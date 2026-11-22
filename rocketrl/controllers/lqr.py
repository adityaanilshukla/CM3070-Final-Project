"""Classical soft-landing baseline for the continuous RocketLander env.

APPROACH CHOSEN: cascaded PID (feedback-linearized), NOT a discrete-ARE LQR.

Why not LQR.  An LQR gain is only as good as the LTI model it is solved from,
and this plant violates the LTI assumptions in three ways that matter:

  1. The two "position" actuators are RATE commands, not forces.  action[0] is a
     gimbal *rate* (gimbal += action[0] * 0.15/FPS) and action[1] a throttle
     *rate* (throttle += action[1] * 0.5/FPS).  Both integrate into clamped
     states, and both slew-limit hard: gimbal moves at most 0.15 rad/s over a
     +-0.4 rad range (~5 s end to end) and throttle at most 0.5/s over [0,1]
     (~2 s end to end).  A proper LQR would have to augment throttle and gimbal
     as states and would then spend its authority fighting those rate limits.
  2. The main-engine thrust is discontinuous at throttle=0: power jumps from 0
     straight to MIN_THROTTLE (0.4) the instant throttle>0.  There is no small
     thrust; the linearization around hover is a lie on one side.
  3. The third actuator (side thruster) is a discrete three-way {-1,0,+1} impulse,
     not a continuous force.

After augmentation + rate clamps + the discontinuity + the discrete thruster, a
"solved" LQR degrades into a saturating gain schedule, i.e. a PID with extra
ceremony.  So we implement the honest thing: a cascaded, feedback-linearized PID.
Every gain below is derived from constants imported from the env or from short
empirical system-id probes (documented inline), so there are no bare magic
numbers.  The class is named CascadedPIDController to stay honest; the factory is
still make_classical_controller() and the file is still lqr.py per the harness.

Cascade structure (all three loops run every step):
  * Vertical loop  : glide-slope target descent speed  -> feedback-linearized
                     throttle (solve the exact thrust needed for a commanded
                     vertical acceleration, invert the throttle->power map).
  * Lateral loop   : (x, vx) PD -> desired small body tilt (a tilted rocket
                     vectors some main thrust sideways to translate).
  * Attitude loop  : (angle - desired_tilt, omega) PD -> gimbal position command
                     (+ side thruster when the slow gimbal saturates), then an
                     inner rate law slews the gimbal/throttle toward those
                     commands at the actuator's max rate.

Sign conventions below were measured, not assumed (see SYSTEM-ID block).
"""

from __future__ import annotations

import numpy as np

# Import the ground-truth physics constants so nothing is hand-copied.
from rocketrl.envs.rocket_lander import (
    FPS,
    MAIN_ENGINE_POWER,
    MIN_THROTTLE,
    GIMBAL_THRESHOLD,
    START_SPEED,
    W,
    H,
    SHIP_HEIGHT,
)

# ---------------------------------------------------------------------------
# Plant parameters (measured once from Box2D; reproduced here with derivation).
# ---------------------------------------------------------------------------
# Gravity: Box2D world is created with the default b2World() gravity (0, -10).
G = 10.0                                   # m/s^2, verified via env.world.gravity

# Total mass the main engine must lift.  The lander body is a rectangle
# ROCKET_WIDTH x ROCKET_HEIGHT at density 1.0 (area ~= 21.24), plus two legs
# (~0.86 each).  Measured from Box2D: lander.mass=21.244, 2 legs=1.717.
MASS = 22.96                               # kg, = lander.mass + 2*leg.mass

# Vertical equilibrium ("hover") throttle.  Thrust = MAIN_ENGINE_POWER*power and
# power = MIN_THROTTLE + throttle*(1-MIN_THROTTLE) for throttle>0.  Setting
# thrust = MASS*G and inverting:
HOVER_POWER = MASS * G / MAIN_ENGINE_POWER            # ~0.410
HOVER_THROTTLE = (HOVER_POWER - MIN_THROTTLE) / (1.0 - MIN_THROTTLE)   # ~0.017
# Note how close to zero this is: the rocket nearly hovers at idle throttle, and
# the useful control band is throttle in roughly [0, 0.5].  Braking authority
# above hover is strong (~+14 m/s^2 at full throttle); the only way to descend
# *faster* than free-fall-minus-nothing is to cut throttle to 0 (a -10 m/s^2
# cliff).  The vertical profile is therefore designed to only ever BRAKE in the
# terminal phase (arrive slow, coast down near hover).

# --- Actuator rate limits (from env.step integration constants) --------------
GIMBAL_RATE = 0.15 / FPS                   # rad per step at |action[0]|=1
THROTTLE_RATE = 0.5 / FPS                  # throttle units per step at |action[1]|=1

# ---------------------------------------------------------------------------
# SYSTEM-ID probes (measured by holding pose fixed and reading domega/dvx per
# step; baseline-subtracted to remove the constant leg-joint torque bias):
#   +gimbal 0.2  -> domega = -0.0231/step  => d2theta/dgimbal ~ -6.9 rad/s^2 per rad
#   +angle 0.15  -> dvx    = -0.045/step   => a_x ~ -T*sin(angle), T~thrust accel
#   side +1      -> dvx = -0.028/step (push left),  small +omega
#   side -1      -> dvx = +0.028/step (push right), stronger -omega
# Takeaways used below:
#   * POSITIVE gimbal produces NEGATIVE angular accel.
#   * POSITIVE body angle produces NEGATIVE (leftward) horizontal accel.
#   * side force_dir=+1 pushes LEFT (-x); force_dir=-1 pushes RIGHT (+x).
GIMBAL_TORQUE_GAIN = 6.9      # |angular accel| (rad/s^2) per unit gimbal (rad)

# ---------------------------------------------------------------------------
# Observation -> SI unit conversions (inverting the env's obs encoding).
# ---------------------------------------------------------------------------
#   obs[0] = 2*(x - W/2)/W         -> x_m   = obs[0] * W/2
#   obs[1] = 2*(y_dist - 0.5)      -> h_m   = (obs[1]/2 + 0.5) * (H - shipheight)
#   obs[2] = angle (rad)           -> theta = obs[2]
#   obs[5] = 2*(throttle - 0.5)    -> throttle = obs[5]/2 + 0.5
#   obs[6] = gimbal/GIMBAL_THRESH  -> gimbal = obs[6]*GIMBAL_THRESHOLD
#   obs[7] = vx/START_SPEED        -> vx_m  = obs[7]*START_SPEED
#   obs[8] = vy/START_SPEED        -> vy_m  = obs[8]*START_SPEED
#   obs[9] = omega (rad/s)         -> omega = obs[9]
SHIPHEIGHT = H / 20.0 + SHIP_HEIGHT        # terrainheigth + SHIP_HEIGHT, = pad top y
HALF_W = W / 2.0                           # pad half-width in meters (~134.75)
ALT_SPAN = H - SHIPHEIGHT                  # meters mapped by y_distance (~364.5)

# ---------------------------------------------------------------------------
# VERTICAL loop gains.
# ---------------------------------------------------------------------------
# Glide slope: a LINEAR height->speed schedule, v_target = -(V_TOUCH + K*h),
# capped at V_DESC_MAX.  A linear slope (vs a sqrt constant-decel profile) has
# no flare discontinuity, so the rocket keeps descending smoothly into the pad
# instead of over-braking into a hover.  At h=0 the target is -V_TOUCH, i.e. the
# rocket flies into the deck at a safe speed rather than hovering above it.
#   The 3000-step budget favours a SLOW descent: a shallow K_SLOPE and a low
#   V_DESC_MAX, so the rocket sinks gently while it finishes centering.
V_TOUCH = 2.5           # m/s, terminal descent speed at the pad (well under 8 gate)
# Deliberately SLOW descent.  The episode budget is 3000 steps (50 s), so the
# winning strategy is to brake to a low speed early and sink gently while
# centering over the pad the whole way down (a steep, fast glide leaves no time
# to finish centering, which is the dominant failure).  A low V_DESC_MAX caps
# the whole mid-descent; a shallow K_SLOPE keeps it slow well above the pad.
V_DESC_MAX = 22.0       # m/s, cap on commanded descent speed (slow, patient fall)
K_SLOPE = 0.07          # (m/s per m) glide-slope gradient: gentle (reaches cap ~280 m)
# Vertical-speed feedback bandwidth.  Thrust can be re-pointed within a few
# tenths of a second, so a slightly-above-1 rad/s speed loop is realistic.
KP_VY = 1.5             # (m/s^2) per (m/s) vertical-speed error
# Max commanded vertical accel.  The achievable value is thrust-limited AND
# falls with tilt (vertical thrust component ~ Tmax*cos(tilt) - G), so it is
# clamped DYNAMICALLY at runtime (see THRUST_ACCEL_MAX); this static value is
# only a conservative absolute ceiling.
A_Y_MAX = 15.0          # m/s^2, absolute ceiling (set above the ~14 feasible
                        # max so the dynamic tilt-aware clamp below is what binds)
# Peak thrust acceleration = MAIN_ENGINE_POWER * power(=1) / MASS.  Used to clamp
# the commanded vertical accel to what the engine can actually deliver at the
# current tilt, so the vertical loop never asks for infeasible braking.
THRUST_ACCEL_MAX = MAIN_ENGINE_POWER / MASS      # ~24.4 m/s^2
A_Y_FEASIBLE_MARGIN = 1.0                         # command full feasible thrust when braking

# ---------------------------------------------------------------------------
# LATERAL loop gains  (drive x_m, vx_m -> 0 via a commanded horizontal accel,
# then convert to a tilt through a_x = -T*sin(theta)).
# ---------------------------------------------------------------------------
# Second-order target.  The inner attitude loop (gimbal) is slow, so this outer
# loop was prone to limit-cycling (x, vx circle the origin 90 deg out of phase).
# The cure is NOT just a slow outer loop -- that fails to centre within the
# container-free zone in time -- but a direct, lag-free lateral-velocity damper
# via the side thruster (see SIDE_AX_TRIGGER below), which lets us run a moderate
# omega_n ~ 0.45 rad/s with heavy damping (zeta ~ 1.8).  KP=wn^2, KD=2*zeta*wn.
LAT_WN = 0.40
LAT_ZETA = 1.6
KP_X = LAT_WN ** 2            # ~0.16  (m/s^2 per m)
KD_X = 2 * LAT_ZETA * LAT_WN  # ~1.28  (m/s^2 per m/s)
# Tilt authority.  Kept MODEST on purpose: with the slow 3000-step descent there
# is ample time to centre, and a large tilt cap makes the position loop overshoot
# -- it tilts hard early, builds a large lateral velocity, then spends the rest
# of the fall undoing it (which also steals vertical braking).  A small cap plus
# the lag-free side thruster centres smoothly without that overshoot.
TILT_MAX = 0.25             # rad, max commanded tilt
# Small position gain used only in the settle phase, where the body is held
# upright and the side thruster (lag-free) trims the last few metres onto the
# pad centre.  Much smaller than KP_X so it nudges rather than shoves.
KP_X_SETTLE = 0.05          # (m/s^2 per m) settle-phase centering gain
# Spin gate: suppress the lateral tilt command while the initial spin is being
# arrested (|omega| above OMEGA_GATE_OK), fading to SPIN_GATE_MIN authority.
OMEGA_GATE_OK = 0.35        # rad/s, spin below which full tilt is allowed
OMEGA_GATE_SCALE = 0.9      # rad/s, span over which tilt fades to SPIN_GATE_MIN
SPIN_GATE_MIN = 0.1         # floor on tilt-authority fraction during spin-kill
# Position-chase cutoff: below H_POS_CUT stop steering toward the pad and only
# DAMP lateral velocity (null vx).  Chasing the last few metres of position near
# the ground just tilts the body and builds vx that cannot be nulled before
# touchdown; landing upright and slow a little off-centre beats a fast arrival.
H_POS_CUT = 22.0          # m, below this the position (KP_X) term fades out.
# Kept high so the rocket is still actively steered toward the pad well into the
# final descent (the slow fall gives time to use it); residual centering below
# this is handled by the lag-free side thruster, which does not tilt the body.
T_EST_FLOOR = 6.0           # m/s^2, floor on thrust-accel estimate for tilt mapping
# Leveling taper: lateral steering tilts the body, but a tilted touchdown snaps
# a leg (the env's brokenleg check) even at low speed.  So fade the commanded
# tilt to zero as the pad approaches -- full lateral authority above H_LEVEL_HI,
# strictly upright below H_LEVEL_LO -- trading a little landing-spot precision
# for an upright, leg-safe touchdown.  The window must give the slow gimbal time
# to null the tilt: ~20 m at ~7 m/s descent is ~3 s, enough.
H_LEVEL_HI = 22.0            # m, above this: full lateral steering
H_LEVEL_LO = 3.0            # m, below this: strictly upright (desired_tilt = 0)

# ---------------------------------------------------------------------------
# ATTITUDE loop gains  (angle -> gimbal position command).
# ---------------------------------------------------------------------------
# Attitude bandwidth is limited by the slow gimbal.  Target omega_n ~ 2.4 rad/s.
# The spin-damping term carries an extra 1.6x boost (KD_BOOST): the gimbal rate
# limit makes the attitude loop ring, and extra omega feedback suppresses that
# ringing (which otherwise couples into the lateral channel as a limit cycle).
ATT_WN = 2.4
ATT_ZETA = 0.9
KD_BOOST = 1.6
KP_A = ATT_WN ** 2                        # ~5.76  (rad/s^2 per rad angle error)
KD_A = 2 * ATT_ZETA * ATT_WN * KD_BOOST   # ~6.91  (rad/s^2 per rad/s spin)

# Inner rate laws: command the actuator toward its target at (near) max slew.
# A gain of ~1/RATE saturates the action for any error beyond one step of slew,
# which is exactly "move toward target as fast as allowed".
KR_GIM = 1.0 / GIMBAL_RATE      # drives action[0] to +-1 for any real gimbal error
KR_THR = 1.0 / THROTTLE_RATE    # drives action[1] to +-1 for any real throttle error

# Side thruster as a FAST attitude co-actuator.  The gimbal is strong (up to
# ~2.76 rad/s^2) but slews slowly (full deflection ~2.6 s), so it cannot by
# itself arrest the initial spin (up to +-1.4 rad/s) before the body tilts far
# enough to build large lateral velocity.  The thruster is weak but INSTANT, so
# we fire it continuously whenever the blended attitude error demands more
# corrective angular accel than a small deadband -- this keeps the attitude
# transient (and hence the lateral-velocity it injects) much smaller.
SIDE_ACCEL_TRIGGER = 0.5        # rad/s^2 demanded before the thruster assists
# Side-thruster lateral assist (the limit-cycle breaker).  Below H_VX_DAMP the
# thruster fires in the direction of the lateral controller's commanded accel
# a_x_des (which blends pad-centering and vx-damping), supplying lag-free lateral
# authority.  This is what breaks the terminal limit cycle: it damps vx DIRECTLY
# instead of through the slow tilt->attitude path, spiralling (x, vx) into the
# origin.  It must assist a_x_des (not blindly oppose vx), else it fights the
# corrective velocity the position loop needs to reach the pad.
H_VX_DAMP = 90.0                # m, altitude below which the thruster assists
SIDE_AX_TRIGGER = 0.6           # m/s^2 of commanded lateral accel before firing

# Touchdown / settle.  CRITICAL: latch settle on EITHER leg touching (or very
# low altitude), not both.  With any residual lateral velocity one leg touches
# first; if the controller keeps steering while pivoting on that single leg it
# drives the body over (a slow one-legged tip-over).  On settle we go strictly
# upright, drop all lateral/side steering, and command a gentle descent so the
# second leg plants and the speed stays under the 8 m/s gate.
CONTACT = 0.5                   # obs leg-contact flag threshold
H_SETTLE = 6.0                  # m, below this altitude force the settle phase
# Firm (not feather-soft) settle descent.  A too-gentle settle lets the rocket
# perch on one leg and bob on the soft leg springs without ever planting both
# legs (a truncation).  3 m/s is still far under the 8 m/s gate but firmly seats
# both legs so the landed condition (both contacts + speed gate) latches.
V_SETTLE = 3.0                  # m/s, settle descent speed


class CascadedPIDController:
    """Cascaded, feedback-linearized PID lander (see module docstring)."""

    name = "lqr"   # harness-facing label for the classical baseline slot

    def __init__(self) -> None:
        self.reset()

    # -- per-episode state -------------------------------------------------
    def reset(self) -> None:
        """Clear everything that could leak between episodes.

        This controller is deliberately almost stateless (every loop is PD on
        the current observation); the one piece of per-episode memory is the
        settle latch, which we clear here so no information crosses an episode
        boundary (the old FSM leaked exactly this kind of state).
        """
        self._settling = False          # latched once a leg touches / we are low

    # -- main control law --------------------------------------------------
    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)

        # --- decode observation into SI units ---------------------------
        x_m = obs[0] * HALF_W
        h_m = (obs[1] / 2.0 + 0.5) * ALT_SPAN
        theta = float(obs[2])
        left_contact = obs[3] > CONTACT
        right_contact = obs[4] > CONTACT
        throttle = obs[5] / 2.0 + 0.5
        gimbal = obs[6] * GIMBAL_THRESHOLD
        vx_m = obs[7] * START_SPEED
        vy_m = obs[8] * START_SPEED
        omega = float(obs[9])

        # Latch settle on the FIRST leg contact or once we are very low.  Once
        # latched it stays latched for the episode (no un-latching that could
        # re-start steering and tip the rocket).
        if left_contact or right_contact or h_m < H_SETTLE:
            self._settling = True

        # ================================================================
        # VERTICAL loop -> throttle command (feedback-linearized)
        # ================================================================
        # Dynamically feasible upward accel: the vertical thrust component is
        # Tmax*cos(tilt), so the most the rocket can decelerate a fall is
        # Tmax*cos(tilt) - G.  Clamping to this (with a small margin) keeps the
        # vertical loop from commanding braking the tilted engine cannot deliver.
        a_y_hi = min(A_Y_MAX,
                     THRUST_ACCEL_MAX * max(np.cos(theta + gimbal), 0.3)
                     * A_Y_FEASIBLE_MARGIN - G)

        if self._settling:
            # Keep flying a gentle fixed descent (NOT throttle-off, which would
            # drop the rocket onto the second leg too hard and trip the speed
            # gate).  Reuse the feedback-linearized throttle for v_target=-V_SETTLE.
            a_y_des = KP_VY * (-V_SETTLE - vy_m)
            a_y_des = float(np.clip(a_y_des, -G, a_y_hi))
            axis = max(np.cos(theta + gimbal), 0.3)
            power = (G + a_y_des) / axis * MASS / MAIN_ENGINE_POWER
            throttle_cmd = float(np.clip(
                (power - MIN_THROTTLE) / (1.0 - MIN_THROTTLE), 0.0, 1.0))
        else:
            # Glide-slope target descent speed (negative = downward).  Constant
            # deceleration profile above the flare, gentle fixed speed below it.
            # Linear glide slope and its feedforward deceleration.  WITHOUT the
            # feedforward term a pure proportional speed loop structurally lags
            # the descending profile (it only brakes in proportion to the
            # tracking error), so the rocket always descends faster than planned
            # and arrives too fast.  For v_target = -(V_TOUCH + K*h) the profile
            # deceleration is dv_target/dt = -K*vy (positive, since vy<0).
            v_sched = V_TOUCH + K_SLOPE * h_m
            if v_sched < V_DESC_MAX:
                v_target = -v_sched
                a_y_ff = -K_SLOPE * vy_m          # profile's own deceleration
            else:
                v_target = -V_DESC_MAX
                a_y_ff = 0.0                       # on the speed cap, no FF
            # Feedforward + vertical-speed PD -> desired vertical acceleration.
            a_y_des = a_y_ff + KP_VY * (v_target - vy_m)
            a_y_des = float(np.clip(a_y_des, -G, a_y_hi))  # stay thrust-feasible
            # Feedback linearization: thrust accel needed = G + a_y_des, but the
            # thrust is along the body axis, so the vertical component is
            # cos(theta+gimbal); divide it out so tilt does not sag the altitude.
            axis = np.cos(theta + gimbal)
            axis = max(axis, 0.3)                      # guard near-horizontal
            thrust_accel = (G + a_y_des) / axis
            power = thrust_accel * MASS / MAIN_ENGINE_POWER
            throttle_cmd = (power - MIN_THROTTLE) / (1.0 - MIN_THROTTLE)
            throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))

        # ================================================================
        # LATERAL loop -> desired body tilt
        # ================================================================
        if self._settling:
            # Stay strictly upright (no tilt), but keep a SMALL bounded position
            # term in a_x_des so the lag-free side thruster nudges the rocket onto
            # the pad centre during the settle -- this directly fixes the common
            # "lands soft but just outside |x|<0.073" failure.  The gain is tiny
            # (KP_X_SETTLE << KP_X) so it only trims, never shoves.
            a_x_des = -(KP_X_SETTLE * x_m + KD_X * vx_m)
            desired_tilt = 0.0
        else:
            # PD on (x, vx) -> commanded horizontal acceleration toward the pad.
            # Fade the position (KP_X) term out below H_POS_CUT so the terminal
            # phase only damps lateral velocity (prevents the position chase from
            # re-injecting vx the slow attitude loop cannot null before landing).
            pos_gain = np.clip((h_m - H_LEVEL_LO) / (H_POS_CUT - H_LEVEL_LO),
                               0.0, 1.0)
            a_x_des = -(pos_gain * KP_X * x_m + KD_X * vx_m)
            # Invert a_x = -T*sin(theta): theta = -asin(a_x_des / T).  Use the
            # current thrust-accel estimate (floored) as T.
            t_est = max(MAIN_ENGINE_POWER * _power_of(throttle) / MASS, T_EST_FLOOR)
            ratio = float(np.clip(-a_x_des / t_est, -0.99, 0.99))
            desired_tilt = float(np.clip(np.arcsin(ratio), -TILT_MAX, TILT_MAX))
            # Fade lateral tilt to zero near the ground so we touch down upright.
            level_taper = np.clip((h_m - H_LEVEL_LO) / (H_LEVEL_HI - H_LEVEL_LO),
                                  0.0, 1.0)
            desired_tilt *= level_taper
            # SPIN GATE: while the initial body spin (up to ~1.4 rad/s) is still
            # being arrested, suppress the lateral tilt command.  Otherwise the
            # attitude transient of killing the spin AND chasing a tilt setpoint
            # overshoots -- for a far, same-signed-spin start it tilts the body
            # further off-pad and steals vertical braking (a too-fast crash).
            # This only acts for the ~1 s spin-kill, then full lateral resumes.
            spin_gate = np.clip(1.0 - (abs(omega) - OMEGA_GATE_OK) / OMEGA_GATE_SCALE,
                                SPIN_GATE_MIN, 1.0)
            desired_tilt *= spin_gate

        # ================================================================
        # ATTITUDE loop -> gimbal position command (+ side thruster)
        # ================================================================
        angle_err = theta - desired_tilt
        # Desired angular acceleration to null the attitude error and spin.
        ang_accel_des = -(KP_A * angle_err + KD_A * omega)
        # Invert d2theta = -GIMBAL_TORQUE_GAIN * gimbal  ->  gimbal position.
        gimbal_cmd = -ang_accel_des / GIMBAL_TORQUE_GAIN
        gimbal_cmd = float(np.clip(gimbal_cmd, -GIMBAL_THRESHOLD, GIMBAL_THRESHOLD))

        # Side thruster: PRIMARY role is to damp lateral velocity DIRECTLY, with
        # no attitude-loop lag.  This is the key to breaking the terminal limit
        # cycle: the position->tilt->(slow gimbal)->velocity path has ~90 deg of
        # lag, so (x, vx) circle the origin (when x~0, vx is maximal).  A direct
        # -k*vx force spirals (x, vx) INTO the origin instead.  The sign is also
        # synergistic with leveling: vx>0 (drifting right) typically comes from a
        # negative tilt, and force_dir=+1 both pushes left AND adds +omega, which
        # rotates the body back upright.  Measured: force_dir=+1 pushes -x (left)
        # and gives +omega; force_dir=-1 pushes +x (right) and gives -omega.
        side = 0.0
        # Assist the lateral controller's desired accel a_x_des (which already
        # blends pad-centering and velocity-damping), lag-free.  To push +x fire
        # force_dir=-1; to push -x fire force_dir=+1.  So side = -sign(a_x_des).
        if h_m < H_VX_DAMP and abs(a_x_des) > SIDE_AX_TRIGGER:
            side = -float(np.sign(a_x_des))
        # Attitude emergency override: when a large angular accel is demanded
        # (e.g. killing the initial spin) and the slow gimbal cannot supply it,
        # commandeer the thruster for attitude instead.
        if not self._settling:
            if ang_accel_des > SIDE_ACCEL_TRIGGER:
                side = 1.0     # need +angular accel (force_dir=+1 gives +omega)
            elif ang_accel_des < -SIDE_ACCEL_TRIGGER:
                side = -1.0    # need -angular accel (force_dir=-1 gives -omega)

        # ================================================================
        # INNER rate laws: slew actuators toward their commanded positions.
        # ================================================================
        gimbal_rate_cmd = np.clip(KR_GIM * (gimbal_cmd - gimbal), -1.0, 1.0)
        throttle_rate_cmd = np.clip(KR_THR * (throttle_cmd - throttle), -1.0, 1.0)

        return np.array([gimbal_rate_cmd, throttle_rate_cmd, side], dtype=np.float32)


def _power_of(throttle: float) -> float:
    """Main-engine power fraction for a given throttle (mirrors env.step)."""
    if throttle <= 0.0:
        return 0.0
    return MIN_THROTTLE + throttle * (1.0 - MIN_THROTTLE)


def make_classical_controller() -> CascadedPIDController:
    """Factory: a ready-to-use classical controller for the comparison harness."""
    return CascadedPIDController()


__all__ = ["CascadedPIDController", "make_classical_controller"]
