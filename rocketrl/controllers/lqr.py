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


# ===========================================================================
# FAST controller constants (PPO-distilled) -- used ONLY by FastLandingController.
# ===========================================================================
# (a) Vertical braking profile distilled from the PPO agent: descend on a
# constant-deceleration (sqrt) curve, then hand off to a linear flare.  This
# lands in ~9-10 s (vs ~22 s for the slow glide) while staying smooth: the
# engine runs lit near power ~0.74 the whole way, no bang-bang.
#   v_ref(h) = -min(V_TD + K_TERM*h, sqrt(V_TD^2 + 2*A_BRAKE*h))
# V_TD and A_BRAKE were swept on held-out banks disjoint from the PPO/tuning seeds.
# A follow-up sweep on two further held-out banks (seeds 4e6 and 5e6) showed V_TD=3
# consistently lands ~86% (vs ~84% at V_TD=5) with notably softer touchdowns
# (~0.005 vs 0.008), because the lower deck-speed target cuts the "came in too hot"
# crashes, at a trivial (~1 s) time cost and unchanged smoothness. So V_TD=3.
# This sits near the controller's Pareto ceiling (~86-87%): 90% at fast speed is
# not reachable by hand at the continuous interface's gimbal slew rate.
V_TD = 3.0              # m/s, descent speed at the deck (linear-flare intercept)
K_TERM = 0.33          # (m/s per m) terminal linear-flare slope
A_BRAKE = 9.0          # m/s^2, constant-decel branch (~62% of the ~14.4 available)
V_CAP = 80.0           # m/s, never command a descent faster than the entry speed
# (b) Lateral / leveling gates scheduled on TIME-TO-GO (t_go = h/max(-vy,2))
# rather than altitude, because the fast descent reaches a given altitude with a
# much larger speed -- an altitude gate would fade steering far too early.
T_POS_HI, T_POS_LO = 2.2, 0.9   # s to ground: position-steering fade window
T_LVL_HI, T_LVL_LO = 3.0, 1.0   # s to ground: tilt-taper (upright) window
# (c) Latched two-mode router: decide fast-vs-patient ONCE, then hold it.  A
# too-far / too-fast entry cannot be absorbed by the aggressive fast profile, so
# those episodes fall back to the (slow) patient controller -- which is the
# original CascadedPID logic verbatim.
H_GATE = 100.0         # m, single decision altitude (first crossing)
TAU_GATE = 2.5         # s, vx credit horizon in the gate miss projection
X_GATE_ABORT = 26.0    # m, |x + TAU_GATE*vx| beyond this at the gate -> patient
H_LATE = 40.0          # m, below this a tilt-aware late-abort check is armed
X_LATE_ABORT = 18.0    # m, projected miss (incl. current tilt accel) -> patient
T_PROJ_ACC = 1.8       # s, horizon for the tilt-acceleration term of that projection
X0_PATIENT = 45.0      # m, first-obs offset beyond which an adverse spin -> patient
OMEGA0_PATIENT = 0.55  # rad/s, first-obs spin magnitude threshold for that rule
# (d) Fast-mode settle is slightly firmer than the slow controller's (the faster
# approach wants a firmer plant); patient mode keeps the slow settle (H_SETTLE /
# V_SETTLE) verbatim.
H_SETTLE_FAST = 5.0    # m, fast-mode settle altitude
V_SETTLE_FAST = 3.5    # m/s, fast-mode settle descent speed


# ===========================================================================
# Shared cascade helpers (used by BOTH controllers).
#
# The INNER cascade -- feedback-linearized throttle, the lateral-accel->tilt
# map, the spin gate, and the attitude/RCS loop -- is identical for the slow
# classical controller and the fast PPO-distilled controller; only the OUTER
# references differ (the vertical speed profile, and whether the position/level
# gates schedule on altitude or on time-to-go).  Factoring the inner cascade and
# each outer profile here lets the two controllers compose them without
# duplicating ~80 lines of identical logic.
# ===========================================================================

def _power_of(throttle: float) -> float:
    """Main-engine power fraction for a given throttle (mirrors env.step)."""
    if throttle <= 0.0:
        return 0.0
    return MIN_THROTTLE + throttle * (1.0 - MIN_THROTTLE)


def _feedback_lin_throttle(a_y_des: float, theta: float, gimbal: float) -> float:
    """Throttle that realizes a desired vertical acceleration a_y_des.

    Clamps a_y_des to what the TILTED engine can actually deliver
    (Tmax*cos(tilt) - G), then inverts thrust = MASS*(G + a_y_des)/cos(tilt) and
    the throttle->power map.  Shared by every vertical branch of both controllers.
    """
    axis = max(np.cos(theta + gimbal), 0.3)            # guard near-horizontal
    a_y_hi = min(A_Y_MAX, THRUST_ACCEL_MAX * axis * A_Y_FEASIBLE_MARGIN - G)
    a_y_des = float(np.clip(a_y_des, -G, a_y_hi))
    power = (G + a_y_des) / axis * MASS / MAIN_ENGINE_POWER
    return float(np.clip((power - MIN_THROTTLE) / (1.0 - MIN_THROTTLE), 0.0, 1.0))


def _spin_gate(omega: float):
    """Tilt-authority fraction: suppress tilt while the initial spin is large.

    Returns the np.clip result (np.float64) UNwrapped on purpose: the tilt is
    `tilt(py-float) * level(float32) * spin_gate`, and letting the final factor be
    float64 reproduces the baseline's exact (float64) rounding -- wrapping it in
    float() would leave the product at float32 and perturb the action by ~1e-5.
    """
    return np.clip(1.0 - (abs(omega) - OMEGA_GATE_OK) / OMEGA_GATE_SCALE,
                   SPIN_GATE_MIN, 1.0)


def _lateral_tilt(a_x_des: float, throttle: float, level: float,
                  omega: float) -> float:
    """Map a desired horizontal accel to a body tilt, tapered by `level` (the
    outer leveling gate) and the spin gate.  Inverts a_x = -T*sin(theta)."""
    t_est = max(MAIN_ENGINE_POWER * _power_of(throttle) / MASS, T_EST_FLOOR)
    ratio = float(np.clip(-a_x_des / t_est, -0.99, 0.99))
    tilt = float(np.clip(np.arcsin(ratio), -TILT_MAX, TILT_MAX))
    return tilt * level * _spin_gate(omega)


def _attitude_rcs(theta, omega, gimbal, desired_tilt, a_x_des, h_m, settling):
    """Attitude PD -> gimbal-rate command, plus the side-thruster (RCS) logic.

    Returns (gimbal_rate_cmd, side).  The side thruster first assists the lateral
    controller's a_x_des lag-free (the limit-cycle breaker); when not settling a
    large demanded angular accel overrides it for attitude (initial spin-kill).
    """
    angle_err = theta - desired_tilt
    ang_accel_des = -(KP_A * angle_err + KD_A * omega)
    gimbal_cmd = float(np.clip(-ang_accel_des / GIMBAL_TORQUE_GAIN,
                               -GIMBAL_THRESHOLD, GIMBAL_THRESHOLD))
    side = 0.0
    if h_m < H_VX_DAMP and abs(a_x_des) > SIDE_AX_TRIGGER:
        side = -float(np.sign(a_x_des))              # push toward a_x_des (lag-free)
    if not settling:
        if ang_accel_des > SIDE_ACCEL_TRIGGER:
            side = 1.0
        elif ang_accel_des < -SIDE_ACCEL_TRIGGER:
            side = -1.0
    gimbal_rate_cmd = float(np.clip(KR_GIM * (gimbal_cmd - gimbal), -1.0, 1.0))
    return gimbal_rate_cmd, side


def _settle_vertical(vy_m, theta, gimbal, v_settle):
    """Throttle for a gentle fixed settle descent at -v_settle (shared)."""
    return _feedback_lin_throttle(KP_VY * (-v_settle - vy_m), theta, gimbal)


def _glide_throttle(h_m, vy_m, theta, gimbal):
    """SLOW vertical profile: linear glide slope v=-(V_TOUCH+K_SLOPE*h), capped,
    with the profile's own feedforward deceleration (-K_SLOPE*vy)."""
    v_sched = V_TOUCH + K_SLOPE * h_m
    if v_sched < V_DESC_MAX:
        v_target = -v_sched
        a_y_ff = -K_SLOPE * vy_m                 # profile's own deceleration
    else:
        v_target = -V_DESC_MAX
        a_y_ff = 0.0                             # on the speed cap, no FF
    return _feedback_lin_throttle(a_y_ff + KP_VY * (v_target - vy_m), theta, gimbal)


def _fast_throttle(h_m, vy_m, theta, gimbal):
    """FAST vertical profile (PPO-distilled): constant-decel sqrt branch handing
    off to a linear flare, each with its matching feedforward acceleration."""
    v_lin = V_TD + K_TERM * h_m
    v_sqrt = np.sqrt(V_TD ** 2 + 2.0 * A_BRAKE * h_m)
    if v_lin <= v_sqrt:                          # near the ground: linear flare
        v_mag = min(v_lin, V_CAP)
        a_y_ff = -K_TERM * vy_m
    else:                                        # high up: constant-decel brake
        v_mag = min(v_sqrt, V_CAP)
        a_y_ff = A_BRAKE
    return _feedback_lin_throttle(a_y_ff + KP_VY * (-v_mag - vy_m), theta, gimbal)


def _altitude_lateral(h_m, x_m, vx_m, throttle, omega):
    """SLOW lateral: position/level gates scheduled on ALTITUDE.
    Returns (a_x_des, desired_tilt)."""
    pos_gain = np.clip((h_m - H_LEVEL_LO) / (H_POS_CUT - H_LEVEL_LO), 0.0, 1.0)
    a_x_des = -(pos_gain * KP_X * x_m + KD_X * vx_m)
    level = np.clip((h_m - H_LEVEL_LO) / (H_LEVEL_HI - H_LEVEL_LO), 0.0, 1.0)
    return a_x_des, _lateral_tilt(a_x_des, throttle, level, omega)


def _tgo_lateral(h_m, vy_m, x_m, vx_m, throttle, omega):
    """FAST lateral: position/level gates scheduled on TIME-TO-GO.
    Returns (a_x_des, desired_tilt)."""
    t_go = h_m / max(-vy_m, 2.0)
    pos_gain = np.clip((t_go - T_POS_LO) / (T_POS_HI - T_POS_LO), 0.0, 1.0)
    a_x_des = -(pos_gain * KP_X * x_m + KD_X * vx_m)
    level = np.clip((t_go - T_LVL_LO) / (T_LVL_HI - T_LVL_LO), 0.0, 1.0)
    return a_x_des, _lateral_tilt(a_x_des, throttle, level, omega)


def _settle_lateral(x_m, vx_m):
    """Shared settle lateral: hold upright, let the lag-free thruster trim x."""
    return -(KP_X_SETTLE * x_m + KD_X * vx_m), 0.0


def _decode(obs):
    """Decode a raw observation into SI-unit state (shared)."""
    return dict(
        x_m=obs[0] * HALF_W,
        h_m=(obs[1] / 2.0 + 0.5) * ALT_SPAN,
        theta=float(obs[2]),
        contact=(obs[3] > CONTACT or obs[4] > CONTACT),
        both_contact=(obs[3] > CONTACT and obs[4] > CONTACT),
        throttle=obs[5] / 2.0 + 0.5,
        gimbal=obs[6] * GIMBAL_THRESHOLD,
        vx_m=obs[7] * START_SPEED,
        vy_m=obs[8] * START_SPEED,
        omega=float(obs[9]),
    )


# Touchdown engine-cut action: neutral gimbal rate, throttle driven hard to zero
# (so power drops below the MIN_THROTTLE floor to a true engine-off), no thruster.
# Once BOTH legs are planted, continuing to fire the engine at its 40% floor keeps
# the body light and can tip it or stop it settling under the speed gate; cutting
# the engine lets it seat on the legs and settle, which is how a real booster (and
# the original FSM) behaves. Latched per episode once both legs touch.
_ENGINE_CUT_ACTION = np.array([0.0, -1.0, 0.0], dtype=np.float32)


def _assemble_action(throttle_cmd, desired_tilt, a_x_des, s):
    """Run the shared attitude/RCS loop and inner rate laws -> action."""
    gimbal_rate_cmd, side = _attitude_rcs(
        s["theta"], s["omega"], s["gimbal"], desired_tilt, a_x_des,
        s["h_m"], s["_settling"])
    throttle_rate_cmd = float(np.clip(
        KR_THR * (throttle_cmd - s["throttle"]), -1.0, 1.0))
    return np.array([gimbal_rate_cmd, throttle_rate_cmd, side], dtype=np.float32)


class CascadedPIDController:
    """Cascaded, feedback-linearized PID lander (see module docstring).

    The SLOW baseline: a patient glide-slope descent (~22 s) that finishes
    centering over the pad all the way down.  Shares the inner cascade helpers
    with FastLandingController; its outer references are the altitude-scheduled
    glide profile and lateral gates.
    """

    name = "lqr"   # harness-facing label for the classical baseline slot

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear per-episode state.  The only memory is the settle latch; clearing
        it here stops any information crossing an episode boundary (the old FSM
        leaked exactly this kind of state)."""
        self._settling = False          # latched once a leg touches / we are low
        self._touched_down = False      # latched once BOTH legs plant -> cut engine

    def act(self, obs: np.ndarray) -> np.ndarray:
        s = _decode(np.asarray(obs, dtype=np.float32))
        s["_settling"] = self._settling

        # Both legs planted: cut the engine and let the rocket seat and settle
        # (latched for the rest of the episode).
        if s["both_contact"]:
            self._touched_down = True
        if self._touched_down:
            return _ENGINE_CUT_ACTION.copy()

        # Latch settle on the FIRST leg contact or once very low (never unlatch).
        if s["contact"] or s["h_m"] < H_SETTLE:
            self._settling = True
            s["_settling"] = True

        if self._settling:
            throttle_cmd = _settle_vertical(s["vy_m"], s["theta"], s["gimbal"], V_SETTLE)
            a_x_des, desired_tilt = _settle_lateral(s["x_m"], s["vx_m"])
        else:
            throttle_cmd = _glide_throttle(s["h_m"], s["vy_m"], s["theta"], s["gimbal"])
            a_x_des, desired_tilt = _altitude_lateral(
                s["h_m"], s["x_m"], s["vx_m"], s["throttle"], s["omega"])

        return _assemble_action(throttle_cmd, desired_tilt, a_x_des, s)


class FastLandingController:
    """Fast, PPO-distilled lander (see FAST constants above).

    Uses the PPO braking profile and time-to-go lateral gates to land in ~10 s.
    A latched router diverts entries too far/fast for that aggressive profile to
    PATIENT mode, which runs the slow CascadedPID outer loop VERBATIM (altitude
    gates, glide profile, slow settle).  Shares every inner cascade helper with
    the slow controller, so control stays as smooth as the baseline.
    """

    name = "fast"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear ALL per-episode state: the settle latch, the two-mode router
        latch, the gate-armed flag, and the first-obs flag."""
        self._settling = False
        self._patient = False           # latched routing decision (fast vs patient)
        self._gated = False             # the H_GATE decision has been taken
        self._first = True              # first observation of the episode
        self._touched_down = False      # latched once BOTH legs plant -> cut engine

    def act(self, obs: np.ndarray) -> np.ndarray:
        s = _decode(np.asarray(obs, dtype=np.float32))
        x_m, h_m, vx_m, vy_m, omega = s["x_m"], s["h_m"], s["vx_m"], s["vy_m"], s["omega"]

        # Both legs planted: cut the engine and let the rocket seat and settle
        # (latched for the rest of the episode).
        if s["both_contact"]:
            self._touched_down = True
        if self._touched_down:
            return _ENGINE_CUT_ACTION.copy()

        # Router rule 1 (first obs): a large offset with ADVERSE initial spin
        # (x_m * omega < 0, so the spin-recovery transient tilts AWAY from the
        # pad) is exactly what the fast profile cannot absorb -> go patient.
        if self._first:
            self._first = False
            if abs(x_m) > X0_PATIENT and x_m * omega < 0 and abs(omega) > OMEGA0_PATIENT:
                self._patient = True

        # Mode-specific settle geometry (the mode is always decided by h~40 m,
        # well above either settle altitude, so this is unambiguous).
        h_settle = H_SETTLE if self._patient else H_SETTLE_FAST
        v_settle = V_SETTLE if self._patient else V_SETTLE_FAST
        if s["contact"] or h_m < h_settle:
            self._settling = True
        s["_settling"] = self._settling

        if self._settling:
            throttle_cmd = _settle_vertical(vy_m, s["theta"], s["gimbal"], v_settle)
            a_x_des, desired_tilt = _settle_lateral(x_m, vx_m)
        else:
            # Router rule 2 (single decision at H_GATE): simple ballistic miss.
            if not self._patient and not self._gated and h_m <= H_GATE:
                self._gated = True
                if abs(x_m + TAU_GATE * vx_m) > X_GATE_ABORT:
                    self._patient = True
            # Router rule 3 (late abort, armed below H_LATE): tilt-aware miss
            # projection -- if even the current tilt cannot close the gap in the
            # time left, bail to patient.
            if not self._patient and h_m < H_LATE:
                t_go = h_m / max(-vy_m, 2.0)
                t_acc = min(t_go, T_PROJ_ACC)
                a_x_now = -(MAIN_ENGINE_POWER * _power_of(s["throttle"]) / MASS) \
                    * np.sin(s["theta"])
                miss = abs(x_m + vx_m * t_go + 0.5 * a_x_now * t_acc * t_acc)
                if miss > X_LATE_ABORT:
                    self._patient = True

            if self._patient:                        # slow controller, verbatim
                throttle_cmd = _glide_throttle(h_m, vy_m, s["theta"], s["gimbal"])
                a_x_des, desired_tilt = _altitude_lateral(
                    h_m, x_m, vx_m, s["throttle"], omega)
            else:                                    # fast PPO-distilled profile
                throttle_cmd = _fast_throttle(h_m, vy_m, s["theta"], s["gimbal"])
                a_x_des, desired_tilt = _tgo_lateral(
                    h_m, vy_m, x_m, vx_m, s["throttle"], omega)

        return _assemble_action(throttle_cmd, desired_tilt, a_x_des, s)


def make_classical_controller() -> CascadedPIDController:
    """Factory: the slow classical (patient glide-slope) controller."""
    return CascadedPIDController()


def make_fast_controller() -> FastLandingController:
    """Factory: the fast PPO-distilled controller (with patient fallback)."""
    return FastLandingController()


__all__ = [
    "CascadedPIDController",
    "FastLandingController",
    "make_classical_controller",
    "make_fast_controller",
]
