"""Out-of-distribution perturbation wrappers for the robustness study.

Every controller in this project (PPO, demo-seeded SAC, the two hand-coded
cascaded-PID controllers) was trained / tuned INSIDE a fixed initial-condition
distribution. This module adds optional, default-OFF perturbations so we can
measure how each controller DEGRADES under conditions it never saw, and whether
learned control generalizes better or worse than classical control.

Design rules (so nominal training / evaluation are completely unaffected):

  * Every wrapper defaults to a true NO-OP (std=0, wind=0, mass_scale=1.0):
    it draws no randomness and changes no dynamics, so a sweep's level-0 point
    is the genuine nominal landing rate and can be computed once per controller.
  * All perturbation randomness comes from a SIDE RNG seeded from the episode
    seed, NEVER from env.np_random. Pulling from the env RNG would shift the
    initial-condition draw stream and silently break the paired design (the
    perturbed run must share its initial condition with the nominal run at the
    same seed). Seeding the side RNG identically across controllers also makes
    the noise realization / wind direction identical per seed -> a fair, paired
    comparison.
  * The base env (rocketrl/envs/rocket_lander.py) and the controllers
    (rocketrl/controllers/lqr.py) are untouched.

Perturbation / obs-space notes:

  * SENSOR NOISE is added in the STATIC-NORMALIZED observation space for EVERY
    controller, so the physical corruption is identical across controllers.
    The learned agents consume static-normalized obs directly (static_normalize
    =True env), so they get z*std. The classical controllers decode RAW obs, so
    to inject the *same physical* noise we scale by STATIC_OBS_STD: raw obs get
    z*std*STATIC_OBS_STD. Adding std=0.1 directly to raw obs instead would
    corrupt the classical controller ~10-20x harder on angle/vx/contact (those
    raw components are ~0.05-0.1 in std), which would be a representation-scaling
    artifact, not a real robustness gap. The alternative (raw-space std) is noted
    in the write-up.
  * WIND is a constant horizontal force applied at the CENTER OF MASS
    (ApplyForceToCenter), not at the body origin. The body origin is the rocket
    base; a horizontal force there would add a torque ~ -h*Fx and spin the
    rocket, confounding the wind test with an attitude disturbance that is purely
    an artifact of where the origin sits. Direction (sign) is drawn once per
    episode from the side RNG so it is identical across controllers at a seed.
  * MASS scales every vehicle fixture density (lander body + both legs) and
    rebuilds the body mass data at reset, so the total vehicle mass is exactly
    mass_scale x nominal. The classical controllers invert a fixed plant mass
    (MASS=22.96 kg), so this is a direct test of their model-mismatch tolerance.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from rocketrl.envs.rocket_lander import STATIC_OBS_STD


class SensorNoiseWrapper(gym.ObservationWrapper):
    """Add zero-mean Gaussian sensor noise to the observation (default OFF).

    Args:
        env: the env to wrap. Its observations may be raw or static-normalized.
        std: noise standard deviation IN STATIC-NORMALIZED UNITS. std=0 is an
            exact no-op (no RNG draw, obs returned unchanged).
        obs_is_normalized: True if the wrapped env already emits static-normalized
            observations (the learned agents' env). Then per-component noise std
            is exactly `std`. False for raw-obs envs (the classical controllers'
            env): per-component noise std becomes std*STATIC_OBS_STD[i] so the
            PHYSICAL corruption matches the normalized case.

    The side RNG is reseeded from the episode seed on every reset, so the noise
    realization is deterministic and identical across controllers at a given
    seed. Noise is drawn fresh each step (each observation() call).
    """

    def __init__(self, env: gym.Env, std: float = 0.0,
                 obs_is_normalized: bool = True):
        super().__init__(env)
        self.std = float(std)
        n = env.observation_space.shape[0]
        if obs_is_normalized:
            self._scale = np.ones(n, dtype=np.float64)
        else:
            self._scale = STATIC_OBS_STD[:n].astype(np.float64)
        self._rng = np.random.default_rng(0)

    def reset(self, *, seed=None, options=None):
        # Seed the SIDE RNG from the episode seed BEFORE super().reset() applies
        # observation() to the first obs, so the first observation's noise is part
        # of the same deterministic, per-seed realization as the rest.
        if seed is not None:
            self._rng = np.random.default_rng(seed + 90_001)
        return super().reset(seed=seed, options=options)

    def observation(self, obs):
        if self.std == 0.0:
            return obs
        noise = self._rng.normal(0.0, self.std, size=obs.shape) * self._scale
        return (obs.astype(np.float64) + noise).astype(obs.dtype)


class WindWrapper(gym.Wrapper):
    """Constant horizontal wind force on the lander each step (default OFF).

    Args:
        env: env to wrap.
        wind_mag: wind force magnitude in Box2D force units. For scale, the
            vehicle weight is ~230 (mass ~23 kg * g 10). wind_mag=0 is an exact
            no-op. The sign (left vs right) is drawn once per episode from the
            side RNG seeded by the episode seed, so it is identical across
            controllers at a given seed.

    The force is applied at the center of mass (ApplyForceToCenter) before the
    wrapped env's own step, so it is integrated by that step's single world.Step
    together with the engine forces. Box2D clears forces after every step, so
    this re-applies the wind every step -> a sustained constant wind.
    """

    def __init__(self, env: gym.Env, wind_mag: float = 0.0):
        super().__init__(env)
        self.wind_mag = float(wind_mag)
        self._rng = np.random.default_rng(0)
        self._wind = 0.0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed + 91_001)
        if self.wind_mag > 0.0:
            sign = 1.0 if self._rng.random() < 0.5 else -1.0
            self._wind = sign * self.wind_mag
        else:
            self._wind = 0.0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        if self.wind_mag > 0.0:
            self.env.unwrapped.lander.ApplyForceToCenter((self._wind, 0.0), True)
        return self.env.step(action)


class MassScaleWrapper(gym.Wrapper):
    """Scale the whole vehicle's mass at reset (default OFF = 1.0).

    Args:
        env: env to wrap.
        mass_scale: multiplier on every vehicle fixture density (lander body +
            both legs), with the body mass data rebuilt so the total vehicle mass
            is exactly mass_scale x nominal. mass_scale=1.0 is an exact no-op.

    Note: the env's reset() produces its first observation via one internal
    neutral step BEFORE this wrapper can rescale the mass, so that single step
    (1 of up to 3000, at the top of descent) uses the nominal mass. Every
    subsequent step uses the scaled mass.
    """

    def __init__(self, env: gym.Env, mass_scale: float = 1.0):
        super().__init__(env)
        self.mass_scale = float(mass_scale)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if self.mass_scale != 1.0:
            u = self.env.unwrapped
            bodies = [u.lander] + list(u.legs)
            for body in bodies:
                for fx in body.fixtures:
                    fx.density *= self.mass_scale
                body.ResetMassData()
        return obs, info


__all__ = ["SensorNoiseWrapper", "WindWrapper", "MassScaleWrapper"]
