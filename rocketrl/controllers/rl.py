"""RLController: adapts a trained Stable-Baselines3 model (SAC or PPO) to the
shared `Controller` protocol (see rocketrl/controllers/__init__.py), so a
learned policy can be dropped into the same evaluation harness as the random
baseline and the classical (LQR) controller.

Why this file exists: controllers in this project consume RAW environment
observations (see the layout documented in controllers/__init__.py). SB3
models trained behind a VecNormalize wrapper expect NORMALIZED observations.
This adapter applies the exact same normalization used at train time
(VecNormalize.normalize_obs, stats loaded from the run's vecnormalize.pkl)
before calling model.predict(..., deterministic=True), and returns a raw
action for the env to consume -- VecNormalize never touches actions for this
env (norm_obs=True only), so no de-normalization of the action is needed.

Expected run_dir layout (as produced by rocketrl.train.train):
    <run_dir>/model.zip            final model, saved at the end of training
    <run_dir>/best/best_model.zip  best model checkpoint (EvalCallback)
    <run_dir>/vecnormalize.pkl     saved VecNormalize running stats
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np

from rocketrl.controllers import Controller


def _load_sb3_model(path: str, algo: str):
    algo = algo.lower()
    if algo == "sac":
        from stable_baselines3 import SAC

        return SAC.load(path, device="cpu")
    if algo == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(path, device="cpu")
    raise ValueError(f"unknown algo {algo!r}; expected 'sac' or 'ppo'")


def _resolve_model_path(run_dir: str) -> str:
    """Prefer the final model.zip; fall back to best/best_model.zip.

    SB3's `.load()` accepts a path with or without the '.zip' suffix, so we
    normalize to the file-with-suffix form before checking existence.
    """
    final_path = os.path.join(run_dir, "model.zip")
    if os.path.exists(final_path):
        return final_path
    best_path = os.path.join(run_dir, "best", "best_model.zip")
    if os.path.exists(best_path):
        return best_path
    raise FileNotFoundError(
        f"no model.zip or best/best_model.zip found under run_dir={run_dir!r}"
    )


def _load_vecnormalize(run_dir: str):
    """Load the saved VecNormalize obs-normalization stats, if present.

    Returns None (and warns) if vecnormalize.pkl is missing, so the caller
    can fall back to raw observations rather than crash -- useful for
    smoke-testing a controller before training finishes writing the final
    artifact, but it means the policy will almost certainly perform badly if
    it was actually trained with norm_obs=True.
    """
    vecnorm_path = os.path.join(run_dir, "vecnormalize.pkl")
    if not os.path.exists(vecnorm_path):
        warnings.warn(
            f"no vecnormalize.pkl found under run_dir={run_dir!r}; "
            "evaluating with RAW (unnormalized) observations. If this model "
            "was trained with VecNormalize(norm_obs=True), results will be "
            "misleading.",
            stacklevel=2,
        )
        return None

    # VecNormalize.load(path, venv) only needs `venv` to validate obs/action
    # space shapes and to attach to; we never step this venv, so a single
    # throwaway DummyVecEnv instance is enough to deserialize the stats.
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    import rocketrl.envs  # noqa: F401  (registers RocketLander-v1)

    dummy_venv = DummyVecEnv([lambda: gym.make("RocketLander-v1", continuous=True)])
    vecnorm = VecNormalize.load(vecnorm_path, dummy_venv)
    vecnorm.training = False  # freeze stats: never update running mean/var at eval time
    return vecnorm


class RLController:
    """Wraps an SB3 model + (optional) VecNormalize stats as a Controller."""

    def __init__(self, model, vecnormalize, name: str):
        self.name = name
        self._model = model
        self._vecnorm = vecnormalize  # None => raw obs passed straight through

    def reset(self) -> None:
        # The wrapped policy is a stateless (feedforward) mapping from obs to
        # action for both SAC and PPO's default MlpPolicy, and VecNormalize
        # stats are frozen (training=False) before eval -- so there is no
        # per-episode state to clear here.
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if self._vecnorm is not None:
            # normalize_obs does not mutate running stats; safe to call on a
            # single-row batch of the raw observation.
            norm_obs = self._vecnorm.normalize_obs(obs.reshape(1, -1))[0]
        else:
            norm_obs = obs
        action, _state = self._model.predict(norm_obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)


def load_rl_controller(
    run_dir: str, algo: str, name: Optional[str] = None
) -> Controller:
    """Build an RLController from a training run directory.

    Args:
        run_dir: directory produced by rocketrl.train (contains model.zip or
            best/best_model.zip, and vecnormalize.pkl).
        algo: "sac" or "ppo" -- selects which SB3 class to use for .load().
        name: display name for the controller; defaults to
            "<algo>:<basename(run_dir)>".
    """
    model_path = _resolve_model_path(run_dir)
    model = _load_sb3_model(model_path, algo)
    vecnorm = _load_vecnormalize(run_dir)
    controller_name = name or f"{algo}:{os.path.basename(os.path.normpath(run_dir))}"
    return RLController(model, vecnorm, controller_name)
