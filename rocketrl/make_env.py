"""Environment construction: factory, vectorization, and normalization.

Key correctness points addressed here (from the review):
  * Each parallel env gets a UNIQUE Monitor filename (the original shared one
    'Monitor_Log' across 4 envs, which corrupted the training CSV).
  * Normalization is an explicit VecNormalize wrapper, seeded and saved with
    the model, instead of a constant baked into the env.
  * Episode truncation comes from the registered max_episode_steps (TimeLimit),
    so `truncated` is set correctly for bootstrapping.
"""

from __future__ import annotations

import os

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor)

import rocketrl.envs  # noqa: F401  (registers RocketLander-v1)
from rocketrl.configs import EnvConfig


def make_single_env(cfg: EnvConfig, seed: int, rank: int, monitor_dir: str | None,
                    render_mode: str | None = None):
    """Return a thunk that builds one monitored, seeded env instance."""

    def _init():
        env = gym.make(
            cfg.env_id,
            continuous=cfg.continuous,
            max_episode_steps=cfg.max_episode_steps,
            render_mode=render_mode,
            static_normalize=cfg.static_normalize,
            init_x_range=cfg.init_x_range,
            init_vx_random=cfg.init_vx_random,
            init_spin=cfg.init_spin,
        )
        # Potential-based descent-profile shaping (inside Monitor so the agent
        # sees the shaped reward). Kept below Monitor so the TimeLimit truncation
        # signal still reaches the wrapper correctly.
        if cfg.shape_descent:
            from rocketrl.sacfd import DescentProfileShapingWrapper
            env = DescentProfileShapingWrapper(env, w_shape=cfg.w_shape,
                                               gamma=cfg.shape_gamma)
        monitor_path = None
        if monitor_dir is not None:
            os.makedirs(monitor_dir, exist_ok=True)
            monitor_path = os.path.join(monitor_dir, f"monitor_{rank}")
        # expose terminal outcome keys so success rate is computable from logs
        env = Monitor(env, monitor_path,
                      info_keywords=("landed", "crashed", "touchdown_speed"))
        env.reset(seed=seed + rank)
        return env

    return _init


def build_vec_env(cfg: EnvConfig, seed: int, monitor_dir: str | None,
                  training: bool = True, norm_path: str | None = None):
    """Build the (optionally normalized) vectorized env used for training/eval."""
    thunks = [make_single_env(cfg, seed, i, monitor_dir) for i in range(cfg.n_envs)]
    if cfg.n_envs > 1:
        venv = SubprocVecEnv(thunks)
        venv = VecMonitor(venv)          # episode stats across subprocesses
    else:
        venv = DummyVecEnv(thunks)

    if cfg.normalize:
        if norm_path is not None and os.path.exists(norm_path):
            venv = VecNormalize.load(norm_path, venv)
            venv.training = training
            venv.norm_reward = training
        else:
            venv = VecNormalize(venv, norm_obs=True,
                                norm_reward=cfg.norm_reward and training,
                                clip_obs=10.0, gamma=0.99)
    return venv
