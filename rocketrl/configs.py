"""Experiment configuration as dataclasses.

Every run is fully described by a Config: the algorithm, the seed, the env
options, and the hyperparameters. No magic numbers live in the training code.
Configs serialize to / from YAML so experiments are reproducible and diffable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import yaml


@dataclass
class EnvConfig:
    env_id: str = "RocketLander-v1"
    continuous: bool = True
    n_envs: int = 1               # PPO overrides to 16; SAC uses 1
    normalize: bool = True        # VecNormalize obs normalization (replaces baked-in MEAN/VAR)
    norm_reward: bool = True      # reward normalization; both defaults set it OFF (reward is pre-scaled to [-1,1])
    static_normalize: bool = False  # use the env's built-in static obs normalization (original recipe)
    # The discovery landing takes ~930-1000 sim steps; a 1000 cap amputates exactly
    # those trajectories, so landing is never found. The original used 3000.
    max_episode_steps: int = 3000
    # initial-condition difficulty (defaults = original hard task)
    init_x_range: float = 0.3
    init_vx_random: float = 0.4
    init_spin: float = 1.4


@dataclass
class SACConfig:
    algo: str = "sac"
    policy: str = "MlpPolicy"
    total_timesteps: int = 3_000_000
    learning_rate: float = 3e-4
    # smaller buffer discards the stale crash/hover-era transitions
    buffer_size: int = 300_000
    batch_size: int = 512
    tau: float = 0.005
    # gamma 0.997 (not 0.99) kills the hover local optimum: at 0.99 the discounted
    # cost of hovering (~-0.42) beats crashing (~-1), so hovering is an attractor;
    # at 0.997 hovering costs ~-1.4 and the ~1000-step-away landing payoff is
    # discounted far less. This is a SAC-specific objective change (ablate vs 0.99).
    gamma: float = 0.997
    train_freq: int = 1
    gradient_steps: int = -1      # one update per env step collected (per-env)
    learning_starts: int = 25_000
    # fixed small entropy coef; "auto" collapsed to ~1e-4 immediately and left the
    # actor unregularized, causing the late-run policy churn (stoch > det)
    ent_coef: float = 0.01
    net_arch: list[int] = field(default_factory=lambda: [256, 256])
    device: str = "cuda"          # SAC gradient updates benefit from the GPU


@dataclass
class PPOConfig:
    algo: str = "ppo"
    policy: str = "MlpPolicy"
    total_timesteps: int = 15_000_000
    # Gentle recipe matching the original known-good PPO2 run (lr 1e-4, few epochs).
    # The previous aggressive defaults (lr 3e-4, n_epochs 10) collapsed the policy
    # entropy and failed to solve even LunarLander.
    learning_rate: float = 1e-4
    n_steps: int = 1024
    batch_size: int = 4096
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    net_arch: list[int] = field(default_factory=lambda: [64, 64])
    device: str = "cpu"           # tiny MLP + 16 parallel envs: CPU wins


@dataclass
class Config:
    seed: int = 0
    run_name: str = "run"
    out_dir: str = "runs"
    eval_freq: int = 25_000
    n_eval_episodes: int = 20
    checkpoint_freq: int = 100_000
    env: EnvConfig = field(default_factory=EnvConfig)
    algo: dict[str, Any] = field(default_factory=lambda: asdict(SACConfig()))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    @staticmethod
    def load(path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        env = EnvConfig(**raw.pop("env", {}))
        algo = raw.pop("algo", {})
        return Config(env=env, algo=algo, **raw)


# Convenience builders for the two default experiments -----------------------
def default_sac(seed: int = 0) -> Config:
    return Config(
        seed=seed,
        run_name=f"sac_s{seed}",
        # 8 parallel envs: decorrelate the replay buffer and lift throughput
        # (~190 -> ~600-1000 steps/s, the Box2D sim is the serial bottleneck).
        # static obs normalization (dynamic VecNormalize mis-scales the near-pad
        # region; its stats are dominated by early free-fall states)
        env=EnvConfig(n_envs=8, normalize=False, norm_reward=False, static_normalize=True),
        algo=asdict(SACConfig()),
    )


def reproduction_ppo(seed: int = 0) -> Config:
    """Faithful reproduction of the original known-good PPO2 recipe: discrete
    actions, the env's STATIC observation normalization (no VecNormalize),
    3000-step episodes, lr 1e-4, small minibatches, few epochs, 20M steps.
    Used to confirm the rebuilt env is the same learnable task the original solved."""
    algo = asdict(PPOConfig())
    algo.update(total_timesteps=20_000_000, learning_rate=1e-4,
                n_steps=1024, batch_size=256, n_epochs=3)
    return Config(
        seed=seed,
        run_name=f"ppo_repro_s{seed}",
        env=EnvConfig(n_envs=16, continuous=False, normalize=False,
                      norm_reward=False, static_normalize=True, max_episode_steps=3000),
        algo=algo,
    )


def default_ppo(seed: int = 0) -> Config:
    return Config(
        seed=seed,
        run_name=f"ppo_s{seed}",
        # static obs normalization + no reward normalization, matching the
        # original known-good recipe (dynamic VecNormalize was the worst performer)
        env=EnvConfig(n_envs=16, normalize=False, norm_reward=False, static_normalize=True),
        algo=asdict(PPOConfig()),
    )
