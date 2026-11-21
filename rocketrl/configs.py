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
    normalize: bool = True        # explicit VecNormalize (replaces baked-in MEAN/VAR)
    max_episode_steps: int = 1000


@dataclass
class SACConfig:
    algo: str = "sac"
    policy: str = "MlpPolicy"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    learning_starts: int = 10_000
    ent_coef: str = "auto"
    net_arch: list[int] = field(default_factory=lambda: [256, 256])
    device: str = "cuda"          # SAC gradient updates benefit from the GPU


@dataclass
class PPOConfig:
    algo: str = "ppo"
    policy: str = "MlpPolicy"
    total_timesteps: int = 5_000_000
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 2048
    n_epochs: int = 10
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
        env=EnvConfig(n_envs=1, normalize=True),
        algo=asdict(SACConfig()),
    )


def default_ppo(seed: int = 0) -> Config:
    return Config(
        seed=seed,
        run_name=f"ppo_s{seed}",
        env=EnvConfig(n_envs=16, normalize=True),
        algo=asdict(PPOConfig()),
    )
