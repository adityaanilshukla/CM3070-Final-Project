"""Train SAC or PPO on RocketLander-v1 from a Config.

Usage:
    python -m rocketrl.train --algo sac --seed 0
    python -m rocketrl.train --algo ppo --seed 0 --timesteps 5000000
    python -m rocketrl.train --config runs/sac_s0/config.yaml

Outputs (under out_dir/run_name/):
    config.yaml        the exact config used
    tb/                TensorBoard logs
    monitor/           per-env episode CSVs (unique filenames)
    checkpoints/       periodic model snapshots
    best/              best model by eval
    model.zip          final model
    vecnormalize.pkl   saved normalization statistics
"""

from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback)

from rocketrl.configs import Config, default_ppo, default_sac, reproduction_ppo
from rocketrl.make_env import build_vec_env
from rocketrl.seeding import set_global_seeds


def build_model(cfg: Config, train_env, tb_dir: str):
    a = cfg.algo
    common = dict(
        policy=a["policy"],
        env=train_env,
        seed=cfg.seed,
        verbose=1,
        tensorboard_log=tb_dir,
        device=a["device"],
        gamma=a["gamma"],
        learning_rate=a["learning_rate"],
        policy_kwargs=dict(net_arch=a["net_arch"]),
    )
    if a["algo"] == "sac":
        return SAC(
            **common,
            buffer_size=a["buffer_size"],
            batch_size=a["batch_size"],
            tau=a["tau"],
            train_freq=a["train_freq"],
            gradient_steps=a["gradient_steps"],
            learning_starts=a["learning_starts"],
            ent_coef=a["ent_coef"],
        )
    if a["algo"] == "ppo":
        return PPO(
            **common,
            n_steps=a["n_steps"],
            batch_size=a["batch_size"],
            n_epochs=a["n_epochs"],
            gae_lambda=a["gae_lambda"],
            clip_range=a["clip_range"],
            ent_coef=a["ent_coef"],
        )
    raise ValueError(f"unknown algo {a['algo']!r}")


def train(cfg: Config) -> str:
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    tb_dir = os.path.join(run_dir, "tb")
    monitor_dir = os.path.join(run_dir, "monitor")
    cfg.save(os.path.join(run_dir, "config.yaml"))

    set_global_seeds(cfg.seed)

    train_env = build_vec_env(cfg.env, cfg.seed, monitor_dir, training=True)
    # separate eval env (no reward norm, fresh seed offset) for honest eval
    eval_env = build_vec_env(cfg.env, cfg.seed + 10_000,
                             os.path.join(run_dir, "eval_monitor"), training=False)

    model = build_model(cfg, train_env, tb_dir)

    callbacks = [
        CheckpointCallback(
            save_freq=max(cfg.checkpoint_freq // cfg.env.n_envs, 1),
            save_path=os.path.join(run_dir, "checkpoints"),
            name_prefix="model",
            save_vecnormalize=True,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(run_dir, "best"),
            log_path=os.path.join(run_dir, "eval"),
            eval_freq=max(cfg.eval_freq // cfg.env.n_envs, 1),
            n_eval_episodes=cfg.n_eval_episodes,
            deterministic=True,
        ),
    ]

    try:
        model.learn(total_timesteps=cfg.algo["total_timesteps"], callback=callbacks,
                    progress_bar=False)
    finally:
        # Always persist the final model + normalization stats, even if the
        # progress-bar teardown or a KeyboardInterrupt raises at the very end.
        # (A prior run lost its final save to a rich/tqdm shutdown error here.)
        model.save(os.path.join(run_dir, "model"))
        if cfg.env.normalize:
            train_env.save(os.path.join(run_dir, "vecnormalize.pkl"))
    print(f"[done] {run_dir}")
    return run_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=None,
                   help="override total_timesteps (useful for smoke tests)")
    p.add_argument("--discrete", action="store_true",
                   help="use the discrete action space (control to reproduce the known-good baseline)")
    p.add_argument("--repro", action="store_true",
                   help="faithful reproduction of the original known-good PPO2 recipe")
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    if args.config:
        cfg = Config.load(args.config)
    elif args.repro:
        cfg = reproduction_ppo(args.seed)
    else:
        cfg = default_ppo(args.seed) if args.algo == "ppo" else default_sac(args.seed)
    if args.timesteps is not None:
        cfg.algo["total_timesteps"] = args.timesteps
    if args.discrete:
        cfg.env.continuous = False
        cfg.run_name = cfg.run_name + "_discrete"
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    train(cfg)


if __name__ == "__main__":
    main()
