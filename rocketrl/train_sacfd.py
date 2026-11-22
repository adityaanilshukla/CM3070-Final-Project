"""Train demonstration-seeded SAC (SACfD) on RocketLander-v1.

Pipeline:
  1. Generate (or reuse) landing demonstrations from the fast continuous
     controller -> an .npz of statically-normalized transitions.
  2. Build SAC with a DemoMixReplayBuffer that mixes that frozen demo partition
     into every batch, and a small learning_starts (the buffer is demo-rich from
     step 0).
  3. Train through the shared train() machinery (monitors, eval, checkpoints).

Usage:
    # smoke test (short run, writes under a deletable dir)
    python -m rocketrl.train_sacfd --timesteps 80000 \
        --out-dir runs_smoke_sacfd --run-name sacfd_smoke --demo-episodes 100

    # full run (the coordinator launches this)
    python -m rocketrl.train_sacfd --timesteps 1000000 --seed 0

The demo .npz is deterministic in (seed-base, episodes), so a smoke-generated
set can be reused for the full run via --demo-path.
"""

from __future__ import annotations

import argparse
import os

from rocketrl.configs import EnvConfig, default_sacfd
from rocketrl.sacfd import DEMO_SEED_BASE, generate_demos
from rocketrl.train import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=1_000_000,
                   help="total training steps (coordinator uses ~1,000,000)")
    p.add_argument("--demo-episodes", type=int, default=100,
                   help="fast-controller episodes to roll for demonstrations")
    p.add_argument("--demo-seed-base", type=int, default=DEMO_SEED_BASE,
                   help="seed base for demo episodes (disjoint from the eval bank)")
    p.add_argument("--demo-ratio", type=float, default=0.25,
                   help="fraction of every training batch drawn from the demos")
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--bc-coef", type=float, default=2.0,
                   help="initial weight of the behavior-cloning auxiliary actor loss on demo rows")
    p.add_argument("--bc-coef-final", type=float, default=None,
                   help="if set, decay bc_coef linearly to this value over training (strong early)")
    p.add_argument("--bc-q-filter", action="store_true",
                   help="only apply the BC loss where Q(demo_action) > Q(policy_action)")
    p.add_argument("--gradient-steps", type=int, default=None,
                   help="override gradient_steps (default from default_sacfd is 1)")
    p.add_argument("--shape-descent", action="store_true",
                   help="enable potential-based descent-velocity-profile reward shaping")
    p.add_argument("--w-shape", type=float, default=0.1,
                   help="shaping weight W_SHAPE for Phi = -w_shape*|vy - v_ref(h)|")
    p.add_argument("--demo-path", type=str, default=None,
                   help="reuse an existing demo .npz instead of regenerating")
    p.add_argument("--regen-demos", action="store_true",
                   help="regenerate demos even if the .npz already exists")
    p.add_argument("--out-dir", type=str, default="runs")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--device", type=str, default=None,
                   help="override the SAC device (e.g. cpu, cuda)")
    args = p.parse_args()

    # Resolve the demo .npz path. Default: a shared demos dir under out-dir keyed
    # by seed base + episode count, so repeat runs reuse the same demonstrations.
    demo_path = args.demo_path
    if demo_path is None:
        demo_path = os.path.join(
            args.out_dir, "sacfd_demos",
            f"demos_seed{args.demo_seed_base}_{args.demo_episodes}ep.npz")

    if args.regen_demos or not os.path.exists(demo_path):
        print(f"[sacfd] generating {args.demo_episodes} demo episodes -> {demo_path}")
        generate_demos(
            n_episodes=args.demo_episodes,
            seed_base=args.demo_seed_base,
            env_cfg=EnvConfig(),          # hard defaults, matching training
            keep_timeouts=True,
            out_path=demo_path,
            verbose=True,
        )
    else:
        print(f"[sacfd] reusing existing demos: {demo_path}")

    cfg = default_sacfd(args.seed, demo_path=demo_path,
                        demo_ratio=args.demo_ratio,
                        learning_starts=args.learning_starts,
                        bc_coef=args.bc_coef,
                        bc_coef_final=args.bc_coef_final,
                        bc_q_filter=args.bc_q_filter,
                        shape_descent=args.shape_descent,
                        w_shape=args.w_shape)
    cfg.algo["total_timesteps"] = args.timesteps
    if args.gradient_steps is not None:
        cfg.algo["gradient_steps"] = args.gradient_steps
    cfg.out_dir = args.out_dir
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.device is not None:
        cfg.algo["device"] = args.device

    train(cfg)


if __name__ == "__main__":
    main()
