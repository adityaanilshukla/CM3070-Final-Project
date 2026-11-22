"""Uniform two-bank best-checkpoint eval across PPO reproduction seeds.

For each seed, under one identical selection rule:
  1. Sweep the full training curve (every 1M steps) on a SELECTION seed bank.
  2. Pick that seed's best checkpoint by selection-bank landing rate.
  3. Report that checkpoint's landing rate on a disjoint EVAL seed bank.

The selection and eval banks never overlap, so the reported number is not the
checkpoint cherry-picked on its own test set. Running the SAME sweep window for
every seed removes the selection bias that a per-seed window would reintroduce.

Usage:
  uv run python scripts/multiseed_ci.py 0 1 2          # seeds 0,1,2
  uv run python scripts/multiseed_ci.py 0 1 2 3 4      # full five
"""

from __future__ import annotations

import os
import statistics
import sys

import gymnasium as gym
import numpy as np

import rocketrl.envs  # noqa: F401  (registers RocketLander-v1)
from stable_baselines3 import PPO
from rocketrl.controllers.rl import RLController

RUN_FMT = "runs/ppo_repro_s{seed}/checkpoints"
STEPS = list(range(5_000_000, 20_000_001, 1_000_000))  # 5M..20M every 1M
SEL_BANK_BASE = 3_000_000
N_SEL = 40
EVAL_BANK_BASE = 1_000_000
N_EVAL = 100
MAX_STEPS = 3000


def make_eval_env():
    return gym.make(
        "RocketLander-v1",
        continuous=False,
        max_episode_steps=MAX_STEPS,
        static_normalize=True,
    )


def _run_landed(env, ctrl, seed):
    """One episode; return True iff the env reports a landing. Discrete-safe
    (the shared run_episode assumes a continuous 3-vector action for its
    smoothness metric and raises on scalar/discrete actions)."""
    ctrl.reset()
    obs, info = env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        action = ctrl.act(obs)
        if np.ndim(action) == 0 or np.size(action) == 1:
            action = int(np.asarray(action).reshape(-1)[0])  # discrete env wants a scalar int
        obs, _r, terminated, truncated, info = env.step(action)
    return bool(info.get("landed", False))


def landed_rate(model, env, n, base):
    ctrl = RLController(model, None, "ppo_repro")
    k = sum(_run_landed(env, ctrl, base + i) for i in range(n))
    return k / n


def eval_seed(seed: int, env) -> dict:
    run = RUN_FMT.format(seed=seed)
    best = (-1.0, None)
    for s in STEPS:
        path = os.path.join(run, f"model_{s}_steps.zip")
        if not os.path.exists(path):
            continue
        model = PPO.load(path, device="cpu")
        rate = landed_rate(model, env, N_SEL, SEL_BANK_BASE)
        print(f"  seed {seed} {s//1_000_000:>2}M: sel {rate*100:3.0f}%", flush=True)
        if rate > best[0]:
            best = (rate, s)
    sel_rate, best_step = best
    # report the selected checkpoint on the disjoint eval bank
    model = PPO.load(os.path.join(run, f"model_{best_step}_steps.zip"), device="cpu")
    eval_rate = landed_rate(model, env, N_EVAL, EVAL_BANK_BASE)
    print(f"seed {seed}: best {best_step//1_000_000}M "
          f"(sel {sel_rate*100:.0f}%) -> EVAL {eval_rate*100:.0f}% (n={N_EVAL})",
          flush=True)
    return {"seed": seed, "best_step": best_step, "sel": sel_rate, "eval": eval_rate}


def main():
    seeds = [int(a) for a in sys.argv[1:]] or [0, 1, 2]
    env = make_eval_env()
    rows = [eval_seed(s, env) for s in seeds]
    env.close()
    evals = [r["eval"] * 100 for r in rows]
    mean = statistics.mean(evals)
    std = statistics.pstdev(evals) if len(evals) > 1 else 0.0
    print("\n==== summary (eval bank, full-curve uniform selection) ====", flush=True)
    for r in rows:
        print(f"  seed {r['seed']}: {r['eval']*100:.0f}% "
              f"(best {r['best_step']//1_000_000}M)", flush=True)
    print(f"  {len(evals)}-seed: mean {mean:.1f}%, "
          f"range {min(evals):.0f}-{max(evals):.0f}%, std {std:.1f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
