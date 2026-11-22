"""Robustness / out-of-distribution generalization sweep.

All four controllers in the study (PPO, the demo-seeded SAC, and the two
hand-coded cascaded-PID controllers) were evaluated INSIDE their training
initial-condition distribution. This script measures how each one DEGRADES under
conditions it never trained on, and whether learned control generalizes better or
worse than classical control.

For each of four perturbation types it sweeps the perturbation magnitude over a
few levels (including a nominal level-0) and records the landing rate of every
controller on a fixed, shared, seeded bank of initial conditions (so the
comparison is paired: controller A and controller B see the identical initial
condition and the identical perturbation realization at each seed). Landing is
info['landed'] at the 3000-step episode cap, policies run deterministically.

Perturbations (see rocketrl/robustness.py for the wrappers, all default-OFF):
  * sensor_noise  - Gaussian obs noise, physically matched across controllers
  * init_scale    - wider initial conditions (init_x_range/vx/spin x multiplier)
  * wind          - constant horizontal force at the center of mass
  * mass          - whole-vehicle mass scaled at reset

Outputs (under --out-dir, default assets/robustness/):
  * results.csv            tidy (controller, perturbation, level, landing_rate, n)
  * robustness_<ptype>.png one degradation figure per perturbation
  * FINDINGS.md            summary table + figures + honest write-up

Re-runnable and parameterized:
  uv run python scripts/robustness.py --n-seeds 100
  uv run python scripts/robustness.py --n-seeds 60 --perturbations sensor_noise wind
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import gymnasium as gym

import rocketrl.envs  # noqa: F401  (registers RocketLander-v1)
from rocketrl.robustness import SensorNoiseWrapper, WindWrapper, MassScaleWrapper

EVAL_SEED_OFFSET = 1_000_000  # matches rocketrl.evaluate: disjoint from training seeds
MAX_STEPS = 3000

# PPO is discrete + static-normalized; SAC is continuous + static-normalized; the
# classical controllers are continuous and decode RAW observations. The specified
# best checkpoints (read-only).
PPO_CKPT = "runs/ppo_repro_s0/checkpoints/model_15000000_steps.zip"
SAC_CKPT = "runs/sacfd_s0/checkpoints/model_200000_steps.zip"

# Nominal (training) initial-condition knobs, from rocketrl.configs.
BASE_INIT_X = 0.3
BASE_INIT_VX = 0.4
BASE_INIT_SPIN = 1.4

# Controller display order (also the categorical color slot order in the figures).
CONTROLLERS = ["PPO", "SAC (demo)", "Classical careful", "Classical fast"]


# --------------------------------------------------------------------------- #
# Policies: a uniform act/reset interface over learned agents and controllers.
# --------------------------------------------------------------------------- #
class AgentPolicy:
    """Wrap an SB3 model as a deterministic act()/reset() policy."""

    def __init__(self, model):
        self._model = model

    def reset(self):
        pass

    def act(self, obs):
        action, _ = self._model.predict(obs, deterministic=True)
        return action


def load_policies():
    """Return {name: (policy, continuous, static_normalize)} for all four."""
    # The policies are tiny MLPs run one observation at a time; torch's default
    # multi-threading adds large per-call sync overhead that dominates (10-20x
    # slower) and can stall. Pin to a single thread for fast serial inference.
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO, SAC
    from rocketrl.controllers.lqr import (
        make_classical_controller, make_fast_controller)

    ppo = AgentPolicy(PPO.load(PPO_CKPT, device="cpu"))
    sac = AgentPolicy(SAC.load(SAC_CKPT, device="cpu"))
    return {
        "PPO": (ppo, False, True),
        "SAC (demo)": (sac, True, True),
        "Classical careful": (make_classical_controller(), True, False),
        "Classical fast": (make_fast_controller(), True, False),
    }


# --------------------------------------------------------------------------- #
# Env construction per (controller, perturbation, level).
# --------------------------------------------------------------------------- #
def build_env(continuous: bool, static_normalize: bool, perturbation: str,
              level: float) -> gym.Env:
    """Build the env for one controller under one perturbation level.

    init_scale is realized through the env's own init knobs (not a wrapper); the
    other three are default-off wrappers. Level-0 / x1.0 is an exact nominal
    no-op for every perturbation.
    """
    init_x, init_vx, init_spin = BASE_INIT_X, BASE_INIT_VX, BASE_INIT_SPIN
    if perturbation == "init_scale":
        init_x, init_vx, init_spin = (BASE_INIT_X * level,
                                      BASE_INIT_VX * level,
                                      BASE_INIT_SPIN * level)

    env = gym.make("RocketLander-v1", continuous=continuous,
                   max_episode_steps=MAX_STEPS, static_normalize=static_normalize,
                   init_x_range=init_x, init_vx_random=init_vx, init_spin=init_spin)

    if perturbation == "sensor_noise":
        env = SensorNoiseWrapper(env, std=level, obs_is_normalized=static_normalize)
    elif perturbation == "wind":
        env = WindWrapper(env, wind_mag=level)
    elif perturbation == "mass":
        env = MassScaleWrapper(env, mass_scale=level)
    return env


def landing_rate(policy, env: gym.Env, seeds: list[int]) -> float:
    landed = 0
    for s in seeds:
        policy.reset()
        obs, info = env.reset(seed=s)
        terminated = truncated = False
        while not (terminated or truncated):
            obs, _r, terminated, truncated, info = env.step(
                np.asarray(policy.act(obs), dtype=np.float32))
        landed += int(info.get("landed", False))
    return landed / len(seeds)


# --------------------------------------------------------------------------- #
# Sweep definitions. Level 0 / 1.0 is nominal for each (computed once, reused).
# --------------------------------------------------------------------------- #
@dataclass
class Sweep:
    key: str
    title: str
    xlabel: str
    levels: list[float]
    nominal_level: float


def default_sweeps() -> dict[str, Sweep]:
    return {
        "sensor_noise": Sweep(
            "sensor_noise", "Sensor noise robustness",
            "Observation noise std (static-normalized units)",
            [0.0, 0.02, 0.05, 0.1, 0.2], 0.0),
        "init_scale": Sweep(
            "init_scale", "Wider initial conditions",
            "Initial-condition difficulty multiplier (x training spread)",
            [1.0, 1.25, 1.5], 1.0),
        "wind": Sweep(
            "wind", "Wind robustness",
            "Constant horizontal wind force (Box2D units; vehicle weight ~230)",
            [0.0, 10.0, 20.0, 30.0, 40.0], 0.0),
        "mass": Sweep(
            "mass", "Vehicle mass mismatch",
            "Vehicle mass multiplier (x nominal 22.96 kg)",
            [0.9, 0.95, 1.0, 1.05, 1.1, 1.15], 1.0),
    }


# --------------------------------------------------------------------------- #
# Figures.
# --------------------------------------------------------------------------- #
# Colorblind-safe categorical slots (validated: worst adjacent CVD dE 11.2, floor
# band, mitigated by distinct marker shapes = secondary encoding). Fixed order.
SERIES_COLOR = {
    "PPO": "#2a78d6",                # blue
    "SAC (demo)": "#eb6834",         # orange
    "Classical careful": "#008300",  # green
    "Classical fast": "#4a3aa7",     # violet
}
SERIES_MARKER = {
    "PPO": "o",
    "SAC (demo)": "s",
    "Classical careful": "^",
    "Classical fast": "D",
}
# A self-contained light "card" surface reads on both light and dark pages.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e1e0d9"


def make_figure(sweep: Sweep, results: dict[str, list[float]], out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = np.asarray(sweep.levels, dtype=float)
    for name in CONTROLLERS:
        y = np.asarray(results[name], dtype=float) * 100.0
        ax.plot(x, y, color=SERIES_COLOR[name], marker=SERIES_MARKER[name],
                markersize=7, linewidth=2.0, label=name,
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3)
        # Direct label at the right end (secondary identity cue beyond color).
        ax.annotate(name, (x[-1], y[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=SERIES_COLOR[name])

    ax.set_title(sweep.title, color=INK, fontsize=13, pad=10, loc="left")
    ax.set_xlabel(sweep.xlabel, color=INK_MUTED, fontsize=10)
    ax.set_ylabel("Landing rate (%)", color=INK_MUTED, fontsize=10)
    ax.set_ylim(-3, 103)
    ax.set_xticks(x)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    leg = ax.legend(loc="lower left", frameon=True, fontsize=8.5,
                    facecolor=SURFACE, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(INK)
    # Headroom on the right for the direct labels.
    ax.set_xlim(x[0] - (x[-1] - x[0]) * 0.02, x[-1] + (x[-1] - x[0]) * 0.22)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Markdown write-up.
# --------------------------------------------------------------------------- #
def write_findings(path: str, sweeps: dict[str, Sweep],
                   all_results: dict[str, dict[str, list[float]]],
                   n_seeds: int, fig_names: dict[str, str]):
    lines: list[str] = []
    lines.append("# Robustness and out-of-distribution generalization\n")
    lines.append(
        f"Every controller was evaluated on the identical, shared bank of "
        f"{n_seeds} seeded initial conditions (seeds {EVAL_SEED_OFFSET}.."
        f"{EVAL_SEED_OFFSET + n_seeds - 1}), deterministically, at the "
        f"{MAX_STEPS}-step cap; landing is `info['landed']`. Each perturbation "
        f"is applied identically to every controller at a given seed (shared "
        f"realization), and defaults off so nominal evaluation is unchanged. "
        f"Sensor noise is injected in the static-normalized observation space "
        f"for all controllers (the raw-obs classical controllers receive the "
        f"same noise scaled by `STATIC_OBS_STD`), so the physical corruption is "
        f"identical; applying the std directly in raw-obs units instead would "
        f"corrupt the classical controllers ~10-20x harder and is not what is "
        f"reported here.\n")

    # One table per perturbation.
    for key, sweep in sweeps.items():
        if key not in all_results:
            continue
        res = all_results[key]
        lines.append(f"\n## {sweep.title}\n")
        lines.append(f"![{sweep.title}]({fig_names[key]})\n")
        header = "| Controller | " + " | ".join(
            f"{lvl:g}" for lvl in sweep.levels) + " |"
        sep = "|" + "---|" * (len(sweep.levels) + 1)
        lines.append(header)
        lines.append(sep)
        for name in CONTROLLERS:
            cells = " | ".join(f"{v * 100:.0f}%" for v in res[name])
            lines.append(f"| {name} | {cells} |")
        lines.append(f"\n*x-axis: {sweep.xlabel}. Cells are landing rate "
                     f"(% of {n_seeds} seeds).*\n")

    lines.append("\n## Findings\n")
    lines.append("<!-- Honest interpretation written below; see tables/figures. -->\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-seeds", type=int, default=100)
    p.add_argument("--seed-offset", type=int, default=EVAL_SEED_OFFSET)
    p.add_argument("--out-dir", type=str, default="assets/robustness")
    p.add_argument("--perturbations", nargs="+", default=None,
                   help="subset of: sensor_noise init_scale wind mass")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [args.seed_offset + i for i in range(args.n_seeds)]

    sweeps = default_sweeps()
    if args.perturbations:
        sweeps = {k: v for k, v in sweeps.items() if k in args.perturbations}

    print("loading policies ...")
    policies = load_policies()

    # Nominal landing rate per controller (computed ONCE, reused as every
    # sweep's level-0 / x1.0 point).
    print("computing nominal landing rates ...")
    nominal: dict[str, float] = {}
    for name, (pol, cont, sn) in policies.items():
        env = build_env(cont, sn, "none", 0.0)
        nominal[name] = landing_rate(pol, env, seeds)
        env.close()
        print(f"  nominal {name:20s} {nominal[name] * 100:5.1f}%")

    all_results: dict[str, dict[str, list[float]]] = {}
    fig_names: dict[str, str] = {}

    for key, sweep in sweeps.items():
        print(f"\n== sweep: {key} ==")
        res: dict[str, list[float]] = {name: [] for name in CONTROLLERS}
        for lvl in sweep.levels:
            for name, (pol, cont, sn) in policies.items():
                if lvl == sweep.nominal_level:
                    rate = nominal[name]
                else:
                    env = build_env(cont, sn, key, lvl)
                    rate = landing_rate(pol, env, seeds)
                    env.close()
                res[name].append(rate)
                print(f"  {key} lvl={lvl:<6g} {name:20s} {rate * 100:5.1f}%")
        all_results[key] = res
        fig_names[key] = f"robustness_{key}.png"
        make_figure(sweep, res, os.path.join(args.out_dir, fig_names[key]))
        print(f"  wrote {fig_names[key]}")

    # Tidy CSV.
    csv_path = os.path.join(args.out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["perturbation", "level", "controller", "landing_rate", "n_seeds"])
        for key, res in all_results.items():
            for i, lvl in enumerate(sweeps[key].levels):
                for name in CONTROLLERS:
                    w.writerow([key, lvl, name, f"{res[name][i]:.4f}", args.n_seeds])
    print(f"\nwrote {csv_path}")

    write_findings(os.path.join(args.out_dir, "FINDINGS.md"),
                   sweeps, all_results, args.n_seeds, fig_names)
    print(f"wrote {os.path.join(args.out_dir, 'FINDINGS.md')}")


if __name__ == "__main__":
    main()
