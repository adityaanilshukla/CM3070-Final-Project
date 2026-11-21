"""Paired, seeded evaluation harness for the rocket soft-landing comparison.

Fixes relative to the old project's eval: every controller runs on the
IDENTICAL fixed bank of seeded initial conditions (env.reset(seed=...)), so
comparisons are paired rather than drawn from independent random episodes;
PPO is evaluated deterministically (same as the classical controller and
SAC) instead of stochastically; and all timing is derived from simulation
steps (steps / FPS), never time.time() wall-clock.

Usage:
    uv run python -m rocketrl.evaluate --controllers random --n-episodes 50
    uv run python -m rocketrl.evaluate --controllers random lqr --n-episodes 200 --seed 0
    uv run python -m rocketrl.evaluate --controllers random rl:runs/sac_s0:sac \
        --n-episodes 200

Controller specs accepted by --controllers:
    random              RandomController(seed=<--seed>)
    lqr                 rocketrl.controllers.lqr.make_classical_controller()
                         (imported lazily; skipped with a warning if the
                         module doesn't exist yet or fails to build)
    rl:<run_dir>:<algo>  rocketrl.controllers.rl.load_rl_controller(run_dir, algo)

Outputs, under results/eval/<tag>/ (tag defaults to a timestamp):
    episodes.csv   one row per (controller, episode)
    summary.json   per-controller aggregates (Wilson CI for success rate,
                   bootstrap CI for continuous metrics) and pairwise
                   significance tests + effect sizes between every pair of
                   evaluated controllers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from scipy import stats as scipy_stats

import rocketrl.envs  # noqa: F401  (registers RocketLander-v1)
from rocketrl.controllers import Controller, RandomController
from rocketrl.envs.rocket_lander import FPS

try:
    from rich.console import Console
    from rich.table import Table

    _HAVE_RICH = True
except ImportError:  # pragma: no cover - rich is a soft dependency for display only
    _HAVE_RICH = False

# A fixed additive offset so the evaluation seed bank never collides with the
# seeds rocketrl.train already uses (training env: seed..seed+n_envs-1;
# its own EvalCallback env: seed+10_000+rank). --seed shifts the whole fixed
# bank, but for a given --seed the bank (and thus every controller's episode
# set) is identical and reproducible run to run.
EVAL_SEED_OFFSET = 1_000_000


# --------------------------------------------------------------------------- #
# Per-episode record
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeResult:
    controller: str
    episode_idx: int
    seed: int
    landed: bool
    crashed: bool
    timeout: bool
    touchdown_speed: float
    touchdown_angle: float
    abs_x_landing: float
    fuel_used: float
    episode_steps: int
    episode_time_s: float
    smooth_gimbal: float
    smooth_throttle: float
    smooth_side: float
    smooth_overall: float


def run_episode(env: gym.Env, controller: Controller, seed: int, episode_idx: int) -> EpisodeResult:
    """Run one episode on a fixed seed and return its outcome + metrics.

    Terminal outcome fields (landed/crashed/touchdown_*/x_landing/fuel_used)
    come from `info`, populated by the env only when `terminated` is True
    (see rocketrl/envs/rocket_lander.py). If the episode instead times out
    (`truncated` without `terminated`), those keys are absent: we record
    landed=False, crashed=False, timeout=True, and leave touchdown_*/
    x_landing as NaN (there was no touchdown), but still recover fuel_used
    from the env's running counter since fuel is tracked every step
    regardless of how the episode ends.
    """
    controller.reset()
    obs, info = env.reset(seed=seed)

    terminated = False
    truncated = False
    actions: list[np.ndarray] = []
    steps = 0
    while not (terminated or truncated):
        action = np.asarray(controller.act(obs), dtype=np.float32)
        actions.append(action.copy())
        obs, _reward, terminated, truncated, info = env.step(action)
        steps += 1

    landed = bool(info.get("landed", False))
    crashed = bool(info.get("crashed", False))
    timeout = bool(truncated and not terminated)

    touchdown_speed = float(info.get("touchdown_speed", float("nan")))
    touchdown_angle = float(info.get("touchdown_angle", float("nan")))
    abs_x_landing = float(abs(info["x_landing"])) if "x_landing" in info else float("nan")
    fuel_used = float(info.get("fuel_used", getattr(env.unwrapped, "fuel_used", float("nan"))))

    acts = np.asarray(actions, dtype=np.float32)
    if len(acts) >= 2:
        diffs = np.abs(np.diff(acts, axis=0))
        smooth_gimbal, smooth_throttle, smooth_side = (float(x) for x in diffs.mean(axis=0))
        smooth_overall = float(diffs.mean())
    else:
        smooth_gimbal = smooth_throttle = smooth_side = smooth_overall = float("nan")

    return EpisodeResult(
        controller=controller.name,
        episode_idx=episode_idx,
        seed=seed,
        landed=landed,
        crashed=crashed,
        timeout=timeout,
        touchdown_speed=touchdown_speed,
        touchdown_angle=touchdown_angle,
        abs_x_landing=abs_x_landing,
        fuel_used=fuel_used,
        episode_steps=steps,
        episode_time_s=steps / FPS,
        smooth_gimbal=smooth_gimbal,
        smooth_throttle=smooth_throttle,
        smooth_side=smooth_side,
        smooth_overall=smooth_overall,
    )


# --------------------------------------------------------------------------- #
# Statistics: CIs, significance tests, effect sizes
# --------------------------------------------------------------------------- #
Z_95 = 1.959963984540054


def wilson_ci(successes: int, n: int, z: float = Z_95) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Robust at small n and
    at p near 0 or 1, unlike the normal (Wald) approximation."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return phat, max(0.0, center - half), min(1.0, center + half)


def _clean(x: Any) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float)
    return arr[~np.isnan(arr)]


def bootstrap_mean_ci(
    data: Any, n_boot: int = 10000, ci: float = 0.95, rng: Optional[np.random.Generator] = None
) -> tuple[float, float, float, int]:
    """Percentile bootstrap CI for the mean. Returns (mean, ci_lo, ci_hi, n)
    where n is the count of non-NaN samples actually used. NaNs (e.g. no
    successful landings to average touchdown speed over) propagate rather
    than silently becoming 0."""
    arr = _clean(data)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    if n == 1:
        return float(arr[0]), float("nan"), float("nan"), 1
    rng = rng if rng is not None else np.random.default_rng(0)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(arr.mean()), float(lo), float(hi), n


def bootstrap_diff_ci(
    a: Any, b: Any, n_boot: int = 10000, ci: float = 0.95, rng: Optional[np.random.Generator] = None
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for mean(a) - mean(b), resampling each group
    independently (unpaired)."""
    a_arr, b_arr = _clean(a), _clean(b)
    if len(a_arr) == 0 or len(b_arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng if rng is not None else np.random.default_rng(0)
    idx_a = rng.integers(0, len(a_arr), size=(n_boot, len(a_arr)))
    idx_b = rng.integers(0, len(b_arr), size=(n_boot, len(b_arr)))
    diffs = a_arr[idx_a].mean(axis=1) - b_arr[idx_b].mean(axis=1)
    lo, hi = np.percentile(diffs, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(a_arr.mean() - b_arr.mean()), float(lo), float(hi)


def cohens_d(a: Any, b: Any) -> float:
    a_arr, b_arr = _clean(a), _clean(b)
    na, nb = len(a_arr), len(b_arr)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_var = ((na - 1) * a_arr.var(ddof=1) + (nb - 1) * b_arr.var(ddof=1)) / (na + nb - 2)
    pooled_std = math.sqrt(pooled_var)
    if pooled_std == 0:
        return 0.0
    return float((a_arr.mean() - b_arr.mean()) / pooled_std)


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for a difference of two proportions."""
    if math.isnan(p1) or math.isnan(p2):
        return float("nan")
    return float(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))


def two_proportion_ztest(s1: int, n1: int, s2: int, n2: int) -> tuple[float, float]:
    """Pooled two-proportion z-test, two-sided. Returns (z, p)."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = s1 / n1, s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p = 2 * scipy_stats.norm.sf(abs(z))
    return float(z), float(p)


def mann_whitney(a: Any, b: Any) -> tuple[float, float, float]:
    """Returns (U statistic, two-sided p-value, rank-biserial effect size).

    scipy.stats.mannwhitneyu RAISES ValueError on an empty input, which WILL
    happen whenever a controller has zero successful landings (its
    success-only arrays, e.g. touchdown_speed, are empty) -- guard explicitly
    rather than letting that propagate as a crash.
    """
    a_arr, b_arr = _clean(a), _clean(b)
    if len(a_arr) == 0 or len(b_arr) == 0:
        return float("nan"), float("nan"), float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            u, p = scipy_stats.mannwhitneyu(a_arr, b_arr, alternative="two-sided")
    except ValueError:
        return float("nan"), float("nan"), float("nan")
    r = 1 - (2 * u) / (len(a_arr) * len(b_arr))
    return float(u), float(p), float(r)


# --------------------------------------------------------------------------- #
# Paired statistics (exploit the shared seed bank: rows_a[i] and rows_b[i]
# are the SAME initial condition, so differences can be taken per episode)
# --------------------------------------------------------------------------- #
MIN_PAIRS_FOR_WILCOXON = 6


def mcnemar_test(rows_a: list[EpisodeResult], rows_b: list[EpisodeResult]) -> dict[str, Any]:
    """Paired binary test for success rate across the shared seed bank.

    Builds the discordant cell counts of the paired 2x2 table:
        b = seeds where A landed but B did not,
        c = seeds where B landed but A did not.
    Only discordant pairs carry information under McNemar's null (that the two
    controllers are equally likely to win a discordant seed). Uses the exact
    binomial form for small discordant totals and the chi-square-with-
    continuity-correction form otherwise (statsmodels is not installed here,
    so this is implemented directly on scipy). Returns NaN with a note when
    there are no discordant pairs (e.g. both controllers land nothing).
    """
    n = min(len(rows_a), len(rows_b))
    both = sum(1 for i in range(n) if rows_a[i].landed and rows_b[i].landed)
    a_only = sum(1 for i in range(n) if rows_a[i].landed and not rows_b[i].landed)
    b_only = sum(1 for i in range(n) if rows_b[i].landed and not rows_a[i].landed)
    neither = n - both - a_only - b_only
    discordant = a_only + b_only

    result = {
        "test": "mcnemar",
        "paired": True,
        "n_pairs": n,
        "both_landed": both,
        "a_only_landed": a_only,
        "b_only_landed": b_only,
        "neither_landed": neither,
        "n_discordant": discordant,
    }

    if discordant == 0:
        result.update(
            statistic=float("nan"),
            p_value=float("nan"),
            method="none",
            note="no discordant pairs (A and B agree on every seed); McNemar undefined.",
        )
        return result

    if discordant < 25:
        # Exact binomial: P(X ~ Binom(discordant, 0.5) at least as extreme).
        k = min(a_only, b_only)
        res = scipy_stats.binomtest(k, discordant, 0.5, alternative="two-sided")
        result.update(statistic=float(min(a_only, b_only)), p_value=float(res.pvalue), method="exact_binomial")
    else:
        stat = (abs(a_only - b_only) - 1) ** 2 / discordant
        p = float(scipy_stats.chi2.sf(stat, df=1))
        result.update(statistic=float(stat), p_value=p, method="chi2_continuity")
    return result


def wilcoxon_paired(diffs: np.ndarray) -> dict[str, Any]:
    """Wilcoxon signed-rank test on per-episode paired differences (a - b).

    `diffs` must already be the aligned, NaN-free difference vector for the
    relevant pairing (all seeds, or the both-succeeded subset). Emits NaN with
    a note when there are too few pairs (< MIN_PAIRS_FOR_WILCOXON) or when all
    differences are zero (the test is undefined / uninformative there).
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n_pairs = int(len(diffs))
    median_diff = float(np.median(diffs)) if n_pairs > 0 else float("nan")

    if n_pairs < MIN_PAIRS_FOR_WILCOXON:
        return {
            "test": "wilcoxon_signed_rank",
            "paired": True,
            "n_pairs": n_pairs,
            "statistic": float("nan"),
            "p_value": float("nan"),
            "median_paired_diff": median_diff,
            "note": f"too few paired samples (n_pairs={n_pairs} < {MIN_PAIRS_FOR_WILCOXON}); test not computed.",
        }
    if np.all(diffs == 0):
        return {
            "test": "wilcoxon_signed_rank",
            "paired": True,
            "n_pairs": n_pairs,
            "statistic": float("nan"),
            "p_value": float("nan"),
            "median_paired_diff": median_diff,
            "note": "all paired differences are zero; signed-rank test undefined.",
        }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            stat, p = scipy_stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
    except ValueError as e:
        return {
            "test": "wilcoxon_signed_rank",
            "paired": True,
            "n_pairs": n_pairs,
            "statistic": float("nan"),
            "p_value": float("nan"),
            "median_paired_diff": median_diff,
            "note": f"wilcoxon could not be computed ({e}).",
        }
    return {
        "test": "wilcoxon_signed_rank",
        "paired": True,
        "n_pairs": n_pairs,
        "statistic": float(stat),
        "p_value": float(p),
        "median_paired_diff": median_diff,
    }


def _paired_diffs_all(
    rows_a: list[EpisodeResult], rows_b: list[EpisodeResult], field_name: str
) -> np.ndarray:
    """Per-seed differences (A - B) across ALL paired episodes, dropping any
    seed where either side's value is NaN."""
    n = min(len(rows_a), len(rows_b))
    diffs = []
    for i in range(n):
        va, vb = getattr(rows_a[i], field_name), getattr(rows_b[i], field_name)
        if not (math.isnan(va) or math.isnan(vb)):
            diffs.append(va - vb)
    return np.asarray(diffs, dtype=float)


def _paired_diffs_both_success(
    rows_a: list[EpisodeResult], rows_b: list[EpisodeResult], field_name: str
) -> np.ndarray:
    """Per-seed differences (A - B) over only the seeds where BOTH controllers
    landed -- the valid pairing for success-only metrics."""
    n = min(len(rows_a), len(rows_b))
    diffs = []
    for i in range(n):
        if rows_a[i].landed and rows_b[i].landed:
            va, vb = getattr(rows_a[i], field_name), getattr(rows_b[i], field_name)
            if not (math.isnan(va) or math.isnan(vb)):
                diffs.append(va - vb)
    return np.asarray(diffs, dtype=float)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _cont_stat(rows: list[EpisodeResult], field_name: str, n_boot: int, rng: np.random.Generator) -> dict[str, Any]:
    vals = [getattr(r, field_name) for r in rows]
    mean, lo, hi, n = bootstrap_mean_ci(vals, n_boot=n_boot, rng=rng)
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": n}


def aggregate_controller(rows: list[EpisodeResult], n_boot: int, rng: np.random.Generator) -> dict[str, Any]:
    n = len(rows)
    n_success = sum(1 for r in rows if r.landed)
    n_crash = sum(1 for r in rows if r.crashed)
    n_timeout = sum(1 for r in rows if r.timeout)
    p_hat, lo, hi = wilson_ci(n_success, n)
    succ_rows = [r for r in rows if r.landed]

    return {
        "n_episodes": n,
        "success_rate": {"p": p_hat, "ci_lo": lo, "ci_hi": hi, "n_success": n_success, "n": n},
        "crash_rate": {"p": (n_crash / n) if n else float("nan"), "n_crash": n_crash, "n": n},
        "timeout_rate": {"p": (n_timeout / n) if n else float("nan"), "n_timeout": n_timeout, "n": n},
        # Precision / touchdown kinematics: successful landings only, never
        # averaged together with crashes (per the eval spec).
        "touchdown_speed_success_only": _cont_stat(succ_rows, "touchdown_speed", n_boot, rng),
        "touchdown_angle_success_only": _cont_stat(succ_rows, "touchdown_angle", n_boot, rng),
        "precision_abs_x_landing_success_only": _cont_stat(succ_rows, "abs_x_landing", n_boot, rng),
        # Fuel and timing are defined over the whole episode regardless of
        # outcome (fuel is spent trying even when the episode ends in a crash
        # or timeout).
        "fuel_used": _cont_stat(rows, "fuel_used", n_boot, rng),
        "episode_time_s": _cont_stat(rows, "episode_time_s", n_boot, rng),
        "episode_time_s_success_only": _cont_stat(succ_rows, "episode_time_s", n_boot, rng),
        "smoothness_gimbal": _cont_stat(rows, "smooth_gimbal", n_boot, rng),
        "smoothness_throttle": _cont_stat(rows, "smooth_throttle", n_boot, rng),
        "smoothness_side": _cont_stat(rows, "smooth_side", n_boot, rng),
        "smoothness_overall": _cont_stat(rows, "smooth_overall", n_boot, rng),
    }


def _cont_pair(
    rows_a: list[EpisodeResult],
    rows_b: list[EpisodeResult],
    field_name: str,
    n_boot: int,
    rng: np.random.Generator,
    pairing: str,
) -> dict[str, Any]:
    """Compare a continuous metric between two controllers.

    Reports BOTH an unpaired reference test (Mann-Whitney U over each group's
    full sample, plus a bootstrap difference CI and Cohen's d) AND the
    paired-correct Wilcoxon signed-rank test over per-seed differences.

    `pairing` selects how episodes are matched for the paired test:
        "all"           -> pair across every seed (fuel, smoothness, timing);
                           the unpaired sample is likewise every episode.
        "both_success"  -> pair only over seeds where BOTH controllers landed
                           (touchdown kinematics, precision); the unpaired
                           sample is each controller's successful episodes.
    """
    if pairing == "both_success":
        a_vals = [getattr(r, field_name) for r in rows_a if r.landed]
        b_vals = [getattr(r, field_name) for r in rows_b if r.landed]
        diffs = _paired_diffs_both_success(rows_a, rows_b, field_name)
    elif pairing == "all":
        a_vals = [getattr(r, field_name) for r in rows_a]
        b_vals = [getattr(r, field_name) for r in rows_b]
        diffs = _paired_diffs_all(rows_a, rows_b, field_name)
    else:  # pragma: no cover - guard against a bad call site
        raise ValueError(f"unknown pairing {pairing!r}")

    diff, dlo, dhi = bootstrap_diff_ci(a_vals, b_vals, n_boot=n_boot, rng=rng)
    u, p, r_eff = mann_whitney(a_vals, b_vals)
    return {
        "pairing": pairing,
        # --- unpaired reference ---
        "unpaired": {
            "test": "mann_whitney_u",
            "paired": False,
            "mean_diff_a_minus_b": diff,
            "diff_ci_lo": dlo,
            "diff_ci_hi": dhi,
            "mannwhitney_u": u,
            "mannwhitney_p": p,
            "rank_biserial_r": r_eff,
            "cohens_d": cohens_d(a_vals, b_vals),
        },
        # --- paired-correct test ---
        "paired": wilcoxon_paired(diffs),
    }


def pairwise_stats(
    rows_a: list[EpisodeResult], rows_b: list[EpisodeResult], n_boot: int, rng: np.random.Generator
) -> dict[str, Any]:
    n1, n2 = len(rows_a), len(rows_b)
    s1 = sum(1 for r in rows_a if r.landed)
    s2 = sum(1 for r in rows_b if r.landed)
    p1 = (s1 / n1) if n1 else float("nan")
    p2 = (s2 / n2) if n2 else float("nan")
    z, p = two_proportion_ztest(s1, n1, s2, n2)
    diff, dlo, dhi = bootstrap_diff_ci(
        [float(r.landed) for r in rows_a], [float(r.landed) for r in rows_b], n_boot=n_boot, rng=rng
    )

    return {
        "success_rate": {
            "p_a": p1,
            "p_b": p2,
            "risk_diff_a_minus_b": diff,
            "diff_ci_lo": dlo,
            "diff_ci_hi": dhi,
            # Unpaired reference test.
            "unpaired": {
                "test": "two_proportion_z",
                "paired": False,
                "z": z,
                "p_value": p,
                "cohens_h": cohens_h(p1, p2),
                "note": "unpaired two-proportion z-test over success/failure counts.",
            },
            # Paired-correct test: McNemar on the discordant seeds.
            "paired": mcnemar_test(rows_a, rows_b),
        },
        # Touchdown kinematics + precision: paired over seeds where BOTH
        # controllers landed; unpaired reference over each controller's own
        # successful episodes.
        "touchdown_speed_success_only": _cont_pair(rows_a, rows_b, "touchdown_speed", n_boot, rng, "both_success"),
        "touchdown_angle_success_only": _cont_pair(rows_a, rows_b, "touchdown_angle", n_boot, rng, "both_success"),
        "precision_abs_x_landing_success_only": _cont_pair(
            rows_a, rows_b, "abs_x_landing", n_boot, rng, "both_success"
        ),
        # Fuel, timing, smoothness: paired over ALL seeds.
        "fuel_used": _cont_pair(rows_a, rows_b, "fuel_used", n_boot, rng, "all"),
        "episode_time_s": _cont_pair(rows_a, rows_b, "episode_time_s", n_boot, rng, "all"),
        "smoothness_overall": _cont_pair(rows_a, rows_b, "smooth_overall", n_boot, rng, "all"),
    }


# --------------------------------------------------------------------------- #
# Controller construction from CLI specs
# --------------------------------------------------------------------------- #
def build_controllers(specs: list[str], seed: int) -> list[Controller]:
    controllers: list[Controller] = []
    seen_names: set[str] = set()

    for spec in specs:
        controller: Optional[Controller] = None

        if spec == "random":
            controller = RandomController(seed=seed)

        elif spec == "lqr":
            try:
                from rocketrl.controllers.lqr import make_classical_controller
            except ImportError as e:
                warnings.warn(f"skipping 'lqr': rocketrl.controllers.lqr not importable ({e})")
                continue
            try:
                controller = make_classical_controller()
            except Exception as e:  # noqa: BLE001 - any build failure should be a skip, not a crash
                warnings.warn(f"skipping 'lqr': make_classical_controller() failed ({e})")
                continue

        elif spec.startswith("rl:"):
            parts = spec.split(":", 2)
            if len(parts) != 3:
                warnings.warn(f"skipping malformed controller spec {spec!r}; expected rl:<run_dir>:<algo>")
                continue
            _, run_dir, algo = parts
            try:
                from rocketrl.controllers.rl import load_rl_controller

                controller = load_rl_controller(run_dir, algo)
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"skipping {spec!r}: failed to load RL controller ({e})")
                continue

        else:
            warnings.warn(f"unknown controller spec {spec!r}; skipping")
            continue

        name = controller.name
        if name in seen_names:
            base, i = name, 2
            while f"{base}#{i}" in seen_names:
                i += 1
            warnings.warn(f"duplicate controller name {name!r}; renaming to {base}#{i}")
            controller.name = f"{base}#{i}"
            name = controller.name
        seen_names.add(name)
        controllers.append(controller)

    return controllers


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_csv(path: str, all_rows: dict[str, list[EpisodeResult]]) -> None:
    fieldnames = list(EpisodeResult.__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rows in all_rows.values():
            for r in rows:
                writer.writerow(asdict(r))


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {o!r}")


def _fmt_cont(stat: dict[str, Any]) -> str:
    if math.isnan(stat["mean"]):
        return "n/a (no data)"
    if math.isnan(stat["ci_lo"]):
        return f"{stat['mean']:.3f} (n={stat['n']}, CI n/a)"
    return f"{stat['mean']:.3f} [{stat['ci_lo']:.3f}, {stat['ci_hi']:.3f}] (n={stat['n']})"


def print_summary(aggregate: dict[str, Any], pairwise: dict[str, Any], out_dir: str) -> None:
    if _HAVE_RICH:
        console = Console()
        table = Table(title="Controller evaluation summary (95% CIs)")
        table.add_column("controller")
        table.add_column("success rate")
        table.add_column("touchdown speed (succ.)")
        table.add_column("touchdown |angle| (succ.)")
        table.add_column("|x_landing| (succ.)")
        table.add_column("fuel used")
        table.add_column("landing time (s)")
        table.add_column("smoothness (overall)")
        for name, agg in aggregate.items():
            sr = agg["success_rate"]
            sr_str = f"{sr['p']:.3f} [{sr['ci_lo']:.3f}, {sr['ci_hi']:.3f}] (n={sr['n']})"
            table.add_row(
                name,
                sr_str,
                _fmt_cont(agg["touchdown_speed_success_only"]),
                _fmt_cont(agg["touchdown_angle_success_only"]),
                _fmt_cont(agg["precision_abs_x_landing_success_only"]),
                _fmt_cont(agg["fuel_used"]),
                _fmt_cont(agg["episode_time_s"]),
                _fmt_cont(agg["smoothness_overall"]),
            )
        console.print(table)

        if pairwise:
            ptable = Table(title="Pairwise success rate: unpaired (z) vs paired (McNemar)")
            ptable.add_column("pair")
            ptable.add_column("risk diff (A-B) [95% CI]")
            ptable.add_column("z (unpaired)")
            ptable.add_column("p (unpaired)")
            ptable.add_column("discordant (a_only/b_only)")
            ptable.add_column("p (McNemar, paired)")

            def _p(x: float) -> str:
                return "nan" if math.isnan(x) else f"{x:.4f}"

            for pair, stats_ in pairwise.items():
                sr = stats_["success_rate"]
                up, pa = sr["unpaired"], sr["paired"]
                z_str = "nan" if math.isnan(up["z"]) else f"{up['z']:.3f}"
                ptable.add_row(
                    pair,
                    f"{sr['risk_diff_a_minus_b']:.3f} [{sr['diff_ci_lo']:.3f}, {sr['diff_ci_hi']:.3f}]",
                    z_str,
                    _p(up["p_value"]),
                    f"{pa['a_only_landed']}/{pa['b_only_landed']} (nd={pa['n_discordant']})",
                    _p(pa["p_value"]),
                )
            console.print(ptable)

            wtable = Table(title="Pairwise continuous metrics: Wilcoxon signed-rank (paired)")
            wtable.add_column("pair")
            wtable.add_column("metric")
            wtable.add_column("n_pairs")
            wtable.add_column("median diff (A-B)")
            wtable.add_column("p (Wilcoxon)")
            wtable.add_column("p (Mann-Whitney, unpaired)")
            cont_metrics = [
                "touchdown_speed_success_only",
                "touchdown_angle_success_only",
                "precision_abs_x_landing_success_only",
                "fuel_used",
                "episode_time_s",
                "smoothness_overall",
            ]
            for pair, stats_ in pairwise.items():
                for m in cont_metrics:
                    blk = stats_[m]
                    pr, un = blk["paired"], blk["unpaired"]
                    md = "nan" if math.isnan(pr["median_paired_diff"]) else f"{pr['median_paired_diff']:.3f}"
                    wtable.add_row(
                        pair, m, str(pr["n_pairs"]), md, _p(pr["p_value"]), _p(un["mannwhitney_p"])
                    )
            console.print(wtable)
    else:
        for name, agg in aggregate.items():
            print(f"[{name}] success_rate={agg['success_rate']}")
        for pair, stats_ in pairwise.items():
            print(f"[{pair}] success_rate={stats_['success_rate']}")

    print(f"\nwrote episodes.csv and summary.json under {out_dir}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Paired evaluation harness for rocket soft-landing controllers")
    p.add_argument(
        "--controllers",
        nargs="+",
        required=True,
        help="controller specs, e.g.: random lqr rl:runs/sac_s0:sac",
    )
    p.add_argument("--n-episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0, help="shifts the fixed eval seed bank (see EVAL_SEED_OFFSET)")
    p.add_argument("--max-episode-steps", type=int, default=1000)
    p.add_argument("--out-dir", type=str, default="results/eval")
    p.add_argument("--tag", type=str, default=None, help="output subdir name; default is a UTC timestamp")
    p.add_argument("--n-boot", type=int, default=10000, help="bootstrap resamples for CIs")
    args = p.parse_args()

    controllers = build_controllers(args.controllers, args.seed)
    if not controllers:
        raise SystemExit("no controllers to evaluate (all specs were skipped); see warnings above")

    seed_bank = [EVAL_SEED_OFFSET + args.seed + i for i in range(args.n_episodes)]

    env = gym.make("RocketLander-v1", continuous=True, max_episode_steps=args.max_episode_steps)

    all_rows: dict[str, list[EpisodeResult]] = {}
    try:
        for controller in controllers:
            rows = [run_episode(env, controller, seed, i) for i, seed in enumerate(seed_bank)]
            all_rows[controller.name] = rows
    finally:
        env.close()

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    write_csv(os.path.join(out_dir, "episodes.csv"), all_rows)

    rng = np.random.default_rng(args.seed)
    aggregate = {name: aggregate_controller(rows, args.n_boot, rng) for name, rows in all_rows.items()}

    pairwise: dict[str, Any] = {}
    names = list(all_rows.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pairwise[f"{a}_vs_{b}"] = pairwise_stats(all_rows[a], all_rows[b], args.n_boot, rng)

    summary = {
        "meta": {
            "n_episodes": args.n_episodes,
            "seed": args.seed,
            "eval_seed_offset": EVAL_SEED_OFFSET,
            "seed_bank_first": seed_bank[0],
            "seed_bank_last": seed_bank[-1],
            "max_episode_steps": args.max_episode_steps,
            "fps": FPS,
            "controllers": [c.name for c in controllers],
            "n_boot": args.n_boot,
            "pairing_note": (
                "Every controller was run on env.reset(seed=seed_bank[i]) for the "
                "same seed_bank, i.e. identical initial conditions per episode "
                "index across controllers (a paired design). Each 'pairwise' "
                "metric reports BOTH an unpaired reference test and the "
                "paired-correct test: success rate -> unpaired two-proportion "
                "z-test + paired McNemar (exact binomial for <25 discordant "
                "pairs, else chi-square with continuity correction); continuous "
                "metrics -> unpaired Mann-Whitney U + paired Wilcoxon signed-rank "
                "on per-seed differences. Success-only metrics "
                "(touchdown_speed/angle, |x_landing|) are paired ONLY over seeds "
                "where BOTH controllers landed (see each block's paired.n_pairs); "
                "all-episode metrics (fuel_used, episode_time_s, smoothness) are "
                "paired over every seed. Look under each metric's 'paired' and "
                "'unpaired' keys."
            ),
        },
        "aggregate": aggregate,
        "pairwise": pairwise,
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    print_summary(aggregate, pairwise, out_dir)


if __name__ == "__main__":
    main()
