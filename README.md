# Rocket Lander: Learned vs. Classical Control (CM3070)

[![CI](https://github.com/adityaanilshukla/CM3070-Final-Project/actions/workflows/ci.yml/badge.svg?branch=showcase-rebuild)](https://github.com/adityaanilshukla/CM3070-Final-Project/actions/workflows/ci.yml)

A simulation study of rocket soft-landing control that compares **learned controllers** (reinforcement learning: PPO and SAC) against **hand-designed controllers** (classical feedback control) on the same task, and then studies **knowledge transfer between the two paradigms**. A simulated booster must land upright and gently on a narrow pad from a wide range of initial offsets, velocities, and spins.

This branch is a ground-up rebuild of an earlier version on a modern, reproducible stack (Gymnasium + Stable-Baselines3 + PyTorch, managed with `uv`), with a seeded evaluation harness and continuous integration.

## Landings

Deterministic policies landing at full difficulty (hover-slam-style for PPO, smooth descent for the classical controller). Rendered with the simulator's own renderer; flame size tracks actual engine thrust.

<table>
  <tr>
    <th align="center">PPO (learned)</th>
    <th align="center">Classical, fast (PPO-distilled)</th>
    <th align="center">Classical, careful</th>
  </tr>
  <tr>
    <td align="center" width="33%"><img src="assets/gifs/ppo.gif" width="200" alt="PPO agent landing"></td>
    <td align="center" width="33%"><img src="assets/gifs/fast.gif" width="200" alt="Fast PPO-distilled controller landing"></td>
    <td align="center" width="33%"><img src="assets/gifs/classical.gif" width="200" alt="Careful classical controller landing"></td>
  </tr>
  <tr>
    <td align="center">~9 s committed descent</td>
    <td align="center">~11 s, flies the learned strategy by hand</td>
    <td align="center">~20 s, careful descent</td>
  </tr>
</table>

> The middle controller is **hand-coded**: it flies the fast "hover-slam" strategy *distilled* from the PPO agent, so it lands about twice as fast as the careful controller at comparable reliability. A GIF of the demonstration-seeded **SAC** agent will be added if that experiment yields a reliable lander.

## Results

All figures at **full difficulty**, strict seeded evaluation (100+ held-out seeds, 3000-step episode cap, deterministic policies), landing defined as both legs down and settled at low speed on the pad.

| Controller | Landing rate | Time to land | Fuel (RCS) | Smoothness |
|---|---|---|---|---|
| **PPO** (on-policy RL) | **~91%** | ~9 s | ~0.00 | n/a (discrete) |
| **Classical, careful** (cascaded PID) | **~89%** | ~22 s | ~1.0 | smooth |
| **Classical, fast** (PPO-distilled) | **~86-90%** | **~11 s** | ~0.6 | smooth |
| **SAC** (off-policy RL) | ~48% (baseline) | — | — | — |

SAC's number is under active investigation; see "Status" below.

## Key findings

1. **Faithful modern reproduction.** PPO on the rebuilt stack reaches ~91% at full difficulty, matching the original project's result, on a seeded, reproducible pipeline.
2. **Learned and classical control reach near-parity** on landing reliability (~91% vs ~89%) on this hard precision task, rather than one paradigm dominating.
3. **The learned agent is far more efficient**, PPO lands ~2.6x faster on near-zero thruster fuel, because the reward rewards time and fuel. The hand-designed controller, tuned purely for reliability, trades speed and fuel for precision.
4. **Knowledge transfer, RL to classical.** PPO's fast "hover-slam" strategy was distilled into a two-parameter braking insight and implemented in a deterministic controller, giving a hand-coded controller that lands ~2x faster and smoother than the careful one, at comparable reliability. This answers the question "can a deterministic controller be improved using behavioral insight from an RL policy?" in the affirmative, with a measured, explained ceiling.
5. **Knowledge transfer, classical to RL** (in progress). SAC fails to discover the sparse landing by exploration alone; seeding its replay buffer with the controller's demonstrations is being used to bootstrap it.

## Research questions

- How do on-policy (PPO) and off-policy (SAC) deep RL compare against principled classical control on a hard precision-landing task, in reliability, speed, and fuel?
- Can a deterministic controller be improved by distilling behavioral insight from a trained RL policy?
- Can a struggling RL agent be bootstrapped from a working controller's demonstrations?

## Setup

Requires [`uv`](https://docs.astral.sh/uv/). The GPU path uses a CUDA 12.8 PyTorch build (for Blackwell / RTX 50-series); CPU also works.

```bash
uv sync                      # create the venv and install everything (pinned in uv.lock)
uv run python scripts/check_env.py   # sanity-check the environment
```

## Usage

```bash
# Train (examples)
uv run python -m rocketrl.train --algo ppo --repro --seed 0      # faithful PPO reproduction
uv run python -m rocketrl.train --algo sac --seed 0              # SAC

# Evaluate controllers on a seeded bank with confidence intervals
uv run python -m rocketrl.evaluate --controllers random lqr --n-episodes 200 --seed 0

# Record landing GIFs
uv run python -m rocketrl.record_gifs --classical --seed 6 --out-dir assets/gifs
```

## Repository layout

| Path | Purpose |
|------|---------|
| `rocketrl/envs/` | Gymnasium RocketLander environment (continuous and discrete) |
| `rocketrl/controllers/` | classical controllers (careful PID, PPO-distilled fast) + RL-policy adapter |
| `rocketrl/train.py`, `configs.py` | training entrypoint and dataclass experiment configs |
| `rocketrl/evaluate.py` | paired seeded evaluation with Wilson / bootstrap CIs and significance tests |
| `rocketrl/record_gifs.py` | landing-GIF recorder |
| `scripts/` | environment checks and utilities |
| `.github/workflows/ci.yml` | CI (environment, env-conformance, controller, and harness smoke tests) |

## Status

Active development on the `showcase-rebuild` branch. In progress: refining the PPO-distilled fast controller, and a demonstration-seeded (SACfD-style) SAC run to lift SAC above its exploration-limited baseline. The `main` branch holds the original (legacy) version.
