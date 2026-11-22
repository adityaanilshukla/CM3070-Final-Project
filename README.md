# Rocket Lander: Learned vs. Classical Control (CM3070)

[![CI](https://github.com/adityaanilshukla/CM3070-Final-Project/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/adityaanilshukla/CM3070-Final-Project/actions/workflows/ci.yml)

## What this project is

This project studies how to land a simulated rocket booster softly and uprightly on a narrow pad, and it compares two ways of building a controller for that task. The first is reinforcement learning, where an agent learns to fly by trial and error from a reward signal; we use Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC). The second is classical feedback control, where a human writes the control rules directly; we use a cascaded PID controller in two forms, one tuned for careful reliability and one tuned for speed. All of them steer the same physics simulation from a wide range of starting positions, velocities, and spins.

Beyond the comparison, the project investigates knowledge transfer between the two approaches in both directions: distilling the strategy a trained RL agent discovered into a hand-written controller, and seeding a struggling RL agent with demonstrations from a working controller. The code is a reproducible rebuild of an earlier version, on a current Gymnasium and Stable-Baselines3 stack, with seeded evaluation and continuous integration.

## Landings

Deterministic policies landing at full difficulty. Flame size tracks the engine thrust; the engine cuts off once both legs are down.

<table>
  <tr>
    <th align="center">PPO (learned)</th>
    <th align="center">Classical, fast (PPO-distilled)</th>
    <th align="center">Classical, careful</th>
    <th align="center">SAC (demo-seeded)</th>
  </tr>
  <tr>
    <td align="center" width="25%"><img src="assets/gifs/ppo.gif" width="170" alt="PPO agent landing"></td>
    <td align="center" width="25%"><img src="assets/gifs/fast.gif" width="170" alt="Fast PPO-distilled controller landing"></td>
    <td align="center" width="25%"><img src="assets/gifs/classical.gif" width="170" alt="Careful classical controller landing"></td>
    <td align="center" width="25%"><img src="assets/gifs/sac.gif" width="170" alt="Demonstration-seeded SAC landing"></td>
  </tr>
  <tr>
    <td align="center">~9 s committed descent</td>
    <td align="center">~11 s, flies the learned strategy by hand</td>
    <td align="center">~20 s, careful descent</td>
    <td align="center">rescued from 0% by demonstrations</td>
  </tr>
</table>

The second controller is hand-coded, yet it flies the fast descent strategy distilled from the PPO agent, so it lands about twice as quickly as the careful controller at comparable reliability.

## Results

All figures are measured at full difficulty under a seeded evaluation (100 or more held-out seeds, a 3000-step episode cap, deterministic policies). A landing counts only when both legs are down and the rocket has settled at low speed on the pad.

| Controller | Landing rate | Time to land | Fuel (RCS) | Smoothness |
|---|---|---|---|---|
| PPO (on-policy RL) | ~91% | ~9 s | ~0.00 | n/a (discrete) |
| Classical, careful (cascaded PID) | ~89% | ~22 s | ~1.0 | smooth |
| Classical, fast (PPO-distilled) | ~86-90% | ~11 s | ~0.6 | smooth |
| SAC (off-policy RL) | 0% vanilla, ~70% demo-seeded | ~19 s | ~0.6 | smooth |

## Key findings

1. The rebuilt PPO agent reaches about 91% at full difficulty on a seeded, reproducible pipeline, matching the earlier version's result.
2. Learned control and classical control land about as reliably as each other, roughly 91% against 89%, rather than one approach dominating.
3. The learned agent is far more efficient: PPO lands about 2.6 times faster on near-zero thruster fuel, because the reward values time and fuel. The careful controller, tuned only for reliability, trades speed and fuel for precision.
4. Knowledge transfers from RL to classical control. PPO's fast hoverslam descent reduces to a two-parameter braking rule, which turns a hand-written controller into one that lands roughly twice as fast and more smoothly, at comparable reliability. There is a measured ceiling: a hand-tuned controller cannot be both fast and as reliable as PPO, because its gimbal slews far slower than the learned policy's discrete controls allow.
5. Transfer from classical control back to RL works, but it is harder, and SAC still trails. On its own SAC never lands: it fails to discover the sparse landing by exploration, and faces a discount dilemma where one discount setting makes it hover forever and the other makes it dive and crash. Seeding its replay memory with the controller's landing demonstrations makes landings appear but then erode, because demonstrations constrain its actions and not its value estimates. Adding a potential-based reward term that rewards tracking the good descent profile reshapes the value landscape and makes the landings persist. The result is a policy that lands about 70% of the time, up from 0%, though still below PPO and the hand-coded controllers. The landing policy also peaks early in training and erodes if trained too long, so the reported figure is from the best checkpoint.
6. The learned agents generalize to conditions they never trained on better than the classical controllers in some respects and worse in others. Under observation noise the learned agents stay reliable while the classical controllers collapse; under a lighter-than-nominal vehicle the classical controllers fail while PPO is near-perfect, and under a heavier vehicle the reverse holds. See [`assets/robustness/FINDINGS.md`](assets/robustness/FINDINGS.md).

## Setup

Requires [`uv`](https://docs.astral.sh/uv/). The GPU path uses a CUDA 12.8 PyTorch build; CPU also works.

```bash
uv sync                               # create the venv and install everything
uv run python scripts/check_env.py    # sanity-check the environment
```

## Usage

```bash
# Train
uv run python -m rocketrl.train --algo ppo --repro --seed 0      # faithful PPO reproduction
uv run python -m rocketrl.train --algo sac --seed 0              # SAC

# Evaluate on a seeded bank with confidence intervals
uv run python -m rocketrl.evaluate --controllers random lqr --n-episodes 200 --seed 0

# Record landing GIFs
uv run python -m rocketrl.record_gifs --classical --seed 6 --out-dir assets/gifs
```

## Repository layout

| Path | Purpose |
|------|---------|
| `rocketrl/envs/` | Gymnasium RocketLander environment (continuous and discrete) |
| `rocketrl/controllers/` | classical controllers (careful PID, PPO-distilled fast) and the RL-policy adapter |
| `rocketrl/train.py`, `configs.py` | training entrypoint and dataclass experiment configs |
| `rocketrl/evaluate.py` | paired seeded evaluation with Wilson and bootstrap confidence intervals |
| `rocketrl/record_gifs.py` | landing-GIF recorder |
| `paper/` | the write-up (LaTeX source and compiled PDF) |
| `.github/workflows/ci.yml` | continuous integration (environment, conformance, controller, and harness checks) |

A longer write-up is in [`paper/main.pdf`](paper/main.pdf).
