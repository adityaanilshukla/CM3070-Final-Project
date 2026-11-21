# Rocket Lander — Showcase Plan for a Graduate ML Application

Target: a portfolio-grade RL study that reads like a mini-paper. Audience: ML/AI research admissions. Budget: ~2-4 weeks. Direction: **SAC as the headline agent, PPO as the on-policy comparison, a correct model-based controller (LQR) as the classical reference**, on a **continuous-action** rocket soft-landing task.

The current repo is a BSc comparison of one PPO run vs a buggy FSM with invalid metrics. The goal is not to patch it but to turn it into a small, honest, reproducible comparative study with a clear research question, multi-seed statistics, and a robustness analysis. That is what distinguishes a showcase from a coursework submission.

---

## 1. Research framing (do this first, in writing)

State a question and a hypothesis up front; the whole project is the answer.

> **Question.** On a continuous-control rocket soft-landing task, how do an off-policy actor-critic (SAC) and an on-policy policy-gradient method (PPO) compare against a model-based optimal controller (LQR), in terms of final landing performance, sample efficiency, and robustness to disturbances and sensor noise?

> **Hypotheses.** (H1) SAC reaches a given success rate in fewer environment steps than PPO. (H2) Both RL agents exceed the linearized LQR controller once the dynamics leave the linear regime (high tilt, strong wind), but LQR is competitive near nominal. (H3) RL robustness to unseen wind/noise depends on whether it was trained with domain randomization.

This framing is the single biggest lever on how the work is perceived. "I trained PPO on a gym env" is coursework; "I tested three control paradigms under distribution shift with proper statistics" is research-shaped.

---

## 2. Task redesign: continuous actions + principled reward

The current env is `Discrete(7)` (gimbal/throttle/thruster nudges). Real rocket landing is continuous, SAC needs continuous actions, and this is a more credible task.

- **Action space.** `Box` of `[main_throttle ∈ [0,1], gimbal ∈ [-g_max, g_max], side_thruster ∈ [-1,1]]`. Map linearly to the existing physics forces in `rocket_lander.py`.
- **Observation.** Keep the physical state (x, y, vx, vy, angle, angular velocity, leg contacts). **Normalize with a `VecNormalize`/`RunningMeanStd` wrapper, not the hardcoded `MEAN`/`VAR` baked into the env** (that opaque constant is the root of the FSM unit bug). Normalization must be observable and reproducible.
- **Reward.** Document every term: shaping toward the pad (distance, velocity, angle), fuel/throttle penalty, a terminal success bonus, crash penalty. Write the reward function in the report and run a reward ablation (Section 6).
- **Gymnasium API.** `reset() -> (obs, info)`, `step() -> (obs, reward, terminated, truncated, info)`. Put `info['landed']`, `info['crashed']`, `info['touchdown_speed']`, `info['fuel_used']` in `info` so evaluation never scrapes stdout again.

Keep a discrete-action variant available only if you want a PPO-discrete-vs-continuous ablation; it is optional.

---

## 3. Infrastructure and hardware (the migration)

This unblocks everything and the GPU.

- **Framework.** Port to **stable-baselines3** (PyTorch). SB3 has `SAC` and `PPO` with the same API, which makes the comparison clean. Delete all TF1 `Session`/`ConfigProto` code.
- **Python/env.** Python 3.11, managed with `uv` or conda; commit a lockfile. Drop the Python 3.7 pin.
- **Blackwell GPU.** Install a PyTorch wheel built for **CUDA 12.8+/cu13** (the RTX 5060 Ti is sm_120; older wheels fail with "no kernel image"). Assert at startup: `torch.cuda.get_device_capability() == (12, 0)` and `torch.cuda.is_available()`.
- **Vectorized envs.** `SubprocVecEnv` with `n_envs=16` for PPO rollout collection (the real speedup on this CPU, not the GPU). SAC uses a single env + replay buffer; its gradient updates are where the GPU earns its place.
- **Config system.** One `dataclass`/YAML config per experiment (algorithm, seed, hyperparameters, env options). No magic numbers in code. This is what makes multi-seed runs and ablations tractable and is itself a reproducibility signal.
- **Logging.** TensorBoard + per-env `Monitor` with **unique filenames per env** (the shared `Monitor_Log` filename is why the current training CSV is corrupted). Log eval metrics to structured CSV/JSON keyed by (algo, seed).
- **Seeding.** Seed Python, NumPy, Torch, and the env's `self.np_random` from the config seed. Switch env resets off global `np.random`.

---

## 4. The three controllers

**SAC (headline).** Off-policy, sample-efficient, automatic entropy tuning, continuous actions, and the GPU accelerates its actor/critic updates over the replay buffer. This is the "better suited to the task and hardware" choice. Config: `buffer_size=1e6`, `batch_size=256`, `tau=0.005`, `gamma=0.99`, `learning_starts=10_000`, `train_freq=1`, `gradient_steps=1`, `ent_coef="auto"`, `device="cuda"`.

**PPO (on-policy comparison).** Keeps your existing investment and gives the on-policy vs off-policy axis. Config: `SubprocVecEnv(n_envs=16)`, `n_steps=512`, `batch_size=2048`, `n_epochs=10`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.01`, `lr=3e-4` (linear schedule). `device="cpu"` is usually faster for this 64x64 MLP.

**LQR (classical reference).** Replace the broken FSM entirely. Linearize the rocket dynamics about the hover/landing equilibrium, discretize, solve the discrete algebraic Riccati equation (`scipy.linalg.solve_discrete_are`) for the gain `K`, apply `u = -Kx` with the continuous action mapping. This is principled, reproducible, has no hand-tuned magic numbers, and gives a legitimate "model-based optimal control" baseline that reads far better than a PID for a research audience. If deriving the linearization is too costly, fall back to a **single correctly-tuned PID/FSM** with documented gains, but LQR is the stronger showcase.

> For an ML/AI research audience the classical controller is supporting cast: get it correct and principled, do not over-invest in MPC.

---

## 5. Evaluation protocol (this is where the rigor lives)

The current eval is unseeded, unpaired, stochastic-for-PPO, and times Python instead of physics. Rebuild it:

- **Multi-seed training.** 5-10 seeds per RL algorithm. Report learning curves as **mean with 95% CI (or ± std) shaded bands** over seeds, x-axis = environment steps. This single plot is the credibility centerpiece.
- **Paired, seeded test set.** A fixed bank of N≈200 held-out initial conditions. **Every controller is evaluated on the identical seeds**, so differences are controller differences, not luck. RL evaluated **deterministically** (`deterministic=True`).
- **Correct metrics**, all derived from simulation steps, not wall-clock:
  - Success rate (landed within pad bounds and velocity/angle limits) with a **Wilson or bootstrap CI**.
  - Landing precision = `|x|` at touchdown, split by outcome (never average crashes with landings), sampled at the true terminal frame for all controllers.
  - Touchdown speed, final tilt, fuel/throttle effort.
  - Control smoothness = `std` or `mean|u[t]-u[t-1]|` (jerk-like), **not** magnitude. Resample all controllers to a common rate first.
- **Statistical tests.** Two-proportion z-test (or Fisher) for success rate; bootstrap CIs and Mann-Whitney U for the continuous, non-normal metrics. Report effect sizes, not just p-values.
- **Sample efficiency.** Steps-to-threshold (e.g. steps to reach 80% success) per algorithm, averaged over seeds, with CIs. Directly tests H1.

---

## 6. The research differentiators (what makes it stand out)

Pick at least one; both if time allows. These turn a comparison into a contribution.

- **Robustness / distribution shift (strongly recommended).** Evaluate all three controllers under perturbations never seen at nominal: wind force sweeps, Gaussian sensor noise on observations, mass/thrust variation, wider initial-condition ranges. Plot **degradation curves** (success rate vs perturbation magnitude) per controller. This is the most interesting result and tests H2/H3. Pair it with a **domain-randomization** training variant for SAC to show robustness is trainable.
- **Residual RL (optional, high-ceiling).** Train an RL policy that outputs a *correction* on top of the LQR action (`u = u_LQR + π_θ(s)`). This unifies the classical and learned threads, is a recognized technique, and gives a clean narrative: "the learned residual recovers the performance LQR loses outside the linear regime." Strong, memorable story for a research application.
- **Reward / design ablations.** Systematically remove reward terms and observation normalization; show which design choices actually drive performance. Ablations are a hallmark of research maturity.

---

## 7. Visualization

- Learning curves with CI bands, all algorithms on one axis, shared scale.
- A single **results table** (success %, precision, touchdown speed, fuel, smoothness) with CIs, controllers as rows.
- Robustness degradation curves (one figure, one line per controller).
- Trajectory overlays (SAC vs PPO vs LQR on the same initial condition), equal aspect ratio, pad/start markers, units.
- Colorblind-safe palette; never red+green together. Consistent model color across every figure.
- A short demo GIF/MP4 of a SAC landing for the README.

---

## 8. The report (mini-paper, ~6-8 pages)

This is the deliverable admissions actually read. Structure: Abstract, Introduction/motivation, brief Related work (deep RL for control, SAC/PPO, model-based control), Task & dynamics, Methods (the three controllers + reward), Experimental setup (seeds, env steps, hardware), Results (learning curves, table, stats), Robustness analysis, Ablations, Discussion, **Limitations**, Reproducibility appendix (one command to reproduce). Write Limitations honestly: state the sim-only scope and any residual confounds. Honesty reads as maturity.

---

## 9. Repo hygiene / reproducibility

- One-command reproduction: `make train`, `make eval`, `make figures`, or a single `run.py --config ...`.
- Committed configs, lockfile, fixed seeds, pretrained checkpoints, and model cards (what each was trained on).
- README as a landing page: the research question, headline result figure, the demo GIF, and exact repro steps. Not the current setup-only README.
- Clean git history and a tagged `v1.0-showcase`.

---

## 10. Timeline (≈3 weeks nominal, compresses to 2 or extends to 4)

- **Week 1 — Foundation.** SB3 + Gymnasium migration; continuous action space + documented reward; config/seeding/logging infra; Blackwell PyTorch verified on GPU. Milestone: SAC and PPO both train and a smoke-test learning curve rises.
- **Week 2 — Experiments.** LQR derivation; multi-seed training for SAC and PPO; the corrected paired evaluation harness; core metrics + statistics. Milestone: the main results table and CI learning-curve figure exist.
- **Week 3 — Differentiator + writeup.** Robustness sweep (and/or residual RL); ablations; all figures finalized; write the report; repo hygiene and demo. Milestone: paper draft + reproducible repo.
- **Week 4 (if available) — Polish.** Second differentiator, more seeds for tighter CIs, proofread, external read.

---

## 11. Risks and cut lines (if time compresses)

- **If LQR derivation stalls:** fall back to a correctly-tuned, documented PID/FSM. Do not ship the current tautological FSM.
- **If multi-seed is too slow:** drop to 5 seeds, 1M-step budgets; SAC's sample efficiency makes this feasible. Vectorized PPO keeps wall-clock down.
- **Protect at all costs:** the paired/seeded evaluation, the CI learning curve, and the honest Limitations section. Cut a differentiator before cutting the statistics. A smaller study done rigorously beats a larger one done loosely.

---

## What carries over from the review

All the correctness fixes from `REVIEW_FINDINGS.md` are absorbed into the phases above: `deterministic=True` eval, seeding off global `np.random`, `info['landed']`, step-based timing, `std`-based smoothness, per-env monitor filenames, and killing the hardcoded normalization. In the rebuild they stop being bug fixes and become part of a correct-by-construction harness.
