# CM3070 Rocket Lander — Code & Report Review

Consolidated findings from a three-way review: PPO model, FSM controller, and the report/analysis/plots. Agents cross-checked each other, so shared root causes below are confirmed from more than one angle.

## What the project is

A RocketLander (Box2D) simulation comparing a trained **PPO** agent against a hand-built **FSM/PID** controller, with an evaluation + plotting pipeline that stands in for the dissertation's results section. Stack: `tensorflow==1.13.2` + `stable-baselines==2.10.2` + `gym==0.21.0`, Python 3.7. All of that is end-of-life.

---

## The headline problem: the comparison is not valid as it stands

The project's entire thesis is "PPO vs FSM," but the comparison currently cannot support a clean conclusion. Four issues, each confirmed by multiple agents:

1. **PPO is evaluated stochastically, the FSM deterministically.** `run_ppo_model.py:48` calls `model.predict(obs)` without `deterministic=True`, so PPO samples its policy every step while the FSM is a fixed PID. You are not comparing the two controllers at their best behavior.
2. **Nothing is seeded, so the two agents face different scenarios.** The env draws initial x and angular velocity from the *global* `np.random` (`rocket_lander.py:226,294`) instead of `self.np_random`, and neither evaluator seeds. The "100 vs 100" result is unpaired: 100 different FSM scenarios vs 100 different PPO scenarios, irreproducible.
3. **The reported timing metrics are physically invalid.** `time_taken_to_land` (~230 ms mean) is Python wall-clock of the run+analysis loop, not flight time; `response_time` (~120 µs) times a post-hoc list iteration over already-collected data. PPO pays neural-net inference cost the FSM does not, so these metrics are *biased toward the FSM* and measure CPU speed, not control. Do not present FSM as "faster to land."
4. **No statistical test.** PPO 91/100 vs FSM 97/100 is a 6-point gap at n=100 with no confidence interval and no significance test. It may be noise.

**Plus a fairness caveat that must land in the writeup:** the FSM baseline is under-validated (see B2–B4 below), so part of any PPO margin is inflated by FSM bugs, not an inherent limitation of classical control. The dissertation must state this in Limitations rather than concluding "RL beats classical control" cleanly.

---

## PPO model (Agent A)

**High**
- `run_ppo_model.py:48` — add `deterministic=True` to `model.predict`.
- Seed the env identically in both evaluators; switch env resets to `self.np_random` (`rocket_lander.py:226,294`).
- Expose success programmatically: set `info['landed']=True` at `rocket_lander.py:408` instead of printing `"LANDED!!!!!!!!!"` and grepping stdout (`eval_ppo.py:50-55`).
- `train_ppo_model.py:120` — `nminibatches=256` on a 4096 batch gives 16-sample minibatches; PPO updates are very noisy. Use `nminibatches=4` (SB2) / `batch_size≈2048` (SB3).

**Medium**
- `ppo_controller.py:21` hardcodes the **18M-step checkpoint**, not the final 20M model. Document which is "best" or load the final.
- `train_ppo_model.py:97` — checkpoint cleanup builds a path missing the `.zip` suffix `model.save` adds, so `os.path.exists` is always false; all 10 checkpoints are kept.
- `noptepochs=3` is low; 4–10 is typical.
- `SubprocVecEnv` is imported but unused; training actually runs `DummyVecEnv` over 4 envs **sequentially** (`train_ppo_model.py:8,49`).

**Low**
- Hardcoded `MEAN`/`VAR` normalization baked into the env (`rocket_lander.py:86-89`) from a prior run; fragile, undocumented.
- `from ppo_controller import *` star-imports env+model+register side effects.
- RAM metric is process RSS sampled once per episode; grows with Python list accumulation, not model memory.

---

## FSM controller (Agent B)

**High**
- **B1 — Thresholds are in the wrong units.** The env returns z-scored observations (`rocket_lander.py:422`), but the FSM treats `obs[]` as raw physical units everywhere (`fsm_controller.py:49,54,111,134,148`). So every threshold and gain (`abs(x)>0.1`, `y<=-1.3`, PID gains 1200/400) is actually in standard-deviation units. Worse, `land_rocket` aims angle→0 (`:148`), which in z-score space is raw angle = -0.016 rad, i.e. **not upright.**
- **B2 — Dead/tautological logic.** `moving_toward_landing_zone` always returns True (`abs(obs[0]) >= 0`, `:116`); `set_throttle` forces `target_vy=45` whenever `vy<0.1` because `abs(x) > 0` is always true (`:60`), overriding the intended branch. The FSM is near-constant-behavior — a partial strawman.
- **B3 — Control is bang-bang, not PID.** PID output is thresholded to a discrete ±0.1 nudge (`:66-71,98-103`); the "do nothing" branch almost never fires on floats, so it chatters every step. `Ki=0`, so the integral term is inert and the large gains are mostly cosmetic.
- **B4 — Metric sampling asymmetry.** The FSM logs one `flight_data` entry per control iteration but steps the env 2–3× per iteration (`:203,217–233`), while PPO logs one per `env.step`. All smoothness/deviation/throttle/action stats are computed on 2–3× coarser samples for the FSM.
- **B5 — Stale landing frame.** The FSM logs state at the top of the loop and exits mid-iteration, so the true touchdown frame is never recorded; `flight_data[-1]` is ~one iteration pre-contact (vs PPO's true terminal state). FSM landing precision is systematically stale.
- **B6 — "Smoothness" measures magnitude, not smoothness.** Both evals take max/min/avg of raw gimbal/throttle magnitude; no variance, no jerk. The FSM's chatter is invisible to it. Compute `std` or `mean|x[t]-x[t-1]|` and relabel.

**Medium**
- PID objects are module globals never reset between episodes; `previous_error` leaks across episodes.
- `within_landing_zone` has no else/return → returns None; `land_rocket` comments mislabel thrusters 4/5 as "gimbal."
- Pervasive undocumented magic numbers.

---

## Report / analysis / plots (Agent C)

**High**
- **C1** — Drop or redefine the invalid `time_taken_to_land` and `response_time` metrics (see headline #3). Use `len(flight_data) * dt`.
- **C2 — Throttle-smoothness stats box uses the wrong variable.** `plot_results.py:142-144` prints mean/std/var of `avg_deviations` instead of `avg_throttle_smoothness` — a copy-paste bug; the throttle figure shows stability numbers.
- **C3 — README is setup-only.** No research question, methodology, metric definitions, results, or reproduction steps (the eval/plot scripts are never mentioned). Needs Abstract, Research Question, Methodology, Results table, Discussion, Limitations.

**Medium**
- **No learning curve plotted.** `Monitor_Log.monitor.csv` holds ~7000 episodes of reward rising from ~-1.4 to ~0.8 over ~5.7h, but nothing plots it — the core RL convergence evidence is absent.
- **The training log is corrupted.** All 4 training envs share one Monitor filename (`train_ppo_model.py:42`), so their writes interleave/truncate (76 run-together reward rows). The curve requires re-training with unique per-env monitor paths; the current CSV cannot be reliably repaired.
- RAM comparison is confounded (FSM recreates env + `gc.collect()` per episode; PPO reuses one env).
- Plots are siloed per-model with independent y-scales; no side-by-side PPO-vs-FSM figure despite comparison being the thesis. Resample to a common rate first (because of B4).
- Red+green line plots are the worst colorblind combination.
- Axis unit labels ("radians") are likely wrong if `flight_data` carries normalized obs (ties to B1).

**Low**
- `x_landing_precision` is labeled "unitless" in the comment but appends " m" in code; plots signed final x under a "precision" title with no x=0 target; stale for FSM, mixes landings with crashes.
- Single-episode FSM titles lack the "(PPO)"-style model tag; trajectory plots lack equal aspect ratio and start/end/pad markers.

---

## Training faster on this machine (RTX 5060 Ti 16GB, CUDA 13.5, 16 cores)

**Can PPO use this GPU? Yes — but not on the current stack, and the GPU is not the real speedup.**

1. **The current TF1/SB2 stack cannot touch this GPU.** TF 1.13 predates Blackwell (sm_120) by a decade; there is no CUDA-13 kernel path. Reaching the GPU *requires* migrating to **stable-baselines3 (PyTorch)** first. Changing the RL algorithm would not help — GPU support is a framework issue, not a PPO issue.
2. **The bottleneck is env stepping, not the GPU.** RocketLander is a Box2D sim with a tiny 64×64 MLP; PPO here is CPU/rollout-bound. **The dominant lever is your 16 cores:** replace the sequential `DummyVecEnv` with `SubprocVecEnv`, `n_envs=16`. That is a multiple-x rollout speedup, far larger than any GPU effect at this network size. For a 64×64 MLP, `device="cpu"` often beats GPU once per-step transfer overhead is counted; only go to CUDA if you enlarge the network/batch.
3. **Blackwell setup gotcha:** install a PyTorch wheel built for CUDA 12.8+/cu13 (older wheels fail with "no kernel image"). Verify `torch.cuda.get_device_capability() == (12, 0)`.
4. **Recommended SB3 config:** `n_envs=16` (SubprocVecEnv), `n_steps=512`, `batch_size=2048`, `n_epochs=10`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.01`, `lr=3e-4` (or linear schedule).

**Optional, unrelated to the GPU:** if wall-clock still matters after vectorizing, an off-policy method like **SAC** typically reaches good continuous-control performance in far fewer environment steps than PPO. That is a sample-efficiency tradeoff worth a paragraph in the report, but it is not required and would mean rebuilding the comparison.

---

## Suggested order of work

1. **Migrate to SB3 + Gymnasium** (unblocks GPU, modern Python, and naturally fixes the stochastic-eval, success-info, and minibatch issues).
2. **Fix the evaluation harness:** seed both evaluators identically for *paired* episodes, `deterministic=True` for PPO, expose `info['landed']`, align env lifecycle, derive timing from sim steps, redefine "smoothness" as variation. This is what makes the comparison publishable.
3. **Fix the FSM baseline** (B1–B3) so it is a fair opponent, then re-run.
4. **Re-train** with `SubprocVecEnv(n_envs=16)` and unique per-env monitor paths, then plot the learning curve.
5. **Rewrite the README into a report** with a results table, 95% CIs, a two-proportion significance test, side-by-side figures, and an explicit Limitations section noting the baseline caveat.
