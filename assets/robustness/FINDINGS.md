# Robustness and out-of-distribution generalization

Every controller was evaluated on the identical, shared bank of 100 seeded initial conditions (seeds 1000000..1000099), deterministically, at the 3000-step cap; landing is `info['landed']`. Each perturbation is applied identically to every controller at a given seed (shared realization), and defaults off so nominal evaluation is unchanged. Sensor noise is injected in the static-normalized observation space for all controllers (the raw-obs classical controllers receive the same noise scaled by `STATIC_OBS_STD`), so the physical corruption is identical; applying the std directly in raw-obs units instead would corrupt the classical controllers ~10-20x harder and is not what is reported here.


## Sensor noise robustness

![Sensor noise robustness](robustness_sensor_noise.png)

| Controller | 0 | 0.02 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|
| PPO | 91% | 94% | 96% | 97% | 95% |
| SAC (demo) | 70% | 72% | 72% | 72% | 28% |
| Classical careful | 89% | 86% | 50% | 0% | 0% |
| Classical fast | 87% | 79% | 48% | 9% | 3% |

*x-axis: Observation noise std (static-normalized units). Cells are landing rate (% of 100 seeds).*


## Wider initial conditions

![Wider initial conditions](robustness_init_scale.png)

| Controller | 1 | 1.25 | 1.5 |
|---|---|---|---|
| PPO | 91% | 68% | 58% |
| SAC (demo) | 70% | 52% | 41% |
| Classical careful | 89% | 72% | 63% |
| Classical fast | 87% | 69% | 56% |

*x-axis: Initial-condition difficulty multiplier (x training spread). Cells are landing rate (% of 100 seeds).*


## Wind robustness

![Wind robustness](robustness_wind.png)

| Controller | 0 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|
| PPO | 91% | 81% | 45% | 17% | 3% |
| SAC (demo) | 70% | 45% | 31% | 13% | 15% |
| Classical careful | 89% | 70% | 42% | 49% | 3% |
| Classical fast | 87% | 68% | 45% | 48% | 4% |

*x-axis: Constant horizontal wind force (Box2D units; vehicle weight ~230). Cells are landing rate (% of 100 seeds).*


## Vehicle mass mismatch

![Vehicle mass mismatch](robustness_mass.png)

| Controller | 0.9 | 0.95 | 1 | 1.05 | 1.1 | 1.15 |
|---|---|---|---|---|---|---|
| PPO | 94% | 93% | 91% | 74% | 24% | 0% |
| SAC (demo) | 4% | 19% | 70% | 59% | 51% | 7% |
| Classical careful | 0% | 35% | 89% | 76% | 63% | 34% |
| Classical fast | 25% | 44% | 87% | 72% | 63% | 46% |

*x-axis: Vehicle mass multiplier (x nominal 22.96 kg). Cells are landing rate (% of 100 seeds).*


## Findings

<!-- Honest interpretation written below; see tables/figures. -->
