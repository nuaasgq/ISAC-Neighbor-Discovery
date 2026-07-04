# Round7 Long-Run Results Index

Date: 2026-07-05

## Status

| Experiment | Status | Raw output | Archived output | Figure output |
|---|---|---|---|---|
| Long CEM training | complete | `05_simulation/results_raw/round7_long_cem_train_n10_b10_600slot` | `06_analysis/paper_tables/round7_long_cem_training` | `06_analysis/paper_figures/round7_long_cem_training` |
| Scale/beam grid | complete | `05_simulation/results_raw/round7_scale_beam_grid_light` | `06_analysis/paper_tables/round7_scale_beam_grid_light` | `06_analysis/paper_figures/round7_scale_beam_grid_light` |
| N=100 multi-mobility | complete | `05_simulation/results_raw/round7_n100_multimobility_600slot` | `06_analysis/paper_tables/round7_n100_multimobility_600slot` | `06_analysis/paper_figures/round7_n100_multimobility_600slot` |
| Error profiles | running | `05_simulation/results_raw/round7_error_profiles_light` | pending | pending |

## Training Summary

Round7 uses a longer 16-generation, 24-population CEM run at `N=10`, 10-degree beams, 600 slots, with three training seeds and three held-out seeds.

| Metric | Value |
|---|---:|
| Best training score | 91.5810 |
| Held-out score | 82.4714 |
| Held-out discovery rate | 0.7506 |
| Held-out mean delay | 314.70 slots |
| Held-out empty-scan ratio | 0.8972 |
| Held-out lambda2 | 4.2645 |

This run is useful as additional convergence evidence, but it is not promoted above the round3/round4 main evidence chain because its small-scale held-out score is lower than the earlier candidate training under a different 1200-slot setup.

## Scale/Beam Grid Snapshot

The completed scale/beam grid evaluates the round7 policy at `N=10,20,50,100`, beamwidths `3,5,10,15,30` degrees, Gauss-Markov mobility, density-preserving scaling, 600 slots, and two seeds.

`N=100`, proposed method:

| Beam | Discovery | Empty scan | Lambda2 | Collisions | Collision-penalized discovery | Discoveries / scan |
|---:|---:|---:|---:|---:|---:|---:|
| 3 deg | 0.0128 | 0.9456 | 0.0000 | 0.0 | 0.0128 | 0.0011 |
| 5 deg | 0.0908 | 0.8251 | 0.8905 | 16.0 | 0.0905 | 0.0077 |
| 10 deg | 0.3738 | 0.5437 | 12.1462 | 1125.0 | 0.3045 | 0.0321 |
| 15 deg | 0.5456 | 0.3958 | 27.4288 | 8887.0 | 0.1956 | 0.0468 |
| 30 deg | 0.4622 | 0.2623 | 27.6886 | 85032.5 | 0.0255 | 0.0397 |

Interpretation:

- Raw discovery is strongest at 15 degrees, consistent with the earlier main result.
- Collision-penalized discovery is strongest at 10 degrees, because 15- and 30-degree beams create far more collision attempts.
- 3- and 5-degree beams remain stress regimes.
- Round7 reinforces the need to report raw discovery together with collision-aware efficiency metrics.

## Promotion Decision

## N=100 Mobility Snapshot

The completed mobility sweep evaluates `N=100`, 10- and 15-degree beams, four mobility models, 600 slots, and three seeds.

Proposed method:

| Mobility | Beam | Discovery | Empty scan | Lambda2 | Collisions | Collision-penalized discovery |
|---|---:|---:|---:|---:|---:|---:|
| Gauss-Markov | 10 deg | 0.3700 | 0.5427 | 12.7548 | 1127.0 | 0.3013 |
| Gauss-Markov | 15 deg | 0.5477 | 0.3942 | 29.0615 | 8769.7 | 0.1980 |
| Random walk | 10 deg | 0.1822 | 0.7922 | 6.8288 | 99.0 | 0.1786 |
| Random walk | 15 deg | 0.4414 | 0.6716 | 25.1352 | 840.0 | 0.3773 |
| Random direction | 10 deg | 0.0777 | 0.3468 | 1.3726 | 930.3 | 0.0654 |
| Random direction | 15 deg | 0.1533 | 0.2818 | 4.4742 | 3891.7 | 0.0859 |
| Random waypoint | 10 deg | 0.0312 | 0.2179 | 0.0000 | 1619.7 | 0.0235 |
| Random waypoint | 15 deg | 0.0629 | 0.1595 | 1.1289 | 8047.7 | 0.0240 |

Interpretation:

- Round7 slightly improves the abrupt-mobility stress cases over the earlier round5 trend, but random-direction and random-waypoint remain weak.
- Random-walk transfer benefits from 15-degree beams and has much lower collision cost than Gauss-Markov at the same beamwidth.
- The existing manuscript boundary remains correct: the method is strongest when beam-cell occupancy evolves smoothly relative to the discovery horizon.

## Promotion Decision

Do not replace the existing main manuscript tables with round7 values yet. Use round7 as robustness and convergence support unless the pending error-profile job provides a clearly stronger result than the current round3/round4 evidence chain.
