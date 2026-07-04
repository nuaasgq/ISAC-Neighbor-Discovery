# Round8 Targeted Follow-Up Results Index

Date: 2026-07-05

## Purpose

Round8 contains targeted follow-up jobs launched after the round7 audit. These jobs are not new training runs; they evaluate the existing `N=10`, 10-degree trained policy and missing baselines in reviewer-sensitive regimes.

## Status

| Experiment | Status | Raw output | Archived output | Notes |
|---|---|---|---|---|
| N=100 mobility missing baselines | complete | `05_simulation/results_raw/round8_n100_multimobility_missing_baselines_600slot` | `06_analysis/paper_tables/round8_n100_multimobility_missing_baselines_600slot` | Adds SkyOrbs-like and vanilla RL without ISAC for four mobility models and 10/15-degree beams. |
| N=100 B=15 error profiles | quick complete; full run still running | full: `05_simulation/results_raw/round8_error_profiles_b15_gm_rw_600slot`; quick: `05_simulation/results_raw/round8_error_profiles_b15_gm_rw_quick` | quick: `06_analysis/paper_tables/round8_error_profiles_b15_gm_rw_quick` | Evaluates B=15 under Gauss-Markov and random-walk mobility with ISAC error profiles. |

## Mobility Missing-Baseline Snapshot

The missing-baseline sweep confirms that the mobility-boundary result is not caused by omitting SkyOrbs-like or vanilla RL baselines. Across all tested mobility models and 10/15-degree beams, these two baselines remain near-zero in finite-time discovery and fail to form connected discovered graphs.

| Mobility | Beam | Baseline | Discovery | Empty scan | Lambda2 |
|---|---:|---|---:|---:|---:|
| Gauss-Markov | 10 deg | SkyOrbs-like | 0.0007 | 0.9020 | 0.0000 |
| Gauss-Markov | 10 deg | RL no-ISAC | 0.0006 | 0.9012 | 0.0000 |
| Random walk | 10 deg | SkyOrbs-like | 0.0007 | 0.9014 | 0.0000 |
| Random walk | 10 deg | RL no-ISAC | 0.0003 | 0.9010 | 0.0000 |
| Random direction | 10 deg | SkyOrbs-like | 0.0004 | 0.9006 | 0.0000 |
| Random direction | 10 deg | RL no-ISAC | 0.0002 | 0.9016 | 0.0000 |
| Random waypoint | 10 deg | SkyOrbs-like | 0.0001 | 0.9037 | 0.0000 |
| Random waypoint | 10 deg | RL no-ISAC | 0.0007 | 0.9032 | 0.0000 |

The compact merged table for all protocols is generated at `06_analysis/paper_tables/round8_n100_multimobility_full_baseline/combined_aggregate_metrics.csv`.

## Promotion Decision

Use this result as supplementary baseline-completeness evidence for the mobility section. It does not change the manuscript's main mobility conclusion: the proposed method is strongest under Gauss-Markov and random-walk mobility, while random-direction and random-waypoint remain stress regimes.

## B=15 Error-Profile Quick Snapshot

The quick B=15 fallback completed with one seed for Gauss-Markov and random-walk mobility, comparing full ISAC against improved no-ISAC. It is a sanity result while the full three-seed B=15 job continues.

| Mobility | Pfa | Pmd | Offset | ISAC discovery | no-ISAC discovery | ISAC lambda2 | ISAC collisions | Collision-penalized |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gauss-Markov | 0.00 | 0.00 | 0.0 | 0.5481 | 0.0030 | 30.2564 | 8230.0 | 0.2058 |
| Gauss-Markov | 0.01 | 0.05 | 0.5 | 0.5828 | 0.0030 | 35.1516 | 6055.0 | 0.2622 |
| Gauss-Markov | 0.05 | 0.15 | 1.0 | 0.5271 | 0.0030 | 32.0803 | 3642.0 | 0.3037 |
| Gauss-Markov | 0.10 | 0.30 | 1.5 | 0.5152 | 0.0030 | 36.1082 | 2162.0 | 0.3585 |
| Random walk | 0.00 | 0.00 | 0.0 | 0.4503 | 0.0038 | 25.6022 | 899.0 | 0.3811 |
| Random walk | 0.01 | 0.05 | 0.5 | 0.3475 | 0.0038 | 19.9646 | 540.0 | 0.3133 |
| Random walk | 0.05 | 0.15 | 1.0 | 0.2782 | 0.0038 | 12.2877 | 236.0 | 0.2655 |
| Random walk | 0.10 | 0.30 | 1.5 | 0.2240 | 0.0038 | 9.3605 | 141.0 | 0.2178 |

Interpretation: B=15 keeps strong raw discovery and connectivity under Gauss-Markov even with configured errors, but collision counts are high. Under random-walk mobility, error severity causes a clearer raw-discovery decline, although full ISAC remains far above the no-ISAC baseline.
