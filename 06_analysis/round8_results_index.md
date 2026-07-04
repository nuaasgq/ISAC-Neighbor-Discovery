# Round8 Targeted Follow-Up Results Index

Date: 2026-07-05

## Purpose

Round8 contains targeted follow-up jobs launched after the round7 audit. These jobs are not new training runs; they evaluate the existing `N=10`, 10-degree trained policy and missing baselines in reviewer-sensitive regimes.

## Status

| Experiment | Status | Raw output | Archived output | Notes |
|---|---|---|---|---|
| N=100 mobility missing baselines | complete | `05_simulation/results_raw/round8_n100_multimobility_missing_baselines_600slot` | `06_analysis/paper_tables/round8_n100_multimobility_missing_baselines_600slot` | Adds SkyOrbs-like and vanilla RL without ISAC for four mobility models and 10/15-degree beams. |
| N=100 B=15 error profiles | complete | full: `05_simulation/results_raw/round8_error_profiles_b15_gm_rw_600slot`; quick: `05_simulation/results_raw/round8_error_profiles_b15_gm_rw_quick` | full: `06_analysis/paper_tables/round8_error_profiles_b15_gm_rw_600slot`; quick: `06_analysis/paper_tables/round8_error_profiles_b15_gm_rw_quick` | Evaluates B=15 under Gauss-Markov and random-walk mobility with ISAC error profiles. |

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
The corresponding full-baseline supplement figures are generated under `06_analysis/paper_figures/round8_n100_multimobility_full_baseline/`:

- `full_baseline_discovery_n100_b10.png`
- `full_baseline_discovery_n100_b15.png`
- `full_baseline_lambda2_n100_b10.png`
- `full_baseline_lambda2_n100_b15.png`
- `full_baseline_collision_penalized_n100_b10.png`
- `full_baseline_collision_penalized_n100_b15.png`

## Promotion Decision

Use this result as supplementary baseline-completeness evidence for the mobility section. It does not change the manuscript's main mobility conclusion: the proposed method is strongest under Gauss-Markov and random-walk mobility, while random-direction and random-waypoint remain stress regimes.

## B=15 Error-Profile Full Snapshot

The full B=15 error-profile sweep completed with three seeds for Gauss-Markov and random-walk mobility. It compares full ISAC, one-slot delayed ISAC, and improved no-ISAC.

| Mobility | Pfa | Pmd | Offset | ISAC disc. | Std. | no-ISAC disc. | ISAC lambda2 | ISAC coll. | Coll.-penalized |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gauss-Markov | 0.00 | 0.00 | 0.0 | 0.5477 | 0.0037 | 0.0037 | 29.0615 | 8769.7 | 0.1980 |
| Gauss-Markov | 0.01 | 0.05 | 0.5 | 0.5852 | 0.0028 | 0.0037 | 32.7013 | 6548.7 | 0.2522 |
| Gauss-Markov | 0.05 | 0.15 | 1.0 | 0.5421 | 0.0106 | 0.0037 | 34.4420 | 3840.0 | 0.3053 |
| Gauss-Markov | 0.10 | 0.30 | 1.5 | 0.5272 | 0.0087 | 0.0037 | 35.8572 | 2380.7 | 0.3560 |
| Random walk | 0.00 | 0.00 | 0.0 | 0.4414 | 0.0092 | 0.0041 | 25.1352 | 840.0 | 0.3773 |
| Random walk | 0.01 | 0.05 | 0.5 | 0.3477 | 0.0058 | 0.0041 | 18.3021 | 497.0 | 0.3160 |
| Random walk | 0.05 | 0.15 | 1.0 | 0.2704 | 0.0084 | 0.0041 | 14.1573 | 209.3 | 0.2595 |
| Random walk | 0.10 | 0.30 | 1.5 | 0.2253 | 0.0013 | 0.0041 | 10.1517 | 141.0 | 0.2191 |

Interpretation: B=15 keeps strong raw discovery and connectivity under Gauss-Markov even with configured errors, but collision counts are high. Under random-walk mobility, error severity causes a clearer raw-discovery decline, although full ISAC remains far above the no-ISAC baseline.

The quick one-seed fallback remains archived at `06_analysis/paper_tables/round8_error_profiles_b15_gm_rw_quick`, but it is superseded by the full three-seed sweep for quantitative reporting.
