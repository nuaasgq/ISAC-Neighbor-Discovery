# Round9 Targeted Stress Results Index

Date: 2026-07-05

## Purpose

Round9 adds one targeted stress check requested by the experiment audit: a full five-baseline comparison for the extreme `N=100`, 3-degree narrow-beam case. This is not a new training run. It evaluates the existing `N=10`, 10-degree trained policy without fine-tuning.

## Experiment

| Experiment | Status | Raw output | Archived output | Figures |
|---|---|---|---|---|
| N=100, 3-degree, five-baseline stress check | complete | `05_simulation/results_raw/round9_n100_b3_full_baselines_600slot` | `06_analysis/paper_tables/round9_n100_b3_full_baselines_600slot` | `06_analysis/paper_figures/round9_n100_b3_full_baselines_600slot` |

Configuration:

- Mobility: Gauss-Markov.
- Range mode: single-hop.
- Scale mode: density-preserving.
- Horizon: 600 slots.
- Seeds: `20290704`, `20291713`, `20292722`.
- Protocols: uniform random, SkyOrbs-like skip scan, RL without ISAC, improved RL without ISAC, improved RL with ISAC.

## Snapshot

| Protocol | Discovery | Std. | Empty scan | Lambda2 | Collision-penalized |
|---|---:|---:|---:|---:|---:|
| Uniform random | 0.0000 | 0.0000 | 0.9870 | 0.0000 | 0.0000 |
| SkyOrbs-like | 0.0000 | 0.0000 | 0.9865 | 0.0000 | 0.0000 |
| RL no-ISAC | 0.0001 | 0.0001 | 0.9868 | 0.0000 | 0.0001 |
| Improved no-ISAC | 0.0000 | 0.0000 | 0.9871 | 0.0000 | 0.0000 |
| Improved ISAC | 0.0131 | 0.0007 | 0.9418 | 0.0000 | 0.0130 |

## Interpretation

The `N=100`, 3-degree setting remains an extreme finite-horizon stress regime. ISAC still reduces empty scanning and produces a small nonzero discovery rate, but it does not form a connected discovered graph within 600 slots. Use this result to bound the 3--30 degree claim, not to advertise performance at 3 degrees.
