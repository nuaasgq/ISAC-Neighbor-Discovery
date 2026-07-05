# Phase-6 Final Long-Evaluation Results - 2026-07-06

## Campaign

- Campaign: `phase6_final_long_eval_b10_b30_10ep_stoch`
- Training setting for all policies: `N=10`, `10 deg`, `300 slots/episode`
- Test setting: zero-shot `N=100`, `3000 slots/episode`, stochastic decentralized execution
- Beamwidths: `10, 15, 30 deg`
- Episodes per point: `10`
- Ranges: `communication_range_m = sensing_range_m = 900`
- Aggregated method-comparison table:
  `06_analysis/paper_tables/marl/phase6_final_long_eval_b10_b30_10ep_stoch_method_comparison/marl_method_comparison.csv`
- Method-comparison figures:
  `06_analysis/paper_figures/marl/phase6_final_long_eval_b10_b30_10ep_stoch_method_comparison/`

All nine evaluation commands and all four aggregation commands returned `0`.

## Main Results

| Method | Beamwidth | Discovery | CPD | Lambda2 | Collisions |
|---|---:|---:|---:|---:|---:|
| legacy_shared | 10 | 0.3958 | 0.0817 | 18.238 | 19677.4 |
| collision_reward | 10 | 0.3958 | 0.0953 | 18.542 | 16036.9 |
| contention_actor | 10 | 0.3700 | 0.1726 | 17.118 | 5685.6 |
| legacy_shared | 15 | 0.4415 | 0.0340 | 22.266 | 59696.0 |
| collision_reward | 15 | 0.4446 | 0.0368 | 23.812 | 54989.1 |
| contention_actor | 15 | 0.4333 | 0.0795 | 21.653 | 22177.0 |
| legacy_shared | 30 | 0.3236 | 0.0087 | 11.419 | 180310.9 |
| collision_reward | 30 | 0.3259 | 0.0105 | 10.784 | 149563.2 |
| contention_actor | 30 | 0.3591 | 0.0185 | 13.888 | 91801.4 |

CPD means `collision_penalized_discovery_rate`.

## Relative To Legacy

| Method | Beamwidth | CPD gain | Collision reduction | Raw discovery change | Lambda2 change |
|---|---:|---:|---:|---:|---:|
| collision_reward | 10 | +16.60% | 18.50% | +0.02% | +1.67% |
| collision_reward | 15 | +8.24% | 7.88% | +0.71% | +6.94% |
| collision_reward | 30 | +20.92% | 17.05% | +0.70% | -5.55% |
| contention_actor | 10 | +111.20% | 71.11% | -6.51% | -6.14% |
| contention_actor | 15 | +133.62% | 62.85% | -1.85% | -2.76% |
| contention_actor | 30 | +113.72% | 49.09% | +10.98% | +21.62% |

## Interpretation

The phase-6 results support the paper's current mechanism claim:

- `collision_reward` improves CPD and reduces collisions without changing the actor network.
- `contention_actor` provides the main network-structure innovation: it strongly suppresses contention in the `N=100` transfer setting.
- The strongest claim is not universal raw-discovery dominance. At `10 deg` and `15 deg`, contention control trades a small amount of raw discovery for much better collision-aware discovery. At `30 deg`, it improves raw discovery, CPD, lambda2, and collisions together.
- This is now a defensible MARL + ISAC cross-layer result for the core `10/15/30 deg` transfer regime.

## Paper Use

Recommended main-figure set from this campaign:

- `marl_method_comparison_collision_penalized_discovery_rate.png`
- `marl_method_comparison_collision_count.png`
- `marl_method_comparison_collisions_per_discovery_censored.png`
- `marl_method_comparison_discovery_rate.png`
- `marl_method_comparison_lambda2.png`

Recommended wording:

The contention-aware ISAC-MARL policy reduces collision load by `49.09%` to
`71.11%` and increases collision-penalized discovery by `111.20%` to `133.62%`
over the legacy shared ISAC-MAPPO baseline across `10/15/30 deg` beams under
zero-shot transfer from `N=10` training to `N=100` testing.

## Remaining Gaps

- `3 deg` and `5 deg` remain stress-boundary tests and should be run separately with `--max-workers 1`.
- The current campaign uses one trained checkpoint per method. A submission-grade extension should add independent training seeds if time permits.
- Deterministic decoding is not part of this main table. If included later, report it separately from stochastic decentralized execution.

