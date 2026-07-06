# Phase9 Strict Shared No-ISAC MAPPO Results

Date: 2026-07-06

## Scope

This report summarizes the strict shared MAPPO no-ISAC replacement campaign required for a fair five-way MARL comparison.

- Campaign: `phase9_mappo_no_isac_strict_100ep_3seed`
- Method: `mappo_no_isac`
- Algorithm/network: shared MAPPO-style actor-critic, legacy reward
- ISAC boundary: `--disable-isac-features`, `env_protocol=structured_marl_no_isac`
- Training scale: `N=10`, `B=10 deg`, `300 slots/episode`, `5 ms/slot`
- Training budget: `100 episodes`, `3 seeds` (`20260731`, `20260732`, `20260733`)
- Aggregated tables: `06_analysis/paper_tables/marl/phase9_mappo_no_isac_strict_100ep_3seed_learning_curves/`
- Figures: `06_analysis/paper_figures/marl/phase9_mappo_no_isac_strict_100ep_3seed_learning_curves/`

The campaign produced `300` episode rows, `198` in-training evaluation rows, and `90000` step-level reward samples. The large step-level aggregate CSV is intentionally ignored by Git; the figures and compact tables are retained.

## Key Results

Final training rows:

| Run | Training step | Discovery rate | CPD | Lambda2 | Collisions | Episode return |
|---|---:|---:|---:|---:|---:|---:|
| seed20260731 | 30000 | 0.0000 | 0.0000 | 0.0000 | 0 | -35.9150 |
| seed20260732 | 30000 | 0.0000 | 0.0000 | 0.0000 | 0 | -35.6750 |
| seed20260733 | 30000 | 0.0000 | 0.0000 | 0.0000 | 0 | -40.8450 |

Last-10 training-episode aggregate:

| Metric | Mean | Std. |
|---|---:|---:|
| Discovery rate | 0.0000 | 0.0000 |
| Collision-penalized discovery | 0.0000 | 0.0000 |
| Lambda2 | 0.0000 | 0.0000 |
| Collision count | 0.0000 | 0.0000 |
| Episode return | -37.5670 | 2.3998 |

Final in-training evaluation at episode 100:

| Phase | Discovery rate | CPD | Lambda2 | Collisions | Episode return |
|---|---:|---:|---:|---:|---:|
| Deterministic | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -22.6994 |
| Stochastic | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -37.7683 |

## Interpretation

This is the fair shared-network counterpart to `contention_no_isac`: same small-scale training setting, same 100-episode budget, and a strict simulator boundary that prevents sensing actions from updating ISAC-derived occupancy belief.

The result is a clean lower-bound signal. Shared MAPPO without ISAC-derived candidate or occupancy assistance fails to discover any training-topology edge after 100 episodes across all three seeds. This should not be overstated as a proof of impossibility, but it strongly supports the paper's mechanism claim: under 3D narrow-beam blind search, slot-level MARL needs ISAC-derived beam evidence or another strong prior to escape the sparse-discovery regime.

## Paper Use

Use this campaign as:

1. the paper-grade `MAPPO w/o ISAC` training-curve evidence;
2. the checkpoint source for the five-way Phase9 transfer comparison;
3. a lower-bound comparator paired with `uniform_random` and `SkyOrbs-like`, not as the sole no-ISAC baseline.

Useful figures:

- `marl_step_reward_curve.png`
- `marl_episode_return_curve.png`
- `marl_episode_discovery_curve.png`
- `marl_eval_discovery_curve.png`
- `marl_policy_loss_curve.png`
- `marl_value_loss_curve.png`
- `marl_resource_rss_curve.png`

## Next Required Use

- The first checkpoint, `seed20260731`, is already being used in `phase9_fiveway_n100_b10_3000slot_10ep_stoch`.
- After the ongoing B=10 five-way run completes, rerun the full paper-grade five-way beam transfer at `B=10/15/30`.
- Use the full three-seed set for robustness or supplementary training-stability plots.
