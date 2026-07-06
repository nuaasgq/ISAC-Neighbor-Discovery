# Phase7 Strict No-ISAC MARL Results

Date: 2026-07-06

## Scope

This report summarizes the strict no-ISAC replacement campaign for the MARL comparison.

- Campaign: `phase7_contention_no_isac_strict_100ep_3seed`
- Training scale: `N=10`, `B=10 deg`, `300 slots/episode`, `5 ms/slot`
- Training budget: `100 episodes`, `3 seeds` (`20260731`, `20260732`, `20260733`)
- Algorithm: shared contention-aware MAPPO-style actor-critic without ISAC-derived features
- Environment protocol: `structured_marl_no_isac`
- Strict boundary: no-ISAC protocols do not update beam belief from `sense` actions.

Artifacts:

- Raw campaign: `05_simulation/results_raw/marl_campaign/phase7_contention_no_isac_strict_100ep_3seed/`
- Tables: `06_analysis/paper_tables/marl/phase7_contention_no_isac_strict_100ep_3seed_learning_curves/`
- Figures: `06_analysis/paper_figures/marl/phase7_contention_no_isac_strict_100ep_3seed_learning_curves/`

The run produced `90000` step-level records, `300` episode records, and `198` in-training evaluation records. The large step-level aggregate CSV is ignored by Git; the plots and compact episode/eval tables are retained.

## Key Results

Final training rows:

| Seed | Discovery rate | Collision-penalized discovery | Lambda2 | Collisions | Empty-scan ratio |
|---:|---:|---:|---:|---:|---:|
| 20260731 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.9838 |
| 20260732 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.9861 |
| 20260733 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.9889 |

Last-10 training-episode average:

| Method | Discovery rate | Collision-penalized discovery | Lambda2 | Collisions | Empty-scan ratio |
|---|---:|---:|---:|---:|---:|
| Contention-aware MAPPO w/o ISAC | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0.0 +- 0.0 | 0.9869 +- 0.0033 |

Episode-100 stochastic evaluation:

| Method | Discovery rate | Collision-penalized discovery | Lambda2 | Collisions |
|---|---:|---:|---:|---:|
| Contention-aware MAPPO w/o ISAC | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | 0.0 +- 0.0 |

Only one nonzero training episode was observed: seed `20260733`, episode `2`, discovery rate `0.0222`, corresponding to one discovered edge out of 45 possible edges. The learned policy converged to near-zero discovery under the strict no-ISAC boundary.

## Interpretation

This campaign should be used as a strict lower-bound learning baseline, not as evidence that no-ISAC MARL is effective. It shows that, when ISAC-derived candidate information and sensing belief updates are removed, a contention-aware MARL actor trained for 100 episodes on 300-slot episodes cannot reliably discover neighbors in the N=10, B=10 training environment.

The result strengthens the paper's cross-layer premise: the ISAC physical-layer abstraction is not a cosmetic feature, but the information source that makes narrow-beam neighbor discovery learnable in this setting.

However, the result can also make the no-ISAC MARL baseline look too weak. The five-way comparison must therefore retain `uniform_random` and `skyorbs_like` baselines, so the proposed method is not compared only against a collapsed learning baseline.

## Figure Use

Main-text candidates:

- `marl_step_reward_curve.png`
- `marl_episode_discovery_curve.png`
- `marl_eval_discovery_curve.png`

Supplementary candidates:

- `marl_episode_empty_scan_curve.png`
- `marl_episode_collision_curve.png`
- `marl_resource_rss_curve.png`

## Next Use

1. Use checkpoint `train_n10_b10_contention_no_isac_100ep_300slot_seed20260731/final_model.pt` as the default strict no-ISAC checkpoint in the five-way campaign.
2. Run five-way N=100, B=10/15/30, 3000-slot comparison after the current B=3 stress eval frees resources.
3. Describe the no-ISAC MARL baseline as strict and difficult, not as an optimized no-ISAC upper bound.
