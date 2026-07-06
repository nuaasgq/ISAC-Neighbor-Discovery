# MARL + ISAC Final Experiment Matrix - 2026-07-06

## Fixed Experimental Contract

- Training source: `N=10`, `B=10 deg`, `300 slots/episode`, `5 ms/slot`.
- Training budget for paper learning curves: `100 episodes`, `3 seeds` where available.
- Transfer/testing horizon: `3000 slots/episode` for the main paper tables.
- Execution: stochastic decentralized policy execution; deterministic rows are diagnostics only unless reported separately.
- Primary metrics: `discovery_rate`, `collision_penalized_discovery_rate`, `collision_count`, `collisions_per_discovery_censored`, `lambda2`, `lcc_ratio`, and censored delay.
- Method identity: checkpoint MAPPO/actor-critic runs are the only MARL evidence. CEM and protocol heuristics are background, baselines, or supplementary probes only.

## Evidence Already Ready

| Evidence block | Status | Main files |
|---|---|---|
| ISAC-MARL multi-seed training curves | Ready | `06_analysis/marl_phase7_isac_multiseed_results_20260706.md`; `06_analysis/paper_figures/marl/phase7_long_training_100ep_3seed_learning_curves/` |
| Strict no-ISAC training curves | Ready as lower-bound baseline | `06_analysis/marl_phase7_strict_no_isac_results_20260706.md`; `06_analysis/paper_figures/marl/phase7_contention_no_isac_strict_100ep_3seed_learning_curves/` |
| Internal ISAC-MARL ablation at `N=100`, `B=10/15/30`, `3000 slots` | Ready | `06_analysis/marl_phase6_final_long_eval_results_20260706.md`; `06_analysis/paper_figures/marl/phase6_final_long_eval_b10_b30_10ep_stoch_method_comparison/` |
| `B=5` stress boundary | Ready | `06_analysis/marl_phase6_b5_long_eval_results_20260706.md`; `06_analysis/paper_figures/marl/phase6_final_long_eval_b5_10ep_stoch_method_comparison/` |

## Active Result Gaps

| Gap | Running campaign | Success condition |
|---|---|---|
| Shared no-ISAC MAPPO 100-episode checkpoint | `phase9_mappo_no_isac_strict_100ep_3seed` | Complete `mappo_no_isac` at `N=10`, `B=10`, `300 slots`, `100 episodes`, preferably 3 seeds. |
| `B=3` narrow-beam stress | `phase6_final_long_eval_b3_10ep_stoch` | Complete 10 stochastic `3000-slot` episodes for `legacy_shared`, `collision_reward`, and `contention_actor`; report as early-checkpoint diagnostic unless rerun with Phase-7 100-episode checkpoints. |
| Five-way large-scale beam transfer | `phase8_fiveway_n100_b10_b15_b30_3000slot_10ep_stoch` | Diagnostic only because `mappo_no_isac` and `contention_actor` use older checkpoints. Use it for sanity checks, then rerun as Phase-9 with 100-episode checkpoints. |
| Five-way node-count transfer | `phase8_fiveway_node_transfer_b10_3000slot_10ep_stoch` | Diagnostic only for the same checkpoint-budget reason. Rerun as Phase-9 after shared no-ISAC 100-episode training completes. |
| Paper-grade internal ablation | planned `phase9_internal_100ep_*` | Rerun `legacy_shared`, `collision_reward`, and `contention_actor` using Phase-7 100-episode checkpoints. |
| Paper-grade five-way transfer | planned `phase9_fiveway_*` | Use 100-episode checkpoints for `mappo_no_isac`, `contention_no_isac`, and `contention_actor`, paired against random and SkyOrbs-like baselines. |
| B=15 five-way follow-up | complete `phase9_fiveway_n100_b15_3000slot_10ep_stoch` | B=30 was explicitly canceled by the user on 2026-07-06; B=15 has been aggregated and reported with B=10. |
| Phase9 area-scaling closure | planned `phase9_fiveway_node_transfer_b10_3000slot_10ep_stoch_area_*` | `run_marl_fiveway_eval_campaign.py` now supports `--area-scale fixed` and `--area-scale density`; both variants must be kept as separate campaigns/tables. |

## Main-Paper Figure Plan

| Figure | Claim supported | Source |
|---|---|---|
| System model and ISAC-assisted ND workflow | Cross-layer problem formulation, not pure PHY beam prediction | `06_analysis/paper_figures/concept/` |
| Contention-aware ISAC-MARL actor-critic structure | Network-structure innovation | Regenerate/update from `06_analysis/scripts/draw_concept_figures.py` if labels lag current model |
| Step reward and episode return curves | Real MARL learning evidence under 300-slot training | Phase-7 learning-curve figures |
| Training discovery/CPD/collision curves | Learning behavior and mechanism tradeoff | Phase-7 learning-curve figures |
| Five-way `N=100, B=10` comparison | Main baseline closure | Phase-8 five-way beam campaign |
| Beamwidth transfer `B=3/5/10/15` | Zero-shot beamwidth generalization and stress boundary | Phase-6 plus Phase-9 B=10/B=15 campaigns; B=30 is not part of the current active run. |
| Node transfer `N=10/20/50/100` | Small-to-large scalable policy transfer | Phase-8 node campaign |
| Internal ablation `legacy -> reward -> contention actor` | Separates reward shaping from network structure | Phase-6 long evaluation |

## Claim Discipline

- ISAC contribution: protocol-layer occupancy/candidate prior that reduces empty-beam exploration and supplies local beam evidence before handshake alignment.
- MARL contribution: shared-parameter CTDE actor-critic trained on slot-level observations/actions/rewards. Current implementation is MAPPO-style PPO with a centralized critic, not a full reference MAPPO stack with GAE/minibatch/parallel rollout.
- Network-structure contribution: contention-aware beam-token actor. The strongest current result is lower collision load and higher collision-penalized discovery, not universal raw-discovery dominance.
- Scalability contribution: train at `N=10`, deploy without fine-tuning to larger `N` and different beamwidths. The claim is zero-shot transfer, not retraining at every test scale.
- No-ISAC baseline: strict no-ISAC is a lower-bound learning baseline under high-dimensional blind search. It should be paired with random and SkyOrbs-like protocol baselines to avoid relying on an artificially weak comparator.

## Paper-Ready Threshold

The result set is paper-ready only when the following are all true:

1. Phase-9 five-way beam transfer has complete rows and figures for `N=100`, `B=10/15`, `3000 slots`, with 100-episode MARL checkpoints. B=30 is currently excluded by user direction rather than treated as a missing result.
2. Phase-9 node transfer has complete rows and figures for `N=10/20/50/100`, `B=10`, `3000 slots`, with 100-episode MARL checkpoints.
3. Phase-9 `N=100` node-transfer evidence is reported under both fixed-area and equal-density scaling, without averaging the two scaling conventions.
4. `B=3` and `B=5` are reported as stress-boundary rows with honest limitations.
5. All main figures use 4:3 aspect ratio, Times New Roman or Times fallback, and a unified color palette.
6. Every result table states `300-slot training` and `3000-slot evaluation` explicitly.
