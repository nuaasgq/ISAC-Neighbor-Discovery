# Claim-Evidence Matrix

Date: 2026-07-05

This matrix tracks which manuscript claims are directly supported by archived experiments and which claims must remain bounded.

## Supported Main Claims

| Claim | Evidence | Manuscript use |
|---|---|---|
| ISAC should be treated as an imperfect link-layer occupancy prior, not as an oracle position estimator. | System model and mechanism figures in `06_analysis/paper_figures/concept/`; model text in `07_paper/ieee_twc_isac_nd/main.tex`. | Core system-model and contribution claim. |
| ISAC-assisted candidate-set refinement is the largest observed mechanism contributor at the main `N=100`, 10-degree operating point. | `06_analysis/paper_tables/round3_robustness/ablation/aggregate_metrics.csv`; candidate-set removal drops `N=100`, 10-degree discovery from 0.3655 to 0.0313 and collapses connectivity. | Main mechanism ablation paragraph and `ablation_discovery_n100_b10.png`; do not generalize this mechanism ranking to all beams and mobility models. |
| The proposed policy shows observed improvement over random, SkyOrbs-like, RL-no-ISAC, and improved-RL-no-ISAC baselines in the main large-scale 10-degree comparison at `N=100`, density scaling. | `06_analysis/paper_tables/round3_robustness/n100_density_multiseed/aggregate_metrics.csv`; main table added in `main.tex` with discovery 0.3655 versus near-zero baselines. | Main baseline comparison table. |
| A policy trained at `N=10`, 10-degree beams can transfer to `N=100` for 10-30 degree beams in the tested finite horizon. | `06_analysis/paper_tables/round3_robustness/n100_density_multiseed/aggregate_metrics.csv` and `n100_fixed_multiseed/aggregate_metrics.csv`; transfer table in `main.tex`. | Transfer result, but not a universal scalability guarantee. |
| The 3--30 degree beamwidth sweep has been explicitly stress-tested at `N=100`. | `06_analysis/paper_tables/round7_scale_beam_grid_light/aggregate_metrics.csv`; figures in `06_analysis/paper_figures/round7_scale_beam_grid_light/`. | Use as supplementary robustness evidence; do not replace the main transfer table yet. |
| Collision-aware metrics change the beamwidth preference in dense finite-horizon tests. | `round7_scale_beam_grid_light` shows raw `N=100` discovery peaking around 15 degrees, while collision-penalized discovery is strongest around 10 degrees. | Supports reporting raw discovery and collision-normalized efficiency together. |
| Density-preserving and fixed-area `N=100` scaling produce similar trends. | Same `round3_robustness/n100_*_multiseed` tables and `area_scale_n100_lambda2.png`. | Supports reporting both scaling conventions. |
| The 5 ms slot duration is not a single tuned point in the Gauss-Markov setting. | `06_analysis/paper_tables/round6_slot_duration_sensitivity/aggregate_metrics.csv`; discovery remains about 0.3564-0.3696 from 1 ms to 20 ms. | Modeling-timescale defense. |
| Moderate sensing errors degrade but do not collapse the ISAC benefit in the tested settings. | `06_analysis/paper_tables/round3_robustness/error_profiles/aggregate_metrics.csv`; paired 600-slot profile sweep. | Robustness claim with moderate-error wording. |
| The round7 three-seed error-profile rerun confirms the moderate-error trend. | `06_analysis/paper_tables/round7_error_profiles_light/aggregate_metrics.csv`; full ISAC remains around 0.3534 under mild error and 0.2887 under moderate error, while no-ISAC remains near zero. | Supplementary confirmation; keep round3 as the main manuscript source unless promoted. |
| B=15 error robustness is strong under Gauss-Markov but more sensitive under random-walk mobility. | `06_analysis/paper_tables/round8_error_profiles_b15_gm_rw_600slot/aggregate_metrics.csv`; full ISAC stays around 0.5272--0.5852 under Gauss-Markov configured errors, while random-walk declines from 0.4414 to 0.2253 as errors increase. | Supplementary evidence for B=15; report collision counts together with raw discovery. |
| One-slot delayed candidate-set use retains much of the benefit while reducing collisions. | `06_analysis/paper_tables/round4_delay_ablation/aggregate_metrics.csv`; discovery 0.2989, lambda2 8.4709, collisions 697.0 versus full ISAC 1050.0. | Implementation-boundary claim. |

## Bounded or Negative Claims

| Claim boundary | Reason | Required wording |
|---|---|---|
| The current learning result is not full neural MARL. | The main method is shared-parameter/CEM policy optimization; actor-critic imitation is still a method probe. | Use "shared-parameter policy optimization" and keep MAPPO/QMIX/GNN as future work or extension. |
| 3-degree and 5-degree beams are not solved. | `N=100` 5-degree discovery is about 0.081; 3-degree regimes are stress cases. | Describe as stress regimes, not successful transfer. |
| Random-direction and random-waypoint mobility are applicability boundaries. | `round5_mobility_transfer` shows weak discovery under abrupt mobility at `N=100`, 10 degrees. | State that smoother occupancy evolution appears more favorable. |
| The round7 15-degree mobility sweep improves some stress cases but does not remove the mobility boundary. | `06_analysis/paper_tables/round7_n100_multimobility_600slot/aggregate_metrics.csv` improves random-walk transfer at 15 degrees, but random-direction and random-waypoint remain weak. | Do not claim universal mobility robustness. |
| Missing SkyOrbs-like and vanilla RL baselines do not change the mobility-boundary interpretation. | `06_analysis/paper_tables/round8_n100_multimobility_full_baseline/combined_aggregate_metrics.csv` merges round7 mobility results with round8 SkyOrbs-like and vanilla RL baselines; these missing baselines remain near-zero across tested mobility models. | Use as supplementary baseline-completeness evidence with figures in `06_analysis/paper_figures/round8_n100_multimobility_full_baseline/`. |
| `Rs/Rc` saturation is model-internal, not a physical sensing law. | Only communication-range neighbors can be confirmed by handshake in the simulator. | Say "in the evaluated communication-neighbor-discovery abstraction." |
| SkyOrbs-like is not a strict SkyOrbs reproduction. | The baseline implements a deterministic 3-D skip-scan schedule inspired by SkyOrbs, not the full original protocol. | Always write "SkyOrbs-like" and explicitly state non-reproduction. |
| Energy efficiency is not yet Joule-level. | No radio-state energy model has been implemented. | Report scan-action and collision-normalized efficiency only. |
| Wider beams are not automatically better. | 30-degree beams reduce empty scans but create very high collision counts. | Treat 15 degrees as the current best discovery-connectivity balance in tested `N=100` results. |

## Current Promotion Decision

Round7 long CEM training has been archived as a candidate run, but it is not promoted to the main result unless downstream transfer/evaluation jobs improve large-scale robustness. The existing main evidence remains the round3/round4/round5/round6 chain.
