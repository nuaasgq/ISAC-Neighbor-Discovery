# Paper Readiness Dashboard

Date: 2026-07-05

## Readiness Verdict

The current data package is potentially sufficient for a bounded IEEE TWC/TCOM-style draft if the paper is framed as:

> ISAC-assisted link-layer beam-cell occupancy priors improve distributed-execution narrow-beam UAV-UAV neighbor discovery and finite-time discovered-neighbor graph formation in tested dynamic regimes.

The data do not yet support a claim of full neural MARL superiority, strict SkyOrbs reproduction, platform-calibrated energy optimality, or universal robustness across all extremely narrow beams and abrupt mobility models.

## Main Evidence Chain

| Claim | Primary artifacts | Current status |
|---|---|---|
| ISAC candidate-beam refinement is the largest observed contributor at the main `N=100`, 10-degree operating point. | `06_analysis/paper_tables/round3_robustness/ablation/aggregate_metrics.csv`, `06_analysis/paper_figures/round4_delay_ablation/ablation_discovery_n100_b10.png` | Strong but operating-point-specific main claim. |
| Proposed ISAC policy shows observed improvement over random, SkyOrbs-like, learned no-ISAC, and enhanced no-ISAC baselines at `N=100`, 10 degrees. | `06_analysis/paper_tables/round14_main_table_10seed_n100_b10/`, `Table~\ref{tab:n100_baseline}` in `07_paper/ieee_twc_isac_nd/main.tex` | Strong ten-seed main comparison; SkyOrbs-like scope note and appendix are present, but strict reproduction is absent. |
| Small-scale training transfers to `N=100` for 10--30 degree beams in tested regimes. | `06_analysis/paper_tables/round3_robustness/n100_*_multiseed/aggregate_metrics.csv`, `Table~\ref{tab:n100_transfer}` | Strong but bounded; 3/5 degrees remain stress regimes. |
| Collision-aware metrics change the preferred beamwidth. | `06_analysis/paper_tables/round7_scale_beam_grid_light/aggregate_metrics.csv` | Supplementary but important for reviewer discussion. |
| Moderate ISAC errors degrade but do not collapse the gain. | `06_analysis/paper_tables/round3_robustness/error_profiles/aggregate_metrics.csv`, `06_analysis/paper_tables/round7_error_profiles_light/aggregate_metrics.csv` | Strong bounded robustness claim. |
| B=15 remains strong under Gauss-Markov error profiles but random-walk is more sensitive. | `06_analysis/paper_tables/round8_error_profiles_b15_gm_rw_600slot/aggregate_metrics.csv` | Supplementary evidence for operating-region nuance. |
| Mobility-boundary results are not caused by missing SkyOrbs-like or vanilla learned-policy baselines. | `06_analysis/paper_tables/round8_n100_multimobility_full_baseline/combined_aggregate_metrics.csv`, `06_analysis/paper_figures/round8_n100_multimobility_full_baseline/` | Supplementary baseline-completeness evidence. |
| The 3-degree beam setting is an explicit stress/failure boundary even with five baselines. | `06_analysis/paper_tables/round9_n100_b3_full_baselines_600slot/aggregate_metrics.csv`, `06_analysis/paper_figures/round9_n100_b3_full_baselines_600slot/` | Supplementary stress evidence. |
| Candidate-constrained neural MARL is feasible under the same local information boundary, but not yet superior to all baselines. | `06_analysis/paper_tables/structured_marl_probe/structured_marl_probe_eval_summary.csv`, `06_analysis/paper_figures/structured_marl_probe/` | Secondary method-probe evidence; do not frame as the main contribution. |
| Focused five-seed paired stability preserves the main N=100/B=10/B=15 raw-discovery ordering. | `06_analysis/paper_tables/round11_paired_seed_campaign_main/`, `06_analysis/paper_figures/round11_paired_seed_campaign_main/` | Strong stability/mechanism evidence; also shows collision-penalized optimization remains open at B=15. |
| Collision-aware local role control mitigates the B=15 collision-penalized boundary. | `06_analysis/paper_tables/round13_collision_energy_10seed/`, `06_analysis/paper_figures/round13_collision_energy_10seed/` | Ten-seed mechanism-refinement evidence with assumed radio-state accounting; not a complete MAC or calibrated energy model. |
| ISAC abstraction has a cite-backed PHY-to-protocol interpretation. | `07_paper/ieee_twc_isac_nd/main.tex`, `07_paper/ieee_twc_isac_nd/supplement.tex`, `06_analysis/phy_to_protocol_isac_mapping_20260705.md` | Text-only mitigation; still not a calibrated waveform/detector appendix. |

## Recommended Main-Text Figures

Use the existing main draft selection:

- System model: `06_analysis/paper_figures/concept/system_model_isac_uav_swarm.png`
- Protocol mechanism: `06_analysis/paper_figures/concept/protocol_mechanism_itap_nd.png`
- Shared-policy architecture: `06_analysis/paper_figures/concept/policy_architecture_shared_agent.png`
- Training-score evolution: `06_analysis/paper_figures/training_round2_candidate/train_reward_curve.png`
- N=100 transfer: `06_analysis/paper_figures/round3_n100_transfer/density_beamwidth_discovery_rate_n100.png`
- Area-scale check: `06_analysis/paper_figures/round3_n100_transfer/area_scale_n100_lambda2.png`
- Mobility boundary: `06_analysis/paper_figures/round5_mobility_transfer/mobility_discovery_n100_b10.png`
- Range sensitivity: `06_analysis/paper_figures/round3_robustness/range_gain_discovery_n100_b10.png`
- ISAC error profile: `06_analysis/paper_figures/round3_robustness/error_profile_discovery_n100_b10.png`
- Mechanism ablation: `06_analysis/paper_figures/round4_delay_ablation/ablation_discovery_n100_b10.png`

Keep round7/round8 stress figures as supplement unless the main draft expands.

## Verification Snapshot

- `pdflatex -interaction=nonstopmode main.tex`: passed; no undefined references/citations and no overfull warnings in the final checked log.
- `pdflatex -interaction=nonstopmode supplement.tex`: passed; no undefined references/citations and no overfull warnings in the final checked log.
- `python -m pytest 05_simulation\tests`: 29 passed.
- `06_analysis/paper_figures`: archived paper-figure pool with the selected manuscript and supplement figures checked at 4:3 aspect ratio.
- Statistical stability summary: `06_analysis/paper_tables/statistical_stability_summary/statistical_stability_summary.csv`.
- Round14 ten-seed main-table stability check: `06_analysis/paper_tables/round14_main_table_10seed_n100_b10/`.
- Round11 paired-seed campaign: `06_analysis/paper_tables/round11_paired_seed_campaign_main/`.
- Round13 collision-aware MAC/energy probe: `06_analysis/paper_tables/round13_collision_energy_10seed/`.
- One-page submission pitch: `06_analysis/submission_pitch_one_page_20260705.md`.

## Literature Boundary Notes

Recent checks support the current scope:

- SkyOrbs is the closest UAV 3-D directional ND reference, but our baseline remains `SkyOrbs-like`, not a strict reproduction; the supplement and `06_analysis/skyorbs_like_baseline_scope_appendix_20260705.md` now make this explicit: `https://ieeexplore.ieee.org/document/10659183/`
- Delay-power UAV ND work supports reporting collision and radio-activity caveats; our work now has assumed radio-state accounting but not a platform-calibrated delay-power model: `https://www.computer.org/csdl/journal/tm/2026/06/11320813/2cTQxGWicIo`
- ISAC predictive beam tracking remains mostly physical-layer/beam-management work, so our protocol layer distinction should remain explicit; the current draft now supports the distinction with 3GPP sensing-service and channel-model citations.

## Remaining Risks

- Full neural MARL is not yet a main contribution; the current learning evidence is shared-parameter protocol tuning plus a structured actor-critic probe.
- Energy efficiency is still model-assumed; round13 reports radio-state accounting, but not a calibrated Joule-level hardware model or energy-aware optimizer.
- 3-degree and 5-degree beams are not solved.
- Random-direction and random-waypoint mobility remain stress regimes.
- The main paper is still compact; the supplement now carries round7/round8 stress results, pre-11 backup trajectories, round11 paired-seed stability, the B=15 collision boundary, round13 collision-aware refinement and energy accounting, structured MARL probe figures, the PHY-to-protocol parameter table, and full std/CI tables.
