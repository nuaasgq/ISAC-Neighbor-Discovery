# Manuscript Figure Selection

Date: 2026-07-05

## Verification Snapshot

- Figure archive checked: `06_analysis/paper_figures`
- PNG count: 354
- Aspect ratio check: all PNG figures are 4:3 within tolerance
- Training-score evolution is covered by `06_analysis/paper_figures/training_round2_candidate/train_reward_curve.png`
- Additional round7 training curves are available under `06_analysis/paper_figures/round7_long_cem_training/`; use them as supplementary evidence unless the round7 transfer evaluations outperform the existing main result chain.
- Main plotting style: Times New Roman and Okabe-Ito color order as documented in existing figure manifests

## Main-Manuscript Figures

These figures should carry the main paper narrative.

| Role | Figure | Why it belongs in the main text |
|---|---|---|
| System model | `06_analysis/paper_figures/concept/system_model_isac_uav_swarm.png` | Defines the distributed UAV swarm and information boundary. |
| Protocol mechanism | `06_analysis/paper_figures/concept/protocol_mechanism_itap_nd.png` | Shows how ISAC feedback enters link-layer discovery without creating edges directly. |
| Policy architecture | `06_analysis/paper_figures/concept/policy_architecture_shared_agent.png` | Clarifies shared-parameter execution and the current learning boundary. |
| Training-score evolution | `06_analysis/paper_figures/training_round2_candidate/train_reward_curve.png` | Provides empirical reward/score stabilization evidence without implying a theoretical convergence guarantee. |
| N=100 beamwidth transfer | `06_analysis/paper_figures/round3_n100_transfer/density_beamwidth_discovery_rate_n100.png` | Directly supports zero-shot transfer from `N=10`, 10-degree training to `N=100`. |
| Area-scaling check | `06_analysis/paper_figures/round3_n100_transfer/area_scale_n100_lambda2.png` | Shows density-preserving and fixed-area scaling give similar connectivity trends. |
| Mobility transfer | `06_analysis/paper_figures/round5_mobility_transfer/mobility_discovery_n100_b10.png` | Separates smooth/diffusive mobility support from abrupt-mobility stress regimes. |
| Range sensitivity | `06_analysis/paper_figures/round3_robustness/range_gain_discovery_n100_b10.png` | Supports the claim that sensing-range benefit saturates near communication range. |
| ISAC error profile | `06_analysis/paper_figures/round3_robustness/error_profile_discovery_n100_b10.png` | Supports bounded robustness under false alarms, missed detections, and angular errors. |
| Mechanism ablation | `06_analysis/paper_figures/round4_delay_ablation/ablation_discovery_n100_b10.png` | Shows candidate-set refinement is the largest observed contributor at the main `N=100`, 10-degree operating point and includes the one-slot-delay boundary. |

## Supplementary Candidates

- Use additional training plots from `06_analysis/paper_figures/training_round2_candidate/` for appendix-level convergence evidence: discovery rate, empty-scan ratio, delay, collision, and connectivity.
- Use the longer round7 CEM plots from `06_analysis/paper_figures/round7_long_cem_training/` as supplementary convergence evidence. They cover reward, score, discovery, empty-scan, delay, collision, and connectivity, but should not replace the round2 candidate training curve unless the main result chain is also updated.
- Use `round3_n100_transfer/*mean_delay*`, `*p95_delay*`, `*empty_scan*`, and `*lambda2*` to support reviewer questions about delayed discovery and topology quality.
- Use `06_analysis/paper_figures/round7_scale_beam_grid_light/` to answer broad beamwidth-sweep questions across 3, 5, 10, 15, and 30 degrees, especially the raw-discovery versus collision-penalized tradeoff at `N=100`.
- Use `06_analysis/paper_figures/round7_n100_multimobility_600slot/` when discussing mobility-model stress tests at `N=100` for 10- and 15-degree beams.
- Use `06_analysis/paper_figures/round8_error_profiles_b15_gm_rw_600slot/` for the B=15 error-profile extension; keep it as operating-region nuance because random-walk mobility is much more error-sensitive than Gauss-Markov mobility.
- Use `06_analysis/paper_figures/round8_n100_multimobility_full_baseline/` and the combined table `06_analysis/paper_tables/round8_n100_multimobility_full_baseline/combined_aggregate_metrics.csv` when reviewers ask whether the large-scale mobility boundary is caused by omitted SkyOrbs-like or vanilla RL baselines.
- Use `round6_slot_duration_sensitivity/` if the 5 ms subframe timescale needs more explicit defense.
- Use `round4_delay_ablation/ablation_collision_penalized_discovery_n100_b10.png` and `ablation_discovery_per_scan_n100_b10.png` if collision or scan-efficiency criticism becomes central.

## Figures to Avoid as Main Evidence

- The full `final_round1` per-metric dynamic plots are useful for traceability but too repetitive for the main paper; Table~\ref{tab:dynamic_comparison} and selected transfer figures are cleaner.
- Full-factor error-grid plots from the 400-slot auxiliary run should not be numerically compared against the 600-slot main transfer numbers. Use them only for qualitative robustness trends.
- High-percentile delay plots should be treated carefully because finite horizons censor P95/P99 in weak baselines and stress regimes.
- The 3-degree and 5-degree transfer figures are valuable as stress evidence, but they should not be used to imply that extremely narrow beams are solved.
- Round7 wide-beam collision figures should not be used alone to claim that wider beams are worse; report them together with raw discovery, empty-scan ratio, and connectivity.
