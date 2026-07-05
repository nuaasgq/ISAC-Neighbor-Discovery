# Experiment Coverage Matrix

Date: 2026-07-05

Purpose: map the requested experiment requirements to archived data, figures, and remaining claim boundaries.

## Coverage Summary

| Requirement | Status | Primary artifacts | Claim boundary |
|---|---|---|---|
| Beamwidth range 3--30 degrees | Covered as main plus supplement | Main: `06_analysis/paper_tables/round3_robustness/n100_density_multiseed/aggregate_metrics.csv` and `n100_fixed_multiseed/aggregate_metrics.csv` for 5/10/15/30 degrees. Stress supplement: `06_analysis/paper_tables/round7_scale_beam_grid_light/aggregate_metrics.csv` and `06_analysis/paper_figures/round7_scale_beam_grid_light/` for 3/5/10/15/30 degrees. Full 3-degree five-baseline stress check: `06_analysis/paper_tables/round9_n100_b3_full_baselines_600slot/` and `06_analysis/paper_figures/round9_n100_b3_full_baselines_600slot/`. | Write "evaluated over 3--30 degrees"; do not write "effective over 3--30 degrees." The 3-degree and 5-degree cases remain stress regimes. |
| Node counts 10--100 | Covered | `06_analysis/paper_tables/round7_scale_beam_grid_light/aggregate_metrics.csv`; `06_analysis/paper_figures/round7_scale_beam_grid_light/`. | Use as scalability/stress evidence; not all baselines are fully crossed with every mobility model and beamwidth. |
| Small-scale training, large-scale testing | Covered | Training artifacts under `06_analysis/paper_tables/round2_transfer/training/`; main N=100 transfer tables under `06_analysis/paper_tables/round3_robustness/`; figures under `06_analysis/paper_figures/round3_n100_transfer/`. | The main trained policy is shared-parameter/CEM policy optimization, not full MAPPO/QMIX/GNN MARL. |
| Train at 10-degree beams, test other beamwidths | Covered | Main 5/10/15/30 degree transfer: `round3_robustness/n100_*_multiseed`; stress 3/5/10/15/30 degree transfer: `round7_scale_beam_grid_light`. | Clearest useful evidence is 10--30 degrees; 3/5 degrees show the hard lower-beamwidth boundary. |
| Density-preserving and fixed-area N=100 scaling | Covered | `06_analysis/paper_tables/round3_robustness/n100_density_multiseed/aggregate_metrics.csv`; `06_analysis/paper_tables/round3_robustness/n100_fixed_multiseed/aggregate_metrics.csv`; `06_analysis/paper_figures/round3_n100_transfer/area_scale_n100_lambda2.png`. | Supports that the N=100 trend is not a one-off scaling artifact. |
| Fully random baseline | Covered | Main baseline table source: `06_analysis/paper_tables/round3_robustness/n100_density_multiseed/aggregate_metrics.csv`; dynamic comparison source under final/round2 tables. | Random remains a baseline, not the closest literature method. |
| Literature-inspired baseline | Covered as approximate | `skyorbs_like_skip_scan` in `round3_robustness/n100_density_multiseed`, `round8_n100_multimobility_full_baseline/combined_aggregate_metrics.csv`, and figures under `06_analysis/paper_figures/round8_n100_multimobility_full_baseline/`. | Always write "SkyOrbs-like" or "SkyOrbs-inspired"; this is not a strict SkyOrbs reproduction. |
| Learned policy without ISAC baseline | Covered | `rl_no_isac` rows in `round3_robustness/n100_*_multiseed` and `round8_n100_multimobility_full_baseline/combined_aggregate_metrics.csv`. | Used to isolate communication-only shared-policy behavior; do not call the main method full neural MARL. |
| Enhanced learned policy without ISAC baseline | Covered | `improved_rl_no_isac` rows in main and supplement tables. | Shows that rule/policy improvement without sensing remains near zero in the evaluated large narrow-beam settings. |
| Enhanced learned policy plus ISAC | Covered | `improved_rl_isac` rows in all main and supplement tables. | Main performance evidence for the proposed protocol. |
| Focused paired-seed stability campaign | Covered | `06_analysis/paper_tables/round11_paired_seed_campaign_main/`; `06_analysis/paper_figures/round11_paired_seed_campaign_main/`. | Five paired seeds at N=100, B=10/15, Gauss-Markov, 600 slots; use as stability/mechanism evidence unless main tables are formally promoted. |
| Collision-aware MAC refinement | Covered as mechanism probe | `06_analysis/paper_tables/round13_collision_energy_10seed/`; `06_analysis/paper_figures/round13_collision_energy_10seed/`. | Shows the B=15 collision-penalized boundary can be mitigated by local role control across 10 paired seeds; includes assumed radio-state accounting but not platform-calibrated energy-aware MAC optimization. |
| Multi-mobility evaluation | Covered as operating-boundary evidence | Main: `06_analysis/paper_tables/round5_mobility_transfer/aggregate_metrics.csv`; stress: `06_analysis/paper_tables/round7_n100_multimobility_600slot/aggregate_metrics.csv`; full missing-baseline check: `06_analysis/paper_tables/round8_n100_multimobility_full_baseline/combined_aggregate_metrics.csv`. | Gauss-Markov and random-walk are favorable tested regimes; random-direction and random-waypoint remain stress regimes. |
| Communication range and sensing range tests | Covered | `06_analysis/paper_tables/round3_robustness/range_rc_rs_grid/aggregate_metrics.csv`; `06_analysis/paper_figures/round3_robustness/range_gain_discovery_n100_b10.png`. | `Rs/Rc` saturation is internal to the communication-neighbor-discovery abstraction; it is not a calibrated radar range law. |
| Slot duration and motion timescale | Covered | `06_analysis/paper_tables/round6_slot_duration_sensitivity/aggregate_metrics.csv`; `06_analysis/paper_figures/round6_slot_duration_sensitivity/`. | Supports that the 5 ms setting is not a single tuned point under Gauss-Markov mobility. |
| ISAC sensing errors | Covered | Main B=10: `06_analysis/paper_tables/round3_robustness/error_profiles/aggregate_metrics.csv`; confirmation: `round7_error_profiles_light`; B=15 extension: `round8_error_profiles_b15_gm_rw_600slot`. | Supports bounded robustness under configured errors; does not imply physical-layer estimator immunity. |
| Training reward or score curve | Covered | `06_analysis/paper_figures/training_round2_candidate/train_reward_curve.png`; `06_analysis/paper_figures/training_round2_candidate/train_score_curve.png`; `06_analysis/paper_figures/round7_long_cem_training/`. | Write "training-score/reward evolution"; do not claim theoretical RL convergence. |
| Structured neural MARL probe | Covered as secondary/probe evidence | `06_analysis/paper_tables/structured_marl_probe/structured_marl_probe_eval_summary.csv`; `06_analysis/paper_figures/structured_marl_probe/`. | Supports candidate-constrained decentralized actor design and ISAC information-boundary feasibility; do not claim the neural MARL branch beats all baselines. |
| Clean no-ISAC neural environment control | Covered | `no_isac_env` rows in `06_analysis/paper_tables/structured_marl_probe/structured_marl_probe_eval_summary.csv`. | Shows neural nonzero behavior depends on ISAC-enabled local observations; not a large-scale transfer result. |
| Residual-strength neural ablation | Covered | `scale_sweep` rows in `06_analysis/paper_tables/structured_marl_probe/structured_marl_probe_eval_summary.csv`; `marl_rule_residual_scale_sweep.png`. | Supports tuning of rule-residual coupling; current best tested setting is not a global optimum proof. |
| More than 16 paper-ready figures, 4:3 aspect, Times-style fonts | Covered | `06_analysis/paper_figures` archives a large figure pool; selected round11 and round13 figures are 1920 x 1440. Figure manifests document 4:3 and Times New Roman with serif fallback. | The final manuscript should select a small subset for the main text and use the rest in supplement. |
| Network-structure or method innovation beyond blind search | Partially covered | Mechanism figures, candidate-set refinement ablation, topology-deficit scoring ablation, one-slot delay ablation. | Current clearest innovation is cross-layer ISAC occupancy prior plus topology-deficit link-layer discovery. Neural MARL architecture innovation remains a future or secondary thread. |

## Highest-Value Supplement Figures

| Question | Recommended figures |
|---|---|
| Does the method still help when mobility changes? | `06_analysis/paper_figures/round8_n100_multimobility_full_baseline/full_baseline_discovery_n100_b10.png`; `full_baseline_discovery_n100_b15.png`; `full_baseline_collision_penalized_n100_b10.png`. |
| What happens across 3--30 degree beams? | `06_analysis/paper_figures/round7_scale_beam_grid_light/density_beamwidth_discovery_rate_n100.png`; `density_beamwidth_collision_penalized_discovery_n100.png`; `density_beamwidth_lambda2_n100.png`. |
| Do all five baselines fail in the 3-degree stress case? | `06_analysis/paper_figures/round9_n100_b3_full_baselines_600slot/n100_b3_full_baselines_discovery.png`; `n100_b3_full_baselines_empty_scan.png`. |
| Is the 5 ms slot assumption brittle? | `06_analysis/paper_figures/round6_slot_duration_sensitivity/slot_duration_discovery_n100_b10.png`. |
| What is the main mechanism? | `06_analysis/paper_figures/round4_delay_ablation/ablation_discovery_n100_b10.png`; `ablation_collision_penalized_discovery_n100_b10.png`. |
| Do ISAC errors break the result? | `06_analysis/paper_figures/round3_robustness/error_profile_discovery_n100_b10.png`; `06_analysis/paper_figures/round8_error_profiles_b15_gm_rw_600slot/ablation_discovery_n100_b15.png`. |
| Is there a neural MARL path under the same information boundary? | `06_analysis/paper_figures/structured_marl_probe/marl_eval_discovery_rate.png`; `marl_eval_empty_scan_ratio.png`; `marl_rule_residual_scale_sweep.png`. |
| Are the pre-11 backup runs consistent over finite time? | `06_analysis/paper_figures/pre11_evidence/round10_cumulative_discovery_b10.png`; `round10_cumulative_discovery_b15.png`; `round10_tradeoff_discovery_rate.png`. |
| Do paired seeds support the main mechanism? | `06_analysis/paper_figures/round11_paired_seed_campaign_main/round11_discovery_rate.png`; `round11_paired_discovery_delta.png`; `round11_collision_penalized.png`. |
| Can collision-aware role control mitigate the B=15 boundary? | `06_analysis/paper_figures/round13_collision_energy_10seed/round13_collision_penalized.png`; `round13_collision_count.png`; `round13_collision_penalized_delta.png`. |
| What happens under assumed radio-state energy accounting? | `06_analysis/paper_figures/round13_collision_energy_10seed/round13_discoveries_per_joule.png`; `round13_energy_per_discovery.png`. |

## Remaining Experiment Gaps

- A strict SkyOrbs reproduction is not implemented.
- Full MAPPO/QMIX/GNN MARL is not yet the main evidence; actor-critic remains a method probe.
- The structured MARL probe has not yet been transferred to N=30/50/100 or B=10/B=15-equivalent deployment without fine-tuning.
- A full Cartesian cross of all node counts, all beamwidths, all mobility models, all baselines, and all ISAC errors is not available.
- Assumed radio-state energy accounting is now available and reported in the round13 ten-seed probe, but platform-calibrated delay-power optimization is not.
- The current range sweep is protocol-level; physical sensing and communication ranges would require calibrated link budgets and waveform/receiver assumptions.
- Round11 shows B=15 raw discovery remains strongest for the proposed protocol, but one-slot delay can be better on collision-penalized discovery. Round13 shows a local collision-aware role-control probe can mitigate this boundary across 10 paired seeds, but full MAC/energy optimization remains open.
