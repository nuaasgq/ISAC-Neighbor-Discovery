# Figure-to-Claim Map - 2026-07-05

This map records why each selected paper figure is in the manuscript package.

## Main Manuscript

| Figure/table | File or source | Claim supported |
|---|---|---|
| Fig. 1 system model | `paper_figures/concept/system_model_isac_uav_swarm.png` | UAV-UAV self-localized, neighbor-unknown, distributed narrow-beam discovery setting. |
| Fig. 2 protocol mechanism | `paper_figures/concept/protocol_mechanism_itap_nd.png` | ISAC occupancy evidence is used as a link-layer prior before handshake confirmation. |
| Fig. 3 shared policy | `paper_figures/concept/policy_architecture_shared_agent.png` | One shared local policy can be trained at small scale and executed by many UAVs. |
| Fig. 4 candidate reward | `training_candidate_trajectory_20260705/cem_candidate_reward_trajectory.png` | Learned policy-search run has a CEM candidate-evaluation trace; no per-step MARL convergence claim. |
| Table I information boundary | `main.tex` | Own state is available; undiscovered neighbor state and global topology are not. |
| Table II dynamic comparison | `main.tex` | ISAC-assisted protocol improves finite-time discovery over random scanning in small dynamic swarms. |
| Table III N=100 transfer | `main.tex` | 10--30 degree beams are useful transfer regimes; 5 degrees is weak. |
| Table IV N=100 baselines | `main.tex` | At N=100/B=10, implemented communication-only baselines remain near zero, while ISAC-assisted policy forms a useful discovered-neighbor graph. |
| Fig. 5 N=100 density transfer | `round3_n100_transfer/density_beamwidth_discovery_rate_n100.png` | Beamwidth transfer trend under density-preserving scaling. |
| Fig. 6 N=100 area scaling | `round3_n100_transfer/area_scale_n100_lambda2.png` | Transfer trend is not solely an area-scaling artifact. |
| Fig. 7 mobility | `round5_mobility_transfer/mobility_discovery_n100_b10.png` | Gauss-Markov/random-walk are favorable; abrupt mobility is a boundary. |
| Fig. 8 range | `round3_robustness/range_gain_discovery_n100_b10.png` | ISAC range support helps until link-confirmation constraints dominate. |
| Fig. 9 error profile | `round3_robustness/error_profile_discovery_n100_b10.png` | Configured ISAC errors degrade but do not erase the main operating-point gain. |
| Fig. 10 ablation | `round4_delay_ablation/ablation_discovery_n100_b10.png` | Candidate-set refinement is the largest observed mechanism contributor at N=100/B=10. |

## Supplement

| Supplement item | File or source | Reviewer question addressed |
|---|---|---|
| Coverage table | `supplement.tex` | Are all requested dimensions covered, and what are the caveats? |
| Training reward and score | `training_candidate_trajectory_20260705/cem_candidate_reward_trajectory.png`, `cem_candidate_score_trajectory.png` | Where is the training trace, and is convergence overclaimed? |
| Scale/beam heatmap | `round7_scale_beam_grid_light/density_heatmap_proposed_discovery_rate.png` | Does N=10 training transfer across N=10--100 and beamwidths 3--30? |
| B=10 node-count curve | `round7_scale_beam_grid_light/density_node_count_discovery_rate_b10.png` | How does the main trained beamwidth scale with node count? |
| Area-scale discovery/connectivity | `round3_n100_transfer/area_scale_n100_discovery_rate.png`, `area_scale_n100_lambda2.png` | Does N=100 performance depend on density-preserving versus fixed-area scaling? |
| Range/timing | `range_gain_discovery_n100_b10.png`, `slot_duration_discovery_n100_b10.png` | Are sensing/communication range and slot-duration assumptions tested? |
| Beamwidth stress | `density_beamwidth_discovery_rate_n100.png`, `density_beamwidth_collision_penalized_discovery_n100.png` | Why are 3 and 5 degrees treated as stress regimes? |
| 3-degree full baseline | `round9_n100_b3_full_baselines_600slot/*.png` | Do all five baselines remain hard at the 3-degree boundary? |
| Mobility full baseline | `round8_n100_multimobility_full_baseline/*.png` | Is the mobility boundary caused by missing communication-only baselines? |
| B=15 error extension | `round8_error_profiles_b15_gm_rw_600slot/*.png` | Does error sensitivity persist beyond the B=10 main operating point? |
| Statistical index | `statistical_stability_summary.csv` | Where are multi-seed mean/std/CI values archived? |
| Finite-horizon backup curves | `pre11_evidence/round10_cumulative_discovery_b10.png`, `round10_cumulative_discovery_b15.png` | Do the extra N=100/B=10 and B=15 seeds preserve the proposed-vs-no-ISAC ordering over time? |
| Long-horizon sensitivity | `round15_long_horizon_3000slot/round15_horizon_discovery_600_vs_3000.png`, `round15_cumulative_discovery_b10.png`, `round15_cumulative_discovery_b15.png` | Is 600 slots too short, and does the method ordering persist at 3000 slots? |
| Backup tradeoff panels | `pre11_evidence/round10_tradeoff_*.png` | How do discovery, collision-penalized discovery, topology, and empty scanning move together in the backup block? |
| Structured MARL probe | `structured_marl_probe/marl_eval_discovery_rate.png`, `marl_eval_empty_scan_ratio.png`, `marl_empty_collision_tradeoff.png` | Does the candidate-constrained shared actor use ISAC-local observations to reduce empty scans under the same execution information boundary? |
| Residual-scale sweep | `structured_marl_probe/marl_rule_residual_scale_sweep.png` | How sensitive is the rule-residual actor to coupling strength, and why is the neural branch framed as a probe rather than the main method? |
| Round11 paired-seed stability | `round11_paired_seed_campaign_main/round11_discovery_rate.png`, `round11_paired_discovery_delta.png` | Do five paired seeds preserve the proposed-vs-baseline ordering at N=100, B=10/B=15? |
| Round11 collision-aware boundary | `round11_paired_seed_campaign_main/round11_collision_penalized.png` | Does higher raw discovery also imply MAC-efficient discovery? No; B=15 shows collision-aware optimization remains open. |
| Round13 collision-aware refinement | `round13_collision_energy_10seed/round13_collision_penalized.png`, `round13_collision_count.png` | Can the B=15 collision boundary be mitigated without changing the ISAC candidate-set interface? Yes, a local role-control probe improves collision-penalized discovery in a ten-seed paired extension. |
| Round13 paired collision delta | `round13_collision_energy_10seed/round13_collision_penalized_delta.png` | Is the collision-aware gain seed-paired? Yes, collision-penalized delta is 10/10 positive versus proposed and one-slot delay at B=10 and B=15. |
| Round13 radio-state accounting | `round13_collision_energy_10seed/round13_discoveries_per_joule.png`, `round13_energy_per_discovery.png` | Does the collision-aware role-control probe also improve assumed energy accounting? Yes, discoveries-per-joule delta is 10/10 positive under stated default radio-state powers, while not claiming platform-calibrated optimality. |
| Round13 power sensitivity | `round13_energy_sensitivity/energy_sensitivity_discoveries_per_joule.png`, `round13_energy_sensitivity/energy_sensitivity_delta_vs_proposed.png` | Is the energy-accounting result robust to simple power reweighting? It is robust versus one-slot delay across tested profiles, but RX-heavy B=10 is a boundary versus the proposed low-latency variant. |
| Round14 main-table stability | `round14_main_table_10seed_n100_b10/round14_main_table_discovery_lambda2.png` | Does the N=100/B=10 main comparison survive a ten-seed baseline check? Yes, the ISAC-assisted policy keeps discovery near 0.365 while four communication-only controls stay near zero, with 10/10 positive paired discovery deltas. |

## Figure Integrity

| Audit | File or source | Claim supported |
|---|---|---|
| LaTeX figure-reference audit | `paper_figure_integrity_audit_20260705.md`, `paper_figure_integrity_audit_20260705.csv` | The current main/supplement package references 47 figure instances, 44 unique existing files, no missing figures, and no 4:3 aspect-ratio violations. |
