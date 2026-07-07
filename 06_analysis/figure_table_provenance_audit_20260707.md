# Figure/Table Provenance Audit - 2026-07-07

Scope: `07_paper/ieee_twc_isac_nd/main.tex` and `07_paper/ieee_twc_isac_nd/supplement.tex`.

This audit supersedes `06_analysis/paper_figure_integrity_audit_20260705.md`, which was tied to an earlier figure selection and does not fully describe the current Phase10-centered manuscript.

## File Presence and Format Check

- Checked manuscript image references: 51 PNG files.
- Main manuscript images: 10.
- Supplement images: 41.
- Missing image files: 0.
- Approximate 4:3 aspect-ratio violations: 0.
- Check method: parse `\includegraphics{...}` entries from `main.tex` and `supplement.tex`, resolve each path from the LaTeX workspace, and inspect PNG width/height with `System.Drawing`.

## Main Evidence Chain

| Manuscript item | Claim role | Data/table source | Figure source | Generation/provenance |
|---|---|---|---|---|
| `fig:system_model`, `fig:protocol_mechanism`, `fig:policy_architecture` | System, protocol, and shared-policy mechanism schematics; not numerical performance evidence. | N/A. | `06_analysis/paper_figures/concept/` | `06_analysis/scripts/draw_concept_figures.py`; `06_analysis/paper_figures/concept/concept_figure_manifest.json`; `06_analysis/paper_figures/concept/README.md`. |
| `fig:training_reward`, `fig:supp_training` | Step-indexed and episode-indexed training diagnostics for the final gated contention ISAC-MAPPO actor. | `06_analysis/paper_tables/marl/p10_gate_training_3seed_100ep_step_curves/` | `06_analysis/paper_figures/marl/p10_gate_training_3seed_100ep_step_curves/` | Manifest reports three N=10, B=10, 100-episode, 300-slot training runs; 90,000 step rows, 300 episode rows, 198 eval rows, and algorithm/network labels `isac_mappo` / `gated_contention_shared`. |
| `tab:n100_transfer`, `fig:marl_cpd` | Final N=100, 3000-slot, B=10/B=15 zero-shot transfer comparison against random, SkyOrbs-like directional baseline, MAPPO without ISAC, and contention actor without ISAC. | `06_analysis/paper_tables/marl/p10_final_b10_b15_method_comparison_with_v4/marl_method_comparison.csv` | `06_analysis/paper_figures/marl/p10_final_b10_b15_method_comparison_with_v4/` | Manifest combines Phase9 five-way baselines with Phase10 gated/adaptive/topology/balanced gate summaries; filters: `slots_per_episode=3000`, `node_count=100`, `beamwidths=[10,15]`, `phase=eval_stochastic`. |
| `tab:n100_baseline`, `fig:gate_topology_tradeoff` | Gate-family operating-point tradeoff among default gated, adaptive v2, topology v3, balanced v4, and contention-heavy variants. | `06_analysis/paper_tables/marl/p10_gate_family_v2_v3_v4_tradeoff_comparison/seed_tradeoff_core_metrics.csv` | `06_analysis/paper_figures/marl/p10_gate_family_v2_v3_v4_tradeoff_comparison/` | Manifest combines Phase9 baselines and Phase10 seed31/32/33 plus adaptive/topology/balanced gate summaries; used for topology-vs-collision tradeoff claims. |

## Main Robustness/Boundary Evidence Chain

| Manuscript item | Claim role | Data/table source | Figure source | Notes |
|---|---|---|---|---|
| `fig:mobility_transfer` | Mobility-model transfer boundary at N=100, B=10. | `06_analysis/paper_tables/round5_mobility_transfer/manifest.json` | `06_analysis/paper_figures/round5_mobility_transfer/` | Supplementary mechanism-boundary evidence; not used as the final Phase10 comparison. |
| `fig:range_gain_heatmap`, `fig:supp_range_timing` | Communication/sensing range and slot-duration sensitivity. | `06_analysis/paper_tables/round3_robustness/range_rc_rs_grid/manifest.json`; `06_analysis/paper_tables/round6_slot_duration_sensitivity/manifest.json` | `06_analysis/paper_figures/round3_robustness/`; `06_analysis/paper_figures/round6_slot_duration_sensitivity/` | Used to justify protocol-level ISAC abstraction and timing sensitivity. |
| `fig:error_profile`, `fig:supp_b15_error` | Error-profile robustness under configured false-alarm, miss, and angular-offset settings. | `06_analysis/paper_tables/round3_robustness/error_profiles/manifest.json`; `06_analysis/paper_tables/round3_robustness/error_robustness/manifest.json`; `06_analysis/paper_tables/round8_error_profiles_b15_gm_rw_600slot/manifest.json` | `06_analysis/paper_figures/round3_robustness/`; `06_analysis/paper_figures/round8_error_profiles_b15_gm_rw_600slot/` | Error severity index is a visualization aid only; profile definitions remain configured protocol-level abstractions. |
| `fig:ablation_discovery` | Mechanism ablation for no ISAC, oracle-like/variant updates, and delay boundary. | `06_analysis/paper_tables/round4_delay_ablation/manifest.json` | `06_analysis/paper_figures/round4_delay_ablation/` | Used as implementation-boundary evidence, separate from the final trained Phase10 method comparison. |

## Supplement-Only Evidence Groups

| Supplement item | Evidence role | Provenance |
|---|---|---|
| `fig:supp_scale_transfer`, `fig:supp_beam_stress` | Archived N and beamwidth transfer/stress sweeps, including 3-/5-degree stress and historical 30-degree boundary points. | `06_analysis/paper_tables/round7_scale_beam_grid_light/manifest.json`; `06_analysis/paper_figures/round7_scale_beam_grid_light/transfer_figure_manifest.json`. |
| `fig:supp_area_scale` | N=100 density-preserving vs fixed-area scaling comparison. | `06_analysis/paper_tables/round3_robustness/n100_density_multiseed/manifest.json`; `06_analysis/paper_tables/round3_robustness/n100_fixed_multiseed/manifest.json`; `06_analysis/paper_figures/round3_n100_transfer/transfer_figure_manifest.json`. |
| `tab:supp_b3`, `fig:supp_b3` | N=100, B=3 stress case with five baselines. | `06_analysis/paper_tables/round9_n100_b3_full_baselines_600slot/manifest.json`; `06_analysis/paper_figures/round9_n100_b3_full_baselines_600slot/n100_b3_full_baselines_figure_manifest.json`. |
| `fig:supp_mobility_full` | Full-baseline mobility comparison at N=100, B=10/B=15. | `06_analysis/paper_tables/round8_n100_multimobility_missing_baselines_600slot/manifest.json`; `06_analysis/paper_figures/round8_n100_multimobility_full_baseline/full_baseline_figure_manifest.json`. |
| `fig:supp_cumulative_discovery`, `fig:supp_tradeoff` | Extra-seed backup trajectories and endpoint tradeoffs. | `06_analysis/paper_tables/pre11_evidence/manifest.json`; `06_analysis/paper_tables/round10_n100_b10_b15_extra_seeds/manifest.json`; `06_analysis/paper_figures/pre11_evidence/`. |
| `fig:supp_long_horizon` | 600-slot vs 3000-slot long-horizon sensitivity. | `06_analysis/paper_tables/round15_long_horizon_3000slot/manifest.json`; `06_analysis/paper_figures/round15_long_horizon_3000slot/`. |
| `fig:supp_round11_stability` | Focused paired-seed stability campaign. | `06_analysis/paper_tables/round11_paired_seed_campaign_main/manifest.json`; `06_analysis/paper_figures/round11_paired_seed_campaign_main/`. |
| `fig:supp_round13_collision_aware`, `fig:supp_round13_energy` | Collision-aware MAC refinement and assumed radio-state energy accounting. | `06_analysis/paper_tables/round13_collision_energy_10seed/manifest.json`; `06_analysis/paper_figures/round13_collision_energy_10seed/`. |
| `fig:supp_structured_marl`, `fig:supp_structured_marl_training` | Earlier structured neural MARL feasibility probe; not final main-method evidence. | `06_analysis/paper_tables/structured_marl_probe/manifest.json`; `06_analysis/paper_figures/structured_marl_probe/`. |
| `tab:supp_stat_examples`, `tab:supp_paired_delta`, `tab:supp_stats_index` | Representative statistical stability and paired-delta summaries. | `06_analysis/paper_tables/statistical_stability_summary/`; manuscript values were checked in the Phase10 claim-evidence audit. |

## Claim Boundary

- Primary manuscript performance claims should be tied to the Phase10 evidence chain: small-scale N=10, B=10 training; N=100, 3000-slot zero-shot transfer; B=10/B=15 final comparison; and the gate-family tradeoff table/figure.
- Round3-Round9 figures are retained as robustness, stress, and boundary evidence unless explicitly promoted by a later audit.
- The 3-degree and 5-degree cases are stress regimes; the 30-degree case is historical boundary evidence and is not part of the final main comparison.
- Concept figures document mechanism and architecture, but they do not independently validate numerical claims.

## Residual Risk Before Submission

- Some supplement-only figures summarize earlier 600-slot campaigns. The current manuscript labels those as boundary or robustness evidence, while the main numerical claim uses 3000-slot Phase10 transfer.
- The current audit is group-level. A stricter pre-submission archive can add per-file SHA256 hashes for every referenced image, CSV, and manifest.
