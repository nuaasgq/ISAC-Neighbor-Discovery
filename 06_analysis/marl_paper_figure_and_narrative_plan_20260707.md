# MARL+ISAC Paper Figure and Narrative Plan

## Main Figure Set

Use a compact main-text figure set. Avoid filling the main paper with pilot, smoke, and single-purpose debug figures.

1. Training convergence.
   - Preferred source: `06_analysis/paper_figures/marl/p10_gate_training_3seed_100ep_step_curves`.
   - Core figures: `marl_step_reward_curve.png`, `marl_eval_cpd_curve.png`, `marl_eval_lambda2_curve.png`, `marl_episode_collision_curve.png`.
   - If balanced gate v4 is promoted to the main method, use `06_analysis/paper_figures/marl/p10_balanced_gate_v4_training_curves` as the updated convergence source.

2. Five-way baseline comparison.
   - Preferred source if v4 does not dominate: `06_analysis/paper_figures/marl/p10_gate31_vs_phase9_b10_b15_method_comparison`.
   - Core figures: `marl_method_comparison_discovery_rate.png`, `marl_method_comparison_collision_penalized_discovery_rate.png`, `marl_method_comparison_lambda2.png`, `marl_method_comparison_collision_count.png`.

3. ISAC search-space compression.
   - Source: `06_analysis/paper_figures/marl/phase9_fiveway_n100_b10_3000slot_10ep_stoch_all_methods` and `06_analysis/paper_figures/marl/phase9_fiveway_n100_b15_3000slot_10ep_stoch_all_methods`.
   - Core figure: `marl_transfer_beam_empty_scan_ratio.png`.
   - Purpose: show that ISAC candidate memory suppresses empty-beam scans compared with blind/random/no-ISAC policies.

4. Gate-family tradeoff.
   - Current source: `06_analysis/paper_figures/marl/p10_gate_family_v2_v3_tradeoff_comparison`.
   - Update after v4 transfer: `p10_gate_family_v2_v3_v4_tradeoff_comparison`.
   - Core figures: CPD, lambda2, collision count.

## Appendix and Ablation Figures

The following should be appendix/supplementary unless one becomes the final selected method:

- `06_analysis/paper_figures/marl/p10_adaptive_gate_v2_seed20260741_training_curves`.
- `06_analysis/paper_figures/marl/p10_topology_gate_v3_seed20260742_training_curves`.
- `06_analysis/paper_figures/marl/p10_gate_seed31_seed32_seed33_tradeoff_comparison`.
- Pilot, smoke, phase1, phase2, and low-episode probe directories.

## Defensible Innovation Statement

The strongest current claim is not that a single learned policy dominates every metric. The defensible contribution is a cross-layer, distributed neighbor-discovery framework:

- ISAC-derived non-empty beam evidence is abstracted as link-layer candidate memory.
- A parameter-shared MARL policy is trained in a small `N=10`, `B=10 deg` scenario and transferred to larger `N=100` swarms and different beamwidths.
- A decentralized contention gate regulates the active access probability from local topology need, candidate confidence, and collision pressure.
- The resulting protocol forms a tunable frontier between collision efficiency and topology growth.

If v4 succeeds, phrase the final method as a balanced topology-aware access gate. If v4 does not improve over gate31, use gate31 as the default low-collision profile and place adaptive v2, topology v3, and balanced v4 in the gate-family ablation.

## Minimum Data Package Before 2026-07-08 09:30

1. v4 large-scale transfer summary:
   - `06_analysis/paper_tables/marl/p10_balanced_gate_v4_n100_b10_b15_3000slot_10ep_stoch/marl_transfer_summary.csv`.

2. v4 method comparison:
   - `06_analysis/paper_figures/marl/p10_balanced_gate_v4_vs_phase9_phase10_method_comparison`.
   - Include discovery rate, CPD, lambda2, collision count.

3. Final gate-family comparison:
   - Include seed31, seed32, seed33, adaptive v2, topology v3, and balanced v4.
   - Use this to justify the final method choice or the tunable-frontier interpretation.

4. Final written result note:
   - State whether v4 is promoted to the main method.
   - If not, state that gate31 is the main low-collision protocol profile and v4 is an ablation.
