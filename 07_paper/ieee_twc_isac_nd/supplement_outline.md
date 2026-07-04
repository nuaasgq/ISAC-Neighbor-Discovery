# Supplementary Material Outline

Purpose: keep the main IEEE draft compact while preserving the full experiment evidence chain.

## S1. Reproducibility Metadata

- Repository commit for the submitted package.
- Simulation entry points:
  - `05_simulation/run_training.py`
  - `05_simulation/run_transfer_sweep.py`
  - `06_analysis/scripts/plot_round3_results.py`
  - `06_analysis/scripts/plot_transfer_results.py`
  - `06_analysis/scripts/plot_mobility_results.py`
  - `06_analysis/scripts/plot_mobility_full_baselines.py`
  - `06_analysis/scripts/plot_baseline_stress.py`
  - `06_analysis/scripts/plot_slot_duration_results.py`
  - `06_analysis/scripts/draw_concept_figures.py`
  - `06_analysis/scripts/build_statistical_summary.py`
  - `06_analysis/scripts/build_mobility_full_baseline_table.py`
- Statistical summary table:
  - `06_analysis/paper_tables/statistical_stability_summary/statistical_stability_summary.csv`

## S2. Training Curves

Use:

- `06_analysis/paper_figures/training_round2_candidate/`
- `06_analysis/paper_figures/round7_long_cem_training/`

Recommended figures:

- Reward/score evolution.
- Discovery-rate evolution.
- Empty-scan evolution.
- Collision and connectivity evolution.

Interpretation rule: empirical training-score stabilization only, not a theoretical convergence proof.

## S3. Full Transfer and Beamwidth Stress

Use:

- `06_analysis/paper_tables/round7_scale_beam_grid_light/aggregate_metrics.csv`
- `06_analysis/paper_figures/round7_scale_beam_grid_light/`
- `06_analysis/paper_tables/round9_n100_b3_full_baselines_600slot/aggregate_metrics.csv`
- `06_analysis/paper_figures/round9_n100_b3_full_baselines_600slot/`

Recommended focus:

- `N=10,20,50,100` and `3,5,10,15,30` degree transfer.
- Raw discovery versus collision-penalized discovery.
- Full five-baseline comparison for the `N=100`, 3-degree stress case.
- State explicitly that 3/5-degree beams are stress regimes.

## S4. Mobility Boundary and Missing Baselines

Use:

- `06_analysis/paper_tables/round7_n100_multimobility_600slot/aggregate_metrics.csv`
- `06_analysis/paper_tables/round8_n100_multimobility_full_baseline/combined_aggregate_metrics.csv`
- `06_analysis/paper_figures/round7_n100_multimobility_600slot/`
- `06_analysis/paper_figures/round8_n100_multimobility_full_baseline/`

Recommended focus:

- Gauss-Markov and random-walk are favorable tested regimes.
- Random-direction and random-waypoint remain stress regimes.
- SkyOrbs-like and vanilla RL without ISAC remain near zero in the same large-scale mobility settings.

## S5. ISAC Error Robustness

Use:

- Main B=10 evidence: `06_analysis/paper_tables/round3_robustness/error_profiles/aggregate_metrics.csv`
- Round7 B=10 confirmation: `06_analysis/paper_tables/round7_error_profiles_light/aggregate_metrics.csv`
- B=15 mobility-sensitive extension: `06_analysis/paper_tables/round8_error_profiles_b15_gm_rw_600slot/aggregate_metrics.csv`

Recommended focus:

- Full ISAC remains useful under configured moderate errors.
- B=15 Gauss-Markov remains strong but collision counts are high.
- B=15 random-walk declines more clearly with error severity.
- Do not imply physical-layer estimator immunity.

## S6. Range and Timing Sensitivity

Use:

- `06_analysis/paper_tables/round3_robustness/range_rc_rs_grid/aggregate_metrics.csv`
- `06_analysis/paper_tables/round6_slot_duration_sensitivity/aggregate_metrics.csv`

Recommended focus:

- The `Rs/Rc` saturation result is model-internal to communication-neighbor discovery.
- The 5 ms slot assumption is not a single tuned point in the Gauss-Markov setting.

## S7. Mechanism and Implementation Boundary

Use:

- `06_analysis/paper_tables/round4_delay_ablation/aggregate_metrics.csv`
- `06_analysis/paper_figures/round4_delay_ablation/`
- `06_analysis/paper_figures/round7_error_profiles_light/ablation_*`
- `06_analysis/paper_figures/round8_error_profiles_b15_gm_rw_600slot/ablation_*`

Recommended focus:

- Candidate-set refinement is the largest observed mechanism contributor at the main `N=100`, 10-degree operating point.
- One-slot delay preserves much of the benefit and reduces collisions.
- Topology-aware and beam-lock refinements are secondary contributors under current evidence.
