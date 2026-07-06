# Phase10 Gated Seed Tradeoff Results

Created: 2026-07-07

## Scope

This note records the seed-level behavior of the gated contention actor trained at N=10, B=10 deg, 300 slots, then transferred to N=100, 3000 slots at B=10 and B=15. It separates checkpoint behavior instead of averaging all gated checkpoints into a single method curve.

## Artifacts

- seed31 transfer summary:
  - `06_analysis/paper_tables/marl/p10_gate31_n100_b10_b15_3000slot_10ep_stoch/marl_transfer_summary.csv`
- seed32 transfer summary:
  - `06_analysis/paper_tables/marl/p10_gate32_n100_b10_b15_3000slot_10ep_stoch/marl_transfer_summary.csv`
- seed33 transfer summary:
  - `06_analysis/paper_tables/marl/p10_gate33_n100_b10_b15_3000slot_10ep_stoch/marl_transfer_summary.csv`
- seed tradeoff tables:
  - `06_analysis/paper_tables/marl/p10_gate_seed31_seed32_seed33_tradeoff_comparison/seed_tradeoff_core_metrics.csv`
  - `06_analysis/paper_tables/marl/p10_gate_seed31_seed32_seed33_tradeoff_comparison/seed_tradeoff_method_comparison.csv`
- seed tradeoff figures:
  - `06_analysis/paper_figures/marl/p10_gate_seed31_seed32_seed33_tradeoff_comparison`
- reproducible plotting script:
  - `06_analysis/scripts/plot_marl_gated_seed_tradeoff.py`

All generated seed tradeoff figures are 1920 x 1440 pixels.

## Key Metrics

| Beamwidth | Checkpoint | Discovery | CPD | Lambda2 | Collisions |
|---:|---|---:|---:|---:|---:|
| 10 deg | seed31 low-collision | 0.3020 | 0.2353 | 11.9043 | 1403.3 |
| 10 deg | seed32 topology/collision-heavy | 0.3653 | 0.1658 | 17.4428 | 5965.8 |
| 10 deg | seed33 topology | 0.3644 | 0.2035 | 16.4072 | 3938.0 |
| 15 deg | seed31 low-collision | 0.3618 | 0.2114 | 16.1730 | 3535.0 |
| 15 deg | seed32 topology/collision-heavy | 0.3937 | 0.1106 | 19.5043 | 12712.1 |
| 15 deg | seed33 topology | 0.4136 | 0.1497 | 21.7437 | 8745.4 |

## Interpretation

The two checkpoints expose a real collision-topology tradeoff:

- seed31 is best for collision-penalized discovery and collision suppression.
- seed33 is best for raw discovery and algebraic connectivity.
- seed32 confirms that high topology recovery can come with excessive contention; it is not the best checkpoint for either CPD or collision count.
- No checkpoint dominates across all metrics.

This is useful rather than fatal: it identifies a concrete mechanism-improvement target. Adaptive gating should suppress repeated high-collision access while preserving topology recovery when candidate confidence and degree deficit are high.

## Next Step

Use the three-seed result to present gated contention as a tunable access-control mechanism rather than a single universally dominant checkpoint. The adaptive gate v2 run should be interpreted as a follow-up attempt to close the CPD-lambda2 gap.
