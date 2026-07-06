# Phase10 Gate-Family Final Results

Created: 2026-07-07

## Scope

This note summarizes the N=10 to N=100 transfer evidence for the gated MARL+ISAC family under the current deadline run. All policies are trained at N=10, B=10 deg, 300 slots, then evaluated at N=100, 3000 slots, stochastic mode, with communication and sensing ranges set to 900 m.

## Main Artifacts

- Three-seed gated tradeoff:
  - `06_analysis/paper_tables/marl/p10_gate_seed31_seed32_seed33_tradeoff_comparison`
  - `06_analysis/paper_figures/marl/p10_gate_seed31_seed32_seed33_tradeoff_comparison`
- Gate-family tradeoff including adaptive v2:
  - `06_analysis/paper_tables/marl/p10_gate_seed_and_adaptive_tradeoff_comparison`
  - `06_analysis/paper_figures/marl/p10_gate_seed_and_adaptive_tradeoff_comparison`
- Adaptive v2 transfer:
  - `06_analysis/paper_tables/marl/p10_adaptive_gate_v2_n100_b10_b15_3000slot_10ep_stoch`
  - `06_analysis/paper_figures/marl/p10_adaptive_gate_v2_n100_b10_b15_3000slot_10ep_stoch`

## Core Comparison

| Beamwidth | Variant | Discovery | CPD | Lambda2 | Collisions |
|---:|---|---:|---:|---:|---:|
| 10 deg | contention actor | 0.3429 | 0.2263 | 14.3814 | 2563.7 |
| 10 deg | gated seed31 | 0.3020 | 0.2353 | 11.9043 | 1403.3 |
| 10 deg | gated seed33 | 0.3644 | 0.2035 | 16.4072 | 3938.0 |
| 10 deg | adaptive v2 | 0.2203 | 0.2027 | 7.5506 | 429.6 |
| 15 deg | contention actor | 0.4233 | 0.1387 | 19.8887 | 10226.9 |
| 15 deg | gated seed31 | 0.3618 | 0.2114 | 16.1730 | 3535.0 |
| 15 deg | gated seed33 | 0.4136 | 0.1497 | 21.7437 | 8745.4 |
| 15 deg | adaptive v2 | 0.2935 | 0.2265 | 10.9925 | 1463.7 |

## Defensible Claims

1. ISAC-derived candidate memory is necessary for useful large-scale transfer. No-ISAC MARL baselines remain near zero discovery in Phase9.
2. Gated contention control improves effective discovery under contention, but the best checkpoint depends on whether the objective prioritizes CPD/collisions or topology.
3. Adaptive v2 verifies that the access gate can intentionally move the operating point toward collision suppression. It is not the final dominant method because it sacrifices raw discovery and lambda2.
4. The paper should present a cross-layer MARL+ISAC access-control mechanism and a tunable CPD-topology tradeoff, not an all-metric dominance claim.

## Recommended Paper Positioning

Use seed31 as the main low-collision gated result, seed33 as the topology-recovery gated result, and adaptive v2 as the mechanism ablation showing that the gate can push collision suppression further. The next method variant should add topology-preserving annealing or a lambda2-aware gate target to combine seed31/adaptive collision control with seed33 topology recovery.
