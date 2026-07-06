# Phase10 Topology-Adaptive Gate v3 Results

## Setup

- Training: `N=10`, `B=10 deg`, `300 slots`, `100 episodes`, seed `20260742`.
- Transfer test: `N=100`, `3000 slots`, stochastic policy.
- Beamwidths tested: `B=10 deg` and `B=15 deg`.
- Network: `topology_adaptive_gated_contention_shared`.
- Reward: `collision_topology`.

## Transfer Results

| Beamwidth | Discovery rate | CPD | Lambda2 | Collisions |
|---:|---:|---:|---:|---:|
| 10 deg | 0.3451 | 0.2000 | 14.052 | 3606.5 |
| 15 deg | 0.3857 | 0.1425 | 18.104 | 8476.1 |

## Comparison Interpretation

The v3 topology-preserving gate recovers large-scale topology growth relative to the aggressive adaptive v2 gate, but it does not improve collision efficiency. At `B=10 deg`, v3 reaches a discovery rate close to the ungated contention actor, but its CPD is lower because collision count is higher. At `B=15 deg`, v3 preserves better topology than adaptive v2, yet collision cost remains high.

This should be treated as a useful ablation rather than the final default method. It supports the claim that ISAC-informed candidate evidence alone is not enough; the access-control layer needs an explicit balance between topology expansion and collision suppression.

## Paper Use

- Keep v3 as a topology-heavy gate ablation.
- Use it to motivate the balanced topology gate v4.
- Main text should avoid claiming monotonic dominance. The more defensible claim is a tunable cross-layer frontier between collision efficiency and topology growth.

## Generated Artifacts

- Training figures: `06_analysis/paper_figures/marl/p10_topology_gate_v3_seed20260742_training_curves`.
- Transfer figures: `06_analysis/paper_figures/marl/p10_topology_gate_v3_n100_b10_b15_3000slot_10ep_stoch`.
- Method comparison: `06_analysis/paper_figures/marl/p10_topology_gate_v3_vs_phase9_b10_b15_method_comparison`.
- Gate-family tradeoff: `06_analysis/paper_figures/marl/p10_gate_family_v2_v3_tradeoff_comparison`.
