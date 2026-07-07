# Phase10 Final Balanced Gate v4 Results

## Setup

- Training: `N=10`, `B=10 deg`, `300 slots/episode`, `100 episodes`.
- Training seeds: `20260743`, `20260744`.
- Transfer: `N=100`, `3000 slots`, stochastic policy.
- Transfer beamwidths: `B=10 deg`, `B=15 deg`.
- Network: `balanced_topology_gated_contention_shared`.

## v4 Combined Transfer Results

| Beamwidth | Eval episodes | Discovery rate | CPD | Lambda2 | Collisions |
|---:|---:|---:|---:|---:|---:|
| 10 deg | 20 | 0.3127 | 0.2086 | 12.224 | 2477.3 |
| 15 deg | 20 | 0.3716 | 0.1598 | 17.547 | 6730.9 |

## Decision

Balanced gate v4 should not replace the current default main method. It improves the gate-family ablation by filling an intermediate point between collision-suppressing and topology-preserving policies, but it does not dominate gate31.

For the paper-facing default protocol profile:

- Use `gated_contention_actor` seed31 as the default low-collision balanced profile.
- Use `adaptive_gated_contention_actor` v2 as the aggressive collision-suppression ablation.
- Use `topology_adaptive_gated_contention_actor` v3 as the topology-heavy ablation.
- Use `balanced_topology_gated_contention_actor` v4 as the attempted balanced adaptive gate.

## Key Comparison

At `B=10 deg`, gate31 has the best CPD:

- gate31: CPD `0.2353`, lambda2 `11.904`, collisions `1403.3`.
- contention actor: CPD `0.2263`, lambda2 `14.381`, collisions `2563.7`.
- balanced v4: CPD `0.2086`, lambda2 `12.224`, collisions `2477.3`.

At `B=15 deg`, adaptive v2 has the highest CPD but weaker topology, while gate31 is the stronger balanced default:

- adaptive v2: CPD `0.2265`, lambda2 `10.993`, collisions `1463.7`.
- gate31: CPD `0.2114`, lambda2 `16.173`, collisions `3535.0`.
- balanced v4: CPD `0.1598`, lambda2 `17.547`, collisions `6730.9`.
- topology v3: CPD `0.1425`, lambda2 `18.104`, collisions `8476.1`.

## Paper Interpretation

The final result supports a tunable cross-layer access-control frontier rather than a single all-metric winner. ISAC-assisted MARL is necessary for large-scale transfer; the contention gate then selects the desired operating profile:

- low-collision balanced discovery: gate31;
- strict collision suppression: adaptive v2;
- topology-heavy discovery: topology v3;
- intermediate balanced adaptive attempt: v4.

This is defensible for a network/protocol paper because the contribution is the cross-layer abstraction and distributed access-control mechanism, not only a neural network score.

## Generated Artifacts

- v4 training curves: `06_analysis/paper_figures/marl/p10_balanced_gate_v4_training_curves`.
- v4 transfer curves: `06_analysis/paper_figures/marl/p10_balanced_gate_v4_n100_b10_b15_3000slot_10ep_stoch`.
- final method comparison: `06_analysis/paper_figures/marl/p10_final_b10_b15_method_comparison_with_v4`.
- final gate-family comparison: `06_analysis/paper_figures/marl/p10_gate_family_v2_v3_v4_tradeoff_comparison`.
