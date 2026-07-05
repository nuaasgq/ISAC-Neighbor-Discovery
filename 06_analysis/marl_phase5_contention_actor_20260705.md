# Phase 5 Contention-Aware Actor Report

## Purpose

Phase 4 confirmed that collision-aware reward shaping reduces large-scale contention, but it still relies mainly on reward feedback. Phase 5 adds a network-structure contribution: a contention/topology-aware shared actor that consumes local contention memory and applies a role-gating prior during decentralized execution.

## Method change

The new `contention_shared` actor adds:

- per-beam local collision memory through `beam_collision`;
- a 10-dimensional `contention_state` vector derived from local success/failure/collision/empty-beam history, candidate sparsity, and topology deficit;
- a contention encoder that feeds the mode head and beam query;
- an explicit role-gating prior that suppresses TX/RX under high local collision pressure and increases sensing/idle exploration when needed;
- a beam-risk residual that suppresses repeatedly colliding beams while preserving ISAC candidate attraction.

The actor still uses only decentralized public observations. No true positions, true neighbor edges, or hidden adjacency are exposed.

## V1 failure and V2 fix

The first contention actor exposed `contention_state` to the network but did not impose an explicit role gate. N=100 probing showed excessive TX behavior, especially at 30 deg, which increased collisions. V2 added the explicit contention mode prior and strengthened beam collision suppression before retraining. This is the version used below.

## N=100 long-horizon comparison

All methods are trained at `N=10`, `10 deg`, `300 slots/episode`, then evaluated at `N=100`, `3000 slots`, single-hop range setting. The table uses 4 evaluation episodes per beamwidth.

| Beamwidth | Legacy discovery | Collision-reward discovery | Contention-actor discovery | Legacy collisions | Collision-reward collisions | Contention-actor collisions | Collision reduction vs legacy | Collision reduction vs collision reward | Legacy CPD | Collision-reward CPD | Contention-actor CPD | Contention lambda2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 deg | 0.3830 | 0.3867 | 0.3611 | 19969.0 | 13862.2 | 5891.5 | 70.50% | 57.50% | 0.0768 | 0.1018 | 0.1648 | 16.149 |
| 15 deg | 0.4265 | 0.4495 | 0.4249 | 60814.2 | 54572.5 | 23017.8 | 62.15% | 57.82% | 0.0323 | 0.0374 | 0.0753 | 19.973 |
| 30 deg | 0.3219 | 0.3476 | 0.3635 | 178328.2 | 168078.2 | 92276.8 | 48.25% | 45.10% | 0.0088 | 0.0100 | 0.0186 | 12.736 |

CPD means collision-penalized discovery rate.

## Interpretation

The contention-aware actor materially improves the large-scale protocol-efficiency metric. Compared with legacy shared MARL, it reduces collisions by 48-70% across 10/15/30 deg and improves collision-penalized discovery at every tested beamwidth. Compared with reward-only collision training, it still reduces collisions by 45-58%, which supports the claim that network structure adds value beyond reward shaping.

The tradeoff is raw discovery. At 10/15 deg, raw discovery is slightly below the reward-only collision model, while at 30 deg it is higher. This is acceptable for the current research direction because the wide-beam/high-density regime is collision-limited; however, the final method should recover some 10/15 deg raw discovery without losing the collision reduction.

## Paper implication

The current method stack is now clearer:

1. ISAC-assisted candidate reduction addresses empty-beam waste under narrow beams.
2. Collision/topology-aware reward addresses post-ISAC large-scale contention.
3. Contention-aware actor structure uses decentralized local memory to further reduce collisions and improve efficiency under N=100 transfer.

This gives both mechanism innovation and method innovation. The next paper-level improvement should be a Pareto-tuned role prior or adaptive role temperature so that 10/15 deg raw discovery recovers while the collision gains remain.

## Artifacts

- Transfer table: `06_analysis/paper_tables/marl/phase5_contention_shared_v2_transfer_probe/marl_transfer_summary.csv`
- Training curves: `06_analysis/paper_figures/marl/phase5_contention_shared_v2_train_curves/`
- Transfer curves: `06_analysis/paper_figures/marl/phase5_contention_shared_v2_transfer_probe/`
- Three-method comparison: `06_analysis/paper_figures/marl/phase5_method_comparison/`
