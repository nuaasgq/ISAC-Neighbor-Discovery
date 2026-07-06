# Phase10 Topology-Adaptive Gate v3 Plan

Created: 2026-07-07

## Motivation

The Phase10 gate-family results show a collision-topology tradeoff:

- seed31 suppresses collisions and improves CPD but gives up some topology recovery.
- seed33 recovers topology but raises collision count.
- adaptive v2 strongly suppresses collisions, especially at B=15, but over-throttles access and loses raw discovery and lambda2.

Topology-adaptive gate v3 is designed to test whether the access gate can preserve active access when local ISAC evidence and topology deficit are both strong.

## Mechanism

Compared with adaptive v2, v3 adds a topology-evidence term:

```text
topology_evidence = topology_need * candidate_score_max
```

This term reduces collision throttling when a node still lacks topology support and has a high-confidence candidate beam. It also adds a small direct active-access credit to Tx/Rx logits and reduces the idle preference under strong topology evidence.

## Hypothesis

The desired outcome is not maximum CPD alone. The target is a middle operating point:

- B=10: CPD closer to seed31 while lambda2 stays closer to seed33/contention actor.
- B=15: CPD above seed31 or near adaptive v2, with lambda2 higher than adaptive v2.

## Current Run

- Network: `topology_adaptive_gated_contention_shared`
- Method label: `topology_adaptive_gated_contention_actor`
- Training: N=10, B=10 deg, 300 slots, 100 episodes
- Seed: 20260742
- Output:
  - `05_simulation/results_raw/marl_campaign/p10_topology_gate_v3/train/train_n10_b10_topology_adaptive_gated_contention_actor_100ep_300slot_seed20260742`

## Acceptance Logic

v3 should be advanced to N=100 transfer only if the 100-episode stochastic small-scale eval is not degenerate and shows a meaningful balance among CPD, discovery, and lambda2. If v3 collapses to deterministic failure and weak stochastic topology like v2, it should remain a negative ablation rather than the main method.
