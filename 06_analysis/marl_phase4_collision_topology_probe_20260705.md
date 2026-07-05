# Phase 4 Collision-Aware Shared MARL Probe

## Purpose

Phase 3 showed that ISAC-assisted MARL can transfer from `N=10` training to `N=100` long-horizon testing, but large-scale contention becomes the dominant bottleneck. This phase tests whether a collision/topology-aware reward can reduce collisions without losing the ISAC-driven discovery gain.

## Setup

- Training: `isac_mappo`, shared actor-critic, `collision_topology` reward.
- Training scale: `N=10`, `10 deg`, `300 slots/episode`, 20 episodes.
- Evaluation: zero-shot `N=100`, `3000 slots`, `10/15/30 deg`.
- Episodes: 4 per beamwidth after combining the 1-episode probe and the 3-episode repeat.
- Ranges: `R_c = R_s = 900 m`, single-hop validation setting.

## Legacy vs collision-aware transfer

| Beamwidth | Legacy discovery | Collision-aware discovery | Delta | Legacy collisions | Collision-aware collisions | Collision reduction | Legacy collision-penalized discovery | Collision-aware collision-penalized discovery | Lambda2 delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 deg | 0.3830 | 0.3867 | +0.0037 | 19969.0 | 13862.2 | 30.58% | 0.0768 | 0.1018 | -0.648 |
| 15 deg | 0.4265 | 0.4495 | +0.0230 | 60814.2 | 54572.5 | 10.26% | 0.0323 | 0.0374 | -0.699 |
| 30 deg | 0.3219 | 0.3476 | +0.0257 | 178328.2 | 168078.2 | 5.75% | 0.0088 | 0.0100 | -0.847 |

## Interpretation

The result is directionally useful. Collision-aware training improves raw discovery rate at all three evaluated beamwidths and reduces collision count, with the strongest effect at `10 deg`. It also improves collision-penalized discovery at all three beamwidths, which is the metric most aligned with large-scale protocol efficiency.

The tradeoff is that algebraic connectivity is slightly lower than the legacy reward under the same long-horizon transfer setting. The drop is small at `10/15 deg` and larger at `30 deg`. This suggests that the collision penalty reduces aggressive simultaneous access, but the topology-shaping term is not yet strong enough to fully preserve graph spectral quality.

## Paper implication

This phase gives a concrete second-layer contribution beyond ISAC candidate-space reduction:

1. ISAC reduces empty-beam waste and makes narrow-beam discovery feasible.
2. Collision/topology-aware MARL then addresses the post-ISAC contention bottleneck at large scale.
3. The remaining challenge is to jointly optimize collision avoidance and topology spectral quality, likely through explicit role-balance and graph-aware actor features rather than reward shaping alone.

## Next optimization target

The next model should keep the collision-aware reward but add a topology-preserving actor structure:

- local TX/RX role-load feature,
- recent collision intensity feature,
- target-degree deficit and neighbor-diversity token,
- graph-aware action prior that suppresses overused beams and roles,
- stronger lambda2 or component-merge shaping only when the discovered graph is weak.

This should be tested first at `N=10` training and transferred to `N=100`, because the current results confirm the 300-slot training and long-horizon transfer protocol is viable.
