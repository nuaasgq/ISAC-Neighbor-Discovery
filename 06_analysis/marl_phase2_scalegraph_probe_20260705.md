# Phase-2 Scalegraph and Collision-Topology Probe

## Purpose

This probe tests whether the new `scalegraph_beam` network and `collision_topology` reward can improve the real MARL + ISAC neighbor-discovery line without changing the core 300-slot training rule.

## Training Runs

All runs use:

- Source setting: N=10, 10 deg beams.
- Training horizon: 20 episodes, 300 slots per episode.
- Evaluation mode: deterministic and stochastic separated.
- Network comparison:
  - `shared + legacy`: original shared beam-token actor.
  - `scalegraph_beam + legacy`: query-based beam-set attention actor.
  - `scalegraph_beam + collision_topology`: same actor with collision/topology reward shaping.

Training-curve outputs:

- Tables: `06_analysis/paper_tables/marl/phase2_train_compare/`
- Figures: `06_analysis/paper_figures/marl/phase2_train_compare/`

## Training Observations

- `scalegraph_beam + legacy` is runnable and transferable across beam codebooks because its parameters do not depend on the number of beams.
- Stochastic execution remains much stronger than deterministic argmax execution.
- `scalegraph_beam + collision_topology` improves the training-environment stochastic evaluation slightly and reduces collisions:
  - At episode 20, stochastic discovery is about 0.781 for collision/topology vs 0.744 for legacy.
  - At episode 20, stochastic collision count is about 49 for collision/topology vs higher legacy collision levels.

## Transfer Probe

Single-episode 3000-slot transfer probes at 15 deg beams:

| Method | N | Discovery rate | Lambda2 | Empty-scan ratio | Collision count | Collision-penalized discovery |
|---|---:|---:|---:|---:|---:|---:|
| scalegraph + legacy | 20 | 0.863 | 13.463 | 0.114 | 2710 | 0.057 |
| scalegraph + collision_topology | 20 | 0.868 | 13.879 | 0.130 | 1790 | 0.083 |
| scalegraph + legacy | 50 | 0.536 | 16.957 | 0.020 | 12091 | 0.049 |
| scalegraph + collision_topology | 50 | 0.535 | 16.864 | 0.024 | 8432 | 0.068 |

Interpretation:

- `scalegraph_beam` alone does not yet beat the current `shared + legacy` phase-1 result on discovery rate.
- `collision_topology` preserves discovery rate while reducing collision count by about 34% at N=20 and about 30% at N=50 in these probes.
- The most defensible method contribution is therefore not "scalegraph directly improves discovery rate"; it is "scalegraph enables a scalable architecture, while collision/topology reward improves collision-normalized efficiency."

## Next Gate

1. Complete the no-ISAC phase-1 baseline matrix.
2. Run 3-episode transfer for `scalegraph_beam + collision_topology` only at the strongest stress points first:
   - N=20, 15 deg, 3000 slots.
   - N=50, 15 deg, 3000 slots.
   - N=50, 10 deg, 3000 slots.
3. If the collision-normalized gain holds, expand to 5/10/15/30 deg and then N=100.
4. If discovery rate stays below shared, keep `shared + legacy` as the main discovery-rate method and use `collision_topology` as a collision-aware variant/ablation.
