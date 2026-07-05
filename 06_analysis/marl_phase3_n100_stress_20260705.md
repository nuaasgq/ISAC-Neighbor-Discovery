# Phase 3 N=100 Long-Horizon MARL Transfer Stress Report

## Experiment scope

- Training policy: `isac_mappo`, shared actor-critic, trained at `N=10`, `10 deg`, `300 slots/episode`.
- Evaluation policy: zero-shot transfer to `N=100`.
- Evaluation horizon: primary long test is `3000 slots` (`15 s` at `5 ms/slot`). The earlier `3 deg, 1200 slots` point is retained only as a short-window stress reference.
- Ranges: communication and sensing ranges are both set to `900 m`, which is larger than the current region diagonal and therefore keeps the first-stage transfer test in a single-hop setting.
- Discovery-rate definition: `discovered_edges / true_edges_seen`, where `true_edges_seen` is the number of one-hop neighbor pairs that are physically discoverable during the episode.

## Main N=100 results

| Beamwidth | Slots | Episodes | Discovery rate | Collision-penalized discovery | Lambda2 | Empty-scan ratio | Collisions/episode | Comment |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 deg | 1200 | 1 | 0.081 | 0.077 | 0.820 | 0.544 | 276 | Short-window extreme narrow-beam reference. |
| 3 deg | 3000 | 1 | 0.137 | 0.112 | 4.660 | 0.358 | 1082 | Long horizon improves connectivity, but beam search remains hard. |
| 5 deg | 3000 | 1 | 0.265 | 0.138 | 13.077 | 0.118 | 4583 | Transition point where full connectivity and useful algebraic connectivity emerge. |
| 10 deg | 3000 | 4 | 0.383 | 0.077 | 16.200 | 0.038 | 19969 | Stable transfer from the training codebook; collisions become significant. |
| 15 deg | 3000 | 4 | 0.426 | 0.032 | 23.406 | 0.010 | 60814 | Best raw discovery/connectivity, but collision penalty is severe. |
| 30 deg | 3000 | 4 | 0.322 | 0.009 | 13.044 | 0.004 | 178328 | Wide beams reduce empty scans but create severe contention/collision. |

## Interpretation

The current evidence supports the core feasibility claim: a policy trained only on `N=10`, `10 deg`, and `300-slot` episodes can transfer to `N=100` and maintain a connected discovered topology under long-horizon evaluation. This is strongest at `5-15 deg`, where `largest_component_size=100` and lambda2 is clearly positive.

The current evidence also exposes the next paper-level bottleneck. ISAC removes the empty-beam search burden, but large-scale contention is now dominant. As beamwidth increases, empty scans approach zero, while collisions grow rapidly. Therefore, the next algorithmic innovation should focus on collision-aware, topology-aware, and scalable coordination rather than only improving beam selection.

## Code and analysis changes in this phase

- Training default changed to `300 slots` in `05_simulation/run_marl_training.py`.
- Evaluation default changed to `3000 slots` in `05_simulation/run_marl_evaluate.py`.
- Evaluation now writes `eval_episode_metrics.csv` and `progress.json` after each episode, so long tests no longer lose all progress if interrupted.
- Shared actor inference now supports batched per-slot action selection over all agents, which is needed for large `N` and high beam counts.
- Transfer plotting now includes collision-penalized discovery, largest component, LCC ratio, isolated-node ratio, collisions per discovery, scan efficiency, and energy efficiency.
- Manifest parsing now accepts UTF-8 BOM so repaired or externally generated manifests are not silently skipped.

## Next experiment decisions

1. Keep training horizon at `300 slots/episode`; use `1200/3000 slots` for evaluation depending on codebook size.
2. Treat `3 deg` and `5 deg` as expensive stress cases; use 1 episode first, then add repeats only after runtime is reduced further.
3. Use `10/15/30 deg` for repeated long-horizon statistics because their runtime is manageable.
4. Prioritize collision-aware MARL structure next: contention-state features, topology-deficit scheduling, adaptive TX/RX role balance, and collision-aware reward.
5. For the paper narrative, present ISAC as enabling candidate-space reduction, and present the MARL/network-structure contribution as resolving the post-ISAC contention bottleneck.
