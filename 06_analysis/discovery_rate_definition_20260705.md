# Discovery-Rate Definition

Date: 2026-07-05

The simulator reports a finite-horizon discovery rate:

```text
discovery_rate = |discovered_edges| / max(1, |first_true_slot|)
```

where:

- `first_true_slot` stores every unordered UAV pair that has entered communication range at least once during the episode.
- `discovered_edges` stores UAV pairs that have completed a bidirectional narrow-beam handshake.
- `finite_time_discovery_rate` is an alias of `discovery_rate` in the CSV writer.

Therefore, in the single-hop N=100 setting, the denominator is normally all 4950 possible undirected pairs. A discovery is not counted merely because ISAC senses an occupied beam cell; the edge is counted only after bidirectional alignment and handshake confirmation.

The collision-penalized variant is:

```text
collision_penalized_discovery_rate = |discovered_edges| / max(1, |first_true_slot| + collision_count)
```

This metric is used as a MAC-cost diagnostic. It is intentionally stricter than raw discovery rate when a policy gains discoveries by concentrating many nodes on the same beam cells and causing collisions.

Source locations:

- `05_simulation/src/isac_nd_sim/simulator.py`: true-edge tracking, discovered-edge tracking, endpoint metrics.
- `05_simulation/src/isac_nd_sim/runner.py`: `finite_time_discovery_rate` alias.
