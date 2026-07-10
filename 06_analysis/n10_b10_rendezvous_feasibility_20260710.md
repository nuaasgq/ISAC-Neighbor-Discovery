# N=10/B=10 Rendezvous Feasibility Analysis (2026-07-10)

## Random Alignment Scaling

With `M` beams, independent TX/RX probability 0.5/0.5, `N` nodes, and `T` slots, the expected number of directed reciprocal random alignments is

```text
lambda = T N (N - 1) (0.5)(0.5) / M^2.
```

Using a Poisson rare-event approximation, the probability of at least one alignment is `1 - exp(-lambda)`.

| Beamwidth | Nodes | Slots | Expected alignments | P(at least one) |
|---:|---:|---:|---:|---:|
| 10 deg | 10 | 300 | 0.0161 | 0.0159 |
| 10 deg | 10 | 3000 | 0.1608 | 0.1485 |
| 10 deg | 50 | 300 | 0.4376 | 0.3544 |
| 10 deg | 100 | 300 | 1.7683 | 0.8294 |
| 15 deg | 10 | 300 | 0.0814 | 0.0782 |
| 15 deg | 10 | 3000 | 0.8138 | 0.5568 |

Thus both node count and horizon explain part of the zero result, but neither is an acceptable fix for the stated small-to-large transfer objective. Training at N=10/B=10 must obtain its signal from sensing-driven rendezvous rather than rare random coincidence.

## Diagnostic Iterations

1. Highest-score candidate lock: many guided actions, zero mutual reports and zero alignment.
2. Sparse position-hash rendezvous windows: exploration recovered, but historical body-frame beam indices became stale under attitude updates.
3. Deterministic sensing bootstrap plus rendezvous windows: more reports, still zero while using stale beam indices.
4. Global position report reprojected into the current body frame: 10 alignments and 10 discoveries across the same three seeds.

The fourth result isolates the useful cross-layer mechanism: ISAC supplies an anonymous global position estimate, navigation supplies current self position/attitude, and the data-link layer converts those measurements into a fresh communication beam and rendezvous state.

## MARL Handoff

The final actor must still output only `TX/RX/beam`. Its local observation should add:

- reprojected per-beam rendezvous score;
- report confidence and normalized staleness;
- whether the current slot matches the anonymous position-pair rendezvous phase;
- signed local role hint (`TX`, `RX`, or unavailable).

The 200-slot deterministic bootstrap remains a diagnostic expert only. It must be removed or ablated in the learned policy. The next training probe should first establish nonzero rewards with these observations and a measurement-derived auxiliary beam objective, then compare against a zero-rendezvous-feature ablation.

## MARL Handoff Result

The handoff gate subsequently passed without the deterministic bootstrap. A zero-initialized learned ISAC evidence adapter and measurement-derived beam/role losses produced eight discoveries across six held-out N=10/B=10/300-slot episodes (mean discovery rate 2.96%, 5/6 nonzero). Uniform random, Wang-table, and adapter-zero controls remained at zero on the same scenarios.

The remaining bottleneck is reciprocal opportunity formation rather than single-agent beam learning: held-out target-beam hit rate reached 86.77% and joint beam-role rate reached 72.28%, while only 65 common-phase pair-slots emerged from 1069 reciprocal-report pair-slots. See `06_analysis/rendezvous_learnability_gate_20260710.md`.
