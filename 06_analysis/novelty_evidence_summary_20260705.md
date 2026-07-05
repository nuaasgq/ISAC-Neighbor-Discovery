# Novelty and Evidence Summary - 2026-07-05

## Working Thesis

This project should be positioned as a cross-layer link-layer neighbor-discovery paper, not as a pure physical-layer ISAC paper and not as a full neural MARL paper.

Core claim:

> An imperfect ISAC beam-cell occupancy prior can be exposed to the data-link layer to reduce empty narrow-beam scanning and improve finite-horizon discovered-neighbor graph formation in distributed-execution UAV-UAV neighbor discovery, while every edge is still confirmed only by bidirectional narrow-beam handshake.

## Main Novelty

1. **Problem setting**: distributed-execution UAV-UAV narrow-beam neighbor discovery with own pose known, but no undiscovered-neighbor identity, pose, beam state, or central scheduler.
2. **Cross-layer abstraction**: ISAC is modeled as an imperfect beam-cell occupancy-prior service with false alarm, missed detection, angular-cell offset, staleness, and sensing-range support.
3. **Protocol mechanism**: blind first probing is followed by ISAC belief update, candidate-beam refinement, exploration-floor protection, beam-lock memory, and local discovered-degree/topology-deficit prioritization.
4. **Scalable tuning path**: a shared-parameter protocol-tuning workflow trains at N=10 and transfers without fine-tuning to larger N and different beamwidths in the tested finite-horizon regimes.
5. **Evidence discipline**: the paper explicitly separates useful operating regions (mainly 10--30 deg in smooth mobility) from stress/failure-boundary regions (3--5 deg and abrupt mobility).

## Strongest Evidence Chain

| Evidence item | Current result | Why it matters |
|---|---:|---|
| N=100, B=10, density-scaled main comparison | Round14 ten-seed proposed discovery 0.3652 vs enhanced no-ISAC 0.0006; lambda2 13.2595 vs 0; 10/10 positive paired discovery deltas versus all four communication-only controls | Establishes the core empty-beam pruning value of ISAC under large narrow-beam search. |
| N=100, B=15, density-scaled transfer | Proposed discovery 0.5440; lambda2 26.84 | Shows the mechanism is not a single B=10 artifact, while collision caveats remain. |
| Round14 paired delta, N=100/B=10 | Discovery deltas are positive in 10/10 paired seeds versus random, SkyOrbs-like, learned no-ISAC, and enhanced no-ISAC controls; proposed discovery is 0.3652 and lambda2 is 13.2595 | Provides the strongest seed-matched main-table stability evidence. |
| Candidate-set ablation | Removing candidate-set refinement drops discovery from 0.3655 to 0.0313 | Identifies the actual ISAC mechanism rather than attributing gains to generic learning. |
| One-slot delay ablation | Delayed ISAC still reaches discovery 0.2989 and lambda2 8.47 | Bounds the low-latency same-slot assumption and improves implementation credibility. |
| Mobility boundary | Gauss-Markov/random-walk remain useful; random-direction/random-waypoint degrade | Converts weak regimes into explicit applicability boundaries. |
| Error profiles | B=10 moderate errors remain useful; B=15 Gauss-Markov remains high but random-walk is more sensitive | Supports bounded robustness, not sensing immunity. |
| Round11 five-seed paired campaign | B=10 proposed discovery 0.3639 vs enhanced no-ISAC 0.0006; B=15 proposed discovery 0.5445 vs 0.0034; all 5/5 paired seeds positive versus random, enhanced no-ISAC, candidate-set ablation, and one-slot delay for raw discovery | Strengthens seed-stability and mechanism evidence while preserving collision caveats. |
| Round11 collision-aware check | B=10 proposed collision-penalized discovery 0.2995 vs one-slot delay 0.2622; B=15 proposed 0.2042 vs one-slot delay 0.2207 | Shows collision-aware MAC control is still open, especially for denser wider-beam operation. |
| Round13 collision-aware MAC refinement | B=10 collision-aware collision-penalized discovery 0.3147 vs proposed 0.2991; B=15 0.2479 vs 0.2017; collision-penalized and discoveries-per-joule deltas are 10/10 positive versus proposed and one-slot delay at both beamwidths | Shows the collision boundary can be mitigated by local role control while preserving the same ISAC candidate-set interface. |

## What Not To Claim

- Do not claim full MAPPO/QMIX/GNN-MARL superiority. The current main result is shared-parameter protocol tuning.
- Do not claim strict SkyOrbs reproduction. The implemented baseline is a deterministic 3-D skip-scan reference inspired by SkyOrbs, and the supplement plus scope appendix state this boundary explicitly.
- Do not claim physical-layer ISAC waveform or estimator design. The paper uses a protocol-level occupancy-prior abstraction.
- Do not claim 3--30 deg are all solved. Write "evaluated over 3--30 deg"; 3--5 deg are stress cases.
- Do not claim instantaneous active communication graph connectivity under arbitrary mobility. The reported lambda2 is for the finite-horizon discovered-neighbor graph/cache.
- Do not claim the proposed protocol is a final collision-optimal MAC. Round13 mitigates the round11 boundary under assumed radio-state accounting, but calibrated energy-aware role scheduling remains future work.

## Likely Reviewer Questions and Preferred Answers

| Question | Short answer |
|---|---|
| Is ISAC acting as an oracle? | No. It only changes beam-selection probabilities; neighbor edges require bidirectional handshake. |
| Why is this not another beam-tracking paper? | Existing UAV-ISAC work mostly assumes a known target link for beamforming/tracking; this paper addresses unknown-neighbor discovery before alignment. |
| Is the topology-aware term really topology optimization? | It is a local discovered-degree/topology-deficit proxy, not a marginal lambda2 estimator. The paper uses lambda2 for evaluation and design motivation. |
| Are results statistically stable? | The main N=100/B=10 baseline table now uses ten paired seeds with 10/10 positive proposed-vs-control discovery deltas. Round11 separately shows 5/5 positive raw-discovery deltas against random, enhanced no-ISAC, candidate-set removal, and one-slot delay at B=10 and B=15; round13 adds ten-seed collision/energy refinement evidence. |
| Why not use complete SkyOrbs? | The current comparison is intentionally a SkyOrbs-like deterministic 3-D skip-scan reference under the same information boundary. Full reproduction requires matching original scan-path and rotation assumptions and remains future work; the supplement and standalone scope appendix now document this boundary. |
| What about delay tails? | P95/P99 are heavily censored by the finite 600-slot horizon in large narrow-beam tests, so the paper does not base main claims on uncensored tail-delay improvement. Discovery rate, empty scans, graph metrics, and paired deltas are the reliable primary metrics. |

## 11:00 Priority If More Work Is Available

1. Run one final independent manuscript audit after the latest edits; the current figure audit already passes with 47 LaTeX figure instances, 44 unique files, 0 missing files, and 0 non-4:3 violations.
2. Keep additional experiments narrow: radio-state accounting sensitivity under stated default powers, platform-calibrated power study, PHY-to-ISAC sensing-parameter mapping, or 10+ seed confirmation only, not a full Cartesian sweep.
3. Prepare the next-paper roadmap for real neural MARL: candidate-set-constrained action heads, graph/local-neighborhood encoders after discovery, and value-decomposition or actor-critic variants evaluated under the same transfer protocol.
