# One-Page Submission Pitch - 2026-07-05

## Positioning

This manuscript is a cross-layer link-layer protocol paper for UAV-UAV narrow-beam neighbor discovery.
It does not claim to design an ISAC waveform, solve physical-layer beamforming, or deliver a full neural MARL algorithm.
The core idea is to expose ISAC sensing as an imperfect beam-cell occupancy prior and use it to reduce the link-layer search space before bidirectional handshake confirmation.

## Main Claim

Within the evaluated single-hop, finite-horizon UAV swarm regimes, ISAC-assisted candidate-space reduction improves distributed narrow-beam neighbor discovery by reducing empty scans and improving the discovered-neighbor graph, while every discovered edge is still created only by a confirmed bidirectional narrow-beam handshake.

## Novelty Points

1. Distributed UAV-UAV narrow-beam ND with own pose/attitude known but no undiscovered-neighbor state, no central scheduler, and no global topology at execution.
2. Protocol-level ISAC abstraction with false alarms, missed detections, angular-cell offsets, staleness, and separate sensing/communication ranges.
3. ISAC-driven candidate-beam refinement with exploration floors and beam-lock memory, so sensing helps without becoming an oracle.
4. Local topology-deficit prioritization that targets finite-time discovered-graph quality rather than only raw pair discovery.
5. Small-scale trained/shared protocol tuning evaluated under N=100, 3--30 degree beamwidth, mobility, range, error, and ablation sweeps.
6. Structured MARL probe showing that the same ISAC candidate interface can feed a decentralized actor, while honestly leaving full MAPPO/QMIX/GNN-style MARL as future work.

## Evidence Chain

| Evidence | Result | Use |
|---|---:|---|
| Main N=100/B=10 density comparison | Proposed discovery 0.3655 vs enhanced no-ISAC 0.0007; lambda2 12.9222 vs 0 | Primary performance result. |
| Main N=100/B=15 density comparison | Proposed discovery 0.5440; lambda2 26.8413 | Shows B=10 is not a one-point artifact. |
| Candidate-set ablation | Discovery drops from 0.3655 to 0.0313 | Identifies ISAC candidate reduction as the mechanism. |
| One-slot delay ablation | Discovery 0.2989, lambda2 8.4709 | Bounds same-slot ISAC update sensitivity. |
| Round10 backup seeds | B=10 proposed 0.1739 vs enhanced no-ISAC 0.0008; B=15 proposed 0.4181 vs 0.0045 | Confirms ordering but flags scenario sensitivity. |
| Round11 five-seed paired campaign | B=10 proposed 0.3639 vs enhanced no-ISAC 0.0006; B=15 proposed 0.5445 vs 0.0034; 5/5 positive paired raw-discovery deltas versus all four controls | Strongest current seed-stability check for the main N=100/B=10/B=15 mechanism. |
| Round11 collision-aware boundary | B=10 proposed collision-penalized 0.2995 vs one-slot delay 0.2622; B=15 proposed 0.2042 vs one-slot delay 0.2207 | Shows collision-aware MAC optimization is not solved by raw discovery maximization. |
| Structured MARL probe | Best structured stochastic actor 0.5978; clean no-ISAC neural stochastic 0.0044 | Supports learning-interface feasibility, not main-method superiority. |

## Literature Anchors

- Directional ND and mmWave ND baseline context: Vasudevan et al., Chen et al., Wang et al., FastND, SkyOrbs.
- ISAC/side-information beam-management context: out-of-band spatial information, radar-assisted predictive beamforming, sensing-assisted NR-V2X/UAV beam management.
- UAV beam-mobility context: mmWave/THz drone beam learning under in-flight mobility uncertainty.
- Range abstraction context: ISAC system-level range analysis plus radar-system references; the paper deliberately keeps `Rc` and `Rs` as protocol support parameters.
- Learning context: wireless MARL/GNN resource allocation and directional ND learning, with full neural MARL explicitly scoped as future work.

## Reviewer Boundaries

- Do not claim strict SkyOrbs superiority; current baseline is SkyOrbs-inspired skip-scan under the same simulator information boundary.
- Do not claim 3--5 degree beams are solved; they are stress/failure-boundary cases.
- Do not claim calibrated physical sensing range laws; `Rs`, `P_fa`, `P_md`, and angular-cell errors are abstraction parameters.
- Do not claim full MARL superiority; structured actor-critic results are a probe and currently trail the flat stochastic student in raw discovery.
- Do not claim consensus convergence; lambda2 is a discovered-graph quality proxy.
- Do not claim collision-optimal scheduling; the collision-penalized results identify a follow-up MAC refinement problem.

## Best Next Experiment

The most defensible next experiment is not a broad Cartesian sweep.
Run a focused 5--10 seed campaign at N=100/B=10 and B=15 for the main proposed, enhanced no-ISAC, one-slot-delay, and candidate-ablation protocols, using paired scenario seeds and reporting median/IQR/bootstrap CI plus collision-penalized discovery.
