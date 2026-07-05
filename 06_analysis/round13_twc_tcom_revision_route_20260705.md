# Round13 TWC/TCOM Revision Route

Date: 2026-07-05

This note turns the round13 ten-seed collision/energy evidence into a paper-writing route.
It is scoped for a cross-layer ISAC-assisted UAV-UAV neighbor-discovery paper, not a pure PHY beamforming, pure MAC scheduling, or full neural-MARL paper.

## Main Story

The paper should be framed around one central mechanism:

> ISAC exposes an imperfect beam-cell occupancy prior to the link layer, allowing a fully distributed UAV-UAV narrow-beam neighbor-discovery protocol to suppress empty beam cells and concentrate finite discovery opportunities on plausible neighbor directions.

The main contribution remains the link-layer candidate-beam refinement protocol and its scalable shared-policy tuning.
Round13 should be used as reviewer-facing evidence that collision and assumed radio-state accounting have been considered, not as a new primary method that changes the paper into a MAC optimization paper.

## What Can Go in the Main Paper

1. System model:
   ISAC is an imperfect occupancy-prior service with false alarms, missed detections, angular-cell offsets, sensing range, and staleness.
   It does not reveal neighbor identity, trajectory, or range-resolved state.

2. Protocol mechanism:
   The key mechanism is candidate-set refinement before confirmed handshake.
   The handshake still creates the edge; sensing only changes beam-cell priority.

3. Main N=100 baseline comparison:
   At N=100 and 10-degree beams, the ISAC-assisted policy achieves discovery 0.3655 and lambda2 12.9222, while implemented communication-only baselines remain near zero.
   Keep the SkyOrbs baseline as "SkyOrbs-like".

4. Transfer:
   Training at N=10 and 10 degrees transfers to N=100 in the tested 10--30 degree single-hop finite-horizon regimes.
   Write "evaluated over 3--30 degrees" only when the supplement/stress cases are included; do not write "effective over 3--30 degrees".

5. Collision discussion:
   The main text can briefly say that raw discovery/connectivity and MAC efficiency are different objectives.
   The supplementary round13 ten-seed probe shows that local role control improves collision-penalized discovery at B=10 and B=15 without changing the ISAC candidate-set interface.

## What Should Stay in the Supplement

| Evidence | Reason to keep in supplement |
|---|---|
| Round7 N=10--100 and 3--30 degree heatmaps | Broad stress coverage; not the cleanest main evidence chain. |
| Round9 3-degree full baselines | Strong failure-boundary evidence, but it weakens the main narrative if over-emphasized. |
| Round10 extra-seed backup | Useful trajectory/stability backup; not a replacement for the main table. |
| Round11 five-seed paired raw-discovery campaign | Focused stability evidence; keep because it also exposes the B=15 collision boundary. |
| Round13 collision-aware MAC/energy probe | Important reviewer response, but still a mechanism-refinement probe under assumed radio-state powers. |
| Structured MARL probe | Shows feasibility of candidate-constrained neural policies, but it is not yet stronger than the rule-driven main method. |

## Round13 Facts to Use

All values below use N=100, Gauss-Markov mobility, 600 slots, density scaling, single-hop range, ten paired seeds, and training at N=10/B=10.

| Beamwidth | Metric | Proposed low-latency ISAC | Collision-aware ISAC | Paired sign result |
|---:|---|---:|---:|---|
| 10 deg | Discovery rate | 0.3652 | 0.3660 | raw delta is mixed, 5/10 positive |
| 10 deg | Collision-penalized discovery | 0.2991 | 0.3147 | 10/10 positive |
| 10 deg | Discoveries per joule | 6.1932 | 6.5417 | 10/10 positive |
| 10 deg | Energy per discovered edge | 0.1616 J | 0.1530 J | 10/10 lower is better |
| 15 deg | Discovery rate | 0.5421 | 0.5647 | 10/10 positive |
| 15 deg | Collision-penalized discovery | 0.2017 | 0.2479 | 10/10 positive |
| 15 deg | Discoveries per joule | 9.2045 | 10.1564 | 10/10 positive |
| 15 deg | Energy per discovered edge | 0.1087 J | 0.0985 J | 10/10 lower is better |

Important wording boundary:

> Round13 supports that local collision-aware role control mitigates the collision-penalized boundary under the stated ten-seed setting and assumed radio-state powers.

Do not make an energy-optimality claim.

Do not write:

> Collision-aware ISAC is uniformly better in raw discovery at B=10.

## Suggested Contribution Wording

Use this contribution shape in the next manuscript pass:

1. We formulate distributed-execution UAV-UAV narrow-beam neighbor discovery with self-localized UAV states and no undiscovered-neighbor state, and abstract ISAC as an imperfect beam-cell occupancy prior rather than a physical-layer oracle.
2. We design an ISAC-assisted link-layer discovery protocol that suppresses sensed-empty beam cells, reinforces candidate occupied cells, and preserves bidirectional handshake as the only edge-confirmation event.
3. We use shared-parameter protocol tuning at small scale and evaluate zero-shot transfer to N=100 over beamwidth, mobility, range, sensing-error, and area-scaling regimes.
4. We expose collision and radio-activity boundaries through collision-penalized and assumed radio-state accounting metrics, including a ten-seed local role-control refinement probe.

## Next Main-Text Edits

1. Abstract:
   Add one phrase that the protocol is "collision- and energy-accounting aware in evaluation", not "energy optimized".

2. Contributions:
   Make the third contribution about scalable shared-policy protocol tuning and bounded zero-shot transfer.
   Make the fourth contribution about evaluation breadth and explicit boundary reporting.

3. Results:
   Keep round13 to one compact paragraph after the ablation discussion or in limitations.
   Refer to supplement for values.

4. Limitations:
   Preserve the clear boundary that complete collision- and platform-calibrated energy-aware MAC design is future work.

5. Supplement:
   Keep round13 figures as the main reviewer-facing answer to "do collisions invalidate the result?"

## One-Paragraph English Rewrite Candidate

This paper studies distributed UAV-UAV neighbor discovery under narrow 3-D beams when no undiscovered-neighbor state is available before alignment. Instead of treating ISAC as a physical-layer beamforming solution, we expose it to the link layer as an imperfect beam-cell occupancy prior with false alarms, missed detections, angular errors, range limits, and staleness. The resulting protocol uses sensing feedback to suppress empty beam cells and prioritize candidate occupied cells while retaining bidirectional handshake as the only link-confirmation mechanism. Experiments show that a policy tuned at N=10 transfers to N=100 in the useful 10--30 degree regimes, strongly outperforming communication-only baselines, while supplementary stress tests identify the 3--5 degree, abrupt-mobility, collision-heavy, and uncalibrated-energy boundaries.
