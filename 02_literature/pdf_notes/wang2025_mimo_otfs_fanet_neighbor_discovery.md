# Wang et al. 2025: MIMO-OTFS ISAC Waveform for FANET Rapid Neighbor Discovery

## Metadata

- Authors: Weidong Wang, Jian Yang, Jiadong Shang, Hui Gao, Qiannan Zhang, Shuo Zhang
- Title: A Rapid Neighbor Discovery Method for Directional Flying Ad-Hoc Networks Based on MIMO-OTFS Integrated Sensing and Communication Waveform
- Venue: Modern Defence Technology, 2025, 53(2): 107-120
- DOI: 10.3969/j.issn.1009-086x.2025.02.012
- Local PDF: user-provided, not committed to repository

## WHY

Directional FANET initialization is slowed by frequent topology changes, link-quality fluctuation, narrow-beam deafness, hidden terminals, and many ineffective handshakes. The paper argues that ISAC can reduce blind scanning by sensing which beam sectors contain potential neighbors, while OTFS improves robustness under high-Doppler airborne channels.

## HOW

The paper combines a physical-layer MIMO-OTFS ISAC multi-target detector with an upper-layer neighbor discovery protocol:

- Nodes use multi-RF UPA beamforming and transmit the same Hello packet through multiple beams.
- Each node maintains a sensing table and a neighbor table.
- A beam flag records whether a sector may still contain undiscovered targets.
- MIMO-OTFS echoes are processed with likelihood detection, false-alarm thresholding, and SIC to estimate delay, Doppler, azimuth, and elevation.
- Sensed delay/angle parameters are mapped to target positions and beam-sector occupancy.
- Successful handshakes exchange both neighbor tables and sensing tables, enabling indirect communication discovery and cooperative sensing.

## WHAT

The paper reports that MIMO-OTFS has better angle-domain sensing performance than MIMO-OFDM in high-mobility settings, that sensing success probability exceeds about 90% near -5 dB SNR in its tested setup, and that cooperative ISAC table exchange reduces initial networking time compared with non-cooperative and communication-only cooperative discovery.

## Main Assumptions

- 3D FANET, all nodes are uniformly deployed in a 10 km x 10 km x 10 km region.
- Node count is up to 50 in the reported protocol simulations.
- All node pairs are within communication range.
- Beamwidth is fixed at 25 degrees in the main simulation table.
- Each node knows its own GPS/location and the total network node count.
- Initial networking is assumed short enough that relative motion can be ignored.
- LoS echo path is assumed for sensing.
- Time synchronization is assumed before discovery.
- Nodes use full-duplex ISAC reception while transmitting.

## Relevance to This Project

This is the closest local-language reference found so far for ISAC-assisted FANET neighbor discovery. It validates several directions we have been using:

- ISAC can be abstracted as a beam-cell occupancy prior rather than only as beam alignment.
- Empty-beam elimination is a valid protocol-level mechanism.
- A sensing table with flags, target counts, SNR/confidence, and beam IDs is a concrete physical-to-link-layer abstraction.
- Cooperative exchange of sensing information is a meaningful protocol primitive, not just a physical-layer detail.
- MIMO-OTFS can motivate a high-Doppler sensing model if reviewers ask why the sensing capability is plausible.

## Key Differences From Our Target Paper

- Their protocol is rule-based randomized scanning with table exchange; ours should be MARL-plus-rules with learned, topology-aware, uncertainty-aware decisions.
- They optimize networking time; our stronger claim is finite-time topology quality and scalable discovery under a limited slot budget.
- They stop at 50 nodes; our target evidence emphasizes N=100 transfer and small-to-large scalability.
- They use one main beamwidth setting around 25 degrees; our study covers narrow-beam sensitivity such as 3-15 degrees in the final evidence and historical 30-degree boundary checks.
- They assume motion can be ignored during initial discovery; our simulations include dynamic mobility, while keeping the slot duration short enough for ISAC freshness.
- They assume all pairs are within communication range and know total node count; we can initially keep single-hop for comparability, but should state this as a first-step scope and later test range/density.
- They do not include MARL algorithm-family comparison, learned network structures, or scale-invariant transfer.
- They do not prioritize algebraic connectivity, critical links, largest connected component, or consensus-relevant topology quality.

## Design Inspiration For Our Method

1. Use a sensing-table abstraction in the system model:
   - beam_id
   - occupancy flag
   - estimated neighbor count
   - confidence/SNR
   - age or freshness
   - discovered-count residual

2. Make ISAC imperfect:
   - miss detection can incorrectly mark an occupied beam as empty
   - false alarms can keep an empty beam active
   - confidence should decay with time and mobility
   - multipath/angle mismatch should be mentioned as a limitation and robustness dimension

3. Separate three protocol information sources:
   - self-state: own position, attitude, beam direction, local table
   - ISAC observation: local beam-cell occupancy evidence
   - handshake observation: confirmed identity and neighbor table exchange

4. Turn rule-based table updates into MARL state tokens:
   - per-beam occupancy/confidence token
   - per-beam residual-undiscovered estimate
   - local topology token from discovered neighbors
   - contention/collision token from recent slots

5. Add topology-aware decision logic beyond this paper:
   - prioritize beams likely to reveal bridging or low-degree neighbors
   - balance raw discovery against collision and redundant discovery
   - reward lambda2 proxy, largest connected component, and isolated-node reduction

## How To Use It In Our Paper

Recommended positioning:

> Wang et al. mapped MIMO-OTFS ISAC sensing into a FANET neighbor discovery protocol by maintaining sensing and neighbor tables and exchanging them after successful handshakes. However, their method remains rule-based, assumes a short quasi-static initialization interval, and mainly optimizes discovery delay. Our work studies scalable distributed learning over the ISAC-derived beam-cell prior and explicitly targets finite-time topology quality under narrow-beam UAV-UAV discovery.

This paper can serve as:

- A close related-work reference in the ISAC-assisted FANET ND paragraph.
- A motivation source for the sensing-table abstraction.
- A comparison point for "rule-based ISAC-assisted discovery" if we implement a simplified baseline.
- A justification for MIMO-OTFS-inspired sensing probability/SNR assumptions without turning our paper into a physical-layer waveform paper.

## Caution

Do not overclaim that our problem is untouched. This paper already combines FANET, directional neighbor discovery, MIMO-OTFS ISAC, and cooperative table exchange. The novelty must be framed around scalable MARL, topology-aware finite-time discovery, imperfect/aged ISAC priors, and broader evaluation rather than merely "ISAC-assisted FANET neighbor discovery."
