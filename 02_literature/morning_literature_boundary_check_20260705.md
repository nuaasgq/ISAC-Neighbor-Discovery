# Literature Boundary Check: ISAC-Assisted UAV Neighbor Discovery

Date: 2026-07-05

Purpose: capture the literature-positioning checks used during the morning experiment-to-paper pass.

## Verified Positioning

The current manuscript should position itself as a cross-layer, data-link-layer neighbor-discovery protocol paper, not as a physical-layer beam-tracking or waveform-design paper.

## Closest Neighbor-Discovery References

- SkyOrbs: Fast 3-D directional neighbor discovery for UAV networks.
  - Source checked: IEEE Xplore, `https://ieeexplore.ieee.org/document/10659183/`
  - Boundary: use `SkyOrbs-like` for our baseline. The current simulator implements a deterministic 3-D skip-scan baseline inspired by SkyOrbs, not a strict reproduction of the full SkyOrbs protocol.
- Joint delay-power neighbor discovery in UAV networks.
  - Source checked: IEEE Computer Society page, `https://www.computer.org/csdl/journal/tm/2026/06/11320813/2cTQxGWicIo`
  - Boundary: this supports discussing delay, collision, and radio activity jointly. Our current paper reports scan-action and collision-normalized efficiency, not Joule-level delay-power optimization.
- Hunting-based directional neighbor discovery in mmWave networks.
  - Source checked: IEEE Xplore search page, `https://ieeexplore.ieee.org/document/7980107/`; DOI and page metadata cross-checked through public metadata/search snippets.
  - Citation used in the draft: Wang, Mao, and Rappaport, IEEE ICDCS 2017, pp. 1704--1713, DOI `10.1109/ICDCS.2017.229`.
  - Boundary: useful as an older deterministic/protocol reference for directional rendezvous under deafness and no prior information; not a UAV-UAV ISAC baseline.

## ISAC / Beam-Management References

- ISAC survey and cross-layer motivation.
  - Source checked: IEEE/ACM DOI page, `https://dl.acm.org/doi/10.1109/JSAC.2022.3156632`
  - Citation used in the draft: Liu et al., IEEE JSAC 2022, DOI `10.1109/JSAC.2022.3156632`.
  - Boundary: supports treating sensing-assisted communications as a cross-layer opportunity; it does not validate the paper's specific neighbor-discovery algorithm.
- ISAC-assisted predictive beam tracking in multipath channels.
  - Source checked: TechRxiv PDF page, `https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.22818182`
  - Boundary: relevant to sensing-aided beam prediction under imperfect observations, but it is not UAV-UAV neighbor discovery and should not be framed as a direct protocol baseline.
- Recent predictive beam tracking for sea-air UAV/USV links.
  - Source checked: arXiv HTML, `https://arxiv.org/html/2606.26569v1`
  - Boundary: confirms that ISAC beam tracking remains an active physical-layer/beam-management theme, reinforcing the need to clarify that our contribution is link-layer discovery.
- Sensing-assisted or CKM/LoS-aided predictive beamforming publications listed by ISAC research groups.
  - Source checked: Fan Liu ISAC publication page, `https://sites.google.com/view/dr-fan-liu/publications`
  - Boundary: use only as background for ISAC beam-management motivation unless a specific cited paper is individually verified.

## Graph and Range-Abstraction References

- Algebraic connectivity.
  - Source checked: EuDML metadata page, `https://eudml.org/doc/12723`
  - Citation used in the draft: Fiedler, Czechoslovak Mathematical Journal 1973.
  - Boundary: supports the definition of algebraic connectivity; it does not itself prove swarm consensus performance in this paper.
- Consensus in networked agents.
  - Source checked: IEEE Xplore page, `https://ieeexplore.ieee.org/document/1333204`
  - Citation used in the draft: Olfati-Saber and Murray, IEEE TAC 2004.
  - Boundary: supports the relevance of graph connectivity to consensus; our simulations only use `lambda2` as a topology proxy and do not close the control loop.
- Radar range equation.
  - Source checked: MIT Lincoln Laboratory radar equation notes, `https://www.ll.mit.edu/media/6946`, which also points to Skolnik's textbook.
  - Citation used in the draft: Skolnik, Introduction to Radar Systems, 3rd ed., McGraw-Hill, 2001.
  - Boundary: supports the statement that sensing range depends on target, waveform, aperture, distance, and medium variables; our `Rs/Rc` sweep remains a protocol-level abstraction, not a calibrated radar-link-budget model.

## Current Manuscript Wording Rules

- Write `evaluated over 3--30 degrees`, not `effective over 3--30 degrees`.
- Write `SkyOrbs-like deterministic 3-D skip-scan baseline`, not `SkyOrbs reproduction`.
- Write `ISAC beam-cell occupancy prior`, not `ISAC oracle` or `neighbor position estimator`.
- Write `shared-parameter policy optimization`, not full MAPPO/QMIX/GNN-MARL.
- Write `scan-action and collision-normalized efficiency`, not Joule-level energy efficiency.
- Treat 3/5-degree beams, random-direction mobility, and random-waypoint mobility as stress regimes.
