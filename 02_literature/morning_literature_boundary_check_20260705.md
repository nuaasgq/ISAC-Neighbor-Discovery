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
- Hunting-based directional neighbor discovery in mmWave ad hoc networks.
  - Source checked: public PDF link returned by search, `https://www.eng.auburn.edu/~szm0001/papers/YuWang_TWC17.pdf`
  - Boundary: useful as an older deterministic/protocol reference for directional rendezvous under deafness and no prior information.

## ISAC / Beam-Management References

- ISAC-assisted predictive beam tracking in multipath channels.
  - Source checked: TechRxiv PDF page, `https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.22818182`
  - Boundary: relevant to sensing-aided beam prediction under imperfect observations, but it is not UAV-UAV neighbor discovery and should not be framed as a direct protocol baseline.
- Recent predictive beam tracking for sea-air UAV/USV links.
  - Source checked: arXiv HTML, `https://arxiv.org/html/2606.26569v1`
  - Boundary: confirms that ISAC beam tracking remains an active physical-layer/beam-management theme, reinforcing the need to clarify that our contribution is link-layer discovery.
- Sensing-assisted or CKM/LoS-aided predictive beamforming publications listed by ISAC research groups.
  - Source checked: Fan Liu ISAC publication page, `https://sites.google.com/view/dr-fan-liu/publications`
  - Boundary: use only as background for ISAC beam-management motivation unless a specific cited paper is individually verified.

## Current Manuscript Wording Rules

- Write `evaluated over 3--30 degrees`, not `effective over 3--30 degrees`.
- Write `SkyOrbs-like deterministic 3-D skip-scan baseline`, not `SkyOrbs reproduction`.
- Write `ISAC beam-cell occupancy prior`, not `ISAC oracle` or `neighbor position estimator`.
- Write `shared-parameter policy optimization`, not full MAPPO/QMIX/GNN-MARL.
- Write `scan-action and collision-normalized efficiency`, not Joule-level energy efficiency.
- Treat 3/5-degree beams, random-direction mobility, and random-waypoint mobility as stress regimes.
