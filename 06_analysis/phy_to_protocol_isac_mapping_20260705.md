# PHY-to-Protocol ISAC Mapping Note

Date: 2026-07-05

Purpose: record the cite-backed interpretation of the protocol-level ISAC abstraction used in the manuscript.
This is a text-only mitigation of the PHY-mapping risk, not a calibrated waveform or detector design.

## Source Anchors

| Source | Role in the manuscript |
|---|---|
| 3GPP TS 22.137, Release 19, Integrated Sensing and Communication ([ETSI PDF](https://www.etsi.org/deliver/etsi_ts/122100_122199/122137/19.01.00_60/ts_122137v190100p.pdf)) | Supports treating ISAC as a sensing service with sensing results and KPI dimensions such as detection reliability, resolution, latency, and refreshing rate. |
| 3GPP TR 38.901, channel model for 0.5--100 GHz ([ETSI PDF](https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/16.01.00_60/tr_138901v160100p.pdf)) | Supports treating communication range as a link-budget/channel feasibility quantity rather than a sensing-detection range. |
| Liu et al., "Integrated Sensing and Communications: Toward Dual-Functional Wireless Networks for 6G and Beyond," IEEE JSAC, 2022, DOI 10.1109/JSAC.2022.3156632 | Supports the general ISAC/perceptive-network framing and sensing-assisted communication view. |
| Skolnik, Introduction to Radar Systems, 3rd ed., 2001 | Supports the radar-range/detection-threshold view behind sensing support, false alarms, and missed detections. |
| Cui et al., "Seeing Is Not Always Believing," IEEE WCL, 2024, DOI 10.1109/LWC.2023.3303949 | Supports the boundary that sensing-derived angle information can be misleading under multipath and should be used as a prior rather than an oracle. |
| Cui et al., "Sensing-Assisted Accurate and Fast Beam Management for Cellular-Connected mmWave UAV Network," China Communications, 2024, DOI 10.23919/JCC.ea.2023-0140.202401 | Supports sensing-assisted UAV beam-management context, while our work differs by moving the emphasis to UAV-UAV neighbor discovery. |

## Mapping Used in the Current Draft

| Protocol parameter | PHY/service interpretation |
|---|---|
| `Rc` | Communication-link support range: bidirectional narrow-beam handshake decoding is feasible under the assumed link-budget, channel, blockage, interference, and receiver-sensitivity conditions. |
| `Rs` | Sensing-observation support range: the sensing receiver can produce useful occupancy evidence under a selected detector operating point and integration/scanning budget. |
| `P_fa` | Detector false-alarm operating point: an empty beam cell can be marked occupied because of clutter, sidelobes, noise, or estimator artifacts. |
| `P_md` | Detector missed-detection operating point: an occupied cell can be missed because of weak return, occlusion, multipath, angular mismatch, or detection thresholding. |
| `sigma_b` | Quantized angular-cell error: AoA/angle-estimation error normalized by the beam-codebook cell width. |
| Staleness / slot age | Sensing-to-use latency relative to UAV mobility and beam footprint; stale evidence is decayed before it can dominate future scans. |

## Manuscript Boundary

The draft should keep saying that sensing does not create a neighbor edge.
ISAC provides a noisy occupancy prior, and only the bidirectional narrow-beam handshake confirms a communication neighbor.

The draft should not claim:

- `Rs = Rc` by physics.
- `Rs > Rc` stops helping as a physical sensing law.
- The configured `P_fa`, `P_md`, or `sigma_b` values are calibrated to a named waveform, CFAR detector, aperture, RCS distribution, or receiver implementation.

The current text-only improvement adds:

- a main-text statement that the abstraction follows 3GPP-style sensing-service/KPI thinking;
- a main-text statement that `Rc` and `Rs` are separated because communication decoding and sensing detection have different success criteria;
- a supplement table that maps protocol parameters to PHY/service interpretations.

## Remaining Upgrade

For a stronger external submission, the next step would be a small analytical appendix:

1. Choose a carrier frequency, bandwidth, aperture, transmit power, noise figure, and target RCS range.
2. Derive an illustrative sensing SNR and detector ROC point for `P_fa/P_md`.
3. Derive a communication link budget for `Rc`.
4. Show that the simulation grid spans plausible `Rs/Rc` and detector-quality regimes.

This would be a calibration appendix, not a change to the protocol contribution.
