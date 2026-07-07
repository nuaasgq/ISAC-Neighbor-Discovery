# Range Abstraction Theory Note - 2026-07-07

## Material Passport

- Origin Skill: academic-paper / citation-compliance and argument-boundary pass
- Verification Status: ANALYZED
- Scope: protocol-level interpretation of communication range `Rc`, sensing range `Rs`, and the matched-support final setting

## Claim Boundary

The manuscript does not assume that communication range and ISAC sensing range are physically identical. It models them as separate protocol support parameters:

- `Rc` is the link-confirmation support for bidirectional narrow-beam packet decoding.
- `Rs` is the support of coarse beam-cell occupancy evidence delivered by an underlying sensing receiver.
- A discovered edge still requires communication-range eligibility and bidirectional handshake alignment; an ISAC observation alone never creates a neighbor edge.

The final Phase10 long-horizon transfer tables use `Rc = Rs = 900 m` under a single-hop matched-support setting. This is a controlled operating point for evaluating neighbor-discovery behavior, not a platform-calibrated statement that communication and sensing links must have the same physical range.

## Source Rationale

- `ThreeGPP22137ISAC` (https://www.3gpp.org/dynareport/22137.htm): 3GPP treats ISAC as a service that exposes sensing results and service-quality dimensions such as detection, accuracy, resolution, latency, and refreshing behavior. This supports exposing a sensing observation to the link layer instead of treating sensing as a decoded-neighbor edge.
- `ThreeGPP38901Channel` (https://www.3gpp.org/dynareport/38901.htm): communication feasibility is tied to channel, blockage, path loss, and link-budget assumptions for the communication waveform and receiver.
- `Liu2022ISAC6G` (https://doi.org/10.1109/JSAC.2022.3156632): ISAC integrates sensing and communication functions but still has distinct sensing and communication performance objectives.
- `Skolnik2001Radar`: radar-like detection depends on target scattering, propagation, aperture, integration, clutter, and detection threshold, which are not the same variables as packet-decoding success.
- `Bomfin2024SystemISAC` (https://doi.org/10.1109/WCNC57260.2024.10571030): system-level ISAC analysis explicitly separates communication and radar/sensing SNR behavior; the relative communication/sensing operating ranges can vary with carrier frequency and system assumptions.

## Evidence in the Current Artifact Set

- Main manuscript: `07_paper/ieee_twc_isac_nd/main.tex` explicitly states that `Rc` and `Rs` are separate abstraction parameters and that the final `Rs=Rc` setting is a matched-support operating point rather than a hardware law.
- Supplement: `07_paper/ieee_twc_isac_nd/supplement.tex` states that `Rc=Rs` is a matched-support single-hop setting and reports the range sweep.
- Range-grid evidence:
  - `06_analysis/paper_tables/round3_robustness/range_rc_rs_grid/aggregate_metrics.csv`
  - `06_analysis/paper_tables/round3_robustness/range_rs_ratio/aggregate_metrics.csv`
- Tested range support:
  - `Rc/D` values: 0.65, 0.85, 1.05.
  - `Rs/Rc` values: 0.5, 0.75, 1.0, 1.25.

## Audit Decision

This requirement is suitable for `PASS` only under the following conservative reading: the paper has explicit communication/sensing range assumptions, cites why the two supports are physically distinct, sweeps `Rc/D` and `Rs/Rc`, and avoids claiming that `Rs=Rc` is hardware-calibrated. It does not prove a universal PHY range law.
