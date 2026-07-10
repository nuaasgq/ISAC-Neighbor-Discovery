# Communication PHY Sensitivity Calibration (2026-07-10)

## Scope

This calibration fixes a defensible nominal communication-PHY operating point and quantifies sensitivity to path-loss exponent, Rician K-factor, log-normal shadowing, sidelobe gain, SINR threshold, and transmit power. It does not fit parameters to measured FANET channel traces.

The ranges are intentionally broader than the nominal LoS FANET setting. The 3GPP TR 38.901 framework covers 0.5-100 GHz, reports scenario-dependent log-normal shadowing values, and includes Rician reference profiles with K-factors of 13.3 dB and 22 dB. It is used as a range reference, not as a claim that this simulator implements a complete 3GPP scenario model.

- 3GPP TR 38.901 record: <https://www.3gpp.org/DynaReport/38.901.htm>
- 3GPP TR 38.901 PDF: <https://www.3gpp.org/ftp/Meetings_3GPP_SYNC/SA3/Inbox/Drafts/tr_138901v140200p.pdf>
- 3GPP TR 36.777 aerial study: <https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3231>
- ITU-R SM.575 thermal-noise reference: <https://www.itu.int/dms_pubrec/itu-r/rec/sm/R-REC-SM.575-2-201310-S!!PDF-E.pdf>

## Reproducible Design

- Script: `05_simulation/run_communication_phy_calibration.py`
- Base configuration: `05_simulation/configs/twc_trainable_n10.yaml`
- Seed: `20260710`
- Samples: 20,000 per link condition
- Statistics: common random numbers within each sweep and Wilson 95% confidence intervals for probabilities
- Distances: 1, 5, 10, 15, and 17.32 km
- Interference cases: equal-power 10 km/10 km; near-far 3 km/10 km; one-sided off-beam interferer at 2 km

The calibration targets are 0.80-0.98 isolated edge coverage, at most 0.20 equal-power decoding, and at least 0.75 near-far capture. The upper edge-coverage target prevents choosing a trivially overpowered operating point that would make PHY sensitivity invisible.

## Nominal Point

| Parameter | Nominal value |
|---|---:|
| Path-loss exponent | 2.1 |
| Rician K-factor | 10 dB |
| Shadowing standard deviation | 2 dB |
| Sidelobe gain | -10 dBi |
| SINR threshold | 5 dB |
| Transmit power | 1 W |

| Link-level metric | Estimate | Wilson 95% CI |
|---|---:|---:|
| Isolated edge coverage | 0.9448 | 0.9415-0.9478 |
| Isolated 10 km coverage | 0.9984 | 0.9977-0.9989 |
| Equal-power decode | 0.1837 | 0.1784-0.1891 |
| Near-far capture | 0.9317 | 0.9281-0.9351 |
| One-sided interference survival | 0.9854 | 0.9836-0.9869 |

The nominal edge SNR has mean 9.70 dB, 10th percentile 6.05 dB, and median 9.81 dB. The 5 dB threshold therefore retains a finite but not negligible edge outage probability.

## Sensitivity Findings

1. **Path loss is the dominant range uncertainty.** Edge coverage changes from 0.9978 at `n=2.0` to 0.9476 at `n=2.1`, 0.5797 at `n=2.2`, and zero at `n>=2.5` under the fixed 1 W budget. Results must therefore report `n`, not only transmit power and range.
2. **K-factor changes both coverage and capture.** Edge coverage rises from 0.7605 at 0 dB to 0.9457 at 10 dB and 0.9935 at 22 dB. A single high-K LoS assumption would materially overstate robustness.
3. **Shadowing is a major stress variable.** Increasing its standard deviation from 2 dB to 6 dB reduces edge coverage from 0.9465 to 0.7703 and near-far capture from 0.9312 to 0.7394.
4. **Sidelobes control spatial interference rejection.** Changing sidelobe gain from -10 dBi to -5 dBi lowers one-sided interference survival from 0.9869 to 0.9103; 0 dBi lowers it to 0.5864. Isolated coverage is unchanged, as expected.
5. **The SINR threshold exposes a coverage/collision tradeoff.** A 5 dB threshold gives edge coverage 0.9498 and equal-power decode 0.1779. At 10 dB these become 0.4719 and 0.0086, but near-far capture also falls to 0.5843.
6. **Power repairs noise-limited coverage, not equal-power interference.** Raising power from 0.1 W to 2 W increases edge coverage from 0.0248 to 0.9928, while equal-power decode only changes from 0.0533 to 0.1907 because desired and interfering powers scale together.

## Recommended Experiment Profiles

| Profile | n | Threshold | Power | Intended use |
|---|---:|---:|---:|---|
| nominal | 2.1 | 5 dB | 1 W | Main training and evaluation |
| balanced_stress | 2.2 | 5 dB | 2 W | Higher attenuation with viable coverage |
| low_power | 2.0 | 5 dB | 0.25 W | Energy-constrained stress |
| high_selectivity | 2.0 | 7.5 dB | 0.5 W | Strong collision rejection |
| highest_feasible_coverage | 2.0 | 5 dB | 0.5 W | Upper-coverage comparison |

## Protocol-Level Check

Five common scenario seeds were run for each profile with `improved_rl_isac_tables` for 300 slots. Only three aligned handshake attempts occurred across each five-episode set, and all three succeeded for every profile. This result is retained as an audit artifact, but it is statistically non-discriminating and must not be used to claim protocol-level insensitivity. Beam alignment and MAC action sparsity, rather than PHY decoding, dominated these short episodes.

The link-level calibration is therefore the supported result. A future paper-level protocol sensitivity experiment should use either many more independent episodes or an explicitly conditioned set of aligned handshake opportunities while preserving identical MAC actions across PHY profiles.

## Artifacts

- Tables: `06_analysis/paper_tables/communication_phy_calibration_20260710/`
- Figures: `06_analysis/paper_figures/communication_phy_calibration_20260710/`
- Raw reproducibility output: `05_simulation/results_raw/communication_phy_calibration_20260710/`

The figure directory contains eight 4:3 Times New Roman plots: six one-at-a-time sensitivity plots, one coverage-distance plot, and one joint path-loss/SINR heatmap.

## Limitations

- Parameters are calibrated to internal link-budget and interference criteria, not flight measurements.
- The close-in model is generic and is not a complete 3GPP UMa/UMi/RMa implementation.
- Fading is reciprocal within a slot but lacks Doppler correlation, blockage states, and packet-length/MCS-dependent BLER.
- The sectored pattern does not represent a measured phased-array pattern.
- Bandwidth, noise figure, carrier frequency, and antenna efficiency remain fixed in this sweep.
