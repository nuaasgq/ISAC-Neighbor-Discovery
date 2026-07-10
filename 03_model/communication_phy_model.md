# Communication PHY Model for Directional UAV Neighbor Discovery

## Purpose

The communication PHY adapter determines whether a directional HELLO/ACK exchange is decodable. It is separate from beam selection and MAC decision logic so that all random, rule-based, Wang-aligned, and MARL methods use the same channel realization and receiver model.

The implementation is in `05_simulation/src/isac_nd_sim/communication_phy.py`. Historical configurations use `model: ideal`; the TWC training configuration explicitly uses `model: close_in_rician_sinr`.

## Large-Scale Path Loss

For carrier frequency `f_c`, reference distance `d_0`, path-loss exponent `n`, and distance `d >= d_0`, the close-in model is

```text
PL(d) = FSPL(d_0) + 10 n log10(d / d_0) + X_sigma + L_sys,
FSPL(d_0) = 20 log10(4 pi f_c d_0 / c).
```

`X_sigma` is one reciprocal, episode-level log-normal shadowing realization per UAV pair. The first single-hop study uses `n=2`, consistent with an unobstructed LoS reference, and must later sweep or calibrate this value. This generic close-in model is not presented as a verbatim 3GPP UMi/UMa formula.

3GPP TR 38.901 supplies the broader 0.5-100 GHz channel-modeling framework, while 3GPP TR 36.777 motivates explicit aerial-link treatment. Relevant official records are:

- 3GPP TR 38.901: <https://www.3gpp.org/DynaReport/38.901.htm>
- 3GPP TR 36.777: <https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3231>

## Directional Antenna Gain

One codebook cell is represented by an ideal sectored beam. With azimuth width `Delta_phi`, elevation width `Delta_theta`, and aperture efficiency `eta`,

```text
Omega_B = Delta_phi * 2 sin(Delta_theta / 2),
G_main = eta * 4 pi / Omega_B.
```

If the selected beam contains the target direction, `G_main` is applied; otherwise the configurable sidelobe gain is applied. Both transmitter and receiver gains enter every desired and interfering link. The model therefore captures beamwidth-dependent array gain and directional interference suppression without assuming that non-selected candidate beams are active.

## Small-Scale Fading and Noise

Rician fading uses

```text
h = sqrt(K/(K+1)) + sqrt(1/(K+1)) g,
g ~ CN(0,1),
```

so `E[|h|^2] = 1`. The pairwise power gain is reciprocal within a slot and independently resampled in the next slot. Shadowing and fading use a channel RNG derived only from `scenario_seed`, independent of protocol/policy seeds, mobility, and sensing RNGs. Paired protocol evaluations therefore share the same communication-channel realization in every slot.

Receiver noise is

```text
N = k T B F,
```

where `T=290 K`, `B` is communication bandwidth, and `F` is receiver noise factor. The equivalent room-temperature density is approximately -174 dBm/Hz, consistent with ITU-R references such as <https://www.itu.int/dms_pubrec/itu-r/rec/sm/R-REC-SM.575-2-201310-S!!PDF-E.pdf>.

## Aggregate Interference and SINR

For desired transmitter `i` and receiver `j`,

```text
P_ij = P_t G_t(i,j) G_r(j,i) |h_ij|^2 / L_ij,
SINR_ij = P_ij / (N + sum_{k != i} P_kj).
```

The interference sum includes every simultaneous emitter in the sub-slot after its actual transmit and receive directional gains. A receiver can decode at most one signal: among candidates above the SINR threshold, it selects the strongest received power. This permits capture instead of imposing an unconditional collision rule.

## Two-Phase Handshake

1. **HELLO phase:** nodes choosing TX emit simultaneously. Each RX node computes all candidate HELLO SINRs and decodes at most one.
2. **ACK phase:** RX nodes that decoded a HELLO transmit an ACK using their selected reciprocal beam. Original TX nodes compute ACK SINRs under aggregate ACK interference and decode at most one.
3. A neighbor edge is created only if the same directed pair succeeds in both phases.

Failures are recorded separately as forward decode failure, ACK decode failure, interference-limited failure, or noise/fading outage. `collision_count` is retained for compatibility but now counts interference-limited failed attempts rather than every simultaneous transmission.

## TWC Training Parameters

The initial `twc_trainable_n10.yaml` values are:

| Parameter | Value |
|---|---:|
| Carrier frequency | 30 GHz |
| Bandwidth | 64 MHz |
| TX power | 1 W |
| Noise figure | 7 dB |
| Path-loss exponent | 2.0 |
| Shadowing standard deviation | 2 dB |
| Rician K-factor | 10 dB |
| SINR threshold | 5 dB |
| Antenna efficiency | 0.70 |
| Sidelobe gain | -10 dBi |

These values define a reproducible first operating point, not a final calibrated claim. Paper experiments must include sensitivity sweeps for SINR threshold, K-factor, shadowing, sidelobe gain, transmit power, bandwidth, and path-loss exponent.

## Known Limitations

- no probabilistic LoS/blockage state;
- no Doppler-dependent waveform decoding or temporal fading correlation;
- no correlated shadowing field across nearby UAV links;
- no adjacent-channel interference or power-control action;
- ideal sectored antenna rather than measured array pattern;
- fixed SINR threshold instead of MCS/packet-length-dependent BLER.

These limitations must remain explicit until the corresponding model and calibration tests are implemented.
