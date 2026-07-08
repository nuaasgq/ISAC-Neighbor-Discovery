# Wang2025 PHY-Aware Reproduction and Comparison Plan

## Purpose

This note records the next experimental route after reading Wang et al. 2025, *A Rapid Neighbor Discovery Method for Directional FANETs Based on MIMO-OTFS ISAC Waveform*.

The key decision is to strengthen the code-level physical-layer abstraction without turning this project into a waveform-design paper. The simulator now supports a calibrated MIMO-OTFS-style radar-SNR sensing model and a Wang2025-like rule baseline in the same neighbor-discovery environment used by our ISAC/MARL methods.

## Why This Is Needed

TWC can accept cross-layer, data-driven, and ad hoc wireless-network work, but the ISAC component must not look like an oracle. A credible submission needs:

- a physical-layer sensing model with distance/SNR dependence;
- sensing false alarm, miss detection, and angular error;
- a clear mapping from MIMO-OTFS sensing outputs to link-layer beam-cell beliefs;
- comparison with a close ISAC-assisted FANET neighbor-discovery rule baseline.

## Code Additions

- `05_simulation/src/isac_nd_sim/phy_sensing.py`
  - Implements radar-equation SNR:
    `SNR = lambda^2 sigma P_t / ((4 pi)^3 r^4 sigma_w^2)`.
  - Maps SNR to detection probability with a calibrated logistic detector.
  - Keeps processing/array/OTFS gain as an explicit dB parameter.

- `wang2025_isac_tables` protocol in `05_simulation/src/isac_nd_sim/simulator.py`
  - Random Tx/Rx selection.
  - Random scan among active sensing-table beam flags.
  - Higher probability of selecting high-belief potential-target beams.
  - Tx-side piggyback ISAC sensing.
  - Successful handshakes exchange neighbor/sensing table information.
  - Indirect table information boosts beam beliefs but does not directly count as a confirmed link.

- `05_simulation/configs/wang2025_reproduction_smoke.yaml`
  - 10 km x 10 km x 10 km Wang-style 3D FANET region.
  - 30 GHz carrier, 64 MHz bandwidth, 1 W transmit power, 1 m^2 RCS.
  - Around 25 degree beam cells via 15 azimuth x 7 elevation beams.
  - 200-slot cap and 5 ms slot duration.
  - Single-hop first-step setting: communication and sensing ranges exceed region diagonal.

## Smoke Result

Command:

```powershell
$env:PYTHONPATH='05_simulation/src'
python -m isac_nd_sim.runner --config 05_simulation/configs/wang2025_reproduction_smoke.yaml --output 05_simulation/results_raw/wang2025_smoke --episodes 1 --slots 40
```

Output rows: 3 protocol rows, 120 slot rows, 120 discovered-edge rows.

| Protocol | Discovery rate | CPD | Empty scan ratio | Lambda2 | Mean sensing Pd |
|---|---:|---:|---:|---:|---:|
| uniform_random | 0.0053 | 0.0053 | 0.8488 | 0.0000 | 0.0000 |
| wang2025_isac_tables | 0.3105 | 0.1960 | 0.6788 | ~0.0000 | 0.8922 |
| improved_rl_isac | 0.3158 | 0.1714 | 0.4877 | 0.7886 | 0.8657 |

This smoke result is not paper evidence. It only confirms that the same-environment comparison pipeline runs and yields meaningful sensing metrics.

## Next Experiments

1. Reproduce Wang-style curves under the same environment:
   - node count: 10, 20, 30, 40, 50;
   - slots: 200;
   - beamwidth around 25 degrees;
   - protocols: no collaboration, communication-only collaboration if implemented, `wang2025_isac_tables`;
   - metrics: consumed slots to completion, finite-time discovery rate, sensing Pd, empty scan ratio.

2. Apply our methods in the same environment:
   - `improved_rl_isac` rule-enhanced method;
   - trained MARL policy variants once checkpoint wiring is selected;
   - ablation without ISAC and without table exchange.

3. Extend beyond Wang2025:
   - beamwidth: 3, 5, 10, 15, 25 degrees;
   - node count: 10 to 100;
   - mobility enabled with 5 ms slots;
   - PHY sensitivity: processing gain, sensing range, false alarm, miss detection, and angular error.

4. Paper claim boundary:
   - Wang2025 establishes feasibility of MIMO-OTFS-assisted FANET ND.
   - Our claim should be scalable, topology-aware, learning/rule-driven exploitation of imperfect ISAC beam-cell priors.
   - Do not claim novelty for MIMO-OTFS waveform design or generic sensing-table exchange.
