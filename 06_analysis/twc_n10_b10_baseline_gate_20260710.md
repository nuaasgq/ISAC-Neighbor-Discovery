# TWC N=10, B=10 Baseline Learnability Gate (2026-07-10)

## Purpose

This probe checks whether the corrected 648-beam environment produces enough reciprocal beam-alignment opportunities to justify MARL training. It is an engineering gate, not paper evidence.

## Common Environment

- Configuration: `05_simulation/configs/twc_canonical_n10_b10.yaml`
- Nodes: 10
- Beamwidth: 10 degrees in azimuth and elevation (648 cells)
- Horizon: 300 slots at 5 ms per slot
- One RF chain; TX-coupled sensing; neighbor and sensing table exchange enabled where specified
- Communication and sensing range: 18 km in a 10 km cube
- Communication PHY: close-in path loss, Rician fading, log-normal shadowing, normalized directional gain, aggregate interference, and SINR decoding
- Shared exogenous realizations: mobility and communication channel depend on scenario seed; sensing events are keyed by scenario, slot, node, beam, and event

## Execution

The original 10-seed command reached the 10-minute hard timeout after completing random and Wang. The incomplete rule-ISAC child directory contained no result file and is not treated as complete. A separate three-seed rule probe was then run and compared only against the first three completed random/Wang seeds.

## Paired Three-Seed Result

| Method | Discoveries/episode | HELLO/episode | TX-RX role pairs/episode | Aligned opportunities | Empty-scan ratio |
|---|---:|---:|---:|---:|---:|
| Uniform random | 0.0 | 1524.3 | 6741.7 | 0 | 0.9876 |
| Wang2025 ISAC tables | 0.0 | 1474.7 | 6728.0 | 0 | 0.9854 |
| Rule ISAC tables | 0.0 | 1659.0 | 6469.3 | 0 | 0.9720 |

All methods produced zero forward decodes, successful handshakes, and discovered edges. The rule method reduced empty scanning, so sensing information affected beam selection, but it did not create reciprocal rendezvous within 300 slots.

## Ten-Seed Completed Subset

Uniform random and Wang each completed ten seeds. Both retained zero aligned opportunities and zero discoveries. Their mean empty-scan ratios were 0.9874 and 0.9857, respectively. These rows confirm that the three-seed zero result is not caused by one unusual seed, but they do not provide a ten-seed comparison against rule ISAC.

## PHY Calibration Check

Under the canonical normalized B=10 antenna, the original `n=2.1`, 5 dB, 1 W point gives 17.32 km isolated-link coverage 0.9990 (95% CI 0.9984-0.9993). The physical link is therefore nearly ideal for most aligned pairs. The zero-discovery result is caused by rendezvous sparsity before PHY decoding, not by path loss or fading.

## Decision

**Gate status: FAILED. Do not start long MARL training.**

The next mechanism must convert a one-sided positive ISAC observation into a persistent, locally observable rendezvous state. Candidate designs may use the anonymous ISAC position estimate, confidence, and staleness as actor inputs, while TX/RX/beam remain the only actions. No protocol may receive current target identity, true direction, or global topology at execution time.

Before training, a diagnostic policy must demonstrate nonzero `aligned_handshake_opportunities` in the same three seeds. Then random, Wang, rule ISAC, and the diagnostic policy must be rerun with paired seeds and the exact same PHY/MAC contract.

## Raw Artifacts

- Completed random/Wang: `05_simulation/results_raw/twc_n10_b10_baseline_gate_20260710/`
- Three-seed rule probe: `05_simulation/results_raw/twc_n10_b10_rule_probe3_20260710/`
- B=10 PHY calibration: `05_simulation/results_raw/communication_phy_calibration_v2_b10_20260710/`
