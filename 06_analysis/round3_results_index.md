# Round3 Robustness and Transfer Results Index

Generated during the long-run goal window ending at 2026-07-05 11:00 (Asia/Shanghai).

## Experiment Queue

| Experiment | Purpose | Status | Raw output | Archived table output | Figure output |
|---|---|---|---|---|---|
| `round3_ablation` | Mechanism attribution: candidate set, beam lock, topology scoring | complete | `05_simulation/results_raw/round3_ablation` | `06_analysis/paper_tables/round3_robustness/ablation` | `06_analysis/paper_figures/round3_robustness` |
| `round3_n100_density_multiseed` | Multi-seed N=100 transfer under density-preserving scaling | complete | `05_simulation/results_raw/round3_n100_density_multiseed` | `06_analysis/paper_tables/round3_robustness/n100_density_multiseed` | `06_analysis/paper_figures/round3_n100_transfer` |
| `round3_n100_fixed_multiseed` | Multi-seed N=100 transfer under fixed-area scaling | complete | `05_simulation/results_raw/round3_n100_fixed_multiseed` | `06_analysis/paper_tables/round3_robustness/n100_fixed_multiseed` | `06_analysis/paper_figures/round3_n100_transfer` |
| `round3_range_rc_rs_grid` | Physical range sensitivity over `Rc/D` and `Rs/Rc` | complete | `05_simulation/results_raw/round3_range_rc_rs_grid` | `06_analysis/paper_tables/round3_robustness/range_rc_rs_grid` | `06_analysis/paper_figures/round3_robustness` |
| `round3_range_rs_ratio` | Auxiliary `Rs/Rc` sensitivity at single-hop `Rc/D=1.05` | complete | `05_simulation/results_raw/round3_range_rs_ratio` | `06_analysis/paper_tables/round3_robustness/range_rs_ratio` | `06_analysis/paper_figures/round3_robustness` |
| `round3_error_profiles` | Paired ISAC false-alarm, miss-detection, angular-error profiles | complete | `05_simulation/results_raw/round3_error_profiles` | `06_analysis/paper_tables/round3_robustness/error_profiles` | `06_analysis/paper_figures/round3_robustness` |
| `round3_error_robustness` | Auxiliary full-factor small ISAC error grid | complete | `05_simulation/results_raw/round3_error_robustness` | `06_analysis/paper_tables/round3_robustness/error_robustness` | `06_analysis/paper_figures/round3_robustness` |
| `round4_delay_ablation` | Implementation-boundary ablation for one-slot delayed candidate-set use plus efficiency metrics | complete | `05_simulation/results_raw/round4_delay_ablation` | `06_analysis/paper_tables/round4_delay_ablation` | `06_analysis/paper_figures/round4_delay_ablation` |

## Completed Results Snapshot

### Mechanism Ablation, N=100, 10-degree Beam, Density Scaling

| Protocol | Discovery rate | Empty-scan ratio | Lambda2 | Collisions |
|---|---:|---:|---:|---:|
| Uniform random | 0.0005 | 0.9015 | 0.0000 | 0.0 |
| Improved-RL without ISAC | 0.0007 | 0.9011 | 0.0000 | 0.0 |
| No topology term | 0.3617 | 0.5011 | 10.3293 | 1083.3 |
| No beam lock | 0.3553 | 0.4904 | 13.1866 | 1096.3 |
| One-slot candidate delay | 0.2989 | 0.4947 | 8.4709 | 697.0 |
| No candidate set | 0.0313 | 0.4985 | 0.0000 | 3.3 |
| Improved-RL + ISAC | 0.3655 | 0.4986 | 12.9222 | 1050.0 |

Interpretation: the candidate-set refinement enabled by ISAC feedback is the most critical mechanism in this implementation. Removing topology or beam-lock refinements preserves much of the finite-time discovery rate, while removing the candidate set nearly collapses discovery and connectivity. The one-slot delayed candidate-set ablation preserves most of the ISAC benefit while reducing collisions, so it is the key implementation-boundary result for the paper.

### Implementation-Boundary Efficiency, N=100, 10-degree Beam

| Protocol | Discovery rate | Discoveries / 1000 scans | Scan actions / discovery | Collision-penalized discovery |
|---|---:|---:|---:|---:|
| Uniform random | 0.0005 | 0.0458 | 22658.1 | 0.0005 |
| Improved-RL without ISAC | 0.0007 | 0.0567 | 19587.9 | 0.0007 |
| One-slot candidate delay | 0.2989 | 25.6069 | 39.10 | 0.2620 |
| Improved-RL + ISAC | 0.3655 | 31.3563 | 31.89 | 0.3015 |

Interpretation: the delayed candidate-set protocol is a conservative, more implementable variant. It retains about 82% of the full ISAC discovery rate at N=100 and 10-degree beams, while reducing collisions from 1050.0 to 697.0. This supports writing the full ISAC protocol as a low-latency upper design point and the one-slot delayed variant as a practical implementation boundary.

### N=100 Multi-Seed Transfer, Proposed Method

| Scaling | Beamwidth | Discovery rate | Empty-scan ratio | Lambda2 | Mean delay |
|---|---:|---:|---:|---:|---:|
| Density | 5 deg | 0.0814 | 0.8161 | 0.8430 | 576.64 |
| Density | 10 deg | 0.3655 | 0.4986 | 12.9222 | 477.99 |
| Density | 15 deg | 0.5440 | 0.3293 | 26.8413 | 406.74 |
| Density | 30 deg | 0.4568 | 0.1815 | 25.9812 | 438.66 |
| Fixed | 5 deg | 0.0822 | 0.8152 | 1.0800 | 576.21 |
| Fixed | 10 deg | 0.3615 | 0.5021 | 11.0417 | 479.00 |
| Fixed | 15 deg | 0.5352 | 0.3315 | 23.2957 | 410.82 |
| Fixed | 30 deg | 0.4487 | 0.1816 | 25.0550 | 442.39 |

Interpretation: density-preserving and fixed-area scaling give similar trends for N=100. The strongest transfer regime is 10-30 degrees, with 15 degrees currently giving the best discovery/connectivity balance. The 5-degree case remains a stress regime and should not be written as solved.

### Rs/Rc Sensitivity at Single-Hop Rc/D=1.05, N=100, 10-degree Beam

| Protocol | Rs/Rc | Discovery rate | Empty-scan ratio | Lambda2 |
|---|---:|---:|---:|---:|
| Improved-RL + ISAC | 0.50 | 0.3570 | 0.5082 | 11.0633 |
| Improved-RL + ISAC | 0.75 | 0.3637 | 0.4970 | 13.9211 |
| Improved-RL + ISAC | 1.00 | 0.3655 | 0.4986 | 12.9222 |
| Improved-RL + ISAC | 1.25 | 0.3655 | 0.4986 | 12.9222 |

Interpretation: increasing sensing range from half the communication range to the communication range gives a small improvement. Increasing sensing beyond the communication range does not further improve communication neighbor discovery in the current model because only communication-range neighbors are immediately discoverable.

### Rc/D x Rs/Rc Range Matrix, Proposed Method, N=100, 10-degree Beam

| Rc/D | Rs/Rc | Discovery rate | Empty-scan ratio | Lambda2 |
|---:|---:|---:|---:|---:|
| 0.65 | 0.50 | 0.2628 | 0.5954 | 7.9064 |
| 0.65 | 0.75 | 0.3569 | 0.5105 | 13.1966 |
| 0.65 | 1.00 | 0.3569 | 0.5028 | 13.9766 |
| 0.65 | 1.25 | 0.3569 | 0.5028 | 13.9766 |
| 0.85 | 0.50 | 0.3366 | 0.5294 | 14.5038 |
| 0.85 | 0.75 | 0.3590 | 0.5010 | 12.1291 |
| 0.85 | 1.00 | 0.3572 | 0.5035 | 10.4264 |
| 0.85 | 1.25 | 0.3572 | 0.5035 | 10.4264 |
| 1.05 | 0.50 | 0.3570 | 0.5082 | 11.0633 |
| 1.05 | 0.75 | 0.3637 | 0.4970 | 13.9211 |
| 1.05 | 1.00 | 0.3655 | 0.4986 | 12.9222 |
| 1.05 | 1.25 | 0.3655 | 0.4986 | 12.9222 |

Interpretation: the useful sensing range saturates around the communication range in this communication-neighbor-discovery model. The most visible degradation appears when both communication range and sensing range are small (`Rc/D=0.65`, `Rs/Rc=0.5`).

### ISAC Error Robustness, N=100, 10-degree Beam

The full-factor error grid uses 400-slot episodes with two seeds. It should be used for robustness trends, not directly compared against the 600-slot main transfer numbers.

| Pfa | Pmd | Angular std | Proposed discovery rate | Proposed empty-scan ratio | No-ISAC discovery rate |
|---:|---:|---:|---:|---:|---:|
| 0.000 | 0.00 | 0.0 | 0.2771 | 0.4929 | 0.0004 |
| 0.000 | 0.02 | 0.0 | 0.2847 | 0.4981 | 0.0004 |
| 0.000 | 0.05 | 0.0 | 0.2772 | 0.4973 | 0.0004 |
| 0.005 | 0.00 | 0.0 | 0.2699 | 0.5015 | 0.0004 |
| 0.005 | 0.02 | 0.0 | 0.2691 | 0.5066 | 0.0004 |
| 0.005 | 0.05 | 0.0 | 0.2747 | 0.5002 | 0.0004 |
| 0.010 | 0.00 | 0.0 | 0.2657 | 0.5005 | 0.0004 |
| 0.010 | 0.02 | 0.0 | 0.2687 | 0.5045 | 0.0004 |
| 0.010 | 0.05 | 0.0 | 0.2694 | 0.5028 | 0.0004 |

The paired 600-slot profile sweep further includes angular error:

| Pfa | Pmd | Angular std | Proposed discovery rate | Proposed empty-scan ratio | Lambda2 |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.00 | 0.0 | 0.3655 | 0.4986 | 12.9222 |
| 0.01 | 0.05 | 0.5 | 0.3357 | 0.5464 | 13.6500 |
| 0.05 | 0.15 | 1.0 | 0.2854 | 0.5934 | 9.8519 |
| 0.10 | 0.30 | 1.5 | 0.2935 | 0.6111 | 12.0446 |

Interpretation: noisy ISAC degrades discovery and increases empty scans, but the proposed protocol remains far above the no-ISAC baseline in these tested regimes. The result supports a bounded robustness claim, not a claim of immunity to sensing errors.

## Figure Inventory

- `06_analysis/paper_figures/round3_robustness/ablation_discovery_n100_b10.png`
- `06_analysis/paper_figures/round3_robustness/ablation_empty_scan_n100_b10.png`
- `06_analysis/paper_figures/round3_robustness/ablation_lambda2_n100_b10.png`
- `06_analysis/paper_figures/round3_robustness/ablation_collision_n100_b10.png`
- `06_analysis/paper_figures/round3_n100_transfer/density_beamwidth_discovery_rate_n100.png`
- `06_analysis/paper_figures/round3_n100_transfer/density_beamwidth_empty_scan_n100.png`
- `06_analysis/paper_figures/round3_n100_transfer/density_beamwidth_lambda2_n100.png`
- `06_analysis/paper_figures/round3_n100_transfer/fixed_beamwidth_discovery_rate_n100.png`
- `06_analysis/paper_figures/round3_n100_transfer/fixed_beamwidth_empty_scan_n100.png`
- `06_analysis/paper_figures/round3_n100_transfer/fixed_beamwidth_lambda2_n100.png`
- `06_analysis/paper_figures/round3_n100_transfer/area_scale_n100_discovery_rate.png`
- `06_analysis/paper_figures/round3_n100_transfer/area_scale_n100_empty_scan.png`
- `06_analysis/paper_figures/round3_n100_transfer/area_scale_n100_lambda2.png`
- `06_analysis/paper_figures/round3_robustness/range_protocol_discovery_n100_b10.png`
- `06_analysis/paper_figures/round3_robustness/range_gain_discovery_n100_b10.png`
- `06_analysis/paper_figures/round3_robustness/range_gain_empty_scan_n100_b10.png`
- `06_analysis/paper_figures/round3_robustness/range_gain_lambda2_n100_b10.png`
- `06_analysis/paper_figures/round3_robustness/error_gain_discovery_n100_b10.png`
- `06_analysis/paper_figures/round3_robustness/error_gain_empty_scan_n100_b10.png`
- `06_analysis/paper_figures/round3_robustness/error_profile_discovery_n100_b10.png`
- `06_analysis/paper_figures/round4_delay_ablation/ablation_discovery_n100_b10.png`
- `06_analysis/paper_figures/round4_delay_ablation/ablation_collision_penalized_discovery_n100_b10.png`
- `06_analysis/paper_figures/round4_delay_ablation/ablation_discovery_per_scan_n100_b10.png`

## Paper-Writing Implications

- Strong claim: ISAC feedback is useful mainly because it creates a candidate beam set after blind probing, not because it directly discovers neighbors.
- Strong claim: a one-slot delayed candidate-set variant remains far above no-ISAC baselines, so the mechanism is not wholly dependent on same-slot sensing-to-handshake reuse.
- Strong claim: the learned small-scale policy transfers to N=100 for 10-30 degree beams under both density-preserving and fixed-area scaling.
- Conservative claim: 3-5 degree beams are still a stress region under the current finite-time horizon.
- Conservative claim: current learning is shared-policy parameter optimization, not yet a full neural MARL implementation.
- Remaining gap: energy-normalized efficiency still requires an explicit action-energy model; current efficiency results are scan-action and collision normalized, not Joule normalized.
