# Rendezvous Learnability Gate (2026-07-10)

## Material Passport

- Type: experiment result and reproducibility audit.
- Scope: `N=10`, 10-degree beams, 648 beam cells, 300 slots, three training seeds, two held-out episodes per seed.
- Verification status: reproduced from saved raw CSVs and checkpoint ablations.
- Claim boundary: learnability gate only; not final TWC performance evidence.

## Method Under Test

The actor still selects only `TX/RX x beam`. The proposed training path adds:

1. local anonymous sensing-position reports reprojected into the current body-frame beam;
2. a position-pair rendezvous phase and signed local TX/RX role hint;
3. beam and role auxiliary losses derived only from those local observations;
4. a zero-initialized learned ISAC evidence adapter, with beam residuals normalized by `log(M)` for codebook size `M`;
5. the existing soft candidate-ranking objective for exploration, without a hard candidate mask or rule action override.

## Paired Results

| Method | Episodes | Nonzero | Mean edges | Std. | Mean discovery rate | Mean empty-scan ratio |
|---|---:|---:|---:|---:|---:|---:|
| Uniform random | 6 | 0 | 0.000 | 0.000 | 0.00% | 98.68% |
| Wang ISAC tables | 6 | 0 | 0.000 | 0.000 | 0.00% | 98.61% |
| Trained policy, adapter zeroed | 6 | 0 | 0.000 | 0.000 | 0.00% | 98.64% |
| MARL + ISAC adapter | 6 | 5 | 1.333 | 1.033 | 2.96% | 95.81% |

The complete method discovered `[1, 3, 2, 1, 1, 0]` edges in the six held-out scenarios. Its mean-edge 95% t interval is `[0.249, 2.417]`, equivalent to a discovery-rate interval of approximately `[0.55%, 5.37%]`. The exploratory one-sided exact sign result against an all-zero control is `p=0.03125` after excluding the one tie. This p-value is descriptive because the first seed was used during method development and the sample is small.

## Mechanism Chain

Across the six complete-method evaluation episodes:

- local rendezvous observations: 635 agent-slots;
- target-beam hits: 551 (86.77%);
- role matches: 528 (83.15%);
- joint beam-role actions: 459 (72.28%);
- reciprocal-report pair-slots: 1069;
- common-phase pair-slots: 65 (6.08% of reciprocal-report pair-slots);
- actor-converted reciprocal pair-slots: 32;
- new reciprocal alignments and successful handshakes: 8.

The adapter fixes the single-agent 648-way beam-selection bottleneck: the no-adapter screen produced only one beam hit after ten training episodes, while the adapter reached held-out beam-hit rates of 82.8%-90.3% after five episodes. All eight aligned opportunities passed the implemented SINR/ACK PHY in this small sample.

## Remaining Failure Mode

Absolute discovery remains low. The dominant bottleneck is now upstream of PHY decoding:

1. TX-piggyback monostatic sensing often creates a one-sided anonymous report.
2. Repeated exploitation can revisit that same target without giving the other endpoint a report.
3. Independently noisy position quantization sometimes maps reciprocal reports to different rendezvous phases.
4. A correct single-agent action is insufficient unless both reports, phases, beams, and roles coincide.

The next mechanism study should therefore compare robust multi-window phase schedules and explicit learned exploration/exploitation factorization. It must not restore the deterministic 200-slot bootstrap or hard-code final actions.

## Artifacts

- Tables and hashes: `06_analysis/paper_tables/marl/rendezvous_learnability_gate_20260710/`.
- Figures: `06_analysis/paper_figures/marl/rendezvous_learnability_gate_20260710/`.
- Rebuild script: `06_analysis/scripts/build_rendezvous_learnability_gate.py`.
- Raw bundle: `05_simulation/results_raw/marl_rendezvous_learnability_gate_20260710/` (git-ignored).
