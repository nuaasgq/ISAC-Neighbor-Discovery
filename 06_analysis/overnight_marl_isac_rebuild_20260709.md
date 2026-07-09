# Overnight MARL-ISAC Rebuild Report (2026-07-09)

## Scope

This bundle consolidates the single-RF, N=10-trained MARL/ISAC runs, N=50/N=100 transfer evaluations, B=10 and B=15 beamwidth checks, Wang-style baselines, rule ISAC baselines, and expert-assisted BC-MARL variants generated overnight.

## Main Findings

- ISAC sensing/candidate information is decisive: non-ISAC random and no-ISAC proxy protocols remain near zero discovery in the narrow-beam 3D setting.
- Wang-style ISAC and collision-aware rule ISAC are currently stronger than trained MARL on the primary target discovery-rate metric.
- Expert-assisted BC-MARL improves large-scale transfer discovery over the earlier trained MARL, but it creates excessive collisions and therefore poor collision-penalized discovery.
- Table exchange is not yet a reliable win in the current implementation; it needs trust-gated fusion instead of unconditional boosting.

## Selected Metrics

| B | N | Method | Discovery | Collisions | CPD | Lambda2 |
|---:|---:|---|---:|---:|---:|---:|
| 10 | 100 | BC-MARL | 0.266 | 118806.7 | 0.011 | 13.017 |
| 10 | 100 | MARL | 0.180 | 21971.3 | 0.033 | 7.732 |
| 10 | 100 | Wang ISAC | 0.681 | 1343.3 | 0.536 | 42.538 |
| 10 | 100 | Collision-aware ISAC | 0.767 | 4676.7 | 0.395 | 43.307 |
| 15 | 100 | BC-MARL | 0.349 | 249641.5 | 0.007 | 16.550 |
| 15 | 100 | Wang ISAC | 0.884 | 14533.0 | 0.227 | 70.278 |

## Paper-Readiness Judgment

The current data are useful for a diagnostic paper section and for motivating the final method, but they are not yet sufficient to claim a high-level TWC/TCOM MARL method that beats Wang-style ISAC baselines. The strongest defensible claim is: ISAC-assisted empty-beam exclusion makes the problem tractable; naive MARL does not inherit that benefit under N=10 to N=100 transfer; rule-guided expert pretraining improves raw discovery but must be paired with stronger collision-constrained MARL.

## Next Technical Move

The next method should factor actions into two timescales: a rule/ISAC candidate beam executor and a learned constrained access controller trained with collision budget or Lagrangian penalty. The gate should learn when to throttle Tx, not overwrite candidate beams. Table exchange should be trust-gated by recency, collision evidence, and peer-table consistency.

## Artifact Locations

- Tables: `06_analysis\paper_tables\marl\overnight_20260709_marl_isac_rebuild`
- Figures: `06_analysis\paper_figures\marl_overnight_20260709`
- Raw results: `05_simulation\results_raw\overnight_20260709`
