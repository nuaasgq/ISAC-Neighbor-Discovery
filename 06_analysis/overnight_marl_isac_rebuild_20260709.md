# Overnight MARL-ISAC Rebuild Report (2026-07-09)

## Scope

This bundle consolidates the single-RF, N=10-trained MARL/ISAC runs, N=50/N=100 transfer evaluations, B=10 and B=15 beamwidth checks, Wang-style baselines, rule ISAC baselines, and expert-assisted BC-MARL variants generated overnight.

## Main Findings

- ISAC sensing/candidate information is decisive: non-ISAC random and no-ISAC proxy protocols remain near zero discovery in the narrow-beam 3D setting.
- Wang-style ISAC and collision-aware rule ISAC are currently stronger than trained MARL on the primary target discovery-rate metric.
- Budgeted collision-aware ISAC gives a stronger constrained-access expert: it trades a small raw-discovery loss against large collision reductions and better collision-penalized discovery.
- Expert-assisted BC-MARL improves large-scale transfer discovery over the earlier trained MARL, but it creates excessive collisions and therefore poor collision-penalized discovery.
- Table exchange is not yet a reliable win in the current implementation; it needs trust-gated fusion instead of unconditional boosting.

## Selected Metrics

| B | N | Method | Discovery | Collisions | CPD | Lambda2 |
|---:|---:|---|---:|---:|---:|---:|
| 10 | 100 | BC-MARL | 0.266 | 118806.7 | 0.011 | 13.017 |
| 10 | 100 | MARL | 0.180 | 21971.3 | 0.033 | 7.732 |
| 10 | 100 | Wang ISAC | 0.681 | 1343.3 | 0.536 | 42.538 |
| 10 | 100 | Collision-aware ISAC | 0.767 | 4676.7 | 0.395 | 43.307 |
| 10 | 100 | Budgeted ISAC | 0.712 | 1347.0 | 0.560 | 38.871 |
| 15 | 100 | BC-MARL | 0.349 | 249641.5 | 0.007 | 16.550 |
| 15 | 100 | Wang ISAC | 0.884 | 14533.0 | 0.227 | 70.278 |
| 15 | 100 | Budgeted ISAC | 0.862 | 12602.0 | 0.245 | 61.981 |

## Paper-Readiness Judgment

The current data now support a stronger protocol-side contribution: density-adaptive, budgeted ISAC access can outperform Wang-style ISAC on collision-penalized discovery in the tested single-hop settings. They are still not sufficient to claim a high-level TWC/TCOM MARL method that beats the best rule experts. The defensible MARL claim is narrower: naive MARL does not inherit ISAC benefits under N=10 to N=100 transfer, while rule-guided expert pretraining improves raw discovery but must be paired with stronger collision-constrained access learning.

## Next Technical Move

The next method should distill `budgeted_collision_aware_isac` into a learned constrained access controller trained with a collision budget or Lagrangian penalty. The gate should learn when to throttle Tx, not overwrite candidate beams. Table exchange should be trust-gated by recency, collision evidence, and peer-table consistency.

## Artifact Locations

- Tables: `06_analysis\paper_tables\marl\overnight_20260709_marl_isac_rebuild`
- Figures: `06_analysis\paper_figures\marl_overnight_20260709`
- Raw results: `05_simulation\results_raw\overnight_20260709`
- Budgeted raw results: `05_simulation\results_raw\round_budgeted_isac_eval`
