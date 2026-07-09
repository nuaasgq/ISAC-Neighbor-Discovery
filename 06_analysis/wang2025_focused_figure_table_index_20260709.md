# Wang2025-Focused Single-RF Figure and Table Index (2026-07-09)

## Scope

This index packages the simplified Chinese-paper-aligned comparison matrix:
single RF, MIMO-OTFS sensing abstraction, about 25-degree beams, `N=10--50`,
200 slots, and five paired episodes per case.

## Table Artifacts

| Artifact | Role | Use |
|---|---|---|
| `06_analysis/paper_tables/wang2025_focused_single_rf_20260709/aggregate_metrics.csv` | Main aggregate table | Mean discovery, CPD, collisions, empty scans, delay, and topology by protocol and node count. |
| `06_analysis/paper_tables/wang2025_focused_single_rf_20260709/per_episode_summary.csv` | Raw paired episode rows | Source for paired deltas and episode-level audit. |
| `06_analysis/paper_tables/wang2025_focused_single_rf_20260709/paired_delta_summary.csv` | Paired delta summary | Budgeted ISAC versus Wang-style baselines by node count. |
| `06_analysis/paper_tables/wang2025_focused_single_rf_20260709/paired_deltas_vs_budgeted.csv` | Per-pair deltas | Episode-level paired evidence for discovery, CPD, collisions, and lambda2. |
| `06_analysis/paper_tables/wang2025_focused_single_rf_20260709/manifest.json` | Provenance | Records configuration, node counts, RF setting, protocols, episode count, and PHY-abstraction boundary. |

## Figure Artifacts

| Figure | Claim Supported |
|---|---|
| `node_scaling_discovery_rate_rf1.png` | Budgeted ISAC keeps competitive raw discovery and becomes strongest at larger node counts. |
| `node_scaling_cpd_rf1.png` | Budgeted ISAC gives the best collision-penalized discovery for every tested node count. |
| `node_scaling_collision_rf1.png` | Budgeted ISAC reduces table-induced collision growth compared with Wang-style table exchange. |
| `node_scaling_empty_scan_rf1.png` | ISAC-assisted methods suppress empty scanning relative to blind random scanning. |
| `node_scaling_lambda2_rf1.png` | Budgeted ISAC preserves or improves discovered-graph connectivity in denser single-RF cases. |
| `node_scaling_consumed_slots_rf1.png` | Completion-slot behavior under the 200-slot finite horizon. |

## Main Claim

Recommended manuscript claim:

> In the Wang-style MIMO-OTFS sensing abstraction and single-RF FANET matrix,
> collision-budgeted ISAC access improves effective finite-horizon neighbor
> discovery over Wang-style no-table, communication-table, and sensing-table
> baselines, especially as the number of UAVs increases.

## Key Numbers

Best-Wang baseline here is `wang2025_isac_no_collab` on CPD. Budgeted ISAC
outperforms it at every tested node count:

| N | Budgeted CPD | Best Wang CPD | CPD Gain | Budgeted Discovery | Best Wang Discovery |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.3742 | 0.2638 | +0.1103 | 0.9200 | 0.8267 |
| 20 | 0.2803 | 0.1808 | +0.0995 | 0.7926 | 0.7295 |
| 30 | 0.2408 | 0.1650 | +0.0758 | 0.7053 | 0.6363 |
| 40 | 0.2142 | 0.1314 | +0.0828 | 0.6279 | 0.5674 |
| 50 | 0.1961 | 0.1115 | +0.0846 | 0.6091 | 0.5211 |

At `N=50`, paired deltas against the three Wang-style baselines are positive
for discovery and CPD, and negative for collisions, in all five paired
episodes:

| Control | Discovery Delta | CPD Delta | Collision Delta | CPD Wins |
|---|---:|---:|---:|---:|
| `wang2025_comm_tables` | +0.0700 | +0.1155 | -4417.2 | 5/5 |
| `wang2025_isac_no_collab` | +0.0880 | +0.0846 | -1940.2 | 5/5 |
| `wang2025_isac_tables` | +0.0944 | +0.1232 | -4891.8 | 5/5 |

## Boundary

This evidence supports a protocol-layer ISAC abstraction with PHY-aware sensing
parameters. It does not claim a new MIMO-OTFS waveform receiver, multi-RF
optimization, or MARL superiority over the best rule protocols.
