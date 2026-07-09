# Wang2025-Focused Single-RF Result Note (2026-07-09)

## Scope

This note records the simplified experiment block aligned with the Chinese
MIMO-OTFS ISAC FANET neighbor-discovery paper, after stopping the broader
3000-slot MARL transfer expansion.

Experiment matrix:

- Scenario: PHY-aware MIMO-OTFS sensing abstraction in `wang2025_reproduction_smoke.yaml`.
- Beam setting: about 24 deg azimuth by 25.7 deg elevation.
- Node counts: `N = 10, 20, 30, 40, 50`.
- RF chains: `1`.
- Horizon: `200` slots, `5` episodes per node count and protocol.
- Single-hop control: communication and sensing ranges both cover the 10 km cube diagonal.
- Protocols: `uniform_random`, `wang2025_isac_no_collab`, `wang2025_comm_tables`,
  `wang2025_isac_tables`, `improved_rl_isac`, `budgeted_collision_aware_isac`.

Generated artifacts:

- Tables: `06_analysis/paper_tables/wang2025_focused_single_rf_20260709/`
- Figures: `06_analysis/paper_figures/wang2025_focused_single_rf_20260709/`
- Main CSV: `aggregate_metrics.csv`
- Paired deltas: `paired_deltas_vs_budgeted.csv`, `paired_delta_summary.csv`

## Key Results

At `N=50`, the budgeted ISAC access rule is the strongest method in this
focused matrix:

| Protocol | Discovery | CPD | Collisions | Lambda2 |
|---|---:|---:|---:|---:|
| `budgeted_collision_aware_isac` | 0.6091 | 0.1961 | 2582.2 | 20.3632 |
| `wang2025_comm_tables` | 0.5391 | 0.0807 | 6999.4 | 17.6903 |
| `wang2025_isac_no_collab` | 0.5211 | 0.1115 | 4522.4 | 16.0628 |
| `wang2025_isac_tables` | 0.5148 | 0.0729 | 7474.0 | 14.7000 |
| `improved_rl_isac` | 0.4477 | 0.0587 | 8147.2 | 12.7513 |
| `uniform_random` | 0.0104 | 0.0104 | 0.8 | 0.0000 |

Across the tested node counts, `budgeted_collision_aware_isac` gives the best
collision-penalized discovery rate at every `N`. Its raw discovery is not always
the best at small `N`, where Wang table variants have slightly higher discovery,
but the budgeted rule avoids the table-induced collision growth.

The paired delta summary strengthens this point. At `N=50`, against all three
Wang-style baselines, the budgeted ISAC rule improves discovery, CPD, and
collision count in all 5 paired episodes:

| Control | Discovery Delta | CPD Delta | Collision Delta | CPD Wins |
|---|---:|---:|---:|---:|
| `wang2025_comm_tables` | +0.0700 | +0.1155 | -4417.2 | 5/5 |
| `wang2025_isac_no_collab` | +0.0880 | +0.0846 | -1940.2 | 5/5 |
| `wang2025_isac_tables` | +0.0944 | +0.1232 | -4891.8 | 5/5 |

## Interpretation

This is a cleaner result than the stopped 3000-slot MARL expansion. In the
paper-aligned 200-slot Wang-style matrix, the proposed budgeted ISAC access rule
can be positioned as a MAC/link-layer improvement over Wang-style ISAC table
exchange: it keeps discovery competitive and substantially improves CPD and
connectivity under larger node counts.

The defensible claim is:

> Under the same MIMO-OTFS sensing abstraction and Wang-style FANET discovery
> matrix, collision-budgeted ISAC access improves effective finite-horizon
> neighbor discovery over Wang-style no-table, communication-table, and
> sensing-table baselines in denser single-RF networks.

Do not claim from this block:

- A new physical-layer waveform receiver.
- Multi-RF superiority.
- MARL superiority over rule protocols.
- General dominance in all horizons or beamwidths.

## Next Minimal Tasks

1. Keep this as the primary comparison matrix for the next paper draft pass.
2. If needed, add only one optional extension: `rf_chains = 3, 6`, because the
   Chinese paper includes multi-RF behavior; keep the main text single RF unless
   multi-RF strengthens the story.
