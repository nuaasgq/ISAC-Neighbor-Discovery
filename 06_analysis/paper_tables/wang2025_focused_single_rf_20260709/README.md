# Wang2025 Extended Comparison

This campaign compares Wang2025-like rule mechanisms and the current topology-aware ISAC method in a shared PHY-aware FANET setting.

- Node counts: [10, 20, 30, 40, 50]
- RF chains: [1]
- Protocols: uniform_random, wang2025_isac_no_collab, wang2025_comm_tables, wang2025_isac_tables, improved_rl_isac, budgeted_collision_aware_isac
- Episode rows: 150
- Slot rows: 30000
- Figures: `06_analysis/paper_figures/wang2025_focused_single_rf_20260709`

Files:

- `per_episode_summary.csv`
- `per_slot_metrics.csv`
- `completion_slots.csv`
- `aggregate_metrics.csv`
- `paired_deltas_vs_budgeted.csv`
- `paired_delta_summary.csv`
- `manifest.json`
