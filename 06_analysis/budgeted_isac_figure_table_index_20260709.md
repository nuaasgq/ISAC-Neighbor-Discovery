# Budgeted ISAC Figure/Table Index (2026-07-09)

## Primary Tables

| Artifact | Use in paper | Key claim supported |
|---|---|---|
| `06_analysis/paper_tables/marl/overnight_20260709_marl_isac_rebuild/transfer_summary.csv` | Main result table source | Budgeted ISAC improves collision-penalized discovery over Wang-style ISAC at N=100 for B=10 and B=15. |
| `06_analysis/paper_tables/marl/overnight_20260709_marl_isac_rebuild/transfer_episode_rows.csv` | Per-episode evidence and rerun audit source | Main averages are computed from raw episode rows, not hand-entered values. |
| `06_analysis/paper_tables/marl/overnight_20260709_marl_isac_rebuild/training_summary.csv` | MARL diagnostic table | BC-MARL improves raw transfer discovery but does not solve collision control. |
| `06_analysis/paper_tables/marl/overnight_20260709_marl_isac_rebuild/manifest.json` | Artifact provenance | Links raw roots, figure directory, and generated row counts. |

## Recommended Main Figures

| Figure | Suggested placement | Message |
|---|---|---|
| `transfer_b10_n100_cpd.png` | Main results | At B=10, N=100, Budgeted ISAC has the best collision-penalized discovery among the tested rule/MARL baselines. |
| `transfer_b10_n100_discovery.png` | Main results | Budgeted ISAC preserves most of the raw-discovery gain of aggressive collision-aware ISAC while beating Wang ISAC on discovery. |
| `transfer_b10_n100_collision.png` | Main results | Budgeted ISAC collapses the collision count of non-budgeted collision-aware ISAC to the Wang-level range. |
| `beamwidth_transfer_n100.png` | Robustness/transfer | Budgeted ISAC remains competitive under B=10 to B=15 beamwidth transfer at N=100. |
| `training_discovery_rate.png` | Learning diagnostic | N=10-trained learned policies improve but remain below rule experts in transfer. |
| `training_expert_bc_loss.png` | Learning diagnostic | Expert-guided training is active and measurable, supporting the rule-to-MARL narrative. |

## Supplementary Figures

| Figure family | Use |
|---|---|
| `transfer_b10_n50_*` | Shows that N=50 behavior is consistent but less favorable to Budgeted ISAC on CPD than N=100/B=10; useful for honest boundary discussion. |
| `transfer_b15_n50_*` | Shows B=15, N=50 mixed behavior; avoid overclaiming universal superiority. |
| `transfer_b15_n100_*` | Supports robustness of Budgeted ISAC at wider beams and larger scale. |
| `training_gate_*` | Documents current gate-action distribution and why learned gating still needs a stronger constraint objective. |

## Recommended Claim Wording

Use:

> In the tested single-hop large-swarm transfer setting, density-adaptive budgeted ISAC access improves collision-penalized discovery relative to Wang-style ISAC while retaining high raw discovery.

Avoid:

> The current MARL method outperforms all rule baselines.

Avoid:

> Table exchange always improves discovery.

## Main Text Integration Plan

1. Replace the older Phase10 result narrative in `main.tex` with the Budgeted ISAC mechanism and result summary.
2. Keep MARL as a diagnostic and forward-looking constrained-access-learning component unless a later learned gate exceeds Budgeted ISAC.
3. Move table-exchange claims to a limitations/ablation paragraph: table exchange without trust gating can amplify stale or congested candidate beams.
4. Use `budgeted_isac_manuscript_insert.tex` as the method subsection source.
