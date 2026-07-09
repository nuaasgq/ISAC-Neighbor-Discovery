# Final Data Package Completion Audit (2026-07-09)

## Scope

This audit checks the active objective:

> Complete a paper-usable data package for ISAC-assisted narrow-beam UAV
> neighbor discovery, including single-RF MARL/ISAC/table-exchange/gating/baseline
> experiments, statistical figures, summary tables, manuscript narrative
> material, and version management.

It uses only current repository evidence.

## Requirement-Level Verdict

| Requirement | Evidence | Verdict |
|---|---|---|
| Single-RF experiment line | `06_analysis/paper_tables/wang2025_focused_single_rf_20260709/manifest.json` records `rf_chains=[1]`, `N=10--50`, 200 slots, five episodes, and Wang-style protocols. | PASS |
| ISAC mechanism evidence | `budgeted_collision_aware_isac`, `improved_rl_isac`, and Wang-style ISAC rows appear in `aggregate_metrics.csv`; ISAC methods strongly beat uniform random. | PASS |
| Wang/reference baselines | `wang2025_isac_no_collab`, `wang2025_comm_tables`, and `wang2025_isac_tables` are included in the focused matrix and paired-delta tables. | PASS |
| Random baseline | `uniform_random` is included in the focused matrix and shows near-zero discovery, e.g., `N=50` discovery 0.0104. | PASS |
| Table-exchange comparison | Wang communication-table and sensing-table baselines are included in the focused matrix; `trust_gated_isac_tables` and `improved_rl_isac_tables` are additionally evaluated in `06_analysis/paper_tables/marl/trust_gate_bc_sweep_20260709/`. | PASS |
| Gate/control evidence | Budgeted ISAC access is the main gate/control mechanism; `bc0p15` Budgeted expert gate BC training curves are in `06_analysis/paper_figures/marl_trust_gate_bc_20260709/`. | PASS with boundary |
| MARL evidence | Formal MARL/no-ISAC/gated/BC diagnostic results are archived under `06_analysis/paper_tables/marl/` and summarized in `06_analysis/paper_data_package_audit_20260709.md`. They are not the main performance claim after the Wang-style pivot. | PASS with boundary |
| Statistical summary tables | Main aggregate, per-episode, paired-delta, and manifest files are in `06_analysis/paper_tables/wang2025_focused_single_rf_20260709/`. | PASS |
| Publication-style figures | Focused matrix figures are in `06_analysis/paper_figures/wang2025_focused_single_rf_20260709/`; checked at 1920 x 1440, 4:3. | PASS |
| Paper narrative material | `06_analysis/wang2025_focused_single_rf_results_20260709.md`, `06_analysis/wang2025_focused_figure_table_index_20260709.md`, and `07_paper/ieee_twc_isac_nd/wang2025_focused_single_rf_insert.tex`. | PASS |
| Version management | Latest pushed commit is `4fe4704 Add Wang-focused manuscript package`; working tree was clean before this audit update. | PASS after committing this audit |

## Main Evidence Chain

The cleanest paper-ready result is now the Wang-style single-RF matrix:

- Configuration: MIMO-OTFS sensing abstraction, about 25-degree beams, single RF,
  `N=10/20/30/40/50`, 200 slots, five paired episodes.
- Main result directory: `06_analysis/paper_tables/wang2025_focused_single_rf_20260709/`.
- Figure directory: `06_analysis/paper_figures/wang2025_focused_single_rf_20260709/`.
- Manuscript insert: `07_paper/ieee_twc_isac_nd/wang2025_focused_single_rf_insert.tex`.

At `N=50`, Budgeted ISAC reaches discovery 0.6091, CPD 0.1961, collisions
2582.2, and lambda2 20.3632. The best Wang-style CPD baseline reaches discovery
0.5211, CPD 0.1115, collisions 4522.4, and lambda2 16.0628. Paired deltas
against all three Wang-style baselines improve discovery, CPD, and collision
count in all five paired episodes.

## Claim Boundaries

Supported:

- ISAC sensing feedback is useful as a link-layer candidate-beam service.
- Table exchange alone can amplify contention.
- Collision-budgeted access to ISAC-suggested beams is the clearest current
  mechanism improvement.
- MARL/gated BC evidence is present as diagnostic support, not as the main
  superiority claim.

Not supported:

- A new physical-layer MIMO-OTFS waveform receiver.
- A learned MARL policy that beats the best rule protocol on both discovery and
  CPD.
- A universal claim over all beamwidths, mobility models, or multi-RF settings.

## Completion Decision

For the active data-package objective, the repository now contains the required
single-RF experiments, baselines, table-exchange evidence, gate/control evidence,
MARL diagnostics, figures, summary tables, narrative material, and GitHub-backed
version history. The remaining work is manuscript integration and optional
multi-RF extension, not a blocker for the requested data package.
