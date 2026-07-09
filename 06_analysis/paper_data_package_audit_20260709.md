# Paper Data Package Audit (2026-07-09)

## Scope

This audit checks whether the current repository contains a paper-usable evidence package for the single-RF ISAC-assisted narrow-beam UAV neighbor-discovery study. It is based on the current worktree, committed analysis artifacts, result CSV files, and the latest simulator/test state.

## Requirement Coverage

| Requirement | Current evidence | Status |
|---|---|---|
| Single-RF setting | `paper_transfer_train_n10_b10_singlehop.yaml`; all overnight and budgeted protocol evaluations use one RF chain unless otherwise specified by the base config. | Covered |
| ISAC-assisted empty-beam/candidate mechanism | `improved_rl_isac`, `collision_aware_isac`, and `budgeted_collision_aware_isac` in `05_simulation/src/isac_nd_sim/simulator.py`; ISAC belief/candidate update metrics in evaluation CSVs. | Covered |
| Wang-style baseline | `wang2025_isac_no_collab`, `wang2025_comm_tables`, `wang2025_isac_tables` in the B=10/B=15 summaries. | Covered |
| Chinese-paper-aligned Wang matrix | `06_analysis/paper_tables/wang2025_focused_single_rf_20260709/` covers single RF, about 25-degree beams, `N=10/20/30/40/50`, 200 slots, and five paired episodes under the MIMO-OTFS sensing abstraction. | Covered, recommended main comparison |
| Completely random baseline | `Uniform random` rows in `transfer_summary.csv`; discovery stays near zero under narrow beams. | Covered |
| MARL without/with ISAC and gate variants | `MARL`, `MARL+gate`, `MARL+tables`, `MARL+tables+gate`, `BC-MARL`, and `BC-MARL+gate` rows in `transfer_summary.csv`. | Covered |
| Table exchange comparison | `MARL+tables`, `MARL+tables+gate`, `Wang ISAC+tables`, and `wang2025_comm_tables` rows. Current results show table exchange is not yet a robust gain. | Covered, negative result |
| Trust-gated table exchange | `trust_gated_isac_tables` is implemented and unit-tested. Formal B=10, N=100, 3000-slot, 5-episode results are under `06_analysis/paper_tables/marl/trust_gate_bc_sweep_20260709/`; the result is negative on CPD because collisions remain high. | Covered, negative B=10 result |
| Budgeted expert gate BC | `run_marl_training.py` now exposes Budgeted expert `access_gate` labels and imitates them in BC loss. The `budgeted_gate_bc_sweep_20260709_v2` 0.15-weight run is still training; transfer results are not yet available. | Implemented, training in progress |
| N=10 training to N=50/N=100 transfer | `training_summary.csv` and `transfer_summary.csv` under `06_analysis/paper_tables/marl/overnight_20260709_marl_isac_rebuild/`. | Covered |
| Beamwidth transfer | B=10 and B=15 are covered in the main package; 3/5/30-degree sweeps remain archived boundary evidence rather than current main claims. | Partially covered |
| 300-slot training and 3000-slot testing | Training rows use 300-slot episodes for formal MARL runs; transfer rows use 3000-slot evaluations. | Covered |
| 4:3 Times New Roman figures | 26 PNGs under `06_analysis/paper_figures/marl_overnight_20260709/`, checked at 1760x1320. | Covered |
| Paper narrative material | `overnight_marl_isac_rebuild_20260709.md` plus the manuscript insert `07_paper/ieee_twc_isac_nd/budgeted_isac_manuscript_insert.tex`. | Covered for current results |
| Wang-focused manuscript material | `06_analysis/wang2025_focused_single_rf_results_20260709.md`, `06_analysis/wang2025_focused_figure_table_index_20260709.md`, and `07_paper/ieee_twc_isac_nd/wang2025_focused_single_rf_insert.tex`. | Covered |
| Independent Budgeted ISAC rerun | `06_analysis/paper_tables/marl/budgeted_isac_paired_rerun_b10_n100/` contains an independently seeded B=10, N=100, 3000-slot paired rerun against Wang ISAC, Collision-aware ISAC, and Uniform random. | Covered |
| Version management | The current evidence package is tracked in Git and should be committed after this audit update. | Covered after next commit |

## Main Quantitative Evidence

The cleanest current main-paper comparison is the Wang-style single-RF matrix,
because it directly follows the Chinese MIMO-OTFS ISAC FANET experiment shape:
about 25-degree beams, `N=10--50`, 200 slots, single RF, and Wang-style
no-table / communication-table / sensing-table baselines. Its tables are under
`06_analysis/paper_tables/wang2025_focused_single_rf_20260709/`.

At the densest focused point, `N=50`, Budgeted ISAC is strongest on discovery,
CPD, collisions, and discovered-graph connectivity:

| Method | Discovery | Collisions | CPD | Lambda2 |
|---|---:|---:|---:|---:|
| Wang ISAC, no table | 0.5211 | 4522.4 | 0.1115 | 16.0628 |
| Wang communication table | 0.5391 | 6999.4 | 0.0807 | 17.6903 |
| Wang sensing table | 0.5148 | 7474.0 | 0.0729 | 14.7000 |
| Ours, non-budgeted ISAC | 0.4477 | 8147.2 | 0.0587 | 12.7513 |
| Ours, Budgeted ISAC | 0.6091 | 2582.2 | 0.1961 | 20.3632 |

Paired deltas at `N=50` improve discovery, CPD, and collision count against all
three Wang-style baselines in all five paired episodes. Against the strongest
Wang CPD baseline, Budgeted ISAC improves CPD by +0.0846 and discovery by
+0.0880 while reducing collisions by 1940.2.

The longer N=100, 3000-slot results remain useful as auxiliary evidence for the
density-adaptive budgeted access rule, but they should not be the first result
shown if the paper is being aligned with the Chinese reference matrix. On the
primary N=100 transfer points:

| Beamwidth | Method | Discovery | Collisions | CPD | Lambda2 |
|---:|---|---:|---:|---:|---:|
| 10 | Wang ISAC | 0.681 | 1343.3 | 0.536 | 42.538 |
| 10 | Collision-aware ISAC | 0.767 | 4676.7 | 0.395 | 43.307 |
| 10 | Budgeted ISAC | 0.712 | 1347.0 | 0.560 | 38.871 |
| 15 | Wang ISAC | 0.884 | 14533.0 | 0.227 | 70.278 |
| 15 | Collision-aware ISAC | 0.873 | 32504.0 | 0.116 | 64.621 |
| 15 | Budgeted ISAC | 0.862 | 12602.0 | 0.245 | 61.981 |

Interpretation:

- At B=10, N=100, Budgeted ISAC improves collision-penalized discovery over Wang ISAC by about 4.5% relative and over Collision-aware ISAC by about 41.7% relative.
- At B=15, N=100, Budgeted ISAC improves collision-penalized discovery over Wang ISAC by about 8.3% relative and reduces collisions by about 13.3%.
- Budgeted ISAC is not uniformly best on raw discovery; it is an access-budget tradeoff, not a pure discovery maximizer.
- Current trained MARL variants are not yet paper-strong against rule experts. BC-MARL improves transfer discovery but causes excessive collisions, so it should be framed as diagnostic evidence and a motivation for distilling Budgeted ISAC into a constrained learned gate.

An independent paired rerun was then executed at the primary B=10, N=100 point using seed `2026071061`, 3000 testing slots, and 3 paired episodes. Its committed tables are under `06_analysis/paper_tables/marl/budgeted_isac_paired_rerun_b10_n100/`.

| Method | Discovery | Collisions | CPD | Lambda2 |
|---|---:|---:|---:|---:|
| Wang ISAC | 0.695 | 1574.3 | 0.530 | 42.031 |
| Collision-aware ISAC | 0.785 | 5300.7 | 0.384 | 46.437 |
| Budgeted ISAC | 0.724 | 1529.0 | 0.554 | 36.524 |
| Uniform random | 0.002 | 0.0 | 0.002 | -0.000 |

The rerun confirms the main tradeoff: Budgeted ISAC improves discovery by +0.0288 and CPD by +0.0244 over Wang ISAC on paired episodes, while reducing collisions by 3771.7 relative to non-budgeted Collision-aware ISAC and improving CPD by +0.1698. Its raw discovery remains below aggressive Collision-aware ISAC, which is why the defensible contribution is collision-budgeted ISAC-assisted neighbor discovery rather than unconstrained discovery maximization.

The new B=10 table-exchange sweep gives a useful negative result. At N=100 and
3000 slots, `improved_rl_isac_tables` and `trust_gated_isac_tables` both reach
about 0.744 raw discovery, but their collision counts are 9247.0 and 9516.4,
so CPD drops to 0.2608 and 0.2569. `budgeted_collision_aware_isac` reaches
lower raw discovery, 0.7022, but substantially higher CPD, 0.5563. This supports
the current mechanism direction: table exchange should be treated as shared
candidate information that still requires learned or rule-based access budgeting.

## Claim Boundary

Supported:

- ISAC occupancy/candidate feedback is necessary for tractable narrow-beam discovery in the tested 3D setting.
- A density-adaptive budgeted ISAC access rule improves collision-penalized discovery in the Wang-style single-RF MIMO-OTFS sensing matrix, and also improves CPD in auxiliary large-scale single-hop transfer tests.
- Naive shared-parameter MARL trained at N=10 does not automatically inherit the rule expert's collision control under N=100 transfer.
- Expert-guided BC-MARL raises raw discovery but currently needs explicit collision-budget learning.

Not yet supported:

- A learned MARL method that beats Budgeted ISAC or Wang ISAC on both raw discovery and collision-penalized discovery.
- A general claim that table exchange improves performance; current B=10
  evidence shows that even trust-gated table exchange is harmful on CPD without
  stronger access budgeting.
- A final TWC/TCOM-ready physical-layer waveform contribution. The current waveform abstraction is protocol-level ISAC service modeling, not a new waveform design.

## Remaining Work Before Full Paper Lock

1. Decide whether `main.tex` should pivot from the older Phase10 MARL/N=100 storyline to the Wang-style single-RF matrix as the primary result.
2. If this pivot is accepted, integrate `wang2025_focused_single_rf_insert.tex` and move MARL to a bounded diagnostic/extension section.
3. Run only one optional extension before paper lock: `rf_chains=3,6` under the same Wang-style matrix, and only if multi-RF comparison is needed for reviewer alignment.
4. Keep B=10 trust-gated table exchange as a negative mechanism boundary unless table exchange becomes a main contribution.
