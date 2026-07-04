# Round7 Long-Goal Run Log

Date: 2026-07-05
Deadline: 2026-07-05 11:00 Asia/Shanghai

## Purpose

This run tracks the long unattended work window for the ISAC-assisted narrow-beam UAV neighbor-discovery paper. The goal is to turn the existing simulation and manuscript assets into a defensible paper-data package, while avoiding overclaiming the current shared-policy optimizer as a complete neural MARL method.

## Active Work Streams

| Stream | Status | Artifact target | Notes |
|---|---|---|---|
| Long CEM training | complete | `05_simulation/results_raw/round7_long_cem_train_n10_b10_600slot` | Trained at `N=10`, `10 deg`, 600-slot horizon with 3 training seeds and 3 held-out seeds. Compact tables are archived under `06_analysis/paper_tables/round7_long_cem_training`. |
| Round7 transfer/evaluation jobs | complete | `05_simulation/results_raw/round7_*` | Scale/beam grid, N=100 multi-mobility, and error-profile evaluation are complete and archived. |
| Round8 targeted follow-ups | complete | `05_simulation/results_raw/round8_*` | The N=100 mobility missing-baseline job and B=15 error-profile job are complete and archived. |
| Neural MARL probe | method probe completed | `05_simulation/run_actor_critic_imitation_probe.py` | Rule-assisted BC works in teacher-forced mode. After fixing the MARL env to expose ISAC piggyback belief updates, stochastic autonomous eval became nonzero but remains a method probe. |
| Experiment-matrix audit | complete | analysis summary | Confirmed that the main evidence chain covers the minimum TWC/TCOM-style variables if claims remain bounded. |
| Paper evidence audit | complete | `07_paper/ieee_twc_isac_nd/main.tex` and result indexes | Added compact N=100 transfer/baseline tables, seed-reporting text, mobility-baseline checks, and B=15 error robustness notes. |

## Strong Current Evidence

- ISAC-assisted candidate-beam refinement is the dominant mechanism. Removing the candidate set collapses `N=100`, 10-degree discovery from about 0.3655 to about 0.0313.
- A one-slot delayed candidate-set variant keeps most of the ISAC gain while reducing collisions, which makes the mechanism more defensible as a data-link-layer protocol abstraction.
- Small-scale shared-policy tuning at `N=10`, `10 deg` transfers to `N=100` for 10-30 degree beams under both density-preserving and fixed-area scaling.
- The sensing range benefit saturates around the communication range in the current communication-neighbor-discovery model.
- Slot duration from 1 ms to 20 ms does not materially change the Gauss-Markov `N=100`, 10-degree conclusion.
- The current figure archive contains 346 PNG figures, all checked at 4:3 aspect ratio. The manuscript also includes an explicit training-score evolution curve, and all 10 figures referenced by the IEEE LaTeX draft exist on disk.

## Morning Manuscript Checkpoint

- Added `Table~\ref{tab:n100_transfer}` to `07_paper/ieee_twc_isac_nd/main.tex`, summarizing `N=100` zero-shot transfer for 5, 10, 15, and 30 degree beams under density-preserving and fixed-area scaling.
- Added an explicit `N=100`, 10-degree baseline table covering uniform random, SkyOrbs-like, RL without ISAC, improved RL without ISAC, and improved RL with ISAC.
- Clarified the small-scale dynamic comparison table header from `Empty` to `Empty (ISAC)` so the column is not misread as a multi-protocol comparison.
- Recompiled the IEEE LaTeX manuscript with `pdflatex` twice after the edit. The final log has no undefined references/citations and no overfull warnings; only normal underfull warnings remain.
- Generated round7 training figures in `06_analysis/paper_figures/round7_long_cem_training`; this includes reward, score, discovery, empty-scan, delay, collision, and connectivity curves.
- Rechecked the full `06_analysis/paper_figures` archive after round8 additions: 346 PNG files, zero aspect-ratio violations under the 4:3 tolerance check.
- The round7 small-scale held-out score is lower than the earlier round2 candidate training result under a different 1200-slot setup, so round7 is not promoted to the main result unless the transfer/evaluation jobs show stronger large-scale robustness.
- The completed round7 scale/beam grid gives a useful caution: at `N=100`, raw discovery peaks around 15 degrees, but collision-penalized discovery is strongest around 10 degrees. This supports reporting collision-aware efficiency alongside raw discovery.
- The completed round7 mobility sweep slightly improves abrupt-mobility stress cases but does not remove the applicability boundary: random-direction and random-waypoint remain weak compared with Gauss-Markov and random walk.
- A quick one-seed round7 error-profile backup completed and was archived under `06_analysis/paper_tables/round7_error_profiles_quick` with one 4:3 figure in `06_analysis/paper_figures/round7_error_profiles_quick`. It is superseded by the full three-seed round7 error-profile archive for quantitative reporting.
- The full three-seed round7 error-profile job completed and was archived under `06_analysis/paper_tables/round7_error_profiles_light`, with figures in `06_analysis/paper_figures/round7_error_profiles_light`. It confirms the moderate-error robustness trend without replacing the existing round3 main evidence chain.
- The round8 missing-baseline mobility job adds SkyOrbs-like and vanilla RL without ISAC for `N=100`, 10/15-degree beams, and four mobility models. The merged full-baseline table is archived under `06_analysis/paper_tables/round8_n100_multimobility_full_baseline`.
- A quick B=15 error-profile fallback completed for Gauss-Markov and random-walk mobility and is archived under `06_analysis/paper_tables/round8_error_profiles_b15_gm_rw_quick`; it is superseded by the full three-seed archive under `06_analysis/paper_tables/round8_error_profiles_b15_gm_rw_600slot`.

## Conservative Boundaries

- The current main learning result is CEM/shared-policy parameter optimization, not a full MAPPO, QMIX, or GNN-MARL algorithm.
- The actor-critic implementation is currently a probe. It should only become a paper result after multi-seed baseline comparisons and transfer tests.
- Round7 imitation probes show that the MARL environment must preserve ISAC piggyback belief updates. With the corrected ISAC-capable env, stochastic student evaluation reached about 0.647 mean discovery rate over five short episodes, but this is not yet a paper main result.
- The SkyOrbs comparison is a SkyOrbs-like baseline, not a strict reproduction of the full original protocol.
  See `06_analysis/skyorbs_baseline_reproduction_checklist.md` before upgrading this wording.
- 3-5 degree beams and abrupt mobility models remain stress regimes under the current finite horizon.
- Energy-normalized discovery is not yet available because a Joule-level action-energy model has not been added.
  See `06_analysis/energy_efficiency_extension_plan.md` for the required radio-state model and implementation touch points.

## Remaining Work Before 11:00

1. Continue polishing the IEEE manuscript around related work, limitations, and result-story continuity.
2. Keep the current learning claim conservative: shared-parameter policy optimization is the main result; neural MARL remains a probe.
3. Do not launch additional heavy sweeps unless a specific paper gap appears. The current evidence is sufficient for a bounded paper draft.
4. If adding tables or figures, prefer supplement/readiness artifacts over overcrowding the 7-page main draft.
5. Before any local commit, run `git status`, stage only compact scripts/docs/tables/figures, and keep raw `results_raw` outputs local.

## Version-Control Rule

All subsequent development, experiment scripts, compact analysis outputs, and manuscript changes should be synchronized through the repository. If remote GitHub push is unavailable, use local git commits as the authoritative version-management record until remote synchronization is restored.
