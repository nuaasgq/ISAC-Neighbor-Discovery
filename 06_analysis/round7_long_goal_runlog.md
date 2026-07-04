# Round7 Long-Goal Run Log

Date: 2026-07-05
Deadline: 2026-07-05 11:00 Asia/Shanghai

## Purpose

This run tracks the long unattended work window for the ISAC-assisted narrow-beam UAV neighbor-discovery paper. The goal is to turn the existing simulation and manuscript assets into a defensible paper-data package, while avoiding overclaiming the current shared-policy optimizer as a complete neural MARL method.

## Active Work Streams

| Stream | Status | Artifact target | Notes |
|---|---|---|---|
| Long CEM training | complete | `05_simulation/results_raw/round7_long_cem_train_n10_b10_600slot` | Trained at `N=10`, `10 deg`, 600-slot horizon with 3 training seeds and 3 held-out seeds. Compact tables are archived under `06_analysis/paper_tables/round7_long_cem_training`. |
| Round7 transfer/evaluation jobs | running | `05_simulation/results_raw/round7_*` | Started N=100 multi-mobility, scale/beam grid, and error-profile evaluations using the round7 trained config. |
| Neural MARL probe | method probe completed | `05_simulation/run_actor_critic_imitation_probe.py` | Rule-assisted BC works in teacher-forced mode. After fixing the MARL env to expose ISAC piggyback belief updates, stochastic autonomous eval became nonzero but remains a method probe. |
| Experiment-matrix audit | running in parallel agent | analysis summary | Checks whether current figures cover the dimensions commonly expected in TWC/TCOM-style ISAC/beam-management papers. |
| Paper evidence audit | active | `07_paper/ieee_twc_isac_nd/main.tex` and result indexes | Added a compact `N=100` transfer table and clarified the dynamic-table empty-scan column. Main risk remains wording: supported claims must be separated from stress-regime and limitation claims. |

## Strong Current Evidence

- ISAC-assisted candidate-beam refinement is the dominant mechanism. Removing the candidate set collapses `N=100`, 10-degree discovery from about 0.3655 to about 0.0313.
- A one-slot delayed candidate-set variant keeps most of the ISAC gain while reducing collisions, which makes the mechanism more defensible as a data-link-layer protocol abstraction.
- Small-scale shared-policy tuning at `N=10`, `10 deg` transfers to `N=100` for 10-30 degree beams under both density-preserving and fixed-area scaling.
- The sensing range benefit saturates around the communication range in the current communication-neighbor-discovery model.
- Slot duration from 1 ms to 20 ms does not materially change the Gauss-Markov `N=100`, 10-degree conclusion.
- The current figure archive contains 228 PNG figures, all checked at 4:3 aspect ratio. The manuscript also includes an explicit training reward convergence curve.

## Morning Manuscript Checkpoint

- Added `Table~\ref{tab:n100_transfer}` to `07_paper/ieee_twc_isac_nd/main.tex`, summarizing `N=100` zero-shot transfer for 5, 10, 15, and 30 degree beams under density-preserving and fixed-area scaling.
- Clarified the small-scale dynamic comparison table header from `Empty` to `Empty (ISAC)` so the column is not misread as a multi-protocol comparison.
- Recompiled the IEEE LaTeX manuscript with `pdflatex` twice after the edit. The final log has no undefined references/citations and no overfull warnings; only normal underfull warnings remain.
- Generated round7 training figures in `06_analysis/paper_figures/round7_long_cem_training`; this includes reward, score, discovery, empty-scan, delay, collision, and connectivity curves.
- The round7 small-scale held-out score is lower than the earlier round2 candidate training result under a different 1200-slot setup, so round7 is not promoted to the main result unless the transfer/evaluation jobs show stronger large-scale robustness.

## Conservative Boundaries

- The current main learning result is CEM/shared-policy parameter optimization, not a full MAPPO, QMIX, or GNN-MARL algorithm.
- The actor-critic implementation is currently a probe. It should only become a paper result after multi-seed baseline comparisons and transfer tests.
- Round7 imitation probes show that the MARL environment must preserve ISAC piggyback belief updates. With the corrected ISAC-capable env, stochastic student evaluation reached about 0.647 mean discovery rate over five short episodes, but this is not yet a paper main result.
- The SkyOrbs comparison is a SkyOrbs-like baseline, not a strict reproduction of the full original protocol.
  See `06_analysis/skyorbs_baseline_reproduction_checklist.md` before upgrading this wording.
- 3-5 degree beams and abrupt mobility models remain stress regimes under the current finite horizon.
- Energy-normalized discovery is not yet available because a Joule-level action-energy model has not been added.
  See `06_analysis/energy_efficiency_extension_plan.md` for the required radio-state model and implementation touch points.

## Trigger Conditions Before 11:00

1. If the long CEM training finishes and improves held-out score or robustness, archive its compact tables and regenerate training figures.
2. If the long CEM training finishes but does not improve results, keep it as robustness evidence and do not replace the existing main result.
3. If the imitation/actor-critic probe returns a stable nonzero discovery curve, add it as a method-innovation appendix/result candidate.
4. If the neural probe remains weak, keep the manuscript wording conservative and describe full neural MARL as future work or an extension path.
5. Before any local commit, run `git status`, stage only compact scripts/docs/tables/figures, and keep raw `results_raw` outputs local.

## Version-Control Rule

All subsequent development, experiment scripts, compact analysis outputs, and manuscript changes should be synchronized through the repository. If remote GitHub push is unavailable, use local git commits as the authoritative version-management record until remote synchronization is restored.
