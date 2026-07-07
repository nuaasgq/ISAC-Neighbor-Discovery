# IEEE TWC/TCOM Draft Workspace

This folder contains the working IEEE journal draft for the ISAC-assisted UAV-UAV narrow-beam neighbor discovery study.

Template basis:

- IEEE official author template selector: https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/ieee-article-templates/
- Draft class: `\documentclass[journal]{IEEEtran}`.

Current status:

- `main.tex` is a conservative IEEE journal draft aligned with the Phase10 final MARL+ISAC evidence line; it is not yet a final submission manuscript.
- `supplement.tex` is a compact supplementary-material draft for coverage, SkyOrbs-like baseline-scope notes, stress, mobility-baseline, error-profile, PHY-to-protocol parameter mapping, paired-stability, collision-aware MAC-refinement, and assumed radio-state energy-accounting evidence.
- `references.bib` only includes references that were locally noted or web-verified during the workflow.
- The current draft compiles to a 9-page PDF with `pdflatex`; the final checked log has no undefined references/citations and no overfull warnings.
- The supplement compiles to a 13-page PDF with `pdflatex`; the final checked log has no undefined references and no overfull warnings.
- The final main MARL evidence trains at `N=10`, 10-degree beams, and 300 slots per episode, then evaluates zero-shot transfer at `N=100`, 3000 slots, and 10-/15-degree beams.
- The 3- and 5-degree results are treated as extremely narrow stress-boundary regimes; the archived 30-degree sweep is retained only as historical boundary evidence outside the final main comparison.
- Main figures are selected in `../../06_analysis/manuscript_figure_selection.md`; current figure/table provenance is audited in `../../06_analysis/figure_table_provenance_audit_20260707.md`.
- The current closeout workflow and repository synchronization rule are recorded in `../../00_workflow/current_execution_workflow.md`.
- A fresh LaTeX figure audit checked 51 referenced PNG files for presence and approximate 4:3 aspect compliance; it supersedes the earlier `../../06_analysis/paper_figure_integrity_audit_20260705.md`.
- A manuscript-facing SHA256 artifact manifest is available at `../../06_analysis/manuscript_artifact_manifest_20260707.csv` and `../../06_analysis/manuscript_artifact_manifest_20260707.json`.
- Phase10 statistical interpretation and fallacy-scan notes are recorded in `../../06_analysis/phase10_statistical_validation_report_20260707.md`.
- Final Phase10 method-to-manifest trace files are in `../../06_analysis/paper_tables/marl/p10_final_method_manifest_trace/`.
- The research-goal coverage audit is in `../../06_analysis/research_goal_coverage_audit_20260707.md` with machine-readable CSVs under `../../06_analysis/paper_tables/research_goal_coverage_audit/`.
- Submission-readiness risks and edits are tracked in `../../06_analysis/submission_readiness_review_20260707.md`.
- Mean/std/95% CI summaries for archived multi-seed sweeps are in `../../06_analysis/paper_tables/statistical_stability_summary/`.
- Experiment coverage against the requested variables is tracked in `../../06_analysis/experiment_coverage_matrix_20260705.md`.
- Round7 outputs are treated as robustness and audit support unless explicitly promoted in `../../06_analysis/round7_results_index.md`.

Build command, if a LaTeX toolchain is installed:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
pdflatex supplement.tex
pdflatex supplement.tex
```
