# IEEE TWC/TCOM Draft Workspace

This folder contains the working IEEE journal draft for the ISAC-assisted UAV-UAV narrow-beam neighbor discovery study.

Template basis:

- IEEE official author template selector: https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/ieee-article-templates/
- Draft class: `\documentclass[journal]{IEEEtran}`.

Current status:

- `main.tex` is a conservative IEEE journal draft with explicit claim boundaries; it is not yet a final submission manuscript.
- `supplement.tex` is a compact supplementary-material draft for coverage, stress, mobility-baseline, error-profile, paired-stability, and collision-aware MAC-refinement evidence.
- `references.bib` only includes references that were locally noted or web-verified during the workflow.
- The current draft compiles to an 8-page PDF with `pdflatex`; the final checked log has no undefined references/citations and no overfull warnings.
- The supplement compiles to a 9-page PDF with `pdflatex`; the final checked log has no undefined references and no overfull warnings.
- Main figures are selected in `../../06_analysis/manuscript_figure_selection.md`; all figure paths referenced by `main.tex` have been checked on disk.
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
