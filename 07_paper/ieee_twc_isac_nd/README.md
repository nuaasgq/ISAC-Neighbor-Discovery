# IEEE TWC/TCOM Draft Workspace

This folder contains the working IEEE journal draft for the ISAC-assisted UAV-UAV narrow-beam neighbor discovery study.

Template basis:

- IEEE official author template selector: https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/ieee-article-templates/
- Draft class: `\documentclass[journal]{IEEEtran}`.

Current status:

- `main.tex` is a conservative technical skeleton, not a submission-ready manuscript.
- `references.bib` only includes references that were locally noted or web-verified during the workflow.
- Round2 transfer figures and training curves can be included immediately.
- Round3 robustness, range, and ablation figures should be regenerated after all long-running jobs finish.

Build command, if a LaTeX toolchain is installed:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

