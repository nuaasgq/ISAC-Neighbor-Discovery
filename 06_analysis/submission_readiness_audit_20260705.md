# Submission Readiness Audit - 2026-07-05 Morning

This note records the post-commit readiness state after commit `6fa250c` (`Add protocol definitions and submission evidence`).

## Current Manuscript Package

| Artifact | Status | Notes |
|---|---:|---|
| `07_paper/ieee_twc_isac_nd/main.tex` | Compiles | 8-page IEEEtran draft, about 3860 texcount total words/caption words, 10 figures, 4 tables. |
| `07_paper/ieee_twc_isac_nd/supplement.tex` | Compiles | 7-page IEEEtran supplement, about 730 texcount total words/caption words, 8 figure blocks, 5 tables. |
| `06_analysis/paper_figures/` | Verified | 358 PNG figures, all within 4:3 aspect tolerance. |
| `06_analysis/paper_tables/statistical_stability_summary/` | Verified | 335 normalized rows, mapped by evidence tier in the supplement. |

## Verification Snapshot

Commands completed after the last manuscript edits:

```powershell
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode supplement.tex
pdflatex -interaction=nonstopmode supplement.tex
Select-String -Path main.log,supplement.log -Pattern 'LaTeX Error|Emergency stop|Fatal error|Undefined control sequence|Citation.*undefined|Reference.*undefined|Overfull \\hbox'
python -m pytest 05_simulation\tests
```

Result: no log errors, no unresolved references or citations, no overfull warnings, `25 passed`, and all 358 paper-figure PNG files pass the 4:3 aspect-ratio check.

## Claim Boundaries Now Reflected in Text

| Claim area | Current wording discipline |
|---|---|
| Learning method | Use "shared-parameter policy optimization" and "learned/shared policy"; do not call the current evidence full MAPPO/QMIX/GNN MARL. |
| SkyOrbs comparison | Use "SkyOrbs-like" or "SkyOrbs-inspired"; the main text explicitly says it is not a strict reproduction. |
| Beamwidth coverage | Write "evaluated over 3--30 degrees"; 10--30 degrees are the main useful operating region, while 3--5 degrees are stress cases. |
| ISAC abstraction | Treat `Rs`, false alarms, missed detections, and angular-cell errors as protocol-level abstraction parameters, not a calibrated radar equation. |
| Distributed setting | Use distributed execution with own navigation-frame pose/position and no central scheduler or undiscovered-neighbor state. |
| Transfer | Bind transfer claims to evaluated single-hop, 600-slot regimes, strongest for smoother mobility and 10--30 degree beams. |

## Evidence Coverage

The main manuscript now carries the concise evidence chain: training evolution, dynamic protocol comparison, N=100 transfer, area scaling, mobility boundary, range sensitivity, error robustness, and mechanism ablation.

The supplement now carries the reviewer-facing evidence chain: coverage matrix, training reward and score curves, N=10--100 scale/beam heatmap, N=100 density/fixed scaling, range and slot-duration sensitivity, 3-degree full-baseline stress, full-baseline mobility checks, B=15 error profiles, and statistical evidence-tier index.

## Remaining Risks Before External Submission

1. The current learned method is still CEM/shared-policy search rather than a modern neural MARL algorithm. This is acceptable only if framed as a protocol paper with a learned policy-search variant, not as a pure MARL contribution.
2. Collision-aware efficiency is not yet fully optimized; dense 15/30-degree cases improve raw discovery but can create many collisions.
3. The SkyOrbs-like baseline is only an inspired communication-only baseline. A complete SkyOrbs reproduction remains future work.
4. The physical-layer ISAC service is abstracted. A TWC/TCOM reviewer may still ask for a stronger mapping from sensing parameters to waveform/estimator assumptions.
5. The 3-degree and 5-degree cases are not success regimes under the current 600-slot horizon.

## Next High-Value Work

1. Add a concise "why this is TWC/TCOM and not pure PHY" paragraph if reviewer-style checks still find ambiguity.
2. Strengthen the method-name consistency around `ITAP-ND` and `L-ITAP-ND` in captions and figure legends.
3. Prepare a one-page response-style evidence map for likely reviewer questions: SkyOrbs, MARL naming, Rs/Rc, 3-degree beams, collisions, and single-hop scope.
4. If time remains before 11:00, create a compact cover-letter style novelty summary and a figure-to-claim mapping table.
