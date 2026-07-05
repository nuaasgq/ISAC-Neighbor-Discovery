# Submission Readiness Audit - 2026-07-05 Morning

This note records the readiness state after commit `3eae6dd` (`Add round11 stability evidence to supplement`) plus the current collision-aware MAC refinement work in progress.

## Current Manuscript Package

| Artifact | Status | Notes |
|---|---:|---|
| `07_paper/ieee_twc_isac_nd/main.tex` | Compiles | 8-page IEEEtran draft, texcount `3669+62+294`, 10 figures, 4 tables. |
| `07_paper/ieee_twc_isac_nd/supplement.tex` | Compiles | 9-page IEEEtran supplement with finite-horizon backup evidence and structured MARL probe figures. |
| `06_analysis/paper_figures/` | Verified | 358 PNG figures, all within 4:3 aspect tolerance. |
| `06_analysis/paper_tables/statistical_stability_summary/` | Verified | 345 normalized rows, mapped by evidence tier in the supplement. |
| `06_analysis/paper_tables/paired_delta_summary/` | Verified | 125 paired treatment-control delta rows with bootstrap descriptive CIs and seed-level sign counts. |
| `06_analysis/paper_tables/round10_n100_b10_b15_extra_seeds/` | Backup | Extra three-seed N=100/B=10/B=15 stability check; preserves qualitative ordering but shows absolute discovery-rate seed sensitivity. |
| `06_analysis/paper_tables/structured_marl_probe/` | Probe | Candidate-constrained shared actor-critic evidence, including clean no-ISAC environment baseline and residual-strength sweep. |

## Verification Snapshot

Commands completed after the last manuscript edits:

```powershell
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode supplement.tex
pdflatex -interaction=nonstopmode supplement.tex
Select-String -Path main.log,supplement.log -Pattern 'LaTeX Error|Emergency stop|Fatal error|Undefined control sequence|Citation.*undefined|Reference.*undefined|Overfull \\hbox|Overfull \\vbox'
python -m pytest 05_simulation\tests
python 06_analysis\scripts\build_paired_delta_summary.py
python 06_analysis\scripts\plot_round11_stability.py
python 06_analysis\scripts\plot_round12_collision_aware.py
```

Result after the latest full check: no log errors, no unresolved references or citations, no overfull warnings, `27 passed`, and generated paper-figure PNG files used for the latest evidence blocks are 4:3.

## Claim Boundaries Now Reflected in Text

| Claim area | Current wording discipline |
|---|---|
| Learning method | Use "shared-parameter protocol tuning" for the main method and "structured MARL probe" for the neural extension; do not claim MAPPO/QMIX/GNN superiority. |
| SkyOrbs comparison | Use "SkyOrbs-like" or "SkyOrbs-inspired"; the main text explicitly says it is not a strict reproduction. |
| Beamwidth coverage | Write "evaluated over 3--30 degrees"; 10--30 degrees are the main useful operating region, while 3--5 degrees are stress cases. |
| ISAC abstraction | Treat `Rs`, false alarms, missed detections, and angular-cell errors as protocol-level abstraction parameters, not a calibrated radar equation. |
| Distributed setting | Use distributed execution with own navigation-frame pose/position and no central scheduler or undiscovered-neighbor state. |
| Transfer | Bind transfer claims to evaluated single-hop, 600-slot regimes, strongest for smoother mobility and 10--30 degree beams. |

## Evidence Coverage

The main manuscript now carries the concise evidence chain: training evolution, dynamic protocol comparison, N=100 transfer, area scaling, mobility boundary, range sensitivity, error robustness, and mechanism ablation.

The supplement now carries the reviewer-facing evidence chain: coverage matrix, training reward and score curves, N=10--100 scale/beam heatmap, N=100 density/fixed scaling, range and slot-duration sensitivity, 3-degree full-baseline stress, full-baseline mobility checks, B=15 error profiles, paired treatment-control deltas, statistical evidence-tier index, finite-horizon round10 trajectories, focused round11 five-seed paired stability, round12 collision-aware MAC refinement, and structured MARL probe results. Round10 is backup trajectory evidence; round11 is the focused paired stability evidence for N=100/B=10/B=15; round12 is a mechanism-refinement probe for the collision boundary.

## Remaining Risks Before External Submission

1. The main learned method is still CEM/shared-policy search; the neural actor-critic branch is a structured MARL probe, not the strongest evidence. This is acceptable only if framed as a protocol paper with a learning-assisted extension, not as a pure MARL contribution.
2. Collision-aware efficiency is only partly addressed by the round12 role-control probe; dense 15/30-degree cases still need a full MAC and energy model.
3. The SkyOrbs-like baseline is only an inspired communication-only baseline. A complete SkyOrbs reproduction remains future work.
4. The physical-layer ISAC service is abstracted. A TWC/TCOM reviewer may still ask for a stronger mapping from sensing parameters to waveform/estimator assumptions.
5. The 3-degree and 5-degree cases are not success regimes under the current 600-slot horizon.
6. Extra round10 seeds preserve the proposed-vs-no-ISAC ordering but show that absolute N=100/B=10 discovery can be scenario-seed sensitive; round11 strengthens the paired raw-discovery ordering while exposing the B=15 collision-penalized boundary, and round12 shows that a local role-control refinement can mitigate that boundary.
7. The best structured stochastic actor reduces empty scanning and improves deterministic nonzero behavior, but still trails the flat stochastic actor in raw discovery rate.

## Next High-Value Work

1. Add a concise "why this is TWC/TCOM and not pure PHY" paragraph if reviewer-style checks still find ambiguity.
2. Strengthen the method-name consistency around `ITAP-ND` and `L-ITAP-ND` in captions and figure legends.
3. Prepare a one-page response-style evidence map for likely reviewer questions: SkyOrbs, MARL naming, Rs/Rc, 3-degree beams, collisions, and single-hop scope.
4. If time remains before 11:00, run one more compile/test check after any manuscript edits and commit the refreshed research records.
