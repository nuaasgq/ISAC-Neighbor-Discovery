## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-15
- Verification Status: ANALYZED
- Version Label: n10_b15_static_ideal_single_mixture_v1
- Statistical unit: trained seed with paired held-out scenarios

## Formal Result

Promotion decision: PASS

| Method | Final discovery | 50-slot | 100-slot | Delay | Curve AUC |
|---|---:|---:|---:|---:|---:|
| Wang2025 | 94.62% | 10.86% | 43.72% | 128.38 | 0.575 |
| ISAC candidate random | 100.00% | 11.47% | 50.95% | 98.41 | 0.675 |
| Direct-ISAC MAPPO | 89.08% | 32.28% | 64.86% | 105.19 | 0.652 |
| Residual-mask MAPPO | 99.97% | 37.64% | 82.18% | 66.40 | 0.782 |

## Attribution Boundary

Residual-mask versus Direct-ISAC isolates the local residual candidate-support mechanism under the same corrected MAPPO distribution. Residual-mask versus candidate-random isolates the learned policy increment within the same class of local residual candidate support.
Residual minus Direct AUC: 0.130 [0.095, 0.171].
Residual minus candidate-random AUC: 0.107 [0.098, 0.116].
Residual final-discovery gap versus candidate-random: -0.03 pp.
Minimum evaluated candidate-mask compliance: 100.00%.

## Eleven-Fallacy Scan

| Check | Status | Boundary |
|---|---|---|
| Causal attribution | Controlled | Direct and Residual differ only in local candidate support. |
| Metric substitution | Controlled | Final discovery and curve AUC are both reported. |
| Baseline fairness | Controlled | All methods use identical paired scenarios and horizons. |
| Information leakage | Controlled | Truth is used only by offline diagnostics. |
| Pseudoreplication | Controlled | Hierarchical bootstrap resamples seeds then scenarios. |
| Seed robustness | Limited | Three training seeds; no stronger population claim is made. |
| Censoring bias | Controlled | Delay and completion metrics remain horizon-censored. |
| Cherry picking | Controlled | Promotion criteria were fixed before formal completion. |
| Multiple comparisons | Descriptive | Intervals support mechanism analysis, not confirmatory multiplicity claims. |
| Extrapolation | Restricted | Result applies to static N=10 ideal-ISAC only. |
| Implementation fidelity | Audited | Double beam-mixture defect was fixed and legacy runs excluded. |
