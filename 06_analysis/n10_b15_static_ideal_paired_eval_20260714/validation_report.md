## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-14
- Verification Status: ANALYZED
- Version Label: n10_b15_static_ideal_paired_v1

## Validation Report

- Source: `C:\Users\沈高青\Documents\通感一体化辅助的窄波束邻居发现方法研究\05_simulation\results_raw\n10_b15_static_ideal_paired_eval_3seed`
- Overall Confidence: CAUTION
- Design: three trained seeds, 50 paired held-out scenarios per seed

### Method Coverage

| Method | Episodes | Discovery rate | Seed SD | Mean delay | Curve AUC |
|---|---:|---:|---:|---:|---:|
| Uniform random | 150 | 22.86% | 0.53 pp | 264.72 | 0.118 |
| Wang2025 | 150 | 94.62% | 0.04 pp | 128.38 | 0.575 |
| ISAC candidate random | 150 | 100.00% | 0.00 pp | 98.41 | 0.675 |
| MAPPO without ISAC | 150 | 35.93% | 2.42 pp | 244.43 | 0.186 |
| Direct-ISAC MAPPO | 150 | 91.38% | 1.50 pp | 104.58 | 0.654 |
| ISAC + auxiliary MAPPO | 150 | 90.07% | 5.22 pp | 100.86 | 0.667 |
| Random role + learned beam | 150 | 82.46% | 1.57 pp | 136.02 | 0.549 |
| Learned role + random beam | 150 | 32.34% | 0.44 pp | 248.30 | 0.173 |

### Primary Interpretation

- Against Wang2025, Direct-ISAC MAPPO is 3.24 pp lower at 300 slots, but its discovery-curve AUC is 0.079 higher (descriptive 95% hierarchical-bootstrap interval [0.055, 0.108]). The supported claim is faster early discovery, not better final coverage.
- Against local residual-table candidate random, Direct-ISAC MAPPO is 8.62 pp lower at 300 slots. Its AUC difference is -0.021 with interval [-0.048, 0.017]; this experiment does not establish an advantage over the strong local candidate-pool rule.
- Replacing only the learned role with uniform TX/RX reduces discovery by 8.92 pp. Replacing only the learned beam with a uniform beam reduces discovery by 59.04 pp. Beam selection is the dominant learned component, while role learning provides a smaller incremental gain.
- The measurement auxiliary has AUC 0.667 versus 0.654 for Direct-ISAC MAPPO, but the paired AUC interval for Direct minus auxiliary [-0.030, 0.007] crosses zero and auxiliary seed SD is 5.22 pp. It is not a robust primary method yet.

### Statistical Boundary

Hierarchical bootstrap intervals resample training seeds and then paired scenarios within seed. Only three independently trained seeds are available, so intervals describe uncertainty but do not support strong asymptotic significance claims.

### Auxiliary Diagnostic

The diagnostic table reports the last-100 training discovery rate, held-out discovery rate, and their gap. A positive gap indicates training-to-test degradation, not failure to converge.

### Fallacy Scan

- Coverage: 11/11 statistical fallacy types checked
- Simpson/ecological/Berkson/collider: no subgroup or conditioning claim is made.
- Base-rate neglect: not applicable to the link-discovery ratio.
- Regression to mean/survivorship: all scheduled seeds and episodes are retained.
- Look-elsewhere/garden of forking paths: CAUTION; multiple metrics and checkpoints exist, so the three-seed final-checkpoint table remains primary.
- Correlation/causation and reverse causality: controlled simulation interventions support mechanism comparisons only within this static ideal environment.

### Reproducibility

- Method: deterministic replay contract with stochastic RNG seeds fixed per episode
- Verdict: PARTIALLY_REPRODUCIBLE until the full paired campaign is independently rerun
