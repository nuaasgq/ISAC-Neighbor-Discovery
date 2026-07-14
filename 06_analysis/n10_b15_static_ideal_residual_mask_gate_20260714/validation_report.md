## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-14
- Verification Status: ANALYZED
- Version Label: n10_b15_static_ideal_residual_mask_gate_v1

## Single-Seed Gate Report

- Training seed: 59262731
- Paired scenarios: 61262731--61262780 (50)
- Statistical unit: paired held-out scenario within one trained seed
- Promotion decision: PASS

| Method | Final discovery | 50-slot | 100-slot | Mean delay | Curve AUC |
|---|---:|---:|---:|---:|---:|
| Direct-ISAC MAPPO | 90.84% | 28.84% | 59.96% | 110.54 | 0.635 |
| ISAC candidate random | 100.00% | 12.31% | 50.67% | 98.22 | 0.676 |
| Wang2025 | 94.67% | 10.62% | 42.76% | 129.56 | 0.571 |
| Residual-mask MAPPO | 99.64% | 23.38% | 65.56% | 87.59 | 0.711 |

### Attribution Boundary

The residual candidate-random comparator uses the same local residual-table feasible set. Residual-mask MAPPO can claim an RL increment only from differences relative to that comparator; its absolute final discovery rate is not an RL contribution.

Residual-mask minus candidate-random AUC: 0.035 [0.020, 0.051].
Residual-mask minus Direct-ISAC AUC: 0.077 [0.062, 0.091].
Residual-mask final coverage versus candidate random: -0.36 pp.

### Statistical Boundary

Paired-scenario bootstrap intervals condition on one trained policy seed. They characterize scenario variation only and cannot establish training-seed robustness or publication-level significance.
The pilot is promoted only to decide whether a 1,000-episode, three-seed run is worth the compute.
