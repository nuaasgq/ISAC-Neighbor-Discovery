# MARL Phase-1 Findings and Next Experiment Gate

## Active Rule

- Training horizon is fixed at 300 slots for the main MARL policy.
- Transfer/testing horizons are 300, 1200, and 3000 slots.
- The trained source setting is N=10 and 10 deg beams.
- The first transfer matrix covers N in {10, 20, 50} and beamwidth in {5, 10, 15, 30} deg.
- Raw training/evaluation outputs remain under `05_simulation/results_raw/`; paper tables and plots are regenerated under `06_analysis/paper_tables/marl/` and `06_analysis/paper_figures/marl/`.

## Completed Implementation Gate

- Real slot-level MARL is implemented in `05_simulation/run_marl_training.py`.
- `isac_mappo` uses ISAC candidate mask, candidate score, topology deficit, and rule residual inputs.
- `mappo` disables ISAC features and uses the no-ISAC environment protocol.
- `scalegraph_beam` is now available as a scale-invariant beam-set actor option via `--network scalegraph_beam`.
- The transfer aggregator separates `train_algorithm` and `train_network`, so shared and scalegraph results will not be averaged together.

## Final Phase-1 Evidence

The complete phase-1 matrix finished with 72 transfer evaluation runs:

- 2 algorithms: `isac_mappo` and no-ISAC `mappo`.
- 3 test horizons: 300, 1200, and 3000 slots.
- 3 node counts: N=10, 20, and 50.
- 4 beamwidths: 5, 10, 15, and 30 deg.

The final tables and figures are regenerated at:

- `06_analysis/paper_tables/marl/phase1_transfer_final/`
- `06_analysis/paper_figures/marl/phase1_transfer_final/`
- `06_analysis/paper_tables/marl/phase1_train_curves_final/`
- `06_analysis/paper_figures/marl/phase1_train_curves_final/`

The complete ISAC-MAPPO 3000-slot transfer rows show strong small-to-medium scale transfer and a clear large-scale bottleneck.

| N | Beamwidth | Discovery rate | Lambda2 | Empty-scan ratio | Collision count |
|---:|---:|---:|---:|---:|---:|
| 10 | 5 | 0.963 | 7.529 | 0.749 | 144 |
| 10 | 10 | 0.970 | 8.000 | 0.519 | 190 |
| 10 | 15 | 0.978 | 9.000 | 0.368 | 277 |
| 10 | 30 | 0.993 | 9.333 | 0.172 | 680 |
| 20 | 5 | 0.732 | 9.249 | 0.566 | 143 |
| 20 | 10 | 0.791 | 10.686 | 0.248 | 400 |
| 20 | 15 | 0.912 | 14.927 | 0.112 | 482 |
| 20 | 30 | 0.809 | 12.138 | 0.028 | 154 |
| 50 | 5 | 0.416 | 10.114 | 0.247 | 270 |
| 50 | 10 | 0.576 | 17.069 | 0.054 | 909 |
| 50 | 15 | 0.602 | 17.055 | 0.019 | 209 |
| 50 | 30 | 0.511 | 14.141 | 0.006 | 604 |

The no-ISAC MAPPO 3000-slot rows stay near zero even with long testing:

| N | Beamwidth | Discovery rate | Lambda2 | Empty-scan ratio |
|---:|---:|---:|---:|---:|
| 10 | 5 | 0.000 | 0.000 | 0.997 |
| 10 | 10 | 0.000 | 0.000 | 0.986 |
| 10 | 15 | 0.000 | 0.000 | 0.971 |
| 10 | 30 | 0.059 | 0.000 | 0.902 |
| 20 | 5 | 0.002 | 0.000 | 0.993 |
| 20 | 10 | 0.000 | 0.000 | 0.972 |
| 20 | 15 | 0.002 | 0.000 | 0.946 |
| 20 | 30 | 0.079 | 0.000 | 0.836 |
| 50 | 5 | 0.000 | 0.000 | 0.982 |
| 50 | 10 | 0.000 | 0.000 | 0.940 |
| 50 | 15 | 0.005 | 0.000 | 0.888 |
| 50 | 30 | 0.080 | 0.299 | 0.733 |

Interpretation:

- ISAC candidate-space reduction is effective: the policy trained at N=10, 10 deg transfers to N=20 and N=50 without retraining.
- Beamwidth transfer is not monotonic. Wider beams reduce empty scans, but collision and role coordination can dominate at dense settings.
- The N=50, 15 deg case is currently the strongest large-scale row among completed 3000-slot ISAC-MAPPO results.
- The no-ISAC MAPPO rows stay near zero even at 3000 slots, supporting the necessity of ISAC assistance under high-dimensional blind search.

## Risks Before Paper Use

- Phase-1 is still single training seed and 3 eval episodes per transfer point.
- The current method should be described as MAPPO-style CTDE, not a full canonical MAPPO implementation.
- Deterministic and stochastic evaluation must remain separated because deterministic argmax can collapse.
- The current `shared` network is not yet enough as a network-structure innovation; `scalegraph_beam` was added to support the next comparison.
- Collision-aware reward/topology-aware reward is not yet implemented.

## Next Gate

1. Use `shared + legacy + ISAC` as the strongest phase-1 discovery-rate baseline.
2. Use no-ISAC MAPPO as a hard failure baseline, not as a competitive method.
3. Treat `scalegraph_beam` and `collision_topology` as phase-2 method innovations; early probes show collision-efficiency gains but not discovery-rate gains.
4. Add N=100 and 3 deg stress tests only after the N=10/20/50 matrix is stable.
5. Add sensing/communication range and ISAC error sweeps after the main MARL comparison is defensible.
