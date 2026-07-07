# Phase10 Independent Re-Run Validation - 2026-07-07

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Verification Status: VERIFIED_PARTIAL
- Scope: gated contention actor, N=100, B=10, 3000-slot stochastic transfer

## Summary

- Original method row: `gated_contention_actor`, B=10.0 degrees.
- Independent re-run episodes: 10 with seed base 41260731--41260740.
- Relative tolerance: 5.00%.
- Verdict: all checked metrics are within tolerance.

## Metric Comparison

| Metric | Original mean | Re-run mean | Re-run std | Re-run CI95 | Rel. diff | Status |
|---|---:|---:|---:|---:|---:|---|
| discovery_rate | 0.301980 | 0.299475 | 0.013942 | 0.008641 | 0.83% | MATCH |
| collision_penalized_discovery_rate | 0.235312 | 0.232290 | 0.009248 | 0.005732 | 1.28% | MATCH |
| empty_scan_ratio | 0.065866 | 0.066129 | 0.002244 | 0.001391 | 0.40% | MATCH |
| lambda2 | 11.904328 | 12.061005 | 1.455609 | 0.902196 | 1.30% | MATCH |
| collision_count | 1403.300000 | 1432.100000 | 176.464318 | 109.373717 | 2.01% | MATCH |

## Interpretation

This is a partial reproducibility check, not a full re-run of the entire Phase10 campaign.
It verifies that the paper-facing default gated profile at B=10 remains within the predeclared tolerance under an independent stochastic seed range using the same checkpoint and transfer configuration.

## Generated Files

- `06_analysis/paper_tables/marl/p10_independent_rerun_gate31_b10_validation/rerun_metric_summary.csv`
- `06_analysis/paper_tables/marl/p10_independent_rerun_gate31_b10_validation/rerun_vs_original_comparison.csv`
- `06_analysis/paper_tables/marl/p10_independent_rerun_gate31_b10_validation/rerun_source_file_hashes.csv`
- `06_analysis/paper_tables/marl/p10_independent_rerun_gate31_b10_validation/manifest.json`
