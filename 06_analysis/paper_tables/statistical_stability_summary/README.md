# Statistical Stability Summary

Generated: 2026-07-04T22:25:58.718891+00:00

This directory contains compact mean/std/95% CI summaries extracted from archived `aggregate_metrics.csv` files for the manuscript and supplement.

Files:

- `statistical_stability_summary.csv`: normalized rows with evidence block, protocol, scenario context, `n_episodes`, metric means, metric standard deviations, and normal-approximation 95% confidence intervals.

Interpretation notes:

- The CI columns use `1.96 * std / sqrt(n_episodes)` and are intended as a concise reporting aid, not a replacement for paired statistical testing.
- Rows tagged `main` support current manuscript tables and figures.
- Rows tagged `main_boundary` support applicability-boundary claims.
- Rows tagged `supplement` are useful for reviewer-facing robustness evidence but should not replace the current main evidence chain without a separate promotion decision.
- Rows tagged `supplement_stress` are failure-boundary or extreme-regime checks; use them to bound claims rather than to advertise main performance.
- Rows tagged `supplement_sanity` are quick or one-seed checks; use them only to track trends while waiting for fuller multi-seed results.
- By default the script includes all archived node counts; pass `--node-count 100` to reproduce an N=100-only table.
- Total rows: 335
