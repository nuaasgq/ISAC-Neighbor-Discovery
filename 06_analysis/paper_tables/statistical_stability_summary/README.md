# Statistical Stability Summary

Generated: 2026-07-04T19:54:29.827195+00:00

This directory contains compact mean/std/95% CI summaries extracted from archived `aggregate_metrics.csv` files for the manuscript and supplement.

Files:

- `statistical_stability_summary.csv`: normalized rows with evidence block, protocol, scenario context, `n_episodes`, metric means, metric standard deviations, and normal-approximation 95% confidence intervals.

Interpretation notes:

- The CI columns use `1.96 * std / sqrt(n_episodes)` and are intended as a concise reporting aid, not a replacement for paired statistical testing.
- Rows tagged `main` support current manuscript tables and figures.
- Rows tagged `main_boundary` support applicability-boundary claims.
- Rows tagged `supplement` are useful for reviewer-facing robustness evidence but should not replace the current main evidence chain without a separate promotion decision.
- Total rows: 189
