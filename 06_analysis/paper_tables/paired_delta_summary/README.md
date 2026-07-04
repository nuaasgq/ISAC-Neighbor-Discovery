# Paired Delta Summary

Generated: 2026-07-04T22:44:21.228556+00:00

This directory contains paired treatment-control deltas computed from archived `per_episode_summary.csv` files. Pairing uses identical scenario seeds and scenario parameters, so each delta compares protocols on the same simulated geometry/mobility draw.

Files:

- `paired_delta_summary.csv`: one row per comparison block and metric, including treatment/control means, mean paired delta, bootstrap percentile 95% CI over paired deltas, and seed-level sign counts.

Interpretation notes:

- Bootstrap intervals are descriptive because several key blocks have only three paired seeds.
- Positive deltas are beneficial for discovery rate, collision-penalized discovery, and lambda2; negative deltas are beneficial for empty-scan ratio, delay, and collision count.
- The SkyOrbs-like comparison is a deterministic 3-D skip-scan reference under this simulator's information boundary, not a strict reproduction of the full SkyOrbs protocol.

Rows: 115

Metric rows: {"collision_count": 23, "collision_penalized_discovery_rate": 19, "discovery_rate": 23, "empty_scan_ratio": 23, "lambda2": 23, "mean_discovery_delay": 4}
