# N=100 Mobility Full-Baseline Table

Generated: 2026-07-04T20:00:08.042136+00:00

This table merges:

- `round7_n100_multimobility_600slot`: uniform random, improved no-ISAC, one-slot delayed ISAC, and full ISAC results.
- `round8_n100_multimobility_missing_baselines_600slot`: SkyOrbs-like skip scan and vanilla learned policy without ISAC.

The output is intended for reviewer-facing baseline-completeness checks under the mobility-transfer setting. It should be interpreted together with the mobility-boundary wording in the manuscript: Gauss-Markov and random-walk regimes are stronger for the proposed method, while random-direction and random-waypoint remain stress regimes.

Files:

- `combined_aggregate_metrics.csv`: compact merged mean/std table.

Total rows: 48
