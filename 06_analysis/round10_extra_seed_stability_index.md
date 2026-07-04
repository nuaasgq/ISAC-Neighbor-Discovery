# Round10 Extra-Seed Stability Check

Date: 2026-07-05

## Purpose

This is a narrow backup stability check, not a replacement for the main manuscript tables.
It reruns only the main large-scale Gauss-Markov setting with new scenario seeds:

- N=100
- Beamwidths: 10 and 15 degrees
- Density-preserving area scaling
- Single-hop communication/sensing support
- 600 slots
- Protocols: uniform random, SkyOrbs-like skip scan, learned no-ISAC, enhanced no-ISAC, enhanced ISAC
- Seeds: 20293731, 20294740, 20295749

Artifacts:

- Raw summary: `06_analysis/paper_tables/round10_n100_b10_b15_extra_seeds/per_episode_summary.csv`
- Aggregate summary: `06_analysis/paper_tables/round10_n100_b10_b15_extra_seeds/aggregate_metrics.csv`

## Aggregate Results

| Beam | Protocol | Discovery | Empty scan | Lambda2 | Collisions |
|---:|---|---:|---:|---:|---:|
| 10 | Uniform random | 0.0003 | 0.9041 | 0.0000 | 0.0 |
| 10 | SkyOrbs-like | 0.0007 | 0.9028 | 0.0000 | 0.0 |
| 10 | Learned no-ISAC | 0.0005 | 0.9028 | 0.0000 | 0.0 |
| 10 | Enhanced no-ISAC | 0.0008 | 0.9032 | 0.0000 | 0.0 |
| 10 | Enhanced ISAC | 0.1739 | 0.6741 | 3.7088 | 150.3 |
| 15 | Uniform random | 0.0026 | 0.8412 | 0.0000 | 0.0 |
| 15 | SkyOrbs-like | 0.0040 | 0.8343 | 0.0000 | 0.0 |
| 15 | Learned no-ISAC | 0.0024 | 0.8416 | 0.0000 | 0.0 |
| 15 | Enhanced no-ISAC | 0.0045 | 0.8376 | 0.0000 | 0.0 |
| 15 | Enhanced ISAC | 0.4181 | 0.5177 | 21.9957 | 2096.7 |

## Interpretation

The additional seeds preserve the qualitative ordering: the ISAC-assisted protocol remains far above all communication-only baselines for discovery rate, empty-scan reduction, and discovered-graph connectivity.
They also show nontrivial seed sensitivity, especially at 10-degree beams, where discovery drops from the main three-seed mean of 0.3655 to 0.1739 on these extra seeds.
This should be interpreted as a robustness-boundary signal, not as a failure of the main mechanism: no-ISAC baselines remain near zero, but the absolute ISAC gain varies with geometry and motion draw.

Recommended use:

- Use the main manuscript table as the primary archived multi-seed result.
- Use round10 as backup evidence if reviewers ask whether the main comparison was seed-specific.
- Do not use round10 to strengthen claims; use it to make the claim more honest: ISAC gives a large advantage over blind/no-ISAC baselines, but absolute discovery rate remains scenario-sensitive.
