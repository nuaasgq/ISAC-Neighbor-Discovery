# Phase10 Learned-Component Ablation - 2026-07-07

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Verification Status: ANALYZED
- Scope: N=100, B=10, 3000-slot stochastic transfer, 3 paired seed episodes

## Summary

- Trained full CPD mean: 0.229741; random-weight CPD mean: 0.159857; zero-weight/rule-only CPD mean: 0.146369.
- Trained full reduces mean collisions by 81.4% vs random weights and 85.2% vs zero-weight/rule-only.
- Raw discovery and lambda2 are not always highest for the trained checkpoint; random/zero-weight rule-dominated policies discover more links by transmitting aggressively, but with much higher collision burden.
- Disabling the hard candidate mask improves CPD in this 3-episode probe but also raises empty-scan ratio sharply, so it is an efficiency tradeoff signal rather than a ready replacement for the final policy.

## Metric Means

| Variant | Discovery | CPD | Collisions | Collisions/discovery | Empty scan | Lambda2 |
|---|---:|---:|---:|---:|---:|---:|
| Trained full | 0.2958 | 0.2297 | 1415.3 | 0.962 | 0.0661 | 10.779 |
| Random weights | 0.4054 | 0.1599 | 7624.7 | 3.791 | 0.0447 | 18.714 |
| Zero weights | 0.4265 | 0.1464 | 9552.3 | 4.517 | 0.0438 | 21.261 |
| No rule residual | 0.2574 | 0.2321 | 536.7 | 0.420 | 0.0947 | 9.214 |
| No candidate mask | 0.2998 | 0.2826 | 299.3 | 0.201 | 0.8173 | 12.782 |

## Interpretation Boundary

This ablation separates learned weights from structured ISAC/rule priors but does not prove a universally dominant learned policy.
The defensible claim is narrower: learned weights materially suppress collisions relative to random/zero-weight policies under the same ISAC features, while rule residuals and candidate-mask design control the discovery/collision/empty-scan tradeoff.

## Generated Figures

- `06_analysis/figures/marl/p10_learned_component_ablation_b10_3ep/learned_ablation_b10_discovery_efficiency.png`
- `06_analysis/figures/marl/p10_learned_component_ablation_b10_3ep/learned_ablation_b10_collision_tradeoff.png`

## Generated Tables

- `06_analysis/paper_tables/marl/p10_learned_component_ablation_b10_3ep/ablation_metric_summary.csv`
- `06_analysis/paper_tables/marl/p10_learned_component_ablation_b10_3ep/ablation_vs_trained_full.csv`
- `06_analysis/paper_tables/marl/p10_learned_component_ablation_b10_3ep/ablation_run_index.csv`
- `06_analysis/paper_tables/marl/p10_learned_component_ablation_b10_3ep/ablation_source_file_hashes.csv`
- `06_analysis/paper_tables/marl/p10_learned_component_ablation_b10_3ep/manifest.json`
