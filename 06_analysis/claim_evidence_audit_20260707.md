# Claim-Evidence Audit for Phase10 Manuscript Alignment

Date: 2026-07-07

Scope: `07_paper/ieee_twc_isac_nd/main.tex` and `supplement.tex` after the Phase10 MARL result update.

Primary evidence files:

- `06_analysis/paper_tables/marl/p10_final_b10_b15_method_comparison_with_v4/marl_method_comparison.csv`
- `06_analysis/paper_tables/marl/p10_gate_family_v2_v3_v4_tradeoff_comparison/seed_tradeoff_core_metrics.csv`
- `06_analysis/paper_tables/paired_delta_summary/paired_delta_summary.csv`
- `06_analysis/paper_tables/round5_mobility_transfer/per_episode_summary.csv`
- `06_analysis/paper_tables/round4_delay_ablation/per_episode_summary.csv`

## Audit Summary

| Claim area | Verdict | Action |
|---|---|---|
| Small-to-large MARL transfer setting | Supported | Manuscript states `N=10`, 10-degree, 300-slot training and `N=100`, 3000-slot transfer. |
| ISAC improves discovery over communication-only baselines | Supported with wording constraint | Corrected the abstract from a universal "more than two orders of magnitude" claim to explicit minimum ratios: 110x at 10 degrees and 22x at 15 degrees against random/directional baselines. |
| ISAC reduces empty scans | Supported with scope constraint | Corrected broad "ISAC-aware policies" wording to the main contention/default-gated profiles because the adaptive 10-degree profile has a higher empty-scan ratio. |
| Gate family exposes a collision/topology frontier | Supported | Final table supports default, adaptive, topology, and balanced gate tradeoffs. |
| Ungated actor maximizes raw discovery | Overstated before correction | Corrected because topology gate slightly exceeds raw discovery at 10 degrees, although ungated is highest at 15 degrees and has high topology connectivity. |
| Supplement baseline coverage | Inconsistent before correction | Updated legacy Phase5 baseline wording to the final Phase10 baseline/gate-family set. |
| Historical neural MARL probe | Supported as traceability only | Updated supersession statement from Phase5 to Phase10. |

## Quantitative Checks

### Discovery Improvement Ratios

For the MARL+ISAC contention actor:

| Beam | Baseline | Baseline discovery | Actor discovery | Ratio |
|---|---:|---:|---:|---:|
| 10 deg | Uniform random | 0.002788 | 0.342909 | 123.0x |
| 10 deg | SkyOrbs-like | 0.003091 | 0.342909 | 110.9x |
| 10 deg | MAPPO no ISAC | 0.000485 | 0.342909 | 707.3x |
| 10 deg | Contention no ISAC | 0.000768 | 0.342909 | 446.7x |
| 15 deg | Uniform random | 0.012929 | 0.423293 | 32.7x |
| 15 deg | SkyOrbs-like | 0.018970 | 0.423293 | 22.3x |
| 15 deg | MAPPO no ISAC | 0.002222 | 0.423293 | 190.5x |
| 15 deg | Contention no ISAC | 0.004343 | 0.423293 | 97.5x |

Conclusion: the two-order-magnitude statement is valid for the 10-degree comparisons and for the 15-degree MAPPO no-ISAC comparison, but not for all 15-degree communication-only baselines.

### Gate-Family Tradeoff

| Beam | Best CPD profile | CPD | Discovery | Lambda2 | Collision count |
|---|---|---:|---:|---:|---:|
| 10 deg | Default gated | 0.235312 | 0.301980 | 11.904 | 1403.3 |
| 15 deg | Adaptive gated | 0.226503 | 0.293495 | 10.993 | 1463.7 |

Raw-discovery ordering:

- 10 deg: topology gate (0.345091) is slightly above the ungated contention actor (0.342909), while the ungated actor has higher `lambda2` than topology gate.
- 15 deg: ungated contention actor is highest in raw discovery (0.423293) and `lambda2` (19.889), but has the largest collision count among the main actors.

Conclusion: the evidence supports a tunable operating frontier, not a single universally dominant profile.

## Remaining Integrity Risks

1. The manuscript still contains supplementary legacy 600-slot evidence. It is acceptable only if described as mechanism-boundary evidence, not as the final MARL evidence line.
2. Reference metadata and claim-to-reference alignment have not yet been fully web-verified in this audit. A separate citation-integrity pass is still required before any submission claim.
3. Figure caption provenance is based on existing CSV/PNG artifacts. A stricter figure package with per-figure source-data and transformation hashes would further reduce review risk.
