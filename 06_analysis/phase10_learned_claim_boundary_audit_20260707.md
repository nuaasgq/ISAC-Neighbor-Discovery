# Phase10 Learned-Claim Boundary Audit - 2026-07-07

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Verification Status: ANALYZED
- Scope: focused learned-component ablation and manuscript claim boundary

## Summary

- Checks passed: 9/9.
- Paired trained-vs-control comparisons: 6.
- Collision/CPD paired improvements: 6/6.
- Bounded learned claim pass: True.

## Boundary Interpretation

The audit supports only a bounded learned-component claim: trained weights improve the collision-efficiency operating point relative to random-weight and zero-weight controls under the same structured ISAC/rule interface.
It does not support universal raw-discovery dominance; the manuscript must keep the current caveats about aggressive access, candidate masks, rule residuals, and the three-episode probe.

## Check Table

| ID | Theme | Status | Evidence | Boundary |
|---|---|---|---|---|
| L01 | required ablation variants | PASS | 5/5 labels present | All learned/control feature variants needed to separate weights, mask, and residual priors are present. |
| L02 | minimum paired episodes | PASS | episode_counts=[3, 3, 3, 3, 3] | This is a focused three-episode probe, not a broad stochastic dominance test. |
| L03 | collision reduction vs random and zero weights | PASS | trained=1415.3, random=7624.7, zero=9552.3 | Supports collision-suppression contribution under the same structured interface. |
| L04 | CPD improvement vs random and zero weights | PASS | trained=0.2297, random=0.1599, zero=0.1464 | Supports collision-efficiency, not raw discovery dominance. |
| L05 | paired seed sign consistency | PASS | 6/6 paired control comparisons improve collision and CPD | Paired controls: random=3, zero=3. |
| L06 | raw discovery non-dominance acknowledged | PASS | trained=0.2958, random=0.4054, zero=0.4265 | Prevents overclaiming learned raw-discovery superiority. |
| L07 | candidate-mask tradeoff | PASS | no_mask_cpd=0.2826, trained_cpd=0.2297, no_mask_empty=0.8173 | No-mask CPD is a boundary signal because empty scans increase sharply. |
| L08 | rule-residual tradeoff | PASS | no_rule_collision=536.7, trained_collision=1415.3, no_rule_discovery=0.2574, trained_discovery=0.2958 | Rule residual raises raw discovery/topology while increasing collisions. |
| L09 | manuscript claim boundary | PASS | 07_paper/ieee_twc_isac_nd/main.tex:366; 07_paper/ieee_twc_isac_nd/supplement.tex:347; 07_paper/ieee_twc_isac_nd/supplement.tex:351; 07_paper/ieee_twc_isac_nd/supplement.tex:352; 07_paper/ieee_twc_isac_nd/supplement.tex:352 |  |

## Paired Episode Evidence

| Seed | Control | Collision delta | CPD delta | Raw discovery delta |
|---|---|---:|---:|---:|
| 52260731 | random_weights_full | -5426.0 | 0.0518 | -0.1188 |
| 52260731 | zero_weights_rule_only | -6772.0 | 0.0581 | -0.1477 |
| 52260732 | random_weights_full | -6482.0 | 0.0800 | -0.0990 |
| 52260732 | zero_weights_rule_only | -8567.0 | 0.0945 | -0.1206 |
| 52260733 | random_weights_full | -6720.0 | 0.0778 | -0.1111 |
| 52260733 | zero_weights_rule_only | -9072.0 | 0.0975 | -0.1238 |

## Generated Files

- `06_analysis/paper_tables/marl/p10_learned_claim_boundary_audit/learned_claim_checks.csv`
- `06_analysis/paper_tables/marl/p10_learned_claim_boundary_audit/paired_episode_claim_checks.csv`
- `06_analysis/paper_tables/marl/p10_learned_claim_boundary_audit/manifest.json`
