## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-07
- Verification Status: VERIFIED_PARTIAL
- Version Label: validation_v1

## Validation Report

- Source: Phase10 manuscript-facing MARL result tables, final manuscript figure/table provenance, the generated manuscript artifact hash manifest, and one independent stochastic Phase10 transfer re-run.
- Primary CSV: `06_analysis/paper_tables/marl/p10_final_b10_b15_method_comparison_with_v4/marl_method_comparison.csv`
- Gate-family CSV: `06_analysis/paper_tables/marl/p10_gate_family_v2_v3_v4_tradeoff_comparison/seed_tradeoff_core_metrics.csv`
- Artifact manifest: `06_analysis/manuscript_artifact_manifest_20260707.csv`
- Overall Confidence: CAUTION for global campaign breadth; PASS for the primary paired ISAC-vs-communication-only discovery/empty-scan test boundary

The performance direction is strong in the current simulator evidence, especially for the ISAC-assisted MARL actor against communication-only baselines. A follow-up paired sign-test addendum now predeclares the primary discovery-rate and empty-scan-ratio families for the final B=10/B=15 ISAC-vs-communication-only comparison; all 16 confirmatory tests pass Holm-adjusted alpha=0.05. The global confidence rating remains CAUTION because gate-family operating points, mobility stress, beamwidth stress, and hardware-facing PHY abstractions are not globally multiple-comparison-corrected claims.

### Statistical Findings

| Metric | Test | Value | Effect Size | Confidence |
|---|---|---:|---|---|
| B=10 discovery gain over random/directional baselines | Ratio from final CSV | min 110.94x; actor discovery 0.3429 +/- 0.0054 CI95 | Very large practical effect; limiting baseline is SkyOrbs-like at 0.0031 | CAUTION |
| B=15 discovery gain over random/directional baselines | Ratio from final CSV | min 22.31x; actor discovery 0.4233 +/- 0.0081 CI95 | Very large practical effect; limiting baseline is SkyOrbs-like at 0.0190 | CAUTION |
| Primary paired discovery/empty-scan tests | Exact paired sign tests with Holm correction | 16/16 confirmatory tests pass; all paired discovery deltas are positive and all paired empty-scan deltas are negative across 10/10 scenario seeds | Strong seed-paired support for the primary simulator claim, restricted to ISAC actor vs communication-only controls | SOLID within stated boundary |
| B=10 default gate collision-penalized discovery | Descriptive comparison | 0.2353 +/- 0.0058 vs ungated 0.2263 +/- 0.0067 | +3.99% CPD with 45.26% fewer collisions | CAUTION |
| B=15 default gate collision-penalized discovery | Descriptive comparison | 0.2114 +/- 0.0061 vs ungated 0.1387 +/- 0.0066 | +52.35% CPD with 65.43% fewer collisions | CAUTION |
| B=15 adaptive gate collision-penalized discovery | Descriptive comparison | 0.2265 +/- 0.0051 vs ungated 0.1387 +/- 0.0066 | Highest B=15 CPD among current gate-family points, but lower raw discovery/lambda2 than topology-heavy profiles | CAUTION |
| Graph connectivity separation | Descriptive comparison | actor/gated lambda2 is 11.90-19.89 for main rows; communication-only baselines are numerically zero | Strong graph-level separation in simulator | CAUTION |
| Training trace coverage | Artifact-manifest check | 90,000 step rows, 300 episode rows, 198 eval rows across three N=10/B=10 training seeds | Supports step-indexed convergence diagnostics, not formal optimality | CAUTION |
| Learned-component ablation | Focused 3-episode B=10 transfer ablation | trained CPD 0.2297 vs random-weight 0.1599 and zero-weight/rule-only 0.1464 | Supports collision-efficiency contribution, not universal learned dominance in raw discovery/lambda2 | CAUTION |

### Warnings

| Type | Detail | Affected |
|---|---|---|
| Sample size | The final Phase10 transfer table mostly uses 10 stochastic evaluation episodes. The primary ISAC-vs-communication-only comparison now has exact paired sign tests, but most gate-family and robustness comparisons remain descriptive. | Final B=10/B=15 method comparison |
| Variant selection | The project iteratively explored multiple gate profiles and reward/network variants. The final manuscript must keep the distinction between selected main profile, aggressive profile, adaptive profile, topology-heavy profile, and supplementary probes. | Gate-family claims |
| Mixed horizon evidence | Main Phase10 claims use 3000-slot N=100 transfer, while several robustness/boundary figures come from earlier 600-slot campaigns. The manuscript currently labels those as boundary/supplementary evidence; this boundary should remain explicit. | Mobility, range, error, ablation figures |
| Simulator-to-platform gap | ISAC sensing, energy accounting, and slot timing are protocol-level abstractions. The results support protocol feasibility in the simulator, not hardware-calibrated energy or PHY-optimal claims. | PHY/protocol interpretation and energy discussion |
| Multiple comparisons | Many scenarios, metrics, and profiles were evaluated. The new paired sign-test addendum corrects only the primary discovery and empty-scan families; there is still no global correction over every explored profile and robustness setting. | All broad superiority claims |
| Learned contribution boundary | Random/zero-weight policies can obtain higher raw discovery and lambda2 by accepting much higher collision load. | Learned-component claims should emphasize collision suppression and tradeoff shaping, not raw discovery dominance. |

### Fallacy Scan

- Coverage: 11/11 fallacy types checked.

| Fallacy | Severity | Detail | Recommendation |
|---|---|---|---|
| Simpson's paradox | NOTE | The final comparison is stratified by beamwidth B=10 and B=15. No pooled cross-beam aggregate is used for the main gain claim. | Keep reporting beamwidth-stratified results. |
| Ecological fallacy | CAUTION | Discovery rate and lambda2 are graph-level metrics. They should not be used to infer per-UAV link quality or per-pair fairness without additional per-node/per-link analysis. | State graph-level interpretation explicitly. |
| Berkson's paradox | NOTE | The main setting intentionally evaluates single-hop eligible neighbors. This is a design boundary rather than a selected-observation sample. | Keep single-hop and range assumptions visible. |
| Collider bias | NOTE | The current tables are direct simulation comparisons, not controlled regressions with potentially collider variables. | No collider adjustment claim is needed. |
| Base rate neglect | CAUTION | Discovery rates depend on eligible edge count, beamwidth, region scale, and range assumptions. Reporting rates without scenario context would be misleading. | Report N, beamwidth, horizon, and single-hop/range settings with rates. |
| Regression to the mean | NOTE | The final transfer table is not a pre/post extreme-group design. | No action beyond avoiding convergence overclaims. |
| Survivorship bias | CAUTION | Current main evidence uses selected completed training/evaluation campaigns. Failed or discarded exploratory attempts are not part of the statistical denominator. | Treat training selection as model-development history, not as a random sample of all possible policies. |
| Look-elsewhere effect | CAUTION | Multiple protocols, beams, mobility models, and gate variants were tested. Large effects against blind baselines are still visible, but broad profile-selection claims should remain conservative. | Tie headline claims to the predefined Phase10 final evidence chain. |
| Garden of forking paths | CAUTION | The method was iteratively refined through reward/network/gate variants. Provenance audits reduce ambiguity but do not equal preregistration. | Describe the final method and use earlier probes as ablations/boundaries. |
| Correlation vs causation | NOTE | Simulator ablations can support causal statements inside the modeled environment, e.g., disabling ISAC reduces discovery under fixed assumptions. They do not prove real-world hardware causality. | Qualify claims as simulator/protocol-level evidence. |
| Reverse causality | NOTE | Reverse causality is not a primary risk for intervention-style simulation comparisons. | No action beyond avoiding field-deployment causal overreach. |

### Reproducibility

- Method: artifact hashing plus manifest/provenance verification, with one independent stochastic re-run for `gated_contention_actor` at N=100, B=10, 3000 slots, and 10 episodes.
- Verdict: VERIFIED_PARTIAL for the selected B=10 gated transfer point; CANNOT_VERIFY for full stochastic Phase10 campaign reproduction.
- Artifact result: `06_analysis/scripts/build_manuscript_artifact_manifest.py` generated 139 current manuscript-facing artifacts with 0 missing paths.
- Figure result: the current figure audit checked 51 referenced PNG instances, with 0 missing files and 0 approximate 4:3 aspect-ratio violations.
- LaTeX result: the current `main.tex` and `supplement.tex` compiled to 9 and 13 pages respectively with no undefined references/citations, no overfull warnings, and no LaTeX hard errors in the checked logs.

| Metric | Original | Re-run | Diff | Status |
|---|---:|---:|---:|---|
| Artifact paths missing | N/A | 0 | N/A | MATCH |
| Manuscript PNG references | N/A | 51 instances | N/A | MATCH |
| Non-4:3 PNG aspect violations | N/A | 0 | N/A | MATCH |
| gated_contention_actor B=10 discovery_rate | 0.301980 | 0.299475 | 0.83% relative | MATCH |
| gated_contention_actor B=10 collision-penalized discovery | 0.235312 | 0.232290 | 1.28% relative | MATCH |
| gated_contention_actor B=10 lambda2 | 11.904328 | 12.061005 | 1.32% relative | MATCH |
| Full MARL transfer reproduction | Existing Phase10 summaries | One selected point re-run only | N/A | CANNOT_VERIFY |

### Interpretation Boundary

The current data are sufficient to support a paper-writing draft around the MARL+ISAC neighbor-discovery mechanism, the small-scale-to-large-scale transfer claim, and the collision/topology gate-family tradeoff. They are not yet sufficient for claims of hardware-calibrated PHY optimality, globally corrected statistical significance across every explored setting, or fairness across individual UAVs.
