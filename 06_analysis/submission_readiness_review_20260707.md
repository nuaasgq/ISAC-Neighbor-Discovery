# Submission Readiness Review - 2026-07-07

Scope: `07_paper/ieee_twc_isac_nd/main.tex`, `supplement.tex`, and the Phase10 manuscript-facing evidence chain.

Reviewer posture: IEEE TWC/TCOM-style pre-submission audit focused on claim strength, methodology transparency, experiment integrity, and reproducibility. This is not a replacement for an external peer review.

## Field and Maturity

- Primary discipline: wireless networking / UAV swarm networking.
- Secondary disciplines: ISAC, directional neighbor discovery, multi-agent reinforcement learning.
- Paradigm: quantitative simulation and algorithm/protocol design.
- Methodology type: stochastic simulation, MARL training/evaluation, ablation and robustness sweeps.
- Target tier: Q1 IEEE communications journal level.
- Current maturity: advanced draft with credible evidence chain; still needs final writing polish, full raw-result reproducibility instructions, and likely reviewer-facing tightening before submission.

## High-Priority Reviewer Risks

| Priority | Risk | Evidence | Status |
|---|---|---|---|
| P1 | Over-strong causal wording around ISAC's role. | Main text used "This confirms..." for the empty-scan mechanism. | Mitigated in this pass by changing to "This supports... in the evaluated setting." |
| P1 | Statistical evidence can be overread as formal hypothesis testing. | Phase10 final table mostly uses 10 stochastic evaluation episodes; validation report marks confidence as CAUTION. | Mitigated in this pass by adding a main-text note that CIs are descriptive and not global multiple-comparison-corrected tests. |
| P1 | Reproducibility package not visible enough in manuscript. | README/provenance had artifact manifests, but main text did not state the replication-package boundary. | Mitigated in this pass by adding a concise replication-package note in Experimental Setup and supplement Reproducibility Notes. |
| P2 | SkyOrbs-like baseline could be challenged as not a faithful reproduction. | Main and supplement already state this is an inspired baseline, not a strict full protocol reproduction. | Acceptable with current caveat; a future stronger paper can add a closer SkyOrbs reproduction. |
| P2 | MARL method family is still a structured MAPPO-like actor-critic, not an exhaustive latest-MARL comparison. | Limitations state that the evidence is not exhaustive against QMIX/HAPPO/MAT/GNN variants. | Acceptable as a limitation; future work can add a focused MARL-family benchmark. |
| P2 | Some robustness figures come from earlier 600-slot campaigns while the final main table is 3000-slot Phase10 evidence. | Provenance audit separates main evidence from boundary/supplement evidence. | Acceptable if this boundary remains explicit in captions/text. |

## Edits Applied

- Added a main-text replication package note naming result CSVs, figure-generation manifests, training-curve tables, and SHA256 artifact manifest.
- Added a main-text statistical boundary note: confidence intervals are descriptive summaries, not global multiple-comparison-corrected hypothesis tests.
- Changed "This confirms..." to "This supports..." for the ISAC search-space compression interpretation.
- Added supplement Reproducibility Notes for `manuscript_artifact_manifest_20260707.csv` / JSON and `phase10_statistical_validation_report_20260707.md`.
- Added `p10_final_method_manifest_trace/` and a supplementary method-trace index table that maps every final Phase10 method group to training and transfer manifest sources.

## Remaining Submission Tasks

1. Add a concise algorithm box if page budget allows; the current prose/equation form is clear but a reviewer may prefer pseudocode.
2. Add a short "Data and Code Availability" statement only after deciding whether anonymous review will use an anonymized repository or post-acceptance release.
3. If compute budget allows, run a small independent re-run of the final Phase10 transfer evaluation for one method/beam pair to convert part of the artifact-integrity status from ANALYZED to VERIFIED.
4. Before submission, run one final pass for line-level IEEE style, title concision, abstract length, duplicate labels, and BibTeX journal abbreviation consistency.
