# Final Status Report Draft - 2026-07-05

This draft is intended for the 11:00 handoff.

## Repository State

- Branch: `master`
- Latest synced commit before this draft: `d437871` (`Add pre-11am task board`)
- GitHub remote: `https://github.com/nuaasgq/ISAC-Neighbor-Discovery.git`
- Working tree after last push: clean

## Manuscript Package

- Main draft: `07_paper/ieee_twc_isac_nd/main.tex`
- Main PDF: `07_paper/ieee_twc_isac_nd/main.pdf`
- Supplement: `07_paper/ieee_twc_isac_nd/supplement.tex`
- Supplement PDF: `07_paper/ieee_twc_isac_nd/supplement.pdf`
- References: `07_paper/ieee_twc_isac_nd/references.bib`

Current compiled state:

- Main: 8 pages, texcount `3669+62+294`.
- Supplement: 7 pages, texcount `483+40+275`.
- Compile status: clean LaTeX log, no unresolved citation/reference, no overfull hbox.

## Verification

Latest checks completed:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
python -m pytest 05_simulation\tests
python 06_analysis\scripts\build_paired_delta_summary.py
python 06_analysis\scripts\build_statistical_summary.py
```

Results:

- Unit tests: 25 passed.
- Paper figures: 358 PNG files checked, all 4:3.
- Main PDF pages: 8.
- Supplement PDF pages: 7.

## Main Evidence

| Evidence | Result |
|---|---:|
| N=100/B=10 main proposed discovery | 0.3655 |
| N=100/B=10 enhanced no-ISAC discovery | 0.0007 |
| N=100/B=10 proposed lambda2 | 12.9222 |
| Candidate-set removal discovery | 0.0313 |
| One-slot delayed ISAC discovery | 0.2989 |
| N=100/B=15 proposed discovery | 0.5440 |
| N=100/B=30 proposed discovery | 0.4568 |
| N=100/B=3 stress proposed discovery | 0.0131 |

Paired deltas:

- Main N=100/B=10 proposed vs enhanced no-ISAC: discovery +0.3648, empty scan -0.4025, lambda2 +12.92.
- Main N=100/B=15 proposed vs enhanced no-ISAC: discovery +0.5403, empty scan -0.5052, lambda2 +26.84.
- Candidate-set ablation: discovery +0.3341 vs removing candidate-set refinement.
- One-slot delayed ISAC vs enhanced no-ISAC: discovery +0.2982.

Round10 extra seeds:

- Extra N=100/B=10 proposed discovery: 0.1739 vs enhanced no-ISAC 0.0008.
- Extra N=100/B=15 proposed discovery: 0.4181 vs enhanced no-ISAC 0.0045.
- Interpretation: qualitative ordering is stable, but absolute discovery is scenario-seed sensitive.

## Claim Boundaries

Use these boundaries consistently:

- The paper is a cross-layer link-layer protocol paper.
- ISAC is an imperfect beam-cell occupancy prior, not an oracle.
- Edges are created only by bidirectional narrow-beam handshake.
- The topology metric is for the finite-horizon discovered-neighbor graph/cache, not arbitrary active-link connectivity.
- The current learning evidence is shared-parameter protocol tuning, not full MAPPO/QMIX/GNN-MARL.
- SkyOrbs-like is an inspired deterministic 3-D skip-scan reference, not a strict reproduction.
- 3--5 degree beams and abrupt mobility are stress regimes.

## Key Artifacts

- Claim map: `06_analysis/claim_evidence_matrix.md`
- Figure map: `06_analysis/figure_to_claim_map_20260705.md`
- Reviewer Q&A: `06_analysis/reviewer_question_response_map_20260705.md`
- Novelty/evidence summary: `06_analysis/novelty_evidence_summary_20260705.md`
- Submission readiness audit: `06_analysis/submission_readiness_audit_20260705.md`
- Pre-11 task board: `06_analysis/pre_11am_task_board_20260705.md`
- Round10 extra-seed index: `06_analysis/round10_extra_seed_stability_index.md`
- Candidate-constrained MARL roadmap: `06_analysis/marl_architecture_roadmap_20260705.md`

## Readiness Judgment

Current package is strong enough for an internal paper draft and advisor discussion.
It is not yet final TWC/TCOM submission-ready because a reviewer could still request:

- more seeds for main N=100/B=10/B=15 results,
- a stricter SkyOrbs reproduction or clearer de-emphasis,
- collision-aware protocol optimization,
- Joule-level energy model,
- calibrated physical-layer mapping for ISAC error parameters.

Next neural-method direction is documented in `06_analysis/marl_architecture_roadmap_20260705.md`.
The recommended route is not a flat categorical beam policy, but a candidate-constrained shared actor-critic with ISAC mask/top-k beam proposal, rule-residual logits, topology-deficit conditioning, and centralized critic only during training.
The first code interface for this route has been added: `MarlNeighborDiscoveryEnv` now exposes a local `candidate_mask`, and `SharedBeamActorCritic(..., use_candidate_mask=True)` can apply it without changing existing default behavior.

The most defensible paper angle is:

> ISAC-assisted candidate-beam refinement is a viable cross-layer mechanism for narrowing the search space of distributed UAV-UAV narrow-beam neighbor discovery, with clear gains over blind/no-ISAC baselines and explicit boundaries under extreme beamwidths, abrupt mobility, and collision-heavy regimes.
