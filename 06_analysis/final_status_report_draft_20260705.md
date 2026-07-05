# Final Status Report Draft - 2026-07-05

This draft is intended for the 11:00 handoff.

## Repository State

- Branch: `master`
- Latest synced commit: current `master` HEAD after the round11 paired-seed stability evidence commit.
- GitHub remote: `https://github.com/nuaasgq/ISAC-Neighbor-Discovery.git`
- Working tree at last check: clean

## Manuscript Package

- Main draft: `07_paper/ieee_twc_isac_nd/main.tex`
- Main PDF: `07_paper/ieee_twc_isac_nd/main.pdf`
- Supplement: `07_paper/ieee_twc_isac_nd/supplement.tex`
- Supplement PDF: `07_paper/ieee_twc_isac_nd/supplement.pdf`
- References: `07_paper/ieee_twc_isac_nd/references.bib`

Current compiled state:

- Main: 8 pages.
- Supplement: 9 pages.
- Compile status: no unresolved citation/reference and no overfull warnings in the checked final logs.
- Remaining LaTeX messages are narrow-column underfull warnings only.

## Verification

Latest checks completed:

```powershell
python -m pytest 05_simulation\tests
python -m py_compile 05_simulation\run_actor_critic_imitation_probe.py `
  06_analysis\scripts\analyze_structured_marl_probe.py `
  06_analysis\scripts\plot_pre11_evidence.py
python 06_analysis\scripts\plot_round11_stability.py
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode supplement.tex
pdflatex -interaction=nonstopmode supplement.tex
```

Results:

- Unit tests: 27 passed.
- Main PDF pages: 8.
- Supplement PDF pages: 9.
- New pre-11 and round11 evidence figures: 18 PNGs, all checked at 1920x1440 (4:3).

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

Round10 extra-seed backup:

- Extra N=100/B=10 proposed discovery: 0.1739 vs enhanced no-ISAC 0.0008.
- Extra N=100/B=15 proposed discovery: 0.4181 vs enhanced no-ISAC 0.0045.
- Interpretation: qualitative ordering is stable, but absolute discovery is scenario-seed sensitive.

Round11 five-seed paired campaign:

- N=100/B=10 proposed discovery: 0.3639 vs enhanced no-ISAC 0.0006.
- N=100/B=15 proposed discovery: 0.5445 vs enhanced no-ISAC 0.0034.
- Proposed raw-discovery deltas are positive in all 5/5 paired seeds against random, enhanced no-ISAC, candidate-set removal, and one-slot delay at both B=10 and B=15.
- Collision-aware boundary: B=10 proposed collision-penalized discovery is 0.2995 vs one-slot delay 0.2622, but at B=15 proposed is 0.2042 vs one-slot delay 0.2207. Collision-aware MAC control remains open.

## Structured Neural MARL Probe

Implemented:

- Local `candidate_mask`, `candidate_score`, `topology_deficit`, and `rule_mode_logits`.
- Optional actor flags for candidate masking, candidate-score features, topology-deficit context, and rule-residual logits.
- `--eval-both` for deterministic and stochastic evaluation of the same trained policy.
- `--env-protocol` for a clean no-ISAC neural baseline.

Core N=10/B=72/80-slot probe:

- Flat stochastic discovery: 0.6322.
- Full structured residual stochastic discovery: 0.5571.
- Flat deterministic discovery: 0.0015.
- Full structured residual deterministic discovery: 0.0643, with 14/15 deterministic evaluations nonzero.
- Full structured residual stochastic empty-scan ratio: 0.1112 vs flat stochastic 0.6901.

Follow-up:

- RL10 fine-tune improves full structured stochastic discovery to 0.5865 but still does not beat flat stochastic.
- Residual-strength sweep finds `rule_residual_scale=0.25` as the best balanced tested setting: stochastic discovery 0.5978 and deterministic discovery 0.0837 with 15/15 deterministic evaluations nonzero.
- Clean no-ISAC neural baseline gives deterministic discovery 0 and stochastic discovery 0.0044.
- Interpretation: ISAC-enabled local candidate observations are necessary for the neural probe's nonzero behavior, but collision coordination and stochastic dominance are not solved.

## New Artifacts

- Pre-11 evidence figures/tables:
  - `06_analysis/paper_figures/pre11_evidence`
  - `06_analysis/paper_tables/pre11_evidence`
- Structured MARL figures/tables:
  - `06_analysis/paper_figures/structured_marl_probe`
  - `06_analysis/paper_tables/structured_marl_probe`
- Round11 paired-seed stability figures/tables:
  - `06_analysis/paper_figures/round11_paired_seed_campaign_main`
  - `06_analysis/paper_tables/round11_paired_seed_campaign_main`
- One-page submission pitch:
  - `06_analysis/submission_pitch_one_page_20260705.md`
- Progress report:
  - `06_analysis/pre11_morning_progress_report_20260705.md`
- MARL scripts:
  - `06_analysis/scripts/run_marl_probe_task.ps1`
  - `06_analysis/scripts/analyze_structured_marl_probe.py`

## Claim Boundaries

Use these boundaries consistently:

- The paper is a cross-layer link-layer protocol paper.
- ISAC is an imperfect beam-cell occupancy prior, not an oracle.
- Edges are created only by bidirectional narrow-beam handshake.
- The topology metric is for the finite-horizon discovered-neighbor graph/cache, not arbitrary active-link connectivity.
- The main method remains the rule-driven ISAC-assisted protocol.
- Neural MARL is currently a structured method probe, not the main result.
- SkyOrbs-like is an inspired deterministic 3-D skip-scan reference, not a strict reproduction.
- 3--5 degree beams and abrupt mobility are stress regimes.
- Raw discovery and collision-penalized discovery should be discussed separately; the current method is not collision-optimal.

## Readiness Judgment

The package is now strong for advisor discussion and an internal paper draft.
The central paper angle is defensible:

> ISAC-assisted candidate-beam refinement is a viable cross-layer mechanism for narrowing the search space of distributed UAV-UAV narrow-beam neighbor discovery, with clear gains over blind/no-ISAC baselines and explicit boundaries under extreme beamwidths, abrupt mobility, and collision-heavy regimes.

It is still not final TWC/TCOM submission-ready because a reviewer could request more main-result seeds, a strict SkyOrbs reproduction, collision-aware protocol optimization, a Joule-level energy model, or a calibrated PHY mapping for ISAC error parameters.
