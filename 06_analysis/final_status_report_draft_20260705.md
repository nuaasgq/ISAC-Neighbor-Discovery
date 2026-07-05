# Final Status Report Draft - 2026-07-05

This draft is intended for the 11:00 handoff.

## Repository State

- Branch: `master`
- Latest synced commit: current `master` HEAD after the final gap-triage, round13 reproducibility, reference-metadata, supplement-procedure, round14 ten-seed main-table, cite-backed PHY-to-protocol ISAC mapping, and SkyOrbs-like baseline-scope updates.
- GitHub remote: `https://github.com/nuaasgq/ISAC-Neighbor-Discovery.git`
- Working tree at last check: clean after the final local commit/push.

## Manuscript Package

- Main draft: `07_paper/ieee_twc_isac_nd/main.tex`
- Main PDF: `07_paper/ieee_twc_isac_nd/main.pdf`
- Supplement: `07_paper/ieee_twc_isac_nd/supplement.tex`
- Supplement PDF: `07_paper/ieee_twc_isac_nd/supplement.pdf`
- References: `07_paper/ieee_twc_isac_nd/references.bib`

Current compiled state:

- Main: 9 pages.
- Supplement: 10 pages.
- Compile status: no unresolved citation/reference and no overfull warnings in the checked final logs.
- Remaining LaTeX messages are narrow-column underfull warnings only.

## Verification

Latest checks completed:

```powershell
python -m pytest 05_simulation\tests
python -m py_compile 05_simulation\run_actor_critic_imitation_probe.py `
  06_analysis\scripts\analyze_structured_marl_probe.py `
  06_analysis\scripts\plot_pre11_evidence.py `
  06_analysis\scripts\plot_round12_collision_aware.py
python 06_analysis\scripts\plot_round11_stability.py
python 05_simulation\run_transfer_sweep.py --config 05_simulation\configs\paper_transfer_train_n10_b10_singlehop.yaml --trained-config 06_analysis\paper_tables\round2_transfer\training\best_config.yaml --output 05_simulation\results_raw\round13_collision_energy_10seed --node-counts 100 --beamwidth-degs 10,15 --mobilities gauss_markov --seeds 20290704,20291713,20292722,20293731,20294740,20295749,20296758,20297767,20298776,20299785 --episodes-per-seed 1 --slots 600 --slot-metric-period 1 --area-scale density --range-mode singlehop --protocols uniform_random,improved_rl_no_isac,ablation_isac_one_slot_delay,improved_rl_isac,collision_aware_isac --train-node-count 10 --train-beamwidth-deg 10 --name round13_collision_energy_10seed
python 06_analysis\scripts\plot_round12_collision_aware.py --source 05_simulation\results_raw\round13_collision_energy_10seed --output 06_analysis\paper_tables\round13_collision_energy_10seed --figures 06_analysis\paper_figures\round13_collision_energy_10seed --tag round13
python 05_simulation\run_transfer_sweep.py --config 05_simulation\configs\paper_transfer_train_n10_b10_singlehop.yaml --trained-config 06_analysis\paper_tables\round2_transfer\training\best_config.yaml --output 05_simulation\results_raw\round14_main_table_10seed_n100_b10 --node-counts 100 --beamwidth-degs 10 --mobilities gauss_markov --seeds 20290704,20291713,20292722,20293731,20294740,20295749,20296758,20297767,20298776,20299785 --episodes-per-seed 1 --slots 600 --slot-metric-period 0 --area-scale density --range-mode singlehop --protocols uniform_random,skyorbs_like_skip_scan,rl_no_isac,improved_rl_no_isac,improved_rl_isac --train-node-count 10 --train-beamwidth-deg 10 --name round14_main_table_10seed_n100_b10
python 06_analysis\scripts\analyze_round14_main_table.py --source 05_simulation\results_raw\round14_main_table_10seed_n100_b10 --output 06_analysis\paper_tables\round14_main_table_10seed_n100_b10 --figures 06_analysis\paper_figures\round14_main_table_10seed_n100_b10
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode supplement.tex
pdflatex -interaction=nonstopmode supplement.tex
```

Results:

- Unit tests: 29 passed.
- Main PDF pages: 9.
- Supplement PDF pages: 10.
- Selected pre-11, round11, round13, and round14 evidence figures are checked at 1920x1440 (4:3); round13 contributes 8 current collision/energy figures and round14 contributes 1 current main-table stability figure.
- Referenced figures in `main.tex` and `supplement.tex`: all present on disk. The figure audit checks 47 LaTeX figure instances, 44 unique figure files, 0 missing files, and 0 non-4:3 aspect-ratio violations.
- The final checked LaTeX logs contain no unresolved citations/references and no overfull warnings after adding the 3GPP ISAC/channel references and the supplement PHY-to-protocol parameter table.

## Main Evidence

| Evidence | Result |
|---|---:|
| N=100/B=10 main proposed discovery | 0.3652 |
| N=100/B=10 enhanced no-ISAC discovery | 0.0006 |
| N=100/B=10 proposed lambda2 | 13.2595 |
| Candidate-set removal discovery | 0.0313 |
| One-slot delayed ISAC discovery | 0.2989 |
| N=100/B=15 proposed discovery | 0.5440 |
| N=100/B=30 proposed discovery | 0.4568 |
| N=100/B=3 stress proposed discovery | 0.0131 |

Paired deltas:

- Round14 main N=100/B=10 proposed vs enhanced no-ISAC: discovery +0.3646, empty scan -0.4018, lambda2 +13.26, with 10/10 positive paired discovery deltas.
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

Round13 collision-aware MAC refinement probe:

- Same N=100/B=10/B=15, ten paired seeds, Gauss-Markov, 600-slot, density-scaled, single-hop setting as round11.
- B=10 collision-aware discovery/collision-penalized discovery: 0.3660 / 0.3147 vs proposed 0.3652 / 0.2991.
- B=15 collision-aware discovery/collision-penalized discovery: 0.5647 / 0.2479 vs proposed 0.5421 / 0.2017.
- Collision-aware collision-penalized deltas are positive in 10/10 paired seeds versus both the proposed low-latency protocol and the one-slot delayed variant at B=10 and B=15.
- Assumed radio-state accounting: discoveries per joule improve from 6.1932 to 6.5417 at B=10 and from 9.2045 to 10.1564 at B=15 versus the proposed low-latency protocol, also with 10/10 positive paired deltas.
- Interpretation: the B=15 collision boundary is mitigated in this ten-seed probe by local role control, but full collision- and platform-calibrated energy-aware MAC design remains open.

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
- Round13 collision-aware MAC and energy figures/tables:
  - `06_analysis/paper_figures/round13_collision_energy_10seed`
  - `06_analysis/paper_tables/round13_collision_energy_10seed`
  - `06_analysis/scripts/plot_round12_collision_aware.py`
- Round14 ten-seed main-table stability figures/tables:
  - `06_analysis/paper_figures/round14_main_table_10seed_n100_b10`
  - `06_analysis/paper_tables/round14_main_table_10seed_n100_b10`
  - `06_analysis/scripts/analyze_round14_main_table.py`
- PHY-to-protocol ISAC mapping note:
  - `06_analysis/phy_to_protocol_isac_mapping_20260705.md`
- SkyOrbs-like baseline-scope appendix:
  - `06_analysis/skyorbs_like_baseline_scope_appendix_20260705.md`
- Paper figure integrity audit:
  - `06_analysis/paper_figure_integrity_audit_20260705.md`
  - `06_analysis/paper_figure_integrity_audit_20260705.csv`
  - `06_analysis/scripts/audit_paper_figures.py`
- Round13 paper-writing route:
  - `06_analysis/round13_twc_tcom_revision_route_20260705.md`
- Pre-submission gap triage:
  - `06_analysis/pre_submission_gap_triage_20260705.md`
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
- `Rc`, `Rs`, false alarms, missed detections, angular-cell error, and staleness are cite-backed sensing-service/link-support abstraction parameters, not calibrated waveform/detector outputs.
- Edges are created only by bidirectional narrow-beam handshake.
- The topology metric is for the finite-horizon discovered-neighbor graph/cache, not arbitrary active-link connectivity.
- The main method remains the rule-driven ISAC-assisted protocol.
- Neural MARL is currently a structured method probe, not the main result.
- SkyOrbs-like is an inspired deterministic 3-D skip-scan reference, not a strict reproduction; the supplement and standalone scope appendix now state this boundary explicitly.
- 3--5 degree beams and abrupt mobility are stress regimes.
- Raw discovery, collision-penalized discovery, and assumed energy accounting should be discussed separately; round13 mitigates the collision boundary but does not provide a final MAC or platform-calibrated Joule-level energy model.

## Readiness Judgment

The package is now strong for advisor discussion and an internal paper draft.
The central paper angle is defensible:

> ISAC-assisted candidate-beam refinement is a viable cross-layer mechanism for narrowing the search space of distributed UAV-UAV narrow-beam neighbor discovery, with clear gains over blind/no-ISAC baselines and explicit boundaries under extreme beamwidths, abrupt mobility, and collision-heavy regimes.

It is still not final TWC/TCOM submission-ready because a reviewer could request broader B=15/mobility seed campaigns, a strict SkyOrbs reproduction, stronger energy-aware MAC validation, a calibrated radio-state power model, or a calibrated PHY appendix for ISAC error parameters. The current SkyOrbs risk is now framed as a deliberate non-reproduction boundary rather than an ambiguous baseline claim.
