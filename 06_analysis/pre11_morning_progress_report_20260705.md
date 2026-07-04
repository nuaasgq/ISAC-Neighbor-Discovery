# Pre-11 Morning Progress Report - 2026-07-05

## Completed Since 07:10

- Added structured neural MARL hooks:
  - local `candidate_mask`,
  - local `candidate_score`,
  - scalar local `topology_deficit`,
  - local `rule_mode_logits`,
  - optional actor flags for candidate masking, candidate-score features, topology-deficit context, and rule-residual logits.
- Added `--eval-both` so one trained policy is evaluated with deterministic and stochastic execution.
- Added a reusable PowerShell runner for single MARL probe tasks:
  - `06_analysis/scripts/run_marl_probe_task.ps1`.
- Added MARL probe aggregation:
  - `06_analysis/scripts/analyze_structured_marl_probe.py`,
  - `06_analysis/paper_tables/structured_marl_probe`,
  - `06_analysis/paper_figures/structured_marl_probe`.
- Added pre-11 evidence figures:
  - cumulative discovery curves,
  - raw discovery / collision-penalized discovery / lambda2 / empty-scan tradeoff bars,
  - outputs under `06_analysis/paper_figures/pre11_evidence` and `06_analysis/paper_tables/pre11_evidence`.
- Updated the IEEE supplement with:
  - finite-horizon cumulative discovery trajectories,
  - collision-aware tradeoff figures,
  - structured neural MARL probe figures and scoped interpretation.

## Key Result Takeaways

- Round10 extra-seed backup:
  - N=100, B=10 proposed discovery = 0.1739 vs enhanced no-ISAC = 0.0008.
  - N=100, B=15 proposed discovery = 0.4181 vs enhanced no-ISAC = 0.0045.
  - These are supplementary seed-sensitivity results, not replacements for main round3 results.
- Core structured MARL probe, N=10/B=72/80 slots:
  - Flat stochastic discovery = 0.6322.
  - Full structured residual stochastic discovery = 0.5571.
  - Flat deterministic discovery = 0.0015.
  - Full structured residual deterministic discovery = 0.0643 with 14/15 deterministic evals nonzero.
  - Full structured residual stochastic empty-scan ratio = 0.1112 vs flat stochastic = 0.6901.
- RL10 follow-up:
  - Full structured residual stochastic discovery improves from 0.5571 to 0.5865.
  - Deterministic discovery remains about 0.0658.
  - Still not enough to promote neural MARL to the main method.

## Interpretation Boundary

The rule-driven ISAC-assisted protocol remains the paper's main method.
The neural MARL result is useful as feasibility evidence for a future structured neural extension: candidate constraints and rule residuals reduce deterministic collapse and empty scanning, but collision coordination and stochastic-performance dominance are not solved yet.

## Verification

- `python -m pytest 05_simulation\tests` passed earlier after the MARL interface changes: 27 tests.
- `python -m py_compile 06_analysis\scripts\analyze_structured_marl_probe.py 06_analysis\scripts\plot_pre11_evidence.py` passed.
- `pdflatex -interaction=nonstopmode supplement.tex` run twice after supplement edits.
- Final supplement compile: 9 pages, no undefined-reference or overfull warnings in the checked log output.
