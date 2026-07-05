# MARL Paper Alignment Update - 2026-07-05

## Purpose

This update aligns the manuscript and experiment workflow with the rebuilt real MARL + ISAC research line.
The active rule is:

- Train the main MARL policies at `N=10`, `10 deg`, `300 slots/episode`.
- Use longer horizons only for evaluation: `300`, `1200`, and `3000 slots`.
- Treat older `600-slot` and CEM-style results as supplementary mechanism or historical evidence, not as the main MARL claim.

## Code Alignment

- `run_marl_training.py` already defaults to `--slots 300` and logs step rewards, episode metrics, losses, entropy, and resources.
- `run_marl_evaluate.py` now logs `resource_log.csv` during long evaluations and enforces RSS/system-memory limits.
- `run_marl_campaign.py` now defaults to step-level training logs (`--step-log-period 1`), guarded resource limits, and deterministic plus stochastic evaluation unless `--eval-stochastic-only` is explicitly selected.
- `mvp.yaml`, `marl_mvp.yaml`, and `marl_algorithm_sweep.yaml` now use `300` as the default training slot count and list `[300, 1200, 3000]` as evaluation horizons.

## Manuscript Alignment

- The main method name is now `MARL-I-TAP-ND`.
- The learning component is described as slot-level shared-parameter actor-critic MARL with CTDE and decentralized execution.
- The strongest current evidence is phase5:
  - legacy shared ISAC-MAPPO,
  - collision-reward ISAC-MAPPO,
  - contention-aware ISAC-MAPPO.
- Main N=100 transfer evidence uses `3000-slot` evaluation after `300-slot` training.
- The paper emphasizes collision-penalized discovery and collision reduction rather than claiming universal raw-discovery dominance.
- Older 600-slot mobility/error/mechanism sweeps are explicitly marked as supplementary mechanism-boundary evidence.

## Verification

- `python -m py_compile 05_simulation/run_marl_evaluate.py 05_simulation/run_marl_campaign.py 05_simulation/run_marl_training.py`
- `python -m pytest 05_simulation/tests -q`: `33 passed`
- Campaign dry-run confirms:
  - training uses `--slots 300`,
  - training logs every step,
  - evaluation covers `300/1200/3000`,
  - evaluation uses `--eval-both`,
  - resource limits are propagated to training and evaluation commands.
- LaTeX build check succeeds in a temporary output directory:
  - `main.tex` builds to `main_check.pdf`,
  - `supplement.tex` builds to `supplement_check.pdf`.

## Remaining Paper Risks

- Phase5 uses 20 training episodes and four N=100 evaluation episodes per main beamwidth; it is enough for a current method probe but not yet a high-confidence multi-seed final campaign.
- The contention-aware actor improves collision-penalized discovery and collision counts, but at 10/15 degrees its raw discovery is slightly below the collision-reward actor.
- B=3/B=5 remain stress cases and should not be claimed as solved regimes.
- A future final campaign should add more evaluation seeds for phase5 at 10/15/30 degrees before submission.
