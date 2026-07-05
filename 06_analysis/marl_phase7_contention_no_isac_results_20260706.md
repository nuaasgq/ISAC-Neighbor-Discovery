# Phase-7 Contention No-ISAC Training Results - 2026-07-06

## Scope

- Campaign: `phase7_contention_no_isac_100ep_3seed`
- Method: `MAPPO + contention_shared + collision_topology`
- ISAC features: disabled
- Environment protocol: `structured_marl_no_isac`
- Training setting: `N=10`, `B=10 deg`, `300 slots/episode`, `100 episodes`
- Seeds: `20260731`, `20260732`, `20260733`

This is a true-MARL no-ISAC counterpart, not a rule proxy or CEM-tuned policy.
However, it was produced before the strict no-ISAC simulator patch that prevents
explicit `sense` actions from updating occupancy belief. Treat it as a
pre-strict conservative reference, not the final five-class paper result.

## Artifacts

- Tables: `06_analysis/paper_tables/marl/phase7_contention_no_isac_100ep_3seed_learning_curves/`
- Figures: `06_analysis/paper_figures/marl/phase7_contention_no_isac_100ep_3seed_learning_curves/`
- Rows exported:
  - step rows: `90000`
  - episode rows: `300`
  - eval rows: `198`
- Figures exported: 15 step/episode/eval/resource curves using `training_step`
  as the x-axis.

## Key Observations

Final training rows are zero-link for all three seeds:

| Seed | Training step | Final discovery | Final lambda2 | Final collisions |
|---:|---:|---:|---:|---:|
| 20260731 | 30000 | 0.0 | 0.0 | 0 |
| 20260732 | 30000 | 0.0 | 0.0 | 0 |
| 20260733 | 30000 | 0.0 | 0.0 | 0 |

Across all 300 training episodes, the maximum discovery rate is only `0.0222`
and the mean discovery rate is `0.00015`. Across 198 deterministic/stochastic
evaluation rows, the maximum discovery rate is also `0.0222` and the mean is
`0.00011`. The maximum observed `lambda2` is `0` in both training and evaluation.

## Interpretation

The no-ISAC contention-aware MAPPO actor does not learn an effective discovery
policy in the high-dimensional blind narrow-beam setting, even with the same
network class and collision/topology reward shaping used by the ISAC variant.
This supports the paper's mechanism claim that ISAC-derived empty-beam
elimination and candidate-beam reinforcement are not just auxiliary features;
they are needed to make the link-layer learning problem searchable.

The result should be stated carefully. It shows failure of this no-ISAC MARL
counterpart under the current controlled setting, not an impossibility theorem
for every possible no-ISAC algorithm.

## Paper Use

- Use as pre-strict training-side evidence that no-ISAC MARL is already weak
  even under a looser information boundary.
- Replace with `phase7_contention_no_isac_strict_100ep_3seed` before making
  final five-class endpoint claims at `N=100`.
- In figures, prefer:
  - `marl_episode_discovery_curve.png`
  - `marl_episode_lambda2_curve.png`
  - `marl_step_reward_curve.png`
  - `marl_eval_discovery_curve.png`
