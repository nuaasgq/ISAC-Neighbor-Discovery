# Phase10 Gated Contention Actor Smoke

Date: 2026-07-06

## Material Passport

- Scope: method-innovation smoke test, not a paper-grade result.
- Objective: reduce the collision bottleneck exposed by Phase9 B15 while preserving the ISAC-assisted candidate-beam prior.
- New network: `gated_contention_shared`
- New method label: `gated_contention_actor`
- Training smoke: `N=10, B=10 deg, 300 slots/episode, 5 episodes, seed=20260731`
- Checkpoint-load transfer smoke: `N=20, B=15 deg, 300 slots, 1 stochastic episode`
- Raw smoke path: `05_simulation/results_raw/marl_campaign/phase10_gated_contention_actor_smoke/`

## Mechanism

The new actor keeps the existing decentralized action schema `Action(mode, beam)` and adds a learned active-access gate inside the contention actor. The gate is computed from each UAV's public local contention/topology state and adjusts only mode logits:

- increases `tx/rx` preference when local topology need and candidate confidence are high;
- suppresses active access when local collision/failure pressure is high;
- shifts probability mass toward `sense/idle` under high contention.

This avoids changing the environment action contract, PPO log-probability shape, or old `contention_shared` checkpoints.

## Smoke Results

Training smoke final episode:

| Episode | Discovery rate | CPD | Lambda2 | Collision count | LCC ratio |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.6889 | 0.2109 | 3.6595 | 102 | 1.0 |

Checkpoint-load transfer smoke:

| Scenario | Discovery rate | CPD | Lambda2 | Collision count | LCC ratio |
|---|---:|---:|---:|---:|---:|
| `N=20, B=15, 300 slots` | 0.5263 | 0.2119 | 5.8235 | 282 | 1.0 |

These numbers prove that the new branch is trainable, serializable, and transferable through `run_marl_evaluate.py`. They are too small to support paper claims.

## Verification

- `python -m py_compile 05_simulation/src/isac_nd_sim/neural_contention_actor_critic.py 05_simulation/run_marl_training.py 05_simulation/run_marl_evaluate.py 05_simulation/run_marl_training_stability_campaign.py 05_simulation/run_marl_campaign.py 06_analysis/scripts/plot_marl_transfer_results.py 06_analysis/scripts/plot_marl_method_comparison.py 06_analysis/scripts/plot_marl_learning_curves.py`
- `python -m pytest 05_simulation/tests/test_actor_critic_probe.py -q` passed `9/9`.
- `python -m pytest 05_simulation/tests/test_marl_fiveway_eval_campaign.py 05_simulation/tests/test_marl_training_stability_campaign.py -q` passed `4/4`.
- `python -m pytest 05_simulation/tests -q` passed `40/40`.
- Training and checkpoint-load transfer smoke both exited with code `0`.

## Next Formal Experiment

Run paper-grade Phase10 training with `300 slots`, three seeds, and the new network:

```powershell
python 05_simulation/run_marl_training_stability_campaign.py --campaign phase10_gated_contention_actor_100ep_3seed --methods gated_contention_actor --seeds 20260731 20260732 20260733 --episodes 100 --slots 300 --eval-episodes 3 --eval-interval 10 --checkpoint-interval 50 --hidden-dim 64 --ppo-epochs 2 --torch-threads 2 --step-log-period 1 --resource-log-period 100 --max-rss-mb 10000 --max-system-memory-percent 90 --command-timeout-seconds 0
```

After this finishes, evaluate the best/representative seed against Phase9 baselines at `N=100`, `B=10/15`, `3000 slots`, and compare three metrics jointly: discovery rate, CPD, and collision count.

