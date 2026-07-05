# Phase7 ISAC-MARL Multi-Seed Training Results

Date: 2026-07-06

## Scope

This report summarizes the first complete multi-seed real MARL training campaign for the ISAC-assisted narrow-beam neighbor-discovery study.

- Campaign: `phase7_long_training_100ep_3seed`
- Training scale: `N=10`, `B=10 deg`, `300 slots/episode`, `5 ms/slot`
- Training budget: `100 episodes`, `3 seeds` (`20260731`, `20260732`, `20260733`)
- Algorithms:
  - `legacy_shared`: shared actor + centralized critic ISAC-MAPPO-style PPO, legacy reward
  - `collision_reward`: shared actor + centralized critic ISAC-MAPPO-style PPO, collision/topology reward
  - `contention_actor`: contention-aware shared actor + centralized critic ISAC-MAPPO-style PPO, collision/topology reward

The run produced `270000` step-level reward records, `900` episode records, and `594` in-training evaluation records. The learning-curve artifacts are in:

- Tables: `06_analysis/paper_tables/marl/phase7_long_training_100ep_3seed_learning_curves/`
- Figures: `06_analysis/paper_figures/marl/phase7_long_training_100ep_3seed_learning_curves/`

The large step-level aggregate CSV is intentionally ignored by Git; the step-level convergence figures and compact episode/eval tables are retained.

## Key Results

Final stochastic evaluation at episode 100:

| Method | Discovery rate | Collision-penalized discovery | Lambda2 | Collisions |
|---|---:|---:|---:|---:|
| Legacy ISAC-MAPPO | 0.7877 +- 0.0471 | 0.2108 +- 0.0338 | 4.8284 +- 0.7858 | 127.5 +- 31.8 |
| Collision-reward ISAC-MAPPO | 0.7975 +- 0.0475 | 0.2560 +- 0.0482 | 4.6887 +- 0.9877 | 99.6 +- 26.8 |
| Contention-aware ISAC-MAPPO | 0.7358 +- 0.0475 | 0.3840 +- 0.0612 | 4.3378 +- 0.6896 | 43.1 +- 13.4 |

Last-10 training-episode averages:

| Method | Discovery rate | Collision-penalized discovery | Lambda2 | Collisions |
|---|---:|---:|---:|---:|
| Legacy ISAC-MAPPO | 0.7800 +- 0.0629 | 0.2340 +- 0.0557 | 4.7403 +- 0.8405 | 112.0 +- 34.2 |
| Collision-reward ISAC-MAPPO | 0.7793 +- 0.0586 | 0.2714 +- 0.0543 | 4.8411 +- 1.0210 | 88.9 +- 26.6 |
| Contention-aware ISAC-MAPPO | 0.7681 +- 0.0546 | 0.3790 +- 0.0541 | 4.6520 +- 0.7084 | 47.8 +- 13.6 |

## Interpretation

The contention-aware actor does not maximize raw discovery rate in the small training environment. Its main gain is reducing contention and improving the collision-penalized discovery metric. This supports a narrower, defensible claim: the proposed contention-aware actor changes the network behavior toward more efficient neighbor discovery under handshake contention.

The collision/topology reward alone also improves over the legacy reward on collision-penalized discovery, but the network-structure change gives the strongest collision reduction.

Deterministic evaluation is poor for all three methods, so the current paper evidence should report stochastic policy evaluation explicitly. The deterministic rows are useful as a diagnostic, not as the main performance claim.

## Figure Set

Useful main-text candidates:

- `marl_step_reward_curve.png`
- `marl_episode_discovery_curve.png`
- `marl_episode_lambda2_curve.png`
- `marl_eval_discovery_curve.png`
- `marl_episode_collision_curve.png`

Supplementary candidates:

- `marl_step_discovery_curve.png`
- `marl_episode_empty_scan_curve.png`
- `marl_policy_loss_curve.png`
- `marl_value_loss_curve.png`
- `marl_resource_rss_curve.png`
- `marl_resource_memory_curve.png`

## Caveats

- This is MAPPO-style CTDE with shared actor, centralized pooled critic, Monte-Carlo returns, and PPO clipping. It should not be described as a full standard MAPPO implementation with GAE, minibatch rollout, or parallel environments.
- `contention_actor` is a contention/topology-aware beam-token actor, not a GNN or inter-agent communication network.
- These in-training eval rows use the training horizon (`300 slots`); long-horizon transfer claims must use the separate `3000-slot` evaluation campaigns.
- The strict no-ISAC replacement campaign is required before final five-way comparison.

## Next Required Experiments

1. Finish `phase6_final_long_eval_b5_10ep_stoch` and aggregate the B=5 long-horizon transfer results.
2. Finish `phase7_contention_no_isac_strict_100ep_3seed`, then generate strict no-ISAC learning curves.
3. Run five-way N=100, B=10/15/30, 3000-slot comparison using the strict no-ISAC checkpoint.
4. Add B=3 stress transfer only after the current heavy jobs finish, because it is the most expensive beamwidth case.
