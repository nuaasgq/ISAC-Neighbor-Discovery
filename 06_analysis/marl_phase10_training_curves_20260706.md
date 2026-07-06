# Phase10 Gated Contention Training Curves

Created: 2026-07-06

## Scope

This note records the formal three-seed training curves for the gated contention actor. Each model was trained at N=10, B=10 deg, 300 slots per episode, and 100 episodes.

## Runs

| Seed | Run directory |
|---:|---|
| 20260731 | `05_simulation/results_raw/marl_campaign/phase10_gated_contention_actor_100ep_3seed/train/train_n10_b10_gated_contention_actor_100ep_300slot_seed20260731` |
| 20260732 | `05_simulation/results_raw/marl_campaign/p10_gate_train_32_33/train/train_n10_b10_gated_contention_actor_100ep_300slot_seed20260732` |
| 20260733 | `05_simulation/results_raw/marl_campaign/p10_gate_train_32_33/train/train_n10_b10_gated_contention_actor_100ep_300slot_seed20260733` |

## Artifacts

- Tables: `06_analysis/paper_tables/marl/p10_gate_training_3seed_100ep_step_curves`
- Figures: `06_analysis/paper_figures/marl/p10_gate_training_3seed_100ep_step_curves`
- Figure count: 19 PNGs
- Figure size: 1920 x 1440 pixels for every PNG
- X-axis: true training step for step, episode, evaluation, and resource curves

## Final Train Rows

| Seed | Final discovery | Final CPD | Final lambda2 | Final collisions |
|---:|---:|---:|---:|---:|
| 20260731 | 0.6000 | 0.3140 | 3.3251 | 41 |
| 20260732 | 0.6444 | 0.2358 | 3.1573 | 78 |
| 20260733 | 0.8000 | 0.4000 | 5.4187 | 45 |

## Last-10-Episode Mean

| Seed | Discovery | CPD | Lambda2 | Collisions |
|---:|---:|---:|---:|---:|
| 20260731 | 0.6889 | 0.3984 | 3.8413 | 34.5 |
| 20260732 | 0.7089 | 0.2762 | 3.8953 | 71.8 |
| 20260733 | 0.7800 | 0.3041 | 4.8483 | 71.8 |

## Interpretation

The three seeds all converge to non-trivial discovery and connected topologies in the N=10 training environment. The final performance varies across seeds, so paper figures should show mean bands rather than a single cherry-picked trace.

The strongest training-side evidence is that the policy learns to maintain high discovery while keeping the collision-penalized discovery rate positive and stable. The weakness is seed variability in CPD and collisions, which should motivate either a stability-tuned reward schedule or a topology-recovery gate refinement before claiming a mature final algorithm.

