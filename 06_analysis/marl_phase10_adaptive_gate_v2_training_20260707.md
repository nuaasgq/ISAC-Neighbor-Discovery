# Phase10 Adaptive Gate v2 Training Results

Created: 2026-07-07

## Scope

Adaptive gate v2 is a follow-up network-structure variant of the gated contention actor. It keeps the same CTDE MARL framework and decentralized execution interface, but changes the access gate rule to be more collision-adaptive:

- candidate confidence and topology deficit increase active access;
- collision and failure pressure throttle access;
- negative gate values explicitly shift probability toward sense/idle instead of using a symmetric active gate.

The training condition is N=10, B=10 deg, 300 slots, 100 episodes, seed 20260741.

## Artifacts

- Raw training:
  - `05_simulation/results_raw/marl_campaign/p10_adaptive_gate_v2/train/train_n10_b10_adaptive_gated_contention_actor_100ep_300slot_seed20260741`
- Training curves:
  - `06_analysis/paper_tables/marl/p10_adaptive_gate_v2_seed20260741_training_curves`
  - `06_analysis/paper_figures/marl/p10_adaptive_gate_v2_seed20260741_training_curves`
- N=100 transfer:
  - `05_simulation/results_raw/marl_campaign/p10_adaptive_gate_v2_n100/b10`
  - `05_simulation/results_raw/marl_campaign/p10_adaptive_gate_v2_n100/b15`
  - `06_analysis/paper_tables/marl/p10_adaptive_gate_v2_n100_b10_b15_3000slot_10ep_stoch/marl_transfer_summary.csv`
  - `06_analysis/paper_figures/marl/p10_adaptive_gate_v2_n100_b10_b15_3000slot_10ep_stoch`
- Gate-family tradeoff:
  - `06_analysis/paper_tables/marl/p10_gate_seed_and_adaptive_tradeoff_comparison/seed_tradeoff_core_metrics.csv`
  - `06_analysis/paper_figures/marl/p10_gate_seed_and_adaptive_tradeoff_comparison`
- Network implementation:
  - `05_simulation/src/isac_nd_sim/neural_contention_actor_critic.py`

All generated training and transfer figures are 1920 x 1440 pixels.

## Key Observations

At 100 training episodes, stochastic evaluation showed high collision-penalized discovery but lower raw discovery and topology quality:

| Eval phase | Discovery range | CPD range | Lambda2 range | Interpretation |
|---|---:|---:|---:|---|
| deterministic | 0.0000-0.0444 | 0.0000-0.0444 | 0.0000 | deterministic policy is not usable |
| stochastic | 0.5556-0.5778 | 0.4386-0.4561 | 2.1902-2.9109 | strong collision suppression but weak topology |

The final training episode itself had discovery 0.5556, CPD 0.4545, lambda2 3.0173, and 10 collisions.

## N=100 Transfer Results

| Beamwidth | Discovery | CPD | Lambda2 | Collisions |
|---:|---:|---:|---:|---:|
| 10 deg | 0.2203 | 0.2027 | 7.5506 | 429.6 |
| 15 deg | 0.2935 | 0.2265 | 10.9925 | 1463.7 |

Compared with the seed31 low-collision gated checkpoint:

| Beamwidth | Variant | Discovery | CPD | Lambda2 | Collisions |
|---:|---|---:|---:|---:|---:|
| 10 deg | seed31 | 0.3020 | 0.2353 | 11.9043 | 1403.3 |
| 10 deg | adaptive v2 | 0.2203 | 0.2027 | 7.5506 | 429.6 |
| 15 deg | seed31 | 0.3618 | 0.2114 | 16.1730 | 3535.0 |
| 15 deg | adaptive v2 | 0.2935 | 0.2265 | 10.9925 | 1463.7 |

## Interpretation

Adaptive gate v2 successfully moves the policy toward collision suppression, but it overshoots: topology recovery and deterministic deployment quality drop. This is still useful as an ablation because it verifies that the gate can control the CPD/collision side of the tradeoff.

The N=100 transfer confirms the tradeoff. At B=15, adaptive v2 improves CPD over seed31 and cuts collisions by more than half, but it loses raw discovery and lambda2. At B=10, adaptive v2 suppresses collisions even more aggressively, but CPD also drops below seed31.

The paper should frame adaptive v2 as evidence for a controllable access-gating tradeoff and motivate a topology-preserving adaptive gate, not claim adaptive v2 is the final dominant method.
