# Wang-Candidate Stable-Reward MARL Retraining Results (2026-07-09)

## Setup

- Environment: N=10, 105 beams, 200 slots, 5 ms/slot, Wang 2025 sensing-table candidate source, single RF, no separate Sense action.
- Training: MAPPO-style `isac_mappo`, contention-shared network, `discovery_access_stable` reward, 100 episodes, learning rate 2e-4, entropy coefficient 0.02.
- Evaluation: paired seeds 2026084001-2026084005, stochastic execution, full policy vs. same learned mode policy with Wang-candidate random beam executor.

## Key Numbers

- `wang_random_mode_random_beam`: discovery=0.2489, delay=181.71, lambda2=0.000, empty=0.6410
- `matched_marginal_random_mode`: discovery=0.1689, delay=186.39, lambda2=-0.000, empty=0.6568
- `retrained_ckpt40_full_policy`: discovery=0.7022, delay=127.28, lambda2=3.597, empty=0.3339
- `retrained_ckpt40_mode_wang_random_beam`: discovery=0.7022, delay=133.96, lambda2=3.457, empty=0.3479
- `stable_reward_ckpt60_full_policy`: discovery=0.5644, delay=151.06, lambda2=2.222, empty=0.3800
- `stable_reward_ckpt60_mode_wang_random_beam`: discovery=0.5467, delay=153.60, lambda2=2.012, empty=0.3761
- `stable_reward_ckpt100_full_policy`: discovery=0.6000, delay=139.82, lambda2=2.392, empty=0.3669
- `stable_reward_ckpt100_mode_wang_random_beam`: discovery=0.6000, delay=148.09, lambda2=2.850, empty=0.3718

## Beam-Head Ablation

- `ckpt100` full-minus-random-beam: discovery delta=0.0000, delay delta=-8.28 slots, lambda2 delta=-0.459.
- `ckpt60` full-minus-random-beam: discovery delta=0.0178, delay delta=-2.55 slots, lambda2 delta=0.210.

## Interpretation

- The new reward avoided the previous final-checkpoint idle collapse: final evaluation kept roughly 920 TX, 808 RX, and 272 idle actions per episode, with discovery around 0.54 in the internal 6-row final-eval group.
- It did not beat the earlier `discovery_first` best checkpoint. The old ckpt40 full policy remains the strongest observed result here: discovery 0.7022 and mean delay 127.28 slots.
- The learned beam head is still not a robust contribution. At ckpt60 it gives only +0.0178 discovery over Wang-candidate random beam; at ckpt100 the discovery delta is 0.0000, although full policy has lower delay. The defensible contribution remains learned access control plus ISAC/Wang table candidate pruning, not autonomous beam prioritization.
- For the next optimization round, the beam decision needs a stronger learning signal or a different policy factorization. Merely shaping access stability improves collapse behavior but does not solve beam learning.

## Files

- `aggregate_metrics.csv`
- `manifest.json`
- `paired_beam_head_deltas.csv`
- `stable_reward_ablation_metrics.png`
- `stable_reward_eval_per_episode.csv`
- `stable_reward_training_curves.png`
- `training_episode_metrics.csv`
- `training_eval_metrics.csv`
- `training_eval_summary.csv`
