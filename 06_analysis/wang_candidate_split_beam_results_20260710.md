# Separated Beam-Loss MARL Probe Results (2026-07-10)

## Setup

- Environment: N=10, 105 beams, 200 slots, Wang 2025 sensing-table candidate source, single RF, no standalone Sense action.
- Mechanism change: mode/beam/gate PPO log-probabilities are tracked separately; optional beam-ranking auxiliary loss fits local candidate-score rankings.
- Heavy probe: beam_loss_coef=1.25, beam_rank_aux_coef=0.04, 80 episodes. Light probe: beam_loss_coef=0.6, beam_rank_aux_coef=0.01, 40 episodes.

## Key Numbers

- `wang_random_mode_random_beam`: discovery=0.2489, delay=181.71, lambda2=0.000, idle=0.0.
- `matched_marginal_random_mode`: discovery=0.1689, delay=186.39, lambda2=-0.000, idle=254.0.
- `retrained_ckpt40_full_policy`: discovery=0.7022, delay=127.28, lambda2=3.597, idle=66.4.
- `retrained_ckpt40_mode_wang_random_beam`: discovery=0.7022, delay=133.96, lambda2=3.457, idle=60.4.
- `split_light_ckpt20_full_policy`: discovery=0.6889, delay=135.46, lambda2=3.404, idle=129.0.
- `split_light_ckpt20_mode_wang_random_beam`: discovery=0.5822, delay=150.06, lambda2=2.507, idle=109.6.
- `split_light_ckpt40_full_policy`: discovery=0.6267, delay=146.16, lambda2=3.185, idle=219.6.
- `split_light_ckpt40_mode_wang_random_beam`: discovery=0.5022, delay=160.38, lambda2=1.647, idle=183.4.

## Beam-Head Ablation

- `ckpt20` full-minus-random-beam: discovery delta=0.1067, delay delta=-14.60 slots, lambda2 delta=0.897.
- `ckpt40` full-minus-random-beam: discovery delta=0.1244, delay delta=-14.22 slots, lambda2 delta=1.538.

## Interpretation

- The heavy split setting was counterproductive: active beam fraction dropped and final/internal evaluation stayed around 0.43-0.50 discovery. This indicates beam auxiliary learning can suppress access learning when over-weighted.
- The light split setting recovered active access and produced the first clear beam-head ablation gain. At ckpt20, full policy reached discovery 0.6889 versus 0.5822 with the same learned mode policy and Wang-candidate random beams, a paired +0.1067 gain.
- The old discovery-first ckpt40 is still slightly higher in absolute discovery at 0.7022, but it had essentially zero beam-head delta. The new result is therefore a better mechanism candidate, not yet a final performance winner.
- Next optimization should keep the light split regime, add an anti-idle/access-preservation term or mode-loss rescaling, and train multiple seeds around 20-40 episodes to verify the beam delta is stable.

## Files

- `aggregate_metrics.csv`
- `manifest.json`
- `paired_beam_head_deltas.csv`
- `split_aux_heavy_train_episode_metrics.csv`
- `split_aux_heavy_train_eval_episode_metrics.csv`
- `split_aux_heavy_train_step_rewards.csv`
- `split_aux_heavy_training_eval_summary.csv`
- `split_beam_ablation_metrics.png`
- `split_beam_training_diagnostics.png`
- `split_light_eval_per_episode.csv`
- `split_light_train_episode_metrics.csv`
- `split_light_train_eval_episode_metrics.csv`
- `split_light_train_step_rewards.csv`
- `split_light_training_eval_summary.csv`
