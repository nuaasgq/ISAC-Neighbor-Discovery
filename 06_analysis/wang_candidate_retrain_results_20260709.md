# Wang-candidate MARL retraining results, 2026-07-09

## Purpose

This run follows the revised research direction: use Wang's sensing-table rule (`Flag/Node_num/Dis_num`) as the shared ISAC candidate-pool abstraction, then test whether MARL contributes adaptive access control and whether the learned beam head adds value beyond Wang-candidate random beam selection.

Setting: `N = 10`, `RF = 1`, `200` slots, paired evaluation seeds `2026084001` to `2026084005`, `env_protocol = wang2025_isac_tables`, `candidate_source = wang_table`, standalone SENSE disabled.

## Training

Model: `isac_mappo`, `contention_shared`, discovery-first reward, candidate mask/score, topology deficit, rule residual. Training was run for 160 episodes, but policy quality peaked early and then collapsed into an over-idle regime.

Key training observation:

- `checkpoint_ep00040.pt` is the best evaluated checkpoint.
- `checkpoint_ep00080.pt` remains usable but weaker.
- `final_model.pt` is not a good final policy; later PPO updates pushed the actor toward excessive idle and low TX.

This is important for the method section: Wang-candidate MARL needs early stopping or a stronger access-entropy / TX-budget regularizer.

## Results

| Method | Discovery rate | Mean delay | Lambda2 | Empty-scan ratio | Idle actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wang random mode + random candidate beam | 0.2489 | 181.71 | 0.000 | 0.641 | 0.0 |
| Matched marginal random mode + Wang random beam | 0.1689 | 186.39 | 0.000 | 0.657 | 254.0 |
| Old MARL mode + Wang random beam | 0.6133 | 147.95 | 2.812 | 0.385 | 269.4 |
| Retrained ckpt40 mode + Wang random beam | 0.7022 | 133.96 | 3.457 | 0.348 | 60.4 |
| Retrained ckpt40 full mode + beam | 0.7022 | 127.28 | 3.597 | 0.334 | 66.4 |
| Retrained ckpt80 mode + Wang random beam | 0.5911 | 146.86 | 2.383 | 0.325 | 254.4 |
| Retrained ckpt80 full mode + beam | 0.6178 | 141.83 | 2.827 | 0.310 | 270.2 |
| Retrained final full mode + beam | 0.4444 | 161.63 | 1.455 | 0.344 | 675.6 |

## Beam-head evidence

The beam-head evidence is still weak.

Paired discovery-rate deltas for `full policy - same-checkpoint mode + Wang random beam`:

- `checkpoint_ep00040`: mean delta `0.0000`, seed deltas `[-0.0222, 0.0222, 0.1111, -0.0444, -0.0667]`.
- `checkpoint_ep00080`: mean delta `0.0267`, seed deltas `[0.0667, 0.1111, 0.1111, 0.0000, -0.1556]`.

The full policy reduces empty scans and slightly improves lambda2 at ckpt80, but it does not yet give a robust discovery-rate advantage over Wang-candidate random beam selection. The honest claim is therefore:

1. Wang's sensing-table candidate pool is a strong ISAC abstraction and should be retained.
2. MARL provides a clear adaptive access-control gain over Wang's random TX/RX.
3. Learned beam selection remains an open method gap; current evidence does not justify claiming a strong learned beam-control contribution.

## Artifacts

- `06_analysis/paper_tables/wang_candidate_retrain_20260709/aggregate_metrics.csv`
- `06_analysis/paper_tables/wang_candidate_retrain_20260709/paired_beam_head_deltas.csv`
- `06_analysis/paper_tables/wang_candidate_retrain_20260709/wang_candidate_retrain_training_curves.png`
- `06_analysis/paper_tables/wang_candidate_retrain_20260709/wang_candidate_retrain_ablation_metrics.png`
- raw training/evaluation root: `05_simulation/results_raw/marl_campaign/wang_candidate_retrain_20260709/`
