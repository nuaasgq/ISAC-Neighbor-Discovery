# Wang candidate-pool MARL ablation, 2026-07-09

## Question

This ablation tests whether the current MARL advantage mainly comes from our candidate-pool design or from the learned action controller itself.

All rows use `N = 10`, `RF = 1`, `200` slots, paired seeds `2026084001` to `2026084005`, and the same `wang2025_isac_tables` environment. The Wang baseline uses the strict sensing-table implementation with `Flag/Node_num/Dis_num`.

## Compared rows

- `wang_random_mode_random_beam`: Wang baseline. TX/RX is random and beam is sampled uniformly from the Wang `Flag = 1` candidate pool.
- `marl_mode_policy_beam_wang_random`: MARL decides TX/RX/IDLE, but the executed beam is sampled uniformly from the Wang `Flag = 1` candidate pool. This isolates the learned access/role controller.
- `marl_mode_policy_beam_policy_wang_candidate`: MARL sees the Wang candidate mask/score and chooses both mode and beam under that mask.
- `marl_mode_policy_beam_policy_default_candidate`: original MARL checkpoint rerun with its default candidate-mask source.

The three MARL rows use the existing checkpoint trained before this Wang candidate-source ablation. Therefore this is a diagnostic result, not the final retrained comparison.

## Results

| Method | Discovery rate | Mean delay | Lambda2 | LCC ratio | Empty-scan ratio | TX | IDLE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wang random mode + random candidate beam | 0.2489 | 181.71 | 0.0000 | 0.88 | 0.641 | 988.4 | 0.0 |
| MARL mode + Wang random candidate beam | 0.6133 | 147.95 | 2.812 | 1.00 | 0.385 | 942.4 | 269.4 |
| MARL mode + beam, Wang candidate mask | 0.5733 | 146.91 | 2.418 | 1.00 | 0.335 | 924.0 | 283.8 |
| MARL mode + beam, default candidate mask | 0.4444 | 157.81 | 1.349 | 1.00 | 0.344 | 945.8 | 255.4 |

## Interpretation

The Wang candidate pool alone does not explain the MARL gain. Under the same Wang `Flag = 1` candidate pool and even with beam selection forced to Wang-style random sampling, the MARL mode controller reaches a discovery rate of `0.6133`, compared with `0.2489` for Wang's random TX/RX and random candidate-beam selection.

The most plausible explanation is that the current MARL checkpoint learns a better access rhythm: it uses fewer TX actions than Wang, introduces about `269` idle actions per episode, and still discovers more links with lower empty-scan ratio. In this N=10 setting, the learned TX/RX/IDLE controller appears more important than learned beam ranking inside the Wang candidate pool.

The beam-policy result is also informative: `marl_mode_policy_beam_wang_random` is slightly higher than `marl_mode_policy_beam_policy_wang_candidate`. This suggests the existing checkpoint was not trained specifically for Wang's broad candidate-mask distribution, so its beam head is not yet optimized for that candidate source. A final fair claim should retrain MARL with `candidate_source=wang_table`.

## Artifacts

- `06_analysis/paper_tables/wang_candidate_ablation_20260709/aggregate_metrics.csv`
- `06_analysis/paper_tables/wang_candidate_ablation_20260709/per_episode_summary.csv`
- `06_analysis/paper_tables/wang_candidate_ablation_20260709/wang_candidate_ablation_metrics.png`
- `05_simulation/results_raw/marl_campaign/wang_candidate_ablation_20260709/`

