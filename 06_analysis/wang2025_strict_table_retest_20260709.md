# Wang2025 sensing-table strict retest, 2026-07-09

## Scope

This retest fixes the Wang baseline implementation to follow the paper's sensing-table semantics more strictly. The action policy no longer uses the previous broad `belief >= 0.55` / success-priority candidate heuristic.

Strict Wang table rules now implemented:

- Each node initializes all sensing-table beam entries with `Flag = 1`.
- A transmission beam is selected uniformly from beams whose sensing-table `Flag = 1`.
- If ISAC sensing observes no potential target in a beam, that beam is closed with `Flag = 0`.
- If ISAC sensing observes potential target(s), the simulator records `Node_num` and keeps the beam active while `Dis_num < Node_num`.
- A successful interaction increments `Dis_num` for both endpoints' corresponding beams and closes the beam when `Dis_num >= Node_num`.
- Neighbor-table and sensing-table exchange only boosts beams that are still flagged and still have undiscovered potential targets.

Implemented files:

- `05_simulation/src/isac_nd_sim/simulator.py`
- `05_simulation/tests/test_phy_sensing_and_wang2025.py`
- `06_analysis/scripts/visualize_candidate_dynamics.py`

## N=10 comparison

Setting: `N = 10`, `RF = 1`, `200` slots, paired seeds `2026084001` to `2026084005`, same environment and fixed-handshake contract. MARL uses the existing checkpoint trained before this strict Wang-table correction, so this row is useful as a compatibility check but is not the final fair retrained result.

| Method | Discovery rate | Mean delay | LCC ratio | Lambda2 | Empty-scan ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform TX/RX/IDLE random | 0.0044 | 199.14 | 0.12 | 0.0000 | 0.9272 |
| Budgeted ISAC rule | 0.0622 | 193.68 | 0.30 | 0.0000 | 0.7847 |
| Wang strict sensing-table policy | 0.2489 | 181.71 | 0.88 | 0.0000 | 0.6410 |
| MARL + Wang ISAC tables, existing checkpoint | 0.4444 | 157.81 | 1.00 | 1.3493 | 0.3438 |

Result files:

- `06_analysis/paper_tables/wang2025_strict_table_n10_20260709/combined_aggregate.csv`
- `06_analysis/paper_tables/wang2025_strict_table_n10_20260709/paired_baseline_per_episode.csv`
- `06_analysis/paper_tables/wang2025_strict_table_n10_20260709/marl_existing_checkpoint_per_episode.csv`

## Candidate-set audit

The previous Wang trace was not credible because its candidate abstraction could leave dozens of "positive" candidates for only ten nodes. After switching to explicit `Flag/Node_num/Dis_num` table fields:

| Method and metric | Network mean | Min | Max |
| --- | ---: | ---: | ---: |
| MARL candidate count | 20.263 | 11 | 105 |
| Wang active `Flag=1` count | 13.174 | 0 | 105 |
| Wang positive open-target count | 4.158 | 0 | 12 |

For node 0, Wang active beams decrease from `105,96,90,84,79,73,64,64,58,58` in the first ten slots to `4,3,3,1,1,0,1,1,2,1` in the last ten slots. This is now qualitatively consistent with a sensing table that rapidly excludes empty directions.

Candidate dynamics outputs:

- `06_analysis/paper_tables/candidate_dynamics_strict_wang_20260709/node0_candidate_count.png`
- `06_analysis/paper_tables/candidate_dynamics_strict_wang_20260709/network_mean_candidate_count.png`
- `06_analysis/paper_tables/candidate_dynamics_strict_wang_20260709/node0_candidate_heatmaps.png`
- `06_analysis/paper_tables/candidate_dynamics_strict_wang_20260709/node0_selected_beams.png`

## Interpretation

The strict Wang baseline is much stronger and cleaner than the earlier random-like reproduction, but it still remains below the existing MARL checkpoint under this N=10 test. The observed MARL advantage is not yet a final paper claim because the checkpoint was trained before the strict Wang-table correction. The next necessary step is to retrain MARL in exactly this corrected environment and repeat the same paired-seed comparison.
