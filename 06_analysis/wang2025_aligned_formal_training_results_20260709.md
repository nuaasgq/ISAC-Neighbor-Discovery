# Wang-Aligned Formal MARL Training Results, 2026-07-09

## Run

- Script: `06_analysis/scripts/run_wang2025_aligned_marl_matrix.py`
- Raw root: `05_simulation/results_raw/marl_campaign/wang2025_aligned_formal_20260709`
- Paper tables: `06_analysis/paper_tables/wang2025_aligned_formal_20260709`
- Train setting: N=10, 200 slots, 100 episodes, stochastic MAPPO-style training
- Transfer setting: N=10/20/30/40/50, 200 slots, 5 episodes each
- Common environment: `wang2025_isac_tables`
- Main action space: TX/RX/IDLE plus beam; standalone SENSE disabled
- MARL network: non-gated `contention_shared`

## Fairness Check

The main comparison satisfies the current fairness contract:

- common environment fixed to `wang2025_isac_tables`
- neighbor/sensing table exchange enabled for all main rows
- Wang-style duplicate-edge no-reply enabled
- collision handling shared by all methods
- no standalone SENSE in all main rows

All aggregate main rows have `sense_actions_mean = 0.0`.

## Main Discovery-Rate Results

| N | Wang table action | Uniform TX/RX/IDLE | Budgeted ISAC rule | MARL final |
|---:|---:|---:|---:|---:|
| 10 | 0.9822 | 0.9156 | 0.8089 | 0.8489 |
| 20 | 0.8274 | 0.8221 | 0.6958 | 0.5537 |
| 30 | 0.7228 | 0.7062 | 0.6538 | 0.3963 |
| 40 | 0.6228 | 0.6331 | 0.6215 | 0.3287 |
| 50 | 0.5357 | 0.5433 | 0.5512 | 0.2722 |

The current MARL final model does not beat the Wang-style action policy or the
uniform TX/RX/IDLE baseline under this strict common-environment comparison.
It is only competitive at N=10, then degrades sharply as node count increases.

## Checkpoint-50 Transfer Probe

Because the 50-episode held-out training evaluation was stronger than the
75-episode point, `checkpoint_ep00050.pt` was evaluated separately under the
same transfer matrix.

| N | MARL checkpoint 50 discovery rate |
|---:|---:|
| 10 | 0.8622 |
| 20 | 0.5568 |
| 30 | 0.4317 |
| 40 | 0.3433 |
| 50 | 0.2878 |

Checkpoint 50 is slightly better than the final model for N=30/40/50, but it
still remains far below the Wang/random/rule baselines at scale.

## Interpretation

This run is useful as a negative but important result:

- The strict common-environment comparison is now correct.
- The current MARL policy learns an N=10 behavior but does not transfer to
  larger N.
- The main failure mode is scalable contention: as N grows, the learned policy
  produces high collision counts and low discovery coverage.
- The current `contention_shared` actor with discovery-first reward is not
  sufficient for a paper-quality MARL contribution.

## Next Technical Direction

The next training round should focus on scalable contention control without
adding unfair hidden rules:

- Train with randomized node count, not only N=10.
- Add local density/contention normalization features that are invariant to N.
- Make role probability adaptive to estimated neighbor density and recent
  collision pressure.
- Consider an explicit learned TX probability head or contention-temperature
  head while keeping the executed action space TX/RX/IDLE plus beam.
- Compare against Wang under the same common environment after each change.

The current result should not be used as a positive paper result; it should be
used to motivate the next MARL architecture/reward redesign.
