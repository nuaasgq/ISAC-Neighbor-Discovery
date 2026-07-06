# Phase10 09:30 Acceleration Plan

Created: 2026-07-07 00:45 CST
Deadline: 2026-07-07 09:30 CST

## Current State

The small-scale training condition is fixed at N=10, B=10 deg, and 300 slots per episode. Large-scale transfer tests use N=100, 3000 slots, and stochastic evaluation. B=30 is excluded for this deadline window.

Completed evidence:

- Phase9 five-way N=100 transfer comparisons at B=10 and B=15.
- Phase10 gated-contention training for seeds 20260731, 20260732, and 20260733.
- Phase10 seed31 N=100 transfer at B=10/B=15.
- Phase10 seed33 N=100 transfer at B=10/B=15.
- Step-axis learning curves for the three gated-contention training seeds.
- Adaptive gated-contention actor v2 code path and smoke tests.

Running at creation time:

- seed32 transfer, B=10: `05_simulation/results_raw/marl_campaign/p10_gate32_n100/b10`
- seed32 transfer, B=15: `05_simulation/results_raw/marl_campaign/p10_gate32_n100/b15`

## P0 Before 03:30

- Finish seed32 N=100 transfer at B=10/B=15.
- Aggregate seed32 transfer results.
- Regenerate gated seed-level comparison with seed31, seed32, and seed33.
- Record the actual three-seed tradeoff:
  - seed31: strongest collision suppression.
  - seed33: strongest topology recovery.
  - seed32: tie-breaker for stability.

## P1 Before 06:30

- If CPU and memory remain stable, train one adaptive gated-contention v2 seed:
  - N=10, B=10 deg, 300 slots.
  - 100 episodes, 3 eval episodes every 10 episodes.
  - `--torch-threads 2`.
- Do not start adaptive v2 transfer until the training curve is at least not clearly degenerate.

## P2 Before 09:30

- If adaptive v2 training finishes by 07:30 and looks viable, run only B=10/B=15 N=100 transfer for that checkpoint.
- Otherwise, spend the remaining time on figure QA, result notes, and local git commit.

## Resource Guard

- Maximum two N=100 evaluation processes in parallel.
- Add one N=10 training process only after at least one N=100 evaluation finishes or CPU is below 60% with at least 10 GB free RAM.
- Stop adding new jobs if system memory exceeds 85%.
- Keep all figures at 4:3 with Times-compatible serif fonts.

## Paper Claim Guard

The current result supports a qualified claim, not an all-metric dominance claim:

> ISAC-assisted decentralized MARL can use local sensing-derived candidate evidence and contention feedback to improve finite-time effective discovery under N=10 to N=100 transfer, but the access-control rule must balance collision suppression against topology recovery.

The key open question for the next run is whether adaptive gating can combine seed31-like CPD with seed33-like lambda2.
