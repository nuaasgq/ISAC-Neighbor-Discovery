# MARL+ISAC Deadline Plan to 2026-07-08 09:30

## Objective

By 2026-07-08 09:30 Beijing time, produce a compact paper-ready experiment package for ISAC-assisted narrow-beam UAV neighbor discovery. The package must support the main claim with real MARL training, ISAC/no-ISAC comparisons, random/reference baselines, scalable transfer from small training to large testing, and clear learning curves.

## Acceleration Rules

- Keep training small-scale: `N=10`, `B=10 deg`, `300 slots/episode`.
- Keep transfer testing paper-facing: `N=100`, `3000 slots`, `B=10 deg` and `B=15 deg`.
- Do not run `B=30 deg`; it was explicitly removed from the campaign.
- Prefer two-seed parallel training for one high-value method over many weak variants.
- Limit each training/evaluation process to `torch_threads=2`; monitor CPU and memory before launching more work.
- Stop a candidate method if early small-scale stochastic evaluation collapses or if large-scale transfer is dominated on all four core metrics.

## Core Metrics

- Discovery rate.
- Collision-penalized discovery rate (CPD).
- Algebraic connectivity `lambda2`.
- Collision count and collisions per discovery.

## Remaining Batches

1. Archive Phase10 v3 results and figures.
2. Train the balanced topology gate v4 with two seeds.
3. Transfer-evaluate the stronger v4 checkpoint at `N=100`, `B=10/15`, `3000 slots`, stochastic policy.
4. Aggregate v4 against random, reference-like, no-ISAC MARL, contention actor, and gated/adaptive variants.
5. Select the paper main method and move weaker variants into ablation.
6. Generate final tables, learning curves, transfer figures, and method tradeoff figures.
7. Update the paper outline and experiment narrative around the final result.

## Decision Rule

The main method should not be chosen by discovery rate alone. A method is paper-favorable if it improves CPD or collision efficiency while preserving enough topology growth to keep `lambda2` competitive under `N=100` transfer. If no single method dominates, present the result as a tunable cross-layer access-control frontier and use the best CPD method as the default protocol profile.
