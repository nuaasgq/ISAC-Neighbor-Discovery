# Phase10 Deadline Plan

Created: 2026-07-06 21:30 CST
Deadline: 2026-07-07 09:30 CST

## Objective

Before the deadline, prioritize evidence that can support a paper-grade claim for scalable MARL+ISAC narrow-beam UAV neighbor discovery. The key question is whether the gated contention actor trained only on the small-scale case (N=10, B=10 deg, 300 slots) transfers to larger swarms and different beamwidths while improving collision-penalized discovery and topology quality.

## Priority Order

### P0: Must Finish

- Complete Phase10 gated contention actor training for three seeds:
  - seed 20260731: done
  - seed 20260732: running
  - seed 20260733: pending
- Run N=100, 3000-slot transfer tests for the completed seed20260731 checkpoint:
  - B=10 deg, stochastic, 10 episodes
  - B=15 deg, stochastic, 10 episodes
- Generate step-axis learning curves from `step_rewards.csv`, not episode-only traces.
- Compare Phase10 gated contention actor against Phase9 baselines for B=10 and B=15:
  - uniform random
  - SkyOrbs-like reference baseline
  - MAPPO without ISAC
  - contention actor without ISAC
  - contention actor with ISAC
  - gated contention actor with ISAC

### P1: Finish If P0 Is Stable

- Run transfer grid for the best available gated checkpoint:
  - N in {10, 20, 50, 100}
  - B in {3, 5, 10, 15} deg
  - 3000 slots, stochastic
- Add N=100 area-scaling tests:
  - fixed-area dense case: 260 x 260 x 90 m
  - density-preserving isotropic scaling
  - anisotropic horizontal scaling

### P2: Finish If Resources Allow

- Repeat N=100 B=10/B=15 transfer on seed20260732 and seed20260733 final checkpoints for training-seed robustness.
- Add ISAC imperfection sweeps:
  - miss detection
  - false alarm
  - angular offset
  - sensing period
- Draft paper-ready result notes and figure captions.

## Resource Policy

- Keep at most one MARL training process and two N=100 evaluation processes running concurrently.
- Add only light N=10/20/50 evaluations when CPU is below 60% and system memory is below 85%.
- Do not run B=30 experiments; B=15 is the widest beam case for the current deadline window.
- Prefer shorter output paths to avoid Windows path-length failures.

## Current Running Jobs

- Training campaign: `phase10_gated_contention_actor_100ep_3seed`
- Fast-track transfer campaign: `p10_gate31_n100`
  - `b10`: N=100, B=10 deg, 3000 slots, 10 episodes
  - `b15`: N=100, B=15 deg, 3000 slots, 10 episodes

## Acceptance Check at Deadline

The deadline package should include:

- raw CSV metrics
- aggregated summary tables
- step-axis training curves
- N=100 B=10/B=15 method comparison figures
- short interpretation note that explicitly states whether the current results support the innovation claim

