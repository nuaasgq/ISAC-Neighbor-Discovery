# Clean CTDE Planar N=10/B=15 Screen (2026-07-10)

## Contract

- Actor inputs: own state, local ISAC belief/history, local candidate score/mask, local topology deficit, and post-handshake exchanged neighbor/sensing tables.
- Actor exclusions: pair-derived rendezvous phase, deterministic TX/RX role hint, behavior cloning, beam/role action targets, rule residual, and handcrafted mode prior.
- Critic: centralized pooled training state; absent during evaluation.
- Action: one RF chain, `TX/RX x 24 azimuth beams`; no standalone Sense or Idle.

## Setup

- `N=10`, 2-D Gauss-Markov mobility, 15-degree azimuth beams;
- 3500 m by 3500 m plane, communication and sensing ranges 18 km;
- 300 slots per episode, 5 ms per slot;
- 30 training episodes (9000 environment steps), one training seed;
- 10 paired held-out scenarios.

## Held-Out Results

| Method | Discovery rate | Mean edges | TX fraction | Mean aligned opportunities |
|---|---:|---:|---:|---:|
| Clean CTDE, discovery-first | 0.2956 | 13.3 | 0.5909 | 16.2 |
| Clean CTDE, stable reward | 0.2733 | 12.3 | 0.5289 | 14.5 |
| Wang2025 | 0.5444 | 24.5 | 0.4977 | 30.4 |
| Uniform random | 0.1822 | 8.2 | 0.4978 | 9.7 |

Paired clean discovery-first minus random: +0.1133, 95% CI `[+0.0386, +0.1881]`, 9 wins/0 ties/1 loss, exact sign-flip `p=0.013672`.

Paired clean discovery-first minus Wang2025: -0.2489, 95% CI `[-0.3561, -0.1417]`, 0 wins/0 ties/10 losses, exact sign-flip `p=0.001953`.

## Training Diagnosis

- Discovery-first improves from a first-10-episode mean of 0.1844 to a last-10-episode mean of 0.2800.
- The shared policy first collapses toward RX and then overshoots toward TX. Its last-10-episode TX fraction is about 0.706.
- Stable reward improves held-out TX balance from 0.5909 to 0.5289 but reduces discovery by 0.0222; the paired difference is not significant (`p=0.429688`).
- The observed clean gain over random is real preliminary learnability evidence. It does not yet beat the Wang candidate-table protocol and is not paper-ready.

## Decision

Keep discovery-first as the current clean baseline. Do not select the stable reward variant. The next method task is an action-teacher-free role-balance objective or role-decorrelation network component trained from centralized batches while keeping each actor's execution observation local. Validate that component first on the same N=10 screen and then on untouched training/evaluation seeds.

Focused tables and figures:

- `06_analysis/paper_tables/clean_ctde_planar_n10_b15_focused_20260710/`
- `06_analysis/paper_figures/clean_ctde_planar_n10_b15_focused_20260710/`
