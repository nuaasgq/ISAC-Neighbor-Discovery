# Discovery-First MARL Round 1

Date: 2026-07-09

## Purpose

This round skips the previous CPD/collision-centered comparison line and starts
the combined WP3/WP4 path:

1. redesign MARL reward toward neighbor discovery and topology quality;
2. train at small scale (`N=10`) and evaluate zero-shot transfer to larger
   swarms and other beam widths.

This is a fast feedback run, not a final paper-data run.

## Code Changes

- Added `reward_version = discovery_first`.
- Kept standalone `SENSE` disabled for campaigns by default. ISAC feedback is
  obtained through TX-coupled piggyback sensing.
- Discovery-first reward emphasizes:
  - new direct links;
  - earlier links;
  - links that connect components;
  - links involving low-degree or isolated nodes;
  - light empty-scan and failed-access costs.
- Collision is retained only as a light failed-access cost, not the dominant
  objective.
- Fixed ideal-sensing accounting so deterministic/ideal TX-coupled sensing also
  increments sensing observation counters.

## Verification

- `pytest -q`: 54 passed.
- MARL env contract now tests `discovery_first`.
- TX-coupled ideal ISAC now records sensing observations even when
  `sense_actions = 0`.

## Smoke Run

Output:

- `05_simulation/results_raw/marl_campaign/discovery_first_smoke_20260709/`

Command profile:

- Train: `N=10`, 24 slots, 2 episodes.
- Eval: `N=10/20`, 24 slots, 1 episode.
- Reward: `discovery_first`.
- Network: `topology_adaptive_gated_contention_shared`.
- Standalone SENSE: disabled.

Result: training, checkpoint save, and transfer evaluation completed.

## Round 1 Run

Output:

- `05_simulation/results_raw/marl_campaign/discovery_first_round1_20260709/`

Command profile:

- Train: `N=10`, `B=10`, 300 slots, 12 episodes.
- Eval: stochastic only, 300 slots, 2 episodes per point.
- Transfer: `N=10/20/50`, `B=10/15`.
- RF: 1.
- Standalone SENSE: disabled.

## Training Snapshot

| Episode | Discovery | Edges | Mean delay | LCC | Lambda2 | Sense |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.533 | 24 | 180.8 | 1.000 | 2.484 | 0 |
| 1 | 0.556 | 25 | 186.9 | 1.000 | 2.304 | 0 |
| 2 | 0.511 | 23 | 195.7 | 1.000 | 2.575 | 0 |
| 3 | 0.511 | 23 | 190.3 | 1.000 | 1.870 | 0 |
| 6 | 0.600 | 27 | 182.2 | 1.000 | 2.987 | 0 |
| 9 | 0.533 | 24 | 182.0 | 1.000 | 3.099 | 0 |
| 11 | 0.622 | 28 | 174.5 | 1.000 | 3.348 | 0 |

The short training run does not prove convergence, but it no longer collapses
to passive behavior. Episode returns and discovery remain positive under
`TX/RX/IDLE` with standalone SENSE disabled.

## Transfer Snapshot

| Test | Discovery | Edges | Mean delay | p95 delay | LCC | Lambda2 | Empty scan | Sense |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N10 B10 | 0.5222 | 23.5 | 190.3 | 300.0 | 1.000 | 1.956 | 0.870 | 0 |
| N10 B15 | 0.6111 | 27.5 | 159.3 | 300.0 | 1.000 | 3.360 | 0.719 | 0 |
| N20 B10 | 0.2868 | 54.5 | 244.2 | 300.0 | 1.000 | 1.927 | 0.743 | 0 |
| N20 B15 | 0.3316 | 63.0 | 227.9 | 300.0 | 1.000 | 2.416 | 0.506 | 0 |
| N50 B10 | 0.1188 | 145.5 | 278.3 | 300.0 | 1.000 | 1.364 | 0.556 | 0 |
| N50 B15 | 0.2122 | 260.0 | 259.9 | 300.0 | 1.000 | 3.986 | 0.228 | 0 |

## Interpretation

The discovery-first reward is moving in the right direction at small scale:
`N=10, B=10` reaches about `0.52` discovery in the 300-slot transfer run, and
the final training episode reaches `0.622`.

Transfer to larger swarms still degrades strongly. At `N=50`, discovery is
`0.1188` for `B=10` and `0.2122` for `B=15`. The topology becomes connected
(`LCC=1.0`) in these tests, but the discovery fraction and tail delays are not
paper-strong yet. The p95 delay is censored at 300 slots in all transfer cases,
so the policy still discovers too slowly.

The result supports continuing the discovery-first MARL line, but not yet a
claim that MARL solves the final problem.

## Caveat

The round-1 discovery, delay, and topology numbers are valid. However, this run
was completed before fixing the ideal-sensing statistics bug, so sensing
observation columns from this run undercount ideal TX-coupled sensing activity.
Future formal runs should be repeated after the fix.

## Next Training Step

Run a longer small-scale training pass before any paper claim:

- `N=10`, `B=10`, 300 slots.
- 100-300 training episodes.
- stochastic evaluation.
- `N=10/20/50/100`, `B=10/15`.
- include a no-ISAC MARL ablation using the same reward and network so the ISAC
  contribution is isolated.

Do not use CPD or collision count to select the best checkpoint. Select by
validation discovery rate, mean/p95 delay, and topology quality.

