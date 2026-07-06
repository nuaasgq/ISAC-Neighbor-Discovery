# Phase9 B15 Five-Way MARL Transfer Results

Date: 2026-07-06

## Material Passport

- Campaign: `phase9_fiveway_n100_b15_3000slot_10ep_stoch`
- Training source: small-scale checkpoints trained at `N=10, B=10 deg, 300 slots/episode`
- Transfer test: `N=100, B=15 deg, 3000 slots/episode, 10 stochastic episodes`
- Beam grid: `24 x 12 = 288` beams
- Slot duration assumption: `5 ms`; each evaluation episode corresponds to `15 s`
- Communication range and sensing range: both `900 m`, intentionally larger than the test-region diagonal for the current single-hop validation stage
- Compared methods:
  - `uniform_random`: blind random scan baseline
  - `skyorbs_like`: reference-inspired skip-scan protocol baseline, not an exact reproduction of SkyOrbs
  - `mappo_no_isac`: shared MAPPO checkpoint without ISAC observations
  - `contention_no_isac`: contention-aware MAPPO checkpoint without ISAC observations
  - `contention_actor`: contention-aware ISAC-MARL checkpoint with structured ISAC-assisted neighbor-discovery observations
- User-directed scope boundary: `B=30` follow-up is cancelled; the current beamwidth transfer evidence is `B=10` and `B=15`.

## Output Artifacts

- Method table: `06_analysis/paper_tables/marl/phase9_fiveway_n100_b15_3000slot_10ep_stoch_method_comparison/marl_method_comparison.csv`
- Full transfer table: `06_analysis/paper_tables/marl/phase9_fiveway_n100_b15_3000slot_10ep_stoch_all_methods/marl_transfer_summary.csv`
- Method figures: `06_analysis/paper_figures/marl/phase9_fiveway_n100_b15_3000slot_10ep_stoch_method_comparison/`
- Full transfer figures: `06_analysis/paper_figures/marl/phase9_fiveway_n100_b15_3000slot_10ep_stoch_all_methods/`
- Figure QA: all newly generated PNG files are `1920 x 1440`, i.e., `4:3`.

## Main B15 Results

| Method | Discovery rate | Collision-penalized discovery | Lambda2 | LCC ratio | Empty-scan ratio | Collisions |
|---|---:|---:|---:|---:|---:|---:|
| contention_actor | 0.4233 +/- 0.0081 | 0.1387 +/- 0.0066 | 19.8887 +/- 2.1303 | 1.000 | 0.0211 | 10226.9 |
| skyorbs_like | 0.0190 +/- 0.0012 | 0.0190 +/- 0.0012 | ~0 | 0.760 | 0.8324 | 0.4 |
| uniform_random | 0.0129 +/- 0.0011 | 0.0129 +/- 0.0011 | ~0 | 0.388 | 0.8366 | 0.4 |
| contention_no_isac | 0.0043 +/- 0.0005 | 0.0043 +/- 0.0005 | ~0 | 0.059 | 0.8374 | 0.0 |
| mappo_no_isac | 0.0022 +/- 0.0004 | 0.0022 +/- 0.0004 | ~0 | 0.031 | 0.8376 | 0.0 |

The B15 transfer result is strong on discovery efficiency and graph quality. The proposed `contention_actor` reaches a fully connected discovered-neighbor graph in every episode (`LCC ratio = 1.0`, isolated-node ratio `0.0`) and raises algebraic connectivity from near zero to `lambda2 = 19.8887`.

## Gain Summary

| Comparison | Discovery gain | CPD gain |
|---|---:|---:|
| vs. uniform_random | 32.74x | 10.73x |
| vs. skyorbs_like | 22.31x | 7.31x |
| vs. mappo_no_isac | 190.48x | 62.43x |
| vs. contention_no_isac | 97.46x | 31.94x |

## B10 to B15 Interpretation

The earlier B10 transfer campaign gave `contention_actor` discovery rate `0.3429`, CPD `0.2263`, and `lambda2 = 14.3814`. B15 increases the transferred discovery rate to `0.4233` and `lambda2` to `19.8887`, but reduces CPD to `0.1387` because the larger beam footprint and denser successful candidate set induce substantially more contention/collisions.

This means the current evidence supports the core ISAC-assisted neighbor-discovery claim, but the paper should not claim that the current actor fully solves distributed MAC contention. The honest claim is:

> ISAC-derived non-empty-beam evidence gives a scalable, transferable search prior that dramatically improves neighbor discovery and topology formation under narrow beams; the remaining bottleneck is collision-aware contention scheduling after many agents become aware of overlapping useful beams.

## Paper-Usable Claims

1. Small-scale training transfers to a 100-node swarm: the `N=10, B=10` trained `contention_actor` remains effective at `N=100, B=15`.
2. ISAC observations are necessary in this setting: both no-ISAC MARL variants remain near blind-search performance under transfer.
3. The proposed method is not just increasing pair discoveries; it changes graph quality, producing connected discovered-neighbor graphs with nonzero algebraic connectivity.
4. The B15 result strengthens the cross-beamwidth transfer story after the already completed B10 campaign.

## Risk and Next Method Target

The dominant weakness is collision cost. `contention_actor` averages `10226.9` collisions over a 3000-slot episode, while blind/reference baselines have almost no collisions because they also discover very few useful links. The next method innovation should add an explicit distributed contention gate, e.g., a learned per-beam access probability head, local queue/backoff state, or graph-neural contention suppression module, while preserving the ISAC-assisted candidate-beam prior.

