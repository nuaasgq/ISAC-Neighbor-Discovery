# Phase9 Five-Way B=10/N=100 MARL+ISAC Results

## Material Passport

- Artifact: experiment result report
- Date: 2026-07-06
- Campaign: `phase9_fiveway_n100_b10_3000slot_10ep_stoch`
- Training rule: MARL checkpoints are trained at `N=10`, beamwidth `10 deg`,
  `300 slots/episode`, `100 episodes`, slot duration `5 ms`.
- Evaluation rule: zero-shot transfer to `N=100`, beamwidth `10 deg`,
  `648` 3D beam cells, `3000 slots/episode`, `10` stochastic episodes.
- Range setting: communication range `900 m`, sensing range `900 m`, single-hop
  stress setting for this first paper-grade check.
- Source tables:
  `06_analysis/paper_tables/marl/phase9_fiveway_n100_b10_3000slot_10ep_stoch_method_comparison/marl_method_comparison.csv`
- Source figures:
  `06_analysis/paper_figures/marl/phase9_fiveway_n100_b10_3000slot_10ep_stoch_method_comparison/`
  and
  `06_analysis/paper_figures/marl/phase9_fiveway_n100_b10_3000slot_10ep_stoch_all_methods/`

## Result Summary

All results below are mean values over 10 stochastic 3000-slot evaluation
episodes. The MARL policies are not trained for 3000-slot episodes; 3000 slots
are used only for long-horizon transfer testing.

| Method | Discovery | CPD | Lambda2 | LCC | Empty scan | Collisions |
|---|---:|---:|---:|---:|---:|---:|
| `contention_actor` (improved MARL + ISAC) | 0.3429 +/- 0.0054 | 0.2263 +/- 0.0067 | 14.381 +/- 0.988 | 1.000 | 0.054 | 2563.7 |
| `skyorbs_like` (reference baseline) | 0.0031 +/- 0.0005 | 0.0031 +/- 0.0005 | 0.000 | 0.041 | 0.902 | 0.0 |
| `uniform_random` | 0.0028 +/- 0.0006 | 0.0028 +/- 0.0006 | 0.000 | 0.037 | 0.902 | 0.0 |
| `contention_no_isac` (improved MARL, no ISAC) | 0.0008 +/- 0.0002 | 0.0008 +/- 0.0002 | 0.000 | 0.021 | 0.902 | 0.0 |
| `mappo_no_isac` (shared MAPPO, no ISAC) | 0.0005 +/- 0.0001 | 0.0005 +/- 0.0001 | 0.000 | 0.020 | 0.902 | 0.0 |

CPD is the collision-penalized discovery rate. The ISAC method has much more
contention, but even after the collision penalty it remains far ahead of all
controls.

## Paper-Useful Findings

- ISAC + improved MARL gives a 110.9x discovery gain and 73.2x CPD gain over the
  SkyOrbs-like reference baseline in the `N=100`, `B=10 deg`, `3000-slot`
  transfer test.
- Against fully random blind discovery, the gain is 123.0x in discovery and
  81.2x in CPD.
- Against strict no-ISAC MAPPO, the gain is 707.3x in discovery and 466.7x in
  CPD.
- Against the improved no-ISAC contention network, the gain is 446.7x in
  discovery and 294.8x in CPD.
- Topology quality separates sharply: all four no-ISAC/blind baselines have
  effectively zero algebraic connectivity, while `contention_actor` reaches
  `lambda2 = 14.381` and `LCC = 1.0`.
- The empty-scan ratio drops from about `0.902` for blind/no-ISAC methods to
  `0.054` with ISAC, which directly supports the cross-layer mechanism claim:
  ISAC beam evidence suppresses empty beams and concentrates slot decisions on
  likely neighbor directions.

## Interpretation

This result supports the core research direction. Under a large 3D beam search
space, pure random access, reference-style scanning, and no-ISAC MARL fail to
build useful one-hop topology within the 3000-slot test horizon. The
ISAC-assisted contention actor converts physical-layer beam evidence into a
network-layer neighbor-discovery policy and forms a connected graph in all 10
episodes.

The result is strong enough to justify the paper's main mechanism and method
claim, but it is not yet the full paper matrix. The next required checks are
beamwidth transfer (`3/5/10/15/30 deg`), node-count transfer (`10/20/50/100`),
and equal-density versus fixed-area scaling for `N=100`.

## Caveats

- This report covers only the B=10, N=100, single-hop transfer row.
- The ISAC method trades collisions for much higher discovery and topology
  quality; collision-aware scheduling should be the next method refinement.
- `skyorbs_like` is a reference-inspired protocol baseline, not a one-to-one
  reproduction of a specific paper's full physical-layer design.
- Stochastic decentralized execution is reported here. Deterministic decoding
  should be reported separately if used.

## Figure Set

Primary method-comparison figures:

- `marl_method_comparison_discovery_rate.png`
- `marl_method_comparison_collision_penalized_discovery_rate.png`
- `marl_method_comparison_lambda2.png`
- `marl_method_comparison_collision_count.png`
- `marl_method_comparison_collisions_per_discovery_censored.png`

All checked method-comparison PNGs are `1920 x 1440`, i.e., 4:3 aspect ratio.
