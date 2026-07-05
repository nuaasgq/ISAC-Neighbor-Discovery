# Phase6 B=5 Long-Horizon MARL Transfer Results

Date: 2026-07-06

## Scope

This report summarizes the completed B=5 stress evaluation for the real MARL transfer campaign.

- Campaign: `phase6_final_long_eval_b5_10ep_stoch`
- Training source: `N=10`, `B=10 deg`, `300 slots/episode`
- Test setting: `N=100`, `B=5 deg`, `3000 slots/episode`, `10 stochastic episodes`
- Communication and sensing range: `900 m`, single-hop coverage assumption
- Compared methods:
  - `legacy_shared`: legacy shared ISAC-MAPPO
  - `collision_reward`: shared ISAC-MAPPO with collision/topology reward
  - `contention_actor`: contention-aware ISAC-MAPPO actor with collision/topology reward

Artifacts:

- Raw eval campaign: `05_simulation/results_raw/marl_campaign/phase6_final_long_eval_b5_10ep_stoch/`
- Tables: `06_analysis/paper_tables/marl/phase6_final_long_eval_b5_10ep_stoch_*`
- Figures: `06_analysis/paper_figures/marl/phase6_final_long_eval_b5_10ep_stoch_*`

## Summary

| Method | Discovery rate | Collision-penalized discovery | Lambda2 | Collisions |
|---|---:|---:|---:|---:|
| Legacy ISAC-MAPPO | 0.2591 +- 0.0075 | 0.1444 +- 0.0187 | 10.6331 +- 1.0419 | 4089.0 +- 1415.5 |
| Collision-reward ISAC-MAPPO | 0.2566 +- 0.0068 | 0.1530 +- 0.0143 | 9.9721 +- 1.4259 | 3427.0 +- 926.7 |
| Contention-aware ISAC-MAPPO | 0.2271 +- 0.0130 | 0.1816 +- 0.0099 | 8.7099 +- 1.3146 | 1253.6 +- 439.8 |

The B=5 condition is a narrow-beam stress case. All methods discover only about one quarter of the N=100 single-hop edges within 3000 slots. The contention-aware actor reduces collision count by about 69 percent relative to the legacy shared actor and about 63 percent relative to the collision-reward shared actor.

## Interpretation

The result supports a constrained but useful claim: under very narrow beams, the contention-aware actor sacrifices some raw discovery and algebraic connectivity to substantially reduce handshake contention, improving the collision-penalized discovery objective.

This should not be described as a universal gain in discovery rate. It is stronger evidence for contention control and medium-access efficiency than for raw topology completion under B=5.

The B=5 result should be placed beside B=10/15/30 and the future B=3 stress result. For a TWC/TCOM paper, B=5 should be framed as a difficult operating point where narrow beams expose the collision/coordination tradeoff.

## Next Use

1. Merge B=5 with the existing B=10/15/30 and future B=3 results into a unified beamwidth-transfer figure.
2. Use `collision_penalized_discovery_rate`, `collision_count`, and `collisions_per_discovery_censored` as primary B=5 evidence.
3. Keep `discovery_rate` and `lambda2` in the same table to avoid overstating the proposed actor.
4. Use the patched stochastic-eval seed policy for all future transfer and five-way campaigns.
