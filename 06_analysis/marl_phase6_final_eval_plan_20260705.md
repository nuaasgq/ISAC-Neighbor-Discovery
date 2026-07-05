# Phase-6 Final MARL Long-Evaluation Plan - 2026-07-05

## Purpose

Phase 6 is the paper-evidence campaign for the rebuilt real MARL + ISAC line.
It does not train new policies. It reuses fixed small-scale policies trained at
`N=10`, `10 deg`, and `300 slots/episode`, then evaluates zero-shot transfer
under longer test horizons.

This keeps the experimental claim clean:

- Training horizon: `300 slots/episode`.
- Primary test horizon: `3000 slots/episode`.
- Optional intermediate test horizon: `1200 slots/episode`.
- Slot duration: `5 ms`.

## Checkpoints

| Method key | Checkpoint | Training identity |
|---|---|---|
| `legacy_shared` | `05_simulation/results_raw/marl_campaign/phase1_short_train_long_eval/train/train_n10_b10_isac_mappo_300slot/final_model.pt` | shared ISAC-MAPPO, legacy reward |
| `collision_reward` | `05_simulation/results_raw/marl_campaign/phase4_shared_collision_train/train/train_n10_b10_isac_mappo_shared_collision_topology_300slot/final_model.pt` | shared ISAC-MAPPO, collision/topology reward |
| `contention_actor` | `05_simulation/results_raw/marl_campaign/phase5_contention_shared_v2_train/train/train_n10_b10_isac_mappo_contention_shared_collision_topology_300slot/final_model.pt` | contention-aware shared ISAC-MAPPO, collision/topology reward |

## Main Matrix

- Methods: `legacy_shared`, `collision_reward`, `contention_actor`.
- Test nodes: `N=100`.
- Test beamwidths: `3, 5, 10, 15, 30 deg`.
- Test episodes per point: default `10`.
- Policy decoding: stochastic by default; deterministic can be added with
  `--eval-both` and must be reported separately.
- Communication and sensing ranges: default `900 m` for both, deliberately
  non-binding for the first single-hop validation step. This is an experimental
  control, not a general physical-layer range equivalence claim.

## Commands

Dry-run:

```powershell
python 05_simulation/run_marl_final_eval_campaign.py --dry-run --campaign phase6_final_long_eval_10ep_stoch --eval-episodes 10 --eval-slots 3000 --node-counts 100 --beamwidths 3 5 10 15 30 --methods legacy_shared collision_reward contention_actor --torch-threads 2
```

Full long evaluation:

```powershell
python 05_simulation/run_marl_final_eval_campaign.py --campaign phase6_final_long_eval_10ep_stoch --eval-episodes 10 --eval-slots 3000 --node-counts 100 --beamwidths 3 5 10 15 30 --methods legacy_shared collision_reward contention_actor --torch-threads 2
```

Staged lower-risk execution:

```powershell
python 05_simulation/run_marl_final_eval_campaign.py --campaign phase6_final_long_eval_b10_b30_10ep_stoch --eval-episodes 10 --eval-slots 3000 --node-counts 100 --beamwidths 10 15 30 --methods legacy_shared collision_reward contention_actor --torch-threads 2
python 05_simulation/run_marl_final_eval_campaign.py --campaign phase6_final_long_eval_b5_10ep_stoch --eval-episodes 10 --eval-slots 3000 --node-counts 100 --beamwidths 5 --methods legacy_shared collision_reward contention_actor --torch-threads 2
python 05_simulation/run_marl_final_eval_campaign.py --campaign phase6_final_long_eval_b3_10ep_stoch --eval-episodes 10 --eval-slots 3000 --node-counts 100 --beamwidths 3 --methods legacy_shared collision_reward contention_actor --torch-threads 2
```

## Outputs

The campaign script writes raw runs under:

`05_simulation/results_raw/marl_campaign/<campaign>/eval/<method>/<run>/`

Each run must contain:

- `eval_episode_metrics.csv`
- `resource_log.csv`
- `manifest.json`
- `progress.json`

Unless `--no-aggregate` is used, the script refreshes:

- `06_analysis/paper_tables/marl/<campaign>_<method>/marl_transfer_summary.csv`
- `06_analysis/paper_figures/marl/<campaign>_<method>/`
- `06_analysis/paper_tables/marl/<campaign>_method_comparison/marl_method_comparison.csv`
- `06_analysis/paper_figures/marl/<campaign>_method_comparison/`

## Success Criteria

The most defensible paper claim is not "maximum raw discovery under every
beamwidth." The current evidence points to a stronger cross-layer claim:

- ISAC-assisted MARL improves useful candidate selection under narrow beams.
- Collision/topology reward shaping improves network-quality-aware discovery.
- The contention-aware actor should reduce collisions and improve
  collision-penalized discovery, especially at larger swarms.

For submission-grade evidence, Phase 6 should show:

- materially lower `collision_count` and `collisions_per_discovery_censored`
  for `contention_actor` versus `legacy_shared`;
- higher or comparable `collision_penalized_discovery_rate`;
- no severe collapse of `lambda2` or largest-component metrics;
- explicit stress-boundary reporting if `3 deg` or `5 deg` remains weak.

## Resource Discipline

`3 deg` means `120 x 60 = 7200` beams. With `N=100` and `3000` slots, this is
expensive. The final campaign therefore:

- runs commands sequentially;
- defaults to `--torch-threads 2`;
- logs resources every `500` slots;
- enforces `--max-rss-mb 10000` and `--max-system-memory-percent 90`;
- skips completed runs unless `--force` is set.
