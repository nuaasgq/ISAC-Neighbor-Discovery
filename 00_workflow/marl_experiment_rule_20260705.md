# MARL Experiment Rule

This file is the active rule for the rebuilt MARL + ISAC experiment line.

## Method Identity

- The main method must be trained through the real slot-level `MarlNeighborDiscoveryEnv`.
- CEM / heuristic parameter search can only be reported as a baseline, teacher, or warm-start source.
- The paper must not describe heuristic policy search as MARL.

## Training Horizon

- Main training uses the small-scale setting:
  - `N = 10`
  - beamwidth `10 deg`
  - `300 slots` per episode
  - slot duration `5 ms`
- Training configs and commands must keep `300 slots` as the default episode length.
  Do not extend MARL training episodes to `1200` or `3000` slots unless a separate
  ablation is explicitly marked as long-horizon training.
- Long horizons are evaluation-only:
  - `300 slots`
  - `1200 slots`
  - `3000 slots`
- Treat older `600-slot` figures as historical or supplementary stress-window
  evidence only. New MARL transfer claims should be based on the 300-slot-trained
  policies evaluated at the declared test horizons above.

## Version Control

- The canonical development repository is `nuaasgq/ISAC-Neighbor-Discovery`.
- Every code, configuration, analysis, figure, and paper update must be committed
  to git before it is treated as part of the active research workflow.
- If GitHub push is unavailable, keep local commits as the minimum versioned
  record and push when network/credential access is restored.

## Transfer Tests

- Train small and test large without fine-tuning.
- Phase-1 transfer matrix:
  - `N = 10, 20, 50`
  - beamwidth `5, 10, 15, 30 deg`
  - evaluation horizon `300, 1200, 3000 slots`
- Phase-2 stress matrix after resource validation:
  - add `N = 100`
  - add beamwidth `3 deg`
  - compare fixed-area and equal-density scaling.

## Required Logs

Every MARL run must save:

- per-step reward curve,
- per-episode return curve,
- policy/value loss,
- entropy,
- discovery rate,
- mean/P95/P99 discovery delay,
- empty-scan ratio,
- collision count,
- discovered edges,
- `lcc_ratio`,
- `lambda2`,
- RSS memory and system memory usage.

## Evaluation Policy

- Report stochastic and deterministic evaluation separately when both are available.
- If deterministic argmax decoding collapses, the paper must state this and justify stochastic decentralized execution.
- Stochastic results must not be silently relabeled as deterministic results.

## Current Campaign

The current campaign entry point is:

```powershell
python 05_simulation/run_marl_campaign.py --campaign phase1_short_train_long_eval --train-episodes 20 --train-slots 300 --eval-episodes 3 --eval-slots 300 1200 3000 --node-counts 10 20 50 --beamwidths 5 10 15 30 --algorithms isac_mappo mappo --hidden-dim 64 --ppo-epochs 2 --torch-threads 2
```

## Final Long Evaluation Campaign

The final paper-grade transfer evaluation reuses fixed 300-slot-trained MARL
checkpoints and only extends the test horizon:

```powershell
python 05_simulation/run_marl_final_eval_campaign.py --campaign phase6_final_long_eval_10ep_stoch --eval-episodes 10 --eval-slots 3000 --node-counts 100 --beamwidths 3 5 10 15 30 --methods legacy_shared collision_reward contention_actor --torch-threads 2 --max-workers 2
```

For a lower-cost intermediate check, use `--eval-slots 1200 3000` with fewer
episodes or narrower beamwidth subsets. Do not use this script for training.
The evaluator uses an exact fast-evaluation path by default: it skips per-slot
metrics and rich step info that are not consumed by transfer evaluation, while
keeping final `summarize()` metrics unchanged.
