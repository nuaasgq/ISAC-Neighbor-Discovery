# MARL Rebuild Status - 2026-07-05

## Current Correction

The previous main training evidence was a CEM-style heuristic policy-parameter search, not a conventional MARL result. It is now demoted to a heuristic baseline / teacher candidate. The active main line is rebuilt around real slot-level multi-agent reinforcement learning.

## Implemented

- Added `05_simulation/run_marl_training.py`.
  - Uses `MarlNeighborDiscoveryEnv.step()` at every slot.
  - Logs per-step rewards to `step_rewards.csv`.
  - Logs per-episode returns and discovery/topology metrics to `episode_metrics.csv`.
  - Supports `ippo`, `mappo`, and `isac_mappo` modes.
  - Supports CTDE-style pooled centralized critic for MAPPO-style runs.
  - Supports ISAC candidate mask, candidate score, topology deficit, and rule-residual architecture switches.
  - Saves `final_model.pt` and optional checkpoints.
  - Logs resource usage with Windows memory fallback.

- Added `05_simulation/run_marl_evaluate.py`.
  - Loads a trained shared policy checkpoint.
  - Evaluates zero-shot transfer under different node counts and beam codebooks.
  - Supports deterministic, stochastic, and dual evaluation modes.

- Added `06_analysis/scripts/plot_marl_learning_curves.py`.
  - Aggregates real MARL step, episode, eval, and resource logs.
  - Produces 4:3 Times-style PNGs using training step / episode axes.

- Fixed idle-action log-prob accounting in `SharedBeamActorCritic`.
  - Idle actions no longer include a spurious beam log-prob term.

## Smoke And Pilot Results

Smoke tests passed and wrote real MARL outputs.

Pilot setting:

- Train: `N=10`, `10 deg` beam grid (`36 x 18 = 648` beams), `300 slots`, `5 episodes`.
- Eval: stochastic and deterministic evaluation both recorded.

Observed pilot behavior:

- ISAC-MAPPO stochastic eval produced high early discovery in the short pilot, around `0.76-0.84` discovery rate at `N=10`.
- Deterministic argmax evaluation can collapse to near-zero discovery, so final experiments must either justify stochastic execution or add a deterministic decoding/coordination layer.
- No-ISAC MAPPO remained at `0` discovery in the same short pilot, with very high empty-scan ratio.
- Zero-shot transfer from trained `N=10, 10 deg` policy to `N=20, 15 deg` stochastic evaluation ran successfully.
  - ISAC-MAPPO pilot transfer produced about `0.52` discovery rate.
  - No-ISAC transfer remained at `0` discovery.

These are pilot diagnostics, not final paper results.

## Immediate Next Steps

1. Add a campaign runner for controlled training and transfer sweeps.
2. Train with short episodes (`300 slots`) and evaluate with longer horizons (`1200/3000 slots`).
3. Compare:
   - fully random,
   - literature-like scan baseline,
   - MAPPO without ISAC,
   - IPPO/MAPPO with ISAC,
   - proposed ISAC architecture with mask/score/topology/rule-residual switches.
4. Evaluate transfer from `N=10, 10 deg` training to:
   - `N = 20, 50, 100`,
   - beamwidth `3, 5, 10, 15, 30 deg`,
   - equal-density and fixed-area scaling.
5. Only after stable multi-seed results should the paper draft claim MARL contribution.

## Training Horizon Correction

A `3000-slot` training attempt was stopped after the first trajectory exceeded practical memory limits. The working protocol is now:

- Training horizon: `300 slots`.
- Test horizons: `300`, `1200`, and `3000 slots`.
- Rationale: learning uses short finite-time interaction windows, while long-horizon tests verify whether the learned policy sustains discovery and topology formation over more slots without making training memory scale with the full test horizon.

The helper `05_simulation/run_marl_campaign.py` encodes this short-train / long-test workflow.
