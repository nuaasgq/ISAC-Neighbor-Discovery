# Structured MARL Probe Status - 2026-07-05 Morning

## Implemented Interface

- `MarlNeighborDiscoveryEnv` now exposes local `candidate_mask`, `candidate_score`, `topology_deficit`, and `rule_mode_logits`.
- `SharedBeamActorCritic` can optionally use:
  - `use_candidate_mask=True` for hard execution-time beam masking,
  - `use_candidate_score=True` for per-beam score features,
  - `use_topology_deficit=True` for local discovered-degree deficit context,
  - `use_rule_residual=True` for local rule-prior residual logits.
- `run_actor_critic_imitation_probe.py` exposes matching CLI flags:
  - `--candidate-mask`
  - `--candidate-score`
  - `--topology-deficit`
  - `--rule-residual`
  - `--rule-residual-scale`
  - `--eval-both`

## Verification

- `python -m pytest 05_simulation\tests` passed: 27 tests.
- The initial structured smoke command completed:

```powershell
python 05_simulation\run_actor_critic_imitation_probe.py `
  --output 05_simulation\results_raw\structured_imitation_smoke_20260705 `
  --bc-episodes 2 --rl-episodes 1 --eval-episodes 1 --stochastic-eval `
  --slots 12 --node-count 6 --azimuth-cells 6 --elevation-cells 3 `
  --communication-range 900 --sensing-range 900 `
  --false-alarm-rate 0.02 --miss-detection-rate 0.15 `
  --angular-cell-offset-std 0.5 --sensing-period-slots 1 `
  --hidden-dim 24 --learning-rate 0.001 `
  --candidate-mask --candidate-score --topology-deficit --rule-residual `
  --rule-residual-scale 1.0 --seed 2026070501
```

## Completed Probe Blocks

- Initial smoke result:
  - Teacher-forced BC final row: `env_discovery_rate=0.4545`, `env_lambda2=0.3820`.
  - Held-out stochastic student final row: `env_discovery_rate=0.3571`, `env_lcc_ratio=0.8333`, `env_lambda2=0.0`.
- Core N=10/B=72/80-slot probe, 3 training seeds, 5 evaluation episodes per seed:
  - Flat stochastic actor: discovery rate 0.6322.
  - Full structured residual actor (`rule_residual_scale=1.0`): stochastic discovery 0.5571, deterministic discovery 0.0643, with 14/15 deterministic evaluations nonzero.
  - Structured actor stochastic empty-scan ratio: 0.1112 versus 0.6901 for the flat stochastic actor.
- RL10 fine-tune block:
  - Full structured residual stochastic discovery improved to 0.5865.
  - Deterministic discovery remained low at about 0.0658.
- Clean no-ISAC neural baseline:
  - Environment protocol and expert protocol both set to `improved_rl_no_isac`.
  - Deterministic discovery rate: 0.
  - Stochastic discovery rate: 0.0044.
  - Interpretation: the nonzero neural behavior depends on the ISAC-enabled local observation stream rather than only on the shared actor implementation.
- Rule-residual scale sweep:
  - Best balanced tested setting: `rule_residual_scale=0.25`.
  - Stochastic discovery rate: 0.5978.
  - Deterministic discovery rate: 0.0837, with 15/15 deterministic evaluations nonzero.

## Current Interpretation

The structured MARL path now provides useful method evidence but should not be promoted as the main contribution yet.
The positive evidence is that candidate masking, candidate scores, topology-deficit context, and rule-residual logits reduce empty scanning and prevent the worst deterministic zero-discovery collapse.
The limiting evidence is that the best stochastic structured actor still remains below the flat stochastic actor in discovery rate, and deterministic discovery is nonzero but weak.
This supports a careful paper framing:

- Main contribution: ISAC-assisted candidate-space reduction and distributed protocol design.
- Learning contribution: candidate-constrained, rule-residual shared actor interface and probe evidence.
- Limitation: neural MARL is not yet a standalone replacement for the rule-driven ISAC protocol.

## Remaining Gate

Before MARL can be promoted beyond a method probe:

- train with at least 5 independent seeds at N=10/B=72,
- evaluate zero-shot transfer to N=30/50/100 and B=10/B=15 equivalent codebooks,
- report collision-penalized discovery and topology metrics beside discovery rate,
- improve stochastic-vs-deterministic consistency, likely through explicit TX/RX coordination or entropy/anti-collapse regularization,
- preserve the execution information boundary: no undiscovered-neighbor pose, identity, or centralized graph truth.
