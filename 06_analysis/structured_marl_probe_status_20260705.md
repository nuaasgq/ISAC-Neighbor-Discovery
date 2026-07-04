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
- Structured smoke command completed:

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

## Smoke Result

- Teacher-forced BC final row: `env_discovery_rate=0.4545`, `env_lambda2=0.3820`.
- Held-out stochastic student final row: `env_discovery_rate=0.3571`, `env_lcc_ratio=0.8333`, `env_lambda2=0.0`.
- Interpretation: the structured neural path is no longer a zero-signal flat-action probe, but this is still not a paper-quality MARL result.

## Next Gate

Before MARL can be promoted beyond a method probe:

- run at least 3 seeds at N=10/B=72 with 50-100 BC episodes and 25-50 RL fine-tune episodes,
- compare flat BC, candidate-mask only, full structured residual, enhanced no-ISAC, and rule expert,
- report deterministic and stochastic student evaluation separately,
- transfer the same N=10-trained student to N=30/50/100 and at least B=10/B=15 equivalent codebooks.
