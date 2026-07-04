# Shared Actor-Critic Probe Status

Generated during the long-run goal window ending at 2026-07-05 11:00 (Asia/Shanghai).

## What Was Added

- `05_simulation/src/isac_nd_sim/neural_shared_actor_critic.py`
  - Shared actor with local per-agent observations only.
  - Per-beam token encoder over belief, age, success, and failure features.
  - Mode head, beam-selection head, and value head.
- `05_simulation/run_actor_critic_probe.py`
  - Lightweight on-policy actor-critic training probe over `MarlNeighborDiscoveryEnv`.
  - Writes `training_history.csv` and `manifest.json`.
- `05_simulation/tests/test_actor_critic_probe.py`
  - Verifies valid action sampling and a one-episode training smoke run.

## Smoke Result

Command:

```powershell
python 05_simulation/run_actor_critic_probe.py --config 05_simulation/configs/mvp.yaml --output 05_simulation/results_raw/actor_critic_probe_smoke --episodes 12 --slots 30 --node-count 6 --azimuth-cells 6 --elevation-cells 3 --hidden-dim 32 --learning-rate 0.001 --seed 20260705
```

Final smoke metrics:

| Metric | Value |
|---|---:|
| Discovery rate | 0.0000 |
| Lambda2 | 0.0000 |
| Empty-scan ratio | 0.9010 |
| Mean reward | -0.0132 |
| Beam count | 18 |
| Episodes | 12 |
| Slots per episode | 30 |

## Interpretation

This is a software-infrastructure smoke test, not a performance result. The actor-critic path executes and writes reproducible logs, but the short run does not learn discovery behavior. It should not be used as evidence that neural MARL improves the protocol.

## Next Required MARL Work

- Add imitation pretraining or rule-residual logits from `improved_rl_isac` / `I-TAP-ND`.
- Train on N=6-10 with longer horizons and compare against the current CEM shared-policy result.
- Add policy-gradient and value-based variants only after the actor-critic path produces nonzero discovery under a small codebook.
- Keep the paper claim conservative until a neural MARL variant beats or matches the rule-driven shared-policy baseline.
