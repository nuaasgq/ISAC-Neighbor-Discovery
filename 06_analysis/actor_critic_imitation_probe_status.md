# Actor-Critic Imitation Probe Status

Created on 2026-07-05 (Asia/Shanghai).

## Purpose

This file tracks a lightweight MARL method probe that adds rule-expert-assisted behavior cloning before optional actor-critic fine-tuning. It is intended to address the earlier shared actor-critic smoke result where the executable path worked but produced zero discovery under a very short run.

## Added Entry Point

- `05_simulation/run_actor_critic_imitation_probe.py`
  - Reuses `MarlNeighborDiscoveryEnv` for the MARL rollout interface.
  - Reuses `SharedBeamActorCritic` for the shared decentralized actor and value head.
  - Uses `NeighborDiscoverySimulator(..., protocol="improved_rl_isac")` as the rule expert action source.
  - Trains behavior-cloning losses for expert mode and active-beam targets.
  - Optionally runs short on-policy actor-critic fine-tuning after BC.
  - Writes `training_history.csv` and `manifest.json`.

## Output Contract

The output directory contains:

- `training_history.csv`: one row per BC/RL/evaluation episode, including teacher-forced MARL-env metrics, expert metrics, BC losses, accuracy proxies, and deterministic student-eval metrics.
- `manifest.json`: run metadata, selected config overrides, final teacher-forced row, final deterministic student-eval row, and an explicit method-probe warning.

## Important Limitation

This is only a method probe and software-integration draft. It must not be used as a paper main result. A paper-ready result still requires longer training, multi-seed repeats, unchanged comparison baselines, held-out mobility/scenario evaluation, and a clear separation between teacher-forced discovery and autonomous student-policy discovery.

## Suggested Smoke Command

```powershell
python 05_simulation/run_actor_critic_imitation_probe.py --config 05_simulation/configs/mvp.yaml --output 05_simulation/results_raw/actor_critic_imitation_probe_smoke --bc-episodes 2 --rl-episodes 1 --eval-episodes 1 --slots 20 --node-count 6 --azimuth-cells 4 --elevation-cells 2 --communication-range 800 --sensing-range 900 --false-alarm-rate 0 --miss-detection-rate 0 --angular-cell-offset-std 0 --sensing-period-slots 1 --hidden-dim 32 --learning-rate 0.001 --seed 20260705
```

Expected interpretation: the BC teacher-forced rows should show nonzero discovery in the MARL env under the small smoke setup. The deterministic student-eval row may still be weak after only a few episodes and should be treated as a wiring sanity check, not a learned-performance claim.

## Smoke Result

The command above was run successfully.

| Row | Discovery rate | Discovered edges | Lambda2 | Empty-scan ratio | Note |
|---|---:|---:|---:|---:|---|
| Final BC teacher-forced MARL env | 0.4000 | 6 | 0.6571 | 0.1933 | Expert actions executed through `MarlNeighborDiscoveryEnv`. |
| Final BC expert simulator | 0.3333 | 5 | 1.0000 | 0.2101 | Same `improved_rl_isac` action source with expert-side ISAC bookkeeping. |
| Deterministic student eval | 0.0000 | 0 | 0.0000 | 0.6296 | Expected to remain weak after this tiny smoke run. |

Additional verification:

```powershell
python -m py_compile 05_simulation/run_actor_critic_imitation_probe.py
python -m pytest 05_simulation/tests/test_actor_critic_probe.py 05_simulation/tests/test_marl_env_contract.py
```

Result: `6 passed`.

## Round7 Medium Probe

A larger local probe was also run to test whether the autonomous student policy has a usable signal beyond the tiny smoke test.

Important correction: the first medium probes used `MarlNeighborDiscoveryEnv` before the environment default was changed to the ISAC-capable `isac_structured_marl` simulator protocol. Those early probes were useful for detecting deterministic mode collapse, but they understated the neural policy path because TX/RX piggyback ISAC did not update the MARL environment belief.

Deterministic eval after BC+short actor-critic fine-tuning:

```powershell
python 05_simulation/run_actor_critic_imitation_probe.py --config 05_simulation/configs/mvp.yaml --output 05_simulation/results_raw/actor_critic_imitation_probe_round7 --bc-episodes 20 --rl-episodes 20 --eval-episodes 3 --slots 80 --node-count 10 --azimuth-cells 12 --elevation-cells 6 --communication-range 1200 --sensing-range 1200 --false-alarm-rate 0 --miss-detection-rate 0 --angular-cell-offset-std 0 --sensing-period-slots 1 --hidden-dim 64 --learning-rate 0.001 --seed 20260705
```

Result: teacher-forced expert actions still produced discovery, but deterministic student evaluation collapsed to one dominant mode and achieved zero discovery.

BC-only deterministic eval:

```powershell
python 05_simulation/run_actor_critic_imitation_probe.py --config 05_simulation/configs/mvp.yaml --output 05_simulation/results_raw/actor_critic_imitation_probe_round7_bc_only --bc-episodes 40 --rl-episodes 0 --eval-episodes 3 --slots 80 --node-count 10 --azimuth-cells 12 --elevation-cells 6 --communication-range 1200 --sensing-range 1200 --false-alarm-rate 0 --miss-detection-rate 0 --angular-cell-offset-std 0 --sensing-period-slots 1 --hidden-dim 64 --learning-rate 0.001 --seed 20260705
```

Result: deterministic student evaluation again achieved zero discovery, with all agents selecting the same communication mode.

Stochastic eval after BC-only training:

```powershell
python 05_simulation/run_actor_critic_imitation_probe.py --config 05_simulation/configs/mvp.yaml --output 05_simulation/results_raw/actor_critic_imitation_probe_round7_stochastic --bc-episodes 40 --rl-episodes 0 --eval-episodes 5 --stochastic-eval --slots 80 --node-count 10 --azimuth-cells 12 --elevation-cells 6 --communication-range 1200 --sensing-range 1200 --false-alarm-rate 0 --miss-detection-rate 0 --angular-cell-offset-std 0 --sensing-period-slots 1 --hidden-dim 64 --learning-rate 0.001 --seed 20260705
```

Result summary:

| Eval mode | Episodes | Mean discovery rate | Mean discovered edges | Mean empty-scan ratio | Nonzero episodes |
|---|---:|---:|---:|---:|---:|
| Stochastic student | 5 | 0.0140 | 0.6 | 0.9009 | 2 |

After fixing `MarlNeighborDiscoveryEnv` to use the ISAC-capable simulator protocol by default, the same stochastic BC-only probe was rerun:

```powershell
python 05_simulation/run_actor_critic_imitation_probe.py --config 05_simulation/configs/mvp.yaml --output 05_simulation/results_raw/actor_critic_imitation_probe_round7_stochastic_isacenv --bc-episodes 40 --rl-episodes 0 --eval-episodes 5 --stochastic-eval --slots 80 --node-count 10 --azimuth-cells 12 --elevation-cells 6 --communication-range 1200 --sensing-range 1200 --false-alarm-rate 0 --miss-detection-rate 0 --angular-cell-offset-std 0 --sensing-period-slots 1 --hidden-dim 64 --learning-rate 0.001 --seed 20260705
```

Result summary:

| Eval mode | Episodes | Mean discovery rate | Mean discovered edges | Mean lambda2 | Mean empty-scan ratio | Nonzero episodes |
|---|---:|---:|---:|---:|---:|---:|
| Stochastic student with ISAC env | 5 | 0.6469 | 28.6 | 3.2474 | 0.6883 | 5 |

Interpretation: the neural path is viable only when the MARL environment exposes the same ISAC piggyback belief updates used by the proposed protocol. This remains a method probe because it lacks multi-seed baselines and transfer tests, but it is no longer a zero-signal path. The next neural architecture should not rely on a flat categorical over all 3D beam cells. It should use at least one of:

- ISAC top-k beam masking or candidate-set constrained beam heads.
- Rule-residual logits added to learned logits.
- Separate mode coordination prior to avoid all-TX/all-RX deterministic collapse.
- Graph/local-neighborhood pooling once discovered neighbors exist.
