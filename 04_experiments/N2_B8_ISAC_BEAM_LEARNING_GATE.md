# N=2, B=8 ISAC Beam-Learning Gate

## Purpose

This matrix restores learned beam control after the antisymmetric TX/RX role gate. It isolates the value of raw local ISAC measurements and local post-action credit without adding candidate masks, candidate-score priors, expert actions, rendezvous hints, or global actor information.

## Arms

| Arm | Beam policy | ISAC actor input | Local ISAC beam credit | Role policy |
|---|---|---|---|---|
| `random_beam_antisymmetric_role` | exactly uniform | no | no | learned antisymmetric |
| `learned_beam_no_isac` | recurrent learned, 10% uniform support | no | no | learned antisymmetric |
| `learned_beam_raw_isac` | recurrent learned, 10% uniform support | raw anonymous measurements | no | learned antisymmetric |
| `learned_beam_raw_isac_local_credit` | recurrent learned, 10% uniform support | raw anonymous measurements | yes | learned antisymmetric |

All actors use decentralized local observations. The centralized critic is training-only. ISAC measurements arise only after the node transmits on the selected beam; standalone sensing is disabled.

## Commands

Syntax and contract check without running training:

```powershell
python 05_simulation/scripts/run_n2_b8_isac_beam_learning_matrix.py --profile smoke --dry-run
```

Short functional run:

```powershell
python 05_simulation/scripts/run_n2_b8_isac_beam_learning_matrix.py --profile smoke --max-parallel 2
```

Formal three-seed run, 100,000 environment steps per arm and seed:

```powershell
python 05_simulation/scripts/run_n2_b8_isac_beam_learning_matrix.py --profile formal --run-root 05_simulation/results_raw/n2_b8_isac_beam_learning_formal_100k_20260712 --max-parallel 2
```

Resume monitoring without overwriting completed runs:

```powershell
python 05_simulation/scripts/run_n2_b8_isac_beam_learning_matrix.py --profile formal --run-root 05_simulation/results_raw/n2_b8_isac_beam_learning_formal_100k_20260712 --max-parallel 2 --skip-completed
```

Read-only progress and resource summary:

```powershell
python 05_simulation/scripts/monitor_n2_b8_isac_beam_learning_matrix.py --profile formal --run-root 05_simulation/results_raw/n2_b8_isac_beam_learning_formal_100k_20260712
```

## Completion Contract

Each of the 12 run directories must contain `final_model.pt`, `manifest.json`, `episode_metrics.csv`, and `eval_episode_metrics.csv`. The trainer records resource usage and enforces the configured RSS and system-memory limits.

Primary evaluation uses stochastic held-out scenarios. Report discovery rate together with the execution-time funnel `B/A`, `O/B`, and `S/O`. The raw-ISAC claim passes only if `B/A` improves over the no-ISAC learned-beam arm while the antisymmetric role head retains its `O/B` gain.
