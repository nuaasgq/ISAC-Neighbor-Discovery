# N=10, B=15-Degree Static Ideal-ISAC MAPPO Matrix

## Purpose

This stage tests whether the local measurement-prediction mechanism remains useful when the
minimal two-node gate is expanded to ten nodes. It deliberately removes mobility and physical
layer uncertainty so that failures can be attributed to multi-agent coordination and beam
learning rather than channel outage or stale sensing.

## Common Environment

- 10 nodes in a 3500 m x 3500 m planar area
- 24 azimuth beams (15 degrees), one elevation cell, one RF chain
- 300 slots per episode and 5 ms per slot
- static nodes with self position and attitude
- communication and sensing ranges larger than the area diagonal
- ideal communication decoding and ideal anonymous count-ISAC
- TX-coupled sensing only; standalone Sense and Idle are disabled
- decentralized actors and centralized training-only critic

## Trained Arms

| Arm | Local ISAC input | Measurement auxiliary | Candidate rule |
|---|---|---:|---:|
| `mappo_no_isac` | none | 0 | none |
| `mappo_direct_isac` | count, variance, confidence | 0 | none |
| `mappo_direct_isac_measurement_aux` | count, variance, confidence | 0.1 | none |

Every arm uses the same recurrent shared actor, beam-conditioned antisymmetric role head,
10% uniform beam exploration, discovery-first reward, and MAPPO optimizer settings. Candidate
masks, candidate scores, expert imitation, rule residuals, rendezvous hints, and global actor
information are disabled.

The formal profile uses 1000 episodes, or 300,000 environment steps, for each of three
independent training seeds. Each final checkpoint is evaluated on 50 held-out episodes.

## Non-Trained Baselines

`uniform_random` and `wang2025_isac_tables` are evaluated directly for 50 episodes. They are
not trained and must use the same configuration and evaluation seed.

## MATD3 Boundary

The adapted reference uses the official `marlbenchmark/off-policy` implementation pinned at
commit `41fd5eb46d12df2847e1c2e29842997ff2c24998`. Its MATD3 policy extends MADDPG with twin
critics and delayed actor updates. The environment adapter uses the reference implementation's
discrete one-hot/Gumbel path with two categorical heads (`TX/RX` and 24 beams). The actor sees
only the clean local direct-ISAC observation, while its centralized training critic receives the
concatenation of those local observations. Candidate masks, candidate scores, expert actions,
the antisymmetric role head, and the measurement-prediction auxiliary are disabled.

This arm is an algorithm reference rather than an architecture-controlled MAPPO ablation. The
adapter and manifest explicitly record one correction to the upstream runner behavior: the
trainer update counter is incremented so the official delayed-actor schedule actually advances.
The formal MATD3 profile also uses 1000 episodes per seed, 300 slots per episode, three seeds,
and 50 held-out evaluation episodes, but runs the seeds sequentially to limit CPU and memory.

## Commands

Smoke:

```powershell
python 05_simulation/scripts/run_n10_b15_static_ideal_mappo_matrix.py `
  --profile smoke --seeds 59260713 `
  --run-root 05_simulation/results_raw/n10_b15_static_ideal_mappo_smoke_1seed `
  --max-parallel 1
```

Formal:

```powershell
python 05_simulation/scripts/run_n10_b15_static_ideal_mappo_matrix.py `
  --profile formal `
  --run-root 05_simulation/results_raw/n10_b15_static_ideal_mappo_formal_3seed `
  --max-parallel 2
```

Monitor:

```powershell
python 05_simulation/scripts/monitor_n10_b15_static_ideal_mappo_matrix.py `
  --profile formal `
  --run-root 05_simulation/results_raw/n10_b15_static_ideal_mappo_formal_3seed `
  --watch --interval-seconds 30
```

MATD3 formal:

```powershell
python 05_simulation/scripts/run_n10_b15_static_ideal_matd3_reference_matrix.py `
  --profile formal `
  --run-root 05_simulation/results_raw/n10_b15_static_ideal_matd3_reference_formal_3seed `
  --torch-threads 1
```

MATD3 progress:

```powershell
python 05_simulation/scripts/monitor_n10_b15_static_ideal_matd3_reference_matrix.py `
  --run-root 05_simulation/results_raw/n10_b15_static_ideal_matd3_reference_formal_3seed
```
