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

The reference implementation is `marlbenchmark/off-policy`. Its MATD3 implementation extends
MADDPG with twin critics and delayed actor updates. Integration is a separate branch because
the current joint action is categorical (`TX/RX x beam`) and requires the reference code's
discrete one-hot/Gumbel action path rather than a continuous-action TD3 interface. MATD3 does
not block the three-arm MAPPO matrix and must be labeled as an adapted reference implementation
if any environment or action adapter is added.

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
