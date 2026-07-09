# MARL Stochastic Temperature Sweep Results (2026-07-09)

## Purpose

This run tests whether the weak MARL-vs-Wang result is partly caused by
stochastic deployment calibration rather than only by training failure. It does
not change training and does not introduce expert distillation.

## Setup

- Checkpoints: first-pass Wang-style MARL checkpoints trained at `N=10`.
- Evaluation node count: `N=50`.
- Slots: 200.
- Episodes per point: 3.
- Action space: `TX/RX/IDLE`; standalone `SENSE` remains disabled.
- ISAC feedback: TX-coupled piggyback sensing.
- Temperatures: `0.7, 1.0, 1.3, 1.6, 2.0`.
- Temperature is applied to mode, beam, and gate logits during stochastic
  deployment only.

## Output

- Tables: `06_analysis/paper_tables/marl_temperature_sweep_20260709/`.
- Figures: `06_analysis/paper_figures/marl_temperature_sweep_20260709/`.
- Raw evaluations: `05_simulation/results_raw/marl_campaign/marl_temperature_sweep_20260709/`.

## Key Results at N=50

### MARL Without ISAC

| Temperature | Discovery | CPD | Collisions | Lambda2 | TX ratio | Idle ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 0.7 | 0.0008 | 0.0008 | 0.0 | 0.0000 | 0.654 | 0.616 |
| 1.0 | 0.0024 | 0.0024 | 0.0 | 0.0000 | 0.602 | 0.529 |
| 1.3 | 0.0014 | 0.0014 | 0.0 | 0.0000 | 0.573 | 0.487 |
| 1.6 | 0.0011 | 0.0011 | 0.0 | 0.0000 | 0.561 | 0.457 |
| 2.0 | 0.0035 | 0.0035 | 0.0 | 0.0000 | 0.553 | 0.435 |

The no-ISAC trained policy remains essentially unusable under all temperatures.

### MARL + TX-Coupled ISAC

| Temperature | Discovery | CPD | Collisions | Lambda2 | TX ratio | Idle ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 0.7 | 0.2762 | 0.1717 | 745.7 | 5.4821 | 0.167 | 0.536 |
| 1.0 | 0.3039 | 0.1310 | 1643.7 | 6.6097 | 0.233 | 0.487 |
| 1.3 | 0.3118 | 0.0881 | 3120.7 | 7.6261 | 0.281 | 0.464 |
| 1.6 | 0.2980 | 0.0822 | 3216.7 | 6.8616 | 0.318 | 0.434 |
| 2.0 | 0.3026 | 0.0614 | 4812.3 | 6.9394 | 0.354 | 0.421 |

Lower temperature improves CPD by suppressing excessive access. At `T=0.7`,
the policy reaches CPD `0.1717`, which is higher than the previous Wang no-table
baseline CPD `0.1177`, but its discovery rate `0.2762` is still far below Wang
no-table discovery `0.5170`.

### MARL + TX-Coupled ISAC + Gate BC

| Temperature | Discovery | CPD | Collisions | Lambda2 | TX ratio | Idle ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 0.7 | 0.3135 | 0.0748 | 3969.0 | 6.4521 | 0.132 | 0.052 |
| 1.0 | 0.3439 | 0.0521 | 6881.3 | 7.5062 | 0.198 | 0.122 |
| 1.3 | 0.3293 | 0.0499 | 6940.3 | 8.0905 | 0.240 | 0.158 |
| 1.6 | 0.3222 | 0.0455 | 7452.0 | 6.6151 | 0.268 | 0.188 |
| 2.0 | 0.3189 | 0.0449 | 7543.0 | 7.0040 | 0.295 | 0.210 |

The gate-BC checkpoint improves raw discovery but still creates too many
collisions. Temperature reduction helps, but not enough to make this variant
CPD-competitive.

## Conclusions

1. Stochastic temperature matters. The default `T=1.0` was not necessarily the
   best deployment mode.
2. For the current `MARL + TX-coupled ISAC` checkpoint, `T=0.7` is the best CPD
   point among the tested temperatures.
3. Temperature calibration alone can make MARL competitive with Wang no-table on
   CPD, but not on raw discovery or topology quality.
4. Budgeted ISAC rule remains the strongest current method overall: in the
   first-pass Wang matrix at `N=50`, it reached discovery `0.5888`, CPD `0.1914`,
   collisions `2560.7`, and lambda2 `18.7574`.
5. The next MARL improvement should not start with distillation yet. A more
   focused next step is to tune deployment temperature over the whole
   `N=10/20/30/40/50` Wang matrix for the non-gated `MARL + TX-coupled ISAC`
   checkpoint, likely using `T=0.7` as the first candidate.

## Claim Boundary

This result supports a narrow claim:

> MARL should be evaluated as a randomized decentralized protocol, and sampling
> temperature is a material deployment parameter for ISAC-assisted neighbor
> discovery.

It does not yet support:

> The current MARL policy outperforms Budgeted ISAC or fully solves the
> MARL-vs-Wang comparison.
