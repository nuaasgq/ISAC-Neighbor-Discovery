# Wang2025-Style Real MARL Comparison First Pass (2026-07-09)

## Why This Run Was Needed

The previous Wang-style matrix compared Wang baselines with rule/heuristic
protocols such as `improved_rl_isac` and `budgeted_collision_aware_isac`. That
was not a strict comparison between trained MARL and Wang-style methods.

This first-pass corrective campaign adds real trained MARL checkpoints to the
same Wang-style single-RF matrix.

## Experiment Matrix

- Configuration: `05_simulation/configs/wang2025_reproduction_smoke.yaml`.
- Beam setting: 15 azimuth cells x 7 elevation cells, about 25-degree beams.
- RF chains: 1.
- Training: `N=10`, 300 slots per episode, 20 training episodes.
- Evaluation: `N=10/20/30/40/50`, 200 slots, 3 episodes per point.
- MARL action space: `TX/RX/IDLE` only.
- Standalone `SENSE` is disabled for MARL with `--forbid-sense`.
- ISAC feedback comes from TX-coupled piggyback sensing.

## Compared Methods

Baselines:

- `uniform_random`
- `wang2025_isac_no_collab`
- `wang2025_comm_tables`
- `wang2025_isac_tables`
- `budgeted_collision_aware_isac`

Real MARL methods:

- `marl_no_isac_txrxidle`
- `marl_tx_isac_txrxidle`
- `marl_tx_isac_gate_bc_txrxidle`

## Output Directories

- Deterministic MARL evaluation tables: `06_analysis/paper_tables/wang2025_marl_vs_wang_firstpass_20260709/`.
- Deterministic MARL figures: `06_analysis/paper_figures/wang2025_marl_vs_wang_firstpass_20260709/`.
- Stochastic MARL evaluation tables: `06_analysis/paper_tables/wang2025_marl_vs_wang_firstpass_stochastic_20260709/`.
- Stochastic MARL figures: `06_analysis/paper_figures/wang2025_marl_vs_wang_firstpass_stochastic_20260709/`.
- Raw MARL training/evaluation root: `05_simulation/results_raw/marl_campaign/wang2025_marl_vs_wang_firstpass_20260709/`.

## N=50 Results

### Deterministic MARL Deployment

| Method | Discovery | CPD | Collisions | Lambda2 | Standalone Sense | Piggyback ISAC |
|---|---:|---:|---:|---:|---:|---:|
| Uniform random | 0.0090 | 0.0090 | 0.0 | 0.0000 | 0.0 | 0.0 |
| Wang ISAC, no table | 0.5170 | 0.1177 | 4201.7 | 15.5967 | 0.0 | 4993.3 |
| Wang communication table | 0.5401 | 0.0827 | 6796.3 | 16.0006 | 0.0 | 5016.7 |
| Wang sensing table | 0.5361 | 0.0819 | 6815.0 | 17.2360 | 0.0 | 4955.7 |
| Budgeted ISAC rule | 0.5888 | 0.1914 | 2560.7 | 18.7574 | 0.0 | 9666.0 |
| MARL, no ISAC | 0.0000 | 0.0000 | 0.0 | 0.0000 | 0.0 | 0.0 |
| MARL + TX-coupled ISAC | 0.0528 | 0.0463 | 174.3 | 0.0000 | 0.0 | 7514.0 |
| MARL + TX-coupled ISAC + gate BC | 0.0003 | 0.0003 | 0.0 | 0.0000 | 0.0 | 10000.0 |

Deterministic argmax deployment is not usable yet. The trained policies often
collapse into overly passive RX-heavy behavior, causing very low discovery.

### Stochastic MARL Deployment

| Method | Discovery | CPD | Collisions | Lambda2 | Standalone Sense | Piggyback ISAC |
|---|---:|---:|---:|---:|---:|---:|
| Uniform random | 0.0090 | 0.0090 | 0.0 | 0.0000 | 0.0 | 0.0 |
| Wang ISAC, no table | 0.5170 | 0.1177 | 4201.7 | 15.5967 | 0.0 | 4993.3 |
| Wang communication table | 0.5401 | 0.0827 | 6796.3 | 16.0006 | 0.0 | 5016.7 |
| Wang sensing table | 0.5361 | 0.0819 | 6815.0 | 17.2360 | 0.0 | 4955.7 |
| Budgeted ISAC rule | 0.5888 | 0.1914 | 2560.7 | 18.7574 | 0.0 | 9666.0 |
| MARL, no ISAC | 0.0024 | 0.0024 | 0.0 | 0.0000 | 0.0 | 0.0 |
| MARL + TX-coupled ISAC | 0.2833 | 0.1152 | 1790.3 | 5.0014 | 0.0 | 5119.0 |
| MARL + TX-coupled ISAC + gate BC | 0.3244 | 0.0576 | 5671.3 | 8.0745 | 0.0 | 8862.0 |

Stochastic deployment is much better than deterministic deployment, but it still
does not beat the strongest Wang-style baseline on raw discovery, and it does
not beat the Budgeted ISAC rule on CPD.

## Main Conclusions

1. The strict MARL-vs-Wang comparison has now been run in the same Wang-style
   single-RF environment. The earlier `improved_rl_isac` row should not be
   described as trained MARL.
2. The corrected MARL action model is physically cleaner: standalone `SENSE`
   is disabled and all ISAC observations are TX-coupled piggyback observations.
   This is verified by `sense_actions_mean = 0` for all MARL rows.
3. ISAC is still valuable for MARL. Stochastic `MARL + TX-coupled ISAC` at
   `N=50` improves discovery from 0.0024 without ISAC to 0.2833 with ISAC.
4. The current MARL policy is not paper-strong as the main method. It does not
   dominate Wang ISAC or the Budgeted ISAC rule.
5. Gate BC currently increases raw discovery but causes excessive collisions,
   reducing CPD. This suggests that the learned gate imitates access activation
   but not the collision-budget discipline strongly enough.
6. Deterministic deployment collapse is a serious training/calibration problem.
   The next MARL work should target mode-balance regularization, collision-budget
   constraints, or distillation from Budgeted ISAC with explicit CPD-oriented
   loss terms.

## Paper Implication

The defensible current paper claim remains:

> ISAC-assisted candidate-beam feedback plus link-layer budgeted access control
> improves narrow-beam FANET neighbor discovery under the Wang-style matrix.

The MARL claim should be limited to:

> A real TX/RX/IDLE MARL implementation confirms that ISAC observations are
> useful for learned policies, but current learned policies require stronger
> collision-budget learning before they can replace the Budgeted ISAC rule.
