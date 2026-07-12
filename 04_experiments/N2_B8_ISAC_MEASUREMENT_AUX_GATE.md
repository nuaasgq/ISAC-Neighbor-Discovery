# N=2, B=8 Local Measurement-Prediction Gate

## Purpose

This gate tests whether a confidence-weighted local measurement-prediction objective stabilizes ISAC beam learning without supplying action labels or changing the neighbor-discovery reward.

## Matrix

| Arm | Measurement inputs | Prediction auxiliary | Action-level ISAC credit |
|---|---|---:|---:|
| `learned_beam_no_isac` | none | 0 | 0 |
| `learned_beam_direct_isac` | count, variance, confidence | 0 | 0 |
| `learned_beam_direct_isac_measurement_aux` | count, variance, confidence | 0.10 | 0 |
| `learned_beam_residual_isac_measurement_aux` | direct + interaction/residual count | 0.10 | 0 |

Every arm uses the same recurrent beam policy, 10% uniform exploration support, and antisymmetric learned TX/RX role head. Candidate masks, candidate scores, score priors, expert imitation, rendezvous hints, rule logits, and global actor information are disabled.

The auxiliary target is generated only when the node actually transmits and obtains a fresh anonymous piggyback measurement. Positive and negative samples are balanced within the episode and weighted by local measurement confidence. Auxiliary gradients reach the beam encoder and prediction head but not the role head.

## Commands

Contract-only dry run:

```powershell
python 05_simulation/scripts/run_n2_b8_isac_measurement_aux_matrix.py --profile smoke --dry-run
```

Short smoke gate:

```powershell
python 05_simulation/scripts/run_n2_b8_isac_measurement_aux_matrix.py --profile smoke --max-parallel 2
```

Three-seed 10k-step pilot:

```powershell
python 05_simulation/scripts/run_n2_b8_isac_measurement_aux_matrix.py --profile pilot --max-parallel 2
```

Read-only monitoring:

```powershell
python 05_simulation/scripts/monitor_n2_b8_isac_measurement_aux_matrix.py --profile pilot --run-root 05_simulation/results_raw/n2_b8_isac_measurement_aux_pilot_3seed
```

## Gate Decision

Do not launch the formal profile unless both auxiliary arms satisfy all of the following across every training seed:

1. Positive occupied-minus-empty evidence-response contrast.
2. Held-out `B/A` above the paired no-ISAC policy.
3. No TX/RX collapse.
4. Stable or improving final training blocks.

The direct-versus-residual contrast determines whether engineered residual features are necessary. A direct-measurement success is preferred because it supports the cleaner data-driven claim.
