# N=2, B=8 ISAC Beam-Learning Pilot: 10k-Step Diagnosis

## Material Passport

- Artifact type: experiment validation and failure diagnosis
- Verification status: ANALYZED
- Raw run: `05_simulation/results_raw/n2_b8_isac_beam_learning_pilot_3seed`
- Design: four methods, three paired training seeds, 10,000 environment steps per run
- Evaluation: 100 matched held-out stochastic-policy episodes per trained seed
- Inference unit: independently trained policy seed

## Controlled Result

| Method | Seed discovery rates | Mean discovery | Mean `B/A` | Mean `O/B` | Mean TX ratio |
|---|---|---:|---:|---:|---:|
| Random beam + antisymmetric role | 17%, 16%, 16% | 16.33% | 1.48% | 74.28% | 49.64% |
| Learned beam, no ISAC | 15%, 17%, 19% | 17.00% | 1.50% | 75.98% | 45.41% |
| Learned beam, raw local ISAC | 79%, 17%, 20% | 38.67% | 3.73% | 84.73% | 47.31% |
| Raw local ISAC + action credit | 20%, 7%, 8% | 11.67% | 0.91% | 85.65% | 14.30% |

`S/O` is 100% for every defined seed because this gate uses an ideal communication PHY. All deterministic evaluations are 0% because a fixed deterministic beam cannot implement the stochastic exploration contract.

## Conclusions

1. Learned beams without identifying information do not beat uniform beams. Their discovery rates and `B/A` values are effectively the same. This is the expected identifiability control.
2. Raw local ISAC measurements are sufficient for a high-performing policy. One seed reaches 79% discovery and `B/A = 8.18%`, compared with a roughly 16%-17% random/no-ISAC level.
3. The raw-ISAC result is not robust. The other two seeds remain at 17% and 20%, so the 38.67% mean is dominated by one successful training run and is not evidence of stable superiority.
4. The current local action-credit term is harmful. All three runs trend toward RX-dominant behavior; held-out TX ratios are 19.75%, 11.25%, and 11.91%. Its final 125-episode training discovery rates fall to 6.4%, 5.6%, and 8.0%.
5. The antisymmetric role mechanism remains functional. `O/B` stays high whenever bilateral alignment occurs. The unresolved problem is reliable beam-evidence learning, not aligned-role complementarity.

## Evidence-Response Probe

A synthetic local-observation probe changes one beam from unknown to either occupied or confidently empty, then measures the checkpoint's probability of selecting that beam. It uses actor-visible fields only.

| Raw-ISAC seed | Occupied-beam probability | Empty-beam probability | Contrast |
|---|---:|---:|---:|
| 39260711 | 68.72% | 2.24% | +66.48 pp |
| 39261720 | 14.07% | 12.86% | +1.21 pp |
| 39262729 | 13.79% | 12.36% | +1.43 pp |

The successful seed learned a strong local evidence-to-beam mapping. The other seeds did not. Therefore, the main failure is sparse-reward optimization instability, not missing actor information or an insufficient 16-slot horizon.

## Why Local Action Credit Fails

The feedback is positive for an occupied TX beam, negative for an empty TX beam, and zero for RX. Since most exploratory TX beams are empty, its empirical mean is negative. Beam choice also conditions the antisymmetric role distribution. The policy can therefore reduce exposure to negative beam credit by selecting directions associated with RX, producing a TX-avoidance loophole even though the role and beam parameter towers are separated.

This term must not be tuned or promoted to formal experiments in its current form.

## Next Task

Replace action-level ISAC reward credit with a local measurement-prediction auxiliary objective:

1. Add a per-beam evidence head sharing the raw local ISAC beam encoder.
2. After a piggyback TX measurement, predict anonymous occupancy/count confidence for the measured beam at the next observation.
3. Weight the auxiliary loss by measurement confidence and stop its gradient before the role head.
4. Do not alter environment rewards, TX/RX advantages, executed actions, candidate masks, or candidate-score priors.
5. Verify with unit tests that occupied evidence raises and empty evidence lowers the learned beam score, while role probabilities are unchanged by the auxiliary target itself.

The next experiment should compare no-ISAC learned beam, raw-ISAC PPO, and raw-ISAC PPO plus the measurement-prediction auxiliary at 10,000 steps. It should proceed to 100,000 steps only if the ISAC evidence-response contrast and `B/A` gain are positive in all training seeds.
