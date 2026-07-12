# N=2, B=8 Local Measurement-Auxiliary Pilot: 10k-Step Validation

## Material Passport

- Artifact type: experiment validation and formal-gate decision
- Verification status: ANALYZED
- Raw run: `05_simulation/results_raw/n2_b8_isac_measurement_aux_pilot_3seed`
- Design: four methods, three independent training seeds, 10,000 environment steps per run
- Evaluation: 100 matched held-out stochastic-policy episodes per trained seed
- Inference unit: independently trained policy seed
- Analysis script: `06_analysis/scripts/analyze_n2_b8_measurement_aux_gate.py`
- Tables: `06_analysis/paper_tables/n2_b8_isac_measurement_aux_pilot_10k_20260712`
- Figures: `06_analysis/paper_figures/n2_b8_isac_measurement_aux_pilot_10k_20260712`

## Contract Audit

The machine audit passed all 290 checks. Every arm uses the same `N=2`, `B=8`, 16-slot
environment, discovery-first reward, recurrent actor, antisymmetric beam-conditioned role
head, PPO settings, 10% uniform beam exploration, and held-out scenario-seed sequence.
Candidate masks, candidate scores, score priors, expert imitation, rule residuals,
rendezvous observations, action-level ISAC credit, and global actor information are disabled.
The maximum rollout replay log-probability error is zero in every run.

The intended treatment differences are:

1. The no-ISAC control does not receive sensing measurements.
2. The direct-ISAC arm receives local anonymous count, variance, and confidence features.
3. The direct auxiliary arm adds a local measurement-prediction loss with coefficient 0.1.
4. The residual auxiliary arm additionally exposes local interaction and residual-count fields.

The direct-ISAC and direct-auxiliary arms therefore differ only by the auxiliary objective and
its 65-parameter prediction head. In this `N=2` gate, post-discovery table exchange cannot
explain performance because there is no third node and the discovery event terminates the
useful rendezvous interval.

## Held-Out Evaluation

| Method | Seed discovery rates | Mean discovery | 95% t interval | Mean `B/A` | Mean TX ratio |
|---|---|---:|---:|---:|---:|
| MARL, no ISAC | 15%, 17%, 19% | 17.00% | [12.03%, 21.97%] | 1.50% | 45.41% |
| MARL + direct ISAC | 5%, 16%, 74% | 31.67% | not informative | 3.16% | 49.18% |
| Direct ISAC + auxiliary | 88%, 83%, 88% | 86.33% | [79.16%, 93.50%] | 10.79% | 60.36% |
| Residual ISAC + auxiliary | 83%, 76%, 80% | 79.67% | [70.94%, 88.39%] | 9.40% | 57.05% |

The direct auxiliary improves discovery over the paired no-ISAC policy by 73, 66, and 69
percentage points. It also improves `B/A` in all three training seeds. The raw direct-ISAC
arm remains optimization-unstable: one seed learns the evidence mapping and two do not.
`S/O` is 100% whenever defined because this gate deliberately uses an ideal communication
PHY; it is not evidence of robustness to fading, interference, or SINR outage.

The 95% intervals use the three independent training seeds as the inference units. The 100
held-out episodes per seed reduce evaluation noise but are not treated as 300 independent
training replicates. With only three training seeds, the intervals remain preliminary.

## Counterfactual Evidence Response

The probe changes one actor-visible beam measurement from confidently empty to occupied,
holding all other local observation fields fixed. It averages over 32 reset scenarios, both
agents, and all eight target beams, for 512 samples per condition and checkpoint.

| Direct auxiliary seed | Occupied action probability | Empty action probability | Action contrast | Prediction-head contrast |
|---|---:|---:|---:|---:|
| 39260711 | 41.05% | 4.10% | +36.95 pp | +72.87 pp |
| 39261720 | 64.28% | 2.73% | +61.55 pp | +81.46 pp |
| 39262729 | 65.49% | 3.13% | +62.36 pp | +96.20 pp |

The no-ISAC controls have exactly zero counterfactual contrast, as expected because those
measurement fields are absent from their actor input. The direct auxiliary therefore passes
the evidence-to-action test in all seeds without an action mask, expert label, or candidate
score prior.

## Training Stability

The final 125-episode direct-auxiliary discovery rates are 79.2%, 84.0%, and 86.4%.
All are at least as high as the preceding block within the predeclared five-percentage-point
tolerance. Their final TX ratios remain between 60.1% and 64.0%, and the auxiliary objective
receives roughly 19-20 local measurement samples per episode.

The residual arm is not equally clean. Seed 39261720 declines from 74.4% to 69.6% and then
61.6% over its last three blocks. Its held-out result is still 76%, but it fails the strict
late-training stability gate. This is consistent with the simpler direct features being the
preferred representation rather than evidence that engineered residual inputs are necessary.

## Gate Decision

- Primary three-arm formal gate: **PASS**.
- Full four-arm formal gate including residual features: **FAIL**.

The formal run should include only:

1. `learned_beam_no_isac`
2. `learned_beam_direct_isac`
3. `learned_beam_direct_isac_measurement_aux`

Each formal run uses 100,000 environment steps and 200 held-out evaluation episodes. The
residual arm remains a documented pilot ablation and must not be described as a stable formal
method unless a later experiment addresses its late-training regression.

## Scope Boundary

This result establishes a narrow mechanism claim: a local anonymous ISAC
measurement-prediction objective can make beam-evidence learning reproducible in the
minimal two-node, static, ideal-PHY gate. It does not yet establish an `N=10` neighbor
discovery advantage, table-exchange benefit, mobility robustness, realistic-PHY performance,
or TWC-level end-to-end superiority. Those claims require later matched-environment stages.
