# Beam-only RL algorithm comparison gate

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-07-11
- Verification Status: ANALYZED
- Scope: one training seed per algorithm, 30 episodes x 300 slots, 10 paired held-out scenarios

## Common contract

Shared-IDQN, VDN, QMIX, and beam-only MAPPO use the same `N=10`, 15-degree planar codebook, 300-slot episode, mobility, PHY, ISAC measurement, table exchange, reward, scenario sequence, and local `residual_table` observation contract. Every UAV independently samples TX/RX with probability `0.5`; role is absent from every learned action and loss.

All checkpoints are evaluated by the same external seven-policy evaluator. The evaluator uses identical scenario, role, mixture-gate, and candidate-choice random streams and records role, beam, and candidate-mask trajectory hashes. Value methods use epsilon-greedy training (`1.0` to `0.1`, no persistent floor); MAPPO uses categorical policy sampling and entropy coefficient `0.01`, also without a persistent random floor. These are algorithm-specific training exploration mechanisms and are not described as equivalent.

## Evaluation results

| Policy | Shared-IDQN | VDN | QMIX | Beam-only MAPPO |
|---|---:|---:|---:|---:|
| pure learned beam | 51.33% | **56.22%** | 55.33% | 48.00% |
| learned + random, mix 0.2 | 51.33% | 54.44% | 53.11% | 50.44% |
| learned + random, mix 0.5 | 55.56% | 50.00% | 52.00% | 55.33% |
| learned + random, mix 0.8 | 57.33% | **58.22%** | 55.56% | 55.11% |
| candidate-uniform random | 51.56% | 51.56% | 51.56% | 51.56% |
| candidate-score argmax | 47.33% | 47.33% | 47.33% | 47.33% |
| candidate-score proportional | 52.89% | 52.89% | 52.89% | 52.89% |

The three non-learning rows are exactly identical across all four checkpoint types on every scenario, including discovery, delay, role, beam, and candidate-mask traces.

## Pure-policy attribution

| Pure policy comparison | Difference | 95% CI | Exact p | Holm p |
|---|---:|---:|---:|---:|
| Shared-IDQN - candidate random | -0.22 pp | [-12.20, 11.75] | 1.000 | 1.000 |
| Shared-IDQN - score proportional | -1.56 pp | [-9.49, 6.38] | 0.711 | 1.000 |
| VDN - candidate random | +4.67 pp | [-3.66, 12.99] | 0.256 | 1.000 |
| VDN - score proportional | +3.33 pp | [-3.38, 10.05] | 0.316 | 1.000 |
| QMIX - candidate random | +3.78 pp | [-5.43, 12.99] | 0.395 | 1.000 |
| QMIX - score proportional | +2.44 pp | [-2.32, 7.21] | 0.336 | 1.000 |
| MAPPO - candidate random | -3.56 pp | [-13.19, 6.08] | 0.479 | 1.000 |
| MAPPO - score proportional | -4.89 pp | [-15.24, 5.46] | 0.344 | 1.000 |

No pure learned policy passes the claim gate. VDN and QMIX have the best point estimates, but neither resolves an improvement over the stronger score-proportional rule. The 58.22% VDN result uses 80% evaluation-time random beam selection and cannot be attributed to the learned policy.

## Training behavior

| Algorithm | First 10 discovery | Last 10 discovery | First 10 return/UAV | Last 10 return/UAV |
|---|---:|---:|---:|---:|
| Shared-IDQN | 60.89% | 55.78% | 11.04 | 10.34 |
| VDN | 56.00% | 54.22% | 10.31 | 9.95 |
| QMIX | 57.33% | 52.22% | 10.44 | 9.46 |
| Beam-only MAPPO | 60.44% | 55.78% | 10.90 | 10.22 |

All four last-ten means are below their first-ten means. Thirty episodes therefore do not support a convergence claim. MAPPO's weaker pure result may reflect on-policy sample inefficiency under this short budget, while VDN/QMIX may benefit from centralized team-value credit assignment; both statements are hypotheses, not established algorithm conclusions.

## Audits

- Training role hashes are identical across all algorithms for every one of 30 episodes.
- Every held-out scenario has 28 identical role traces: four algorithms times seven execution variants.
- All 30 non-learning control/scenario rows match across algorithms on nine metrics and trace fields.
- MAPPO has no mode head, mode log-prob, mode entropy, access gate, rule residual, behavior cloning, or auxiliary action target.
- MAPPO's actor uses local observations only; its pooled global state is critic-only during CTDE training.
- Algorithm-to-algorithm differences are descriptive fixed-checkpoint contrasts. Multiple independent training seeds are required for inference about the training algorithms.

## Decision

The fair comparison is complete, but no method is yet paper-ready. VDN is the leading candidate (`56.22%` pure), followed by QMIX (`55.33%`); Shared-IDQN is neutral and MAPPO is currently below both non-learning controls. The next compute should prioritize VDN and QMIX with longer training and multiple independent seeds while retaining Shared-IDQN and MAPPO as fixed-budget references. Transfer and large-network tests remain blocked until a pure policy beats both candidate-random and score-proportional controls.

## Statistical fallacy scan

Coverage: **11/11 checked**. The report makes no cross-configuration robustness claim, does not infer individual competence from team discovery, retains every planned algorithm and scenario, uses unchanged paired denominators, avoids post-outcome selection and post-treatment controls, Holm-corrects the 12 exploratory control comparisons, does not treat the one-seed screen as algorithm inference, and limits causal interpretation to the controlled simulator.

## Artifacts

- `06_analysis/paper_tables/beam_only_algorithm_gate_20260711/evaluation_summary.csv`
- `06_analysis/paper_tables/beam_only_algorithm_gate_20260711/paired_control_comparisons.csv`
- `06_analysis/paper_tables/beam_only_algorithm_gate_20260711/run_contracts.csv`
- `06_analysis/paper_tables/beam_only_algorithm_gate_20260711/training_role_sequence_audit.csv`
- `06_analysis/paper_tables/beam_only_algorithm_gate_20260711/evaluation_role_sequence_audit.csv`
- `06_analysis/paper_tables/beam_only_algorithm_gate_20260711/nonlearning_control_reproducibility.csv`
- `06_analysis/paper_tables/beam_only_algorithm_gate_20260711/algorithm_pure_vs_nonlearning_controls.png`
- `06_analysis/paper_tables/beam_only_algorithm_gate_20260711/algorithm_evaluation_random_mix.png`
- `06_analysis/paper_tables/beam_only_algorithm_gate_20260711/algorithm_training_discovery.png`
