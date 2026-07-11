# Beam-only learning transparency gate

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-07-11
- Verification Status: ANALYZED
- Scope: one training seed, 30 episodes x 300 slots, 10 paired held-out scenarios, seven execution policies

## Exact action contract

TX/RX is not learned. Each UAV independently samples TX with probability `0.5` every slot; the Q network outputs only 24 planar-beam values. Role, random-mixture gate, and random-candidate quantile use separate RNG streams with fixed draw counts. Every evaluation row records role, beam, and candidate-mask trajectory hashes.

The following sources of guidance are not interchangeable:

1. **Mechanism guidance:** every policy uses the locally observable ISAC/table `residual_table` mask.
2. **Rule features:** the Q network can read `candidate_score` and other local anonymous sensing/table features.
3. **Training exploration:** epsilon decays from `1.0` to `0.1` over 6000 steps; the guided checkpoint additionally has a persistent random floor of `0.8`.
4. **Evaluation mixture:** learned and candidate-uniform random beam selection are mixed at `0`, `0.2`, `0.5`, `0.8`, or `1.0`.

Thus, `pure_learned_beam` means zero evaluation-time random mixture. It still uses the residual mask, rule features, and a checkpoint trained with epsilon exploration. `random_candidate_beam` is not blind all-codebook random.

## Seven-policy result

| Evaluation policy | Standard epsilon training | Persistent 0.8-guidance training |
|---|---:|---:|
| pure learned beam, mix 0.0 | 51.33% | 54.67% |
| learned + random, mix 0.2 | 51.33% | 50.44% |
| learned + random, mix 0.5 | 55.56% | **60.22%** |
| learned + random, mix 0.8 | 57.33% | 53.33% |
| candidate-uniform random, mix 1.0 | 51.56% | 51.56% |
| candidate-score argmax rule | 47.33% | 47.33% |
| candidate-score proportional rule | 52.89% | 52.89% |

The non-learning rows are exactly identical across checkpoints on every scenario, including beam and candidate-mask trajectory hashes. This confirms that checkpoint identity does not leak into those controls.

## Attribution tests

| Paired comparison | Difference | 95% CI | Exact p | Holm p |
|---|---:|---:|---:|---:|
| standard pure - candidate random | -0.22 pp | [-12.20, 11.75] | 1.000 | 1.000 |
| standard pure - score proportional | -1.56 pp | [-9.49, 6.38] | 0.711 | 1.000 |
| persistent pure - candidate random | +3.11 pp | [-6.55, 12.77] | 0.523 | 1.000 |
| persistent pure - score proportional | +1.78 pp | [-2.77, 6.32] | 0.455 | 1.000 |
| persistent mix 0.5 - candidate random | +8.67 pp | [1.19, 16.14] | 0.039 | 0.469 |

The 60.22% mixed-policy point estimate is exploratory: it includes 50% evaluation-time random beam selection, is based on one trained checkpoint, and does not survive the 12-comparison Holm family. It cannot be reported as pure RL performance or as confirmed superiority.

## Training behavior

| Training regime | First 10 discovery | Last 10 discovery | First 10 return/UAV | Last 10 return/UAV |
|---|---:|---:|---:|---:|
| standard epsilon | 60.89% | 55.78% | 11.04 | 10.34 |
| persistent 0.8 guidance | 54.00% | 53.78% | 9.90 | 9.88 |

Neither curve supports a convergence claim. A small final TD loss does not imply a useful beam ranking. The checkpoint difference is descriptive only; training-regime inference requires multiple independent training seeds.

## Audit and provenance

- All 20 role-sequence audits contain recorded hashes and match independent Bernoulli(0.5) reconstruction.
- All 30 non-learning control audits match across checkpoints on nine result and trace fields.
- Evaluation ran from clean tracked worktree commit `23abc5e` and records checkpoint SHA-256 values.
- The checkpoint processes were launched at commit `9371f38`; their end-of-run manifest queried Git after commit `23abc5e` was created. Training-relevant beam selection code did not change between those commits, but this timing caveat is retained rather than hidden.

## Decision

The action isolation and fairness controls pass. The learned-beam claim fails this gate: neither pure checkpoint resolves an advantage over candidate-random or candidate-score proportional selection. Do not start transfer or large-scale experiments from these checkpoints.

The next gate should keep TX/RX fixed and preserve these seven controls. It should first improve credit assignment for beam choice, then compare a local/difference-style beam reward and a beam-only policy-gradient learner against shared IDQN. Random and rule baselines must be evaluated before and during training, and only a pure learned policy that beats both can advance to multi-seed confirmation.

## Statistical fallacy scan

Coverage: **11/11 checked**. One configuration prevents Simpson-type robustness claims; team discovery is not treated as per-UAV competence; no outcome-conditioned selection, post-treatment control, or changed denominator is used; regression-to-mean and survivorship claims are avoided; 12 exploratory tests use Holm correction; the one-seed gate is not confirmatory; simulator contrasts do not establish real-flight causality; reverse causality is not applicable.

## Artifacts

- `06_analysis/paper_tables/beam_only_gate_20260711/evaluation_summary.csv`
- `06_analysis/paper_tables/beam_only_gate_20260711/paired_comparisons.csv`
- `06_analysis/paper_tables/beam_only_gate_20260711/role_sequence_audit.csv`
- `06_analysis/paper_tables/beam_only_gate_20260711/nonlearning_control_reproducibility.csv`
- `06_analysis/paper_tables/beam_only_gate_20260711/beam_only_evaluation_random_mix.png`
- `06_analysis/paper_tables/beam_only_gate_20260711/beam_only_training_discovery.png`
