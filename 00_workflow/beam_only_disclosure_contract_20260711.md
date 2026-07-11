# Beam-only learning and disclosure contract

## Fixed role contract

- TX/RX is not a neural-network output and has no policy loss or Q value.
- Every UAV independently samples TX with probability `0.5` and RX with probability `0.5` every slot.
- Role RNG is separate from beam RNG.
- Beam mixture-gate RNG is separate from candidate-choice RNG; both consume one draw per UAV per slot under every evaluation mixture.
- Every evaluation variant uses the same role RNG seed, so its TX/RX sequence is identical across beam-policy comparisons.
- Every evaluation row records a per-slot role-sequence hash. Missing or mismatched hashes invalidate the comparison.

## Three guidance layers that must be reported separately

1. **Mechanism guidance:** every beam-only learner uses the local `residual_table` candidate mask derived from anonymous ISAC/table state. A masked beam is unavailable to both learned and random policies.
2. **Training exploration:** DQN uses epsilon-greedy beam exploration. The start, end, decay steps, and any persistent random-mixture floor must be reported.
3. **Evaluation random guidance:** the learned beam distribution may be mixed with a uniform distribution over the current residual candidate set. The exact mixture must appear in every result label and row.

Therefore `pure_learned_beam` means **zero evaluation-time random beam mixture inside the residual candidate set**. It does not mean an end-to-end policy without ISAC candidate guidance, and it was still trained with an explicitly reported exploration schedule.

## Required evaluation variants

| Label | Learned beam contribution | Candidate-uniform random contribution |
|---|---:|---:|
| `pure_learned_beam` | 100% | 0% |
| `learned_beam_random_mix_0.2` | 80% | 20% |
| `learned_beam_random_mix_0.5` | 50% | 50% |
| `learned_beam_random_mix_0.8` | 20% | 80% |
| `random_candidate_beam` | 0% | 100% |

Two non-learning controls are also mandatory because the learner observes the hand-designed `candidate_score`:

- `candidate_score_argmax`: choose the largest score inside the same residual mask.
- `candidate_score_proportional`: sample in proportion to nonnegative scores inside the same residual mask.

Learning gain must be measured against both candidate-uniform random and the stronger of these score controls. Otherwise a network that only copies the hand-designed ranking could be misreported as an RL contribution.

The `random_candidate_beam` row is the causal control for learned beam selection. Blind all-codebook random remains a separate capability lower bound and must not be used to attribute beam-learning gains.

## Training regimes

- `standard_epsilon`: epsilon decays from `1.0` to `0.1`; no persistent candidate-uniform mixture floor.
- `persistent_mix_0.8`: the same epsilon schedule, but candidate-uniform beam mixture never falls below `0.8`.

Both checkpoints must be evaluated under all five variants above. This separates training-time guidance from evaluation-time guidance.

A single checkpoint per regime supports only a diagnostic checkpoint comparison. Claims about the training regimes themselves require multiple independent training seeds, with training seed as the inferential unit.

## Claim gate

A beam-learning claim requires `pure_learned_beam` to improve over `random_candidate_beam` on paired held-out scenarios. A mixed policy may be reported as protocol performance, but it cannot be used as evidence of learned-policy quality unless its pure component is reported alongside it.
