# Prior Audit and Clean-MARL Probe (2026-07-10)

## Purpose

This audit checks whether the previous split-beam MARL result can be attributed to learned beam selection, or whether it depends on handcrafted priors. The environment is kept aligned with the Wang-style sensing-table setting: N=10, 105 beams, 200 slots, single RF, TX-coupled sensing, no standalone Sense action.

Update: local candidate ranking/masking and post-handshake table exchange remain allowed because they use independently available node observations. Pair-derived rendezvous phase, deterministic TX/RX role hints, and supervised action targets are not allowed in the clean main method even when they are computed from noisy local position reports.

## Code-Level Prior Check

- No oracle neighbor identity, hidden adjacency, or true position is directly exposed to the decentralized actor.
- `candidate_mask` and `candidate_score` are local ISAC/table observations. They are acceptable as physical-layer sensing abstractions if shared by all compared methods.
- `rule_residual` is a strong handcrafted prior: it adds `rule_mode_logits` to mode logits and adds `candidate_score - 0.85 * beam_collision_norm` to beam logits. This contaminates claims that the beam head learned the selection rule independently.
- `_contention_mode_prior()` was previously added unconditionally to mode logits. A new `--disable-contention-mode-prior` switch now makes this prior explicit and auditable.

## Checkpoint Re-Evaluation

Checkpoint: `split_light_train/checkpoint_ep00020.pt`.

20 paired stochastic episodes, seeds `2026084001` to `2026084020`:

| Setting | Full discovery | Wang-random-beam discovery | Full minus random | Full delay | Random delay | Full lambda2 | Random lambda2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| residual on, mode prior on | 0.6444 | 0.6000 | +0.0444 | 141.43 | 149.60 | 2.912 | 2.643 |
| residual scale 0, mode prior on | 0.5256 | 0.5011 | +0.0244 | 157.75 | 157.80 | 2.135 | 1.722 |
| residual scale 0, mode prior off | 0.4867 | 0.4489 | +0.0378 | 162.93 | 165.99 | 1.610 | 1.175 |

Paired discovery-delta signs:

| Setting | Positive | Negative | Zero |
|---|---:|---:|---:|
| residual on, mode prior on | 12 | 7 | 1 |
| residual scale 0, mode prior on | 12 | 6 | 2 |
| residual scale 0, mode prior off | 13 | 6 | 1 |

Interpretation: the beam head signal is positive but weak. It does not meet the current robustness gate of `+0.05` discovery-rate improvement over same-mode Wang-random beam. Therefore the previous `+0.1067` result from 5 episodes should not be used as a main claim.

## Clean Re-Training Probe

Two 30-episode clean probes were trained with `--rule-residual-scale 0.0` and `--disable-contention-mode-prior`.

| Probe | Beam-rank aux | Best/last eval discovery | Main failure mode |
|---|---:|---:|---|
| clean split aux | 0.01 | about 0.30 | over-selects TX, too few RX actions |
| clean split no-aux | 0.00 | about 0.38 | excessive idle and weak active access |

Interpretation: once the handcrafted priors are removed, the current policy/reward setup does not learn a strong access/beam strategy within 30 episodes. The next research step should not be scalability or paper writing; it should be mechanism redesign for clean MARL learning.

## Decision

Current claim boundary:

- Acceptable: ISAC candidate-table information improves the action space and can support learned policies.
- Not yet acceptable: the current MARL beam head has a robust standalone contribution beyond Wang-candidate random beam.
- Not yet acceptable: the current clean MARL setup is paper-ready.

Next safe step: redesign the clean policy objective around balanced TX/RX access and beam learning without injecting handcrafted logits directly into the policy. Any new mechanism must be evaluated first by same-mode random-beam ablation before expanding to N=100 or beamwidth transfer.

## Output Directories

- `05_simulation/results_raw/marl_campaign/prior_audit_20260710`
- `05_simulation/results_raw/marl_campaign/prior_audit_20260710_20ep`
- `05_simulation/results_raw/marl_campaign/prior_audit_20260710_training`
