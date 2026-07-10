# Value-based MARL algorithm gate

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-11
- Verification Status: ANALYZED
- Scope: one training seed, 30 episodes x 300 slots, 10 paired held-out scenarios

## Questions

### Is one DQN per UAV still MARL?

Yes. Independent DQN/IQL treats every UAV as an independent learner in a multi-agent environment. It removes the centralized critic or mixer, but it does not turn the problem into a single-agent problem.

Strict independent DQN was implemented with one online network, target network, optimizer, replay stream, and local TD loss per UAV. The team-reward version broadcasts the same training-only mean reward to every local replay; the local-reward version uses only each UAV's own reward.

### Is centralized value decomposition necessary?

Not supported by the current screen. VDN and QMIX do not improve over shared-IDQN or the same-mechanism random control. Their assumptions are also structurally mismatched with complementary TX/RX roles: VDN is additive, while QMIX requires monotonic local action ordering despite role preference reversals and interference.

### Is MAPPO currently the best choice?

No. MAPPO is theoretically more expressive than VDN/QMIX, but the current MAPPO result is not better than strict IDQN, shared-IDQN, or random control with statistical resolution. Its current factorized role/beam action head also differs from the value learner's joint `2B` Q head, so the screen cannot isolate optimizer choice from action-head expressiveness.

## Common-contract results

All rows use the residual candidate mechanism, anonymous `noisy_count` sensing, the same PHY/MAC/table rules, 24 planar beams, and the same held-out scenarios.

| Method | Matched support | Greedy | Random role + learned beam | Random control |
|---|---:|---:|---:|---:|
| strict IDQN, team reward | 0.5622 | 0.2000 | 0.4911 | 0.5356 |
| strict IDQN, local reward | 0.5244 | 0.1844 | 0.5044 | 0.5356 |
| shared-IDQN | 0.5111 | 0.0289 | 0.5311 | 0.5356 |
| VDN | 0.4800 | 0.0000 | 0.5200 | 0.5356 |
| QMIX | 0.5178 | 0.3422 | 0.5267 | 0.5356 |
| residual-v2 MAPPO | 0.5311 | not rerun with a joint head | 0.5222* | 0.5778* |

`*` The MAPPO head controls came from the preceding common-measurement screen and used its recorded policy RNG. They share physical scenarios but are not the within-run value-policy random control.

Strict team-reward IDQN is `+0.0267` above its random control, but the 95% interval is `[-0.0346, 0.0879]` and exact sign-flip `p=0.391`. It is `+0.0311` above MAPPO, with interval `[-0.0334, 0.0957]` and `p=0.332`. Neither difference is resolved.

The learned components do not support a learning claim:

- strict IDQN greedy versus random: `-0.3356`, `p=0.00195`;
- strict IDQN learned role plus random beam versus random: `-0.1978`, `p=0.00195`;
- strict IDQN random role plus learned beam versus random: `-0.0444`, `p=0.309`;
- local-reward IDQN matched support versus random: `-0.0111`, `p=0.826`.

Shared-IDQN and VDN training discovery also decreases as epsilon decays. This indicates that random exploration, rather than learned Q ordering, carries much of their final protocol performance.

## Interpretation

For symmetric independent role selection, if each side selects TX with probability `p`, the complementary probability is `2p(1-p)`, whose maximum is `0.5` at uniform TX/RX. Without a reciprocal pair-specific signal, learning a global role bias cannot improve this probability. This explains why IDQN, VDN, QMIX, and MAPPO can all collapse toward TX or RX and why a strong random wrapper masks weak learned policies.

The current actor exposes count, confidence, residual opportunity, history, and candidate scores, but not an explicitly antisymmetric anonymous target token that two reciprocal detections could use to break role symmetry. Therefore the present algorithm comparison is primarily diagnosing an information/action-parameterization limitation, not identifying a winning optimizer.

## Decision

Do not continue generic MAPPO optimization on the current factorized action contract. The conditions "MARL is empirically necessary" and "MAPPO is better" are not met.

The next gate should compare two scientifically motivated formulations:

1. **Beam-only learning:** keep TX/RX uniformly random as a protocol mechanism and train shared-IDQN versus a beam-only PPO actor under the same encoder and evaluation control.
2. **Anonymous antisymmetric role learning:** expose only locally sensed anonymous relative-position tokens and use an antisymmetric/equivariant role head; no target ID, true adjacency, oracle role label, or hard rendezvous schedule is allowed.

Only if one formulation beats its same-mechanism random control should it receive multi-seed long training. Strict node-specific IDQN remains an `N=10` diagnostic because its parameter count and inference work grow with `N`; it cannot support the small-to-large transfer claim.

## Statistical fallacy scan

Coverage: **11/11 checked**.

| Check | Assessment |
|---|---|
| Simpson's paradox | No subgroup aggregation claim is made; only one `N/B` configuration is analyzed. Robustness across configurations remains untested. |
| Ecological fallacy | Group discovery rates are not used to infer an individual UAV's learned competence. |
| Berkson's paradox | No outcome-conditioned scenario selection is used. |
| Collider bias | No post-treatment covariate is controlled in the paired comparison. |
| Base-rate neglect | Not a diagnostic-classification study; direct discovery denominator is reported. |
| Regression to the mean | Methods were not selected from extreme held-out scores for a pre/post claim. |
| Survivorship bias | All launched v2 runs and all held-out scenarios are retained; the superseded pre-screen is explicitly excluded from formal comparisons. |
| Look-elsewhere effect | **Caution:** several algorithms and five execution variants are screened without multiplicity correction. P-values are diagnostic and no positive significance claim is made. |
| Garden of forking paths | **Caution:** this is an exploratory one-seed gate following earlier debugging. It is not confirmatory paper evidence. |
| Correlation versus causation | Paired controlled simulation supports within-simulator method contrasts only, not real-flight causal effectiveness. |
| Reverse causality | Not applicable to the randomized simulator intervention; training behavior is not inferred from a cross-sectional association. |
