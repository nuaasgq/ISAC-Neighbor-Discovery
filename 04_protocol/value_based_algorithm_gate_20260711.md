# Value-based versus MAPPO algorithm gate

## Question decomposition

1. A DQN maintained and trained by every UAV is Independent DQN/IQL. It is a multi-agent learning method even though it has no centralized critic or mixer.
2. Independent-parameter IDQN satisfies decentralized execution and can satisfy decentralized training, but it creates `N` node-specific models. A model set trained at `N=10` cannot directly supply 90 new node-specific policies at `N=100`.
3. Shared-IDQN uses one homogeneous local Q network for all UAVs. It preserves decentralized execution and is the simplest scalable value-based candidate.
4. VDN and QMIX use centralized value aggregation only during training. Their local utility networks still execute from local observations.
5. Current evidence does not establish that MAPPO is better than these alternatives.

## Structural fit

The local action is discrete: `TX/RX x beam`. Off-policy replay can reuse rare successful discoveries, which is favorable to DQN-style methods. The difficulty is that a discovery requires complementary roles and reciprocal beams, while interference adds action-dependent negative externalities.

If two nodes independently choose TX with the same probability `p` and have no pair-specific shared role signal, their complementary-role probability is

```text
P(complementary) = p(1-p) + (1-p)p = 2p(1-p) <= 1/2.
```

Uniform TX/RX is therefore already optimal inside this symmetric independent-role class. A learned global TX/RX bias cannot improve rendezvous probability and can easily collapse toward one role. Learning can improve role selection only if local observations contain a reciprocal symmetry-breaking variable, such as anonymous target position relative to the UAV's own GNSS position, or if the protocol keeps roles random and learns only beam choice.

For one aligned pair, ignoring equal action costs, the role payoff has the form:

| | UAV j: TX | UAV j: RX |
|---|---:|---:|
| UAV i: TX | 0 | G |
| UAV i: RX | G | 0 |

VDN cannot represent this table additively because the two diagonal values sum to `0`, while the two off-diagonal values sum to `2G`. QMIX is more expressive, but its monotonic individual-utility ordering cannot exactly represent the preference reversal: the preferred action of UAV i changes when UAV j switches between TX and RX. Therefore VDN and QMIX are useful baselines and approximations, not theoretically privileged solutions.

MAPPO has no additive or monotonic value-factorization constraint and naturally represents stochastic protocols. However, its current centralized team advantage has high credit-assignment variance, and the existing factorized mode/beam action head is less expressive than the value learner's joint `2B` action head. A direct result therefore compares complete algorithm packages, not only the optimizer. If value learning wins the first gate, a joint-action MAPPO control is required before rejecting policy-gradient learning.

## Common gate contract

- `N=10`, 15-degree planar beams (`B=24`), 300 slots, 5 ms per slot, one RF chain.
- Common mobility, communication PHY, anonymous `noisy_count` sensing, table exchange, duplicate suppression, residual candidate mask, and held-out scenario seeds.
- Execution input is local anonymous sensing/table state only.
- Action encoding is `a_i = role_i * B + beam_i`, with `role=0` for TX and `role=1` for RX.
- MAPPO and team-value methods use `mean_i(r_i)` as the training team reward.
- Strict IDQN owns one online network, target network, optimizer, replay stream, and TD loss per UAV. Team reward is broadcast only in the fair algorithm gate; a local-reward IDQN is a separate reward-design ablation.
- Shared-IDQN, VDN, and QMIX share one local utility network. Only QMIX reads global simulator state during training.
- Training interaction budget, local encoder size, target update schedule, candidate mask, stochastic support, and evaluation scenarios are recorded in manifests.

Every value policy is evaluated in five modes:

1. matched stochastic support;
2. pure greedy Q action;
3. learned role plus random candidate beam;
4. random role plus learned beam;
5. random role and random candidate beam.

Mode 5 is the same-mechanism causal control. A learning claim requires a paired improvement over it.

## Decision rule

- If strict IDQN wins only at `N=10`, treat node specialization as a diagnostic, not a scalable main method.
- If shared-IDQN matches or exceeds VDN, QMIX, and MAPPO, prefer it as the lowest-complexity scalable learner.
- If VDN/QMIX significantly exceeds shared-IDQN, centralized value decomposition adds useful training information despite its representational constraints.
- Continue the MAPPO architecture route only if a same-contract MAPPO variant beats the best scalable value learner and the same-mechanism random control.
- If no learner beats random control, retain the residual ISAC/table mechanism as the supported contribution and redesign the learning target/action parameterization before any transfer experiment.

## Primary references

- Mnih et al., "Human-level control through deep reinforcement learning," Nature, 2015: https://doi.org/10.1038/nature14236
- Sunehag et al., "Value-Decomposition Networks for Cooperative Multi-Agent Learning," 2017: https://arxiv.org/abs/1706.05296
- Rashid et al., "Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning," JMLR, 2020: https://jmlr.org/papers/v21/20-081.html
- Yu et al., "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games," 2021: https://arxiv.org/abs/2103.01955
