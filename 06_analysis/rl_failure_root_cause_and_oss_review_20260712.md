# RL Failure Root-Cause and Open-Source Review (2026-07-12)

## Material Passport

- Mode: experiment validation + focused open-source review
- Evidence: final three-seed 100,200-step A/B/C/D gate, current implementation audit, prior Neighbor_Discovery code audit, public repositories and primary papers
- Verification status: ANALYZED
- Scope: explain why the current learner does not beat the matched ISAC rule baseline; no new training or hyperparameter selection

## Executive Finding

The current result does not show that reinforcement learning is ineffective for neighbor discovery. It shows that the present actor and training recipe do not expose a learnable advantage over the matched ISAC baseline.

The dominant causes are:

1. The policy called `joint_role_beam` actually factorizes the action as independent role and beam distributions. The role head is not conditioned on the selected beam.
2. The decoupled role tower cannot observe per-beam tokens or a counterpart's current intention. It can mainly adjust the marginal TX probability.
3. For two statistically symmetric agents with independent role choices, the complementary-role probability is `2p(1-p)`, whose maximum is `0.5` at `p=0.5`. A learned marginal TX probability therefore cannot systematically beat the matched Bernoulli(0.5) role baseline without a legitimate symmetry-breaking signal.
4. The score-proportional baseline already consumes the local ISAC residual count, confidence, uncertainty, action mask, and exchanged-table state. The neural beam policy is only learning a residual over a strong baseline.
5. The beam policy does learn to reduce empty scans and create more alignments, but it converts fewer aligned opportunities into successful handshakes because it concentrates traffic on attractive beams without observing concurrent peer intentions or instantaneous interference.
6. The custom training recipe is materially weaker than a standard RMAPPO recipe: one rollout environment, 100.2k steps, 334 updates, finite-horizon MC returns, no GAE, no value normalization, five PPO epochs, and entropy coefficient `0.001`.

## Direct Experimental Evidence

The frozen dev20 gate uses the same 20 scenario seeds for every arm and three independently trained policies.

| Arm | Beam executor | Role executor | Discovery rate |
|---|---|---|---:|
| A | score proportional | Bernoulli(0.5) | 52.11% |
| B | learned | Bernoulli(0.5) | 51.37% |
| C | score proportional | learned | 50.93% |
| D | learned | learned | 50.00% |

The component effects are `B-A=-0.74 pp`, `C-A=-1.19 pp`, `D-B=-1.37 pp`, and interaction `-0.19 pp`. Every learned effect is positive in only one of three training seeds.

More importantly, the policy is not behaviorally inert:

| Arm | Empty scans | Aligned opportunities | Success/alignment | PHY outage | Interference-limited failures |
|---|---:|---:|---:|---:|---:|
| A | 44.4% | 29.45 | 79.6% | 4.60 | 1.40 |
| B | 38.3% | 31.97 | 72.3% | 7.17 | 1.68 |
| D | 37.9% | 31.28 | 71.9% | 6.73 | 2.05 |

Thus the learned beam policy removes about `6.1 pp` of empty scans and creates about `2.5` more aligned opportunities per episode, but loses about `7.3 pp` in handshake conversion. The learner has found occupancy exploitation, not reliable distributed rendezvous.

The last-20 training windows contain only `28.8` to `34.3` aligned opportunities per 3,000 agent actions, approximately `1%`. Sampled training logs have no new edge in more than `92%` of logged slots. The reward is well aligned with discovery (`episode return` and `discovery rate` correlation is about `0.986`), but action credit is sparse and conjunctive.

Training diagnostics support premature concentration rather than useful convergence:

- Three-seed first-20 to last-20 discovery: `52.96% -> 52.19%`.
- Total policy entropy: `1.792 -> 0.841`.
- Normalized candidate entropy: `0.957 -> 0.260`.
- Mean approximate KL: about `0.00134`.
- Mean clip fraction: about `0.0039`.
- Explained variance at the end: about `0.455`.
- Critic gradient norm exceeds its clipping threshold in most episodes, while actor gradients remain small.

## Structural Coordination Limit

A discovery requires the product of several events:

`undiscovered true edge x complementary roles x reciprocal beams x successful PHY handshake`.

The current actor samples mode and beam independently. In `neural_recurrent_contention_actor_critic.py`, `mode_dist` and `beam_dist` are separate categorical distributions. In the decoupled architecture, the role encoder reads aggregate contention state, local summary, previous mode, topology deficit, and candidate statistics, but not the selected beam or per-beam tokens.

This prevents a physically meaningful decentralized coordination mechanism. With an even beam codebook and a common global orientation, reciprocal bearings differ by `pi`. A valid local mechanism can therefore choose a beam first and condition the role on its global beam direction so opposite beams obtain complementary roles. This requires an autoregressive policy

`P(beam | local observation) P(role | selected beam, local observation)`,

not the current factorization

`P(beam | local observation) P(role | local observation)`.

The user's assumption that every UAV knows its own position, attitude, and orientation makes this beam-conditioned symmetry breaking feasible without exposing another UAV's identity, trajectory, or global state. It is both a protocol mechanism and a network-structure hypothesis that can be falsified in a two-node test.

## Why Published RL Results Often Look Easier

### Directed antenna Q-learning code

Repository: [kk-1/dantenna-neighbor-disc](https://github.com/kk-1/dantenna-neighbor-disc), GPL-3.0.

The public Q-learning simulator has eight sectors, a static grid, and an eight-action Q table. It does not model TX/RX roles, a two-phase handshake, fading, shadowing, SINR, interference capture, or noisy sensing. A reward of `+1` is returned immediately for a new mutually visible neighbor and `-2` for a repeated pair. Training and evaluation occur online in the same topology. Its reported `N=10` grid is a `10 x 10` deployment, not ten nodes. This code is useful as a minimal sanity task, but copying its numerical gain would not validate our problem.

### DQN aerial neighbor discovery

Paper: [DQN-Driven Adaptive Neighbor Discovery for Directional Aerial Networks](https://arxiv.org/abs/2605.12552).

This work uses eight full-duplex directional transceivers. The agent chooses only a sector, not a TX/RX role. Its local state stores the last ten sector actions, outcomes (`no discovery`, `collision`, `discovery`), and probing distribution. The reward is a dense ternary sign of the change in a local moving-average objective. Training lasts 20,000 neighbor-discovery intervals with a replay buffer of 20,000 and mini-batches of 128. It compares against uniform random sector selection. No public source-code link was found in the paper.

### BeamManagement6G

Repository: [nicomeyer96/qrl-benchmark](https://github.com/nicomeyer96/qrl-benchmark), Apache-2.0.

This is a useful statistical benchmarking reference, but its beam task is single-agent and dense-reward. The action selects an antenna, the environment automatically chooses that antenna's optimal codebook element, reward equals received energy at every step, and reset starts from the globally optimal antenna. Its result cannot be used as evidence that two-ended decentralized discovery should be easy.

### TCOM codebook learning

Repository: [YuZhang-GitHub/Codebook_Learning_RL](https://github.com/YuZhang-GitHub/Codebook_Learning_RL), CC BY-NC-SA 4.0.

This code optimizes physical beamforming vectors using DDPG, dense beamforming gain, a replay buffer, target networks, and a static channel dataset. It is valuable for waveform/codebook design, but it does not solve rendezvous or multi-agent role coordination.

### DDQN beam selection

Repository: [hqyyqh888/DDQN_BeamSelection](https://github.com/hqyyqh888/DDQN_BeamSelection).

This JSAC implementation uses DDQN, replay, and dense spectral-efficiency feedback for centralized beam selection and precoding. It is useful for off-policy engineering patterns, not as a matched neighbor-discovery baseline. No permissive license should be assumed without repository-level verification.

### Official MARL implementations

- [marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy), MIT: official MAPPO/RMAPPO implementation.
- [marlbenchmark/off-policy](https://github.com/marlbenchmark/off-policy), MIT: QMIX, VDN, MADDPG, MATD3, recurrent variants, and prioritized replay.
- [facebookresearch/BenchMARL](https://github.com/facebookresearch/BenchMARL): standardized multi-algorithm benchmarking framework.

The official MAPPO defaults include 32 rollout environments, 10 million environment steps, GAE with `lambda=0.95`, value normalization, feature normalization, clipped value loss, Huber loss, recurrent chunks of length 10, 15 PPO epochs, and entropy coefficient `0.01`. These settings should not be copied blindly, but the gap means our present trainer should be described as a custom PPO implementation, not a validated standard MAPPO reproduction.

## Prior Neighbor_Discovery Repository

The previously submitted local project uses a much easier static 2-D environment, no path loss/fading/SINR/noisy ISAC, strong opportunity-direction features, and dense rewards. Direct discovery is `+10`, and every gossip information item can add another `+10`. The default training budget is 48 million environment steps. Its ordinary `mappo` critic is not fully centralized; IPPO is sometimes as good as or better than MAPPO, while the graph variant provides the larger gain.

That project confirms that an RNN policy can beat open-loop random scanning when the observation contains actionable directional state and the reward is dense. It does not show that the present 100k-step, noisy, two-ended Dec-POMDP should converge under the same conditions.

## Recommended Causal Validation Ladder

No further N=10 transfer training should start before these gates pass.

1. **Algorithm sanity environment.** Two nodes, eight beams, static geometry, fixed complementary roles, no fading/interference, and direct local reward. Run official recurrent DQN/VDN and RMAPPO. Each must beat uniform beam selection. Failure indicates an interface or algorithm implementation problem.
2. **Beam-only ISAC environment.** Keep fixed Bernoulli roles, add noisy ISAC candidate observations and table exchange, but retain ideal PHY. Compare official recurrent DQN/QMIX/RMAPPO against candidate-random and score-proportional executors.
3. **Beam-conditioned role mechanism.** Implement `beam -> conditional role`, using only own attitude, common slot index, and the selected beam's global direction. First test a deterministic reciprocal-direction rule, then a learned residual. A two-node reciprocal-beam test must show role complementarity above 0.5 without global execution information.
4. **PHY restoration.** Add path loss, fading, shadowing, interference, and SINR one component at a time. Track both aligned opportunities and success/alignment so occupancy exploitation cannot masquerade as discovery improvement.
5. **N=10 matched gate.** Require at least five training seeds, common evaluation scenarios, mean discovery gain of at least 3 pp over the score-proportional matched baseline, and positive gain in at least four seeds before any scale or beamwidth transfer.

## Recommended Algorithm Direction

The first reusable implementation should be an official off-policy recurrent value method, not another custom PPO tuning cycle. The action is discrete, reward is sparse, and replay can reuse rare handshake events. Start with recurrent VDN and QMIX from `marlbenchmark/off-policy`; retain official RMAPPO as the on-policy control. Use the same environment wrapper, action mask, observation contract, training budget, and evaluation scenarios for all algorithms.

The publishable method hypothesis should be:

> ISAC narrows the local beam support; a reciprocal beam-conditioned role policy breaks two-ended rendezvous symmetry; an interference-aware recurrent value learner optimizes the residual scheduling decision under decentralized observations.

This hypothesis contains a physical mechanism, a protocol mechanism, and a learning component. Each can be isolated with the causal ladder above.
