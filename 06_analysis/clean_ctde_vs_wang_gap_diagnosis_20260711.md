# Clean CTDE versus Wang: performance-gap diagnosis

## Scope and evidence level

- Scenario: `N=10`, planar Gauss-Markov mobility, 24 azimuth beams (15 degrees), one RF chain, 300 slots at 5 ms/slot.
- Evaluation: 10 held-out scenario seeds (`26260720` to `26260729`).
- Training evidence: one Clean CTDE training seed, 30 episodes x 300 steps = 9,000 environment steps.
- This is a diagnostic screen, not final paper evidence. The paired sample is small and the trained policy has only one training seed.

## Main results

| Method | Discovery rate | Edges | Empty-beam actions | Undiscovered-target beam actions | Known-only beam actions | Aligned opportunities | TX fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean CTDE | 0.2956 | 13.3 | 0.6489 | 0.3115 | 0.0396 | 16.2 | 0.5909 |
| Wang 2025 | 0.5444 | 24.5 | 0.4368 | 0.4253 | 0.1379 | 30.4 | 0.4977 |
| Uniform random | 0.1822 | 8.2 | 0.7203 | not logged | not logged | 9.7 | 0.4978 |

Clean CTDE is 0.2489 below Wang (paired 95% t interval `[-0.3561, -0.1417]`, exact sign-flip `p=0.00195`). It is above uniform random by 0.1133 in the earlier paired screen, but this is insufficient for the intended contribution.

## Causal action decomposition

The same checkpoint and scenario seeds were evaluated with execution-only diagnostic substitutions. No hidden topology was supplied to any actor.

| Beam executor | Mode executor | Discovery rate |
|---|---|---:|
| learned | learned | 0.2956 |
| learned | uniform TX/RX | 0.3156 |
| local candidate random | learned | 0.2689 |
| local candidate random | uniform TX/RX | 0.2844 |

The 10-scenario paired intervals all cross zero for these diagnostic differences. Therefore the defensible conclusion is not that the mode head is harmful, but that there is currently no evidence that it improves over 50/50 TX/RX. The learned beam head adds only about 0.031 over random sampling from the same local mask when the mode executor is held uniform. Most of the gain over blind random comes from the ISAC/table mechanism, not from a strong learned action policy.

## Why the gap occurs

### 1. The PHY is not the bottleneck

Clean and Wang both decode about 83% of aligned opportunities. Clean obtains 16.2 aligned opportunities versus Wang's 30.4, and the edge-count ratio closely follows the aligned-opportunity ratio. Path loss, fading, interference, and SINR rejection therefore do not explain the main gap.

### 2. TX/RX imbalance is secondary

Clean uses TX in 59.1% of active actions, while Wang is near 50%. The ideal complementary-role factor changes from `2*0.5909*0.4091=0.4835` to `0.5`, only about a 3.4% relative loss. This cannot explain a 45.7% relative discovery deficit. The execution ablation reaches the same conclusion.

### 3. Empty-beam memory is too weak and expires too quickly

Clean spends 64.9% of active beam actions on true empty sectors. Its local candidate mask:

- keeps an empty beam active while its decaying belief remains above the exploration floor;
- generally needs repeated negative observations before exclusion;
- reopens an excluded beam after 50 slots, i.e. only 250 ms.

The measured mean displacement over the full 1.5 s episode is only about 10 m. At kilometer-scale separation, the angular change over 250 ms is far below a 15-degree beam cell in typical cases. A fixed 250 ms reopen time therefore discards valid negative evidence much faster than mobility requires.

Wang closes a beam immediately after sensing no target and keeps it closed unless table information reactivates it. Its empty-action ratio is consequently 21.2 percentage points lower.

### 4. Wang has a residual-target state that Clean lacks

The Wang paper's sensing table stores `Flag`, `Node_num` (remaining potential targets), `Dis_num` (interacted targets), SNR, and target locations. A direction is closed after all targets in that direction have been interacted with. Clean exposes occupancy belief, success/failure history, age, and a candidate score, but no estimate of how many undiscovered targets remain in a beam.

This is a state-design deficit, not merely a PPO hyperparameter issue. The actor cannot infer an explicit residual opportunity from the current inputs.

### 5. The current Wang reproduction has a favorable count side channel

After a binary positive sensing result, `wang2025_sensing_target_count()` counts all true in-range targets in that beam and writes the exact count into `wang_node_num`. A positive detection of any one target can therefore reveal the exact total target count, including targets that were not independently detected.

The paper motivates target counts through multi-target MIMO-OTFS detection and SIC, but a realistic abstraction should apply detection and angular errors per target. The current implementation is useful as an ideal-sensing upper bound, but it is not yet a strictly common physical interface for Wang and MARL.

### 6. Table exchange is information-poor for Clean

The generic sensing report stores only the nearest anonymous target position per beam. It does not exchange a multi-target list or residual count. Shared reports boost belief and success history, but do not tell the actor whether the direction still contains undiscovered targets. This is weaker than the paper's multi-target sensing-table abstraction.

### 7. Training is short, but more training is not the first fix

Thirty episodes and one training seed are insufficient for a final MARL conclusion. However, training longer with the current state will not supply missing residual-target information or repair the 250 ms empty-beam forgetting rule. The observation/mechanism contract should be corrected before a long campaign.

## Next tasks, in order

### P0: establish one common physical-to-link sensing interface

1. Replace the binary-observation-plus-exact-count split with one per-target sensing measurement generator.
2. Apply range-dependent `Pd`, angular error, and false alarms to each anonymous target estimate.
3. Feed the identical measurement object to Wang and MARL.
4. Record per beam: detected-count mean/uncertainty, anonymous positions, SNR/confidence, age, and source.
5. Report direct-link discovery separately from indirect neighbor knowledge acquired through table exchange.

Required ablations: ideal exact count (upper bound), noisy count, and binary occupancy only.

### P1: implement a mobility-aware residual opportunity table

For node `i` and beam `b`, maintain an actor-local estimate of

`R_i,b(t) = max(0, estimated targets - confirmed interacted targets)`

with confidence and age. Empty evidence should decay according to an angular-motion transition model based on beamwidth, elapsed time, own motion, and a configured relative-speed bound, rather than a fixed 50-slot timeout. Table fusion must remain confidence- and age-aware and must never use global truth at execution.

### P2: retrain a clean MARL policy on the corrected table

1. Encode per-beam tokens with circular beam adjacency and temporal memory.
2. Use a mode-conditioned beam policy rather than fully independent mode and beam heads.
3. Retain CTDE: only the critic sees global training state.
4. Use a weak batch-level TX/RX balance regularizer if needed; do not add deterministic role labels or rule imitation.

### P3: use explicit go/no-go gates before transfer

On `N=10`, 15-degree, 300-slot tests with at least 3 training seeds and 20-30 paired evaluation seeds:

1. The corrected residual-table mechanism must materially reduce empty-beam actions versus the current Clean implementation.
2. Learned beams must beat random selection from the same corrected local candidate set.
3. Learned modes must beat or at least not degrade uniform TX/RX.
4. Full MARL must beat the realistic Wang baseline with a paired confidence interval excluding zero.

Do not start `N=100`, 3D, or beamwidth transfer experiments until these gates pass.

## Immediate decision

The next implementation should be the common multi-target sensing measurement and residual opportunity table. Longer training, role-reward tuning, and transfer experiments should be deferred until that interface is in place.
