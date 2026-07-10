# Fair comparison contract for ISAC-assisted neighbor discovery

## Scope

The invariants below apply to methods admitted to the decision-policy comparison. A capability ablation may explicitly remove ISAC or table exchange, but it must be labelled as an ablation and cannot support a causal claim about the action policy.

## Invariants shared by every decision-policy comparator

1. Identical scenario seed, mobility trajectory, slot duration, node geometry, beam codebook, RF-chain count, communication range, and sensing range.
2. Identical communication PHY: path loss, directional gain, shadowing, Rician fading, interference, SINR threshold, and two-phase HELLO/ACK decision.
3. Identical physical-to-link ISAC measurement object. Each anonymous target is detected independently from range-dependent `Pd` and angular uncertainty. No protocol may read true target count or target identity before a successful handshake.
4. Identical table payload and exchange trigger. A new direct handshake atomically exchanges pre-exchange snapshots of confirmed neighbor records and anonymous sensing reports.
5. Anonymous sensing reports retain opaque detection ID, origin node, origin slot, SNR, confidence, and position. Forwarding preserves origin time and deduplicates by detection ID.
6. Identical duplicate-edge suppression and direct-link definition.
7. Methods may differ only in local table update, candidate construction, and stochastic action policy unless an ablation explicitly states otherwise.

## Comparator hierarchy

- `uniform_random` is the blind-search lower bound. It removes ISAC and table assistance, so its gap measures the combined value of sensing, table processing, and decision logic.
- Wang uses the common PHY/MAC and common table payload but retains the paper-specific local candidate-table update. Its gap measures an end-to-end protocol difference, not MARL alone.
- Residual-candidate random beam plus uniform TX/RX uses exactly the same local sensing/table/candidate mechanism as residual MARL. It is the primary causal control for learned decision quality.
- Learned-beam/uniform-mode and random-beam/learned-mode controls isolate the two policy heads.

Only comparisons within the same capability level may be described as action-policy gains.

## Sensing information modes

- `ideal_count`: all targets in the sensed beam are detected with exact count and position. This is an upper bound.
- `noisy_count`: each target is detected independently; count, position, confidence, and variance come only from successful anonymous detections. This is the primary comparison mode.
- `binary_occupancy`: the same noisy detector is compressed to one occupied/empty bit and at most one anonymous position. This is an information ablation.

## Metrics that must not be conflated

- `discovery_rate`: fraction of true undirected links that completed a direct bidirectional handshake.
- `neighbor_knowledge_recall`: fraction of directed node records known locally, including indirect table acquisition.
- `indirect_knowledge_rate`: directed records acquired without establishing the corresponding direct edge.
- `networking_completion_slot_censored`: first slot when every node knows all other node records; censored at the episode limit.
- `per_target_sensing_recall`, `sensing_count_mae`, and `sensing_position_rmse_m`: PHY-to-link sensing fidelity.

Wang-paper networking completion and direct-link discovery answer different questions. Both must be reported.

## MARL execution boundary

Actors may use own kinematics, local anonymous sensing/table state, confirmed or indirectly received neighbor records, local interaction history, and locally computed candidates. Only the training critic may use global state. No target ID, true adjacency, pair-derived role label, oracle beam, behavior-cloning action, or deterministic rendezvous schedule may enter execution.

## Required causal controls

Every learned policy is evaluated with the same physical, MAC, sensing, table, and candidate mechanisms under:

1. full learned mode and beam;
2. learned beam plus uniform TX/RX;
3. local-candidate random beam plus learned mode;
4. local-candidate random beam plus uniform TX/RX.

The learned method contributes only if it beats control 4 on paired held-out scenarios. Beating blind random or Wang while losing to the same-mechanism random control is not sufficient evidence of MARL value.
