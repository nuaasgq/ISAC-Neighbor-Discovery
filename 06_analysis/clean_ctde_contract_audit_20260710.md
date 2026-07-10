# Clean CTDE Contract Audit (2026-07-10)

## Information Boundary

The corrected boundary is based on information provenance, not on whether a feature is produced by a rule:

- Allowed for the decentralized actor: own kinematics, local ISAC beam observations, local history, candidate ranking/masking derived from those observations, and tables received after a successful handshake.
- Allowed for the centralized training critic and reward calculation: global positions, true/discovered adjacency, and global topology metrics.
- Forbidden for the actor: global truth, hidden neighbor state, pair-specific rendezvous phase, deterministic complementary TX/RX recommendation, and action-supervision labels derived from those mechanisms.

Candidate processing remains part of the cross-layer protocol. The learning contribution must be measured after giving all compared learned variants the same local candidate/table interface.

## Table-Exchange Path

The default ISAC-MARL environment protocol is `improved_rl_isac_tables`. After a successful handshake, both endpoints exchange confirmed neighbor records and recent anonymous sensing reports. The receiver projects received positions into its own beam grid and updates local `belief`, `success_count`, `age`, and `last_positive_slot`. These updated local arrays feed the actor's `beam_belief` and `candidate_score` observations.

Tests cover both directions of the boundary:

- exchanged table entries change the receiving actor's local candidate features;
- an edge inserted only into global truth does not create a table-derived candidate without a received record.

## Enforcement Smoke

Command profile: `run_marl_training.py --clean-ctde --candidate-mask --topology-deficit`.

Resolved checkpoint/manifest fields:

- `training_contract_version = clean_local_ctde_v1`;
- centralized critic state present;
- `rendezvous_observation_enabled = false`;
- local candidate mask and score enabled;
- `rule_residual = false` and `contention_mode_prior = false`;
- behavior cloning and all beam/role action-target auxiliary coefficients equal zero;
- environment protocol `improved_rl_isac_tables` keeps post-handshake table exchange active.

The 74% planar result has been relabeled `Rule-guided MARL` and remains an upper bound only. It is not clean-MARL evidence.
