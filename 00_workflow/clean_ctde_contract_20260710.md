# Clean CTDE Contract (2026-07-10)

## Decision

The main learned method must use centralized training and decentralized execution without action-teacher rules. Results produced by the position-pair rendezvous phase, deterministic TX/RX role hint, rendezvous action labels, or learned adapters fitted to those labels are classified as **rule-guided upper bounds**, not clean MARL evidence.

## Decentralized Actor Boundary

At execution time, actor `i` may use only information independently available at UAV `i`:

- own position, velocity, attitude, synchronized slot index, and previous action;
- local per-beam ISAC belief, uncertainty, age, empty evidence, sensing confidence, and sensing SNR;
- locally observed handshake success/failure, PHY outage, collision feedback, and per-beam history;
- the node's own discovered-neighbor table and information received after a successful handshake;
- local discovered degree or another statistic computable from that table.

The actor must not receive:

- undiscovered-neighbor identity, true position, true beam, hidden adjacency, or global topology;
- a position-pair hash or any pair-specific rendezvous phase;
- a deterministic complementary TX/RX role hint;
- an action label derived from a rule protocol, including behavior cloning and role/beam cross-entropy targets;
- rule action logits or deterministic mode-logit priors in the main clean method.

The following local protocol processing is allowed and should be shared fairly across learned methods and baselines:

- ranking beams from the node's own ISAC occupancy belief, uncertainty, recency, and local success/failure history;
- excluding beams using a mask derived only from local empty-beam evidence, with an exploration/fallback rule to avoid permanent false-negative exclusion;
- updating local beam belief from neighbor and sensing tables received after a successful handshake;
- projecting a received table entry into the receiver's own beam grid using its own position and attitude.

These operations transform independently available local observations. They must not query current global truth or output a pair-specific TX/RX schedule.

The PHY simulator may use global truth to generate noisy local measurements and determine channel outcomes. That truth must not be exposed directly or through a derived action recommendation to the actor.

## Centralized Critic Boundary

During training only, the critic may use `training_state()` fields such as all UAV positions, velocities, true/discovered adjacency, and pooled global beam belief. The critic produces values or advantages only. It must not generate execution actions, masks, role hints, beam targets, or actor observations.

Evaluation and deployment instantiate actors without the critic and without access to `training_state()`.

## Enforced Training Profile

Use `run_marl_training.py --clean-ctde`. The profile:

- requires MAPPO/ISAC-MAPPO-style centralized training;
- forces rendezvous observations off even if a YAML enables them;
- permits local candidate score/mask and exchanged-table features;
- disables rule residual, contention mode prior, rendezvous adapter, behavior cloning, and beam/role action-target auxiliary losses;
- rejects conflicting options with a startup error;
- records `actor_observation_contract=clean_local_ctde_v1` and `action_teacher_free=true` in checkpoints/manifests.

The current 2-D rule-guided result is retained only as a feasibility upper bound. A new clean run must first beat uniform random at `N=10` before comparison with Wang or any transfer experiment.
