# TWC-Oriented Training Contract (2026-07-10)

## Scope

This contract defines the first scientifically auditable MARL environment after the 2026-07-10 refactor. It establishes a trustworthy training baseline; it does **not** mean that the simulator or results are already sufficient for an IEEE TWC submission.

## Environment Contract

- Network: fully distributed UAV-to-UAV neighbor discovery; an actor sees only its own state and local protocol memory.
- Mobility: three-dimensional Gauss-Markov motion by default, with 5 ms slots.
- Radio: one RF chain; one selected beam is active in one slot.
- Action: `TX/RX x beam` by default. Standalone `SENSE` and `IDLE` require explicit opt-in.
- ISAC: sensing is piggybacked on TX only. One TX observes one sensing cell unless an explicit footprint radius is configured.
- Handshake: mutually aligned TX/RX candidates are decoded through aggregate-interference SINR. A receiver decodes at most one strongest eligible HELLO, so near-far capture is possible; the reciprocal ACK must also decode.
- Duplicate handling: all methods use the same duplicate-response suppression rule.
- Range: sensing uses `sensing_range_m`; communication success remains bounded by `communication_range_m`.
- Table exchange: confirmed neighbor positions and noisy anonymous sensing-position estimates may be shared after a successful handshake. Table exchange cannot query the current global target identity, direction, or topology.
- Metrics: `lambda2` and connected-component metrics use currently active discovered links; `knowledge_lambda2` records the cumulative discovery-memory graph.

## Clean MARL Defaults

- Algorithm: ISAC-MAPPO with CTDE and decentralized parameter-shared actors.
- Network: contention-aware shared actor-critic.
- ISAC input: soft local candidate score enabled.
- Disabled by default: hard candidate mask, topology-deficit token, rule residual, contention mode prior, behavior cloning, standalone sensing, and idle action.
- Sensing and policy random-number streams are independent.
- Evaluation saves/restores training RNG state and uses a checkpoint/config fingerprint for resume safety.
- Checkpoints store the resolved environment protocol, feature flags, and contract version.

The optional beam-ranking loss is measurement-driven: its target is the local candidate score created from the node's own ISAC/protocol memory. It does not use true neighbor beams. Any paper result using it must report its coefficient and include a zero-coefficient ablation.

## Reference Configuration

`05_simulation/configs/twc_canonical_n10_b10.yaml` is the authoritative pre-training configuration:

- `N=10`, 10-degree azimuth/elevation cells, 648 beams;
- one RF chain and 300 slots per episode;
- communication and sensing radii of 18 km, exceeding the 10 km cube diagonal;
- radar-SNR sensing abstraction with MIMO-OTFS metadata;
- close-in/Rician/SINR communication PHY with an energy-normalized sectored antenna;
- one shared waveform TX power for communication, sensing, and radio-energy accounting;
- Gauss-Markov UAV mobility.

Recommended first convergence run:

```powershell
python 05_simulation/run_marl_training.py `
  --episodes 2000 --slots 300 `
  --separate-action-loss --beam-rank-aux-coef 0.05 `
  --eval-interval 50 --eval-episodes 10 --stochastic-eval `
  --checkpoint-interval 50 `
  --output 05_simulation/results_raw/twc_n10_b10_seed20260705
```

This is a campaign command, not a claim that 2000 episodes is optimal. Convergence, multiple seeds, and held-out comparison must determine the final budget.

## Verification Completed

- Full test suite: 96 tests passed after the PHY/MAC and rendezvous-observation updates.
- Three-episode/20-slot smoke: checkpoints, losses, per-step logs, held-out evaluation, runtime metadata, and resource logs were generated.
- Policy update check: 28 of 30 policy tensors changed between episode 1 and the final checkpoint; parameter delta L2 was approximately 0.0302.
- One-episode/300-slot TX/RX smoke: 3000 active actions, zero standalone sensing actions, zero idle actions, finite PPO/value/beam-ranking losses, and peak memory below configured limits.

The 2026-07-10 baseline gate found zero reciprocal beam-alignment opportunities for random, Wang-table, and rule-ISAC methods in the same first three B=10/300-slot seeds. This is a failed learnability gate, not evidence of method performance. MARL training remains blocked until the sensing-to-rendezvous mechanism yields nonzero opportunities without simulator-truth leakage.

## Invalidated Evidence

Results generated before this contract must not be combined with post-refactor results when any of the following old semantics were active:

- candidate-pool beams could create a handshake without being selected;
- a selected beam implicitly sensed a 3-by-3 cell area;
- RX actions generated ISAC sensing feedback;
- a single TX could discover multiple receivers in one slot;
- table exchange inferred hidden targets from current simulator truth;
- MARL motion used stale distance/beam caches;
- ISAC-MAPPO silently forced hard masks and rule priors.

## Remaining TWC Blockers

1. Calibrate the implemented close-in/Rician/SINR communication PHY against cited UAV/mmWave operating points, then add blockage, Doppler/temporal correlation, and BLER sensitivity where justified.
2. Calibrate the sensing detector and communication PHY against cited MIMO-OTFS/mmWave parameters; do not treat waveform metadata as waveform simulation.
3. Establish learnability with multi-seed convergence runs before any scale-transfer campaign.
4. Run common-environment comparisons against random, Wang-aligned, MARL without ISAC, base MARL with ISAC, and the proposed network/learning enhancements.
5. Report confidence intervals, paired-seed effect sizes, failure cases, ablations, and compute/resource budgets.

## Version Control

The repository of record remains `https://github.com/nuaasgq/ISAC-Neighbor-Discovery`. Code, configs, tests, and documents are committed and pushed after review; large raw results remain ignored and are referenced by reproducible commands and manifests.
