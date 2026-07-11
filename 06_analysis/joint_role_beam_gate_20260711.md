# Joint Role-Beam Gate Audit (2026-07-11)

## Reward-Counter Correction

The original `discovery_first` reward read `success_count` and `fail_count` as handshake events. Those arrays also store TX-side ISAC sensing evidence. Consequently, a learned role policy could increase its reward by selecting TX more often even without improving a handshake.

This path was removed in commit `911cb8b`. The simulator now maintains dedicated per-node handshake success/failure counters, while sensing evidence remains available only through decentralized candidate observations. A regression test verifies that two aligned TX nodes may obtain positive ISAC evidence but receive no handshake-success reward.

All trained checkpoints produced before this correction are diagnostic only and are not valid final MARL results.

## Corrected Single-Seed Screens

Under the corrected reward (`handshake_counters_v2`):

- Beam-only recurrent score-residual MAPPO: `48.33%`.
- Matched score-proportional rule: `53.00%`.
- Shared-trunk joint actor with role floor `0.05`: `53.67%`.
- The same learned beam with forced uniform roles: `53.44%`.

Thus the corrected beam-only learner did not beat the local rule, and the shared joint actor produced only a `+0.22 pp` role increment.

## Factorized ISAC Credit

A beam-only auxiliary credit was implemented from post-action local anonymous ISAC occupancy and confidence. It enters only the beam PPO loss; the mode PPO loss continues to use direct-discovery credit.

With a shared actor trunk, feedback coefficients `0`, `0.1`, and `0.3` gave `51.89%`, `54.00%`, and `45.56%`, respectively. The `0.1` and `0.3` runs shifted the TX ratio to about `70.5%` and `80.5%` because beam gradients changed the shared role representation. This is not an acceptable isolated-credit mechanism.

The final `v4` actor therefore uses disjoint trainable role and beam towers. A gradient test verifies that mode loss cannot update the recurrent beam tower and beam loss cannot update the role tower. In this architecture, coefficient `0.1` still changed role behavior through the observation trajectory and reduced discovery (`52.56%`) relative to coefficient `0` (`53.89%`). The auxiliary feedback is not part of the current main configuration.

## Three-Seed 2x2 Gate

The selected short-gate configuration is:

- Decoupled local role tower.
- Recurrent convolutional score-residual beam tower.
- MPNN centralized critic, per-agent finite-horizon MC returns.
- Separate mode/beam PPO losses.
- Role probability floor `0.05`.
- Beam ISAC feedback coefficient `0`.
- `20 x 300 = 6,000` environment steps per training seed.

Execution arms:

- A: score-proportional beam, uniform TX/RX.
- B: learned beam, uniform TX/RX.
- C: score-proportional beam, learned role.
- D: learned beam, learned role.

| Training seed | A | B | C | D | D-B | Interaction |
|---:|---:|---:|---:|---:|---:|---:|
| 29260711 | 52.11% | 50.22% | 51.78% | 53.89% | +3.67 pp | +4.00 pp |
| 29261711 | 52.11% | 52.44% | 50.67% | 52.44% | +0.00 pp | +1.44 pp |
| 29262711 | 52.11% | 50.89% | 54.11% | 50.56% | -0.33 pp | -2.33 pp |
| Mean | 52.11% | 51.19% | 52.19% | 52.30% | +1.11 pp | +1.04 pp |

The apparent single-seed interaction did not replicate. `D-B` was positive in only one of three seeds, and the method did not pass the `+3 pp` gate.

## Decision

The actor receives only local observations, joint recurrent replay error is zero, role/beam gradients are structurally isolated, and no global action teacher is enabled. The remaining uncertainty is optimization duration: 6k steps provide only 20 policy updates. The next and final architecture check is a fresh `30,000`-step three-seed run with no model or reward changes. Failure at 30k terminates this joint architecture; success is required before any 100k run.

## Result Locations

- Corrected reward screen: `05_simulation/results_raw/handshake_reward_v2_screen_seed29260711_20260711`
- Shared factorized screen: `05_simulation/results_raw/factorized_isac_credit_screen_seed29260711_20260711`
- Decoupled screen: `05_simulation/results_raw/decoupled_factorized_screen_seed29260711_20260711`
- Decoupled three-seed gate: `05_simulation/results_raw/decoupled_joint_6k_three_seed_gate_20260711`
