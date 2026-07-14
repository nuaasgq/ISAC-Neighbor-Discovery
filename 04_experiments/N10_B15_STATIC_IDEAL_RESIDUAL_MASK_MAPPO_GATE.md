# N=10 Static Ideal-ISAC Residual-Mask MAPPO Gate

## Material Passport

- Artifact type: experiment contract
- Created: 2026-07-14
- Verification status: IN_PROGRESS
- Configuration: `05_simulation/configs/n10_b15_static_ideal_isac.yaml`
- Launcher: `05_simulation/scripts/run_n10_b15_static_ideal_residual_mask_mappo.py`

## Mechanism Boundary

Residual-mask MAPPO uses the same recurrent MAPPO actor, antisymmetric beam-conditioned role
head, clean CTDE critic, reward, optimizer, and direct local measurement features as the formal
Direct-ISAC MAPPO arm. It changes only the following action-support mechanism:

```text
candidate_source: default -> residual_table
candidate_mask: false -> true
```

For node `i` and beam `b`, the local residual opportunity is the latest anonymous count-ISAC
estimate minus node `i`'s confirmed interactions in that beam. Locally known empty or exhausted
beams are removed from the feasible action set. Unknown beams remain feasible, and an empty set
reopens locally selected stale beams. Post-handshake neighbor and sensing table exchange remains
enabled.

The mask uses decentralized actor-visible state only. It does not expose true neighbor identity,
true position, true adjacency, or another node's current action before handshake. It is a rule-based
feasible-set constraint and must be reported as such, not as a learned contribution.

The following guidance remains disabled:

- candidate-score prior;
- bounded score residual or rule-logit residual;
- expert behavior cloning;
- rendezvous observation or pair-derived action recommendation;
- auxiliary measurement/ranking loss;
- global information in the actor.

## Gate Design

1. Run a two-episode smoke test to verify masked sampling, PPO replay log-probability consistency,
   checkpoint writing, and held-out evaluation.
2. Train one 100-episode pilot on seed `59262731`, the weakest formal Direct-ISAC seed.
3. Evaluate the pilot on the same 50 held-out scenarios `61262731--61262780` used by Direct-ISAC
   MAPPO, Wang2025, and residual candidate random.
4. Record one censored row per true edge and report discovery at 50/100/150/200/300 slots, curve
   AUC, censored mean delay, and final discovery rate.

The residual candidate-random arm is the attribution control for the mask. A high final discovery
rate alone does not establish an RL contribution. Promotion to a 1,000-episode, three-seed run
requires a repeatable improvement over residual candidate random in early discovery or curve AUC
without materially reducing its final coverage. Direct-ISAC MAPPO remains the no-hard-mask neural
comparator.

## Pilot Command

```powershell
python 05_simulation/scripts/run_n10_b15_static_ideal_residual_mask_mappo.py `
  --profile pilot `
  --run-root 05_simulation/results_raw/n10_b15_static_ideal_residual_mask_pilot `
  --seeds 59262731 --torch-threads 1
```
