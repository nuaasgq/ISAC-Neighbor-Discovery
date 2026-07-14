# N=10 Static Ideal-ISAC Paired Evaluation and Role-Head Ablation

## Material Passport

- Artifact type: experiment contract
- Created: 2026-07-14
- Verification status: PARTIALLY_ANALYZED
- Paired evaluation: completed (24/24 method-seed combinations; 1,200 episodes)
- Paired analysis: `06_analysis/n10_b15_static_ideal_paired_eval_20260714`
- Independent-role ablation: running (seed 59262731)
- Configuration: `05_simulation/configs/n10_b15_static_ideal_isac.yaml`
- Training source: `05_simulation/results_raw/n10_b15_static_ideal_mappo_formal_3seed`

## Paired Evaluation Contract

All methods use the same three held-out scenario blocks. Each block contains 50 episodes and
each episode contains 300 slots.

| Training seed | Held-out scenario seeds |
|---:|---:|
| 59260713 | 61260713--61260762 |
| 59261722 | 61261722--61261771 |
| 59262731 | 61262731--61262780 |

The evaluated methods are:

- uniform random TX/RX and uniform random beam selection without ISAC guidance;
- Wang2025 ISAC sensing-table beam pool and table exchange;
- random TX/RX with random selection from the local residual-table ISAC candidate pool;
- no-ISAC MAPPO;
- direct-ISAC MAPPO;
- direct-ISAC MAPPO with measurement-prediction auxiliary loss;
- random TX/RX with the direct-ISAC MAPPO beam policy;
- direct-ISAC MAPPO role policy with a uniformly random beam.

The ISAC-candidate random arm loads zero neural weights and replaces both role and beam actions.
Its only action guidance is the local residual-table candidate pool; it retains the same
post-handshake table-exchange environment as the direct-ISAC MAPPO arm.

For the learned-role/random-beam intervention, the random beam is selected before the role.
The antisymmetric role head is evaluated conditional on that executed beam. This avoids the
invalid alternative in which the role is conditioned on one neural beam and a different beam is
then executed.

Every evaluation records one censored row per true edge. Discovery curves use an end-of-slot
convention: an edge discovered at zero-based slot `s` contributes from elapsed slot `s+1`.

## Primary Metrics

- edge discovery rate at 50, 100, 150, 200, and 300 slots;
- normalized area under the cumulative discovery curve;
- mean censored edge discovery delay;
- time to 50%, 80%, and 90% edge discovery, censored at 300 slots;
- final discovery rate and networking-completion slot.

The primary report uses three-seed means and seed-level standard deviations. Hierarchical
bootstrap intervals resample training seeds and then paired scenarios within each seed. Because
only three independently trained seeds are available, these intervals are descriptive and must
not be presented as strong asymptotic significance evidence.

## Role-Head Training Ablation

The independent-role ablation changes only:

```text
role_factorization: beam_conditioned_antisymmetric -> independent
```

It retains the decoupled role tower, recurrent beam policy, direct local ISAC observations,
clean CTDE critic, reward, exploration, optimizer, 1000 episodes, 300 slots per episode, and
all disabled-rule settings. The first formal run uses seed `59262731`, which is the weakest
direct-ISAC seed. The existing antisymmetric checkpoint on this seed is the paired comparator.

## Commands

Paired evaluation:

```powershell
python 05_simulation/scripts/run_n10_b15_static_ideal_paired_eval.py `
  --profile formal `
  --run-root 05_simulation/results_raw/n10_b15_static_ideal_paired_eval_3seed `
  --max-parallel 2 --torch-threads 1
```

Independent-role formal run:

```powershell
python 05_simulation/scripts/run_n10_b15_static_ideal_role_head_ablation.py `
  --profile formal `
  --run-root 05_simulation/results_raw/n10_b15_static_ideal_independent_role_formal `
  --seeds 59262731 --torch-threads 1
```
