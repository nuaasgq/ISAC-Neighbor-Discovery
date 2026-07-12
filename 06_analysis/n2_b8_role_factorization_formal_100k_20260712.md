# N=2 Role-Factorization Gate: Formal 100k-Step Result

## Material Passport

- Artifact type: experiment validation report
- Verification status: ANALYZED
- Raw run: `05_simulation/results_raw/n2_b8_role_factorization_formal_100k_20260712`
- Analysis bundle: `06_analysis/tables/n2_b8_role_factorization_formal_100k_20260712`
- Analysis date: 2026-07-12
- Inference unit: independently trained policy seed

## Question

Can a distributed recurrent MAPPO policy learn complementary TX/RX roles from local observations when beam selection is held to the same uniformly random process for every method?

This is a causal role-learning gate, not an end-to-end ISAC neighbor-discovery benchmark. It deliberately removes beam-policy differences before testing the role-network contribution.

## Controlled Design

- Scenario: static two-UAV diagnostic, planar 8-beam codebook, 16 slots per episode. Node geometry varies between training episodes and is held out for evaluation.
- Training: 6,250 episodes, 100,000 environment steps per method and seed.
- Replication: three paired training seeds, for nine runs and 900,000 total environment steps.
- Evaluation: 200 matched held-out stochastic-policy scenarios per trained policy. Deterministic evaluation is retained only as a diagnostic because a fixed deterministic beam cannot realize the uniformly randomized beam contract.
- Actor information: local distributed observations only; no centralized actor input, ISAC actor feature, candidate score, rendezvous beam, sensing action, or rule-based role recommendation.
- Critic: centralized training information is allowed under the CTDE contract.
- Beam control: the executed beam distribution is exactly uniform for all three methods (`beam_uniform_mixture=1.0`).
- Learned control: TX/RX role only.

The compared role heads are:

1. Independent role: role logits do not depend on the selected beam direction.
2. Beam-conditioned role: an unconstrained MLP conditions the role on the selected beam.
3. Antisymmetric conditioned role: the two role logits are constrained to opposite signs along a shared learned global-direction axis. The axis is learned; no side is hard-coded as transmitter.

## Coordination Funnel

Let `A` be active undiscovered pair-slots, `B` bilateral beam-alignment slots, `O` aligned slots with complementary TX/RX roles, and `S` successful handshakes. The diagnostic rates are `B/A`, `O/B`, and `S/O`.

| Method | Discovery rate | `B/A` | `O/B` | `S/O` | Mean TX ratio |
|---|---:|---:|---:|---:|---:|
| Independent | 8.83% | 1.706% | 34.30% | 100% | 43.22% |
| Beam-conditioned | 11.33% | 1.661% | 48.74% | 100%* | 32.61% |
| Antisymmetric | 17.67% | 1.593% | 76.25% | 100% | 50.59% |

`*` The conversion rate is defined in only two ordinary-conditioned seeds because one seed produced no aligned complementary-role opportunity.

The pooled event counts give the same mechanism-level picture:

| Method | `A` | `B` | `O` | `S` | Pooled `O/B` |
|---|---:|---:|---:|---:|---:|
| Independent | 9,147 | 156 | 53 | 53 | 33.97% |
| Beam-conditioned | 9,068 | 151 | 68 | 68 | 45.03% |
| Antisymmetric | 8,733 | 139 | 106 | 106 | 76.26% |

All beam-alignment rates remain close to the random-codebook reference `1/8^2 = 1.5625%`. Therefore, the discovery improvement is not caused by privileged beam selection. Under the ideal-PHY gate, every aligned complementary opportunity succeeds, so the measurable gain occurs at the role-coordination stage.

## Training-Seed Results

| Method | Seed 1 | Seed 2 | Seed 3 | Mean | Seed-level 95% t interval |
|---|---:|---:|---:|---:|---:|
| Independent | 6.0% | 9.5% | 11.0% | 8.83% | [2.46%, 15.21%] |
| Beam-conditioned | 0.0% | 20.5% | 13.5% | 11.33% | [-14.55%, 37.22%] |
| Antisymmetric | 19.0% | 17.0% | 17.0% | 17.67% | [14.80%, 20.54%] |

The ordinary conditioned head is seed-unstable: one run collapses to all-RX behavior (`TX ratio = 0`), while another reaches 20.5% discovery. This is consistent with uncontrolled spontaneous symmetry breaking, not a reliable method gain.

The antisymmetric head improves discovery over the paired independent policies in all three seeds by 13.0, 7.5, and 6.0 percentage points. The mean paired gain is 8.83 points; its three-seed t interval is [-0.32, 17.99] points, so this small gate is not sufficient for a definitive superiority claim. The mechanism-level `O/B` gain is 41.95 points with paired t interval [11.07, 72.82] points and is positive in all seeds.

## Interpretation

The gate supports one specific method claim: encoding reciprocal directional antisymmetry resolves a genuine decentralized TX/RX coordination ambiguity more reliably than an unconstrained role head. It does not yet support the complete paper claim because beam learning, ISAC sensing, table exchange, realistic PHY, larger networks, and mobility are absent here.

The next experiment should preserve this validated role head and restore beam learning in stages:

1. N=2, B=8 with locally observable ISAC beam evidence, while keeping the same role-factorization ablation.
2. Verify that beam-selection gain appears in `B/A` and role gain remains in `O/B`.
3. Only after both stages pass, move to N=10 and compare random, Wang-aligned protocol, non-ISAC MARL, and ISAC-MARL under one shared execution contract.

## Statistical Boundaries

- The three training seeds are the inference units; evaluation episodes are not treated as independent policy replications.
- Seed-level t intervals are very wide at `n=3`; bootstrap intervals over three seeds are descriptive only.
- No multiple-hypothesis correction was applied because this was a predeclared mechanism gate with discovery and funnel diagnostics.
- The experiment isolates causal structure but uses an ideal PHY and cannot establish TWC-level external validity.
