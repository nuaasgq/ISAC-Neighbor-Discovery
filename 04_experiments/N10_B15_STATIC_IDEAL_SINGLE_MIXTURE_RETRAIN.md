# N=10, B=15 Static Ideal-ISAC Single-Mixture Retrain

## Purpose

This campaign replaces the development results produced before the recurrent
beam-distribution audit. It retrains Direct ISAC MAPPO and Residual-mask MAPPO
under one corrected stochastic-action contract before making a causal comparison.

## Implementation audit

The recurrent policy previously applied `beam_uniform_mixture=0.1` twice:

1. once in inherited stochastic-support regularization; and
2. once when constructing the recurrent categorical beam distribution.

The first pass also converted masked zero probabilities to finite log probabilities.
The second pass therefore assigned probability to masked-out beams. The effective
unmasked exploration mixture was approximately `1 - (1 - 0.1)^2 = 0.19`, and a
candidate mask was not strict.

The corrected implementation applies the role floor in recurrent support
regularization and applies the beam mixture exactly once in the recurrent beam
distribution. Candidate-mask support remains zero outside the local feasible set.

## Invalidated development runs

The following interrupted runs are retained only as audit artifacts and must not
enter paper tables:

- `n10_b15_static_ideal_independent_role_formal`: stopped at 330/1000 episodes.
- `n10_b15_static_ideal_residual_mask_formal_3seed`: stopped at 40/1000 episodes
  for the first seed.

Earlier recurrent MAPPO results are labeled pre-fix development evidence. Wang
and external candidate-random protocol results do not traverse the recurrent beam
distribution and may be reused on the same paired scenarios.

## Material Passport

- Environment: static planar N=10, 24 azimuth beams (15 degrees), one RF chain.
- Horizon: 300 slots per episode.
- PHY: ideal communication and ideal anonymous count-ISAC.
- Protocol state: local sensing history and exchanged neighbor tables.
- Actor information: decentralized local observations only.
- Critic information: centralized training information allowed by CTDE.
- Methods: Direct ISAC MAPPO and Residual-mask MAPPO.
- Residual-mask difference: local residual-table candidate support plus a hard
  actor action mask; no candidate scores, rule residuals, expert imitation,
  rendezvous guidance, or action executor override.
- Training seeds: 59260713, 59261722, 59262731.
- Training budget: 1000 episodes x 300 slots per method and seed.
- Evaluation: stochastic policy, 50 held-out paired episodes per seed, seed offset
  2,000,000, 300 slots.
- Primary metrics: final discovery rate and normalized discovery-curve AUC.
- Mechanism diagnostics: candidate-set size, true-undiscovered beam retention,
  empty-beam exclusion, selected-undiscovered-beam rate, and mask compliance.

## Promotion boundary

Residual-mask MAPPO is promoted only if the three-seed paired evaluation shows:

1. higher discovery-curve AUC than corrected Direct ISAC MAPPO;
2. final discovery rate no more than 2 percentage points below candidate-random;
3. strict candidate-mask compliance in every evaluated slot; and
4. no single training seed reverses the primary AUC conclusion by a material margin.

One-seed pilots and scenario-only bootstrap intervals are not publication-level
robustness evidence.
