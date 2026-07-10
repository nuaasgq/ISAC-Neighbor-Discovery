# Common-measurement and residual-v2 screen

## Scope

- `N=10`, planar Gauss-Markov mobility, 24 azimuth beams (15 degrees), one RF chain.
- 300 slots at 5 ms/slot; all nodes are within communication and sensing range.
- Primary sensing mode: per-target `noisy_count`.
- Results use 10 held-out paired scenarios and one training seed. They are diagnostic, not final paper evidence.

## Fairness corrections completed

- Wang and MARL now consume the same anonymous per-target measurement object.
- Wang no longer obtains exact true target count after one positive detection.
- Multi-target reports preserve opaque detection IDs and original timestamps, and repeated forwarding is deduplicated.
- Neighbor-table entries are exchanged as IDs and positions but remain separate from direct edges.
- Direct discovery, indirect knowledge, sensing recall, count MAE, and position RMSE are reported separately.
- An unsensed Wang beam remains open after a passive interaction until later sensing resolves its target count.

## Measurement-mode calibration

| Wang sensing mode | Direct discovery | Neighbor knowledge | Empty scan | Count MAE | Per-target recall |
|---|---:|---:|---:|---:|---:|
| ideal count | 0.5489 | 0.8689 | 0.4340 | 0.0000 | 1.0000 |
| noisy count | 0.5200 | 0.8211 | 0.4566 | 0.0830 | 0.9341 |
| binary occupancy | 0.5178 | 0.8356 | 0.4642 | 0.2749 | 0.9348 |

With 10 scenarios, ideal-versus-noisy and noisy-versus-binary discovery differences are not statistically resolved. The modes nevertheless have the expected sensing-information ordering, especially in count MAE.

## Residual mechanism gate

With both beam and TX/RX decisions replaced by local random execution:

| Candidate mechanism | Direct discovery | Empty scan |
|---|---:|---:|
| previous default table | 0.2822 | 0.6892 |
| motion-aged residual table | 0.5267 | 0.4946 |

The paired discovery gain is `+0.2444`, with 95% interval `[0.1641, 0.3248]` and exact sign-flip `p=0.00391`. This passes the mechanism gate.

The residual table uses only local estimated count, count variance, confidence, confirmed interactions, report age, own speed, configured speed bound, beamwidth, and deployment scale. It does not use true adjacency or a recommended action.

## MARL screen

`uniform random` is a blind-search lower bound, Wang is an end-to-end protocol comparator, and `residual candidate random + uniform mode` is the same-capability causal control for MARL. Their gaps therefore have different interpretations.

| Method | Direct discovery | Neighbor knowledge | Empty scan | TX fraction |
|---|---:|---:|---:|---:|
| uniform random | 0.2333 | 0.2333 | 0.7150 | 0.5030 |
| corrected Wang, noisy count | 0.5200 | 0.8211 | 0.4566 | 0.4966 |
| residual candidate random + uniform mode | 0.5778 | 0.8789 | 0.4540 | 0.4977 |
| residual-v2 MARL, unconstrained | 0.3622 | 0.6189 | 0.5248 | 0.2849 |
| residual-v2 MARL, stochastic support | 0.5311 | 0.8589 | 0.4710 | 0.4503 |

The unconstrained policy fails because its held-out mode distribution collapses toward RX. A 0.30 role-probability floor and 0.80 local-candidate uniform mixture restore performance: constrained MARL versus Wang is `+0.0111`, 95% interval `[-0.1132, 0.1354]`, exact `p=0.877`.

However, constrained MARL remains `0.0467` below the same residual mechanism with random beam and uniform mode; its interval `[-0.1279, 0.0345]` crosses zero. Separate beam and mode controls also do not show a positive learned contribution. Therefore the current result supports the sensing/table mechanism but not yet the MARL innovation.

## Decision

Do not start multi-seed long training, 3D, or scale transfer yet. The next method task is to change the learned objective/parameterization so that the policy improves over residual-candidate random execution. Longer training with the current PPO factorization is not justified by this screen.
