# Recurrent MAPPO Development-Gate Audit (2026-07-11)

## Scope

- Training: `N=10`, planar `B=15 deg` (`24` beams), `300` slots per episode.
- Execution: decentralized local observations, one RF chain, fixed independent TX/RX probability `0.5`.
- Training-only critic: centralized two-pass MPNN.
- Protocol: `improved_rl_isac_tables`, noisy-count piggyback ISAC, neighbor/sensing table exchange.
- Primary learned-policy executor: direct categorical sampling (`pure_learned_stochastic`).
- All results below are development results. The frozen final holdout was not used.

## No-Prior 30k Gate

The recurrent actor plus MPNN critic was trained for `100 x 300 = 30,000` environment steps with three seeds.

| Training seed | Recurrent MAPPO | Feedforward MAPPO | Random candidate | Score proportional |
|---:|---:|---:|---:|---:|
| 29260711 | 52.89% | 49.22% | 51.11% | 53.00% |
| 29261711 | 53.44% | 52.11% | 51.11% | 53.00% |
| 29262711 | 51.89% | 53.56% | 51.11% | 53.00% |
| Mean | 52.74% | 51.63% | 51.11% | 53.00% |

The method did not pass the predeclared `+3 percentage-point` gate. It was `-0.26 pp` below score proportional and `+1.63 pp` above random on average.

## Unbounded Score-Residual 6k Gate

The actor was initialized exactly from the local score-proportional distribution and learned an unbounded recurrent residual. Three `20 x 300 = 6,000`-step runs produced learned discovery rates of `51.22%`, `55.78%`, and `52.44%` (mean `53.15%`). The matched score-proportional control was `53.00%`.

The residual parameters moved away from zero, but the learned policy improved by only `+0.15 pp` on average. This is evidence of parameter learning, not a reliable performance gain.

## Local-Memory Diagnostic

A development diagnostic initially appeared to show that score power `0.5` reached `54.89%` versus `53.00%` for score proportional. This result was invalid for policy-difference attribution: the diagnostic consumed an extra random number even when beam persistence was disabled, so its action sampling stream was not paired with the learned stochastic executor.

Beam persistence (`0.25`, `0.5`, and `0.75`) also reduced discovery in that exploratory run. There is no evidence that simple previous-beam persistence is a useful mechanism in the current environment.

## Bounded Calibrated Residual and CRN Correction

The bounded residual actor used

```text
beam logits = tau * log(local candidate-score probability)
            + alpha * tanh(recurrent residual),
```

with `tau=0.5` initially and a trainable residual bound initialized to `alpha=0.1`. The residual gate remained near `0.10-0.107` after 6k steps and the learned score powers remained near `0.5`.

The evaluator was then corrected so that the learned stochastic policy and the zero-persistence score policy consume a common action-randomness stream. A unit test now verifies identical actions when their probability distributions are identical.

| Training seed | Bounded learned policy | Score power 0.5 | Learned minus rule |
|---:|---:|---:|---:|
| 29260711 | 52.22% | 50.67% | +1.56 pp |
| 29261711 | 50.11% | 50.67% | -0.56 pp |
| 29262711 | 51.44% | 50.67% | +0.78 pp |
| Mean | 51.26% | 50.67% | +0.59 pp |

The corrected score-power result also shows that the earlier apparent `54.89%` advantage was a sampling-stream artifact. The bounded method remains below the original score-proportional control (`53.00%`) and does not pass the gate.

## Frozen-Advantage PPO Correction

An implementation audit found that the critic was updated inside each PPO reuse epoch and then used to recompute the advantage for the next epoch. This violates the fixed behavior-rollout surrogate contract. The implementation was corrected to snapshot values and normalized advantages once before the PPO epoch loop.

Using the otherwise unchanged unbounded score-residual actor, the corrected three-seed 6k gate was:

| Training seed | Frozen-advantage learned policy | Score proportional | Learned minus rule |
|---:|---:|---:|---:|
| 29260711 | 55.11% | 53.00% | +2.11 pp |
| 29261711 | 52.89% | 53.00% | -0.11 pp |
| 29262711 | 53.11% | 53.00% | +0.11 pp |
| Mean | 53.70% | 53.00% | +0.70 pp |

The correction improves the mean relative to the dynamic-advantage run, but it still fails the `+3 pp` gate and is not a sufficient paper-level RL contribution.

## GAE and Local Potential Shaping Screens

Fixed-rollout finite-horizon GAE was implemented and verified against Monte Carlo returns at `lambda=1`. On training seed `29260711`, the 20-scenario results were:

| Advantage estimator | Learned policy | Score proportional | Learned minus rule |
|---|---:|---:|---:|
| Frozen MC | 55.11% | 53.00% | +2.11 pp |
| GAE, lambda=0.95 | 53.33% | 53.00% | +0.33 pp |
| GAE, lambda=0.99 | 52.44% | 53.00% | -0.56 pp |

GAE did not improve the short gate and was not expanded to three seeds.

A potential-based ISAC shaping term was then tested. Its potential uses only the decentralized actor's candidate-mask size and candidate-score entropy. The terminal potential is zero, and a unit test verifies that the discounted shaping return telescopes to an action-independent initial-potential constant.

| Shaping coefficient | Learned policy | Score proportional | Learned minus rule |
|---:|---:|---:|---:|
| 0.05 | 53.78% | 53.00% | +0.78 pp |
| 0.10 | 48.11% | 53.00% | -4.89 pp |
| 0.20 | 50.33% | 53.00% | -2.67 pp |

No shaping coefficient improved on the unshaped frozen-MC run for the same seed. The shaping hypothesis was therefore rejected without a multi-seed expansion.

## Current Conclusion

1. Recurrent replay is numerically consistent (`max absolute log-probability replay error = 0`).
2. The actor receives nonzero updates, but neither an unconstrained nor a bounded local-score residual yields a robust gain.
3. Simple persistence and score-temperature tuning are not supported as innovations by the corrected evidence.
4. The next experiment must isolate MAPPO training correctness and variance control. Candidate rules, reward, actor observations, PHY, and protocol should remain fixed.
5. A 100k run is not justified until a three-seed short gate exceeds the strongest matched local rule by at least `3 pp`.

## Result Locations

- No-prior 30k: `05_simulation/results_raw/recurrent_mpnn_30k_three_seed_gate_20260711`
- Unbounded score residual 6k: `05_simulation/results_raw/score_residual_6k_three_seed_screen_20260711`
- Local diagnostic: `05_simulation/results_raw/local_memory_diagnostics_dev20_20260711`
- Bounded score residual 6k: `05_simulation/results_raw/bounded_score_residual_6k_three_seed_gate_20260711`
- Frozen-advantage score residual 6k: `05_simulation/results_raw/frozen_advantage_v2_6k_three_seed_gate_20260711`
- GAE screen: `05_simulation/results_raw/gae_lambda_screen_seed29260711_20260711`
- Local potential screen: `05_simulation/results_raw/local_potential_coef_screen_seed29260711_20260711`
