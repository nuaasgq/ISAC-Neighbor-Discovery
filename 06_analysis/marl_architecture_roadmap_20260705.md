# Candidate-Constrained MARL Roadmap - 2026-07-05

## Current Code Foundation

The repository already has the minimum scaffolding for a neural MARL extension:

- `05_simulation/src/isac_nd_sim/marl_env.py`
  - `MarlNeighborDiscoveryEnv` exposes decentralized per-agent observations.
  - Public actor observations include self pose/velocity/attitude, beam belief, beam age, beam success/fail counts, last mode/beam, and local discovered-neighbor summary.
  - `training_state()` exposes centralized truth only for training/critic use.
  - Default protocol is `isac_structured_marl`, so piggyback ISAC belief updates are available.

- `05_simulation/src/isac_nd_sim/neural_shared_actor_critic.py`
  - `SharedBeamActorCritic` already implements a shared decentralized actor with a mode head and flat beam head.
  - Beam tokens are built from belief, age, success, fail, and optional candidate-score features.
  - After the 2026-07-05 morning update, the actor can optionally apply a local candidate mask to beam logits via `use_candidate_mask=True`.
  - The actor also supports optional local candidate scores, topology-deficit conditioning, and rule-residual logits via `use_candidate_score=True`, `use_topology_deficit=True`, and `use_rule_residual=True`.

- `05_simulation/run_actor_critic_imitation_probe.py`
  - Provides behavior cloning from the rule expert and optional actor-critic fine-tuning.
  - Existing status notes show that the neural path becomes nonzero only when the MARL env exposes ISAC piggyback belief updates.
  - The 2026-07-05 update also supports `--eval-both` and `--env-protocol`, so deterministic/stochastic evaluation and clean no-ISAC environment baselines can be reported from the same runner.

## Why the Flat Beam Head Is Weak

The current neural probe uses a categorical distribution over all 3-D beam cells.
That is a poor fit for narrow-beam discovery because:

- Most beam cells are empty, especially at 10 degrees and below.
- The useful action support is sparse and changes after each ISAC observation.
- Deterministic evaluation can collapse to a single mode or beam.
- Exploration over all beams wastes almost all probability mass in high-dimensional codebooks.
- The rule protocol's key advantage is not a generic neural policy; it is candidate-set refinement after sensing.

## Proposed Neural Mechanism

The next neural method should be an ISAC-candidate-constrained shared actor-critic:

1. **Mode head**
   - Predicts `sense`, `tx`, `rx`, or `idle`.
   - Uses local summary and short history.
   - Add an entropy floor or coordination prior to avoid all-TX/all-RX collapse.

2. **Candidate beam proposal**
   - Build a per-agent candidate mask from:
     - top-k beam beliefs,
     - beams above an occupancy threshold,
     - recent beam-lock / near-miss beams,
     - a small exploration subset.
   - Enforce at least one random exploration beam so sensing errors cannot permanently prune cells.
   - Current implementation status: `MarlNeighborDiscoveryEnv` now exposes local `candidate_mask` and `candidate_score` observations derived only from belief, age, success/fail memory, recency, and last beam; it does not use undiscovered-neighbor truth.

3. **Masked beam head**
   - Compute logits for all beams, then set non-candidate logits to a large negative value before sampling.
   - Keep a separate exploration probability that samples outside the mask.
   - Report the mask size as a protocol overhead metric.
   - Current implementation status: `SharedBeamActorCritic(..., use_candidate_mask=True)` applies this mask during action sampling; default remains `False` to preserve existing probes.

4. **Rule-residual logits**
   - Add learned logits to normalized rule scores:
     \[
       \ell_i(b)=\ell_i^{rule}(b)+\ell_i^{nn}(b)
     \]
   - This keeps the neural method close to the mechanism that already works while allowing data-driven corrections.
   - Current implementation status: `rule_mode_logits` are exposed by the environment and can be added to the neural logits with `use_rule_residual=True`.

5. **Topology-deficit conditioning**
   - Include discovered degree, target-degree deficit, component-local summaries, and recent discovery count in the context encoder.
   - Do not claim global lambda2 optimization unless a centralized critic or graph estimator is actually trained.
   - Current implementation status: a scalar local `topology_deficit` is exposed and can be included in the actor context with `use_topology_deficit=True`.

6. **Centralized critic**
   - During training only, consume `training_state()`:
     - discovered adjacency,
     - true communication adjacency,
     - global empty/collision counts,
     - lambda2 of discovered graph.
   - Actor remains decentralized and uses only local observations.

## Completed 8-Hour Probe

Target: produce a method-probe result, not a main manuscript result. This target has been reached.

1. Add candidate mask generation to `MarlNeighborDiscoveryEnv._observation_for()` or a helper:
   - `candidate_mask`
   - `candidate_score`
   - `topology_deficit`
   - `rule_mode_logits`
   - Status: implemented and covered by MARL contract tests. `candidate_rank` and `candidate_count` remain optional extensions.

2. Add optional masking to `SharedBeamActorCritic.act()`:
   - `mask_mode="none|topk|threshold|hybrid"`
   - `top_k=8` for B=10/N=10 probe.
   - Status: optional mask path, candidate score, topology-deficit context, and rule-residual path are implemented and covered by tests.

3. Re-run behavior cloning:

```powershell
python 05_simulation/run_actor_critic_imitation_probe.py `
  --config 05_simulation/configs/mvp.yaml `
  --output 05_simulation/results_raw/candidate_mask_actor_critic_probe `
  --bc-episodes 80 `
  --rl-episodes 0 `
  --eval-episodes 10 `
  --stochastic-eval `
  --slots 80 `
  --node-count 10 `
  --azimuth-cells 12 `
  --elevation-cells 6 `
  --communication-range 1200 `
  --sensing-range 1200 `
  --false-alarm-rate 0 `
  --miss-detection-rate 0 `
  --angular-cell-offset-std 0 `
  --sensing-period-slots 1 `
  --hidden-dim 64 `
  --learning-rate 0.001 `
  --candidate-mask `
  --candidate-score `
  --topology-deficit `
  --rule-residual `
  --seed 20260705
```

Smoke status: `05_simulation/results_raw/structured_imitation_smoke_20260705` verified the structured path with 2 BC episodes, 1 RL episode, and 1 stochastic evaluation episode. The held-out stochastic student reached nonzero discovery (`env_discovery_rate=0.3571`) in a very small N=6/B=18/12-slot smoke. This is a wiring and feasibility signal only, not a manuscript result.

4. Compare against:
   - rule expert,
   - current flat shared actor-critic,
   - random,
   - enhanced no-ISAC.

5. Current reportable probe evidence:
   - Core N=10/B=72, 80-slot, 3-seed block reports both stochastic and deterministic evaluation.
   - Full structured residual actor (`rule_residual_scale=1.0`) reduces stochastic empty-scan ratio to 0.1112, versus 0.6901 for the flat stochastic actor.
   - Clean no-ISAC neural baseline (`env_protocol=improved_rl_no_isac`, `expert_protocol=improved_rl_no_isac`) collapses to deterministic discovery 0 and stochastic discovery 0.0044.
   - Residual-scale sweep finds `rule_residual_scale=0.25` as the best balanced tested value, with stochastic discovery 0.5978 and deterministic discovery 0.0837.
   - The best structured stochastic actor still remains below the flat stochastic actor's discovery rate of 0.6322, so this is evidence for an architectural direction rather than proof of a superior learned policy.

## Open Neural Design Problems

The current actor constrains the beam search effectively, but it does not yet solve distributed handshake coordination.
The next neural iteration should therefore focus on:

- TX/RX role balance: add explicit coordination regularization or a local role-prior residual so deterministic policies do not converge to one-sided transmission/reception.
- Collision-aware value shaping: penalize repeated many-to-one contention and reward unique reciprocal alignments, not just discovered-edge count.
- Candidate-mask exploration: keep a small probability of out-of-mask beams to recover from sensing errors while preventing the flat action space from dominating.
- Transfer-normalized inputs: use beam-index-relative features and mask-size-normalized scores so N=10/B=72 training can better transfer to B=10/B=15-equivalent codebooks and N=30/50/100 tests.
- Deterministic policy calibration: select deployment actions through a temperature/epsilon schedule or top-k randomized serving policy if pure argmax remains brittle.

## Promotion Gate for Main Paper

The neural MARL method can be promoted from "method probe" to "main method" only after:

- N=10 training is repeated with at least 5 seeds.
- N=100 transfer is evaluated without fine-tuning at B=10 and B=15.
- It beats enhanced no-ISAC and flat neural actor-critic.
- It is competitive with the current rule-tuned ISAC protocol or clearly improves collision-penalized discovery.
- It preserves the information boundary: no undiscovered-neighbor pose/identity at execution.

Until then, the current manuscript should keep learning as shared-parameter protocol tuning and present neural MARL as future work or a separate extension.

## Manuscript Framing

Use the MARL result as a supplementary cross-layer intelligence probe:

- It validates that the ISAC-derived candidate set can be exposed as a decentralized learning interface.
- It shows that the same local information boundary can drive both rule-based and neural policies.
- It supports the claim that the hard part is not only learning a better beam distribution, but using sensing to collapse the action support before learning.
- It should not be written as "MARL beats all baselines"; the current data do not support that claim.
