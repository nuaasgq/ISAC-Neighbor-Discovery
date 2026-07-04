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
  - Beam tokens are built from belief, age, success, and fail features.

- `05_simulation/run_actor_critic_imitation_probe.py`
  - Provides behavior cloning from the rule expert and optional actor-critic fine-tuning.
  - Existing status notes show that the neural path becomes nonzero only when the MARL env exposes ISAC piggyback belief updates.

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

3. **Masked beam head**
   - Compute logits for all beams, then set non-candidate logits to a large negative value before sampling.
   - Keep a separate exploration probability that samples outside the mask.
   - Report the mask size as a protocol overhead metric.

4. **Rule-residual logits**
   - Add learned logits to normalized rule scores:
     \[
       \ell_i(b)=\ell_i^{rule}(b)+\ell_i^{nn}(b)
     \]
   - This keeps the neural method close to the mechanism that already works while allowing data-driven corrections.

5. **Topology-deficit conditioning**
   - Include discovered degree, target-degree deficit, component-local summaries, and recent discovery count in the context encoder.
   - Do not claim global lambda2 optimization unless a centralized critic or graph estimator is actually trained.

6. **Centralized critic**
   - During training only, consume `training_state()`:
     - discovered adjacency,
     - true communication adjacency,
     - global empty/collision counts,
     - lambda2 of discovered graph.
   - Actor remains decentralized and uses only local observations.

## Minimal 8-Hour Experiment

Target: produce a method-probe result, not a main manuscript result.

1. Add candidate mask generation to `MarlNeighborDiscoveryEnv._observation_for()` or a helper:
   - `candidate_mask`
   - `candidate_rank`
   - `candidate_count`

2. Add optional masking to `SharedBeamActorCritic.act()`:
   - `mask_mode="none|topk|threshold|hybrid"`
   - `top_k=8` for B=10/N=10 probe.

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
  --seed 20260705
```

4. Compare against:
   - rule expert,
   - current flat shared actor-critic,
   - random,
   - enhanced no-ISAC.

5. Report only if:
   - autonomous student discovery is nonzero in all held-out seeds,
   - empty-scan ratio is below flat actor-critic,
   - behavior remains nonzero under at least one mobility change,
   - deterministic and stochastic evaluation are both reported.

## Promotion Gate for Main Paper

The neural MARL method can be promoted from "method probe" to "main method" only after:

- N=10 training is repeated with at least 5 seeds.
- N=100 transfer is evaluated without fine-tuning at B=10 and B=15.
- It beats enhanced no-ISAC and flat neural actor-critic.
- It is competitive with the current rule-tuned ISAC protocol or clearly improves collision-penalized discovery.
- It preserves the information boundary: no undiscovered-neighbor pose/identity at execution.

Until then, the current manuscript should keep learning as shared-parameter protocol tuning and present neural MARL as future work or a separate extension.
