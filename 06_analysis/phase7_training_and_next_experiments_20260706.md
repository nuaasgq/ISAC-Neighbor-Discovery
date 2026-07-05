# Phase-7 Training Stability and Next Experiment Plan - 2026-07-06

## Fixed Rule

- Training episodes remain fixed at `300 slots/episode`.
- Long horizons are evaluation-only, with `3000 slots/episode` as the main paper setting.
- The main transfer claim is zero-shot: train at `N=10`, `B=10 deg`, then test at larger `N`, different beamwidths, and longer horizons without fine-tuning.

## Active Runs

### Phase-6 B=5 Stress Transfer

- Campaign: `phase6_final_long_eval_b5_10ep_stoch`
- Training checkpoints: fixed 300-slot-trained MARL checkpoints from Phase 1/4/5.
- Test setting: `N=100`, `B=5 deg`, `3000 slots`, `10 episodes`, stochastic decentralized execution.
- Methods: `legacy_shared`, `collision_reward`, `contention_actor`.
- Resource guard: `max_workers=1`, `torch_threads=2`, `max_rss_mb=10000`, `max_system_memory_percent=90`.

### Phase-7 Multi-Seed Training Stability

- Campaign: `phase7_long_training_100ep_3seed`
- Script: `05_simulation/run_marl_training_stability_campaign.py`
- Setting: `N=10`, `B=10 deg`, `100 episodes`, `300 slots/episode`.
- Seeds: `20260731`, `20260732`, `20260733`.
- Methods: `legacy_shared`, `collision_reward`, `contention_actor`.
- Purpose: produce step-indexed reward, return, policy/value loss, entropy, discovery, collision, and topology curves with multiple independent seeds.

### Phase-7 Improved No-ISAC MARL Training

- Campaign: `phase7_contention_no_isac_100ep_3seed`
- Script: `05_simulation/run_marl_training_stability_campaign.py`
- Setting: `N=10`, `B=10 deg`, `100 episodes`, `300 slots/episode`.
- Method: `contention_no_isac`, implemented as `MAPPO + contention_shared + collision_topology` with `--disable-isac-features` and `env_protocol=structured_marl_no_isac`.
- Purpose: provide the true-MARL counterpart for the five-class baseline closure, separating network-structure benefit from ISAC-derived occupancy/candidate priors.
- Strict no-ISAC update: with the current simulator patch,
  `structured_marl_no_isac` no longer lets explicit `sense` actions update
  occupancy belief. Earlier `phase7_contention_no_isac_100ep_3seed`
  curves are useful as a conservative pre-strict reference because they already
  fail despite the weaker information boundary, but paper-grade five-way results
  should use a fresh strict-no-ISAC training/evaluation campaign.

### Protocol Baseline Adapter

- Script: `05_simulation/run_protocol_baseline_eval.py`
- Purpose: emit `uniform_random` and `skyorbs_like_skip_scan` endpoint results in the same `scope=marl_transfer_evaluation` schema used by checkpoint-based MARL evaluation.
- Plotting support: `plot_marl_transfer_results.py` now preserves `method/method_label`; `plot_marl_method_comparison.py` now supports dynamic method sets rather than only the three ISAC-MAPPO variants.

### Five-Way MARL-Compatible Campaign

- Script: `05_simulation/run_marl_fiveway_eval_campaign.py`
- Default comparison: `uniform_random`, `skyorbs_like`, `mappo_no_isac`,
  `contention_no_isac`, and `contention_actor`.
- Default test setting: `N=100`, `B in {10, 15, 30} deg`, `3000 slots`,
  `10 stochastic episodes`.
- The script keeps scenario seeds paired across all five methods for each
  `(N, B, slots)` setting and emits the same `marl_transfer_evaluation` schema
  for protocol baselines and checkpoint-based MARL policies.
- `contention_no_isac` defaults to the first Phase-7 100-episode checkpoint.
  Multi-training-seed closure remains a later robustness step after all Phase-7
  seeds complete.
- Wording boundary: `uniform_random` and `skyorbs_like` are protocol baselines;
  `mappo_no_isac`, `contention_no_isac`, and `contention_actor` are checkpoint
  MARL policies. The legacy `rl_no_isac`, `improved_rl_no_isac`, and
  `improved_rl_isac` names are simulator proxy protocols and must not be called
  MARL checkpoint baselines.
- No-ISAC MARL wording boundary: after the strict simulator patch, no-ISAC
  checkpoints remove ISAC-derived candidate mask/score/rule-residual assistance,
  piggyback sensing, and sensing-based belief updates. The actor still receives
  generic beam-memory tensors for failed/successful communication attempts, so
  use "without ISAC-derived sensing/candidate assistance" rather than "without
  local memory".

## Immediate Priority

1. Complete and aggregate Phase-6 `B=5`.
2. Run Phase-6 `B=3` with the same settings after `B=5`.
3. Aggregate `B=3/5/10/15/30` into one MARL transfer table and figure set.
4. Build Phase-7 multi-seed learning-curve figures using `training_step` as the x-axis.
5. Launch the strict `contention_no_isac` replacement campaign and regenerate
   no-ISAC learning curves.
6. Build the five-class MARL-compatible comparison after strict no-ISAC
   checkpoints are available.
7. Update the IEEE draft so all text, tables, and figure paths use the
   10-episode Phase-6 data and strict five-class closure.

## Learning-Curve Plot Rule

- `06_analysis/scripts/plot_marl_learning_curves.py` must use `training_step` on
  the x-axis for step, episode, evaluation, and resource curves.
- Multi-seed paper figures should be method-level mean curves with a standard
  deviation band, not one color per raw run.
- Raw per-run CSV rows remain exported for auditability.

## Next Matrix After B=3/B=5

### Five-Class Baseline Closure

Use the same large-scale transfer setting:

- `N=100`
- `B in {10, 15, 30} deg`
- `3000 slots`
- `10 stochastic episodes`
- same communication/sensing single-hop control setting unless explicitly swept

Comparison classes:

1. `uniform_random`
2. `SkyOrbs-like` deterministic 3D reference
3. `MAPPO no ISAC`
4. improved RL no-ISAC actor/reward counterpart (`contention_no_isac`)
5. improved ISAC-MARL with contention-aware actor

### Scale Transfer Matrix

- Training: fixed `N=10`, `B=10 deg`, `300 slots/episode`.
- Testing: `N in {10, 20, 50, 100}`, `B in {10, 15, 30}`, `3000 slots`.
- Include both fixed-area and density-preserving area scaling for `N=100`.

### ISAC and Range Robustness

Start with `N=100`, `B in {10, 15}`, `3000 slots`:

- `Rs/Rc` sweep after a short link-budget/radar-range justification.
- `false_alarm_rate`, `miss_detection_rate`, `angular_cell_offset_std`.
- sensing period/staleness sweep.

### TWC/TCOM-Style Coverage Upgrades

Prioritize these after the current B=5/B=3 and three-seed training closures:

- Codebook and candidate-set complexity: map `B in {3, 5, 10, 15, 30} deg`
  to codebook size, then sweep `candidate_topk in {1, 2, 4, 8, 16}` or an
  equivalent quantized beam subset size.
- Multipath or clutter-misleading ISAC priors: add a stress mode where the
  strongest sensing direction is not always the best communication direction,
  and report wrong-prior ratio, discovery, CPD, and a post-discovery link-quality
  proxy.
- Processing delay and sensing staleness: evaluate `processing_delay_ms in
  {0, 1, 3, 5, 10}` and `sensing_age_slots in {0, 1, 2, 5, 10}` under the
  5 ms slot model.
- Mobility stress: sweep speed bands such as `{5, 10, 20, 30} m/s` and include
  crossing or B-spline-like trajectories to stress angular-rate tracking.
- Self-state uncertainty: sweep localization and attitude errors separately
  from ISAC angular-cell errors.
- Baseline expansion for reviewer alignment: keep the five current classes, then
  add exhaustive, hierarchical/iterative, position-aided, EKF-ISAC-prior, and
  oracle upper-bound baselines as supplementary evidence.

## Claim Discipline

- Do not claim complete `3-30 deg` support until `B=3` and `B=5` complete.
- Do not claim universal raw-discovery dominance for `contention_actor`; current evidence supports collision reduction and CPD gain.
- Do not claim strict SkyOrbs reproduction unless the original protocol is faithfully reimplemented.
- Do not claim physical-layer ISAC waveform validation; the current abstraction is a protocol-level occupancy/candidate prior.
- Do not present `Rs=Rc` as physically necessary; treat it as an initial controlled single-hop setting until range sweeps and theory are added.
- Do not describe legacy CEM/proxy protocols as MARL policies; the paper's
  MARL claims must be tied to checkpoint-based MAPPO/actor-critic runs and
  their step-indexed reward/return curves.
