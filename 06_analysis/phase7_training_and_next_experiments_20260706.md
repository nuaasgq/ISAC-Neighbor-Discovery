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

## Immediate Priority

1. Complete and aggregate Phase-6 `B=5`.
2. Run Phase-6 `B=3` with the same settings after `B=5`.
3. Aggregate `B=3/5/10/15/30` into one MARL transfer table and figure set.
4. Build Phase-7 multi-seed learning-curve figures using `training_step` as the x-axis.
5. Update the IEEE draft so all text, tables, and figure paths use the 10-episode Phase-6 data.

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
4. improved RL no-ISAC actor/reward counterpart
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

## Claim Discipline

- Do not claim complete `3-30 deg` support until `B=3` and `B=5` complete.
- Do not claim universal raw-discovery dominance for `contention_actor`; current evidence supports collision reduction and CPD gain.
- Do not claim strict SkyOrbs reproduction unless the original protocol is faithfully reimplemented.
- Do not claim physical-layer ISAC waveform validation; the current abstraction is a protocol-level occupancy/candidate prior.
- Do not present `Rs=Rc` as physically necessary; treat it as an initial controlled single-hop setting until range sweeps and theory are added.
