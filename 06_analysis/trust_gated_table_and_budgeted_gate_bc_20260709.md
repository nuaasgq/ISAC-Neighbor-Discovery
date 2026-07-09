# Trust-Gated Tables and Budgeted Expert Gate BC (2026-07-09)

## Purpose

This update strengthens the protocol/MARL mechanism line without changing the physical-layer abstraction or introducing multi-RF assumptions.

Two implementation gaps were closed:

1. `trust_gated_isac_tables` adds a trust gate to our table-exchange protocol. Peer table hints are no longer blindly fused; they are weighted or rejected using local belief, success, failure, collision, and recency evidence.
2. Budgeted and collision-aware rule experts now expose an `access_gate` label for MARL behavior cloning. The BC loss can imitate `backoff`, `normal`, and `aggressive` gate decisions instead of forcing the auxiliary gate target to `normal`.

## Code Changes

- `05_simulation/src/isac_nd_sim/simulator.py`
  - Added `trust_gated_isac_tables` to the ISAC/table protocol families.
  - Added `shared_table_trust_weight(...)`.
  - Extended `boost_shared_beam(...)` with a bounded `strength` parameter.
- `05_simulation/run_marl_training.py`
  - Added `expert_access_gate_for_env(...)`.
  - Expert actions now carry `Action(mode, beam, access_gate)`.
  - `behavior_cloning_loss(...)` now uses the expert gate label.
  - Training manifests now record `expert_bc_weight`, `expert_protocol`, and `expert_gate_imitation`.
- `05_simulation/run_protocol_baseline_eval.py`
  - Added display label for `trust_gated_isac_tables`.
- `05_simulation/run_marl_training_stability_campaign.py`
  - Added `--expert-bc-weights` and `--expert-protocol` for reproducible Budgeted expert BC sweeps.
- `pytest.ini`
  - Constrains pytest collection to `05_simulation/tests` and avoids `tmp/pydeps` permission traps.

## Verification

Targeted tests:

```text
python -m pytest 05_simulation/tests/test_protocol_comparison_contract.py 05_simulation/tests/test_phy_sensing_and_wang2025.py 05_simulation/tests/test_actor_critic_probe.py -q
26 passed
```

Full configured test entry:

```text
python -m pytest -q
52 passed
```

Smoke experiments:

```text
python 05_simulation/run_protocol_baseline_eval.py --protocols improved_rl_isac_tables trust_gated_isac_tables --eval-episodes 1 --slots 30 ...
```

This confirmed that both table protocols write MARL-compatible evaluation artifacts. The one-episode numbers are not used as paper evidence.

```text
python 05_simulation/run_marl_training.py --network balanced_topology_gated_contention_shared --reward-version collision_topology --episodes 1 --slots 8 --expert-bc-weight 0.3 --expert-protocol budgeted_collision_aware_isac ...
```

This confirmed the Budgeted expert BC path writes manifests and non-normal gate counts during training:

| Gate | Training ratio |
|---|---:|
| backoff | 0.1875 |
| normal | 0.6667 |
| aggressive | 0.1458 |

Campaign dry-run:

```text
python 05_simulation/run_marl_training_stability_campaign.py --campaign budgeted_gate_bc_sweep_20260709 --methods balanced_topology_gated_contention_actor --seeds 20260751 --episodes 100 --slots 300 --expert-bc-weights 0.15 0.30 0.50 --expert-protocol budgeted_collision_aware_isac --dry-run --quiet
```

This generated three planned training runs with the expected Budgeted expert BC weights and command-line arguments.

## Claim Boundary

Supported now:

- The simulator has a concrete trust-gated table-exchange mechanism.
- The MARL training path can imitate a Budgeted ISAC expert's access gate.
- Unit/smoke tests cover both mechanisms.
- A formal B=10, N=100, 3000-slot, 5-episode protocol evaluation has now
  closed the first table-exchange question. In this setting, table exchange
  improves raw discovery but creates too many collisions to improve CPD:
  `improved_rl_isac_tables` reaches discovery 0.7438 with 9247.0 collisions
  and CPD 0.2608; `trust_gated_isac_tables` reaches discovery 0.7446 with
  9516.4 collisions and CPD 0.2569; `budgeted_collision_aware_isac` remains
  stronger on CPD at 0.5563 with 1300.8 collisions.

Not yet supported:

- Trust-gated table exchange improves CPD. Current B=10 evidence is negative,
  so any paper claim should say that neighbor-table sharing needs access
  budgeting and collision control rather than being a standalone gain.
- B=15 trust-gated table behavior is not yet measured in this sweep.
- Budgeted expert gate BC improves N=10 to N=100 transfer. The 0.15-weight
  run is still in progress at the time of this update and has not yet produced
  a final model for transfer evaluation.

## Next Experiment Block

Recommended next long run:

1. Train `balanced_topology_gated_contention_shared` at N=10, B=10, 300 slots, using `expert_protocol=budgeted_collision_aware_isac`, `expert_bc_weight` sweep in `{0.15, 0.30, 0.50}`.
2. Evaluate checkpoints at N=100, B=10 and B=15, 3000 slots, 5-10 episodes.
3. Add protocol baselines: `trust_gated_isac_tables`, `improved_rl_isac_tables`, `wang2025_isac_tables`, `budgeted_collision_aware_isac`, `uniform_random`.
4. Promote the result only if the learned gate improves CPD or collision count without collapsing raw discovery.

Current generated artifacts:

- `06_analysis/paper_tables/marl/trust_gate_bc_sweep_20260709/protocol_eval_summary.csv`
- `06_analysis/paper_tables/marl/trust_gate_bc_sweep_20260709/bc_training_runs.csv`
- `06_analysis/paper_figures/marl_trust_gate_bc_20260709/protocol_cpd_b10_n100.png`
- `06_analysis/paper_figures/marl_trust_gate_bc_20260709/bc_eval_cpd_by_checkpoint.png`
