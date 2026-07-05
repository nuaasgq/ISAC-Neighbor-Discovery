# Energy-Efficiency Accounting Status and Extension Plan

Date: 2026-07-05

## Current Status

The simulator already records action-level counters:

- `tx_actions`
- `rx_actions`
- `sense_actions`
- `idle_actions`
- `piggyback_sense_actions`
- `scan_actions`
- `collision_count`

The original paper-ready efficiency metrics were scan-action and collision normalized:

- `discovery_per_scan_action`
- `discoveries_per_1000_scan_actions`
- `scan_actions_per_discovery_censored`
- `collision_normalized_efficiency`
- `collision_penalized_discovery_rate`

Round12 added the simulator-side assumed radio-state accounting metrics, and round13 reports the ten-seed collision/energy probe:

- `energy_j`
- `discoveries_per_joule`
- `energy_per_discovery_censored_j`

These are useful reviewer diagnostics, but they are still not platform-calibrated energy-optimality results.

## What Is Now Implemented

The configuration files define action probabilities such as `p_tx`, `p_rx`, and `p_sense`; these remain probabilities, not powers.
Optional radio-state powers are now loaded from an `energy` section when present, with defaults used otherwise.
The MARL planning file still keeps power actions disabled with `include_power: false`, so this is accounting rather than energy-aware control.

Implemented optional parameters and current defaults:

| Parameter | Meaning | Example unit |
|---|---|---|
| `tx_power_w` | transmit-chain power during TX slots | W, default 1.0 |
| `rx_power_w` | receive-chain power during RX slots | W, default 0.6 |
| `sense_power_w` | dedicated sensing power during sense slots | W, default 1.2 |
| `idle_power_w` | idle baseline power | W, default 0.05 |
| `piggyback_sense_power_w` | additional sensing/processing cost during piggyback sensing | W, default 0.2 |

Then compute

```text
energy_j =
  slot_duration_s * (
    tx_actions * tx_power_w
    + rx_actions * rx_power_w
    + sense_actions * sense_power_w
    + idle_actions * idle_power_w
    + piggyback_sense_actions * piggyback_sense_power_w
  )
```

Derived metrics:

- `energy_per_discovery_censored_j = energy_j / max(1, discovered_edges)`
- `discoveries_per_joule = discovered_edges / max(eps, energy_j)`

## Implemented Touch Points

1. `SimulationConfig` and YAML loading in `05_simulation/src/isac_nd_sim/config.py`.
2. `EpisodeResult`, `summarize()`, and `radio_energy_j()` in `05_simulation/src/isac_nd_sim/simulator.py`.
3. Focused deterministic energy-accounting test in `05_simulation/tests/test_protocol_comparison_contract.py`.
4. Round13 plotting hooks in `06_analysis/scripts/plot_round12_collision_aware.py` using `--tag round13`.

## Manuscript Wording

Use:

> The supplement reports assumed radio-state energy accounting under stated default powers. These metrics are diagnostic and not platform-calibrated energy-optimality results.

Avoid making an energy-optimality claim for the proposed method.

Avoid:

> The radio-state power model is calibrated to a UAV hardware platform.
