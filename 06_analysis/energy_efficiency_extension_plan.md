# Energy-Efficiency Extension Plan

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

The current paper-ready efficiency metrics are therefore scan-action and collision normalized:

- `discovery_per_scan_action`
- `discoveries_per_1000_scan_actions`
- `scan_actions_per_discovery_censored`
- `collision_normalized_efficiency`
- `collision_penalized_discovery_rate`

These are useful protocol-efficiency metrics, but they are not Joule-normalized energy metrics.

## Why a Joule Model Is Not Yet Claimed

The configuration files currently define action probabilities such as `p_tx`, `p_rx`, and `p_sense`; these are not transmit, receive, or sensing power values. The MARL planning file explicitly keeps power actions disabled with `include_power: false`.

Therefore, the manuscript should not claim energy-normalized discovery or delay-power product performance until a radio-state energy model is added.

## Minimal Energy Model to Add Later

Add optional power parameters to the simulation config:

| Parameter | Meaning | Example unit |
|---|---|---|
| `power_tx_w` | transmit-chain power during TX slots | W |
| `power_rx_w` | receive-chain power during RX slots | W |
| `power_sense_w` | dedicated sensing power during sense slots | W |
| `power_idle_w` | idle baseline power | W |
| `power_piggyback_sense_w` | additional sensing/processing cost during piggyback sensing | W |

Then compute

```text
energy_j =
  slot_duration_s * (
    tx_actions * power_tx_w
    + rx_actions * power_rx_w
    + sense_actions * power_sense_w
    + idle_actions * power_idle_w
    + piggyback_sense_actions * power_piggyback_sense_w
  )
```

Derived metrics:

- `energy_per_discovery_j = energy_j / max(1, discovered_edges)`
- `discoveries_per_joule = discovered_edges / max(eps, energy_j)`
- `delay_energy_product = mean_delay_censored * energy_per_discovery_j`

## Implementation Touch Points

1. Extend `SimulationConfig` and YAML loading in `05_simulation/src/isac_nd_sim/config.py`.
2. Add fields to `EpisodeResult` and `summarize()` in `05_simulation/src/isac_nd_sim/simulator.py`.
3. Add aliases in `05_simulation/src/isac_nd_sim/runner.py` if needed.
4. Add plotting hooks in `06_analysis/scripts/plot_paper_results.py` and `06_analysis/scripts/plot_round3_results.py`.
5. Add a focused test confirming that the same action counters produce deterministic energy metrics.

## Manuscript Wording Until Then

Use:

> The present draft reports scan-action and collision-normalized efficiency. A Joule-level energy comparison requires an explicit radio-state power model and is left for the next implementation step.

Avoid:

> The proposed method is energy efficient.

Avoid:

> Energy-normalized discovery improves by ...
