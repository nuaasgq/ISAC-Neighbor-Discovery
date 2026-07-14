from __future__ import annotations

from typing import Any, Iterable


DEFAULT_DISCOVERY_MILESTONES = (50, 100, 150, 200, 300)


def discovery_timeline_rows(
    simulator: Any,
    *,
    eval_episode: int,
    scenario_seed: int,
    method: str,
    phase: str = "eval_stochastic",
) -> list[dict[str, Any]]:
    """Return one censored timeline row for every edge that became active."""

    horizon = int(simulator.cfg.slots_per_episode)
    rows: list[dict[str, Any]] = []
    for edge, first_true_slot in sorted(simulator.first_true_slot.items()):
        discovery_slot = simulator.discovery_slot.get(edge)
        discovered = discovery_slot is not None
        delay = (
            int(discovery_slot) - int(first_true_slot) + 1
            if discovered
            else max(1, horizon - int(first_true_slot))
        )
        rows.append(
            {
                "phase": str(phase),
                "method": str(method),
                "eval_episode": int(eval_episode),
                "scenario_seed": int(scenario_seed),
                "edge_i": int(edge[0]),
                "edge_j": int(edge[1]),
                "first_true_slot": int(first_true_slot),
                "discovered": bool(discovered),
                "discovery_slot_zero_based": int(discovery_slot) if discovered else "",
                "discovery_time_slots": int(discovery_slot) + 1 if discovered else horizon,
                "delay_slots_censored": int(delay),
                "horizon_slots": horizon,
            }
        )
    return rows


def discovery_curve_summary(
    simulator: Any,
    milestones: Iterable[int] = DEFAULT_DISCOVERY_MILESTONES,
) -> dict[str, float]:
    """Summarize cumulative edge discovery without changing simulator state."""

    horizon = int(simulator.cfg.slots_per_episode)
    first_true = simulator.first_true_slot
    discovery = simulator.discovery_slot
    summary: dict[str, float] = {}
    for requested in milestones:
        slot = min(horizon, max(1, int(requested)))
        eligible = [edge for edge, first in first_true.items() if int(first) < slot]
        found = sum(edge in discovery and int(discovery[edge]) < slot for edge in eligible)
        summary[f"discovery_rate_at_{int(requested)}_slots"] = found / max(1, len(eligible))

    if not first_true:
        summary["discovery_curve_auc_normalized"] = 0.0
        for target in (50, 80, 90):
            summary[f"time_to_{target}pct_censored_slots"] = float(horizon)
        return summary

    rates: list[float] = []
    for slot in range(1, horizon + 1):
        eligible = [edge for edge, first in first_true.items() if int(first) < slot]
        found = sum(edge in discovery and int(discovery[edge]) < slot for edge in eligible)
        rates.append(found / max(1, len(eligible)))
    summary["discovery_curve_auc_normalized"] = sum(rates) / horizon
    for target in (0.5, 0.8, 0.9):
        reached = next((slot for slot, rate in enumerate(rates, start=1) if rate >= target), horizon)
        summary[f"time_to_{int(100 * target)}pct_censored_slots"] = float(reached)
    return summary
