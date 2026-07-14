from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def candidate_pool_snapshot(
    simulator: Any,
    observations: Sequence[dict[str, Any]],
    actions: Sequence[Any],
    *,
    eval_episode: int,
    scenario_seed: int,
    slot: int,
    method: str,
    phase: str = "eval_stochastic",
) -> dict[str, Any]:
    """Summarize actor-visible candidate support using truth only for offline diagnostics."""

    n_nodes = int(simulator.cfg.n_nodes)
    n_beams = int(simulator.cfg.n_beams)
    if len(observations) != n_nodes or len(actions) != n_nodes:
        raise ValueError("Candidate diagnostics require one observation and action per node.")

    masks = np.vstack(
        [
            np.asarray(observation.get("candidate_mask", np.ones(n_beams)), dtype=float) > 0.5
            for observation in observations
        ]
    )
    if masks.shape != (n_nodes, n_beams):
        raise ValueError(f"Expected candidate masks {(n_nodes, n_beams)}, received {masks.shape}.")

    true_edges = simulator.true_edges(float(simulator.cfg.communication_range_m))
    undiscovered_edges = sorted(true_edges.difference(simulator.discovered_edges))
    positive_beams = [set() for _ in range(n_nodes)]
    directed_pair_total = 0
    directed_pair_retained = 0
    for node_a, node_b in undiscovered_edges:
        for source, target in ((node_a, node_b), (node_b, node_a)):
            beam = int(simulator.beam_from_to(source, target))
            positive_beams[source].add(beam)
            directed_pair_total += 1
            directed_pair_retained += int(masks[source, beam])

    positive_beam_total = sum(len(beams) for beams in positive_beams)
    positive_beam_retained = sum(
        int(masks[node, beam])
        for node, beams in enumerate(positive_beams)
        for beam in beams
    )
    total_node_beams = n_nodes * n_beams
    candidate_total = int(masks.sum())
    empty_opportunity_total = total_node_beams - positive_beam_total
    excluded_total = total_node_beams - candidate_total
    false_exclusion_total = positive_beam_total - positive_beam_retained
    excluded_empty_total = excluded_total - false_exclusion_total
    directed_retention_defined = directed_pair_total > 0
    positive_recall_defined = positive_beam_total > 0

    active_action_count = 0
    selected_candidate_count = 0
    selected_positive_count = 0
    tx_count = 0
    for node, action in enumerate(actions):
        mode = str(action.mode)
        if mode == "tx":
            tx_count += 1
        if mode == "idle":
            continue
        beam = int(action.beam)
        active_action_count += 1
        selected_candidate_count += int(masks[node, beam])
        selected_positive_count += int(beam in positive_beams[node])

    return {
        "phase": str(phase),
        "method": str(method),
        "eval_episode": int(eval_episode),
        "scenario_seed": int(scenario_seed),
        "slot_zero_based": int(slot),
        "elapsed_slots": int(slot) + 1,
        "candidate_count_mean": float(masks.sum(axis=1).mean()),
        "candidate_count_min": int(masks.sum(axis=1).min()),
        "candidate_count_max": int(masks.sum(axis=1).max()),
        "candidate_fraction": candidate_total / max(1, total_node_beams),
        "candidate_reduction_fraction": excluded_total / max(1, total_node_beams),
        "undiscovered_directed_pairs": int(directed_pair_total),
        "undiscovered_pair_beam_retention_defined": directed_retention_defined,
        "undiscovered_pair_beam_retention": (
            directed_pair_retained / directed_pair_total if directed_retention_defined else 1.0
        ),
        "undiscovered_positive_beams": int(positive_beam_total),
        "positive_beam_recall_defined": positive_recall_defined,
        "positive_beam_recall": (
            positive_beam_retained / positive_beam_total if positive_recall_defined else 1.0
        ),
        "false_exclusion_rate": (
            false_exclusion_total / positive_beam_total if positive_recall_defined else 0.0
        ),
        "empty_opportunity_beams": int(empty_opportunity_total),
        "empty_opportunity_exclusion_rate": excluded_empty_total / max(1, empty_opportunity_total),
        "candidate_positive_precision": positive_beam_retained / max(1, candidate_total),
        "active_actions": int(active_action_count),
        "selected_candidate_compliance": selected_candidate_count / max(1, active_action_count),
        "selected_undiscovered_beam_rate": selected_positive_count / max(1, active_action_count),
        "tx_fraction": tx_count / max(1, n_nodes),
        "discovered_edges_pre_slot": int(len(simulator.discovered_edges.intersection(true_edges))),
        "true_edges": int(len(true_edges)),
        "discovery_rate_pre_slot": len(simulator.discovered_edges.intersection(true_edges))
        / max(1, len(true_edges)),
    }
