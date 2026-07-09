from __future__ import annotations

from dataclasses import replace

import numpy as np

from isac_nd_sim.config import SimulationConfig, load_config
from isac_nd_sim.runner import run_detailed
from isac_nd_sim.simulator import MODE_TX, NeighborDiscoverySimulator


PAPER_PROTOCOLS = [
    "skyorbs_like_skip_scan",
    "uniform_random",
    "rl_no_isac",
    "improved_rl_no_isac",
    "improved_rl_isac",
]

NO_ISAC_PROTOCOLS = [
    "rl_no_isac",
    "improved_rl_no_isac",
    "skyorbs_like_skip_scan",
    "uniform_random",
]

ISAC_PROTOCOLS = [
    "improved_rl_isac",
    "collision_aware_isac",
    "budgeted_collision_aware_isac",
    "ablation_isac_one_slot_delay",
    "isac_only",
    "itap_nd",
]


def compact_config() -> SimulationConfig:
    cfg = load_config("05_simulation/configs/mvp.yaml")
    return replace(
        cfg,
        episodes=1,
        slots_per_episode=10,
        n_nodes=6,
        azimuth_cells=8,
        elevation_cells=4,
        communication_range_m=350.0,
        sensing_range_m=400.0,
        baselines=tuple(PAPER_PROTOCOLS),
    )


def initialized_simulator(cfg: SimulationConfig, protocol: str, seed: int = 2027) -> NeighborDiscoverySimulator:
    simulator = NeighborDiscoverySimulator(cfg, protocol, seed)
    simulator.reset()
    simulator.age.fill(1.0)
    simulator.success_count.fill(0.0)
    simulator.fail_count.fill(0.0)
    return simulator


def test_runner_runs_five_paper_comparison_protocols() -> None:
    cfg = compact_config()
    rows, slot_rows, edge_rows = run_detailed(cfg, PAPER_PROTOCOLS)

    assert [row["protocol"] for row in rows] == PAPER_PROTOCOLS
    assert len(slot_rows) == cfg.slots_per_episode * len(PAPER_PROTOCOLS)
    assert isinstance(edge_rows, list)
    for row in rows:
        assert "p99_discovery_delay" in row
        assert "p99_delay_censored" in row
        assert "scan_actions" in row
        assert "discovery_per_scan_action" in row
        assert "collision_penalized_discovery_rate" in row
        assert "collision_normalized_efficiency" in row
        assert "energy_j" in row
        assert "discoveries_per_joule" in row
        assert row["p99_discovery_delay"] == row["p99_delay_censored"]


def test_episode_result_and_runner_rows_include_p99_delay() -> None:
    cfg = compact_config()
    simulator = initialized_simulator(cfg, "uniform_random")
    episode_row = simulator.run_episode(episode=0).as_dict()

    assert "p99_delay_censored" in episode_row
    rows, _slot_rows, _edge_rows = run_detailed(cfg, ["uniform_random"])
    assert "p99_discovery_delay" in rows[0]


def test_no_isac_protocol_beam_selection_ignores_belief_values() -> None:
    cfg = replace(compact_config(), softmax_beta=3.0, exploration_floor=0.05)
    true_edges = {(0, 1)}
    oracle_occupied = {3, 7}

    for protocol in NO_ISAC_PROTOCOLS:
        low_belief = initialized_simulator(cfg, protocol)
        high_belief = initialized_simulator(cfg, protocol)
        low_belief.belief.fill(0.0)
        high_belief.belief.fill(0.0)
        high_belief.belief[0, 5] = 1.0
        high_belief.belief[0, 17] = 0.8
        low_belief.rng = np.random.default_rng(12345)
        high_belief.rng = np.random.default_rng(12345)

        assert low_belief.select_beam(0, 4, MODE_TX, oracle_occupied, true_edges) == high_belief.select_beam(
            0,
            4,
            MODE_TX,
            oracle_occupied,
            true_edges,
        )


def test_isac_protocol_beam_selection_can_follow_belief_values() -> None:
    cfg = replace(
        compact_config(),
        alpha_occupancy=50.0,
        softmax_beta=5.0,
        exploration_floor=1e-9,
    )
    true_edges = {(0, 1)}
    oracle_occupied: set[int] = set()

    for protocol in ISAC_PROTOCOLS:
        beam_a = initialized_simulator(cfg, protocol)
        beam_b = initialized_simulator(cfg, protocol)
        beam_a.belief.fill(0.0)
        beam_b.belief.fill(0.0)
        beam_a.belief[0, 3] = 1.0
        beam_b.belief[0, 21] = 1.0
        beam_a.rng = np.random.default_rng(9)
        beam_b.rng = np.random.default_rng(9)

        assert beam_a.select_beam(0, 4, MODE_TX, oracle_occupied, true_edges) == 3
        assert beam_b.select_beam(0, 4, MODE_TX, oracle_occupied, true_edges) == 21


def test_one_slot_delay_handshake_uses_pre_sensing_candidate_snapshot() -> None:
    cfg = replace(compact_config(), exploration_floor=1e-9)
    delayed = initialized_simulator(cfg, "ablation_isac_one_slot_delay")
    immediate = initialized_simulator(cfg, "improved_rl_isac")
    target_beam = 5

    delayed.belief.fill(0.0)
    immediate.belief.fill(0.0)
    delayed.snapshot_pre_sensing_candidates(slot=3)

    for simulator in (delayed, immediate):
        simulator.belief[0, target_beam] = 1.0
        simulator.success_count[0, target_beam] = 1.0
        simulator.last_positive_slot[0, target_beam] = 3
        simulator._candidate_pool_cache.clear()

    assert target_beam in set(immediate.handshake_candidate_pool(0, 3).tolist())
    assert target_beam not in set(delayed.handshake_candidate_pool(0, 3).tolist())


def test_collision_aware_isac_reduces_tx_probability_under_candidate_pressure() -> None:
    cfg = replace(compact_config(), target_degree=4)
    simulator = initialized_simulator(cfg, "collision_aware_isac")
    simulator.belief.fill(0.0)
    simulator.belief[0, [2, 5, 7, 11]] = 1.0
    simulator.collision_fail_count[0, [2, 5, 7, 11]] = 2.0

    collision_aware = simulator.collision_aware_role_probabilities(0, slot=12, degree_need=1.0)
    baseline_tx_probability = 0.50 + 0.05

    assert collision_aware[0] < baseline_tx_probability
    assert collision_aware[1] > 1.0 - 0.01 - baseline_tx_probability
    np.testing.assert_allclose(collision_aware.sum(), 1.0)


def test_budgeted_collision_aware_isac_uses_lower_tx_budget_than_collision_aware() -> None:
    cfg = replace(compact_config(), target_degree=4, n_nodes=40, azimuth_cells=12, elevation_cells=4)
    collision_aware = initialized_simulator(cfg, "collision_aware_isac")
    budgeted = initialized_simulator(cfg, "budgeted_collision_aware_isac")
    for simulator in (collision_aware, budgeted):
        simulator.belief.fill(0.0)
        simulator.belief[0, [2, 5, 7, 11]] = 1.0
        simulator.collision_fail_count[0, [2, 5, 7, 11]] = 2.0

    collision_probs = collision_aware.collision_aware_role_probabilities(0, slot=12, degree_need=1.0)
    budgeted_probs = budgeted.budgeted_collision_aware_role_probabilities(0, slot=12, degree_need=1.0)

    assert budgeted_probs[0] < collision_probs[0]
    assert budgeted_probs[1] > collision_probs[1]
    np.testing.assert_allclose(budgeted_probs.sum(), 1.0)


def test_radio_energy_uses_configured_state_powers() -> None:
    cfg = replace(
        compact_config(),
        slot_duration_s=0.01,
        tx_power_w=2.0,
        rx_power_w=1.0,
        sense_power_w=3.0,
        idle_power_w=0.1,
        piggyback_sense_power_w=0.5,
    )
    simulator = initialized_simulator(cfg, "improved_rl_isac")
    simulator.tx_actions = 3
    simulator.rx_actions = 5
    simulator.sense_actions = 2
    simulator.idle_actions = 7
    simulator.piggyback_sense_actions = 4

    np.testing.assert_allclose(
        simulator.radio_energy_j(),
        0.01 * (3 * 2.0 + 5 * 1.0 + 2 * 3.0 + 7 * 0.1 + 4 * 0.5),
    )
