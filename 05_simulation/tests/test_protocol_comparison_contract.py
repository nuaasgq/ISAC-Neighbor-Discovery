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

ISAC_PROTOCOLS = ["improved_rl_isac", "isac_only", "itap_nd"]


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
