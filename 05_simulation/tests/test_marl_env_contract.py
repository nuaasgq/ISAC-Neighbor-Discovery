from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from isac_nd_sim.config import load_config
from isac_nd_sim.marl_env import MarlNeighborDiscoveryEnv


FORBIDDEN_KEYS = {
    "true_edges",
    "true_adjacency",
    "positions",
    "velocities",
    "neighbor_positions",
    "all_positions",
    "undiscovered_neighbors",
}


def _small_cfg():
    cfg = load_config("05_simulation/configs/mvp.yaml")
    return replace(
        cfg,
        episodes=1,
        slots_per_episode=4,
        n_nodes=6,
        azimuth_cells=6,
        elevation_cells=3,
        communication_range_m=260.0,
        sensing_range_m=320.0,
    )


def _assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        leaked = FORBIDDEN_KEYS.intersection(value.keys())
        assert not leaked
        for child in value.values():
            _assert_no_forbidden_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_forbidden_keys(child)


def test_marl_env_reset_observation_contract_has_no_neighbor_truth() -> None:
    cfg = _small_cfg()
    env = MarlNeighborDiscoveryEnv(cfg)

    observations, info = env.reset(seed=123)

    assert len(observations) == cfg.n_nodes
    assert info["n_agents"] == cfg.n_nodes
    assert info["n_beams"] == cfg.n_beams
    _assert_no_forbidden_keys(info)
    _assert_no_forbidden_keys(observations)

    first = observations[0]
    assert first["agent_id"] == 0
    assert first["self_state"].shape == (9,)
    assert first["beam_belief"].shape == (cfg.n_beams,)
    assert first["beam_age"].shape == (cfg.n_beams,)
    assert first["beam_success"].shape == (cfg.n_beams,)
    assert first["beam_fail"].shape == (cfg.n_beams,)
    assert first["last_mode"].shape == (4,)
    assert first["last_beam"].shape == (1,)
    assert first["local_summary"].shape == (4,)


def test_marl_env_step_accepts_mode_beam_actions_and_keeps_info_safe() -> None:
    cfg = _small_cfg()
    env = MarlNeighborDiscoveryEnv(cfg)
    assert env.protocol == "isac_structured_marl"
    observations, _ = env.reset(seed=7)
    assert len(observations) == cfg.n_nodes

    actions = [
        {"mode": "sense", "beam": 0},
        {"mode": "tx", "beam": 1},
        {"mode": "rx", "beam": 2},
        {"mode": "idle"},
        (1, 3),
        (2, 4),
    ]
    next_observations, rewards, terminated, truncated, info = env.step(actions)

    assert len(next_observations) == cfg.n_nodes
    assert rewards.shape == (cfg.n_nodes,)
    assert rewards.dtype == np.float32
    assert terminated is False
    assert truncated is False
    assert info["slot"] == 1
    assert "new_edges_count" in info
    assert info["scan_actions"] == 4
    assert info["tx_actions"] == 2
    assert info["rx_actions"] == 2
    assert info["sense_actions"] == 1
    assert info["idle_actions"] == 1
    assert info["piggyback_sense_actions"] == 4
    assert "discovery_per_scan_action" in info
    _assert_no_forbidden_keys(info)
    _assert_no_forbidden_keys(next_observations)


def test_training_state_is_explicitly_separate_from_public_info() -> None:
    cfg = _small_cfg()
    env = MarlNeighborDiscoveryEnv(cfg)
    _, info = env.reset(seed=11)
    actions = [{"mode": "sense", "beam": node % cfg.n_beams} for node in range(cfg.n_nodes)]
    _, _, _, _, step_info = env.step(actions)
    state = env.training_state()

    _assert_no_forbidden_keys(info)
    _assert_no_forbidden_keys(step_info)
    assert "true_edges" in state
    assert "true_adjacency" in state
    assert "positions" in state
    assert state["positions"].shape == (cfg.n_nodes, 3)
    assert state["velocities"].shape == (cfg.n_nodes, 3)
    assert state["attitudes"].shape == (cfg.n_nodes, 3)
    assert state["true_adjacency"].shape == (cfg.n_nodes, cfg.n_nodes)
    assert state["discovered_adjacency"].shape == (cfg.n_nodes, cfg.n_nodes)


def test_marl_env_truncates_at_episode_horizon() -> None:
    cfg = _small_cfg()
    env = MarlNeighborDiscoveryEnv(cfg)
    env.reset(seed=19)
    actions = [{"mode": "idle"} for _ in range(cfg.n_nodes)]

    truncated = False
    for _ in range(cfg.slots_per_episode):
        _, _, _, truncated, _ = env.step(actions)

    assert truncated is True
