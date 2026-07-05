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
    assert first["beam_collision"].shape == (cfg.n_beams,)
    assert first["candidate_mask"].shape == (cfg.n_beams,)
    assert first["candidate_score"].shape == (cfg.n_beams,)
    assert first["topology_deficit"].shape == (1,)
    assert first["contention_state"].shape == (10,)
    assert first["rule_mode_logits"].shape == (4,)
    assert np.count_nonzero(first["candidate_mask"]) >= 1
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


def test_collision_topology_reward_version_keeps_public_contract_safe() -> None:
    cfg = _small_cfg()
    env = MarlNeighborDiscoveryEnv(cfg, reward_version="collision_topology")
    observations, _ = env.reset(seed=8)
    assert len(observations) == cfg.n_nodes

    actions = [
        {"mode": "sense", "beam": 0},
        {"mode": "tx", "beam": 1},
        {"mode": "rx", "beam": 1},
        {"mode": "tx", "beam": 2},
        {"mode": "rx", "beam": 2},
        {"mode": "idle"},
    ]
    next_observations, rewards, terminated, truncated, info = env.step(actions)

    assert rewards.shape == (cfg.n_nodes,)
    assert np.isfinite(rewards).all()
    assert terminated is False
    assert truncated is False
    assert info["reward_version"] == "collision_topology"
    _assert_no_forbidden_keys(info)
    _assert_no_forbidden_keys(next_observations)


def test_no_isac_marl_protocol_does_not_update_belief_from_sense_action() -> None:
    cfg = _small_cfg()
    env = MarlNeighborDiscoveryEnv(cfg, protocol="structured_marl_no_isac")
    env.reset(seed=9)
    env._sim.belief[:, :] = 0.0
    env._sim.belief[0, 0] = 1.0

    actions = [{"mode": "sense", "beam": 0} for _ in range(cfg.n_nodes)]
    next_observations, _, _, _, info = env.step(actions)

    assert info["sense_actions"] == cfg.n_nodes
    assert info["piggyback_sense_actions"] == 0
    assert env._sim.belief[0, 0] == np.float64(cfg.confidence_decay)
    assert np.count_nonzero(env._sim.belief[0]) == 1
    assert next_observations[0]["beam_belief"][0] == np.float32(cfg.confidence_decay)


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


def test_fast_eval_env_preserves_final_summary_without_slot_metrics() -> None:
    cfg = _small_cfg()
    rich = MarlNeighborDiscoveryEnv(cfg, collect_slot_metrics=True, rich_info=True)
    fast = MarlNeighborDiscoveryEnv(cfg, collect_slot_metrics=False, rich_info=False)
    rich.reset(seed=31)
    fast.reset(seed=31)

    for slot in range(cfg.slots_per_episode):
        actions = []
        for node in range(cfg.n_nodes):
            mode = ["sense", "tx", "rx", "idle"][(node + slot) % 4]
            actions.append({"mode": mode, "beam": (node + 2 * slot) % cfg.n_beams})
        rich_next, rich_rewards, rich_terminated, rich_truncated, rich_info = rich.step(actions)
        fast_next, fast_rewards, fast_terminated, fast_truncated, fast_info = fast.step(actions)

        assert len(rich_next) == len(fast_next) == cfg.n_nodes
        assert np.array_equal(rich_rewards, fast_rewards)
        assert rich_terminated == fast_terminated
        assert rich_truncated == fast_truncated
        assert rich_info["new_edges_count"] == fast_info["new_edges_count"]
        assert "lambda2" in rich_info
        assert "lambda2" not in fast_info

    assert len(rich._sim.per_slot_rows) == cfg.slots_per_episode
    assert len(fast._sim.per_slot_rows) == 0
    assert rich._sim.summarize(0).as_dict() == fast._sim.summarize(0).as_dict()
