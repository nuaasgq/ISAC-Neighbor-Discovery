from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from isac_nd_sim.beam import beam_matches
from isac_nd_sim.config import load_config
from isac_nd_sim.marl_env import MarlNeighborDiscoveryEnv
from isac_nd_sim.simulator import Action, MODE_RX, MODE_TX, NeighborDiscoverySimulator


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
    assert first["last_access_gate"].shape == (3,)
    assert first["last_beam"].shape == (1,)
    assert first["local_summary"].shape == (4,)


def test_wang_table_candidate_source_exposes_sensing_flags() -> None:
    cfg = _small_cfg()
    env = MarlNeighborDiscoveryEnv(cfg, protocol="wang2025_isac_tables", candidate_source="wang_table")
    observations, _ = env.reset(seed=124)

    first = observations[0]
    assert np.count_nonzero(first["candidate_mask"]) == cfg.n_beams
    assert np.all(first["candidate_score"] >= 0.0)
    assert np.all(first["candidate_score"] <= 1.0)

    env._sim.wang_sensing_flag[0, :] = 0.0
    env._sim.wang_sensing_flag[0, [2, 5]] = 1.0
    updated = env._observation_for(0)

    assert set(np.flatnonzero(updated["candidate_mask"] > 0.5).tolist()) == {2, 5}


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


def test_access_gate_action_modulates_effective_role() -> None:
    cfg = _small_cfg()
    env = MarlNeighborDiscoveryEnv(cfg)
    env.reset(seed=17)
    env._sim.collision_fail_count[0, 1] = 1.0

    actions = [
        {"mode": "tx", "beam": 1, "access_gate": "backoff"},
        {"mode": "rx", "beam": 1, "access_gate": "aggressive"},
        {"mode": "tx", "beam": 2, "access_gate": "normal"},
        {"mode": "rx", "beam": 2},
        {"mode": "idle", "access_gate": "aggressive"},
        (2, 3, 0),
    ]
    next_observations, _rewards, _terminated, _truncated, info = env.step(actions)

    assert info["tx_actions"] == 2
    assert info["rx_actions"] == 3
    assert info["idle_actions"] == 1
    assert env._last_actions[0].mode == "rx"
    assert env._last_actions[0].access_gate == "backoff"
    assert env._last_actions[1].mode == "tx"
    assert env._last_actions[1].access_gate == "aggressive"
    assert next_observations[0]["last_access_gate"].tolist() == [1.0, 0.0, 0.0]


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


def test_discovery_first_reward_version_keeps_public_contract_safe() -> None:
    cfg = _small_cfg()
    env = MarlNeighborDiscoveryEnv(cfg, reward_version="discovery_first")
    observations, _ = env.reset(seed=8)
    assert len(observations) == cfg.n_nodes

    actions = [
        {"mode": "tx", "beam": 1},
        {"mode": "rx", "beam": 1},
        {"mode": "tx", "beam": 2},
        {"mode": "rx", "beam": 2},
        {"mode": "idle"},
        {"mode": "idle"},
    ]
    next_observations, rewards, terminated, truncated, info = env.step(actions)

    assert rewards.shape == (cfg.n_nodes,)
    assert rewards.dtype == np.float32
    assert np.isfinite(rewards).all()
    assert terminated is False
    assert truncated is False
    assert info["reward_version"] == "discovery_first"
    _assert_no_forbidden_keys(info)
    _assert_no_forbidden_keys(next_observations)


def test_discovery_access_stable_reward_penalizes_all_idle_more_than_discovery_first() -> None:
    cfg = _small_cfg()
    first_env = MarlNeighborDiscoveryEnv(cfg, reward_version="discovery_first")
    stable_env = MarlNeighborDiscoveryEnv(cfg, reward_version="discovery_access_stable")
    first_env.reset(seed=18)
    stable_env.reset(seed=18)

    actions = [{"mode": "idle"} for _ in range(cfg.n_nodes)]
    _first_obs, first_rewards, _terminated, _truncated, _first_info = first_env.step(actions)
    _stable_obs, stable_rewards, _terminated, _truncated, stable_info = stable_env.step(actions)

    assert stable_info["reward_version"] == "discovery_access_stable"
    assert np.all(stable_rewards < first_rewards)


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


def test_tx_coupled_ideal_isac_records_sensing_observations_without_sense_action() -> None:
    cfg = replace(
        load_config("05_simulation/configs/paper_transfer_train_n10_b10_singlehop.yaml"),
        episodes=1,
        slots_per_episode=1,
        n_nodes=4,
        azimuth_cells=6,
        elevation_cells=3,
        communication_range_m=1000.0,
        sensing_range_m=1000.0,
    )
    env = MarlNeighborDiscoveryEnv(cfg, reward_version="discovery_first")
    env.reset(seed=23)

    actions = [
        {"mode": "tx", "beam": 0},
        {"mode": "rx", "beam": 1},
        {"mode": "tx", "beam": 2},
        {"mode": "rx", "beam": 3},
    ]
    _, _, _, _, info = env.step(actions)

    assert info["sense_actions"] == 0
    assert info["piggyback_sense_actions"] == cfg.n_nodes
    assert info["sensing_observations"] > 0
    assert info["sensing_target_observations"] >= 0


def test_wang_table_env_uses_tx_only_piggyback_and_neighbor_table_exchange() -> None:
    cfg = replace(
        load_config("05_simulation/configs/wang2025_reproduction_smoke.yaml"),
        episodes=1,
        slots_per_episode=1,
        n_nodes=3,
        azimuth_cells=6,
        elevation_cells=3,
        communication_range_m=20000.0,
        sensing_range_m=20000.0,
        false_alarm_rate=0.0,
        miss_detection_rate=0.0,
        angular_cell_offset_std=0.0,
    )
    env = MarlNeighborDiscoveryEnv(cfg, protocol="wang2025_isac_tables", reward_version="discovery_first")
    env.reset(seed=37)

    env._sim.discovered_edges.add((1, 2))
    beam_02 = env._sim.beam_from_to(0, 2)
    env._sim.belief[0, beam_02] = 0.0
    env._sim.success_count[0, beam_02] = 0.0
    env._sim.exchange_neighbor_and_sensing_tables(0, 1, slot=0)

    assert env._sim.belief[0, beam_02] > 0.0
    assert env._sim.success_count[0, beam_02] > 0.0

    beam_01 = env._sim.beam_from_to(0, 1)
    beam_10 = env._sim.beam_from_to(1, 0)
    actions = [
        {"mode": "tx", "beam": beam_01},
        {"mode": "rx", "beam": beam_10},
        {"mode": "rx", "beam": env._sim.beam_from_to(2, 0)},
    ]
    _, _, _, _, info = env.step(actions)

    assert info["sense_actions"] == 0
    assert info["piggyback_sense_actions"] == 1
    assert (0, 1) in env._sim.discovered_edges


def test_wang_table_env_does_not_reply_to_already_discovered_edge() -> None:
    cfg = replace(
        load_config("05_simulation/configs/wang2025_reproduction_smoke.yaml"),
        episodes=1,
        slots_per_episode=1,
        n_nodes=2,
        azimuth_cells=6,
        elevation_cells=3,
        communication_range_m=20000.0,
        sensing_range_m=20000.0,
        false_alarm_rate=0.0,
        miss_detection_rate=0.0,
        angular_cell_offset_std=0.0,
    )
    env = MarlNeighborDiscoveryEnv(cfg, protocol="wang2025_isac_tables", reward_version="discovery_first")
    env.reset(seed=41)
    env._sim.discovered_edges.add((0, 1))
    env._sim.discovery_slot[(0, 1)] = 0

    actions = [
        {"mode": "tx", "beam": env._sim.beam_from_to(0, 1)},
        {"mode": "rx", "beam": env._sim.beam_from_to(1, 0)},
    ]
    _, _, _, _, info = env.step(actions)

    assert info["new_edges_count"] == 0
    assert env._sim.edge_rows == []


def test_isac_candidate_pool_does_not_create_extra_handshake_beam() -> None:
    cfg = replace(
        load_config("05_simulation/configs/wang2025_reproduction_smoke.yaml"),
        episodes=1,
        slots_per_episode=1,
        n_nodes=2,
        azimuth_cells=6,
        elevation_cells=3,
        communication_range_m=20000.0,
        sensing_range_m=20000.0,
        alignment_tolerance_cells=0,
    )
    sim = NeighborDiscoverySimulator(cfg, protocol="wang2025_isac_tables", seed=53, scenario_seed=53)
    sim.reset()
    true_edges = {(0, 1)}
    beam_01 = sim.beam_from_to(0, 1)
    beam_10 = sim.beam_from_to(1, 0)
    sim.belief.fill(0.0)
    sim.belief[0, beam_01] = 1.0
    sim.belief[1, beam_10] = 1.0

    assert beam_01 in set(sim.handshake_candidate_pool(0, slot=0).tolist())
    assert beam_10 in set(sim.handshake_candidate_pool(1, slot=0).tolist())

    wrong_01 = next(
        beam
        for beam in range(cfg.n_beams)
        if not beam_matches(beam, beam_01, cfg.azimuth_cells, cfg.alignment_tolerance_cells)
    )
    wrong_10 = next(
        beam
        for beam in range(cfg.n_beams)
        if not beam_matches(beam, beam_10, cfg.azimuth_cells, cfg.alignment_tolerance_cells)
    )
    wrong_actions = [Action(MODE_TX, wrong_01), Action(MODE_RX, wrong_10)]

    assert sim.resolve_discoveries(0, wrong_actions, true_edges) == []
    assert sim.discovered_edges == set()

    correct_actions = [Action(MODE_TX, beam_01), Action(MODE_RX, beam_10)]
    assert sim.resolve_discoveries(0, correct_actions, true_edges) == [(0, 1)]
    assert sim.discovered_edges == {(0, 1)}


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
