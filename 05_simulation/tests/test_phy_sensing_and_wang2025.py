from __future__ import annotations

from dataclasses import replace

import numpy as np

from isac_nd_sim.config import load_config
from isac_nd_sim.mobility import NodeState
from isac_nd_sim.phy_sensing import detection_probability, radar_snr_db
from isac_nd_sim.runner import run
from isac_nd_sim.simulator import Action, MODE_IDLE, MODE_RX, MODE_TX, NeighborDiscoverySimulator


def test_radar_snr_sensing_probability_decreases_with_distance() -> None:
    cfg = load_config("05_simulation/configs/mvp.yaml")
    cfg = replace(
        cfg,
        isac_sensing_model="radar_snr",
        isac_processing_gain_db=70.0,
        detection_midpoint_snr_db=-10.0,
        detection_slope_per_db=0.5,
    )

    near_snr = radar_snr_db(500.0, cfg)
    far_snr = radar_snr_db(5000.0, cfg)
    assert near_snr > far_snr
    assert detection_probability(500.0, cfg) > detection_probability(5000.0, cfg)


def test_default_config_keeps_constant_error_sensing_model() -> None:
    cfg = load_config("05_simulation/configs/mvp.yaml")
    assert cfg.isac_sensing_model == "constant_error"
    assert cfg.isac_waveform == "abstract"


def test_wang2025_isac_tables_runs_with_phy_sensing_metrics() -> None:
    cfg = load_config("05_simulation/configs/mvp.yaml")
    cfg = replace(
        cfg,
        episodes=1,
        slots_per_episode=20,
        n_nodes=8,
        azimuth_cells=12,
        elevation_cells=4,
        communication_range_m=900.0,
        sensing_range_m=900.0,
        false_alarm_rate=0.01,
        miss_detection_rate=0.05,
        angular_cell_offset_std=0.0,
        sensing_period_slots=1,
        slot_duration_s=0.005,
        isac_sensing_model="radar_snr",
        isac_waveform="mimo_otfs",
        isac_processing_gain_db=85.0,
        detection_midpoint_snr_db=-10.0,
        baselines=("uniform_random", "wang2025_isac_tables"),
    )
    rows = run(cfg, list(cfg.baselines))
    assert len(rows) == 2
    wang = next(row for row in rows if row["protocol"] == "wang2025_isac_tables")
    assert wang["piggyback_sense_actions"] > 0
    assert wang["sensing_observations"] > 0
    assert "mean_sensing_snr_db" in wang
    assert 0.0 <= wang["sensing_detection_rate"] <= 1.0


def test_wang2025_beam_selection_uses_sensing_table_flag() -> None:
    cfg = load_config("05_simulation/configs/mvp.yaml")
    cfg = replace(cfg, n_nodes=4, azimuth_cells=8, elevation_cells=3)
    simulator = NeighborDiscoverySimulator(cfg, "wang2025_isac_tables", seed=11)
    simulator.reset()
    simulator.wang_sensing_flag[0, :] = 0.0
    simulator.wang_sensing_flag[0, [3, 7]] = 1.0

    selected = {simulator.wang2025_table_beam(0) for _ in range(100)}

    assert selected <= {3, 7}
    assert selected == {3, 7}


def test_wang2025_sensing_table_closes_beam_after_discovered_target_count() -> None:
    cfg = load_config("05_simulation/configs/mvp.yaml")
    cfg = replace(cfg, n_nodes=2, azimuth_cells=8, elevation_cells=3, communication_range_m=1000.0)
    simulator = NeighborDiscoverySimulator(cfg, "wang2025_isac_tables", seed=12)
    simulator.reset()
    beam = 5
    simulator.wang_node_num[0, beam] = 1.0
    simulator.wang_dis_num[0, beam] = 0.0
    simulator.wang_sensing_flag[0, beam] = 1.0

    simulator.mark_wang2025_interaction(0, beam, slot=4)

    assert simulator.wang_dis_num[0, beam] == 1.0
    assert simulator.wang_sensing_flag[0, beam] == 0.0


def test_ours_table_exchange_boosts_beam_to_peer_known_neighbor() -> None:
    cfg = load_config("05_simulation/configs/mvp.yaml")
    cfg = replace(
        cfg,
        n_nodes=3,
        azimuth_cells=8,
        elevation_cells=4,
        communication_range_m=1000.0,
        sensing_range_m=1000.0,
        belief_update_rho=0.6,
    )
    simulator = NeighborDiscoverySimulator(cfg, "improved_rl_isac_tables", seed=20260708)
    simulator.reset()
    simulator.states = [
        NodeState(np.asarray([0.0, 0.0, 0.0]), np.zeros(3)),
        NodeState(np.asarray([100.0, 0.0, 0.0]), np.zeros(3)),
        NodeState(np.asarray([0.0, 100.0, 0.0]), np.zeros(3)),
    ]
    simulator._beam_matrix_cache = None
    simulator._distance_matrix_cache = None

    target_beam = simulator.beam_from_to(0, 2)
    before = float(simulator.belief[0, target_beam])
    simulator.discovered_edges.add((1, 2))
    simulator.neighbor_records[1][2] = (simulator.states[2].position.copy(), 3)

    simulator.exchange_neighbor_and_sensing_tables(0, 1, slot=3)

    assert simulator.belief[0, target_beam] > before
    assert simulator.success_count[0, target_beam] > 0.0
    assert simulator.last_positive_slot[0, target_beam] == 3


def test_table_exchange_does_not_infer_targets_from_global_truth() -> None:
    cfg = replace(load_config("05_simulation/configs/twc_trainable_n10.yaml"), n_nodes=3, azimuth_cells=8, elevation_cells=4)
    simulator = NeighborDiscoverySimulator(cfg, "improved_rl_isac_tables", seed=20260710)
    simulator.reset()
    target_beam = simulator.beam_from_to(0, 2)
    before = float(simulator.belief[0, target_beam])
    simulator.discovered_edges.add((1, 2))

    simulator.exchange_neighbor_and_sensing_tables(0, 1, slot=3)

    assert simulator.belief[0, target_beam] == before


def test_policy_rng_consumption_does_not_change_sensing_noise() -> None:
    cfg = replace(
        load_config("05_simulation/configs/twc_trainable_n10.yaml"),
        n_nodes=2,
        azimuth_cells=8,
        elevation_cells=4,
        false_alarm_rate=0.5,
        miss_detection_rate=0.5,
    )
    first = NeighborDiscoverySimulator(cfg, "improved_rl_isac_tables", seed=80, scenario_seed=81)
    second = NeighborDiscoverySimulator(cfg, "improved_rl_isac_tables", seed=80, scenario_seed=81)
    first.reset()
    second.reset()
    second.rng.random(100)
    first.states = [state for state in first.states]
    second.states = [state for state in first.states]
    first.invalidate_geometry_cache()
    second.invalidate_geometry_cache()

    first_value = first.sample_sensing_observation(0, 0, cfg.sensing_range_m, piggyback=True)
    second_value = second.sample_sensing_observation(0, 0, cfg.sensing_range_m, piggyback=True)

    assert first_value == second_value


def test_trust_gated_table_exchange_rejects_high_collision_hint() -> None:
    cfg = load_config("05_simulation/configs/mvp.yaml")
    cfg = replace(
        cfg,
        n_nodes=3,
        azimuth_cells=8,
        elevation_cells=4,
        communication_range_m=1000.0,
        sensing_range_m=1000.0,
        belief_update_rho=0.6,
    )
    simulator = NeighborDiscoverySimulator(cfg, "trust_gated_isac_tables", seed=20260709)
    simulator.reset()
    simulator.states = [
        NodeState(np.asarray([0.0, 0.0, 0.0]), np.zeros(3)),
        NodeState(np.asarray([100.0, 0.0, 0.0]), np.zeros(3)),
        NodeState(np.asarray([0.0, 100.0, 0.0]), np.zeros(3)),
    ]
    simulator._beam_matrix_cache = None
    simulator._distance_matrix_cache = None

    target_beam = simulator.beam_from_to(0, 2)
    simulator.collision_fail_count[0, target_beam] = 5.0
    simulator.discovered_edges.add((1, 2))
    simulator.neighbor_records[1][2] = (simulator.states[2].position.copy(), 3)
    belief_before = float(simulator.belief[0, target_beam])
    success_before = float(simulator.success_count[0, target_beam])

    simulator.exchange_neighbor_and_sensing_tables(0, 1, slot=3)

    assert simulator.belief[0, target_beam] == belief_before
    assert simulator.success_count[0, target_beam] == success_before


def test_single_rf_tx_observes_exactly_one_sensing_cell() -> None:
    cfg = replace(
        load_config("05_simulation/configs/twc_trainable_n10.yaml"),
        n_nodes=2,
        azimuth_cells=8,
        elevation_cells=4,
        sensing_footprint_radius_cells=0,
        sensing_period_slots=1,
    )
    simulator = NeighborDiscoverySimulator(cfg, "improved_rl_isac_tables", seed=71)
    simulator.reset()

    actions = [Action(MODE_TX, 3), Action(MODE_IDLE, 0)]
    simulator.update_action_counts(actions, slot=0)
    simulator.update_sensing(actions, slot=0)

    assert simulator.piggyback_sense_actions == 1
    assert simulator.sensing_observations == 1
    assert simulator.sensing_sectors_for_action(0, Action(MODE_TX, 3), 0).tolist() == [3]


def test_rx_does_not_trigger_piggyback_sensing() -> None:
    cfg = replace(load_config("05_simulation/configs/twc_trainable_n10.yaml"), n_nodes=2)
    simulator = NeighborDiscoverySimulator(cfg, "improved_rl_isac_tables", seed=72)
    simulator.reset()

    actions = [Action(MODE_RX, 0), Action(MODE_IDLE, 0)]
    simulator.update_action_counts(actions, slot=0)
    simulator.update_sensing(actions, slot=0)

    assert simulator.piggyback_sense_actions == 0
    assert simulator.sensing_observations == 0


def test_sensing_range_is_not_clipped_to_communication_range() -> None:
    cfg = replace(
        load_config("05_simulation/configs/twc_trainable_n10.yaml"),
        n_nodes=2,
        azimuth_cells=8,
        elevation_cells=4,
        communication_range_m=100.0,
        sensing_range_m=1000.0,
        isac_sensing_model="constant_error",
        false_alarm_rate=0.0,
        miss_detection_rate=0.0,
        angular_cell_offset_std=0.0,
        sensing_position_error_std_m=0.0,
    )
    simulator = NeighborDiscoverySimulator(cfg, "improved_rl_isac_tables", seed=73)
    simulator.reset()
    simulator.states = [
        NodeState(np.asarray([0.0, 0.0, 0.0]), np.zeros(3)),
        NodeState(np.asarray([500.0, 0.0, 0.0]), np.zeros(3)),
    ]
    simulator.invalidate_geometry_cache()
    target_beam = simulator.beam_from_to(0, 1)

    simulator.update_sensing([Action(MODE_TX, target_beam), Action(MODE_IDLE, 0)], slot=0)

    assert simulator.sensing_target_observations == 1
    assert simulator.belief[0, target_beam] > 0.5


def test_bidirectional_handshake_rejects_one_tx_serving_two_receivers() -> None:
    cfg = replace(
        load_config("05_simulation/configs/twc_trainable_n10.yaml"),
        n_nodes=3,
        azimuth_cells=8,
        elevation_cells=4,
        communication_range_m=1000.0,
    )
    simulator = NeighborDiscoverySimulator(cfg, "improved_rl_isac_tables", seed=74)
    simulator.reset()
    simulator.states = [
        NodeState(np.asarray([0.0, 0.0, 0.0]), np.zeros(3)),
        NodeState(np.asarray([100.0, 0.0, 0.0]), np.zeros(3)),
        NodeState(np.asarray([200.0, 0.0, 0.0]), np.zeros(3)),
    ]
    simulator.invalidate_geometry_cache()
    actions = [
        Action(MODE_TX, simulator.beam_from_to(0, 1)),
        Action(MODE_RX, simulator.beam_from_to(1, 0)),
        Action(MODE_RX, simulator.beam_from_to(2, 0)),
    ]

    discovered = simulator.resolve_discoveries(0, actions, simulator.true_edges(cfg.communication_range_m))

    assert discovered == []
    assert simulator.collision_count == 2
    assert simulator.node_collision_count.tolist() == [2, 1, 1]
