from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from isac_nd_sim.communication_phy import (
    CommunicationPhy,
    beam_solid_angle_sr,
    close_in_path_loss_db,
    db_to_linear,
    free_space_path_loss_db,
    link_received_power_w,
    main_lobe_gain_db,
    sample_rician_power,
    thermal_noise_power_w,
)
from isac_nd_sim.config import load_config
from isac_nd_sim.mobility import NodeState
from isac_nd_sim.simulator import Action, MODE_RX, MODE_TX, NeighborDiscoverySimulator


def _deterministic_phy_config(**overrides):
    cfg = replace(
        load_config("05_simulation/configs/twc_trainable_n10.yaml"),
        n_nodes=3,
        azimuth_cells=24,
        elevation_cells=12,
        communication_fading_enabled=False,
        communication_shadowing_enabled=False,
        communication_shadowing_std_db=0.0,
        communication_sidelobe_gain_db=-30.0,
        communication_sinr_threshold_db=5.0,
    )
    return replace(cfg, **overrides)


def _three_node_aligned_inputs():
    selected_beams = np.asarray([0, 0, 1], dtype=int)
    true_beams = np.zeros((3, 3), dtype=int)
    true_beams[0, 2] = 0
    true_beams[1, 2] = 0
    true_beams[2, 0] = 1
    true_beams[2, 1] = 1
    candidate = np.zeros((3, 3), dtype=bool)
    candidate[0, 2] = True
    candidate[1, 2] = True
    tx_mask = np.asarray([True, True, False])
    rx_mask = np.asarray([False, False, True])
    return selected_beams, true_beams, candidate, tx_mask, rx_mask


def test_close_in_path_loss_is_monotonic_and_uses_friis_reference() -> None:
    carrier = 30.0e9
    fspl_1m = free_space_path_loss_db(1.0, carrier)
    losses = close_in_path_loss_db(np.asarray([1.0, 10.0, 100.0]), carrier, 2.0, 1.0)

    assert losses[0] == fspl_1m
    assert np.all(np.diff(losses) > 0.0)
    assert np.allclose(np.diff(losses), [20.0, 20.0])


def test_directional_gain_and_noise_budget_are_physical() -> None:
    wide = _deterministic_phy_config(azimuth_cells=12, elevation_cells=6)
    narrow = _deterministic_phy_config(azimuth_cells=24, elevation_cells=12)

    assert main_lobe_gain_db(narrow) > main_lobe_gain_db(wide)
    assert main_lobe_gain_db(narrow) > narrow.communication_sidelobe_gain_db
    assert 1e-13 < thermal_noise_power_w(narrow) < 1e-11


def test_normalized_sector_gain_conserves_integrated_realized_gain() -> None:
    cfg = replace(
        _deterministic_phy_config(),
        communication_antenna_gain_mode="normalized_sector",
        communication_sidelobe_gain_db=-10.0,
        communication_antenna_efficiency=0.70,
    )
    solid_angle = beam_solid_angle_sr(cfg)
    main_gain = float(db_to_linear(main_lobe_gain_db(cfg)))
    side_gain = float(db_to_linear(cfg.communication_sidelobe_gain_db))
    integrated_gain = main_gain * solid_angle + side_gain * (4.0 * np.pi - solid_angle)

    np.testing.assert_allclose(integrated_gain, 4.0 * np.pi * cfg.communication_antenna_efficiency)


def test_fixed_main_gain_is_independent_of_codebook_resolution() -> None:
    wide = replace(
        _deterministic_phy_config(azimuth_cells=12, elevation_cells=6),
        communication_antenna_gain_mode="fixed_main_gain",
        communication_fixed_main_lobe_gain_db=21.0,
    )
    narrow = replace(wide, azimuth_cells=36, elevation_cells=18)

    assert main_lobe_gain_db(wide) == 21.0
    assert main_lobe_gain_db(narrow) == 21.0


def test_rician_power_is_normalized_to_unit_mean() -> None:
    values = sample_rician_power(np.random.default_rng(101), (200_000,), 10.0)

    assert abs(float(values.mean()) - 1.0) < 0.01
    assert np.all(values >= 0.0)


def test_isolated_two_phase_link_passes_sinr_threshold() -> None:
    cfg = _deterministic_phy_config(n_nodes=2)
    phy = CommunicationPhy(cfg, np.random.default_rng(102))
    phy.reset(2)
    candidate = np.asarray([[False, True], [False, False]])
    selected = np.asarray([0, 1])
    true_beams = np.asarray([[0, 0], [1, 0]])
    distance = np.asarray([[0.0, 1000.0], [1000.0, 0.0]])

    result = phy.resolve_handshake(
        candidate,
        selected,
        true_beams,
        distance,
        np.asarray([True, False]),
        np.asarray([False, True]),
        slot=0,
    )

    assert result.forward_decoded_matrix[0, 1]
    assert result.success_matrix[0, 1]
    assert result.forward_sinr_db[0, 1] > cfg.communication_sinr_threshold_db
    assert result.ack_sinr_db[0, 1] > cfg.communication_sinr_threshold_db


def test_runtime_received_power_matches_common_link_budget() -> None:
    cfg = _deterministic_phy_config(n_nodes=2)
    phy = CommunicationPhy(cfg, np.random.default_rng(110))
    phy.reset(2)
    selected = np.asarray([0, 1])
    true_beams = np.asarray([[0, 0], [1, 0]])
    distance = np.asarray([[0.0, 1000.0], [1000.0, 0.0]])
    channel = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    received = phy.received_power_matrix(
        selected,
        true_beams,
        distance,
        np.asarray([True, False]),
        np.asarray([False, True]),
        channel,
    )
    main_gain = float(db_to_linear(main_lobe_gain_db(cfg)))
    expected = link_received_power_w(cfg, 1000.0, main_gain, main_gain)

    np.testing.assert_allclose(received[0, 1], expected)


def test_equal_power_aligned_interferers_are_interference_limited() -> None:
    cfg = _deterministic_phy_config()
    phy = CommunicationPhy(cfg, np.random.default_rng(103))
    phy.reset(3)
    selected, true_beams, candidate, tx_mask, rx_mask = _three_node_aligned_inputs()
    distance = np.asarray(
        [[0.0, 1000.0, 1000.0], [1000.0, 0.0, 1000.0], [1000.0, 1000.0, 0.0]]
    )

    result = phy.resolve_handshake(candidate, selected, true_beams, distance, tx_mask, rx_mask, slot=0)

    assert not result.success_matrix.any()
    assert result.forward_interference_failures.sum() == 2
    assert result.forward_outage_failures.sum() == 0


def test_sinr_capture_selects_strong_link_and_completes_ack() -> None:
    cfg = _deterministic_phy_config()
    phy = CommunicationPhy(cfg, np.random.default_rng(104))
    phy.reset(3)
    selected, true_beams, candidate, tx_mask, rx_mask = _three_node_aligned_inputs()
    distance = np.asarray(
        [[0.0, 1000.0, 100.0], [1000.0, 0.0, 1000.0], [100.0, 1000.0, 0.0]]
    )

    result = phy.resolve_handshake(candidate, selected, true_beams, distance, tx_mask, rx_mask, slot=0)

    assert result.success_matrix[0, 2]
    assert not result.success_matrix[1, 2]
    assert result.forward_interference_failures[1, 2]


def test_seeded_shadowing_and_fading_are_reproducible() -> None:
    cfg = replace(
        _deterministic_phy_config(),
        communication_fading_enabled=True,
        communication_shadowing_enabled=True,
        communication_shadowing_std_db=2.0,
    )
    first = CommunicationPhy(cfg, np.random.default_rng(105))
    second = CommunicationPhy(cfg, np.random.default_rng(105))
    first.reset(3)
    second.reset(3)

    assert np.array_equal(first._shadowing_db, second._shadowing_db)
    assert np.array_equal(first.channel_power(3, slot=7), second.channel_power(3, slot=7))


def test_simulator_records_phy_outage_when_threshold_is_unreachable() -> None:
    cfg = _deterministic_phy_config(
        n_nodes=2,
        communication_sinr_threshold_db=100.0,
        communication_range_m=2000.0,
    )
    simulator = NeighborDiscoverySimulator(cfg, "improved_rl_isac_tables", seed=106)
    simulator.reset()
    simulator.states = [
        NodeState(np.asarray([0.0, 0.0, 0.0]), np.zeros(3)),
        NodeState(np.asarray([1000.0, 0.0, 0.0]), np.zeros(3)),
    ]
    simulator.invalidate_geometry_cache()
    actions = [
        Action(MODE_TX, simulator.beam_from_to(0, 1)),
        Action(MODE_RX, simulator.beam_from_to(1, 0)),
    ]

    discovered = simulator.resolve_discoveries(0, actions, {(0, 1)})

    assert discovered == []
    assert simulator.handshake_attempts == 1
    assert simulator.role_compatible_pairs == 1
    assert simulator.aligned_handshake_opportunities == 1
    assert simulator.forward_decodes == 0
    assert simulator.forward_decode_failures == 1
    assert simulator.phy_outage_failures == 1
    assert simulator.interference_limited_failures == 0
    assert simulator.handshake_attempts == (
        simulator.handshake_successes
        + simulator.forward_decode_failures
        + simulator.ack_decode_failures
    )


def test_sinr_phy_rejects_unsupported_multiple_rf_chains() -> None:
    cfg = _deterministic_phy_config(rf_chains=2)

    with pytest.raises(ValueError, match="exactly one RF chain"):
        NeighborDiscoverySimulator(cfg, "improved_rl_isac_tables", seed=109)


def test_policy_rng_does_not_perturb_communication_channel() -> None:
    cfg = replace(
        _deterministic_phy_config(n_nodes=2),
        communication_fading_enabled=True,
        communication_shadowing_enabled=True,
        communication_shadowing_std_db=2.0,
    )
    first = NeighborDiscoverySimulator(cfg, "improved_rl_isac_tables", seed=107, scenario_seed=108)
    second = NeighborDiscoverySimulator(cfg, "improved_rl_isac_tables", seed=999, scenario_seed=108)
    first.reset()
    second.reset()
    second.rng.random(100)
    first.states = [
        NodeState(np.asarray([0.0, 0.0, 0.0]), np.zeros(3)),
        NodeState(np.asarray([1000.0, 0.0, 0.0]), np.zeros(3)),
    ]
    second.states = [state for state in first.states]
    first.invalidate_geometry_cache()
    second.invalidate_geometry_cache()
    actions = [
        Action(MODE_TX, first.beam_from_to(0, 1)),
        Action(MODE_RX, first.beam_from_to(1, 0)),
    ]

    first_result = first.resolve_discoveries(0, actions, {(0, 1)})
    second_result = second.resolve_discoveries(0, actions, {(0, 1)})

    assert first_result == second_result
    assert first.handshake_sinr_samples_db == second.handshake_sinr_samples_db
