from __future__ import annotations

import importlib.util
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from isac_nd_sim.config import load_config, with_communication_tx_power
from isac_nd_sim.marl_env import MarlNeighborDiscoveryEnv
from isac_nd_sim.mobility import NodeState
from isac_nd_sim.neural_contention_actor_critic import ContentionGraphActorCritic
from isac_nd_sim.simulator import Action


ROOT = Path(__file__).resolve().parents[2]
TRAINING_SCRIPT = ROOT / "05_simulation" / "run_marl_training.py"
EVALUATION_SCRIPT = ROOT / "05_simulation" / "run_marl_evaluate.py"


def load_training_module():
    spec = importlib.util.spec_from_file_location("twc_training_contract", TRAINING_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_evaluation_module():
    spec = importlib.util.spec_from_file_location("twc_evaluation_contract", EVALUATION_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_isac_mappo_defaults_to_soft_isac_observation_without_rule_priors() -> None:
    module = load_training_module()
    args = Namespace(
        algorithm="isac_mappo",
        disable_isac_features=False,
        candidate_mask=False,
        candidate_score=None,
        topology_deficit=False,
        rule_residual=False,
    )

    assert module.resolved_feature_flags(args) == {
        "candidate_mask": False,
        "candidate_score": True,
        "topology_deficit": False,
        "rule_residual": False,
    }
    assert module.contention_mode_prior_enabled(Namespace(contention_mode_prior=False)) is False
    assert module.disabled_modes_from_args(
        Namespace(allow_standalone_sense=False, allow_idle=False, forbid_sense=False)
    ) == ("sense", "idle")


def test_clean_ctde_profile_disables_action_teacher_features() -> None:
    module = load_training_module()
    args = Namespace(
        clean_ctde=True,
        algorithm="isac_mappo",
        network="contention_shared",
        candidate_source="default",
        candidate_mask=True,
        candidate_score=None,
        topology_deficit=True,
        rule_residual=False,
        contention_mode_prior=False,
        rendezvous_adapter=False,
        rendezvous_observation=None,
        expert_bc_weight=0.0,
        beam_rank_aux_coef=0.0,
        rendezvous_beam_aux_coef=0.0,
        rendezvous_role_aux_coef=0.0,
    )

    assert module.clean_ctde_violations(args) == []
    assert module.resolved_feature_flags(args) == {
        "candidate_mask": True,
        "candidate_score": True,
        "topology_deficit": True,
        "rule_residual": False,
    }
    assert module.contention_mode_prior_enabled(args) is False


def test_clean_ctde_rejects_rule_guided_action_targets() -> None:
    module = load_training_module()
    args = Namespace(
        clean_ctde=True,
        algorithm="isac_mappo",
        network="contention_shared",
        candidate_source="wang_table",
        candidate_mask=True,
        candidate_score=True,
        rule_residual=True,
        contention_mode_prior=True,
        rendezvous_adapter=True,
        rendezvous_observation=True,
        expert_bc_weight=0.1,
        beam_rank_aux_coef=0.1,
        rendezvous_beam_aux_coef=0.1,
        rendezvous_role_aux_coef=0.1,
    )

    violations = module.clean_ctde_violations(args)
    assert "rule residual" in violations
    assert "contention mode prior" in violations
    assert "rendezvous adapter" in violations
    assert "rendezvous observation" in violations
    assert "behavior cloning" in violations
    assert "beam-ranking action target" in violations
    assert "rendezvous beam target" in violations
    assert "rendezvous role target" in violations


def test_exchanged_neighbor_table_updates_local_actor_candidate_features() -> None:
    cfg = replace(
        load_config("05_simulation/configs/twc_trainable_n10.yaml"),
        n_nodes=3,
        azimuth_cells=8,
        elevation_cells=1,
        belief_update_rho=0.6,
    )
    env = MarlNeighborDiscoveryEnv(cfg, protocol="improved_rl_isac_tables")
    env.reset(seed=20260710)
    env._sim.states = [
        NodeState(np.asarray([0.0, 0.0, 0.0]), np.zeros(3)),
        NodeState(np.asarray([100.0, 0.0, 0.0]), np.zeros(3)),
        NodeState(np.asarray([0.0, 100.0, 0.0]), np.zeros(3)),
    ]
    env._sim.invalidate_geometry_cache()
    env._slot = 3
    target_beam = env._sim.beam_from_to(0, 2)
    before = env._observation_for(0)
    env._sim.neighbor_records[1][2] = (env._sim.states[2].position.copy(), 3)

    env._sim.exchange_neighbor_and_sensing_tables(0, 1, slot=3)
    after = env._observation_for(0)

    assert after["beam_belief"][target_beam] > before["beam_belief"][target_beam]
    assert after["candidate_score"][target_beam] > before["candidate_score"][target_beam]
    assert env._sim.last_positive_slot[0, target_beam] == 3


def test_local_candidate_mask_keeps_unknown_cells_and_reopens_stale_empty_cells() -> None:
    cfg = replace(
        load_config("05_simulation/configs/twc_planar_n10_b15_random20.yaml"),
        n_nodes=2,
        azimuth_cells=8,
        elevation_cells=1,
    )
    env = MarlNeighborDiscoveryEnv(cfg, protocol="improved_rl_isac_tables")
    observations, _ = env.reset(seed=20260711)
    empty_beam = 3

    assert np.all(observations[0]["candidate_mask"] == 1.0)

    env._sim.empty_beam_count[0, empty_beam] = 1.0
    env._sim.belief[0, empty_beam] = 0.0
    env._sim.success_count[0, empty_beam] = 0.0
    env._sim.age[0, empty_beam] = 0.0
    rejected = env._observation_for(0)

    assert rejected["candidate_mask"][empty_beam] == 0.0
    assert np.all(np.delete(rejected["candidate_mask"], empty_beam) == 1.0)

    env._sim.age[0, empty_beam] = 50.0
    reopened = env._observation_for(0)

    assert reopened["candidate_mask"][empty_beam] == 1.0


def test_clean_ctde_forces_rendezvous_observation_off() -> None:
    module = load_training_module()
    cfg = load_config("05_simulation/configs/twc_canonical_n10_b10.yaml")
    args = Namespace(
        episodes=1,
        slots=300,
        seed=20260710,
        node_count=None,
        azimuth_cells=None,
        elevation_cells=None,
        communication_range=None,
        sensing_range=None,
        false_alarm_rate=None,
        miss_detection_rate=None,
        angular_cell_offset_std=None,
        sensing_period_slots=None,
        area_size_m=None,
        mobility_model=None,
        spatial_dimensions=None,
        rendezvous_observation=None,
        clean_ctde=True,
    )

    resolved = module.override_config(cfg, args)

    assert not resolved.rendezvous_observation_enabled


def test_canonical_config_uses_b10_shared_power_and_sinr_phy() -> None:
    cfg = load_config("05_simulation/configs/twc_canonical_n10_b10.yaml")

    assert (cfg.azimuth_cells, cfg.elevation_cells) == (36, 18)
    assert cfg.slots_per_episode == 300
    assert cfg.communication_phy_model == "close_in_rician_sinr"
    assert cfg.communication_antenna_gain_mode == "normalized_sector"
    assert cfg.shared_waveform_power_enabled
    assert cfg.tx_power_w == cfg.isac_tx_power_w == cfg.communication_tx_power_w == 1.0

    low_power = with_communication_tx_power(cfg, 0.25)
    assert low_power.tx_power_w == low_power.isac_tx_power_w == low_power.communication_tx_power_w == 0.25


def test_rendezvous_observation_can_be_ablated_without_changing_yaml() -> None:
    cfg = load_config("05_simulation/configs/twc_canonical_n10_b10.yaml")
    common_args = {
        "slots": 300,
        "seed": 20260705,
        "node_count": None,
        "azimuth_cells": None,
        "elevation_cells": None,
        "communication_range": None,
        "sensing_range": None,
        "false_alarm_rate": None,
        "miss_detection_rate": None,
        "angular_cell_offset_std": None,
        "sensing_period_slots": None,
        "mobility_model": None,
        "rendezvous_observation": False,
    }

    training_cfg = load_training_module().override_config(cfg, Namespace(episodes=1, **common_args))
    evaluation_cfg = load_evaluation_module().override_config(
        cfg,
        Namespace(eval_episodes=1, area_size_m=None, **common_args),
    )

    assert not training_cfg.rendezvous_observation_enabled
    assert not evaluation_cfg.rendezvous_observation_enabled


def test_rendezvous_action_diagnostics_separate_beam_and_mode_learning() -> None:
    module = load_training_module()
    observation = {
        "candidate_score": np.asarray([0.1, 1.0, 0.2], dtype=np.float32),
        "candidate_mask": np.ones(3, dtype=np.float32),
        "rendezvous_beam_score": np.asarray([0.0, 0.8, 0.0], dtype=np.float32),
        "rendezvous_role_hint": np.asarray([1.0], dtype=np.float32),
    }

    beam_only = module.beam_selection_diagnostics([observation], [Action("rx", 1)])
    mode_only = module.beam_selection_diagnostics([observation], [Action("tx", 2)])
    joint = module.beam_selection_diagnostics([observation], [Action("tx", 1)])

    assert beam_only["rendezvous_beam_hit_count"] == 1
    assert beam_only["rendezvous_mode_match_count"] == 0
    assert mode_only["rendezvous_beam_hit_count"] == 0
    assert mode_only["rendezvous_mode_match_count"] == 1
    assert joint["rendezvous_joint_action_rate"] == 1.0


def test_rendezvous_auxiliary_losses_use_only_exposed_local_targets() -> None:
    torch = pytest.importorskip("torch")
    module = load_training_module()
    cfg = replace(
        load_config("05_simulation/configs/twc_canonical_n10_b10.yaml"),
        n_nodes=2,
        azimuth_cells=4,
        elevation_cells=2,
        sensing_position_error_std_m=0.0,
    )
    env = MarlNeighborDiscoveryEnv(cfg)
    env.reset(seed=903)
    env._sim.states = [
        NodeState(np.asarray([100.0, 200.0, 300.0]), np.zeros(3)),
        NodeState(np.asarray([900.0, 800.0, 700.0]), np.zeros(3)),
    ]
    env._sim.invalidate_geometry_cache()
    estimate = env._sim.states[1].position.copy()
    env._sim.sensing_report_position[0, 0] = estimate
    env._sim.sensing_report_confidence[0, 0] = 1.0
    env._sim.sensing_report_slot[0, 0] = 200
    env._slot = env._sim.position_pair_rendezvous_phase(env._sim.states[0].position, estimate, 16) + 208
    observation = env._observation_for(0)
    policy = ContentionGraphActorCritic(
        cfg.n_beams,
        hidden_dim=16,
        use_candidate_score=True,
        disabled_modes=("sense", "idle"),
    )

    beam_loss, role_loss = module.rendezvous_auxiliary_losses(policy, [[observation]], torch)

    assert torch.isfinite(beam_loss) and float(beam_loss.item()) > 0.0
    assert torch.isfinite(role_loss) and float(role_loss.item()) > 0.0


def test_zero_initialized_rendezvous_adapter_is_an_exact_behavioral_ablation() -> None:
    torch = pytest.importorskip("torch")
    cfg = replace(
        load_config("05_simulation/configs/twc_canonical_n10_b10.yaml"),
        n_nodes=2,
        azimuth_cells=4,
        elevation_cells=2,
        sensing_position_error_std_m=0.0,
    )
    env = MarlNeighborDiscoveryEnv(cfg)
    env.reset(seed=904)
    env._sim.states = [
        NodeState(np.asarray([100.0, 200.0, 300.0]), np.zeros(3)),
        NodeState(np.asarray([900.0, 800.0, 700.0]), np.zeros(3)),
    ]
    env._sim.invalidate_geometry_cache()
    estimate = env._sim.states[1].position.copy()
    env._sim.sensing_report_position[0, 0] = estimate
    env._sim.sensing_report_confidence[0, 0] = 1.0
    env._sim.sensing_report_slot[0, 0] = 200
    env._slot = env._sim.position_pair_rendezvous_phase(env._sim.states[0].position, estimate, 16) + 208
    observation = env._observation_for(0)

    torch.manual_seed(905)
    base = ContentionGraphActorCritic(cfg.n_beams, hidden_dim=16, use_candidate_score=True)
    torch.manual_seed(905)
    adapted = ContentionGraphActorCritic(
        cfg.n_beams,
        hidden_dim=16,
        use_candidate_score=True,
        use_rendezvous_adapter=True,
    )
    with torch.no_grad():
        base_mode, base_beam, _ = base.logits_value(observation)
        initial_mode, initial_beam, _ = adapted.logits_value(observation)
        adapted.model.rendezvous_beam_adapter.weight[0, 0] = 1.0
        learned_mode, learned_beam, _ = adapted.logits_value(observation)

    target_beam = int(np.argmax(observation["rendezvous_beam_score"]))
    expected_delta = np.log(cfg.n_beams) * float(observation["rendezvous_beam_score"][target_beam])
    assert torch.allclose(base_mode, initial_mode)
    assert torch.allclose(base_beam, initial_beam)
    assert float((learned_beam[target_beam] - initial_beam[target_beam]).item()) == pytest.approx(expected_delta)
    assert torch.allclose(learned_mode, initial_mode)


def test_reciprocal_rendezvous_diagnostics_are_offline_only() -> None:
    module = load_training_module()
    cfg = replace(
        load_config("05_simulation/configs/twc_canonical_n10_b10.yaml"),
        n_nodes=2,
        azimuth_cells=4,
        elevation_cells=2,
        sensing_position_error_std_m=0.0,
    )
    env = MarlNeighborDiscoveryEnv(cfg)
    env.reset(seed=906)
    env._sim.states = [
        NodeState(np.asarray([100.0, 200.0, 300.0]), np.zeros(3)),
        NodeState(np.asarray([900.0, 800.0, 700.0]), np.zeros(3)),
    ]
    env._sim.invalidate_geometry_cache()
    beam_01 = env._sim.beam_from_to(0, 1)
    beam_10 = env._sim.beam_from_to(1, 0)
    env._sim.sensing_report_position[0, beam_01] = env._sim.states[1].position
    env._sim.sensing_report_position[1, beam_10] = env._sim.states[0].position
    env._sim.sensing_report_confidence[0, beam_01] = 1.0
    env._sim.sensing_report_confidence[1, beam_10] = 1.0
    env._sim.sensing_report_slot[0, beam_01] = 200
    env._sim.sensing_report_slot[1, beam_10] = 200
    env._slot = env._sim.position_pair_rendezvous_phase(
        env._sim.states[0].position,
        env._sim.states[1].position,
        16,
    ) + 208
    observations = env._observations()

    diagnostics = module.rendezvous_pair_diagnostics(
        env,
        observations,
        [Action("tx", beam_01), Action("rx", beam_10)],
    )

    assert diagnostics == {
        "reciprocal_report_pair_count": 1,
        "reciprocal_scheduled_pair_count": 1,
        "reciprocal_actor_pair_count": 1,
    }


def test_marl_observation_reprojects_local_rendezvous_report() -> None:
    cfg = replace(
        load_config("05_simulation/configs/twc_canonical_n10_b10.yaml"),
        n_nodes=2,
        azimuth_cells=8,
        elevation_cells=4,
    )
    env = MarlNeighborDiscoveryEnv(cfg)
    env.reset(seed=902)
    env._sim.states = [
        NodeState(np.asarray([100.0, 200.0, 300.0]), np.zeros(3), yaw=np.pi / 2.0),
        NodeState(np.asarray([900.0, 800.0, 700.0]), np.zeros(3)),
    ]
    env._sim.invalidate_geometry_cache()
    source_beam = 0
    estimate = env._sim.states[1].position.copy()
    env._sim.sensing_report_position[0, source_beam] = estimate
    env._sim.sensing_report_confidence[0, source_beam] = 1.0
    env._sim.sensing_report_slot[0, source_beam] = 200
    phase = env._sim.position_pair_rendezvous_phase(env._sim.states[0].position, estimate, 16)
    env._slot = phase + 208

    observation = env._observation_for(0)
    current_beam = env._sim.beam_from_to(0, 1)

    assert observation["rendezvous_beam_score"][current_beam] > 0.0
    assert observation["rendezvous_beam_role"][current_beam] == 1.0
    assert observation["candidate_score"][current_beam] > 0.0
    assert observation["rendezvous_role_hint"].tolist() == [1.0]
    assert observation["local_summary"][3] == 1.0


def test_disabled_structured_features_do_not_leak_through_aggregate_paths() -> None:
    torch = pytest.importorskip("torch")
    cfg = replace(load_config("05_simulation/configs/twc_trainable_n10.yaml"), n_nodes=1, azimuth_cells=4, elevation_cells=2)
    env = MarlNeighborDiscoveryEnv(cfg)
    observations, _ = env.reset(seed=91)
    base = observations[0]
    changed = {key: value.copy() if hasattr(value, "copy") else value for key, value in base.items()}
    changed["candidate_mask"][:] = 1.0 - changed["candidate_mask"]
    changed["candidate_score"][:] = np.linspace(0.0, 1.0, cfg.n_beams, dtype=np.float32)
    changed["rendezvous_beam_score"][:] = np.linspace(1.0, 0.0, cfg.n_beams, dtype=np.float32)
    changed["rendezvous_beam_role"][:] = 1.0
    changed["rendezvous_role_hint"][:] = 1.0
    changed["local_summary"][3] = 1.0
    changed["topology_deficit"][:] = 1.0
    changed["rule_mode_logits"][:] = np.asarray([9.0, -9.0, 7.0, -7.0], dtype=np.float32)
    changed["contention_state"][1] = 1.0
    changed["contention_state"][6:9] = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    policy = ContentionGraphActorCritic(
        cfg.n_beams,
        hidden_dim=16,
        use_candidate_mask=False,
        use_candidate_score=False,
        use_topology_deficit=False,
        use_rule_residual=False,
        use_contention_mode_prior=False,
    )

    with torch.no_grad():
        base_mode, base_beam, _ = policy.logits_value(base)
        changed_mode, changed_beam, _ = policy.logits_value(changed)

    assert torch.allclose(base_mode, changed_mode)
    assert torch.allclose(base_beam, changed_beam)


def test_training_evaluation_restores_torch_and_numpy_rng_state() -> None:
    torch = pytest.importorskip("torch")
    module = load_training_module()
    cfg = replace(
        load_config("05_simulation/configs/twc_trainable_n10.yaml"),
        n_nodes=2,
        azimuth_cells=4,
        elevation_cells=2,
        slots_per_episode=1,
    )
    policy = ContentionGraphActorCritic(cfg.n_beams, hidden_dim=16, disabled_modes=("sense",))
    args = Namespace(
        eval_both=False,
        eval_episodes=1,
        slots=1,
        reward_version="discovery_first",
        candidate_source="default",
        algorithm="isac_mappo",
    )
    torch.manual_seed(92)
    np.random.seed(92)
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = np.random.get_state()

    module.evaluate_policy(cfg, policy, torch, args, "improved_rl_isac_tables", 1, 9000, True)

    assert torch.equal(torch_before, torch.random.get_rng_state())
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]


def test_marl_step_invalidates_geometry_cache_after_mobility() -> None:
    cfg = replace(
        load_config("05_simulation/configs/twc_trainable_n10.yaml"),
        n_nodes=2,
        azimuth_cells=4,
        elevation_cells=2,
        slots_per_episode=2,
    )
    env = MarlNeighborDiscoveryEnv(cfg)
    env.reset(seed=93)
    env._sim.distance_matrix()
    assert env._sim._distance_matrix_cache is not None

    env.step([{"mode": "idle"}, {"mode": "idle"}])

    assert env._sim._distance_matrix_cache is None
    assert env._sim._beam_matrix_cache is None
