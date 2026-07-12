from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from isac_nd_sim.config import load_config, with_communication_tx_power
from isac_nd_sim.marl_env import MODE_NAMES, MarlNeighborDiscoveryEnv
from isac_nd_sim.mobility import NodeState
from isac_nd_sim.centralized_graph_critic import CentralizedGraphCritic
from isac_nd_sim.neural_contention_actor_critic import ContentionGraphActorCritic
from isac_nd_sim.neural_recurrent_contention_actor_critic import (
    RecurrentContentionGraphActorCritic,
)
from isac_nd_sim.simulator import Action


ROOT = Path(__file__).resolve().parents[2]
TRAINING_SCRIPT = ROOT / "05_simulation" / "run_marl_training.py"
EVALUATION_SCRIPT = ROOT / "05_simulation" / "run_marl_evaluate.py"
BEAM_CHECKPOINT_EVALUATION_SCRIPT = ROOT / "05_simulation" / "evaluate_beam_only_checkpoint.py"


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


def load_beam_checkpoint_evaluation_module():
    simulation_dir = str(BEAM_CHECKPOINT_EVALUATION_SCRIPT.parent)
    if simulation_dir not in sys.path:
        sys.path.insert(0, simulation_dir)
    spec = importlib.util.spec_from_file_location(
        "beam_checkpoint_evaluation_contract", BEAM_CHECKPOINT_EVALUATION_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_beam_only_mappo_has_no_mode_head_and_uses_fixed_half_roles() -> None:
    torch = pytest.importorskip("torch")
    cfg = replace(load_config("05_simulation/configs/mvp.yaml"), n_nodes=4)
    env = MarlNeighborDiscoveryEnv(cfg)
    observations, _ = env.reset(seed=20260801)
    policy = ContentionGraphActorCritic(
        cfg.n_beams,
        hidden_dim=16,
        use_candidate_mask=True,
        use_candidate_score=True,
        action_contract="beam_only_fixed_role",
        disabled_modes=("sense", "idle"),
    )
    role_rng = np.random.default_rng(91)

    step = policy.act(observations, deterministic=False, role_rng=role_rng)

    assert not hasattr(policy.model, "mode_head")
    assert not hasattr(policy.model, "contention_mode_residual")
    assert all(action.mode in {"tx", "rx"} for action in step.actions)
    assert torch.count_nonzero(step.mode_log_probs) == 0
    assert torch.count_nonzero(step.mode_entropies) == 0
    assert torch.allclose(step.log_probs, step.beam_log_probs)
    assert torch.allclose(step.entropies, step.beam_entropies)


def test_beam_only_mappo_role_sequence_is_independent_of_beam_sampling() -> None:
    torch = pytest.importorskip("torch")
    cfg = replace(load_config("05_simulation/configs/mvp.yaml"), n_nodes=4)
    env = MarlNeighborDiscoveryEnv(cfg)
    observations, _ = env.reset(seed=20260802)
    policy = ContentionGraphActorCritic(
        cfg.n_beams,
        hidden_dim=16,
        action_contract="beam_only_fixed_role",
        disabled_modes=("sense", "idle"),
    )
    first_roles: list[str] = []
    second_roles: list[str] = []
    first_rng = np.random.default_rng(92)
    second_rng = np.random.default_rng(92)
    for index in range(50):
        torch.manual_seed(1000 + index)
        first_roles.extend(
            action.mode for action in policy.act(observations, role_rng=first_rng).actions
        )
        torch.manual_seed(2000 + index)
        second_roles.extend(
            action.mode for action in policy.act(observations, role_rng=second_rng).actions
        )

    assert first_roles == second_roles
    assert first_roles.count("tx") / len(first_roles) == pytest.approx(0.5, abs=0.1)


def test_recurrent_joint_actor_starts_at_half_roles_and_replays_joint_log_prob() -> None:
    torch = pytest.importorskip("torch")
    cfg = replace(load_config("05_simulation/configs/mvp.yaml"), n_nodes=4)
    env = MarlNeighborDiscoveryEnv(cfg)
    observations, _ = env.reset(seed=20260826)
    policy = RecurrentContentionGraphActorCritic(
        cfg.n_beams,
        hidden_dim=16,
        action_contract="joint_role_beam",
        disabled_modes=("sense", "idle"),
        use_candidate_mask=True,
        use_candidate_score=True,
        use_candidate_score_prior=True,
        azimuth_cells=cfg.azimuth_cells,
        elevation_cells=cfg.elevation_cells,
    )
    with torch.no_grad():
        mode_logits, _beam_logits, _values = policy.batched_logits_value(
            observations,
            hard_mask=True,
        )
        mode_probabilities = torch.softmax(mode_logits, dim=-1)
    tx_index = MODE_NAMES.index("tx")
    rx_index = MODE_NAMES.index("rx")
    assert torch.allclose(mode_probabilities[:, tx_index], torch.full((4,), 0.5))
    assert torch.allclose(mode_probabilities[:, rx_index], torch.full((4,), 0.5))

    policy.reset_recurrent_state(cfg.n_nodes)
    step = policy.act(observations, deterministic=False)
    replay = policy.evaluate_action_sequence([observations], [step.actions])

    assert all(action.mode in {"tx", "rx"} for action in step.actions)
    assert torch.count_nonzero(step.mode_log_probs) == cfg.n_nodes
    assert torch.allclose(step.log_probs, step.mode_log_probs + step.beam_log_probs)
    assert torch.allclose(replay["log_probs"][0], step.log_probs, atol=1e-6, rtol=1e-6)


def test_decoupled_role_and_beam_towers_have_disjoint_gradients() -> None:
    torch = pytest.importorskip("torch")
    cfg = replace(load_config("05_simulation/configs/mvp.yaml"), n_nodes=4)
    env = MarlNeighborDiscoveryEnv(cfg)
    observations, _ = env.reset(seed=20260829)
    policy = RecurrentContentionGraphActorCritic(
        cfg.n_beams,
        hidden_dim=16,
        action_contract="joint_role_beam",
        disabled_modes=("sense", "idle"),
        use_candidate_mask=True,
        use_candidate_score=True,
        use_candidate_score_prior=True,
        use_decoupled_role_tower=True,
        azimuth_cells=cfg.azimuth_cells,
        elevation_cells=cfg.elevation_cells,
    )

    mode_logits, _beam_logits, _values = policy.batched_logits_value(observations, hard_mask=True)
    mode_logits[:, MODE_NAMES.index("tx")].sum().backward()
    assert policy.model.mode_head.weight.grad is not None
    assert policy.model.beam_query.weight.grad is None

    policy.model.zero_grad(set_to_none=True)
    _mode_logits, beam_logits, _values = policy.batched_logits_value(observations, hard_mask=True)
    beam_logits[torch.isfinite(beam_logits)].sum().backward()
    assert policy.model.beam_query.weight.grad is not None
    assert policy.model.mode_head.weight.grad is None
    assert all(parameter.grad is None for parameter in policy.model.role_encoder.parameters())


def test_beam_only_mappo_ppo_recompute_uses_only_beam_probability() -> None:
    pytest.importorskip("torch")
    module = load_training_module()
    cfg = replace(load_config("05_simulation/configs/mvp.yaml"), n_nodes=4)
    env = MarlNeighborDiscoveryEnv(cfg)
    observations, _ = env.reset(seed=20260803)
    policy = ContentionGraphActorCritic(
        cfg.n_beams,
        hidden_dim=16,
        use_candidate_mask=True,
        action_contract="beam_only_fixed_role",
        disabled_modes=("sense", "idle"),
    )
    step = policy.act(
        observations,
        deterministic=False,
        role_rng=np.random.default_rng(93),
    )

    recomputed = module.evaluate_action_components(
        policy,
        [observations],
        [step.actions],
    )

    assert np.allclose(
        recomputed["log_probs"].detach().cpu().numpy()[0],
        step.log_probs.detach().cpu().numpy(),
    )
    assert np.allclose(
        recomputed["beam_log_probs"].detach().cpu().numpy()[0],
        step.beam_log_probs.detach().cpu().numpy(),
    )
    assert np.count_nonzero(recomputed["mode_log_probs"].detach().cpu().numpy()) == 0
    assert np.count_nonzero(recomputed["mode_entropies"].detach().cpu().numpy()) == 0


def test_recurrent_beam_actor_replays_rollout_log_probs_from_zero_state() -> None:
    torch = pytest.importorskip("torch")
    module = load_training_module()
    cfg = replace(
        load_config("05_simulation/configs/twc_planar_n10_b15_random20.yaml"),
        n_nodes=4,
        azimuth_cells=8,
        elevation_cells=1,
        slots_per_episode=5,
    )
    env = MarlNeighborDiscoveryEnv(
        cfg,
        protocol="improved_rl_isac_tables",
        candidate_source="residual_table",
    )
    observations, _ = env.reset(seed=20260820)
    policy = RecurrentContentionGraphActorCritic(
        cfg.n_beams,
        hidden_dim=16,
        use_candidate_mask=True,
        use_candidate_score=True,
        use_topology_deficit=True,
        use_residual_measurement_features=True,
        disabled_modes=("sense", "idle"),
    )
    policy.reset_recurrent_state(cfg.n_nodes)
    role_rng = np.random.default_rng(20260839)
    observations_by_step = []
    actions_by_step = []
    old_log_probs = []
    for _ in range(cfg.slots_per_episode):
        observations_by_step.append(module.copy_observations(observations))
        with torch.no_grad():
            step = policy.act(observations, role_rng=role_rng)
        old_log_probs.append(step.log_probs.detach())
        actions_by_step.append(step.actions)
        observations, _reward, _terminated, _truncated, _info = env.step(step.actions)

    state_before_replay = policy.clone_recurrent_state()
    replay = policy.evaluate_action_sequence(observations_by_step, actions_by_step)

    assert torch.allclose(replay["log_probs"], torch.stack(old_log_probs), atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(replay["mode_log_probs"]) == 0
    assert torch.count_nonzero(replay["mode_entropies"]) == 0
    assert torch.allclose(policy.recurrent_state, state_before_replay)


def test_recurrent_beam_actor_reset_prevents_episode_state_leakage() -> None:
    torch = pytest.importorskip("torch")
    cfg = replace(load_config("05_simulation/configs/mvp.yaml"), n_nodes=3, azimuth_cells=8, elevation_cells=1)
    env = MarlNeighborDiscoveryEnv(cfg)
    observations, _ = env.reset(seed=20260821)
    policy = RecurrentContentionGraphActorCritic(
        cfg.n_beams,
        hidden_dim=16,
        use_candidate_mask=True,
        disabled_modes=("sense", "idle"),
    )

    policy.reset_recurrent_state(cfg.n_nodes)
    with torch.no_grad():
        first = policy.advance_recurrent_logits(observations)[1].clone()
        second = policy.advance_recurrent_logits(observations)[1].clone()
    policy.reset_recurrent_state(cfg.n_nodes)
    with torch.no_grad():
        repeated_first = policy.advance_recurrent_logits(observations)[1].clone()

    assert torch.allclose(first, repeated_first)
    assert not torch.allclose(first, second)


def test_recurrent_complementary_role_sanity_contract_replays_beam_probability() -> None:
    torch = pytest.importorskip("torch")
    cfg = replace(
        load_config("05_simulation/configs/mvp.yaml"),
        n_nodes=2,
        azimuth_cells=8,
        elevation_cells=1,
    )
    env = MarlNeighborDiscoveryEnv(cfg)
    observations, _ = env.reset(seed=20260841)
    policy = RecurrentContentionGraphActorCritic(
        cfg.n_beams,
        hidden_dim=16,
        action_contract="beam_only_complementary_role",
        azimuth_cells=cfg.azimuth_cells,
        elevation_cells=cfg.elevation_cells,
        disabled_modes=("sense", "idle"),
    )

    policy.reset_recurrent_state(cfg.n_nodes)
    with torch.no_grad():
        step = policy.act(observations, deterministic=True)
    replay = policy.evaluate_action_sequence([observations], [step.actions])

    assert [action.mode for action in step.actions] == ["tx", "rx"]
    assert torch.count_nonzero(step.mode_log_probs) == 0
    assert torch.allclose(replay["log_probs"][0], step.log_probs, atol=1e-6, rtol=1e-6)
    assert policy.model.beam_center_directions.shape == (cfg.n_beams, 3)


def test_beam_conditioned_role_factorization_replays_selected_beam_probability() -> None:
    torch = pytest.importorskip("torch")
    cfg = replace(
        load_config("05_simulation/configs/mvp.yaml"),
        n_nodes=2,
        azimuth_cells=8,
        elevation_cells=1,
    )
    env = MarlNeighborDiscoveryEnv(cfg)
    observations, _ = env.reset(seed=20260842)
    policy = RecurrentContentionGraphActorCritic(
        cfg.n_beams,
        hidden_dim=16,
        action_contract="joint_role_beam",
        role_factorization="beam_conditioned",
        use_decoupled_role_tower=True,
        azimuth_cells=cfg.azimuth_cells,
        elevation_cells=cfg.elevation_cells,
        disabled_modes=("sense", "idle"),
    )

    conditional_logits, beam_logits, _values, _state = policy._step_from_state(
        observations,
        policy._zero_state(cfg.n_nodes),
        hard_mask=True,
    )
    assert conditional_logits.shape == (cfg.n_nodes, cfg.n_beams, len(MODE_NAMES))
    assert torch.allclose(
        torch.softmax(conditional_logits, dim=-1)[..., MODE_NAMES.index("tx")],
        torch.full((cfg.n_nodes, cfg.n_beams), 0.5),
    )

    policy.reset_recurrent_state(cfg.n_nodes)
    with torch.no_grad():
        step = policy.act(observations, deterministic=False)
    replay = policy.evaluate_action_sequence([observations], [step.actions])
    assert torch.allclose(replay["log_probs"][0], step.log_probs, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        replay["mode_tx_probabilities"][0],
        torch.full((cfg.n_nodes,), 0.5),
        atol=1e-6,
    )

    with torch.no_grad():
        policy.model.mode_head.weight.normal_(0.0, 0.1)
    policy.model.zero_grad(set_to_none=True)
    conditional_logits, _beam_logits, _values, _state = policy._step_from_state(
        observations,
        policy._zero_state(cfg.n_nodes),
        hard_mask=True,
    )
    conditional_logits[..., MODE_NAMES.index("tx")].sum().backward()
    assert policy.model.role_beam_encoder[0].weight.grad is not None
    assert policy.model.beam_linear.weight.grad is None


def test_recurrent_beam_grid_wraps_azimuth_but_not_elevation() -> None:
    torch = pytest.importorskip("torch")
    cfg = replace(
        load_config("05_simulation/configs/mvp.yaml"),
        n_nodes=2,
        azimuth_cells=4,
        elevation_cells=3,
    )
    env = MarlNeighborDiscoveryEnv(cfg)
    observations, _ = env.reset(seed=20260823)
    policy = RecurrentContentionGraphActorCritic(
        cfg.n_beams,
        hidden_dim=16,
        azimuth_cells=cfg.azimuth_cells,
        elevation_cells=cfg.elevation_cells,
        use_candidate_mask=True,
        disabled_modes=("sense", "idle"),
    )
    block = policy.model.beam_convolution
    policy.reset_recurrent_state(cfg.n_nodes)
    with torch.no_grad():
        logits = policy.advance_recurrent_logits(observations)[1]

    assert block.azimuth_conv.padding_mode == "circular"
    assert block.elevation_conv.padding_mode == "zeros"
    assert logits.shape == (cfg.n_nodes, cfg.n_beams)
    assert torch.isfinite(logits).all()


def test_recurrent_candidate_score_prior_initializes_to_local_proportional_policy() -> None:
    torch = pytest.importorskip("torch")
    cfg = replace(load_config("05_simulation/configs/mvp.yaml"), n_nodes=2, azimuth_cells=4, elevation_cells=1)
    env = MarlNeighborDiscoveryEnv(cfg)
    observations, _ = env.reset(seed=20260824)
    for observation in observations:
        observation["candidate_mask"] = np.asarray([0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        observation["candidate_score"] = np.asarray([100.0, 1.0, 2.0, 3.0], dtype=np.float32)
    policy = RecurrentContentionGraphActorCritic(
        cfg.n_beams,
        hidden_dim=16,
        azimuth_cells=cfg.azimuth_cells,
        elevation_cells=cfg.elevation_cells,
        use_candidate_mask=True,
        use_candidate_score=True,
        use_candidate_score_prior=True,
        disabled_modes=("sense", "idle"),
    )
    with torch.no_grad():
        _mode_logits, beam_logits, _values = policy.batched_logits_value(observations, hard_mask=True)
        probabilities = torch.softmax(beam_logits, dim=-1)

    expected = torch.tensor([0.0, 1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0])
    assert torch.allclose(probabilities[0], expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(probabilities[1], expected, atol=1e-6, rtol=1e-6)


def test_bounded_score_residual_initializes_to_tempered_local_policy() -> None:
    torch = pytest.importorskip("torch")
    cfg = replace(load_config("05_simulation/configs/mvp.yaml"), n_nodes=2, azimuth_cells=4, elevation_cells=1)
    env = MarlNeighborDiscoveryEnv(cfg)
    observations, _ = env.reset(seed=20260825)
    for observation in observations:
        observation["candidate_mask"] = np.asarray([0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        observation["candidate_score"] = np.asarray([100.0, 1.0, 4.0, 9.0], dtype=np.float32)
    policy = RecurrentContentionGraphActorCritic(
        cfg.n_beams,
        hidden_dim=16,
        azimuth_cells=cfg.azimuth_cells,
        elevation_cells=cfg.elevation_cells,
        use_candidate_mask=True,
        use_candidate_score=True,
        use_candidate_score_prior=True,
        candidate_score_prior_power=0.5,
        use_bounded_score_residual=True,
        score_residual_max_logit=2.0,
        disabled_modes=("sense", "idle"),
    )
    with torch.no_grad():
        _mode_logits, beam_logits, _values = policy.batched_logits_value(observations, hard_mask=True)
        probabilities = torch.softmax(beam_logits, dim=-1)

    expected = torch.tensor([0.0, 1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0])
    assert torch.allclose(probabilities[0], expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(probabilities[1], expected, atol=1e-6, rtol=1e-6)
    initial_scale = 2.0 * torch.sigmoid(policy.model.score_residual_raw_gate)
    assert float(initial_scale.detach()) == pytest.approx(0.1, abs=1e-6)


def test_local_sticky_score_diagnostic_uses_only_valid_previous_beams() -> None:
    module = load_beam_checkpoint_evaluation_module()
    observations = [
        {
            "candidate_mask": np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
            "candidate_score": np.asarray([50.0, 1.0, 2.0, 50.0], dtype=np.float32),
        },
        {
            "candidate_mask": np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "candidate_score": np.asarray([1.0, 50.0, 2.0, 50.0], dtype=np.float32),
        },
    ]
    actions, indices = module.select_tempered_sticky_candidate_score_actions(
        observations,
        np.random.default_rng(7),
        np.random.default_rng(8),
        previous_beams=np.asarray([1, 1]),
        score_power=1.0,
        stay_probability=1.0,
    )

    assert indices[0] == 1
    assert indices[1] in {0, 2}
    assert all(action.mode in {"tx", "rx"} for action in actions)


def test_tempered_score_and_identical_stochastic_logits_share_action_randomness() -> None:
    module = load_beam_checkpoint_evaluation_module()
    observations = [
        {
            "candidate_mask": np.asarray([0.0, 1.0, 1.0, 1.0], dtype=np.float32),
            "candidate_score": np.asarray([100.0, 1.0, 4.0, 9.0], dtype=np.float32),
        }
        for _ in range(4)
    ]
    logits = np.log(np.asarray([[1.0, 1.0, 2.0, 3.0]] * 4, dtype=np.float64))
    stochastic_actions, stochastic_indices = module.select_stochastic_policy_actions(
        logits,
        observations,
        np.random.default_rng(31),
        np.random.default_rng(32),
    )
    score_actions, score_indices = module.select_tempered_sticky_candidate_score_actions(
        observations,
        np.random.default_rng(31),
        np.random.default_rng(32),
        previous_beams=np.full(4, -1, dtype=np.int64),
        score_power=0.5,
        stay_probability=0.0,
    )

    assert np.array_equal(stochastic_indices, score_indices)
    assert [action.mode for action in stochastic_actions] == [action.mode for action in score_actions]


def test_per_agent_return_scope_requires_mpnn_critic() -> None:
    module = load_training_module()
    args = Namespace(
        episodes=1,
        slots=1,
        ppo_epochs=1,
        max_rss_mb=1000.0,
        max_system_memory_percent=90.0,
        return_scope="per_agent",
        critic_network="pooled",
    )

    with pytest.raises(ValueError, match="requires --critic-network mpnn"):
        module.validate_args(args)


def test_centralized_graph_critic_is_agent_permutation_equivariant_and_actor_isolated() -> None:
    torch = pytest.importorskip("torch")
    training = load_training_module()
    cfg = replace(load_config("05_simulation/configs/mvp.yaml"), n_nodes=4, azimuth_cells=8, elevation_cells=1)
    env = MarlNeighborDiscoveryEnv(cfg)
    observations, _ = env.reset(seed=20260822)
    graph = training.central_graph_features(env.training_state(), cfg)
    tensors = {
        key: torch.as_tensor(value).unsqueeze(0)
        for key, value in graph.items()
    }
    critic = CentralizedGraphCritic(*training.central_graph_feature_dims(), hidden_dim=16)
    actor = RecurrentContentionGraphActorCritic(
        cfg.n_beams,
        hidden_dim=16,
        use_candidate_mask=True,
        disabled_modes=("sense", "idle"),
    )
    values = critic(tensors)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = {
        "node_features": tensors["node_features"][:, permutation],
        "edge_features": tensors["edge_features"][:, permutation][:, :, permutation],
        "global_features": tensors["global_features"],
        "edge_mask": tensors["edge_mask"][:, permutation][:, :, permutation],
    }
    permuted_values = critic(permuted)
    values.square().mean().backward()

    assert values.shape == (1, cfg.n_nodes)
    assert torch.allclose(permuted_values, values[:, permutation], atol=1e-6, rtol=1e-6)
    assert all(parameter.grad is None for parameter in actor.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in critic.parameters())
    assert len(observations) == cfg.n_nodes


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


def test_residual_measurement_actor_contract_is_versioned_and_checkpoint_compatible() -> None:
    torch = pytest.importorskip("torch")
    module = load_training_module()
    legacy = ContentionGraphActorCritic(8, hidden_dim=16, use_residual_measurement_features=False)
    residual = ContentionGraphActorCritic(8, hidden_dim=16, use_residual_measurement_features=True)

    assert legacy.model.beam_encoder[0].in_features == 8
    assert residual.model.beam_encoder[0].in_features == 13
    assert module.training_contract_version(Namespace(clean_ctde=True, residual_measurement_features=False)) == (
        "clean_local_ctde_v1"
    )
    assert module.training_contract_version(Namespace(clean_ctde=True, residual_measurement_features=True)) == (
        "clean_local_ctde_residual_v2"
    )
    assert torch.isfinite(next(residual.parameters())).all()


def test_stochastic_support_constraints_preserve_both_roles_and_local_beam_exploration() -> None:
    torch = pytest.importorskip("torch")
    policy = ContentionGraphActorCritic(
        4,
        hidden_dim=16,
        use_candidate_mask=True,
        role_probability_floor=0.30,
        beam_uniform_mixture=0.60,
        disabled_modes=("sense", "idle"),
    )
    mode_logits = torch.tensor([-1.0e9, 10.0, -10.0, -1.0e9])
    beam_logits = torch.tensor([10.0, -10.0, -1.0e9, -1.0e9])
    tensors = {"candidate_mask": torch.tensor([1.0, 1.0, 0.0, 0.0])}

    bounded_mode, mixed_beam = policy._regularize_stochastic_support(mode_logits, beam_logits, tensors)
    mode_probs = torch.softmax(bounded_mode, dim=-1)
    beam_probs = torch.softmax(mixed_beam, dim=-1)

    assert mode_probs[1].item() == pytest.approx(0.70, abs=1e-5)
    assert mode_probs[2].item() == pytest.approx(0.30, abs=1e-5)
    assert beam_probs[0].item() == pytest.approx(0.70, abs=1e-5)
    assert beam_probs[1].item() == pytest.approx(0.30, abs=1e-5)
    assert beam_probs[2:].sum().item() < 1e-8


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


def test_actor_observation_exposes_only_local_anonymous_residual_measurement_state() -> None:
    cfg = replace(
        load_config("05_simulation/configs/twc_planar_n10_b15_random20.yaml"),
        n_nodes=3,
        azimuth_cells=8,
        elevation_cells=1,
        sensing_measurement_mode="ideal_count",
        sensing_position_error_std_m=0.0,
    )
    env = MarlNeighborDiscoveryEnv(cfg, protocol="improved_rl_isac_tables")
    env.reset(seed=20260715)
    env._sim.states = [
        NodeState(np.asarray([0.0, 0.0, 0.0]), np.zeros(3)),
        NodeState(np.asarray([100.0, 0.0, 0.0]), np.zeros(3)),
        NodeState(np.asarray([200.0, 0.0, 0.0]), np.zeros(3)),
    ]
    env._sim.invalidate_geometry_cache()
    target_beam = env._sim.beam_from_to(0, 1)

    env._sim.update_sensing([Action("tx", target_beam), Action("idle", 0), Action("idle", 0)], slot=0)
    observation = env._observation_for(0)

    assert observation["beam_target_count"][target_beam] == pytest.approx(2.0 / cfg.target_degree)
    assert observation["beam_residual_target_count"][target_beam] == pytest.approx(2.0 / cfg.target_degree)
    assert observation["beam_interaction_count"][target_beam] == 0.0
    assert "target_id" not in observation


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


def test_residual_candidate_uses_motion_aged_empty_and_exhausted_local_state() -> None:
    cfg = replace(
        load_config("05_simulation/configs/twc_planar_n10_b15_random20.yaml"),
        n_nodes=2,
        azimuth_cells=8,
        elevation_cells=1,
    )
    env = MarlNeighborDiscoveryEnv(
        cfg,
        protocol="improved_rl_isac_tables",
        candidate_source="residual_table",
    )
    env.reset(seed=20260716)
    empty_beam = 2
    occupied_beam = 5
    env._sim.sensing_target_count_estimate[0, empty_beam] = 0.0
    env._sim.sensing_measurement_confidence[0, empty_beam] = 1.0
    env._sim.sensing_report_slot[0, empty_beam] = 0
    env._sim.sensing_target_count_estimate[0, occupied_beam] = 2.0
    env._sim.sensing_measurement_confidence[0, occupied_beam] = 1.0
    env._sim.sensing_report_slot[0, occupied_beam] = 0
    env._slot = 100

    fresh = env._observation_for(0)

    assert fresh["candidate_mask"][empty_beam] == 0.0
    assert fresh["candidate_mask"][occupied_beam] == 1.0
    assert fresh["candidate_score"][occupied_beam] > fresh["candidate_score"][0]

    env._sim.beam_interaction_count[0, occupied_beam] = 2.0
    exhausted = env._observation_for(0)
    assert exhausted["candidate_mask"][occupied_beam] == 0.0

    env._slot = 10_000
    motion_stale = env._observation_for(0)
    assert motion_stale["candidate_mask"][empty_beam] == 1.0
    assert motion_stale["candidate_mask"][occupied_beam] == 1.0


def test_evaluation_local_candidate_random_executor_uses_actor_visible_mask() -> None:
    module = load_evaluation_module()
    cfg = replace(
        load_config("05_simulation/configs/twc_planar_n10_b15_random20.yaml"),
        n_nodes=1,
        azimuth_cells=8,
        elevation_cells=1,
    )
    env = MarlNeighborDiscoveryEnv(cfg, protocol="improved_rl_isac_tables")
    env.reset(seed=20260712)
    target_beam = 5
    env._sim.belief[0, :] = 0.0
    env._sim.success_count[0, :] = 0.0
    env._sim.empty_beam_count[0, :] = 1.0
    env._sim.empty_beam_count[0, target_beam] = 0.0
    env._sim.age[0, :] = 0.0

    sampled = [module.local_candidate_random_beam(env, 0, "tx") for _ in range(20)]

    assert sampled == [target_beam] * 20


def test_evaluation_candidate_score_executor_uses_only_exposed_local_scores() -> None:
    module = load_evaluation_module()
    cfg = replace(
        load_config("05_simulation/configs/twc_planar_n10_b15_random20.yaml"),
        n_nodes=1,
        azimuth_cells=4,
        elevation_cells=1,
    )
    env = MarlNeighborDiscoveryEnv(cfg, protocol="improved_rl_isac_tables")
    env.reset(seed=20260828)
    env._candidate_features_for = lambda _node: {
        "mask": np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
        "score": np.asarray([100.0, 0.0, 5.0, 100.0], dtype=np.float32),
    }

    sampled = [
        module.local_candidate_score_proportional_beam(env, 0, "tx") for _ in range(20)
    ]

    assert sampled == [2] * 20


def test_evaluation_uniform_mode_executor_has_no_hidden_state_dependency() -> None:
    module = load_evaluation_module()
    cfg = replace(load_config("05_simulation/configs/twc_planar_n10_b15_random20.yaml"), n_nodes=1)
    env = MarlNeighborDiscoveryEnv(cfg, protocol="improved_rl_isac_tables")
    env.reset(seed=20260713)

    sampled = [module.uniform_tx_rx_mode(env) for _ in range(100)]

    assert set(sampled) == {"tx", "rx"}
    assert set(sampled).issubset({"tx", "rx"})


def test_stochastic_mappo_executor_samples_learned_probabilities_within_local_mask() -> None:
    module = load_beam_checkpoint_evaluation_module()
    observations = [
        {
            "candidate_mask": np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
        }
    ]
    logits = np.asarray([[-20.0, -4.0, 4.0, 20.0]], dtype=np.float32)
    actions = []
    for seed in range(30):
        selected, _indices = module.select_stochastic_policy_actions(
            logits,
            observations,
            np.random.default_rng(seed),
            np.random.default_rng(seed + 100),
        )
        actions.append(selected[0])

    assert {action.beam for action in actions}.issubset({1, 2})
    assert sum(action.beam == 2 for action in actions) >= 28
    assert {action.mode for action in actions} == {"tx", "rx"}


def test_selected_beam_target_status_diagnostics_separate_remaining_known_and_empty() -> None:
    cfg = replace(
        load_config("05_simulation/configs/twc_planar_n10_b15_random20.yaml"),
        n_nodes=3,
        azimuth_cells=8,
        elevation_cells=1,
    )
    env = MarlNeighborDiscoveryEnv(cfg, protocol="improved_rl_isac_tables")
    env.reset(seed=20260714)
    env._sim.states = [
        NodeState(np.asarray([0.0, 0.0, 0.0]), np.zeros(3)),
        NodeState(np.asarray([100.0, 0.0, 0.0]), np.zeros(3)),
        NodeState(np.asarray([0.0, 100.0, 0.0]), np.zeros(3)),
    ]
    env._sim.invalidate_geometry_cache()
    true_edges = env._sim.true_edges(cfg.communication_range_m)
    beam_01 = env._sim.beam_from_to(0, 1)

    env._sim.record_selected_beam_target_status([Action("tx", beam_01)], true_edges, slot=0)
    env._sim.discovered_edges.add((0, 1))
    env._sim.record_selected_beam_target_status([Action("tx", beam_01)], true_edges, slot=1)
    env._sim.record_selected_beam_target_status([Action("tx", (beam_01 + 4) % 8)], true_edges, slot=2)

    assert env._sim.undiscovered_target_beam_actions == 1
    assert env._sim.known_only_beam_actions == 1
    assert env._sim.empty_target_beam_actions == 1


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


def test_rollout_advantage_snapshot_is_detached_and_immutable() -> None:
    torch = pytest.importorskip("torch")
    module = load_training_module()
    returns = torch.tensor([[3.0, 1.0], [2.0, 4.0]])
    rollout_values = torch.tensor([[0.5, 0.25], [1.0, 2.0]], requires_grad=True)

    advantages = module.snapshot_normalized_advantages(returns, rollout_values)
    snapshot = advantages.clone()
    with torch.no_grad():
        rollout_values.add_(100.0)

    assert not advantages.requires_grad
    assert torch.equal(advantages, snapshot)
    assert float(advantages.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(advantages.std(unbiased=False)) == pytest.approx(1.0, abs=1e-6)


def test_finite_horizon_gae_lambda_one_matches_monte_carlo_returns() -> None:
    torch = pytest.importorskip("torch")
    module = load_training_module()
    rewards = torch.tensor([[1.0, 0.0], [0.5, 2.0], [3.0, 1.0]])
    rollout_values = torch.tensor([[0.4, 0.2], [0.3, 0.6], [0.8, 0.1]])

    gae_returns, raw_advantages = module.generalized_advantage_estimate(
        rewards,
        rollout_values,
        gamma=0.99,
        gae_lambda=1.0,
        torch_module=torch,
    )
    mc_returns = module.discounted_returns_2d(rewards, 0.99, torch)

    assert torch.allclose(gae_returns, mc_returns, atol=1e-6, rtol=1e-6)
    assert torch.allclose(raw_advantages, mc_returns - rollout_values, atol=1e-6, rtol=1e-6)


def test_local_potential_shaping_telescopes_at_finite_horizon() -> None:
    module = load_training_module()
    sequence = [
        [
            {
                "candidate_mask": np.asarray(mask, dtype=np.float32),
                "candidate_score": np.asarray(score, dtype=np.float32),
            }
        ]
        for mask, score in (
            ([1, 1, 1, 1], [1, 1, 1, 1]),
            ([0, 1, 1, 1], [0, 1, 2, 3]),
            ([0, 0, 1, 1], [0, 0, 1, 4]),
            ([0, 0, 0, 1], [0, 0, 0, 1]),
        )
    ]
    gamma = 0.99
    coefficient = 0.2
    shaping = []
    for step in range(len(sequence) - 1):
        shaping.append(
            module.local_potential_shaping_reward(
                sequence[step],
                sequence[step + 1],
                gamma=gamma,
                terminal=step == len(sequence) - 2,
                coefficient=coefficient,
            )[0]
        )

    discounted_shaping = sum((gamma**step) * value for step, value in enumerate(shaping))
    initial_potential = module.local_candidate_information_potential(sequence[0])[0]
    assert discounted_shaping == pytest.approx(-coefficient * initial_potential, abs=1e-6)
    assert -1.0 <= initial_potential <= 0.0


def test_isac_sensing_evidence_does_not_masquerade_as_handshake_reward() -> None:
    cfg = replace(
        load_config("05_simulation/configs/twc_planar_n10_b15_random20.yaml"),
        n_nodes=2,
        azimuth_cells=8,
        elevation_cells=1,
        sensing_measurement_mode="ideal_count",
        slots_per_episode=1,
    )
    env = MarlNeighborDiscoveryEnv(
        cfg,
        protocol="improved_rl_isac_tables",
        reward_version="discovery_first",
        candidate_source="residual_table",
    )
    env.reset(seed=20260827)
    env._sim.states = [
        NodeState(np.asarray([0.0, 0.0, 0.0]), np.zeros(3)),
        NodeState(np.asarray([100.0, 0.0, 0.0]), np.zeros(3)),
    ]
    env._sim.invalidate_geometry_cache()
    actions = [
        Action("tx", env._sim.beam_from_to(0, 1)),
        Action("tx", env._sim.beam_from_to(1, 0)),
    ]

    _observations, rewards, _terminated, _truncated, info = env.step(actions)

    assert env._sim.success_count.sum() > 0.0
    assert env._sim.node_handshake_success_count.sum() == 0
    assert info["handshake_successes"] == 0
    assert np.allclose(rewards, np.asarray([-0.002, -0.002], dtype=np.float32))


def test_local_isac_feedback_is_signed_for_tx_and_zero_for_rx() -> None:
    module = load_training_module()
    actions = [Action("tx", 1), Action("tx", 2), Action("rx", 1)]
    observations = []
    for _ in actions:
        observations.append(
            {
                "beam_target_count": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                "beam_measurement_confidence": np.asarray(
                    [0.0, 0.8, 0.6], dtype=np.float32
                ),
            }
        )

    feedback = module.local_beam_isac_feedback(actions, observations)

    assert feedback[0] == pytest.approx(0.8)
    assert feedback[1] == pytest.approx(-0.6)
    assert feedback[2] == 0.0


def test_role_balance_regularizer_allows_heterogeneous_local_probabilities() -> None:
    torch = pytest.importorskip("torch")
    module = load_training_module()
    balanced = torch.tensor([[0.2, 0.8], [0.9, 0.1]], requires_grad=True)
    collapsed = torch.tensor([[0.9, 0.9], [0.8, 0.8]])

    balanced_loss, balanced_mean = module.role_balance_regularizer(balanced, torch)
    collapsed_loss, collapsed_mean = module.role_balance_regularizer(collapsed, torch)
    balanced_loss.backward()

    assert float(balanced_loss.detach()) == pytest.approx(0.0, abs=1e-8)
    assert balanced_mean == pytest.approx(0.5)
    assert float(collapsed_loss) > 0.0
    assert collapsed_mean == pytest.approx(0.85)
    assert balanced.grad is not None
