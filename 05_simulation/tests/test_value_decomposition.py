from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from isac_nd_sim.config import load_config
from isac_nd_sim.marl_env import MarlNeighborDiscoveryEnv
from isac_nd_sim.value_decomposition import (
    IndependentReplayBuffer,
    JointTransition,
    MonotonicQMix,
    ValueDecompositionLearner,
    requires_global_training_state,
    select_local_actions,
)


def small_env() -> MarlNeighborDiscoveryEnv:
    cfg = replace(
        load_config("05_simulation/configs/twc_planar_n10_b15_random20.yaml"),
        n_nodes=3,
        azimuth_cells=8,
        elevation_cells=1,
        slots_per_episode=3,
    )
    return MarlNeighborDiscoveryEnv(
        cfg,
        protocol="improved_rl_isac_tables",
        reward_version="discovery_first",
        candidate_source="residual_table",
        collect_slot_metrics=False,
        rich_info=False,
    )


def test_local_value_network_returns_tx_rx_beam_values_from_local_observations() -> None:
    torch = pytest.importorskip("torch")
    env = small_env()
    observations, _ = env.reset(seed=20260721)
    learner = ValueDecompositionLearner(
        "shared_idqn",
        env.n_agents,
        env.n_beams,
        state_dim=23,
        hidden_dim=16,
    )

    with torch.no_grad():
        q_values = learner.q_values(observations)

    assert q_values.shape == (env.n_agents, 2 * env.n_beams)
    assert torch.isfinite(q_values).all()
    assert learner.centralized_training is False


def test_only_qmix_requires_global_training_state() -> None:
    assert not requires_global_training_state("idqn")
    assert not requires_global_training_state("shared_idqn")
    assert not requires_global_training_state("vdn")
    assert requires_global_training_state("qmix")


def test_independent_dqn_owns_one_distinct_network_per_uav() -> None:
    pytest.importorskip("torch")
    env = small_env()
    learner = ValueDecompositionLearner("idqn", env.n_agents, env.n_beams, state_dim=23, hidden_dim=16)

    assert len(learner.q_networks) == env.n_agents
    pointers = {next(network.parameters()).data_ptr() for network in learner.q_networks}
    assert len(pointers) == env.n_agents
    assert len(learner.optimizers) == env.n_agents
    assert learner.independent_parameters is True
    assert learner.centralized_training is False


def test_independent_dqn_replay_samples_each_uav_locally() -> None:
    env = small_env()
    observations, _ = env.reset(seed=20260724)
    transition = JointTransition(
        observations=observations,
        action_indices=np.asarray([0, 1, 2], dtype=np.int64),
        rewards=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        next_observations=observations,
        done=False,
        central_state=np.ones(23, dtype=np.float32),
        next_central_state=np.ones(23, dtype=np.float32),
    )
    replay = IndependentReplayBuffer(env.n_agents, capacity=4, seed=5, reward_scope="team")
    replay.append(transition)

    sampled = replay.sample(1, np.random.default_rng(99))[0]

    assert np.array_equal(sampled.action_indices, transition.action_indices)
    assert np.allclose(sampled.rewards, np.mean(transition.rewards))
    assert sampled.central_state.shape == (1,)
    assert np.allclose(sampled.central_state, 0.0)


def test_qmix_mixer_is_monotonic_in_each_agent_value() -> None:
    torch = pytest.importorskip("torch")
    mixer = MonotonicQMix(n_agents=3, state_dim=5, embed_dim=8)
    state = torch.randn(4, 5)
    baseline = torch.randn(4, 3)

    original = mixer(baseline, state)
    increased = mixer(baseline + torch.tensor([[0.0, 0.5, 0.0]]), state)

    assert torch.all(increased >= original - 1e-7)


def test_value_action_selector_respects_local_candidate_mask() -> None:
    env = small_env()
    observations, _ = env.reset(seed=20260722)
    for observation in observations:
        observation["candidate_mask"][:] = 0.0
        observation["candidate_mask"][3] = 1.0
    q_values = np.zeros((env.n_agents, 2 * env.n_beams), dtype=np.float32)
    q_values[:, env.n_beams + 3] = 5.0

    actions, indices = select_local_actions(
        q_values,
        observations,
        np.random.default_rng(1),
        role_uniform_mixture=0.0,
        beam_uniform_mixture=0.0,
    )

    assert all(action.mode == "rx" and action.beam == 3 for action in actions)
    assert np.array_equal(indices, np.full(env.n_agents, env.n_beams + 3))


@pytest.mark.parametrize("algorithm", ["idqn", "shared_idqn", "vdn", "qmix"])
def test_value_based_update_is_finite_for_every_algorithm(algorithm: str) -> None:
    torch = pytest.importorskip("torch")
    env = small_env()
    observations, _ = env.reset(seed=20260723)
    learner = ValueDecompositionLearner(
        algorithm,
        env.n_agents,
        env.n_beams,
        state_dim=23,
        hidden_dim=16,
        mixer_dim=8,
    )
    with torch.no_grad():
        q_values = learner.q_values(observations).cpu().numpy()
    actions, indices = select_local_actions(
        q_values,
        observations,
        np.random.default_rng(2),
        role_uniform_mixture=1.0,
        beam_uniform_mixture=1.0,
    )
    next_observations, rewards, _terminated, truncated, _info = env.step(actions)
    transition = JointTransition(
        observations=observations,
        action_indices=indices,
        rewards=rewards,
        next_observations=next_observations,
        done=truncated,
        central_state=np.zeros(23, dtype=np.float32),
        next_central_state=np.ones(23, dtype=np.float32),
    )

    losses = learner.update([transition, transition])

    assert losses["td_loss"] >= 0.0
    assert all(np.isfinite(value) for value in losses.values())
