from __future__ import annotations

import pytest

from isac_nd_sim.config import load_config
from isac_nd_sim.evaluation_timeline import discovery_curve_summary, discovery_timeline_rows
from isac_nd_sim.marl_env import MarlNeighborDiscoveryEnv
from isac_nd_sim.neural_recurrent_contention_actor_critic import (
    RecurrentContentionGraphActorCritic,
)


def test_discovery_timeline_records_found_and_censored_edges() -> None:
    cfg = load_config("05_simulation/configs/sanity_planar_n2_b45_ideal.yaml")
    env = MarlNeighborDiscoveryEnv(cfg)
    env.reset(seed=17)
    env._sim.first_true_slot = {(0, 1): 0, (0, 2): 10}
    env._sim.discovery_slot = {(0, 1): 4}

    rows = discovery_timeline_rows(
        env._sim,
        eval_episode=3,
        scenario_seed=17,
        method="test",
    )

    assert rows[0]["discovered"] is True
    assert rows[0]["discovery_time_slots"] == 5
    assert rows[0]["delay_slots_censored"] == 5
    assert rows[1]["discovered"] is False
    assert rows[1]["delay_slots_censored"] == cfg.slots_per_episode - 10


def test_discovery_curve_summary_uses_end_of_slot_convention() -> None:
    cfg = load_config("05_simulation/configs/sanity_planar_n2_b45_ideal.yaml")
    env = MarlNeighborDiscoveryEnv(cfg)
    env.reset(seed=18)
    env._sim.first_true_slot = {(0, 1): 0, (0, 2): 0}
    env._sim.discovery_slot = {(0, 1): 4}

    summary = discovery_curve_summary(env._sim, milestones=(4, 5, 10))

    assert summary["discovery_rate_at_4_slots"] == 0.0
    assert summary["discovery_rate_at_5_slots"] == 0.5
    assert summary["discovery_rate_at_10_slots"] == 0.5
    assert summary["time_to_50pct_censored_slots"] == 5.0


def test_recurrent_policy_conditions_role_on_forced_executed_beams() -> None:
    cfg = load_config("05_simulation/configs/sanity_planar_n2_b45_ideal.yaml")
    env = MarlNeighborDiscoveryEnv(cfg)
    observations, _ = env.reset(seed=19)
    policy = RecurrentContentionGraphActorCritic(
        cfg.n_beams,
        hidden_dim=16,
        disabled_modes=("sense", "idle"),
        action_contract="joint_role_beam",
        azimuth_cells=cfg.azimuth_cells,
        elevation_cells=cfg.elevation_cells,
        use_decoupled_role_tower=True,
        role_factorization="beam_conditioned_antisymmetric",
    )
    forced = [2, 5]

    step = policy.act(observations, deterministic=True, forced_beam_indices=forced)

    assert [action.beam for action in step.actions] == forced
    with pytest.raises(ValueError, match="out-of-range"):
        policy.reset_recurrent_state(env.n_agents)
        policy.act(observations, forced_beam_indices=[-1, 5])
