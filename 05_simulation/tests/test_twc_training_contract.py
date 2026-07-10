from __future__ import annotations

import importlib.util
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from isac_nd_sim.config import load_config, with_communication_tx_power
from isac_nd_sim.marl_env import MarlNeighborDiscoveryEnv
from isac_nd_sim.neural_contention_actor_critic import ContentionGraphActorCritic


ROOT = Path(__file__).resolve().parents[2]
TRAINING_SCRIPT = ROOT / "05_simulation" / "run_marl_training.py"


def load_training_module():
    spec = importlib.util.spec_from_file_location("twc_training_contract", TRAINING_SCRIPT)
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


def test_disabled_structured_features_do_not_leak_through_aggregate_paths() -> None:
    torch = pytest.importorskip("torch")
    cfg = replace(load_config("05_simulation/configs/twc_trainable_n10.yaml"), n_nodes=1, azimuth_cells=4, elevation_cells=2)
    env = MarlNeighborDiscoveryEnv(cfg)
    observations, _ = env.reset(seed=91)
    base = observations[0]
    changed = {key: value.copy() if hasattr(value, "copy") else value for key, value in base.items()}
    changed["candidate_mask"][:] = 1.0 - changed["candidate_mask"]
    changed["candidate_score"][:] = np.linspace(0.0, 1.0, cfg.n_beams, dtype=np.float32)
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
