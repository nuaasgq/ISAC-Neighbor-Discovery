from __future__ import annotations

import importlib.util
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import pytest

from isac_nd_sim.config import load_config
from isac_nd_sim.marl_env import MarlNeighborDiscoveryEnv
from isac_nd_sim.neural_shared_actor_critic import SharedBeamActorCritic


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "05_simulation" / "run_actor_critic_probe.py"
IMITATION_SCRIPT = ROOT / "05_simulation" / "run_actor_critic_imitation_probe.py"


def test_shared_actor_critic_samples_valid_actions() -> None:
    pytest.importorskip("torch")
    cfg = replace(load_config("05_simulation/configs/mvp.yaml"), n_nodes=4, azimuth_cells=4, elevation_cells=2)
    env = MarlNeighborDiscoveryEnv(cfg)
    observations, _ = env.reset(seed=123)
    policy = SharedBeamActorCritic(cfg.n_beams, hidden_dim=16)

    step = policy.act(observations)

    assert len(step.actions) == cfg.n_nodes
    assert step.log_probs.shape == (cfg.n_nodes,)
    assert step.values.shape == (cfg.n_nodes,)
    for action in step.actions:
        assert action.mode in env.modes
        assert 0 <= action.beam < cfg.n_beams


def test_actor_critic_probe_writes_history(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    module = load_probe_module(SCRIPT, "run_actor_critic_probe")
    output = tmp_path / "probe"
    manifest = module.run_probe(
        Namespace(
            config=str(ROOT / "05_simulation" / "configs" / "mvp.yaml"),
            output=str(output),
            episodes=1,
            slots=2,
            node_count=4,
            azimuth_cells=4,
            elevation_cells=2,
            hidden_dim=16,
            learning_rate=1e-3,
            gamma=0.95,
            value_coef=0.5,
            entropy_coef=0.01,
            seed=77,
        )
    )

    assert manifest["algorithm"] == "shared_actor_critic_probe"
    assert (output / "training_history.csv").exists()
    assert (output / "manifest.json").exists()


def test_actor_critic_imitation_probe_writes_history(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    module = load_probe_module(IMITATION_SCRIPT, "run_actor_critic_imitation_probe")
    output = tmp_path / "imitation_probe"
    manifest = module.run_probe(
        Namespace(
            config=str(ROOT / "05_simulation" / "configs" / "mvp.yaml"),
            output=str(output),
            bc_episodes=1,
            rl_episodes=0,
            eval_episodes=0,
            slots=2,
            node_count=4,
            azimuth_cells=4,
            elevation_cells=2,
            communication_range=800.0,
            sensing_range=900.0,
            false_alarm_rate=0.0,
            miss_detection_rate=0.0,
            angular_cell_offset_std=0.0,
            sensing_period_slots=1,
            expert_protocol="improved_rl_isac",
            hidden_dim=16,
            learning_rate=1e-3,
            gamma=0.95,
            bc_coef=1.0,
            beam_bc_coef=1.0,
            value_coef=0.25,
            entropy_coef=0.001,
            seed=78,
            expert_seed_offset=7919,
        )
    )

    assert manifest["algorithm"] == "shared_actor_critic_imitation_probe"
    assert manifest["scope"] == "method_probe_not_paper_result"
    assert (output / "training_history.csv").exists()
    assert (output / "manifest.json").exists()


def load_probe_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
