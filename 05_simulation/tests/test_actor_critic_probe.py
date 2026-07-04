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
    module = load_probe_module()
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


def load_probe_module():
    spec = importlib.util.spec_from_file_location("run_actor_critic_probe", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
