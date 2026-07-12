from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

from isac_nd_sim.config import load_config
from isac_nd_sim.marl_env import MarlNeighborDiscoveryEnv


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "05_simulation" / "run_matd3_reference_training.py"


def load_adapter():
    spec = importlib.util.spec_from_file_location("matd3_reference_adapter", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_flat_matd3_actor_observation_ignores_candidate_rule_fields() -> None:
    module = load_adapter()
    cfg = load_config("05_simulation/configs/n10_b15_static_ideal_isac.yaml")
    env = MarlNeighborDiscoveryEnv(cfg, protocol="improved_rl_isac_tables")
    observations, _ = env.reset(seed=20260713)
    changed = []
    for observation in observations:
        item = {key: value.copy() if hasattr(value, "copy") else value for key, value in observation.items()}
        item["candidate_mask"][:] = 0.0
        item["candidate_score"][:] = 99.0
        changed.append(item)

    original_flat = module.flat_local_observations(observations, cfg.n_beams)
    changed_flat = module.flat_local_observations(changed, cfg.n_beams)

    assert original_flat.shape[0] == cfg.n_nodes
    assert np.allclose(original_flat, changed_flat)


def test_flat_matd3_actor_observation_retains_direct_isac_measurement() -> None:
    module = load_adapter()
    cfg = load_config("05_simulation/configs/n10_b15_static_ideal_isac.yaml")
    env = MarlNeighborDiscoveryEnv(cfg, protocol="improved_rl_isac_tables")
    observations, _ = env.reset(seed=20260714)
    changed = [{key: value.copy() if hasattr(value, "copy") else value for key, value in item.items()} for item in observations]
    changed[0]["beam_target_count"][3] = 1.0
    changed[0]["beam_measurement_confidence"][3] = 1.0

    original_flat = module.flat_local_observations(observations, cfg.n_beams)
    changed_flat = module.flat_local_observations(changed, cfg.n_beams)

    assert not np.allclose(original_flat[0], changed_flat[0])
    assert np.allclose(original_flat[1:], changed_flat[1:])


def test_matd3_multidiscrete_decoder_maps_two_action_heads() -> None:
    module = load_adapter()
    encoded = np.zeros((2, 26), dtype=np.float32)
    encoded[0, 0] = 1.0
    encoded[0, 2 + 7] = 1.0
    encoded[1, 1] = 1.0
    encoded[1, 2 + 19] = 1.0

    actions = module.decode_multidiscrete_actions(encoded)

    assert [(action.mode, action.beam) for action in actions] == [("tx", 7), ("rx", 19)]


def test_metric_writer_accepts_training_fields_that_appear_later(tmp_path: Path) -> None:
    module = load_adapter()
    output = tmp_path / "metrics.csv"

    module.write_rows(output, [{"episode": 0}, {"episode": 1, "critic_loss": 0.25}])

    text = output.read_text(encoding="utf-8")
    assert "critic_loss" in text.splitlines()[0]
    assert "0.25" in text
