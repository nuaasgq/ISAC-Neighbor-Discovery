from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from isac_nd_sim.config import load_config


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "05_simulation" / "scripts" / "run_n10_b15_static_ideal_mappo_matrix.py"
CONFIG = ROOT / "05_simulation" / "configs" / "n10_b15_static_ideal_isac.yaml"


def load_launcher():
    spec = importlib.util.spec_from_file_location("n10_static_ideal_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_static_ideal_config_contract() -> None:
    cfg = load_config(CONFIG)
    assert cfg.n_nodes == 10
    assert cfg.n_beams == 24
    assert cfg.elevation_cells == 1
    assert cfg.slots_per_episode == 300
    assert cfg.mobility["model"] == "static"
    assert cfg.communication_phy_model == "ideal"
    assert cfg.sensing_measurement_mode == "ideal_count"
    assert cfg.communication_range_m > sum(value * value for value in cfg.area_size_m) ** 0.5
    assert cfg.sensing_range_m > sum(value * value for value in cfg.area_size_m) ** 0.5


def test_three_mappo_arms_differ_only_by_isac_and_auxiliary_contract(tmp_path: Path) -> None:
    module = load_launcher()
    assert tuple(module.METHOD_SPECS) == (
        "mappo_no_isac",
        "mappo_direct_isac",
        "mappo_direct_isac_measurement_aux",
    )
    commands = [
        module.training_command("smoke", tmp_path, method, seed, 1)
        for seed in module.DEFAULT_SEEDS
        for method in module.METHOD_SPECS
    ]
    module.validate_matrix_contract(commands, module.DEFAULT_SEEDS)
    assert len(commands) == 9
    for command in commands:
        assert "--clean-ctde" in command
        assert "--no-candidate-score" in command
        assert "--no-candidate-score-prior" in command
        assert "--no-rendezvous-observation" in command
        assert "--candidate-mask" not in command
        assert module.command_value(command, "--role-factorization") == "beam_conditioned_antisymmetric"


def test_formal_profile_is_300k_environment_steps_and_50_eval_episodes() -> None:
    module = load_launcher()
    assert module.PROFILES["formal"]["episodes"] * 300 == 300_000
    assert module.PROFILES["formal"]["eval_episodes"] == 50


def test_method_subset_supports_a_fair_targeted_retrain(tmp_path: Path) -> None:
    module = load_launcher()
    methods = module.parse_methods("mappo_direct_isac")
    commands = [
        module.training_command("smoke", tmp_path, method, seed, 1)
        for seed in (59260713, 59261722)
        for method in methods
    ]

    module.validate_matrix_contract(commands, (59260713, 59261722), methods)
    assert methods == ("mappo_direct_isac",)
    assert len(commands) == 2
