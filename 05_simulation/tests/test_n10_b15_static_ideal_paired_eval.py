from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "05_simulation" / "scripts" / "run_n10_b15_static_ideal_paired_eval.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location("paired_eval_launcher", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_checkpoint_methods_cover_core_and_action_interventions() -> None:
    module = load_launcher()
    labels = {method.label for method in module.CHECKPOINT_METHODS}

    assert labels == {
        "mappo_no_isac",
        "mappo_direct_isac",
        "mappo_direct_isac_measurement_aux",
        "random_role_learned_beam",
        "learned_role_uniform_beam",
        "isac_candidate_pool_random",
    }


def test_paired_checkpoint_command_uses_original_held_out_seed() -> None:
    module = load_launcher()
    method = next(item for item in module.CHECKPOINT_METHODS if item.label == "mappo_direct_isac")
    command = module.checkpoint_command(method, 59260713, 50, Path("output"), 1)

    assert command[command.index("--seed") + 1] == "61260713"
    assert command[command.index("--eval-episodes") + 1] == "50"
    assert command[command.index("--slots") + 1] == "300"
    assert "--collect-discovery-timeline" in command


def test_learned_role_random_beam_conditions_on_executed_beam() -> None:
    module = load_launcher()
    method = next(item for item in module.CHECKPOINT_METHODS if item.label == "learned_role_uniform_beam")
    command = module.checkpoint_command(method, 1, 2, Path("output"), 1)

    assert command[command.index("--mode-executor") + 1] == "policy"
    assert command[command.index("--beam-executor") + 1] == "uniform_random"
    assert "--condition-role-on-executed-beam" in command
