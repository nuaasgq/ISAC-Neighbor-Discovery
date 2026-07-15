from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "05_simulation" / "scripts" / "run_paper_main_n10_b15_paired_eval.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location("paper_main_paired_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_marl_commands_use_the_same_new_held_out_scenarios(tmp_path: Path) -> None:
    module = load_launcher()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    command = module.marl_command(
        checkpoint,
        tmp_path / "output",
        "main",
        module.EVAL_SEED,
        50,
        1,
    )

    assert module.command_value(command, "--seed") == str(module.EVAL_SEED)
    assert module.command_value(command, "--eval-episodes") == "50"
    assert module.command_value(command, "--slots") == "300"
    assert module.command_value(command, "--mode-executor") == "policy"
    assert module.command_value(command, "--beam-executor") == "policy"
    assert "--collect-discovery-timeline" in command


def test_candidate_random_changes_only_external_action_execution(tmp_path: Path) -> None:
    module = load_launcher()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    command = module.marl_command(
        checkpoint,
        tmp_path / "output",
        "candidate_random",
        module.EVAL_SEED,
        50,
        1,
        candidate_random=True,
    )

    assert module.command_value(command, "--mode-executor") == "uniform_tx_rx"
    assert module.command_value(command, "--beam-executor") == "local_candidate_random"
    assert module.command_value(command, "--candidate-source") == "residual_table"
    assert module.command_value(command, "--policy-ablation") == "zero_weights"


def test_protocol_command_uses_identical_config_seed_and_horizon(tmp_path: Path) -> None:
    module = load_launcher()
    command = module.protocol_command(tmp_path, module.EVAL_SEED, 50)

    assert module.command_value(command, "--config") == str(module.CONFIG)
    assert module.command_value(command, "--seed") == str(module.EVAL_SEED)
    assert module.command_value(command, "--eval-episodes") == "50"
    assert command[command.index("--protocols") + 1 : command.index("--eval-episodes")] == [
        "uniform_random",
        "wang2025_isac_tables",
    ]
