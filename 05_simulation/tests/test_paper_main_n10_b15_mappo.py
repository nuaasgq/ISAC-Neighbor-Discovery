from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "05_simulation" / "scripts" / "run_paper_main_n10_b15_mappo.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location("paper_main_launcher", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paper_main_command_freezes_environment_and_removes_rule_priors(tmp_path: Path) -> None:
    module = load_launcher()
    command = module.training_command("formal", tmp_path, 12345, 1)
    module.validate_contract(command, "formal")

    assert module.command_value(command, "--mobility-model") == "gauss_markov"
    assert module.command_value(command, "--sensing-measurement-mode") == "noisy_count"
    assert module.command_value(command, "--candidate-source") == "residual_table"
    assert "--candidate-mask" in command
    assert "--clean-ctde" in command
    assert "--rule-residual" not in command
    assert "--contention-mode-prior" not in command
    assert "--candidate-score-prior" not in command
    assert "--rendezvous-observation" not in command

