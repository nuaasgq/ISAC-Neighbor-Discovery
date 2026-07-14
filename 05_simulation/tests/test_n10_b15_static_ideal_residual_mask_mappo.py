from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "05_simulation" / "scripts" / "run_n10_b15_static_ideal_residual_mask_mappo.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location("residual_mask_mappo_launcher", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_residual_mask_contract_changes_only_local_candidate_support() -> None:
    module = load_launcher()
    command = module.residual_mask_command("pilot", Path("output"), 59262731, 1)

    module.validate_contract(command, "pilot")
    assert module.command_value(command, "--candidate-source") == "residual_table"
    assert "--candidate-mask" in command
    assert "--no-candidate-score" in command
    assert "--no-candidate-score-prior" in command
    assert module.command_value(command, "--expert-bc-weight") == "0.0"
    assert module.command_value(command, "--measurement-prediction-aux-coef") == "0.0"
    assert module.command_value(command, "--episodes") == "100"
