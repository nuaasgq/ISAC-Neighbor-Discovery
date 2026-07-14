from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "05_simulation" / "scripts" / "run_n10_b15_static_ideal_role_head_ablation.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location("role_head_ablation_launcher", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_independent_role_ablation_changes_only_role_factorization_contract() -> None:
    module = load_launcher()
    command = module.ablation_command("formal", Path("output"), 59262731, 1)

    module.validate_contract(command, "formal")
    assert module.command_value(command, "--role-factorization") == "independent"
    assert module.command_value(command, "--measurement-feature-set") == "direct"
    assert module.command_value(command, "--measurement-prediction-aux-coef") == "0.0"
    assert "--decoupled-role-tower" in command
