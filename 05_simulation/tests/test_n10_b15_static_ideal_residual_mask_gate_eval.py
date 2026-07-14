from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "05_simulation" / "scripts" / "run_n10_b15_static_ideal_residual_mask_gate_eval.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location("residual_mask_gate_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gate_eval_uses_exact_paired_seed_without_action_overrides() -> None:
    module = load_launcher()
    command = module.evaluation_command(
        Path("checkpoint.pt"),
        Path("output"),
        59262731,
        50,
        1,
    )

    module.validate_contract(command, 50, 59262731)
    assert command[command.index("--seed") + 1] == "61262731"
    assert "--collect-discovery-timeline" in command
    assert "--candidate-source" not in command
    assert "--beam-executor" not in command
    assert "--mode-executor" not in command
