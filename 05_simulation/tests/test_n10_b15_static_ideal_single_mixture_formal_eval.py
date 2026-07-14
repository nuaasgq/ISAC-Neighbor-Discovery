from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "05_simulation"
    / "scripts"
    / "run_n10_b15_static_ideal_single_mixture_formal_eval.py"
)


def load_launcher():
    spec = importlib.util.spec_from_file_location("single_mixture_formal_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_formal_eval_uses_exact_paired_seed_and_offline_diagnostics() -> None:
    module = load_launcher()
    for method in module.METHODS:
        command = module.evaluation_command(
            Path("checkpoint.pt"), Path("output"), 59260713, method, 1
        )

        module.validate_contract(command, 59260713, method)
        assert command[command.index("--seed") + 1] == "61260713"
        assert command[command.index("--eval-episodes") + 1] == "50"
        assert command[command.index("--slots") + 1] == "300"
        assert command[command.index("--ablation-label") + 1] == method
        assert "--collect-candidate-pool-timeline" in command
        assert "--target-status-diagnostics" in command
        assert "--candidate-source" not in command
        assert "--beam-executor" not in command
        assert "--mode-executor" not in command


def test_method_parser_rejects_unknown_or_duplicate_values() -> None:
    module = load_launcher()

    for value in ("mappo_direct_isac,mappo_direct_isac", "unknown"):
        try:
            module.parse_methods(value)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid method selection must be rejected.")
