from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "05_simulation" / "scripts" / "run_n10_b15_static_ideal_matd3_reference_matrix.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location("n10_matd3_launcher", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_formal_matd3_command_uses_fixed_n10_contract() -> None:
    module = load_launcher()
    output = ROOT / "temporary" / "seed_1"
    command = module.training_command("formal", output, seed=1, threads=1)

    assert command[command.index("--episodes") + 1] == "1000"
    assert command[command.index("--slots") + 1] == "300"
    assert command[command.index("--eval-episodes") + 1] == "50"
    assert command[command.index("--warmup-steps") + 1] == "1500"
    assert command[command.index("--torch-threads") + 1] == "1"


def test_matd3_seed_parser_rejects_duplicates() -> None:
    module = load_launcher()
    try:
        module.parse_seeds("1,1")
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate seeds must be rejected.")
