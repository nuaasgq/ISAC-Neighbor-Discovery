from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "06_analysis" / "scripts" / "analyze_n10_b15_static_ideal_residual_mask_gate.py"


def load_analysis():
    spec = importlib.util.spec_from_file_location("residual_mask_gate_analysis", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_rows(module, mismatch: bool = False):
    rows = []
    for method_index, method in enumerate(module.METHODS):
        for episode in range(2):
            seed = 100 + episode + (1 if mismatch and method_index == 1 else 0)
            rows.append(
                {
                    "method": method,
                    "eval_episode": episode,
                    "scenario_seed": seed,
                }
            )
    return rows


def test_gate_analysis_requires_exact_scenario_pairing() -> None:
    module = load_analysis()
    assert module.validate_pairing(synthetic_rows(module), expected_episodes=2) == (100, 101)
    with pytest.raises(ValueError, match="scenario seeds"):
        module.validate_pairing(synthetic_rows(module, mismatch=True), expected_episodes=2)
