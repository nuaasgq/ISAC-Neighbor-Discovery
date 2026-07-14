from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "06_analysis" / "scripts" / "analyze_n10_b15_static_ideal_single_mixture.py"


def load_analysis():
    spec = importlib.util.spec_from_file_location("single_mixture_analysis", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_rows(module, mismatch: bool = False):
    rows = []
    for method_index, method in enumerate(module.METHODS):
        for episode in range(2):
            scenario = 100 + episode + (1 if mismatch and method_index == 1 else 0)
            rows.append(
                {
                    "method": method,
                    "train_seed": 10,
                    "scenario_seed": scenario,
                }
            )
    return rows


def test_analysis_requires_exact_method_and_scenario_pairing() -> None:
    module = load_analysis()
    assert module.validate_pairing(synthetic_rows(module), expected_per_method=2) == {
        (10, 100),
        (10, 101),
    }
    with pytest.raises(ValueError, match="paired scenarios"):
        module.validate_pairing(synthetic_rows(module, mismatch=True), expected_per_method=2)


def test_training_curve_uses_environment_steps_and_common_episodes() -> None:
    module = load_analysis()
    rows = []
    for method in module.TRAINED_METHODS:
        for seed in (1, 2):
            for episode in range(3):
                rows.append(
                    {
                        "method": method,
                        "train_seed": seed,
                        "episode": episode,
                        "environment_step": (episode + 1) * 300,
                        "episode_return_sum": float(seed + episode),
                        "discovery_rate": 0.1 * episode,
                    }
                )

    curves = module.training_curve_rows(rows, window=2)

    assert len(curves) == 6
    assert {row["environment_step"] for row in curves} == {300, 600, 900}
    assert all(row["rolling_window_episodes"] == 2 for row in curves)
