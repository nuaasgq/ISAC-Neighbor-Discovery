from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "05_simulation" / "run_transfer_sweep.py"


def load_transfer_sweep_module():
    spec = importlib.util.spec_from_file_location("run_transfer_sweep", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_isac_error_profiles_pairs_triples() -> None:
    module = load_transfer_sweep_module()

    profiles = module.parse_isac_error_profiles("0:0:0;0.01:0.05:0.5;0.1:0.3:1.5")

    assert profiles == [(0.0, 0.0, 0.0), (0.01, 0.05, 0.5), (0.1, 0.3, 1.5)]
    assert module.format_error_profiles(profiles) == "(0, 0, 0), (0.01, 0.05, 0.5), (0.1, 0.3, 1.5)"


def test_parse_isac_error_profiles_rejects_non_triples() -> None:
    module = load_transfer_sweep_module()

    with pytest.raises(ValueError, match="triples"):
        module.parse_isac_error_profiles("0.01:0.05")

