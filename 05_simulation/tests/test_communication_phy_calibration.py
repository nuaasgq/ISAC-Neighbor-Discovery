from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "05_simulation" / "run_communication_phy_calibration.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_communication_phy_calibration", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stable_rng_reproduces_common_random_numbers() -> None:
    module = load_module()

    first = module.stable_rng(123, "condition", "edge").normal(size=20)
    second = module.stable_rng(123, "condition", "edge").normal(size=20)
    changed = module.stable_rng(123, "condition", "mid").normal(size=20)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, changed)


def test_rank_operating_points_prioritizes_feasible_rows() -> None:
    module = load_module()
    rows = [
        {
            "edge_coverage_rate": 0.90,
            "equal_power_decode_rate": 0.10,
            "near_far_capture_rate": 0.80,
        },
        {
            "edge_coverage_rate": 1.00,
            "equal_power_decode_rate": 0.00,
            "near_far_capture_rate": 1.00,
        },
    ]

    ranked = module.rank_operating_points(rows)

    assert ranked[0]["feasible"] == 1
    assert ranked[1]["feasible"] == 0


def test_wilson_interval_contains_observed_probability() -> None:
    module = load_module()
    rate, low, high = module.probability_with_wilson_interval(
        np.asarray([True] * 80 + [False] * 20)
    )

    assert rate == 0.8
    assert low < rate < high


def test_calibration_smoke_writes_tables_manifest_and_4_by_3_figures(tmp_path: Path) -> None:
    module = load_module()
    output = tmp_path / "raw"
    paper_output = tmp_path / "paper"
    figure_output = tmp_path / "figures"
    manifest = module.run_calibration(
        Namespace(
            config=str(ROOT / "05_simulation" / "configs" / "twc_trainable_n10.yaml"),
            output=str(output),
            paper_output=str(paper_output),
            figure_output=str(figure_output),
            samples=100,
            seed=456,
            protocol_validation_episodes=0,
            protocol_validation_slots=10,
        )
    )

    assert manifest["scope"] == "communication_phy_sensitivity_calibration"
    assert len(manifest["recommended"]) == 10
    for name in (
        "oat_metrics.csv",
        "coverage_distance.csv",
        "joint_metrics.csv",
        "recommended_operating_points.csv",
        "recommended_profiles.csv",
        "protocol_validation_rows.csv",
        "protocol_validation_summary.csv",
        "manifest.json",
    ):
        assert (output / name).exists()
        assert (paper_output / name).exists()
    figures = sorted(figure_output.glob("*.png"))
    assert len(figures) == 8
