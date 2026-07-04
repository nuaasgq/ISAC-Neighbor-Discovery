from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PLOTTER = ROOT / "06_analysis" / "scripts" / "plot_paper_results.py"


def test_paper_plotting_writes_manifest_and_4x3_figures(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    image_module = pytest.importorskip("PIL.Image")

    table_dir = tmp_path / "gauss_tables"
    training_dir = tmp_path / "training"
    output_dir = tmp_path / "paper_figures"
    write_analysis_tables(table_dir)
    write_training_history(training_dir)

    result = subprocess.run(
        [
            sys.executable,
            str(PLOTTER),
            str(table_dir),
            "--training-dir",
            str(training_dir),
            "--output",
            str(output_dir),
            "--dpi",
            "120",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout_manifest = json.loads(result.stdout)
    manifest_path = output_dir / "paper_figure_manifest.json"
    readme_path = output_dir / "README.md"
    assert manifest_path.exists()
    assert readme_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["style"]["aspect_ratio"] == "4:3"
    assert manifest["capability"]["training_figure_specs"] == 8
    assert manifest["capability"]["test_figure_specs_per_table_dir"] == 12
    assert manifest["counts"]["generated"] >= 16
    assert stdout_manifest["counts"] == manifest["counts"]

    expected = [
        output_dir / "test_gauss_discovery_rate.png",
        output_dir / "test_gauss_slot_discovered_edges.png",
        output_dir / "train_score_curve.png",
    ]
    for path in expected:
        assert path.exists(), path
        with image_module.open(path) as image:
            assert image.size == (768, 576)
            assert image.size[0] / image.size[1] == pytest.approx(4 / 3)

    generated_names = {item["filename"] for item in manifest["figures"] if item["status"] == "generated"}
    assert {path.name for path in expected}.issubset(generated_names)


def write_analysis_tables(table_dir: Path) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        protocol_summary_row(
            "itap_nd",
            discovery_rate=0.82,
            empty_scan_ratio=0.24,
            mean_discovery_delay=7.2,
            p90_discovery_delay=12.0,
            p95_discovery_delay=15.0,
            p99_discovery_delay=19.0,
            lcc_ratio=0.92,
            lambda2=0.31,
            collision_count=0.8,
            discovered_edges=42.0,
        ),
        protocol_summary_row(
            "uniform_random",
            discovery_rate=0.51,
            empty_scan_ratio=0.58,
            mean_discovery_delay=13.4,
            p90_discovery_delay=21.0,
            p95_discovery_delay=27.0,
            p99_discovery_delay=30.0,
            lcc_ratio=0.64,
            lambda2=0.08,
            collision_count=2.6,
            discovered_edges=24.0,
        ),
    ]
    write_csv(table_dir / "protocol_summary.csv", rows)

    slot_rows = []
    for protocol, edge_scale, lcc_start in [("itap_nd", 4.0, 0.5), ("uniform_random", 2.0, 0.3)]:
        for slot in range(6):
            slot_rows.append(
                {
                    "protocol": protocol,
                    "slot": slot,
                    "sample_count": 3,
                    "discovered_edges_mean": edge_scale * (slot + 1),
                    "lcc_ratio_mean": lcc_start + 0.05 * slot,
                }
            )
    write_csv(table_dir / "slot_protocol_summary.csv", slot_rows)


def protocol_summary_row(protocol: str, **metrics: float) -> dict[str, object]:
    row: dict[str, object] = {
        "protocol": protocol,
        "episode_count": 4,
        "run_count": 2,
    }
    for metric, value in metrics.items():
        row[f"{metric}_mean"] = value
        row[f"{metric}_ci95"] = value * 0.05
    return row


def write_training_history(training_dir: Path) -> None:
    training_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for generation in range(1, 8):
        rows.append(
            {
                "generation": generation,
                "elite_score": 10.0 + generation * 1.5,
                "mean_reward": 8.0 + generation,
                "discovery_rate": 0.35 + generation * 0.04,
                "empty_scan_ratio": 0.7 - generation * 0.035,
                "mean_discovery_delay": 22.0 - generation * 1.4,
                "collision_count": 4.0 - generation * 0.25,
                "lcc_ratio": 0.45 + generation * 0.05,
                "lambda2": 0.03 + generation * 0.025,
                "loss": 2.0 / generation,
            }
        )
    write_csv(training_dir / "training_history.csv", rows)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
