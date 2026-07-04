from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PLOTTER = ROOT / "06_analysis" / "scripts" / "plot_mobility_results.py"


def test_mobility_plotter_writes_manifest_and_4x3_figures(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    image_module = pytest.importorskip("PIL.Image")

    sweep_dir = tmp_path / "round5_mobility"
    output_dir = tmp_path / "mobility_figures"
    write_aggregate(sweep_dir, mobility_rows())

    result = subprocess.run(
        [
            sys.executable,
            str(PLOTTER),
            str(sweep_dir),
            "--output",
            str(output_dir),
            "--node-count",
            "100",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout_manifest = json.loads(result.stdout)
    manifest_path = output_dir / "mobility_figure_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stdout_manifest["counts"] == manifest["counts"]
    assert manifest["style"]["aspect_ratio"] == "4:3"
    assert manifest["counts"]["generated"] >= 2

    expected = output_dir / "mobility_discovery_n100_b10.png"
    assert expected.exists()
    with image_module.open(expected) as image:
        assert image.size == (1920, 1440)
        assert image.size[0] / image.size[1] == pytest.approx(4 / 3)


def mobility_rows() -> list[dict[str, object]]:
    rows = []
    for mobility_idx, mobility in enumerate(("gauss_markov", "random_walk", "random_direction")):
        for beamwidth in (10.0, 15.0):
            for protocol_idx, protocol in enumerate(
                ("improved_rl_no_isac", "ablation_isac_one_slot_delay", "improved_rl_isac")
            ):
                discovery = 0.02 + 0.04 * mobility_idx + 0.02 * (beamwidth / 10.0) + 0.12 * protocol_idx
                rows.append(
                    {
                        "protocol": protocol,
                        "mobility_model": mobility,
                        "area_scale": "density",
                        "node_count": 100,
                        "beamwidth_deg": beamwidth,
                        "discovery_rate_mean": discovery,
                        "empty_scan_ratio_mean": 0.8 - discovery,
                        "lambda2_mean": discovery * 20,
                        "collision_count_mean": discovery * 1000,
                        "discovered_edges_mean": discovery * 4950,
                        "true_edges_seen_mean": 4950,
                        "collision_penalized_discovery_rate_mean": discovery / (1.0 + discovery),
                    }
                )
    return rows


def write_aggregate(directory: Path, rows: list[dict[str, object]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "aggregate_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
