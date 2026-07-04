from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PLOTTER = ROOT / "06_analysis" / "scripts" / "plot_round3_results.py"


def test_round3_plotter_writes_robustness_manifest(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    image_module = pytest.importorskip("PIL.Image")

    range_dir = tmp_path / "round3_range"
    error_dir = tmp_path / "round3_error"
    ablation_dir = tmp_path / "round3_ablation"
    output_dir = tmp_path / "round3_figures"
    write_aggregate(range_dir, range_rows())
    write_aggregate(error_dir, error_rows())
    write_aggregate(ablation_dir, ablation_rows())

    result = subprocess.run(
        [
            sys.executable,
            str(PLOTTER),
            str(range_dir),
            str(error_dir),
            str(ablation_dir),
            "--output",
            str(output_dir),
            "--node-count",
            "100",
            "--beamwidth-deg",
            "10",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout_manifest = json.loads(result.stdout)
    manifest_path = output_dir / "round3_figure_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stdout_manifest["counts"] == manifest["counts"]
    assert manifest["style"]["aspect_ratio"] == "4:3"
    assert manifest["counts"]["generated"] >= 6

    expected = output_dir / "range_gain_discovery_n100_b10.png"
    assert expected.exists()
    with image_module.open(expected) as image:
        assert image.size == (1920, 1440)
        assert image.size[0] / image.size[1] == pytest.approx(4 / 3)


def base_row(protocol: str, rc: float = 1.05, rs: float = 1.0, pfa: float = 0.0, pmd: float = 0.0, sigma: float = 0.0) -> dict[str, object]:
    protocol_boost = 0.0
    if protocol == "improved_rl_no_isac":
        protocol_boost = 0.08
    elif protocol == "improved_rl_isac":
        protocol_boost = 0.18
    elif protocol.startswith("ablation_"):
        protocol_boost = 0.12
    discovery = 0.35 + 0.2 * rc + 0.05 * rs + protocol_boost - 0.5 * pfa - 0.2 * pmd - 0.04 * sigma
    return {
        "protocol": protocol,
        "mobility_model": "gauss_markov",
        "area_scale": "density",
        "range_mode": "singlehop",
        "range_sweep_enabled": True,
        "node_count": 100,
        "beamwidth_deg": 10.0,
        "azimuth_cells": 36,
        "elevation_cells": 18,
        "beam_count": 648,
        "communication_range_to_diagonal_ratio": rc,
        "sensing_to_comm_range_ratio": rs,
        "false_alarm_rate": pfa,
        "miss_detection_rate": pmd,
        "angular_cell_offset_std": sigma,
        "n_episodes": 3,
        "discovery_rate_mean": discovery,
        "empty_scan_ratio_mean": 0.75 - discovery / 2,
        "lambda2_mean": discovery * 20,
        "mean_discovery_delay_mean": 600 * (1.0 - discovery),
        "collision_count_mean": 80 * discovery,
        "discovered_edges_mean": 4950 * discovery,
        "scan_actions_mean": 60000 * (0.8 + 0.2 * (1.0 - discovery)),
        "discovery_per_scan_action_mean": discovery / 12.0,
        "scan_actions_per_discovery_censored_mean": 12.0 / max(discovery, 1e-6),
        "collision_normalized_efficiency_mean": discovery / (discovery + 0.2),
        "collision_penalized_discovery_rate_mean": discovery / (1.0 + 0.2 * discovery),
    }


def range_rows() -> list[dict[str, object]]:
    rows = []
    for rc in (0.65, 0.85, 1.05):
        for rs in (0.5, 0.75, 1.0, 1.25):
            for protocol in ("uniform_random", "improved_rl_no_isac", "improved_rl_isac"):
                rows.append(base_row(protocol, rc=rc, rs=rs))
    return rows


def error_rows() -> list[dict[str, object]]:
    rows = []
    for pfa, pmd, sigma in ((0.0, 0.0, 0.0), (0.01, 0.05, 0.0), (0.05, 0.15, 0.0)):
        for protocol in ("improved_rl_no_isac", "improved_rl_isac"):
            rows.append(base_row(protocol, pfa=pfa, pmd=pmd, sigma=sigma))
    return rows


def ablation_rows() -> list[dict[str, object]]:
    return [
        base_row(protocol)
        for protocol in (
            "uniform_random",
            "improved_rl_no_isac",
            "ablation_isac_no_topology",
            "ablation_isac_no_beam_lock",
            "ablation_isac_one_slot_delay",
            "ablation_isac_no_candidate_set",
            "improved_rl_isac",
        )
    ]


def write_aggregate(directory: Path, rows: list[dict[str, object]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "aggregate_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
