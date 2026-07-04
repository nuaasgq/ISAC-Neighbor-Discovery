from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ANALYZER = ROOT / "06_analysis" / "scripts" / "analyze_dynamic_results.py"


def test_dynamic_analysis_writes_tables_and_skips_figures_without_matplotlib(tmp_path: Path) -> None:
    sweep_root = tmp_path / "sweep"
    run_a = sweep_root / "run_a"
    run_b = sweep_root / "nested" / "run_b"
    write_runner_like_run(run_a, "gauss_markov", scale=1.0)
    write_runner_like_run(run_b, "random_walk", scale=1.2)

    output_root = tmp_path / "analysis"
    result = subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            str(sweep_root),
            "--output",
            str(output_root),
            "--no-plots",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout_manifest = json.loads(result.stdout)
    assert stdout_manifest["input_run_count"] == 2
    assert stdout_manifest["episode_rows"] == 8

    protocol_summary = output_root / "tables" / "protocol_summary.csv"
    mobility_summary = output_root / "tables" / "protocol_mobility_summary.csv"
    slot_summary = output_root / "tables" / "slot_protocol_summary.csv"
    source_manifest = output_root / "tables" / "source_runs_manifest.csv"
    figure_manifest = output_root / "figures" / "figure_manifest.json"

    for path in [protocol_summary, mobility_summary, slot_summary, source_manifest, figure_manifest]:
        assert path.exists(), path

    rows = read_csv(protocol_summary)
    protocols = {row["protocol"] for row in rows}
    assert protocols == {"itap_nd", "uniform_random"}
    assert {"episode_count", "discovery_rate_mean", "empty_scan_ratio_mean", "p95_discovery_delay_mean"}.issubset(rows[0])

    itap = next(row for row in rows if row["protocol"] == "itap_nd")
    random = next(row for row in rows if row["protocol"] == "uniform_random")
    assert float(itap["discovery_rate_mean"]) > float(random["discovery_rate_mean"])
    assert float(itap["empty_scan_ratio_mean"]) < float(random["empty_scan_ratio_mean"])

    slot_rows = read_csv(slot_summary)
    assert {"protocol", "slot", "lambda2_mean", "lcc_ratio_mean"}.issubset(slot_rows[0])
    assert len(slot_rows) == 6

    figures = json.loads(figure_manifest.read_text(encoding="utf-8"))
    assert figures
    assert {item["status"] for item in figures} == {"skipped"}


def write_runner_like_run(run_dir: Path, mobility: str, scale: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    episode_rows = []
    for episode in range(2):
        episode_rows.append(
            {
                "protocol": "uniform_random",
                "episode": episode,
                "seed": 100 + episode,
                "slots": 30,
                "mobility_model": mobility,
                "true_edges_seen": 10,
                "discovered_edges": 3,
                "discovery_rate": 0.3 / scale,
                "mean_delay_censored": 19.0 * scale,
                "p95_delay_censored": 28.0 * scale,
                "empty_scan_ratio": 0.72 * scale,
                "collision_count": 3,
                "moved_distance_mean_m": 40.0 * scale,
                "largest_component_size": 4,
                "connected_components": 5,
                "lcc_ratio": 0.5,
                "isolated_node_ratio": 0.25,
                "lambda2": 0.0,
            }
        )
        episode_rows.append(
            {
                "protocol": "itap_nd",
                "episode": episode,
                "seed": 200 + episode,
                "slots": 30,
                "mobility_model": mobility,
                "true_edges_seen": 10,
                "discovered_edges": 7,
                "discovery_rate": 0.7 / scale,
                "mean_delay_censored": 11.0 * scale,
                "p95_delay_censored": 18.0 * scale,
                "empty_scan_ratio": 0.34 * scale,
                "collision_count": 1,
                "moved_distance_mean_m": 40.0 * scale,
                "largest_component_size": 7,
                "connected_components": 2,
                "lcc_ratio": 0.875,
                "isolated_node_ratio": 0.0,
                "lambda2": 0.25,
            }
        )
    write_csv(run_dir / "per_episode_summary.csv", episode_rows)

    slot_rows = []
    for protocol, base_rate, base_lcc, base_lambda2 in [
        ("uniform_random", 1, 0.25, 0.0),
        ("itap_nd", 2, 0.5, 0.1),
    ]:
        for slot in range(3):
            slot_rows.append(
                {
                    "episode": 0,
                    "protocol": protocol,
                    "slot": slot,
                    "mobility_model": mobility,
                    "true_edges": 8,
                    "true_edges_seen": 8 + slot,
                    "discovered_edges": base_rate * slot,
                    "new_edges": base_rate,
                    "empty_scan_ratio": 0.6 / max(1, base_rate),
                    "collision_count": 0,
                    "largest_component_size": 2 + base_rate * slot,
                    "connected_components": 8 - slot,
                    "lcc_ratio": base_lcc + 0.05 * slot,
                    "isolated_node_ratio": 0.5,
                    "lambda2": base_lambda2 + 0.05 * slot,
                }
            )
    write_csv(run_dir / "per_slot_metrics.csv", slot_rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
