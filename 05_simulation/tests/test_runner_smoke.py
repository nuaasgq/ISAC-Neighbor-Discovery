from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

from isac_nd_sim.config import load_config
from isac_nd_sim.runner import run_detailed, write_outputs


def test_mobile_smoke_runner_writes_required_outputs(tmp_path: Path) -> None:
    cfg = load_config("05_simulation/configs/mvp.yaml")
    cfg = replace(
        cfg,
        episodes=1,
        slots_per_episode=12,
        n_nodes=8,
        azimuth_cells=8,
        elevation_cells=4,
        baselines=("uniform_random", "itap_nd"),
    )
    rows, slot_rows, edge_rows = run_detailed(cfg, list(cfg.baselines))
    write_outputs(Path("05_simulation/configs/mvp.yaml"), tmp_path, rows, cfg, slot_rows, edge_rows)

    for file_name in [
        "config.yaml",
        "seed_manifest.json",
        "per_episode_summary.csv",
        "per_slot_metrics.csv",
        "discovered_edges.csv",
        "aggregate_metrics.json",
        "README.md",
    ]:
        assert (tmp_path / file_name).exists()

    with (tmp_path / "per_episode_summary.csv").open("r", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert len(summary_rows) == 2
    required = {
        "protocol",
        "mobility_model",
        "mean_delay_censored",
        "p95_delay_censored",
        "discovery_rate",
        "empty_scan_ratio",
        "largest_component_size",
        "lcc_ratio",
        "lambda2",
    }
    assert required.issubset(summary_rows[0].keys())
    assert summary_rows[0]["mobility_model"] != "static"

    aggregate = json.loads((tmp_path / "aggregate_metrics.json").read_text(encoding="utf-8"))
    assert "uniform_random" in aggregate
    assert "itap_nd" in aggregate

    with (tmp_path / "per_slot_metrics.csv").open("r", encoding="utf-8") as handle:
        slot_metric_rows = list(csv.DictReader(handle))
    assert len(slot_metric_rows) == 24
    assert {"slot", "protocol", "new_edges", "lambda2"}.issubset(slot_metric_rows[0].keys())
