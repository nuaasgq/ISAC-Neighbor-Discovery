from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from isac_nd_sim.config import load_config
from isac_nd_sim.sweep import iter_sweep_cases, run_sweep_from_config


def write_test_sweep_config(path: Path) -> None:
    config = {
        "name": "pytest_dynamic_rule_sweep",
        "base_config": "05_simulation/configs/mobile_smoke.yaml",
        "protocols": ["uniform_random", "itap_nd"],
        "sweep": {
            "episodes": 1,
            "seeds": [101, 202],
            "mobility": ["gauss_markov", "random_walk"],
            "slots": [6],
            "node_counts": [6],
            "beam_cells": [{"azimuth_cells": 6, "elevation_cells": 3}],
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_iter_sweep_cases_expands_expected_grid(tmp_path: Path) -> None:
    config_path = tmp_path / "sweep.yaml"
    write_test_sweep_config(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = load_config("05_simulation/configs/mobile_smoke.yaml")

    cases = list(iter_sweep_cases(raw, base))

    assert len(cases) == 4
    assert {case.mobility_model for case in cases} == {"gauss_markov", "random_walk"}
    assert {case.base_seed for case in cases} == {101, 202}
    assert all(case.node_count == 6 for case in cases)
    assert all(case.beam_cells == 18 for case in cases)


def test_run_sweep_writes_episode_and_aggregate_outputs(tmp_path: Path) -> None:
    config_path = tmp_path / "sweep.yaml"
    output_dir = tmp_path / "results"
    write_test_sweep_config(config_path)

    result = run_sweep_from_config(config_path, output_dir)

    assert result["case_count"] == 4
    assert result["episode_rows"] == 8
    assert result["slot_rows"] == 48
    assert result["aggregate_rows"] == 4
    for file_name in [
        "sweep_config.yaml",
        "base_config_path.txt",
        "per_episode_summary.csv",
        "per_slot_metrics.csv",
        "discovered_edges.csv",
        "aggregate_metrics.csv",
        "aggregate_metrics.json",
        "README.md",
    ]:
        assert (output_dir / file_name).exists()

    with (output_dir / "per_episode_summary.csv").open("r", encoding="utf-8") as handle:
        episode_rows = list(csv.DictReader(handle))
    assert len(episode_rows) == 8
    required_episode_fields = {
        "case_id",
        "base_seed",
        "protocol",
        "mobility_model",
        "node_count",
        "azimuth_cells",
        "elevation_cells",
        "beam_cells",
        "finite_time_discovery_rate",
        "mean_discovery_delay",
        "p95_discovery_delay",
    }
    assert required_episode_fields.issubset(episode_rows[0].keys())
    assert {row["protocol"] for row in episode_rows} == {"uniform_random", "itap_nd"}
    assert {row["mobility_model"] for row in episode_rows} == {"gauss_markov", "random_walk"}

    with (output_dir / "per_slot_metrics.csv").open("r", encoding="utf-8") as handle:
        slot_rows = list(csv.DictReader(handle))
    assert len(slot_rows) == 48
    required_slot_fields = {
        "case_id",
        "base_seed",
        "protocol",
        "slot",
        "mobility_model",
        "node_count",
        "true_edges",
        "new_edges",
        "lambda2",
    }
    assert required_slot_fields.issubset(slot_rows[0].keys())

    with (output_dir / "aggregate_metrics.csv").open("r", encoding="utf-8") as handle:
        aggregate_rows = list(csv.DictReader(handle))
    assert len(aggregate_rows) == 4
    required_aggregate_fields = {
        "protocol",
        "mobility_model",
        "node_count",
        "slots",
        "azimuth_cells",
        "elevation_cells",
        "n_episodes",
        "discovery_rate_mean",
        "empty_scan_ratio_mean",
        "lambda2_mean",
    }
    assert required_aggregate_fields.issubset(aggregate_rows[0].keys())
    assert {row["n_episodes"] for row in aggregate_rows} == {"2"}

    aggregate_json = json.loads((output_dir / "aggregate_metrics.json").read_text(encoding="utf-8"))
    assert len(aggregate_json) == 4
    assert {row["protocol"] for row in aggregate_json} == {"uniform_random", "itap_nd"}
