from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from isac_nd_sim.training import train_from_config


def test_training_pipeline_writes_required_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "training"

    result = train_from_config(
        "05_simulation/configs/mobile_smoke.yaml",
        output_dir=output_dir,
        generations=1,
        population=2,
        episodes=1,
        slots_per_episode=8,
        seeds=[101],
        test_seeds=[202],
        test_episodes=1,
        training_seed=303,
    )

    assert Path(result["run_dir"]) == output_dir
    for file_name in [
        "training_history.csv",
        "elite_history.csv",
        "best_config.yaml",
        "test_summary.csv",
        "manifest.json",
    ]:
        assert (output_dir / file_name).exists()

    with (output_dir / "training_history.csv").open("r", encoding="utf-8") as handle:
        training_rows = list(csv.DictReader(handle))
    assert len(training_rows) == 2
    assert {
        "generation",
        "candidate",
        "score",
        "alpha_occupancy",
        "softmax_beta",
        "exploration_floor",
        "confidence_decay",
        "piggyback_sensing_period_multiplier",
        "discovery_rate_mean",
    }.issubset(training_rows[0].keys())

    with (output_dir / "test_summary.csv").open("r", encoding="utf-8") as handle:
        test_rows = list(csv.DictReader(handle))
    assert {row["seed"] for row in test_rows} == {"202", "all"}

    best_config = yaml.safe_load((output_dir / "best_config.yaml").read_text(encoding="utf-8"))
    assert best_config["trained_protocol"] == "improved_rl_isac"
    assert "shared_policy_parameters" in best_config

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["trained_protocol"] == "improved_rl_isac"
    assert manifest["settings"]["slots_per_episode"] == 8
