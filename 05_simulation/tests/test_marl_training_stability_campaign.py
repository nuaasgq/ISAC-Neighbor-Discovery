from __future__ import annotations

import csv
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "05_simulation" / "run_marl_training_stability_campaign.py"


def load_training_stability_module():
    spec = importlib.util.spec_from_file_location("run_marl_training_stability_campaign", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_episode_rows(path: Path, rows: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", "slots"])
        writer.writeheader()
        for episode in range(rows):
            writer.writerow({"episode": episode, "slots": 300})


def test_complete_training_run_accepts_legacy_manifest_without_seed(tmp_path: Path) -> None:
    module = load_training_stability_module()
    output = tmp_path / "complete"
    output.mkdir()
    (output / "final_model.pt").write_bytes(b"checkpoint")
    write_episode_rows(output / "episode_metrics.csv", 100)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "episodes": 100,
                "slots_per_episode": 300,
                "network": "shared",
                "reward_version": "legacy",
                "env_protocol": "isac_structured_marl",
            }
        ),
        encoding="utf-8",
    )

    assert module.complete_training_run(output, 100, 300, module.METHODS["legacy_shared"], 20260731)


def test_complete_training_run_rejects_wrong_seed_when_present(tmp_path: Path) -> None:
    module = load_training_stability_module()
    output = tmp_path / "wrong_seed"
    output.mkdir()
    (output / "final_model.pt").write_bytes(b"checkpoint")
    write_episode_rows(output / "episode_metrics.csv", 100)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "episodes": 100,
                "slots_per_episode": 300,
                "seed": 20260732,
                "network": "shared",
                "reward_version": "legacy",
                "env_protocol": "isac_structured_marl",
            }
        ),
        encoding="utf-8",
    )

    assert not module.complete_training_run(output, 100, 300, module.METHODS["legacy_shared"], 20260731)


def test_build_plan_includes_budgeted_expert_bc_sweep(tmp_path: Path) -> None:
    module = load_training_stability_module()
    args = Namespace(
        campaign="budgeted_gate_bc",
        config="05_simulation/configs/paper_transfer_train_n10_b10_singlehop.yaml",
        methods=["balanced_topology_gated_contention_actor"],
        seeds=[20260751],
        episodes=100,
        slots=300,
        eval_episodes=3,
        eval_interval=10,
        checkpoint_interval=50,
        hidden_dim=64,
        ppo_epochs=2,
        expert_bc_weights=[0.15, 0.30],
        expert_protocol="budgeted_collision_aware_isac",
        torch_threads=1,
        step_log_period=1,
        resource_log_period=100,
        max_rss_mb=10000.0,
        max_system_memory_percent=90.0,
    )

    plan = module.build_plan(args, tmp_path / "train", tmp_path / "logs")

    assert len(plan["runs"]) == 2
    assert plan["expert_bc_weights"] == [0.15, 0.30]
    assert plan["expert_protocol"] == "budgeted_collision_aware_isac"
    assert "bc0p15_budgeted_collision_aware_isac" in plan["runs"][0]["run_name"]
    assert "--expert-bc-weight" in plan["runs"][0]["command"]
    assert "budgeted_collision_aware_isac" in plan["runs"][0]["command"]
