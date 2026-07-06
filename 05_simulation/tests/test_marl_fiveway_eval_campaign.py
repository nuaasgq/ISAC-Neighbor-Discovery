from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "05_simulation" / "run_marl_fiveway_eval_campaign.py"


def load_fiveway_module():
    spec = importlib.util.spec_from_file_location("run_marl_fiveway_eval_campaign", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fiveway_plan_uses_paired_scenario_seeds_and_unified_outputs(tmp_path: Path) -> None:
    module = load_fiveway_module()
    assert "phase7_contention_no_isac_strict_100ep_3seed" in str(module.DEFAULT_CONTENTION_NO_ISAC)
    args = Namespace(
        campaign="unit_fiveway",
        config=str(ROOT / "05_simulation" / "configs" / "paper_transfer_train_n10_b10_singlehop.yaml"),
        output_root=str(tmp_path / "raw"),
        analysis_output_root=str(tmp_path / "tables"),
        figure_output_root=str(tmp_path / "figures"),
        methods=["uniform_random", "skyorbs_like", "mappo_no_isac", "contention_no_isac", "contention_actor"],
        mappo_no_isac_checkpoint="missing_mappo.pt",
        contention_no_isac_checkpoint="missing_contention_no_isac.pt",
        contention_actor_checkpoint="missing_contention_actor.pt",
        node_counts=[100],
        beamwidths=[10],
        eval_slots=[3000],
        eval_episodes=10,
        area_scale="config",
        base_area_size_m=None,
        train_node_count=10,
        communication_range=900.0,
        sensing_range=900.0,
        seed=20364205,
        torch_threads=2,
        resource_log_period=500,
        max_rss_mb=10000.0,
        max_system_memory_percent=90.0,
        command_timeout_seconds=0,
        comparison_slots=3000,
        comparison_node_count=100,
        comparison_phase="eval_stochastic",
        force=False,
        no_aggregate=False,
        dry_run=True,
        quiet=True,
    )

    plan = module.build_plan(args, tmp_path / "raw" / "unit_fiveway")

    assert len(plan["eval_commands"]) == 5
    assert len(plan["aggregation_commands"]) == 2
    assert len(plan["missing_checkpoints"]) == 3

    seeds = {command[command.index("--seed") + 1] for command in plan["eval_commands"]}
    assert seeds == {"20468205"}

    outputs = [command[command.index("--output") + 1] for command in plan["eval_commands"]]
    assert len(outputs) == len(set(outputs))
    assert any("uniform_random_train_n10_b10_test_n100_b10_3000slot_10ep_stoch" in value for value in outputs)
    assert any("contention_actor_train_n10_b10_test_n100_b10_3000slot_10ep_stoch" in value for value in outputs)

    skyorbs_command = next(command for command in plan["eval_commands"] if "skyorbs_like" in command[command.index("--output") + 1])
    assert skyorbs_command[skyorbs_command.index("--protocols") + 1] == "skyorbs_like_skip_scan"

    neural_commands = [command for command in plan["eval_commands"] if str(module.EVAL_SCRIPT) in command]
    assert len(neural_commands) == 3
    assert all("--stochastic" in command for command in neural_commands)


def test_fiveway_density_area_scale_passes_case_area_to_children(tmp_path: Path) -> None:
    module = load_fiveway_module()
    args = Namespace(
        campaign="unit_density",
        config=str(ROOT / "05_simulation" / "configs" / "paper_transfer_train_n10_b10_singlehop.yaml"),
        output_root=str(tmp_path / "raw"),
        analysis_output_root=str(tmp_path / "tables"),
        figure_output_root=str(tmp_path / "figures"),
        methods=["uniform_random"],
        mappo_no_isac_checkpoint="missing_mappo.pt",
        contention_no_isac_checkpoint="missing_contention_no_isac.pt",
        contention_actor_checkpoint="missing_contention_actor.pt",
        node_counts=[10, 100],
        beamwidths=[10],
        eval_slots=[3000],
        eval_episodes=10,
        area_scale="density",
        base_area_size_m=[260.0, 260.0, 90.0],
        train_node_count=10,
        communication_range=900.0,
        sensing_range=900.0,
        seed=20364205,
        torch_threads=2,
        resource_log_period=500,
        max_rss_mb=10000.0,
        max_system_memory_percent=90.0,
        command_timeout_seconds=0,
        comparison_slots=3000,
        comparison_node_count=100,
        comparison_phase="eval_stochastic",
        force=False,
        no_aggregate=False,
        dry_run=True,
        quiet=True,
    )

    plan = module.build_plan(args, tmp_path / "raw" / "unit_density")

    assert plan["area_scale"] == "density"
    assert len(plan["eval_commands"]) == 2
    n10_command = next(command for command in plan["eval_commands"] if "_test_n10_b10_" in command[command.index("--output") + 1])
    n100_command = next(command for command in plan["eval_commands"] if "_test_n100_b10_" in command[command.index("--output") + 1])
    assert n10_command[n10_command.index("--area-size-m") + 1 : n10_command.index("--area-size-m") + 4] == [
        "260",
        "260",
        "90",
    ]
    n100_area = [float(value) for value in n100_command[n100_command.index("--area-size-m") + 1 : n100_command.index("--area-size-m") + 4]]
    factor = 10.0 ** (1.0 / 3.0)
    assert n100_area == [float(f"{260.0 * factor:g}"), float(f"{260.0 * factor:g}"), float(f"{90.0 * factor:g}")]
    assert "_area_density" in n100_command[n100_command.index("--output") + 1]
