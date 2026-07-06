from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "05_simulation" / "run_marl_evaluate.py"
TRANSFER_PLOT_SCRIPT = ROOT / "06_analysis" / "scripts" / "plot_marl_transfer_results.py"
METHOD_PLOT_SCRIPT = ROOT / "06_analysis" / "scripts" / "plot_marl_method_comparison.py"
DEFAULT_CONFIG = ROOT / "05_simulation" / "configs" / "paper_transfer_train_n10_b10_singlehop.yaml"

BEAMWIDTH_TO_CELLS = {
    3: (120, 60),
    5: (72, 36),
    10: (36, 18),
    15: (24, 12),
    30: (12, 6),
}


@dataclass(frozen=True)
class MethodSpec:
    method: str
    checkpoint: Path
    reward_version: str
    env_protocol: str
    description: str


METHOD_SPECS = {
    "legacy_shared": MethodSpec(
        method="legacy_shared",
        checkpoint=ROOT
        / "05_simulation"
        / "results_raw"
        / "marl_campaign"
        / "phase7_long_training_100ep_3seed"
        / "train"
        / "train_n10_b10_legacy_shared_100ep_300slot_seed20260731"
        / "final_model.pt",
        reward_version="legacy",
        env_protocol="isac_structured_marl",
        description="Shared ISAC-MAPPO actor-critic with the legacy reward, 100-episode checkpoint.",
    ),
    "collision_reward": MethodSpec(
        method="collision_reward",
        checkpoint=ROOT
        / "05_simulation"
        / "results_raw"
        / "marl_campaign"
        / "phase7_long_training_100ep_3seed"
        / "train"
        / "train_n10_b10_collision_reward_100ep_300slot_seed20260731"
        / "final_model.pt",
        reward_version="collision_topology",
        env_protocol="isac_structured_marl",
        description="Shared ISAC-MAPPO actor-critic with collision/topology reward shaping, 100-episode checkpoint.",
    ),
    "contention_actor": MethodSpec(
        method="contention_actor",
        checkpoint=ROOT
        / "05_simulation"
        / "results_raw"
        / "marl_campaign"
        / "phase7_long_training_100ep_3seed"
        / "train"
        / "train_n10_b10_contention_actor_100ep_300slot_seed20260731"
        / "final_model.pt",
        reward_version="collision_topology",
        env_protocol="isac_structured_marl",
        description="Contention-aware shared ISAC-MAPPO actor-critic with collision/topology reward shaping, 100-episode checkpoint.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paper-grade long-horizon transfer evaluation for the fixed 300-slot-trained "
            "MARL checkpoints."
        )
    )
    parser.add_argument("--campaign", default="phase6_final_long_eval_10ep_stoch")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default="05_simulation/results_raw/marl_campaign")
    parser.add_argument("--analysis-output-root", default="06_analysis/paper_tables/marl")
    parser.add_argument("--figure-output-root", default="06_analysis/paper_figures/marl")
    parser.add_argument("--methods", nargs="+", choices=sorted(METHOD_SPECS), default=sorted(METHOD_SPECS))
    parser.add_argument("--legacy-shared-checkpoint", default=str(METHOD_SPECS["legacy_shared"].checkpoint))
    parser.add_argument("--collision-reward-checkpoint", default=str(METHOD_SPECS["collision_reward"].checkpoint))
    parser.add_argument("--contention-actor-checkpoint", default=str(METHOD_SPECS["contention_actor"].checkpoint))
    parser.add_argument("--node-counts", type=int, nargs="+", default=[100])
    parser.add_argument("--beamwidths", type=int, nargs="+", default=[3, 5, 10, 15, 30])
    parser.add_argument("--eval-slots", type=int, nargs="+", default=[3000])
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--communication-range", type=float, default=900.0)
    parser.add_argument("--sensing-range", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--resource-log-period", type=int, default=500)
    parser.add_argument("--max-rss-mb", type=float, default=10000.0)
    parser.add_argument("--max-system-memory-percent", type=float, default=90.0)
    parser.add_argument("--command-timeout-seconds", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=1, help="Bounded parallel eval subprocesses. Aggregation stays serial.")
    parser.add_argument("--eval-both", action="store_true", help="Run deterministic and stochastic evaluation.")
    parser.add_argument("--force", action="store_true", help="Re-run completed evaluation directories.")
    parser.add_argument("--no-aggregate", action="store_true", help="Do not refresh transfer and method-comparison tables.")
    parser.add_argument("--comparison-slots", type=int, default=3000)
    parser.add_argument("--comparison-node-count", type=int, default=100)
    parser.add_argument("--comparison-phase", default="eval_stochastic")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_beamwidths(args.beamwidths)
    output_root = Path(args.output_root) / args.campaign
    output_root.mkdir(parents=True, exist_ok=True)

    plan = build_plan(args, output_root)
    (output_root / "final_eval_campaign_plan.json").write_text(
        json.dumps(plan, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=True, indent=2))
        return

    missing = [item for item in plan["missing_checkpoints"] if item["method"] in set(args.methods)]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    run_commands(
        plan["eval_commands"],
        output_root / "final_eval_run_records.json",
        args,
        max_workers=max(1, int(args.max_workers)),
    )
    if not args.no_aggregate:
        run_commands(plan["aggregation_commands"], output_root / "final_eval_aggregation_records.json", args, max_workers=1)
    if not args.quiet:
        print(
            json.dumps(
                {
                    "status": "complete",
                    "campaign": args.campaign,
                    "output_root": str(output_root),
                    "eval_commands": len(plan["eval_commands"]),
                    "skipped_complete": len(plan["skipped_complete"]),
                    "aggregation_commands": len(plan["aggregation_commands"]),
                },
                ensure_ascii=True,
                indent=2,
            )
        )


def build_plan(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    eval_commands: list[list[str]] = []
    skipped_complete: list[dict[str, Any]] = []
    run_dirs_by_method: dict[str, list[str]] = {method: [] for method in args.methods}
    missing_checkpoints = []
    eval_modes = 2 if bool(args.eval_both) else 1
    expected_rows = int(args.eval_episodes) * eval_modes
    mode_tag = "both" if bool(args.eval_both) else "stoch"
    method_specs = method_specs_from_args(args)

    for method_index, method in enumerate(args.methods):
        spec = method_specs[method]
        if not spec.checkpoint.exists():
            missing_checkpoints.append({"method": method, "checkpoint": str(spec.checkpoint)})
        for slots in args.eval_slots:
            for node_count in args.node_counts:
                for beamwidth in args.beamwidths:
                    azimuth, elevation = beam_cells(beamwidth)
                    eval_name = (
                        f"{method}_train_n10_b10_test_n{node_count}_b{beamwidth}_{slots}slot_"
                        f"{args.eval_episodes}ep_{mode_tag}"
                    )
                    output = output_root / "eval" / method / eval_name
                    run_dirs_by_method[method].append(str(output))
                    expected = expected_manifest(
                        args=args,
                        spec=spec,
                        node_count=node_count,
                        beamwidth=beamwidth,
                        slots=slots,
                        azimuth=azimuth,
                        elevation=elevation,
                    )
                    if complete_eval_run(output, expected_rows, expected) and not bool(args.force):
                        skipped_complete.append(
                            {
                                "method": method,
                                "output": str(output),
                                "expected_rows": expected_rows,
                            }
                        )
                        continue
                    command = [
                        sys.executable,
                        str(EVAL_SCRIPT),
                        "--checkpoint",
                        str(spec.checkpoint),
                        "--config",
                        str(args.config),
                        "--output",
                        str(output),
                        "--eval-episodes",
                        str(args.eval_episodes),
                        "--slots",
                        str(slots),
                        "--node-count",
                        str(node_count),
                        "--azimuth-cells",
                        str(azimuth),
                        "--elevation-cells",
                        str(elevation),
                        "--communication-range",
                        f"{float(args.communication_range):g}",
                        "--sensing-range",
                        f"{float(args.sensing_range):g}",
                        "--seed",
                        str(eval_seed(args.seed, method_index, node_count, beamwidth, slots)),
                        "--torch-threads",
                        str(args.torch_threads),
                        "--reward-version",
                        spec.reward_version,
                        "--env-protocol",
                        spec.env_protocol,
                        "--resource-log-period",
                        str(args.resource_log_period),
                        "--max-rss-mb",
                        str(args.max_rss_mb),
                        "--max-system-memory-percent",
                        str(args.max_system_memory_percent),
                    ]
                    command.append("--eval-both" if bool(args.eval_both) else "--stochastic")
                    eval_commands.append(command)

    aggregation_commands = [] if bool(args.no_aggregate) else build_aggregation_commands(args, run_dirs_by_method)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "campaign": str(args.campaign),
        "train_slots_per_episode": 300,
        "train_node_count": 10,
        "train_beamwidth_deg": 10,
        "eval_slots": list(args.eval_slots),
        "eval_episodes": int(args.eval_episodes),
        "eval_modes": "deterministic_and_stochastic" if bool(args.eval_both) else "stochastic_only",
        "node_counts": list(args.node_counts),
        "beamwidths": list(args.beamwidths),
        "communication_range_m": float(args.communication_range),
        "sensing_range_m": float(args.sensing_range),
        "methods": {method: method_manifest(method_specs[method]) for method in args.methods},
        "resource_limits": {
            "torch_threads": int(args.torch_threads),
            "max_workers": int(args.max_workers),
            "resource_log_period": int(args.resource_log_period),
            "max_rss_mb": float(args.max_rss_mb),
            "max_system_memory_percent": float(args.max_system_memory_percent),
        },
        "missing_checkpoints": missing_checkpoints,
        "skipped_complete": skipped_complete,
        "eval_commands": eval_commands,
        "aggregation_commands": aggregation_commands,
    }


def build_aggregation_commands(args: argparse.Namespace, run_dirs_by_method: dict[str, list[str]]) -> list[list[str]]:
    commands: list[list[str]] = []
    analysis_root = Path(args.analysis_output_root)
    figure_root = Path(args.figure_output_root)
    summary_paths: dict[str, Path] = {}
    for method, run_dirs in run_dirs_by_method.items():
        if not run_dirs:
            continue
        output = analysis_root / f"{args.campaign}_{method}"
        figures = figure_root / f"{args.campaign}_{method}"
        command = [
            sys.executable,
            str(TRANSFER_PLOT_SCRIPT),
            "--output",
            str(output),
            "--figures",
            str(figures),
            "--quiet",
        ]
        for run_dir in run_dirs:
            command.extend(["--run-dir", run_dir])
        commands.append(command)
        summary_paths[method] = output / "marl_transfer_summary.csv"

    required = {"legacy_shared", "collision_reward", "contention_actor"}
    if required.issubset(summary_paths):
        comparison_output = analysis_root / f"{args.campaign}_method_comparison"
        comparison_figures = figure_root / f"{args.campaign}_method_comparison"
        command = [
            sys.executable,
            str(METHOD_PLOT_SCRIPT),
            "--legacy",
            str(summary_paths["legacy_shared"]),
            "--collision",
            str(summary_paths["collision_reward"]),
            "--contention",
            str(summary_paths["contention_actor"]),
            "--output",
            str(comparison_output),
            "--figures",
            str(comparison_figures),
            "--slots",
            str(args.comparison_slots),
            "--node-count",
            str(args.comparison_node_count),
            "--phase",
            str(args.comparison_phase),
            "--beamwidths",
        ]
        command.extend(str(value) for value in args.beamwidths)
        command.append("--quiet")
        commands.append(command)
    return commands


def run_commands(
    commands: list[list[str]],
    records_path: Path,
    args: argparse.Namespace,
    max_workers: int = 1,
) -> None:
    records_path.parent.mkdir(parents=True, exist_ok=True)
    if int(max_workers) <= 1 or len(commands) <= 1:
        run_commands_serial(commands, records_path, args)
        return
    run_commands_parallel(commands, records_path, args, int(max_workers))


def run_commands_serial(commands: list[list[str]], records_path: Path, args: argparse.Namespace) -> None:
    records: list[dict[str, Any]] = []
    timeout = int(args.command_timeout_seconds) if int(args.command_timeout_seconds) > 0 else None
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {' '.join(command)}", flush=True)
        try:
            completed = subprocess.run(command, cwd=ROOT, text=True, timeout=timeout)
            returncode: int | str = int(completed.returncode)
        except subprocess.TimeoutExpired:
            returncode = "timeout"
        records.append(
            {
                "index": index,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "command": command,
                "returncode": returncode,
            }
        )
        records_path.write_text(json.dumps(records, ensure_ascii=True, indent=2), encoding="utf-8")
        if returncode == "timeout":
            raise TimeoutError(f"Command timed out: {' '.join(command)}")
        if returncode != 0:
            raise SystemExit(returncode)


def run_commands_parallel(
    commands: list[list[str]],
    records_path: Path,
    args: argparse.Namespace,
    max_workers: int,
) -> None:
    validate_unique_outputs(commands)
    records: list[dict[str, Any]] = []
    timeout = int(args.command_timeout_seconds) if int(args.command_timeout_seconds) > 0 else None
    env = bounded_thread_env(args)
    pending = list(enumerate(commands, start=1))
    running: list[dict[str, Any]] = []
    while pending or running:
        while pending and len(running) < max_workers:
            index, command = pending.pop(0)
            output_dir = command_output_dir(command)
            output_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = output_dir / "subprocess_stdout.log"
            stderr_path = output_dir / "subprocess_stderr.log"
            stdout_handle = stdout_path.open("w", encoding="utf-8")
            stderr_handle = stderr_path.open("w", encoding="utf-8")
            print(f"[{index}/{len(commands)}] {' '.join(command)}", flush=True)
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                text=True,
                stdout=stdout_handle,
                stderr=stderr_handle,
                env=env,
            )
            running.append(
                {
                    "index": index,
                    "command": command,
                    "process": process,
                    "started_at": time.time(),
                    "stdout_handle": stdout_handle,
                    "stderr_handle": stderr_handle,
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                }
            )
        time.sleep(2.0)
        still_running: list[dict[str, Any]] = []
        for item in running:
            process: subprocess.Popen = item["process"]
            returncode = process.poll()
            timed_out = timeout is not None and (time.time() - float(item["started_at"])) > timeout
            if returncode is None and timed_out:
                process.kill()
                returncode = "timeout"
            if returncode is None:
                still_running.append(item)
                continue
            close_process_logs(item)
            record = {
                "index": item["index"],
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "command": item["command"],
                "returncode": returncode,
                "stdout": item["stdout"],
                "stderr": item["stderr"],
            }
            records.append(record)
            records_path.write_text(
                json.dumps(sorted(records, key=lambda row: row["index"]), ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
            if returncode == "timeout":
                terminate_running(still_running)
                raise TimeoutError(f"Command timed out: {' '.join(item['command'])}")
            if returncode != 0:
                terminate_running(still_running)
                raise SystemExit(returncode)
        running = still_running


def bounded_thread_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    threads = str(max(1, int(args.torch_threads)))
    env["OMP_NUM_THREADS"] = threads
    env["MKL_NUM_THREADS"] = threads
    env["OPENBLAS_NUM_THREADS"] = threads
    env["NUMEXPR_NUM_THREADS"] = threads
    return env


def validate_unique_outputs(commands: list[list[str]]) -> None:
    outputs = [command_output_dir(command) for command in commands if "--output" in command]
    duplicates = sorted({str(path) for path in outputs if outputs.count(path) > 1})
    if duplicates:
        raise ValueError(f"Duplicate output directories in parallel commands: {duplicates}")


def command_output_dir(command: list[str]) -> Path:
    try:
        return Path(command[command.index("--output") + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Command has no --output path: {command}") from exc


def close_process_logs(item: dict[str, Any]) -> None:
    item["stdout_handle"].close()
    item["stderr_handle"].close()


def terminate_running(running: list[dict[str, Any]]) -> None:
    for item in running:
        process: subprocess.Popen = item["process"]
        if process.poll() is None:
            process.kill()
        close_process_logs(item)


def complete_eval_run(output: Path, expected_rows: int, expected: dict[str, Any]) -> bool:
    manifest_path = output / "manifest.json"
    data_path = output / "eval_episode_metrics.csv"
    if not manifest_path.exists() or not data_path.exists() or data_path.stat().st_size == 0:
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return False
    if manifest.get("scope") != "marl_transfer_evaluation":
        return False
    if not manifest_matches(manifest, expected):
        return False
    resource_period = int(expected.get("resource_log_period", 0))
    slots = int(expected.get("slots_per_episode", 0))
    resource_path = output / "resource_log.csv"
    if resource_period > 0 and slots >= resource_period and (not resource_path.exists() or resource_path.stat().st_size == 0):
        return False
    try:
        with data_path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return False
    if len(rows) != int(expected_rows):
        return False
    phases = [str(row.get("phase", "")) for row in rows]
    if bool(expected["eval_both"]):
        return set(phases) == {"eval_deterministic", "eval_stochastic"}
    return bool(expected["stochastic"]) and phases and set(phases) == {"eval_stochastic"}


def expected_manifest(
    *,
    args: argparse.Namespace,
    spec: MethodSpec,
    node_count: int,
    beamwidth: int,
    slots: int,
    azimuth: int,
    elevation: int,
) -> dict[str, Any]:
    return {
        "checkpoint": str(spec.checkpoint),
        "eval_reward_version": spec.reward_version,
        "env_protocol": spec.env_protocol,
        "eval_episodes": int(args.eval_episodes),
        "slots_per_episode": int(slots),
        "node_count": int(node_count),
        "azimuth_cells": int(azimuth),
        "elevation_cells": int(elevation),
        "communication_range_m": float(args.communication_range),
        "sensing_range_m": float(args.sensing_range),
        "deterministic": False,
        "stochastic": not bool(args.eval_both),
        "eval_both": bool(args.eval_both),
        "resource_log_period": int(args.resource_log_period),
        "beamwidth_deg": int(beamwidth),
    }


def manifest_matches(manifest: dict[str, Any], expected: dict[str, Any]) -> bool:
    string_keys = ["checkpoint", "eval_reward_version", "env_protocol"]
    int_keys = ["eval_episodes", "slots_per_episode", "node_count", "azimuth_cells", "elevation_cells"]
    float_keys = ["communication_range_m", "sensing_range_m"]
    bool_keys = ["deterministic", "stochastic", "eval_both"]
    for key in string_keys:
        if str(manifest.get(key, "")) != str(expected[key]):
            return False
    for key in int_keys:
        try:
            if int(manifest.get(key)) != int(expected[key]):
                return False
        except (TypeError, ValueError):
            return False
    for key in float_keys:
        try:
            if abs(float(manifest.get(key)) - float(expected[key])) > 1e-9:
                return False
        except (TypeError, ValueError):
            return False
    for key in bool_keys:
        if bool(manifest.get(key)) != bool(expected[key]):
            return False
    return True


def method_manifest(spec: MethodSpec) -> dict[str, str]:
    return {
        "checkpoint": str(spec.checkpoint),
        "reward_version": spec.reward_version,
        "env_protocol": spec.env_protocol,
        "description": spec.description,
    }


def method_specs_from_args(args: argparse.Namespace) -> dict[str, MethodSpec]:
    return {
        "legacy_shared": MethodSpec(
            method="legacy_shared",
            checkpoint=resolve_repo_path(args.legacy_shared_checkpoint),
            reward_version=METHOD_SPECS["legacy_shared"].reward_version,
            env_protocol=METHOD_SPECS["legacy_shared"].env_protocol,
            description=METHOD_SPECS["legacy_shared"].description,
        ),
        "collision_reward": MethodSpec(
            method="collision_reward",
            checkpoint=resolve_repo_path(args.collision_reward_checkpoint),
            reward_version=METHOD_SPECS["collision_reward"].reward_version,
            env_protocol=METHOD_SPECS["collision_reward"].env_protocol,
            description=METHOD_SPECS["collision_reward"].description,
        ),
        "contention_actor": MethodSpec(
            method="contention_actor",
            checkpoint=resolve_repo_path(args.contention_actor_checkpoint),
            reward_version=METHOD_SPECS["contention_actor"].reward_version,
            env_protocol=METHOD_SPECS["contention_actor"].env_protocol,
            description=METHOD_SPECS["contention_actor"].description,
        ),
    }


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def eval_seed(base_seed: int, method_index: int, node_count: int, beamwidth: int, slots: int) -> int:
    return int(base_seed) + method_index * 1_000_000 + int(node_count) * 1_000 + int(beamwidth) * 100 + int(slots)


def validate_beamwidths(values: list[int]) -> None:
    unsupported = sorted({int(value) for value in values if int(value) not in BEAMWIDTH_TO_CELLS})
    if unsupported:
        raise ValueError(f"Unsupported beamwidths {unsupported}; expected one of {sorted(BEAMWIDTH_TO_CELLS)}.")


def beam_cells(beamwidth: int) -> tuple[int, int]:
    return BEAMWIDTH_TO_CELLS[int(beamwidth)]


if __name__ == "__main__":
    main()
