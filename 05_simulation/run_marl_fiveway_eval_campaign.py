from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "05_simulation" / "run_marl_evaluate.py"
PROTOCOL_EVAL_SCRIPT = ROOT / "05_simulation" / "run_protocol_baseline_eval.py"
TRANSFER_PLOT_SCRIPT = ROOT / "06_analysis" / "scripts" / "plot_marl_transfer_results.py"
METHOD_PLOT_SCRIPT = ROOT / "06_analysis" / "scripts" / "plot_marl_method_comparison.py"
DEFAULT_CONFIG = ROOT / "05_simulation" / "configs" / "paper_transfer_train_n10_b10_singlehop.yaml"

DEFAULT_MAPP0_NO_ISAC = (
    ROOT
    / "05_simulation"
    / "results_raw"
    / "marl_campaign"
    / "phase1_short_train_long_eval"
    / "train"
    / "train_n10_b10_mappo_300slot"
    / "final_model.pt"
)
DEFAULT_CONTENTION_NO_ISAC = (
    ROOT
    / "05_simulation"
    / "results_raw"
    / "marl_campaign"
    / "phase7_contention_no_isac_strict_100ep_3seed"
    / "train"
    / "train_n10_b10_contention_no_isac_100ep_300slot_seed20260731"
    / "final_model.pt"
)
DEFAULT_CONTENTION_ACTOR = (
    ROOT
    / "05_simulation"
    / "results_raw"
    / "marl_campaign"
    / "phase5_contention_shared_v2_train"
    / "train"
    / "train_n10_b10_isac_mappo_contention_shared_collision_topology_300slot"
    / "final_model.pt"
)

BEAMWIDTH_TO_CELLS = {
    3: (120, 60),
    5: (72, 36),
    10: (36, 18),
    15: (24, 12),
    30: (12, 6),
}


@dataclass(frozen=True)
class NeuralMethod:
    method: str
    checkpoint: Path
    reward_version: str
    env_protocol: str
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a five-way MARL-compatible comparison campaign: random, SkyOrbs-like, "
            "MAPPO no-ISAC, contention no-ISAC, and contention-aware ISAC-MARL."
        )
    )
    parser.add_argument("--campaign", default="phase8_fiveway_n100_b10_b30_3000slot_10ep_stoch")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default="05_simulation/results_raw/marl_campaign")
    parser.add_argument("--analysis-output-root", default="06_analysis/paper_tables/marl")
    parser.add_argument("--figure-output-root", default="06_analysis/paper_figures/marl")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["uniform_random", "skyorbs_like", "mappo_no_isac", "contention_no_isac", "contention_actor"],
        choices=["uniform_random", "skyorbs_like", "mappo_no_isac", "contention_no_isac", "contention_actor"],
    )
    parser.add_argument("--mappo-no-isac-checkpoint", default=str(DEFAULT_MAPP0_NO_ISAC))
    parser.add_argument("--contention-no-isac-checkpoint", default=str(DEFAULT_CONTENTION_NO_ISAC))
    parser.add_argument("--contention-actor-checkpoint", default=str(DEFAULT_CONTENTION_ACTOR))
    parser.add_argument("--node-counts", type=int, nargs="+", default=[100])
    parser.add_argument("--beamwidths", type=int, nargs="+", default=[10, 15, 30])
    parser.add_argument("--eval-slots", type=int, nargs="+", default=[3000])
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--communication-range", type=float, default=900.0)
    parser.add_argument("--sensing-range", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=20364205)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--resource-log-period", type=int, default=500)
    parser.add_argument("--max-rss-mb", type=float, default=10000.0)
    parser.add_argument("--max-system-memory-percent", type=float, default=90.0)
    parser.add_argument("--command-timeout-seconds", type=int, default=0)
    parser.add_argument("--comparison-slots", type=int, default=3000)
    parser.add_argument("--comparison-node-count", type=int, default=100)
    parser.add_argument("--comparison-phase", default="eval_stochastic")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_root = Path(args.output_root) / str(args.campaign)
    output_root.mkdir(parents=True, exist_ok=True)

    plan = build_plan(args, output_root)
    plan_path = output_root / "fiveway_eval_campaign_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=True, indent=2))
        return

    missing = [item for item in plan["missing_checkpoints"] if item["method"] in set(args.methods)]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    run_commands(plan["eval_commands"], output_root / "fiveway_eval_run_records.json", args)
    if not bool(args.no_aggregate):
        run_commands(plan["aggregation_commands"], output_root / "fiveway_eval_aggregation_records.json", args)
    if not bool(args.quiet):
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


def validate_args(args: argparse.Namespace) -> None:
    unsupported = sorted({int(value) for value in args.beamwidths if int(value) not in BEAMWIDTH_TO_CELLS})
    if unsupported:
        raise ValueError(f"Unsupported beamwidths {unsupported}; expected one of {sorted(BEAMWIDTH_TO_CELLS)}.")
    if int(args.eval_episodes) <= 0:
        raise ValueError("--eval-episodes must be positive.")
    if any(int(value) <= 0 for value in args.eval_slots):
        raise ValueError("--eval-slots values must be positive.")


def build_plan(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    methods = list(args.methods)
    missing_checkpoints: list[dict[str, str]] = []
    eval_commands: list[list[str]] = []
    skipped_complete: list[dict[str, Any]] = []
    run_dirs: list[str] = []
    neural_methods = neural_method_specs(args)

    for slots in args.eval_slots:
        for node_count in args.node_counts:
            for beamwidth in args.beamwidths:
                for method in methods:
                    output = output_root / "eval" / method / eval_name(method, node_count, beamwidth, slots, args.eval_episodes)
                    run_dirs.append(str(output))
                    if complete_eval_run(output, int(args.eval_episodes)) and not bool(args.force):
                        skipped_complete.append({"method": method, "output": str(output), "expected_rows": int(args.eval_episodes)})
                        continue
                    if method in {"uniform_random", "skyorbs_like"}:
                        eval_commands.append(protocol_command(method, args, output, node_count, beamwidth, slots))
                    else:
                        spec = neural_methods[method]
                        if not spec.checkpoint.exists():
                            missing_checkpoints.append({"method": method, "checkpoint": str(spec.checkpoint)})
                        eval_commands.append(neural_command(spec, args, output, node_count, beamwidth, slots))

    aggregation_commands = [] if bool(args.no_aggregate) else aggregation_commands_for(args, run_dirs)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "campaign": str(args.campaign),
        "train_node_count": 10,
        "train_beamwidth_deg": 10,
        "train_slots_per_episode": 300,
        "scenario_seed_policy": "paired_across_methods_for_each_node_beamwidth_slot_setting",
        "methods": methods,
        "neural_methods": {name: spec_manifest(spec) for name, spec in neural_methods.items() if name in methods},
        "node_counts": [int(value) for value in args.node_counts],
        "beamwidths": [int(value) for value in args.beamwidths],
        "eval_slots": [int(value) for value in args.eval_slots],
        "eval_episodes": int(args.eval_episodes),
        "communication_range_m": float(args.communication_range),
        "sensing_range_m": float(args.sensing_range),
        "missing_checkpoints": missing_checkpoints,
        "skipped_complete": skipped_complete,
        "eval_commands": eval_commands,
        "aggregation_commands": aggregation_commands,
    }


def neural_method_specs(args: argparse.Namespace) -> dict[str, NeuralMethod]:
    return {
        "mappo_no_isac": NeuralMethod(
            method="mappo_no_isac",
            checkpoint=resolve_repo_path(args.mappo_no_isac_checkpoint),
            reward_version="legacy",
            env_protocol="structured_marl_no_isac",
            description="Shared MAPPO checkpoint evaluated without ISAC-derived features.",
        ),
        "contention_no_isac": NeuralMethod(
            method="contention_no_isac",
            checkpoint=resolve_repo_path(args.contention_no_isac_checkpoint),
            reward_version="collision_topology",
            env_protocol="structured_marl_no_isac",
            description="Contention-aware MAPPO checkpoint evaluated without ISAC-derived features.",
        ),
        "contention_actor": NeuralMethod(
            method="contention_actor",
            checkpoint=resolve_repo_path(args.contention_actor_checkpoint),
            reward_version="collision_topology",
            env_protocol="isac_structured_marl",
            description="Contention-aware ISAC-MAPPO checkpoint.",
        ),
    }


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def protocol_command(method: str, args: argparse.Namespace, output: Path, node_count: int, beamwidth: int, slots: int) -> list[str]:
    protocol = "skyorbs_like_skip_scan" if method == "skyorbs_like" else method
    return [
        sys.executable,
        str(PROTOCOL_EVAL_SCRIPT),
        "--config",
        str(args.config),
        "--output",
        str(output),
        "--protocols",
        protocol,
        "--eval-episodes",
        str(args.eval_episodes),
        "--slots",
        str(slots),
        "--node-count",
        str(node_count),
        "--beamwidth-deg",
        str(beamwidth),
        "--communication-range",
        f"{float(args.communication_range):g}",
        "--sensing-range",
        f"{float(args.sensing_range):g}",
        "--seed",
        str(eval_seed(args.seed, node_count, beamwidth, slots)),
        "--slot-metric-period",
        "0",
        "--quiet",
    ]


def neural_command(spec: NeuralMethod, args: argparse.Namespace, output: Path, node_count: int, beamwidth: int, slots: int) -> list[str]:
    azimuth, elevation = BEAMWIDTH_TO_CELLS[int(beamwidth)]
    return [
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
        str(eval_seed(args.seed, node_count, beamwidth, slots)),
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
        "--stochastic",
    ]


def aggregation_commands_for(args: argparse.Namespace, run_dirs: list[str]) -> list[list[str]]:
    output = Path(args.analysis_output_root) / f"{args.campaign}_all_methods"
    figures = Path(args.figure_output_root) / f"{args.campaign}_all_methods"
    transfer = [
        sys.executable,
        str(TRANSFER_PLOT_SCRIPT),
        "--output",
        str(output),
        "--figures",
        str(figures),
        "--quiet",
    ]
    for run_dir in run_dirs:
        transfer.extend(["--run-dir", run_dir])
    comparison_output = Path(args.analysis_output_root) / f"{args.campaign}_method_comparison"
    comparison_figures = Path(args.figure_output_root) / f"{args.campaign}_method_comparison"
    comparison = [
        sys.executable,
        str(METHOD_PLOT_SCRIPT),
        "--combined-summary",
        str(output / "marl_transfer_summary.csv"),
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
    comparison.extend(str(value) for value in args.beamwidths)
    comparison.append("--quiet")
    return [transfer, comparison]


def run_commands(commands: list[list[str]], records_path: Path, args: argparse.Namespace) -> None:
    records: list[dict[str, Any]] = []
    timeout = int(args.command_timeout_seconds) if int(args.command_timeout_seconds) > 0 else None
    for index, command in enumerate(commands, start=1):
        if not bool(args.quiet):
            print(f"[{index}/{len(commands)}] {' '.join(command)}", flush=True)
        try:
            completed = subprocess.run(command, cwd=ROOT, text=True, timeout=timeout)
            returncode: int | str = int(completed.returncode)
        except subprocess.TimeoutExpired:
            returncode = "timeout"
        record = {
            "index": index,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "command": command,
            "returncode": returncode,
        }
        records.append(record)
        records_path.write_text(json.dumps(records, ensure_ascii=True, indent=2), encoding="utf-8")
        if returncode == "timeout":
            raise TimeoutError(f"Command timed out: {' '.join(command)}")
        if returncode != 0:
            raise SystemExit(returncode)


def complete_eval_run(output: Path, expected_rows: int) -> bool:
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
    try:
        with data_path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return False
    return len(rows) == int(expected_rows) and bool(rows)


def eval_name(method: str, node_count: int, beamwidth: int, slots: int, episodes: int) -> str:
    return f"{method}_train_n10_b10_test_n{node_count}_b{beamwidth}_{slots}slot_{episodes}ep_stoch"


def eval_seed(base_seed: int, node_count: int, beamwidth: int, slots: int) -> int:
    return int(base_seed) + int(node_count) * 1_000 + int(beamwidth) * 100 + int(slots)


def spec_manifest(spec: NeuralMethod) -> dict[str, str]:
    return {
        "checkpoint": str(spec.checkpoint),
        "reward_version": spec.reward_version,
        "env_protocol": spec.env_protocol,
        "description": spec.description,
    }


if __name__ == "__main__":
    main()
