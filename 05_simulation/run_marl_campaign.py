from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "05_simulation" / "run_marl_training.py"
EVAL_SCRIPT = ROOT / "05_simulation" / "run_marl_evaluate.py"
DEFAULT_CONFIG = ROOT / "05_simulation" / "configs" / "paper_transfer_train_n10_b10_singlehop.yaml"

BEAMWIDTH_TO_CELLS = {
    3: (120, 60),
    5: (72, 36),
    10: (36, 18),
    15: (24, 12),
    30: (12, 6),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a controlled short-training / long-testing MARL campaign.")
    parser.add_argument("--campaign", default="marl_short_train_long_eval")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default="05_simulation/results_raw/marl_campaign")
    parser.add_argument("--train-episodes", type=int, default=20)
    parser.add_argument("--train-slots", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-slots", type=int, nargs="+", default=[300, 1200, 3000])
    parser.add_argument("--node-counts", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--beamwidths", type=int, nargs="+", default=[5, 10, 15, 30])
    parser.add_argument("--include-n100", action="store_true", help="Also evaluate N=100 transfer cases.")
    parser.add_argument("--algorithms", nargs="+", default=["isac_mappo", "mappo"])
    parser.add_argument(
        "--network",
        choices=[
            "shared",
            "scalegraph_beam",
            "contention_shared",
            "gated_contention_shared",
            "adaptive_gated_contention_shared",
            "topology_adaptive_gated_contention_shared",
            "balanced_topology_gated_contention_shared",
        ],
        default="shared",
    )
    parser.add_argument("--reward-version", choices=["legacy", "collision_topology"], default="legacy")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--step-log-period", type=int, default=1)
    parser.add_argument("--resource-log-period", type=int, default=100)
    parser.add_argument("--max-rss-mb", type=float, default=10000.0)
    parser.add_argument("--max-system-memory-percent", type=float, default=90.0)
    parser.add_argument("--eval-stochastic-only", action="store_true")
    parser.add_argument("--command-timeout-seconds", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root) / args.campaign
    output_root.mkdir(parents=True, exist_ok=True)
    node_counts = list(args.node_counts)
    if args.include_n100 and 100 not in node_counts:
        node_counts.append(100)
    plan = build_plan(args, output_root, node_counts)
    (output_root / "campaign_plan.json").write_text(json.dumps(plan, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    run_records = []
    for index, command in enumerate(plan["commands"], start=1):
        print(f"[{index}/{len(plan['commands'])}] {' '.join(command)}", flush=True)
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                timeout=int(args.command_timeout_seconds) if int(args.command_timeout_seconds) > 0 else None,
            )
        except subprocess.TimeoutExpired:
            run_records.append({"index": index, "command": command, "returncode": "timeout"})
            (output_root / "campaign_run_records.json").write_text(
                json.dumps(run_records, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
            raise
        run_records.append({"index": index, "command": command, "returncode": completed.returncode})
        (output_root / "campaign_run_records.json").write_text(
            json.dumps(run_records, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
    print(json.dumps({"status": "complete", "output_root": str(output_root), "commands": len(plan["commands"])}, indent=2))


def build_plan(args: argparse.Namespace, output_root: Path, node_counts: list[int]) -> dict:
    commands = []
    train_runs = {}
    network_suffix = "" if str(args.network) == "shared" else f"_{args.network}"
    reward_suffix = "" if str(args.reward_version) == "legacy" else f"_{args.reward_version}"
    for algorithm in args.algorithms:
        run_name = f"train_n10_b10_{algorithm}{network_suffix}{reward_suffix}_{args.train_slots}slot"
        output = output_root / "train" / run_name
        command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--config",
            str(args.config),
            "--output",
            str(output),
            "--algorithm",
            algorithm,
            "--network",
            str(args.network),
            "--reward-version",
            str(args.reward_version),
            "--episodes",
            str(args.train_episodes),
            "--slots",
            str(args.train_slots),
            "--eval-episodes",
            str(args.eval_episodes),
            "--eval-interval",
            str(max(1, args.train_episodes // 4)),
            "--eval-both",
            "--checkpoint-interval",
            str(max(1, args.train_episodes // 2)),
            "--hidden-dim",
            str(args.hidden_dim),
            "--ppo-epochs",
            str(args.ppo_epochs),
            "--torch-threads",
            str(args.torch_threads),
            "--resource-log-period",
            str(args.resource_log_period),
            "--step-log-period",
            str(args.step_log_period),
            "--max-rss-mb",
            str(args.max_rss_mb),
            "--max-system-memory-percent",
            str(args.max_system_memory_percent),
        ]
        if algorithm == "mappo":
            command.extend(["--disable-isac-features", "--env-protocol", "structured_marl_no_isac"])
        commands.append(command)
        train_runs[algorithm] = output

    for algorithm, train_output in train_runs.items():
        checkpoint = train_output / "final_model.pt"
        for eval_slots in args.eval_slots:
            for n_nodes in node_counts:
                for beamwidth in args.beamwidths:
                    azimuth, elevation = beam_cells(beamwidth)
                    eval_name = (
                        f"{algorithm}{network_suffix}{reward_suffix}_train_n10_b10_test_n{n_nodes}_b{beamwidth}_{eval_slots}slot"
                    )
                    output = output_root / "eval" / eval_name
                    command = [
                        sys.executable,
                        str(EVAL_SCRIPT),
                        "--checkpoint",
                        str(checkpoint),
                        "--config",
                        str(args.config),
                        "--output",
                        str(output),
                        "--eval-episodes",
                        str(args.eval_episodes),
                        "--slots",
                        str(eval_slots),
                        "--node-count",
                        str(n_nodes),
                        "--azimuth-cells",
                        str(azimuth),
                        "--elevation-cells",
                        str(elevation),
                        "--communication-range",
                        "900",
                        "--sensing-range",
                        "900",
                        "--seed",
                        str(args.seed + 100_000 + n_nodes * 100 + beamwidth + eval_slots),
                        "--torch-threads",
                        str(args.torch_threads),
                        "--reward-version",
                        str(args.reward_version),
                        "--resource-log-period",
                        str(args.resource_log_period),
                        "--max-rss-mb",
                        str(args.max_rss_mb),
                        "--max-system-memory-percent",
                        str(args.max_system_memory_percent),
                    ]
                    if bool(args.eval_stochastic_only):
                        command.append("--stochastic")
                    else:
                        command.append("--eval-both")
                    if algorithm == "mappo":
                        command.extend(["--env-protocol", "structured_marl_no_isac"])
                    commands.append(command)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "campaign": str(args.campaign),
        "train_slots": int(args.train_slots),
        "train_episodes": int(args.train_episodes),
        "eval_slots": list(args.eval_slots),
        "eval_episodes": int(args.eval_episodes),
        "node_counts": node_counts,
        "beamwidths": list(args.beamwidths),
        "algorithms": list(args.algorithms),
        "network": str(args.network),
        "reward_version": str(args.reward_version),
        "step_log_period": int(args.step_log_period),
        "resource_log_period": int(args.resource_log_period),
        "eval_stochastic_only": bool(args.eval_stochastic_only),
        "command_timeout_seconds": int(args.command_timeout_seconds),
        "resource_limits": {
            "max_rss_mb": float(args.max_rss_mb),
            "max_system_memory_percent": float(args.max_system_memory_percent),
        },
        "commands": commands,
    }


def beam_cells(beamwidth: int) -> tuple[int, int]:
    if int(beamwidth) not in BEAMWIDTH_TO_CELLS:
        raise ValueError(f"Unsupported beamwidth {beamwidth}; expected one of {sorted(BEAMWIDTH_TO_CELLS)}.")
    return BEAMWIDTH_TO_CELLS[int(beamwidth)]


if __name__ == "__main__":
    main()
