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
TRAIN_SCRIPT = ROOT / "05_simulation" / "run_marl_training.py"
DEFAULT_CONFIG = ROOT / "05_simulation" / "configs" / "paper_transfer_train_n10_b10_singlehop.yaml"


@dataclass(frozen=True)
class TrainMethod:
    name: str
    algorithm: str
    network: str
    reward_version: str
    description: str
    disable_isac_features: bool = False
    env_protocol: str | None = None
    topology_deficit: bool = False


METHODS = {
    "legacy_shared": TrainMethod(
        name="legacy_shared",
        algorithm="isac_mappo",
        network="shared",
        reward_version="legacy",
        description="Shared ISAC-MAPPO with the legacy reward.",
    ),
    "collision_reward": TrainMethod(
        name="collision_reward",
        algorithm="isac_mappo",
        network="shared",
        reward_version="collision_topology",
        description="Shared ISAC-MAPPO with collision/topology reward shaping.",
    ),
    "contention_actor": TrainMethod(
        name="contention_actor",
        algorithm="isac_mappo",
        network="contention_shared",
        reward_version="collision_topology",
        description="Contention-aware shared ISAC-MAPPO actor.",
    ),
    "gated_contention_actor": TrainMethod(
        name="gated_contention_actor",
        algorithm="isac_mappo",
        network="gated_contention_shared",
        reward_version="collision_topology",
        description="Contention-aware ISAC-MAPPO actor with a learned decentralized access gate.",
    ),
    "adaptive_gated_contention_actor": TrainMethod(
        name="adaptive_gated_contention_actor",
        algorithm="isac_mappo",
        network="adaptive_gated_contention_shared",
        reward_version="collision_topology",
        description="Contention-aware ISAC-MAPPO actor with a collision-adaptive decentralized access gate.",
    ),
    "mappo_no_isac": TrainMethod(
        name="mappo_no_isac",
        algorithm="mappo",
        network="shared",
        reward_version="legacy",
        description="Shared MAPPO actor without ISAC-derived features.",
        disable_isac_features=True,
        env_protocol="structured_marl_no_isac",
    ),
    "contention_no_isac": TrainMethod(
        name="contention_no_isac",
        algorithm="mappo",
        network="contention_shared",
        reward_version="collision_topology",
        description="Contention-aware shared MAPPO actor without ISAC-derived features.",
        disable_isac_features=True,
        env_protocol="structured_marl_no_isac",
        topology_deficit=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run multi-seed MARL training stability campaigns while keeping each episode "
            "fixed at the requested slot length."
        )
    )
    parser.add_argument("--campaign", default="phase7_long_training_100ep_3seed")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default="05_simulation/results_raw/marl_campaign")
    parser.add_argument("--log-root", default="05_simulation/results_raw/marl_campaign_logs")
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=sorted(METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260731, 20260732, 20260733])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--slots", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--step-log-period", type=int, default=1)
    parser.add_argument("--resource-log-period", type=int, default=100)
    parser.add_argument("--max-rss-mb", type=float, default=10000.0)
    parser.add_argument("--max-system-memory-percent", type=float, default=90.0)
    parser.add_argument("--command-timeout-seconds", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_root = Path(args.output_root) / args.campaign
    train_root = output_root / "train"
    log_root = Path(args.log_root) / args.campaign
    output_root.mkdir(parents=True, exist_ok=True)
    train_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    plan = build_plan(args, train_root, log_root)
    (output_root / "training_stability_plan.json").write_text(
        json.dumps(plan, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=True, indent=2))
        return

    records: list[dict[str, Any]] = []
    record_path = output_root / "training_stability_run_records.json"
    for index, item in enumerate(plan["runs"], start=1):
        if item["skip_complete"] and not bool(args.force):
            record = {
                "index": index,
                "run": item["run_name"],
                "method": item["method"],
                "seed": item["seed"],
                "returncode": "skipped_complete",
                "started_at": None,
                "ended_at": datetime.now().isoformat(timespec="seconds"),
            }
            records.append(record)
            write_records(record_path, records)
            if not args.quiet:
                print(f"[{index}/{len(plan['runs'])}] skip complete {item['run_name']}", flush=True)
            continue

        if not args.quiet:
            print(f"[{index}/{len(plan['runs'])}] {' '.join(item['command'])}", flush=True)
        record = run_one(item, args)
        records.append(record)
        write_records(record_path, records)
        if record["returncode"] != 0:
            raise SystemExit(record["returncode"])

    if not args.quiet:
        print(
            json.dumps(
                {
                    "status": "complete",
                    "campaign": args.campaign,
                    "runs": len(plan["runs"]),
                    "records": str(record_path),
                    "output_root": str(output_root),
                },
                ensure_ascii=True,
                indent=2,
            )
        )


def validate_args(args: argparse.Namespace) -> None:
    if int(args.episodes) <= 0:
        raise ValueError("--episodes must be positive.")
    if int(args.slots) <= 0:
        raise ValueError("--slots must be positive.")
    if int(args.eval_episodes) < 0:
        raise ValueError("--eval-episodes cannot be negative.")
    if int(args.eval_interval) <= 0:
        raise ValueError("--eval-interval must be positive.")
    if int(args.torch_threads) <= 0:
        raise ValueError("--torch-threads must be positive.")


def build_plan(args: argparse.Namespace, train_root: Path, log_root: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        for method_name in args.methods:
            method = METHODS[method_name]
            run_name = f"train_n10_b10_{method.name}_{args.episodes}ep_{args.slots}slot_seed{seed}"
            output = train_root / run_name
            command = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--config",
                str(args.config),
                "--output",
                str(output),
                "--algorithm",
                method.algorithm,
                "--network",
                method.network,
                "--reward-version",
                method.reward_version,
                "--episodes",
                str(args.episodes),
                "--slots",
                str(args.slots),
                "--eval-episodes",
                str(args.eval_episodes),
                "--eval-interval",
                str(args.eval_interval),
                "--eval-both",
                "--checkpoint-interval",
                str(args.checkpoint_interval),
                "--hidden-dim",
                str(args.hidden_dim),
                "--ppo-epochs",
                str(args.ppo_epochs),
                "--torch-threads",
                str(args.torch_threads),
                "--step-log-period",
                str(args.step_log_period),
                "--resource-log-period",
                str(args.resource_log_period),
                "--max-rss-mb",
                str(args.max_rss_mb),
                "--max-system-memory-percent",
                str(args.max_system_memory_percent),
                "--seed",
                str(seed),
            ]
            if method.disable_isac_features:
                command.append("--disable-isac-features")
            if method.env_protocol:
                command.extend(["--env-protocol", method.env_protocol])
            if method.topology_deficit:
                command.append("--topology-deficit")
            runs.append(
                {
                    "run_name": run_name,
                    "method": method.name,
                    "seed": int(seed),
                    "description": method.description,
                    "output": str(output),
                    "stdout": str(log_root / f"{run_name}.out.log"),
                    "stderr": str(log_root / f"{run_name}.err.log"),
                    "skip_complete": complete_training_run(output, int(args.episodes), int(args.slots), method, int(seed)),
                    "command": command,
                }
            )
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "campaign": str(args.campaign),
        "train_episodes": int(args.episodes),
        "train_slots": int(args.slots),
        "eval_episodes": int(args.eval_episodes),
        "eval_interval": int(args.eval_interval),
        "methods": list(args.methods),
        "seeds": [int(seed) for seed in args.seeds],
        "runs": runs,
    }


def complete_training_run(output: Path, expected_episodes: int, expected_slots: int, method: TrainMethod, seed: int) -> bool:
    if not (output / "final_model.pt").exists():
        return False
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if int(manifest.get("episodes", -1)) != int(expected_episodes):
        return False
    if int(manifest.get("slots", manifest.get("slots_per_episode", -1))) != int(expected_slots):
        return False
    manifest_seed = manifest.get("seed")
    if manifest_seed is not None and int(manifest_seed) != int(seed):
        return False
    if str(manifest.get("network", "")) != method.network:
        return False
    if str(manifest.get("reward_version", "")) != method.reward_version:
        return False
    if method.env_protocol and str(manifest.get("env_protocol", "")) != method.env_protocol:
        return False
    if bool((manifest.get("args") or {}).get("disable_isac_features", False)) != bool(method.disable_isac_features):
        return False
    return csv_rows(output / "episode_metrics.csv") >= expected_episodes


def csv_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        row_count = sum(1 for _ in reader)
    return max(0, row_count - 1)


def run_one(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started_at = datetime.now().isoformat(timespec="seconds")
    stdout_path = Path(item["stdout"])
    stderr_path = Path(item["stderr"])
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            completed = subprocess.run(
                item["command"],
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=int(args.command_timeout_seconds) if int(args.command_timeout_seconds) > 0 else None,
            )
            returncode: int | str = completed.returncode
        except subprocess.TimeoutExpired:
            returncode = "timeout"
    ended_at = datetime.now().isoformat(timespec="seconds")
    return {
        "run": item["run_name"],
        "method": item["method"],
        "seed": item["seed"],
        "output": item["output"],
        "stdout": item["stdout"],
        "stderr": item["stderr"],
        "started_at": started_at,
        "ended_at": ended_at,
        "returncode": returncode,
    }


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
