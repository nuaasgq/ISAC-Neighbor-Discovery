from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

from run_n10_b15_static_ideal_mappo_matrix import DEFAULT_SEEDS, METHOD_SPECS, PROFILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor the N=10 static ideal MAPPO matrix.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="formal")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def last_row(path: Path) -> tuple[int, dict[str, str] | None]:
    if not path.is_file() or path.stat().st_size == 0:
        return 0, None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows), rows[-1] if rows else None


def collect_status(run_root: Path, profile: str, seeds: tuple[int, ...]) -> dict[str, object]:
    expected_episodes = int(PROFILES[profile]["episodes"])
    expected_steps = expected_episodes * 300
    runs: dict[str, dict[str, object]] = {}
    for method in METHOD_SPECS:
        for seed in seeds:
            output = run_root / method / f"seed_{seed}"
            episodes, row = last_row(output / "episode_metrics.csv")
            _resources, resource = last_row(output / "resource_log.csv")
            complete = (
                episodes == expected_episodes
                and (output / "final_model.pt").is_file()
                and (output / "eval_episode_metrics.csv").is_file()
            )
            step = int(float(row.get("training_step", 0))) if row else 0
            runs[f"{method}/seed_{seed}"] = {
                "state": "complete" if complete else "partial" if episodes else "missing",
                "episodes": episodes,
                "training_step": step,
                "expected_training_steps": expected_steps,
                "progress_percent": 100.0 * step / max(1, expected_steps),
                "rss_mb": float(resource["rss_mb"]) if resource and resource.get("rss_mb") else None,
                "system_memory_percent": (
                    float(resource["system_memory_percent"])
                    if resource and resource.get("system_memory_percent")
                    else None
                ),
            }
    values = list(runs.values())
    completed_steps = sum(int(value["training_step"]) for value in values)
    expected_total = expected_steps * len(values)
    memories = [float(value["system_memory_percent"]) for value in values if value["system_memory_percent"] is not None]
    rss = [float(value["rss_mb"]) for value in values if value["rss_mb"] is not None]
    return {
        "complete_runs": sum(value["state"] == "complete" for value in values),
        "partial_runs": sum(value["state"] == "partial" for value in values),
        "missing_runs": sum(value["state"] == "missing" for value in values),
        "expected_runs": len(values),
        "completed_training_steps": completed_steps,
        "expected_total_training_steps": expected_total,
        "overall_progress_percent": 100.0 * completed_steps / max(1, expected_total),
        "max_run_rss_mb": max(rss) if rss else None,
        "max_logged_system_memory_percent": max(memories) if memories else None,
        "runs": runs,
    }


def print_status(status: dict[str, object]) -> None:
    print(
        f"complete={status['complete_runs']}/{status['expected_runs']} "
        f"partial={status['partial_runs']} missing={status['missing_runs']} "
        f"overall={status['overall_progress_percent']:.2f}% "
        f"step={status['completed_training_steps']}/{status['expected_total_training_steps']}"
    )
    print(
        f"max_run_rss={status['max_run_rss_mb'] or 0:.1f} MB "
        f"system_memory={status['max_logged_system_memory_percent'] or 0:.1f}%"
    )
    for key, value in status["runs"].items():
        print(
            f"{value['state']:8s} {value['progress_percent']:6.2f}% "
            f"step={value['training_step']:7d}/{value['expected_training_steps']:7d} {key}"
        )


def main() -> None:
    args = parse_args()
    if args.watch and args.json:
        raise ValueError("--watch and --json cannot be combined.")
    seeds = tuple(int(part.strip()) for part in args.seeds.split(",") if part.strip())
    while True:
        status = collect_status(args.run_root, args.profile, seeds)
        if args.json:
            print(json.dumps(status, indent=2))
            return
        if args.watch:
            print("\033[2J\033[H", end="")
            print(time.strftime("N=10 MAPPO progress  %Y-%m-%d %H:%M:%S"))
        print_status(status)
        if not args.watch or int(status["complete_runs"]) == int(status["expected_runs"]):
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
