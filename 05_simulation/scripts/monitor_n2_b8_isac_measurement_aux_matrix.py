from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

from run_n2_b8_isac_measurement_aux_matrix import (
    DEFAULT_SEEDS,
    METHOD_SPECS,
    PROFILES,
    parse_methods,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only status summary for the N=2, B=8 measurement-auxiliary matrix."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="formal")
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Refresh the human-readable status until every run completes.",
    )
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--methods", default=",".join(METHOD_SPECS))
    return parser.parse_args()


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain distinct integers.")
    return seeds


def last_csv_row(path: Path) -> tuple[int, dict[str, str] | None]:
    if not path.is_file() or path.stat().st_size == 0:
        return 0, None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows), rows[-1] if rows else None


def run_status(output: Path, expected_episodes: int, expected_steps: int) -> dict[str, object]:
    episode_count, episode = last_csv_row(output / "episode_metrics.csv")
    _resource_count, resource = last_csv_row(output / "resource_log.csv")
    required = (
        output / "final_model.pt",
        output / "manifest.json",
        output / "episode_metrics.csv",
        output / "eval_episode_metrics.csv",
    )
    complete = all(path.is_file() for path in required) and episode_count == expected_episodes
    state = "complete" if complete else "partial" if episode_count else "missing"
    training_step = int(float(episode.get("training_step", 0))) if episode else 0
    return {
        "state": state,
        "episodes": episode_count,
        "expected_episodes": expected_episodes,
        "training_step": training_step,
        "expected_training_steps": expected_steps,
        "progress_percent": 100.0 * training_step / max(1, expected_steps),
        "latest_discovery_rate": float(episode["discovery_rate"]) if episode else None,
        "latest_actor_grad_norm": float(episode["actor_grad_norm"]) if episode else None,
        "latest_rss_mb": float(resource["rss_mb"]) if resource and resource.get("rss_mb") else None,
        "latest_system_memory_percent": (
            float(resource["system_memory_percent"])
            if resource and resource.get("system_memory_percent")
            else None
        ),
    }


def collect_status(
    run_root: Path, profile: str, seeds: tuple[int, ...], methods: tuple[str, ...] | None = None
) -> dict[str, object]:
    expected_episodes = int(PROFILES[profile]["episodes"])
    expected_steps = expected_episodes * 16
    runs: dict[str, object] = {}
    for method in methods or tuple(METHOD_SPECS):
        for seed in seeds:
            key = f"{method}/seed_{seed}"
            runs[key] = run_status(run_root / method / f"seed_{seed}", expected_episodes, expected_steps)
    states = [str(value["state"]) for value in runs.values()]  # type: ignore[index]
    completed_steps = sum(int(value["training_step"]) for value in runs.values())  # type: ignore[index]
    expected_total_steps = expected_steps * len(runs)
    rss_values = [
        float(value["latest_rss_mb"])
        for value in runs.values()  # type: ignore[union-attr]
        if value["latest_rss_mb"] is not None
    ]
    memory_values = [
        float(value["latest_system_memory_percent"])
        for value in runs.values()  # type: ignore[union-attr]
        if value["latest_system_memory_percent"] is not None
    ]
    return {
        "run_root": str(run_root.resolve()),
        "profile": profile,
        "expected_runs": len(runs),
        "complete_runs": states.count("complete"),
        "partial_runs": states.count("partial"),
        "missing_runs": states.count("missing"),
        "completed_training_steps": completed_steps,
        "expected_total_training_steps": expected_total_steps,
        "overall_progress_percent": 100.0 * completed_steps / max(1, expected_total_steps),
        "latest_max_run_rss_mb": max(rss_values) if rss_values else None,
        "latest_max_system_memory_percent": max(memory_values) if memory_values else None,
        "runs": runs,
    }


def print_human_status(status: dict[str, object]) -> None:
    print(
        f"complete={status['complete_runs']}/{status['expected_runs']} "
        f"partial={status['partial_runs']} missing={status['missing_runs']} "
        f"overall={status['overall_progress_percent']:.2f}% "
        f"step={status['completed_training_steps']}/{status['expected_total_training_steps']}"
    )
    memory = status["latest_max_system_memory_percent"]
    memory_text = f"{memory:.1f}%" if memory is not None else "n/a"
    rss = status["latest_max_run_rss_mb"]
    rss_text = f"{rss:.1f} MB" if rss is not None else "n/a"
    print(f"max_run_rss={rss_text} system_memory={memory_text}")
    for key, value in status["runs"].items():  # type: ignore[union-attr]
        print(
            f"{value['state']:8s} {value['progress_percent']:6.2f}% "
            f"step={value['training_step']:6d}/{value['expected_training_steps']:6d} {key}"
        )


def main() -> None:
    args = parse_args()
    if args.watch and args.json:
        raise ValueError("--watch and --json cannot be combined.")
    if args.interval_seconds <= 0.0:
        raise ValueError("--interval-seconds must be positive.")
    seeds = parse_seeds(args.seeds)
    methods = parse_methods(args.methods)
    while True:
        status = collect_status(args.run_root, args.profile, seeds, methods)
        if args.json:
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return
        if args.watch:
            print("\033[2J\033[H", end="")
            print(time.strftime("ISAC experiment progress  %Y-%m-%d %H:%M:%S"))
        print_human_status(status)
        if not args.watch or int(status["complete_runs"]) == int(status["expected_runs"]):
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
