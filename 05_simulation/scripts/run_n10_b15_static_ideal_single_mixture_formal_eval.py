from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "05_simulation" / "configs" / "n10_b15_static_ideal_isac.yaml"
EVALUATOR = ROOT / "05_simulation" / "run_marl_evaluate.py"
TRAIN_SEEDS = (59260713, 59261722, 59262731)
METHODS = ("mappo_direct_isac", "mappo_residual_mask_isac")
EVAL_SEED_OFFSET = 2_000_000
EXPECTED_TRAIN_EPISODES = 1000
EVAL_EPISODES = 50
SLOTS = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate each completed single-mixture MAPPO checkpoint on the exact paired "
            "N=10 scenarios, optionally waiting for sequential training checkpoints."
        )
    )
    parser.add_argument("--train-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--train-seeds", default=",".join(str(seed) for seed in TRAIN_SEEDS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--wait-for-checkpoints", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_train_root() -> Path:
    return (
        ROOT
        / "05_simulation"
        / "results_raw"
        / "n10_b15_static_ideal_single_mixture_formal_3seed"
    )


def default_output_root() -> Path:
    return (
        ROOT
        / "05_simulation"
        / "results_raw"
        / "n10_b15_static_ideal_single_mixture_formal_eval_3seed"
    )


def parse_distinct(text: str, *, flag: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{flag} must contain distinct values.")
    return values


def parse_seeds(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in parse_distinct(text, flag="--train-seeds"))


def parse_methods(text: str) -> tuple[str, ...]:
    methods = parse_distinct(text, flag="--methods")
    unknown = sorted(set(methods).difference(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    return methods


def csv_row_count(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def checkpoint_complete(seed_root: Path) -> bool:
    return (
        (seed_root / "final_model.pt").is_file()
        and (seed_root / "manifest.json").is_file()
        and csv_row_count(seed_root / "episode_metrics.csv") == EXPECTED_TRAIN_EPISODES
    )


def evaluation_complete(output: Path) -> bool:
    return (
        (output / "manifest.json").is_file()
        and csv_row_count(output / "eval_episode_metrics.csv") == EVAL_EPISODES
        and csv_row_count(output / "candidate_pool_timeline.csv") == EVAL_EPISODES * SLOTS
        and csv_row_count(output / "edge_discovery_timeline.csv") > 0
    )


def evaluation_command(
    checkpoint: Path,
    output: Path,
    train_seed: int,
    method: str,
    torch_threads: int,
) -> list[str]:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    return [
        sys.executable,
        str(EVALUATOR),
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(CONFIG),
        "--output",
        str(output),
        "--eval-episodes",
        str(EVAL_EPISODES),
        "--slots",
        str(SLOTS),
        "--stochastic",
        "--seed",
        str(int(train_seed) + EVAL_SEED_OFFSET),
        "--torch-threads",
        str(torch_threads),
        "--policy-ablation",
        "trained",
        "--ablation-label",
        method,
        "--collect-discovery-timeline",
        "--collect-candidate-pool-timeline",
        "--target-status-diagnostics",
    ]


def validate_contract(command: list[str], train_seed: int, method: str) -> None:
    required_pairs = {
        "--eval-episodes": str(EVAL_EPISODES),
        "--slots": str(SLOTS),
        "--seed": str(int(train_seed) + EVAL_SEED_OFFSET),
        "--policy-ablation": "trained",
        "--ablation-label": method,
    }
    for flag, expected in required_pairs.items():
        index = command.index(flag)
        if command[index + 1] != expected:
            raise ValueError(f"{flag} changed from the formal contract value {expected}.")
    for flag in (
        "--stochastic",
        "--collect-discovery-timeline",
        "--collect-candidate-pool-timeline",
        "--target-status-diagnostics",
    ):
        if flag not in command:
            raise ValueError(f"Missing formal evaluation flag {flag}.")
    forbidden = ("--candidate-source", "--beam-executor", "--mode-executor")
    if any(flag in command for flag in forbidden):
        raise ValueError("Formal evaluation must load candidate support and actions from the checkpoint.")


def limited_cpu_environment(threads: int) -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = str(threads)
    env["PYTHONHASHSEED"] = "0"
    return env


def wait_until_checkpoint(seed_root: Path, seed: int, method: str, poll_seconds: float) -> None:
    while not checkpoint_complete(seed_root):
        print(
            json.dumps(
                {
                    "phase": "waiting_for_checkpoint",
                    "method": method,
                    "train_seed": seed,
                    "completed_training_episodes": csv_row_count(
                        seed_root / "episode_metrics.csv"
                    ),
                    "expected_training_episodes": EXPECTED_TRAIN_EPISODES,
                }
            ),
            flush=True,
        )
        time.sleep(poll_seconds)


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.train_seeds)
    methods = parse_methods(args.methods)
    if int(args.torch_threads) < 1:
        raise ValueError("--torch-threads must be positive.")
    if float(args.poll_seconds) <= 0:
        raise ValueError("--poll-seconds must be positive.")
    train_root = (args.train_root or default_train_root()).resolve()
    output_root = (args.output_root or default_output_root()).resolve()

    for seed in seeds:
        for method in methods:
            seed_root = train_root / method / f"seed_{seed}"
            checkpoint = seed_root / "final_model.pt"
            output = output_root / method / f"seed_{seed}"
            command = evaluation_command(checkpoint, output, seed, method, int(args.torch_threads))
            validate_contract(command, seed, method)
            if evaluation_complete(output) and args.skip_completed:
                print(json.dumps({"phase": "skip_completed", "output": str(output)}), flush=True)
                continue
            if args.dry_run:
                print(json.dumps({"phase": "dry_run", "command": command}), flush=True)
                continue
            if not checkpoint_complete(seed_root):
                if not args.wait_for_checkpoints:
                    print(
                        json.dumps(
                            {
                                "phase": "checkpoint_not_ready",
                                "method": method,
                                "train_seed": seed,
                                "completed_training_episodes": csv_row_count(
                                    seed_root / "episode_metrics.csv"
                                ),
                            }
                        ),
                        flush=True,
                    )
                    continue
                wait_until_checkpoint(seed_root, seed, method, float(args.poll_seconds))
            print(json.dumps({"phase": "run", "command": command}), flush=True)
            subprocess.run(
                command,
                cwd=ROOT,
                env=limited_cpu_environment(int(args.torch_threads)),
                check=True,
            )
            if not evaluation_complete(output):
                raise RuntimeError(f"Formal evaluation output is incomplete: {output}")


if __name__ == "__main__":
    main()
