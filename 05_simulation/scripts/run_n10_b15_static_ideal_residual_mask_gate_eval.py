from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "05_simulation" / "configs" / "n10_b15_static_ideal_isac.yaml"
EVALUATOR = ROOT / "05_simulation" / "run_marl_evaluate.py"
DEFAULT_TRAIN_SEED = 59262731
EVAL_SEED_OFFSET = 2_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate residual-mask MAPPO on the exact paired N=10 held-out scenarios."
    )
    parser.add_argument("--profile", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_checkpoint() -> Path:
    return (
        ROOT
        / "05_simulation"
        / "results_raw"
        / "n10_b15_static_ideal_residual_mask_pilot"
        / "mappo_residual_mask_isac"
        / f"seed_{DEFAULT_TRAIN_SEED}"
        / "final_model.pt"
    )


def default_output(profile: str, seed: int) -> Path:
    suffix = "formal" if profile == "formal" else "smoke"
    return (
        ROOT
        / "05_simulation"
        / "results_raw"
        / f"n10_b15_static_ideal_residual_mask_gate_eval_{suffix}"
        / "mappo_residual_mask_isac"
        / f"seed_{seed}"
    )


def evaluation_command(
    checkpoint: Path,
    output: Path,
    train_seed: int,
    eval_episodes: int,
    torch_threads: int,
) -> list[str]:
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
        str(eval_episodes),
        "--slots",
        "300",
        "--stochastic",
        "--seed",
        str(int(train_seed) + EVAL_SEED_OFFSET),
        "--torch-threads",
        str(torch_threads),
        "--policy-ablation",
        "trained",
        "--ablation-label",
        "mappo_residual_mask_isac",
        "--collect-discovery-timeline",
        "--no-resume",
    ]


def validate_contract(command: list[str], eval_episodes: int, train_seed: int) -> None:
    required_pairs = {
        "--eval-episodes": str(eval_episodes),
        "--slots": "300",
        "--seed": str(int(train_seed) + EVAL_SEED_OFFSET),
        "--policy-ablation": "trained",
        "--ablation-label": "mappo_residual_mask_isac",
    }
    for flag, expected in required_pairs.items():
        index = command.index(flag)
        if command[index + 1] != expected:
            raise ValueError(f"{flag} changed from gate contract value {expected}.")
    for flag in ("--stochastic", "--collect-discovery-timeline", "--no-resume"):
        if flag not in command:
            raise ValueError(f"Missing gate contract flag {flag}.")
    forbidden = ("--candidate-source", "--beam-executor", "--mode-executor")
    if any(flag in command for flag in forbidden):
        raise ValueError("Gate evaluation must load candidate support and actions from the checkpoint.")


def completed(output: Path, eval_episodes: int) -> bool:
    required = (
        output / "manifest.json",
        output / "eval_episode_metrics.csv",
        output / "edge_discovery_timeline.csv",
    )
    if not all(path.is_file() for path in required):
        return False
    with (output / "eval_episode_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle)) == eval_episodes


def main() -> None:
    args = parse_args()
    if int(args.torch_threads) < 1:
        raise ValueError("--torch-threads must be positive.")
    eval_episodes = 50 if args.profile == "formal" else 2
    checkpoint = (args.checkpoint or default_checkpoint()).resolve()
    output = (args.output or default_output(args.profile, int(args.train_seed))).resolve()
    command = evaluation_command(
        checkpoint,
        output,
        int(args.train_seed),
        eval_episodes,
        int(args.torch_threads),
    )
    validate_contract(command, eval_episodes, int(args.train_seed))
    if completed(output, eval_episodes) and args.skip_completed:
        print(json.dumps({"phase": "skip_completed", "output": str(output)}), flush=True)
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    if not checkpoint.is_file() and not args.dry_run:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    print(
        json.dumps({"phase": "dry_run" if args.dry_run else "run", "command": command}),
        flush=True,
    )
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
