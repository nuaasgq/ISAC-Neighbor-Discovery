from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "05_simulation" / "run_marl_training.py"
CONFIG = ROOT / "05_simulation" / "configs" / "paper_main_n10_b15.yaml"
DEFAULT_SEEDS = (69260715, 69261724, 69262733)
METHOD = "residual_mask_mappo"

PROFILES = {
    "smoke": {
        "episodes": 2,
        "eval_episodes": 2,
        "checkpoint_interval": 0,
        "flush_interval": 1,
    },
    "pilot": {
        "episodes": 100,
        "eval_episodes": 10,
        "checkpoint_interval": 25,
        "flush_interval": 5,
    },
    "formal": {
        "episodes": 1000,
        "eval_episodes": 50,
        "checkpoint_interval": 100,
        "flush_interval": 10,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the frozen paper-main Residual-mask MAPPO method."
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain distinct integers.")
    return seeds


def default_run_root(profile: str) -> Path:
    return ROOT / "05_simulation" / "results_raw" / f"paper_main_n10_b15_{profile}_3seed"


def command_value(command: list[str], flag: str) -> str | None:
    if flag not in command:
        return None
    index = command.index(flag)
    return command[index + 1] if index + 1 < len(command) else None


def completed(output: Path, expected_episodes: int) -> bool:
    required = (
        output / "final_model.pt",
        output / "manifest.json",
        output / "episode_metrics.csv",
        output / "eval_episode_metrics.csv",
    )
    if not all(path.is_file() for path in required):
        return False
    with (output / "episode_metrics.csv").open(encoding="utf-8-sig") as handle:
        return max(0, sum(1 for _line in handle) - 1) == expected_episodes


def training_command(profile: str, run_root: Path, seed: int, torch_threads: int) -> list[str]:
    settings = PROFILES[profile]
    return [
        sys.executable,
        str(TRAINER),
        "--config",
        str(CONFIG),
        "--output",
        str(run_root / METHOD / f"seed_{seed}"),
        "--algorithm",
        "isac_mappo",
        "--network",
        "recurrent_contention_shared",
        "--action-contract",
        "joint_role_beam",
        "--decoupled-role-tower",
        "--role-factorization",
        "beam_conditioned_antisymmetric",
        "--episodes",
        str(settings["episodes"]),
        "--slots",
        "300",
        "--eval-episodes",
        str(settings["eval_episodes"]),
        "--eval-interval",
        "0",
        "--checkpoint-interval",
        str(settings["checkpoint_interval"]),
        "--flush-interval-episodes",
        str(settings["flush_interval"]),
        "--training-scenario-mode",
        "varying",
        "--evaluation-scenario-mode",
        "held_out",
        "--node-count",
        "10",
        "--azimuth-cells",
        "24",
        "--elevation-cells",
        "1",
        "--spatial-dimensions",
        "2",
        "--mobility-model",
        "gauss_markov",
        "--sensing-measurement-mode",
        "noisy_count",
        "--env-protocol",
        "improved_rl_isac_tables",
        "--candidate-source",
        "residual_table",
        "--candidate-mask",
        "--measurement-feature-set",
        "direct",
        "--clean-ctde",
        "--no-candidate-score",
        "--no-candidate-score-prior",
        "--no-bounded-score-residual",
        "--no-rendezvous-observation",
        "--forbid-sense",
        "--separate-action-loss",
        "--beam-loss-coef",
        "1.0",
        "--beam-isac-feedback-coef",
        "0.0",
        "--measurement-prediction-aux-coef",
        "0.0",
        "--beam-rank-aux-coef",
        "0.0",
        "--local-potential-shaping-coef",
        "0.0",
        "--expert-bc-weight",
        "0.0",
        "--role-balance-coef",
        "0.0",
        "--role-probability-floor",
        "0.0",
        "--beam-uniform-mixture",
        "0.1",
        "--hidden-dim",
        "128",
        "--critic-hidden-dim",
        "128",
        "--learning-rate",
        "0.0003",
        "--gamma",
        "0.99",
        "--advantage-estimator",
        "gae",
        "--gae-lambda",
        "0.95",
        "--ppo-epochs",
        "5",
        "--entropy-coef",
        "0.01",
        "--stochastic",
        "--torch-threads",
        str(torch_threads),
        "--step-log-period",
        "300",
        "--resource-log-period",
        "1000",
        "--max-rss-mb",
        "4096",
        "--max-system-memory-percent",
        "85",
        "--seed",
        str(seed),
    ]


def validate_contract(command: list[str], profile: str) -> None:
    settings = PROFILES[profile]
    required = {
        "--config": str(CONFIG),
        "--algorithm": "isac_mappo",
        "--episodes": str(settings["episodes"]),
        "--slots": "300",
        "--node-count": "10",
        "--azimuth-cells": "24",
        "--elevation-cells": "1",
        "--spatial-dimensions": "2",
        "--mobility-model": "gauss_markov",
        "--sensing-measurement-mode": "noisy_count",
        "--candidate-source": "residual_table",
        "--measurement-feature-set": "direct",
        "--expert-bc-weight": "0.0",
        "--role-probability-floor": "0.0",
        "--beam-uniform-mixture": "0.1",
    }
    for flag, expected in required.items():
        if command_value(command, flag) != expected:
            raise ValueError(f"{flag} changed from the frozen paper value {expected}.")
    required_flags = (
        "--candidate-mask",
        "--clean-ctde",
        "--no-candidate-score",
        "--no-candidate-score-prior",
        "--no-bounded-score-residual",
        "--no-rendezvous-observation",
        "--decoupled-role-tower",
    )
    if any(flag not in command for flag in required_flags):
        raise ValueError("The paper-main training contract is incomplete.")
    forbidden = (
        "--rule-residual",
        "--contention-mode-prior",
        "--candidate-score-prior",
        "--rendezvous-observation",
    )
    if any(flag in command for flag in forbidden):
        raise ValueError("The paper-main method enabled a forbidden rule/action prior.")


def limited_cpu_environment(threads: int) -> dict[str, str]:
    env = os.environ.copy()
    value = str(threads)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = value
    env["PYTHONHASHSEED"] = "0"
    return env


def run_one(command: list[str], threads: int) -> None:
    subprocess.run(command, cwd=ROOT, env=limited_cpu_environment(threads), check=True)


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    if args.torch_threads < 1 or args.max_parallel < 1:
        raise ValueError("Thread and parallel counts must be positive.")
    run_root = (args.run_root or default_run_root(args.profile)).resolve()
    expected_episodes = int(PROFILES[args.profile]["episodes"])
    commands: list[list[str]] = []
    for seed in seeds:
        command = training_command(args.profile, run_root, seed, args.torch_threads)
        validate_contract(command, args.profile)
        output = Path(command_value(command, "--output") or "")
        if completed(output, expected_episodes) and args.skip_completed:
            print(json.dumps({"phase": "skip_completed", "output": str(output)}), flush=True)
            continue
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
        commands.append(command)
        print(
            json.dumps({"phase": "dry_run" if args.dry_run else "queued", "command": command}),
            flush=True,
        )
    if args.dry_run or not commands:
        return
    with ThreadPoolExecutor(max_workers=min(args.max_parallel, len(commands))) as pool:
        futures = [pool.submit(run_one, command, args.torch_threads) for command in commands]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()

