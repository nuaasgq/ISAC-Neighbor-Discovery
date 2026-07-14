from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_n10_b15_static_ideal_mappo_matrix import (  # noqa: E402
    command_value,
    completed,
    limited_cpu_environment,
    training_command,
)


DEFAULT_SEED = 59262731
METHOD = "mappo_direct_isac_independent_role"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the N=10 direct-ISAC MAPPO role-head architecture ablation."
    )
    parser.add_argument("--profile", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--seeds", default=str(DEFAULT_SEED))
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain distinct integers.")
    return seeds


def default_run_root(profile: str) -> Path:
    return (
        ROOT
        / "05_simulation"
        / "results_raw"
        / f"n10_b15_static_ideal_independent_role_{profile}"
    )


def replace_command_value(command: list[str], flag: str, value: str) -> None:
    if flag not in command:
        raise ValueError(f"Missing required command flag: {flag}")
    command[command.index(flag) + 1] = value


def ablation_command(
    profile: str,
    run_root: Path,
    seed: int,
    torch_threads: int,
) -> list[str]:
    command = training_command(
        profile,
        run_root,
        "mappo_direct_isac",
        seed,
        torch_threads,
    )
    output = run_root / METHOD / f"seed_{seed}"
    replace_command_value(command, "--output", str(output))
    replace_command_value(command, "--role-factorization", "independent")
    return command


def validate_contract(command: list[str], profile: str) -> None:
    expected_episodes = "1000" if profile == "formal" else "2"
    required = {
        "--episodes": expected_episodes,
        "--slots": "300",
        "--node-count": "10",
        "--measurement-feature-set": "direct",
        "--measurement-prediction-aux-coef": "0.0",
        "--role-factorization": "independent",
    }
    for flag, expected in required.items():
        if command_value(command, flag) != expected:
            raise ValueError(f"{flag} changed from the ablation contract value {expected}.")
    if "--decoupled-role-tower" not in command:
        raise ValueError("The independent role ablation must retain the decoupled role tower.")


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    if int(args.torch_threads) < 1:
        raise ValueError("--torch-threads must be positive.")
    run_root = (args.run_root or default_run_root(args.profile)).resolve()
    expected_episodes = 1000 if args.profile == "formal" else 2
    commands = [
        ablation_command(args.profile, run_root, seed, args.torch_threads)
        for seed in seeds
    ]
    for command in commands:
        validate_contract(command, args.profile)
        output = Path(command_value(command, "--output") or "")
        if completed(output, expected_episodes) and args.skip_completed:
            print(json.dumps({"phase": "skip_completed", "output": str(output)}), flush=True)
            continue
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
        print(
            json.dumps({"phase": "dry_run" if args.dry_run else "run", "command": command}),
            flush=True,
        )
        if not args.dry_run:
            subprocess.run(
                command,
                cwd=ROOT,
                env=limited_cpu_environment(args.torch_threads),
                check=True,
            )


if __name__ == "__main__":
    main()
