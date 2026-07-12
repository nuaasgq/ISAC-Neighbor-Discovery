from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "05_simulation" / "run_matd3_reference_training.py"
CONFIG = ROOT / "05_simulation" / "configs" / "n10_b15_static_ideal_isac.yaml"
DEFAULT_SEEDS = (69260713, 69261722, 69262731)

PROFILES = {
    "smoke": {"episodes": 2, "eval_episodes": 2, "warmup_steps": 0},
    "formal": {"episodes": 1000, "eval_episodes": 50, "warmup_steps": 1500},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official-code MATD3 reference sequentially for N=10."
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
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
    name = f"n10_b15_static_ideal_matd3_reference_{profile}_3seed"
    return ROOT / "05_simulation" / "results_raw" / name


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


def training_command(profile: str, output: Path, seed: int, threads: int) -> list[str]:
    settings = PROFILES[profile]
    return [
        sys.executable,
        str(TRAINER),
        "--config",
        str(CONFIG),
        "--output",
        str(output),
        "--episodes",
        str(settings["episodes"]),
        "--slots",
        "300",
        "--eval-episodes",
        str(settings["eval_episodes"]),
        "--seed",
        str(seed),
        "--hidden-dim",
        "128",
        "--buffer-size",
        "5000",
        "--batch-size",
        "32",
        "--warmup-steps",
        str(settings["warmup_steps"]),
        "--train-interval",
        "100",
        "--epsilon-anneal-steps",
        "100000",
        "--learning-rate",
        "0.0003",
        "--torch-threads",
        str(threads),
    ]


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


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive.")
    settings = PROFILES[args.profile]
    run_root = (args.run_root or default_run_root(args.profile)).resolve()
    commands: list[list[str]] = []
    for seed in seeds:
        output = run_root / "matd3_direct_isac_reference" / f"seed_{seed}"
        if completed(output, int(settings["episodes"])) and args.skip_completed:
            print(json.dumps({"phase": "skip_completed", "output": str(output)}), flush=True)
            continue
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output}")
        command = training_command(args.profile, output, seed, args.torch_threads)
        print(
            json.dumps({"phase": "dry_run" if args.dry_run else "run", "command": command}),
            flush=True,
        )
        commands.append(command)

    if args.dry_run:
        return
    for command in commands:
        subprocess.run(
            command,
            cwd=ROOT,
            env=limited_cpu_environment(args.torch_threads),
            check=True,
        )


if __name__ == "__main__":
    main()
