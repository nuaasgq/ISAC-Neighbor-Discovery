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
ANALYZER = ROOT / "06_analysis" / "scripts" / "analyze_n2_b8_rmappo_role_factorization_matrix.py"
CONFIG = ROOT / "05_simulation" / "configs" / "sanity_planar_n2_b45_ideal.yaml"
FACTORIZATIONS = (
    "independent",
    "beam_conditioned",
    "beam_conditioned_antisymmetric",
)
DEFAULT_SEEDS = (29260711, 29261720, 29262729)
PROFILES = {
    "smoke": {
        "episodes": 100,
        "eval_episodes": 50,
        "checkpoint_interval": 0,
        "flush_interval": 1,
        "step_log_period": 16,
    },
    "formal": {
        "episodes": 6250,
        "eval_episodes": 200,
        "checkpoint_interval": 1250,
        "flush_interval": 25,
        "step_log_period": 16,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the paired three-seed N=2, B=8, no-ISAC recurrent MAPPO "
            "joint-role factorization matrix."
        )
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Raw result root. Defaults to a profile-specific directory under 05_simulation/results_raw.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Exactly three comma-separated independent training seeds.",
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip a run only when its final checkpoint, manifest, and train/eval CSV files all exist.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without creating files.")
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run the paired aggregation script after all six runs complete.",
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=None,
        help="Optional analysis output directory used with --analyze.",
    )
    return parser.parse_args()


def parse_seeds(text: str) -> tuple[int, int, int]:
    seeds = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("--seeds must contain exactly three distinct integers.")
    return seeds  # type: ignore[return-value]


def default_run_root(profile: str) -> Path:
    suffix = "smoke_3seed" if profile == "smoke" else "formal_100k_3seed"
    return ROOT / "05_simulation" / "results_raw" / f"n2_b8_noisac_rmappo_role_factorization_{suffix}"


def completed(output: Path) -> bool:
    required = (
        output / "final_model.pt",
        output / "manifest.json",
        output / "episode_metrics.csv",
        output / "eval_episode_metrics.csv",
    )
    return all(path.is_file() for path in required)


def training_command(
    profile: str,
    run_root: Path,
    factorization: str,
    seed: int,
    torch_threads: int,
) -> list[str]:
    settings = PROFILES[profile]
    output = run_root / factorization / f"seed_{seed}"
    return [
        sys.executable,
        str(TRAINER),
        "--config",
        str(CONFIG),
        "--output",
        str(output),
        "--algorithm",
        "mappo",
        "--network",
        "recurrent_contention_shared",
        "--action-contract",
        "joint_role_beam",
        "--decoupled-role-tower",
        "--role-factorization",
        factorization,
        "--episodes",
        str(settings["episodes"]),
        "--slots",
        "16",
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
        "2",
        "--azimuth-cells",
        "8",
        "--elevation-cells",
        "1",
        "--spatial-dimensions",
        "2",
        "--env-protocol",
        "structured_marl_no_isac",
        "--clean-ctde",
        "--disable-isac-features",
        "--no-candidate-score",
        "--no-rendezvous-observation",
        "--forbid-sense",
        "--beam-uniform-mixture",
        "1.0",
        "--hidden-dim",
        "64",
        "--critic-hidden-dim",
        "64",
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
        "--role-balance-coef",
        "0.0",
        "--eval-both",
        "--torch-threads",
        str(torch_threads),
        "--step-log-period",
        str(settings["step_log_period"]),
        "--resource-log-period",
        "250",
        "--seed",
        str(seed),
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


def print_command(phase: str, command: list[str]) -> None:
    print(json.dumps({"phase": phase, "command": command}, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be at least one.")
    if not 1 <= args.max_parallel <= 3:
        raise ValueError("--max-parallel must be between one and three.")
    run_root = (args.run_root or default_run_root(args.profile)).resolve()
    commands = [
        training_command(args.profile, run_root, factorization, seed, args.torch_threads)
        for seed in seeds
        for factorization in FACTORIZATIONS
    ]

    pending: list[list[str]] = []
    for command in commands:
        output = Path(command[command.index("--output") + 1])
        if completed(output) and args.skip_completed:
            print(json.dumps({"phase": "skip_completed", "output": str(output)}), flush=True)
            continue
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty run directory: {output}. "
                "Use a new --run-root or --skip-completed."
            )
        print_command("dry_run" if args.dry_run else "run", command)
        if not args.dry_run:
            pending.append(command)

    def execute(command: list[str]) -> None:
        subprocess.run(
            command,
            cwd=ROOT,
            env=limited_cpu_environment(args.torch_threads),
            check=True,
        )

    if pending:
        with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
            list(pool.map(execute, pending))

    if args.analyze:
        analysis_command = [sys.executable, str(ANALYZER), "--run-root", str(run_root)]
        if args.analysis_output is not None:
            analysis_command.extend(("--output", str(args.analysis_output.resolve())))
        print_command("dry_run_analysis" if args.dry_run else "analyze", analysis_command)
        if not args.dry_run:
            subprocess.run(analysis_command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
