from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "05_simulation" / "run_marl_training.py"
CONFIG = ROOT / "05_simulation" / "configs" / "n10_b15_static_ideal_isac.yaml"
DEFAULT_SEEDS = (59260713, 59261722, 59262731)


@dataclass(frozen=True)
class MethodSpec:
    algorithm: str
    env_protocol: str
    measurement_feature_set: str
    measurement_prediction_aux_coef: float


METHOD_SPECS = {
    "mappo_no_isac": MethodSpec(
        algorithm="mappo",
        env_protocol="structured_marl_no_isac",
        measurement_feature_set="none",
        measurement_prediction_aux_coef=0.0,
    ),
    "mappo_direct_isac": MethodSpec(
        algorithm="isac_mappo",
        env_protocol="improved_rl_isac_tables",
        measurement_feature_set="direct",
        measurement_prediction_aux_coef=0.0,
    ),
    "mappo_direct_isac_measurement_aux": MethodSpec(
        algorithm="isac_mappo",
        env_protocol="improved_rl_isac_tables",
        measurement_feature_set="direct",
        measurement_prediction_aux_coef=0.1,
    ),
}

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
        "checkpoint_interval": 200,
        "flush_interval": 10,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the N=10, B=15-degree static ideal-ISAC MAPPO ablation matrix."
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--methods", default=",".join(METHOD_SPECS))
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain distinct integers.")
    return seeds


def parse_methods(text: str) -> tuple[str, ...]:
    methods = tuple(part.strip() for part in text.split(",") if part.strip())
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("--methods must contain distinct method names.")
    unknown = sorted(set(methods).difference(METHOD_SPECS))
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    return methods


def default_run_root(profile: str) -> Path:
    return ROOT / "05_simulation" / "results_raw" / f"n10_b15_static_ideal_mappo_{profile}_3seed"


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


def training_command(
    profile: str,
    run_root: Path,
    method: str,
    seed: int,
    torch_threads: int,
) -> list[str]:
    settings = PROFILES[profile]
    spec = METHOD_SPECS[method]
    output = run_root / method / f"seed_{seed}"
    command = [
        sys.executable,
        str(TRAINER),
        "--config",
        str(CONFIG),
        "--output",
        str(output),
        "--algorithm",
        spec.algorithm,
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
        "static",
        "--env-protocol",
        spec.env_protocol,
        "--candidate-source",
        "default",
        "--measurement-feature-set",
        spec.measurement_feature_set,
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
        str(spec.measurement_prediction_aux_coef),
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
        "--seed",
        str(seed),
    ]
    if spec.measurement_feature_set == "none":
        command.extend(("--no-residual-measurement-features", "--disable-isac-features"))
    return command


def validate_matrix_contract(
    commands: list[list[str]],
    seeds: tuple[int, ...],
    methods: tuple[str, ...] | None = None,
) -> None:
    selected_methods = methods or tuple(METHOD_SPECS)
    expected = len(selected_methods) * len(seeds)
    if len(commands) != expected:
        raise ValueError(f"Expected {expected} commands, received {len(commands)}.")
    for command in commands:
        method = Path(command_value(command, "--output") or "").parent.name
        spec = METHOD_SPECS[method]
        required = {
            "--node-count": "10",
            "--azimuth-cells": "24",
            "--elevation-cells": "1",
            "--spatial-dimensions": "2",
            "--mobility-model": "static",
            "--slots": "300",
            "--candidate-source": "default",
            "--measurement-feature-set": spec.measurement_feature_set,
            "--measurement-prediction-aux-coef": str(spec.measurement_prediction_aux_coef),
            "--beam-isac-feedback-coef": "0.0",
            "--expert-bc-weight": "0.0",
            "--beam-uniform-mixture": "0.1",
        }
        for flag, expected_value in required.items():
            if command_value(command, flag) != expected_value:
                raise ValueError(f"{method}: {flag} changed from {expected_value}.")
        forbidden = (
            "--candidate-mask",
            "--candidate-score-prior",
            "--rendezvous-observation",
        )
        if any(flag in command for flag in forbidden):
            raise ValueError(f"{method}: forbidden action guidance is enabled.")


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
    methods = parse_methods(args.methods)
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive.")
    if not 1 <= args.max_parallel <= 2:
        raise ValueError("--max-parallel must be one or two.")
    settings = PROFILES[args.profile]
    run_root = (args.run_root or default_run_root(args.profile)).resolve()
    commands = [
        training_command(args.profile, run_root, method, seed, args.torch_threads)
        for seed in seeds
        for method in methods
    ]
    validate_matrix_contract(commands, seeds, methods)

    pending: list[list[str]] = []
    for command in commands:
        output = Path(command_value(command, "--output") or "")
        if completed(output, int(settings["episodes"])) and args.skip_completed:
            print(json.dumps({"phase": "skip_completed", "output": str(output)}), flush=True)
            continue
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output}")
        print(json.dumps({"phase": "dry_run" if args.dry_run else "run", "command": command}), flush=True)
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


if __name__ == "__main__":
    main()
