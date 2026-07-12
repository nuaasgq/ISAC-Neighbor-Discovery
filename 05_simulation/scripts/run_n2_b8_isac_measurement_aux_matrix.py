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
CONFIG = ROOT / "05_simulation" / "configs" / "sanity_planar_n2_b45_ideal.yaml"
DEFAULT_SEEDS = (39260711, 39261720, 39262729)


@dataclass(frozen=True)
class MethodSpec:
    algorithm: str
    env_protocol: str
    measurement_feature_set: str
    beam_uniform_mixture: float
    measurement_prediction_aux_coef: float


METHOD_SPECS = {
    "learned_beam_no_isac": MethodSpec(
        algorithm="mappo",
        env_protocol="structured_marl_no_isac",
        measurement_feature_set="none",
        beam_uniform_mixture=0.10,
        measurement_prediction_aux_coef=0.0,
    ),
    "learned_beam_direct_isac": MethodSpec(
        algorithm="isac_mappo",
        env_protocol="improved_rl_isac_tables",
        measurement_feature_set="direct",
        beam_uniform_mixture=0.10,
        measurement_prediction_aux_coef=0.0,
    ),
    "learned_beam_direct_isac_measurement_aux": MethodSpec(
        algorithm="isac_mappo",
        env_protocol="improved_rl_isac_tables",
        measurement_feature_set="direct",
        beam_uniform_mixture=0.10,
        measurement_prediction_aux_coef=0.10,
    ),
    "learned_beam_residual_isac_measurement_aux": MethodSpec(
        algorithm="isac_mappo",
        env_protocol="improved_rl_isac_tables",
        measurement_feature_set="residual",
        beam_uniform_mixture=0.10,
        measurement_prediction_aux_coef=0.10,
    ),
}

PROFILES = {
    "smoke": {
        "episodes": 100,
        "eval_episodes": 50,
        "checkpoint_interval": 0,
        "flush_interval": 1,
    },
    "pilot": {
        "episodes": 625,
        "eval_episodes": 100,
        "checkpoint_interval": 125,
        "flush_interval": 5,
    },
    "formal": {
        "episodes": 6250,
        "eval_episodes": 200,
        "checkpoint_interval": 1250,
        "flush_interval": 25,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the paired N=2, B=8 beam-learning matrix with a common "
            "antisymmetric role head and local measurement-prediction ablation."
        )
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Exactly three comma-separated independent training seeds.",
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument(
        "--methods",
        default=",".join(METHOD_SPECS),
        help="Comma-separated subset of the registered matrix methods.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_seeds(text: str) -> tuple[int, int, int]:
    seeds = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("--seeds must contain exactly three distinct integers.")
    return seeds  # type: ignore[return-value]


def parse_methods(text: str) -> tuple[str, ...]:
    methods = tuple(part.strip() for part in text.split(",") if part.strip())
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("--methods must contain distinct registered method names.")
    unknown = sorted(set(methods) - set(METHOD_SPECS))
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    return methods


def default_run_root(profile: str) -> Path:
    return ROOT / "05_simulation" / "results_raw" / f"n2_b8_isac_measurement_aux_{profile}_3seed"


def completed(output: Path) -> bool:
    required = (
        output / "final_model.pt",
        output / "manifest.json",
        output / "episode_metrics.csv",
        output / "eval_episode_metrics.csv",
    )
    return all(path.is_file() for path in required)


def command_value(command: list[str], flag: str) -> str | None:
    if flag not in command:
        return None
    index = command.index(flag)
    return command[index + 1] if index + 1 < len(command) else None


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
        str(spec.beam_uniform_mixture),
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
        "--eval-both",
        "--torch-threads",
        str(torch_threads),
        "--step-log-period",
        "16",
        "--resource-log-period",
        "250",
        "--seed",
        str(seed),
    ]
    if spec.measurement_feature_set == "none":
        command.extend(("--no-residual-measurement-features", "--disable-isac-features"))
    return command


def validate_matrix_contract(commands: list[list[str]], method_count: int | None = None) -> None:
    if method_count is None:
        method_count = len(
            {
                Path(command_value(command, "--output") or "").parent.name
                for command in commands
            }
        )
    expected = method_count * 3
    if len(commands) != expected:
        raise ValueError(f"Expected {expected} commands, received {len(commands)}.")
    for command in commands:
        method = Path(command_value(command, "--output") or "").parent.name
        spec = METHOD_SPECS[method]
        if command_value(command, "--role-factorization") != "beam_conditioned_antisymmetric":
            raise ValueError(f"{method}: role-factorization contract changed.")
        if "--candidate-mask" in command or "--candidate-score-prior" in command:
            raise ValueError(f"{method}: hard or rule-prior candidate guidance is forbidden.")
        if command_value(command, "--beam-uniform-mixture") != str(spec.beam_uniform_mixture):
            raise ValueError(f"{method}: exploration contract changed.")
        if command_value(command, "--measurement-feature-set") != spec.measurement_feature_set:
            raise ValueError(f"{method}: measurement feature contract changed.")
        if command_value(command, "--measurement-prediction-aux-coef") != str(
            spec.measurement_prediction_aux_coef
        ):
            raise ValueError(f"{method}: measurement auxiliary contract changed.")
        if command_value(command, "--beam-isac-feedback-coef") != "0.0":
            raise ValueError(f"{method}: action-level ISAC credit is forbidden.")


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
        raise ValueError("--torch-threads must be at least one.")
    if not 1 <= args.max_parallel <= 3:
        raise ValueError("--max-parallel must be between one and three.")
    run_root = (args.run_root or default_run_root(args.profile)).resolve()
    commands = [
        training_command(args.profile, run_root, method, seed, args.torch_threads)
        for seed in seeds
        for method in methods
    ]
    validate_matrix_contract(commands, len(methods))

    pending: list[list[str]] = []
    for command in commands:
        output = Path(command_value(command, "--output") or "")
        if completed(output) and args.skip_completed:
            print(json.dumps({"phase": "skip_completed", "output": str(output)}), flush=True)
            continue
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty run directory: {output}. "
                "Use a new --run-root or --skip-completed."
            )
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
