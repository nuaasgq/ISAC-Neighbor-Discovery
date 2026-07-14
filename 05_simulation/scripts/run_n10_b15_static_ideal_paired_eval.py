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
CONFIG = ROOT / "05_simulation" / "configs" / "n10_b15_static_ideal_isac.yaml"
MARL_EVALUATOR = ROOT / "05_simulation" / "run_marl_evaluate.py"
PROTOCOL_EVALUATOR = ROOT / "05_simulation" / "run_protocol_baseline_eval.py"
TRAIN_ROOT = ROOT / "05_simulation" / "results_raw" / "n10_b15_static_ideal_mappo_formal_3seed"
TRAIN_SEEDS = (59260713, 59261722, 59262731)
EVAL_SEED_OFFSET = 2_000_000


@dataclass(frozen=True)
class CheckpointMethod:
    label: str
    checkpoint_arm: str
    mode_executor: str = "policy"
    beam_executor: str = "policy"
    candidate_source: str | None = None
    policy_ablation: str = "trained"
    condition_role_on_executed_beam: bool = False


CHECKPOINT_METHODS = (
    CheckpointMethod("mappo_no_isac", "mappo_no_isac"),
    CheckpointMethod("mappo_direct_isac", "mappo_direct_isac"),
    CheckpointMethod(
        "mappo_direct_isac_measurement_aux",
        "mappo_direct_isac_measurement_aux",
    ),
    CheckpointMethod(
        "random_role_learned_beam",
        "mappo_direct_isac",
        mode_executor="uniform_tx_rx",
    ),
    CheckpointMethod(
        "learned_role_uniform_beam",
        "mappo_direct_isac",
        beam_executor="uniform_random",
        condition_role_on_executed_beam=True,
    ),
    CheckpointMethod(
        "isac_candidate_pool_random",
        "mappo_direct_isac",
        mode_executor="uniform_tx_rx",
        beam_executor="local_candidate_random",
        candidate_source="residual_table",
        policy_ablation="zero_weights",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired N=10 static ideal-ISAC evaluation on the MAPPO held-out seeds."
    )
    parser.add_argument("--profile", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--train-seeds", default=",".join(str(seed) for seed in TRAIN_SEEDS))
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--train-seeds must contain distinct integers.")
    return seeds


def default_run_root(profile: str) -> Path:
    suffix = "3seed" if profile == "formal" else "smoke"
    return ROOT / "05_simulation" / "results_raw" / f"n10_b15_static_ideal_paired_eval_{suffix}"


def completed(output: Path, expected_episodes: int) -> bool:
    required = (
        output / "manifest.json",
        output / "eval_episode_metrics.csv",
        output / "edge_discovery_timeline.csv",
    )
    if not all(path.is_file() for path in required):
        return False
    with (output / "eval_episode_metrics.csv").open(encoding="utf-8-sig") as handle:
        return max(0, sum(1 for _ in handle) - 1) == expected_episodes


def checkpoint_command(
    method: CheckpointMethod,
    train_seed: int,
    eval_episodes: int,
    output: Path,
    threads: int,
) -> list[str]:
    checkpoint = TRAIN_ROOT / method.checkpoint_arm / f"seed_{train_seed}" / "final_model.pt"
    command = [
        sys.executable,
        str(MARL_EVALUATOR),
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
        str(train_seed + EVAL_SEED_OFFSET),
        "--torch-threads",
        str(threads),
        "--mode-executor",
        method.mode_executor,
        "--beam-executor",
        method.beam_executor,
        "--policy-ablation",
        method.policy_ablation,
        "--ablation-label",
        method.label,
        "--collect-discovery-timeline",
        "--no-resume",
    ]
    if method.candidate_source is not None:
        command.extend(("--candidate-source", method.candidate_source))
    if method.condition_role_on_executed_beam:
        command.append("--condition-role-on-executed-beam")
    return command


def protocol_command(train_seed: int, eval_episodes: int, output: Path) -> list[str]:
    return [
        sys.executable,
        str(PROTOCOL_EVALUATOR),
        "--config",
        str(CONFIG),
        "--output",
        str(output),
        "--protocols",
        "uniform_random",
        "wang2025_isac_tables",
        "--eval-episodes",
        str(eval_episodes),
        "--slots",
        "300",
        "--seed",
        str(train_seed + EVAL_SEED_OFFSET),
        "--collect-discovery-timeline",
        "--quiet",
    ]


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


def validate_contract(commands: list[list[str]], eval_episodes: int) -> None:
    for command in commands:
        joined = " ".join(command)
        if "--slots 300" not in joined or f"--eval-episodes {eval_episodes}" not in joined:
            raise ValueError("Paired evaluation command changed its horizon or episode count.")
        if "--collect-discovery-timeline" not in command:
            raise ValueError("Every paired evaluation command must record edge timelines.")


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.train_seeds)
    if not 1 <= int(args.max_parallel) <= 2:
        raise ValueError("--max-parallel must be one or two.")
    eval_episodes = 50 if args.profile == "formal" else 2
    run_root = (args.run_root or default_run_root(args.profile)).resolve()
    jobs: list[tuple[Path, list[str], bool]] = []
    for seed in seeds:
        seed_name = f"seed_{seed}"
        protocol_root = run_root / "protocol_baselines" / seed_name
        protocol_done = all(
            completed(protocol_root / protocol, eval_episodes)
            for protocol in ("uniform_random", "wang2025_isac_tables")
        )
        jobs.append((protocol_root, protocol_command(seed, eval_episodes, protocol_root), protocol_done))
        for method in CHECKPOINT_METHODS:
            output = run_root / method.label / seed_name
            jobs.append(
                (
                    output,
                    checkpoint_command(method, seed, eval_episodes, output, args.torch_threads),
                    completed(output, eval_episodes),
                )
            )
    validate_contract([command for _output, command, _done in jobs], eval_episodes)

    pending: list[list[str]] = []
    for output, command, is_complete in jobs:
        if is_complete and args.skip_completed:
            print(json.dumps({"phase": "skip_completed", "output": str(output)}), flush=True)
            continue
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
        print(
            json.dumps({"phase": "dry_run" if args.dry_run else "run", "command": command}),
            flush=True,
        )
        if not args.dry_run:
            pending.append(command)

    if args.dry_run:
        return

    def execute(command: list[str]) -> None:
        subprocess.run(
            command,
            cwd=ROOT,
            env=limited_cpu_environment(args.torch_threads),
            check=True,
        )

    with ThreadPoolExecutor(max_workers=int(args.max_parallel)) as pool:
        list(pool.map(execute, pending))


if __name__ == "__main__":
    main()
