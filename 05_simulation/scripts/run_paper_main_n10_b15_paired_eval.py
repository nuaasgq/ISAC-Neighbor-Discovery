from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "05_simulation" / "configs" / "paper_main_n10_b15.yaml"
MARL_EVALUATOR = ROOT / "05_simulation" / "run_marl_evaluate.py"
PROTOCOL_EVALUATOR = ROOT / "05_simulation" / "run_protocol_baseline_eval.py"
TRAIN_ROOT = (
    ROOT
    / "05_simulation"
    / "results_raw"
    / "paper_main_n10_b15_formal_3seed"
    / "residual_mask_mappo"
)
TRAIN_SEEDS = (69260715, 69261724, 69262733)
EVAL_SEED = 79260715
EVAL_EPISODES = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen paper-main paired evaluation on 50 common held-out scenarios."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "05_simulation" / "results_raw" / "paper_main_n10_b15_paired_eval_50ep",
    )
    parser.add_argument("--train-seeds", default=",".join(str(seed) for seed in TRAIN_SEEDS))
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--train-seeds must contain distinct integers.")
    return seeds


def command_value(command: list[str], flag: str) -> str | None:
    if flag not in command:
        return None
    index = command.index(flag)
    return command[index + 1] if index + 1 < len(command) else None


def completed(output: Path, expected_episodes: int) -> bool:
    required = (
        output / "manifest.json",
        output / "eval_episode_metrics.csv",
        output / "edge_discovery_timeline.csv",
    )
    if not all(path.is_file() for path in required):
        return False
    with (output / "eval_episode_metrics.csv").open(encoding="utf-8-sig") as handle:
        return max(0, sum(1 for _line in handle) - 1) == expected_episodes


def marl_command(
    checkpoint: Path,
    output: Path,
    label: str,
    eval_seed: int,
    eval_episodes: int,
    threads: int,
    *,
    candidate_random: bool = False,
) -> list[str]:
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
        str(eval_seed),
        "--torch-threads",
        str(threads),
        "--ablation-label",
        label,
        "--collect-discovery-timeline",
        "--resource-log-period",
        "1000",
        "--max-rss-mb",
        "4096",
        "--max-system-memory-percent",
        "85",
    ]
    if candidate_random:
        command.extend(
            (
                "--mode-executor",
                "uniform_tx_rx",
                "--beam-executor",
                "local_candidate_random",
                "--candidate-source",
                "residual_table",
                "--policy-ablation",
                "zero_weights",
            )
        )
    else:
        command.extend(
            (
                "--mode-executor",
                "policy",
                "--beam-executor",
                "policy",
                "--policy-ablation",
                "trained",
            )
        )
    return command


def protocol_command(output: Path, eval_seed: int, eval_episodes: int) -> list[str]:
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
        str(eval_seed),
        "--collect-discovery-timeline",
        "--quiet",
    ]


def build_jobs(
    run_root: Path,
    train_seeds: tuple[int, ...],
    eval_seed: int,
    eval_episodes: int,
    threads: int,
) -> list[tuple[Path, list[str], bool]]:
    jobs: list[tuple[Path, list[str], bool]] = []
    protocol_root = run_root / "protocol_baselines"
    protocol_done = all(
        completed(protocol_root / protocol, eval_episodes)
        for protocol in ("uniform_random", "wang2025_isac_tables")
    )
    jobs.append(
        (
            protocol_root,
            protocol_command(protocol_root, eval_seed, eval_episodes),
            protocol_done,
        )
    )
    for train_seed in train_seeds:
        checkpoint = TRAIN_ROOT / f"seed_{train_seed}" / "final_model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint}")
        output = run_root / "residual_mask_mappo" / f"seed_{train_seed}"
        jobs.append(
            (
                output,
                marl_command(
                    checkpoint,
                    output,
                    f"residual_mask_mappo_seed_{train_seed}",
                    eval_seed,
                    eval_episodes,
                    threads,
                ),
                completed(output, eval_episodes),
            )
        )
    reference_checkpoint = TRAIN_ROOT / f"seed_{train_seeds[0]}" / "final_model.pt"
    candidate_output = run_root / "residual_candidate_random"
    jobs.append(
        (
            candidate_output,
            marl_command(
                reference_checkpoint,
                candidate_output,
                "residual_candidate_random",
                eval_seed,
                eval_episodes,
                threads,
                candidate_random=True,
            ),
            completed(candidate_output, eval_episodes),
        )
    )
    return jobs


def validate_contract(
    jobs: list[tuple[Path, list[str], bool]],
    eval_seed: int,
    eval_episodes: int,
) -> None:
    if len(jobs) != len(TRAIN_SEEDS) + 2:
        raise ValueError("Paired evaluation must contain one protocol job, three MAPPO jobs, and one control job.")
    for _output, command, _done in jobs:
        required = {
            "--config": str(CONFIG),
            "--eval-episodes": str(eval_episodes),
            "--slots": "300",
            "--seed": str(eval_seed),
        }
        for flag, expected in required.items():
            if command_value(command, flag) != expected:
                raise ValueError(f"{flag} changed from the paired value {expected}.")
        if "--collect-discovery-timeline" not in command:
            raise ValueError("Every method must record the discovery timeline.")


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


def execute(command: list[str], threads: int) -> None:
    subprocess.run(command, cwd=ROOT, env=limited_cpu_environment(threads), check=True)


def main() -> None:
    args = parse_args()
    train_seeds = parse_seeds(args.train_seeds)
    if train_seeds != TRAIN_SEEDS:
        raise ValueError(f"The frozen campaign requires training seeds {TRAIN_SEEDS}.")
    if args.eval_episodes < 1 or args.max_parallel < 1 or args.torch_threads < 1:
        raise ValueError("Episode, parallel, and thread counts must be positive.")
    run_root = args.run_root.resolve()
    jobs = build_jobs(
        run_root,
        train_seeds,
        int(args.eval_seed),
        int(args.eval_episodes),
        int(args.torch_threads),
    )
    validate_contract(jobs, int(args.eval_seed), int(args.eval_episodes))
    pending: list[list[str]] = []
    for output, command, is_complete in jobs:
        if is_complete and args.skip_completed:
            print(json.dumps({"phase": "skip_completed", "output": str(output)}), flush=True)
            continue
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
        print(
            json.dumps({"phase": "dry_run" if args.dry_run else "queued", "command": command}),
            flush=True,
        )
        if not args.dry_run:
            pending.append(command)
    if args.dry_run or not pending:
        return
    with ThreadPoolExecutor(max_workers=min(args.max_parallel, len(pending))) as pool:
        futures = [pool.submit(execute, command, int(args.torch_threads)) for command in pending]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()

