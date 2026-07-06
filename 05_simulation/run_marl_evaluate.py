from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "05_simulation" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isac_nd_sim.config import SimulationConfig, load_config  # noqa: E402
from isac_nd_sim.marl_env import MarlNeighborDiscoveryEnv  # noqa: E402
from isac_nd_sim.neural_contention_actor_critic import (  # noqa: E402
    AdaptiveGatedContentionGraphActorCritic,
    ContentionGraphActorCritic,
    GatedContentionGraphActorCritic,
)
from isac_nd_sim.neural_scalegraph_beam_actor_critic import ScaleGraphBeamActorCritic  # noqa: E402
from isac_nd_sim.neural_shared_actor_critic import SharedBeamActorCritic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained shared MARL policy under transfer settings.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="05_simulation/configs/paper_transfer_train_n10_b10_singlehop.yaml")
    parser.add_argument("--output", default="05_simulation/results_raw/marl_eval")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--slots", type=int, default=3000)
    parser.add_argument("--node-count", type=int, default=None)
    parser.add_argument("--area-size-m", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--azimuth-cells", type=int, default=None)
    parser.add_argument("--elevation-cells", type=int, default=None)
    parser.add_argument("--communication-range", type=float, default=None)
    parser.add_argument("--sensing-range", type=float, default=None)
    parser.add_argument("--false-alarm-rate", type=float, default=None)
    parser.add_argument("--miss-detection-rate", type=float, default=None)
    parser.add_argument("--angular-cell-offset-std", type=float, default=None)
    parser.add_argument("--sensing-period-slots", type=int, default=None)
    parser.add_argument("--mobility-model", default=None)
    parser.add_argument("--env-protocol", default=None)
    parser.add_argument("--deterministic", action="store_true", help="Use argmax actions.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions.")
    parser.add_argument("--eval-both", action="store_true", help="Run deterministic and stochastic evaluation.")
    parser.add_argument("--reward-version", choices=["legacy", "collision_topology"], default=None)
    parser.add_argument("--seed", type=int, default=30260705)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--resource-log-period", type=int, default=500)
    parser.add_argument("--max-rss-mb", type=float, default=12000.0)
    parser.add_argument("--max-system-memory-percent", type=float, default=92.0)
    parser.add_argument(
        "--full-step-info",
        action="store_true",
        help="Collect per-slot metrics and rich step info during evaluation. Disabled by default for exact fast eval.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume from an existing partial eval_episode_metrics.csv in the output directory.",
    )
    return parser.parse_args()


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for run_marl_evaluate.py") from exc

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    ensure_resource_args(args)
    cfg = override_config(load_config(args.config), args)
    checkpoint = load_checkpoint(args.checkpoint, torch)
    train_args = checkpoint.get("args", {})
    feature_flags = {
        "candidate_mask": bool(train_args.get("candidate_mask", False)),
        "candidate_score": bool(train_args.get("candidate_score", False)),
        "topology_deficit": bool(train_args.get("topology_deficit", False)),
        "rule_residual": bool(train_args.get("rule_residual", False)),
    }
    if str(train_args.get("algorithm", "")) == "isac_mappo" and not bool(train_args.get("disable_isac_features", False)):
        feature_flags = {
            "candidate_mask": True,
            "candidate_score": True,
            "topology_deficit": True,
            "rule_residual": True,
    }
    hidden_dim = int(train_args.get("hidden_dim", 128))
    train_network = str(train_args.get("network", "shared"))
    reward_version = str(args.reward_version or train_args.get("reward_version", "legacy"))
    policy = build_policy(
        train_network,
        cfg.n_beams,
        hidden_dim=hidden_dim,
        device="cpu",
        use_candidate_mask=feature_flags["candidate_mask"],
        use_candidate_score=feature_flags["candidate_score"],
        use_topology_deficit=feature_flags["topology_deficit"],
        use_rule_residual=feature_flags["rule_residual"],
        rule_residual_scale=float(train_args.get("rule_residual_scale", 1.0)),
    )
    policy.model.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env_protocol = str(args.env_protocol or train_args.get("env_protocol") or inferred_env_protocol(train_args))
    resource_rows: list[dict[str, Any]] = []
    eval_rows = evaluate_policy(
        cfg,
        policy,
        torch,
        args,
        env_protocol,
        reward_version,
        progress_dir=output,
        resource_rows=resource_rows,
    )
    write_rows(output / "eval_episode_metrics.csv", eval_rows)
    write_rows(output / "resource_log.csv", resource_rows)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "marl_transfer_evaluation",
        "checkpoint": str(args.checkpoint),
        "train_algorithm": str(train_args.get("algorithm", checkpoint.get("algorithm", "unknown"))),
        "train_network": train_network,
        "train_reward_version": str(train_args.get("reward_version", "legacy")),
        "eval_reward_version": reward_version,
        "config": str(args.config),
        "output": str(args.output),
        "eval_episodes": int(args.eval_episodes),
        "slots_per_episode": int(cfg.slots_per_episode),
        "node_count": int(cfg.n_nodes),
        "area_size_m": [float(value) for value in cfg.area_size_m],
        "area_diagonal_m": math.sqrt(sum(float(value) ** 2 for value in cfg.area_size_m)),
        "beam_count": int(cfg.n_beams),
        "azimuth_cells": int(cfg.azimuth_cells),
        "elevation_cells": int(cfg.elevation_cells),
        "communication_range_m": float(cfg.communication_range_m),
        "sensing_range_m": float(cfg.sensing_range_m),
        "env_protocol": env_protocol,
        "feature_flags": feature_flags,
        "deterministic": bool(args.deterministic),
        "stochastic": bool(args.stochastic),
        "eval_both": bool(args.eval_both),
        "policy_rng_seed_policy": "torch_and_numpy_episode_seed",
        "resource_limits": {
            "max_rss_mb": float(args.max_rss_mb),
            "max_system_memory_percent": float(args.max_system_memory_percent),
        },
        "resume": {
            "enabled": not bool(args.no_resume),
            "existing_rows_used": int(getattr(args, "_resume_existing_rows", 0)),
        },
        "fast_eval": {
            "collect_slot_metrics": bool(getattr(args, "full_step_info", False)),
            "rich_info": bool(getattr(args, "full_step_info", False)),
        },
        "final_eval": eval_rows[-1] if eval_rows else {},
        "files": ["eval_episode_metrics.csv", "resource_log.csv", "manifest.json"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_checkpoint(path: str | Path, torch_module: Any) -> dict[str, Any]:
    try:
        return torch_module.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location="cpu")


def ensure_resource_args(args: argparse.Namespace) -> None:
    defaults = {
        "area_size_m": None,
        "full_step_info": False,
        "no_resume": False,
        "resource_log_period": 500,
        "max_rss_mb": 12000.0,
        "max_system_memory_percent": 92.0,
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)


def build_policy(
    network: str, *args: Any, **kwargs: Any
) -> (
    SharedBeamActorCritic
    | ScaleGraphBeamActorCritic
    | ContentionGraphActorCritic
    | GatedContentionGraphActorCritic
    | AdaptiveGatedContentionGraphActorCritic
):
    if str(network) == "shared":
        return SharedBeamActorCritic(*args, **kwargs)
    if str(network) == "scalegraph_beam":
        return ScaleGraphBeamActorCritic(*args, **kwargs)
    if str(network) == "contention_shared":
        return ContentionGraphActorCritic(*args, **kwargs)
    if str(network) == "gated_contention_shared":
        return GatedContentionGraphActorCritic(*args, **kwargs)
    if str(network) == "adaptive_gated_contention_shared":
        return AdaptiveGatedContentionGraphActorCritic(*args, **kwargs)
    raise ValueError(f"Unsupported network in checkpoint: {network}")


def override_config(config: SimulationConfig, args: argparse.Namespace) -> SimulationConfig:
    replacements: dict[str, Any] = {
        "slots_per_episode": int(args.slots),
        "episodes": int(args.eval_episodes),
        "seed": int(args.seed),
    }
    optional_fields = {
        "node_count": "n_nodes",
        "azimuth_cells": "azimuth_cells",
        "elevation_cells": "elevation_cells",
        "communication_range": "communication_range_m",
        "sensing_range": "sensing_range_m",
        "false_alarm_rate": "false_alarm_rate",
        "miss_detection_rate": "miss_detection_rate",
        "angular_cell_offset_std": "angular_cell_offset_std",
        "sensing_period_slots": "sensing_period_slots",
    }
    for arg_name, field_name in optional_fields.items():
        value = getattr(args, arg_name)
        if value is not None:
            replacements[field_name] = value
    mobility = dict(config.mobility)
    if args.mobility_model is not None:
        mobility["model"] = str(args.mobility_model)
    replacements["mobility"] = mobility
    if args.area_size_m is not None:
        replacements["area_size_m"] = tuple(float(value) for value in args.area_size_m)
    return replace(config, **replacements)


def inferred_env_protocol(train_args: dict[str, Any]) -> str:
    if bool(train_args.get("disable_isac_features", False)):
        return "structured_marl_no_isac"
    return "isac_structured_marl"


def evaluate_policy(
    cfg: SimulationConfig,
    policy: SharedBeamActorCritic,
    torch_module: Any,
    args: argparse.Namespace,
    env_protocol: str,
    reward_version: str,
    progress_dir: Path | None = None,
    resource_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = existing_eval_rows(progress_dir) if progress_dir is not None and not bool(args.no_resume) else []
    args._resume_existing_rows = len(rows)
    completed = {
        (str(row.get("phase", "")), int(row.get("eval_episode", -1)))
        for row in rows
        if str(row.get("eval_episode", "")).strip() != ""
    }
    if bool(args.eval_both):
        eval_modes = (False, True)
    elif bool(args.stochastic):
        eval_modes = (True,)
    else:
        eval_modes = (False,)
    with torch_module.no_grad():
        for mode_index, use_stochastic in enumerate(eval_modes):
            phase = "eval_stochastic" if use_stochastic else "eval_deterministic"
            for episode in range(int(args.eval_episodes)):
                if (phase, int(episode)) in completed:
                    continue
                seed = int(args.seed) + 10_000 * mode_index + episode
                torch_module.manual_seed(seed)
                np.random.seed(seed % (2**32 - 1))
                env = MarlNeighborDiscoveryEnv(
                    cfg,
                    seed=seed,
                    protocol=env_protocol,
                    reward_version=reward_version,
                    collect_slot_metrics=bool(getattr(args, "full_step_info", False)),
                    rich_info=bool(getattr(args, "full_step_info", False)),
                )
                observations, _ = env.reset(seed=seed)
                rewards = []
                truncated = False
                slot = 0
                while not truncated:
                    step = policy.act(observations, deterministic=not use_stochastic)
                    observations, reward, _terminated, truncated, _info = env.step(step.actions)
                    rewards.append(torch_module.as_tensor(reward, dtype=torch_module.float32))
                    slot += 1
                    if int(args.resource_log_period) > 0 and slot % int(args.resource_log_period) == 0:
                        snapshot = resource_snapshot()
                        snapshot.update(
                            {
                                "phase": "eval_stochastic" if use_stochastic else "eval_deterministic",
                                "eval_episode": episode,
                                "slot": slot,
                                "seed": seed,
                                "node_count": int(cfg.n_nodes),
                                "beam_count": int(cfg.n_beams),
                                "slots_per_episode": int(cfg.slots_per_episode),
                            }
                        )
                        if resource_rows is not None:
                            resource_rows.append(snapshot)
                            if progress_dir is not None:
                                write_rows(progress_dir / "resource_log.csv", resource_rows)
                        enforce_resource_limits(snapshot, args)
                rewards_tensor = torch_module.stack(rewards)
                summary = env._sim.summarize(episode).as_dict()
                row = {
                    "phase": phase,
                    "eval_episode": episode,
                    "seed": seed,
                    "env_protocol": env_protocol,
                    "episode_return_sum": float(rewards_tensor.sum().item()),
                    "episode_return_mean_per_agent": float(rewards_tensor.sum(dim=0).mean().item()),
                    "step_reward_mean": float(rewards_tensor.mean().item()),
                }
                row.update(summary)
                rows.append(row)
                if progress_dir is not None:
                    write_rows(progress_dir / "eval_episode_metrics.csv", rows)
                    progress = {
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "completed_rows": len(rows),
                        "resumed_existing_rows": int(getattr(args, "_resume_existing_rows", 0)),
                        "eval_episodes": int(args.eval_episodes),
                        "mode": "stochastic" if use_stochastic else "deterministic",
                        "eval_episode": episode,
                        "slots_per_episode": int(cfg.slots_per_episode),
                        "node_count": int(cfg.n_nodes),
                        "beam_count": int(cfg.n_beams),
                        "latest": row,
                    }
                    (progress_dir / "progress.json").write_text(
                        json.dumps(progress, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                print(
                    json.dumps(
                        {
                            "phase": "eval_progress",
                            "completed_rows": len(rows),
                            "mode": "stochastic" if use_stochastic else "deterministic",
                            "eval_episode": episode,
                            "discovery_rate": row.get("discovery_rate"),
                            "lambda2": row.get("lambda2"),
                            "collision_count": row.get("collision_count"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    return rows


def resource_snapshot() -> dict[str, Any]:
    try:
        import psutil
    except ImportError:
        return windows_resource_snapshot()
    process = psutil.Process()
    memory = psutil.virtual_memory()
    return {
        "rss_mb": process.memory_info().rss / (1024.0 * 1024.0),
        "process_cpu_percent": process.cpu_percent(interval=None),
        "system_memory_percent": memory.percent,
        "system_available_mb": memory.available / (1024.0 * 1024.0),
    }


def windows_resource_snapshot() -> dict[str, Any]:
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return empty_resource_snapshot()
    if not hasattr(ctypes, "windll"):
        return empty_resource_snapshot()

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        memory = MemoryStatusEx()
        memory.dwLength = ctypes.sizeof(MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            return empty_resource_snapshot()
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(ProcessMemoryCounters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            rss_mb = current_process_rss_mb_fallback()
        else:
            rss_mb = counters.WorkingSetSize / (1024.0 * 1024.0)
        available_mb = memory.ullAvailPhys / (1024.0 * 1024.0)
        total_mb = max(memory.ullTotalPhys / (1024.0 * 1024.0), 1e-9)
        return {
            "rss_mb": rss_mb,
            "process_cpu_percent": "",
            "system_memory_percent": 100.0 * (1.0 - available_mb / total_mb),
            "system_available_mb": available_mb,
        }
    except Exception:
        return empty_resource_snapshot()


def current_process_rss_mb_fallback() -> float | str:
    try:
        import subprocess

        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Get-Process -Id {os.getpid()}).WorkingSet64",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        value = completed.stdout.strip()
        return float(value) / (1024.0 * 1024.0) if value else ""
    except Exception:
        return ""


def empty_resource_snapshot() -> dict[str, Any]:
    return {
        "rss_mb": "",
        "process_cpu_percent": "",
        "system_memory_percent": "",
        "system_available_mb": "",
    }


def enforce_resource_limits(snapshot: dict[str, Any], args: argparse.Namespace) -> None:
    rss = snapshot.get("rss_mb")
    memory_percent = snapshot.get("system_memory_percent")
    if isinstance(rss, (int, float)) and rss > float(args.max_rss_mb):
        raise RuntimeError(f"RSS memory limit exceeded: {rss:.1f} MB > {float(args.max_rss_mb):.1f} MB")
    if isinstance(memory_percent, (int, float)) and memory_percent > float(args.max_system_memory_percent):
        raise RuntimeError(
            f"System memory limit exceeded: {memory_percent:.1f}% > {float(args.max_system_memory_percent):.1f}%"
        )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def existing_eval_rows(progress_dir: Path) -> list[dict[str, Any]]:
    data_path = progress_dir / "eval_episode_metrics.csv"
    if not data_path.exists() or data_path.stat().st_size == 0:
        return []
    try:
        with data_path.open("r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def main() -> None:
    print(json.dumps(run_evaluation(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
