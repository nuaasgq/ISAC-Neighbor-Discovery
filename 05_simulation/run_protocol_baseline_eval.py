from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "05_simulation" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isac_nd_sim.config import SimulationConfig, load_config  # noqa: E402
from isac_nd_sim.runner import stable_protocol_offset, with_metric_aliases  # noqa: E402
from isac_nd_sim.simulator import NeighborDiscoverySimulator  # noqa: E402


BEAMWIDTH_TO_CELLS = {
    3: (120, 60),
    5: (72, 36),
    10: (36, 18),
    15: (24, 12),
    30: (12, 6),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate non-neural protocol baselines and emit MARL-compatible evaluation artifacts."
    )
    parser.add_argument("--config", default="05_simulation/configs/paper_transfer_train_n10_b10_singlehop.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["uniform_random", "skyorbs_like_skip_scan"],
        help="One or more simulator protocols. Multiple protocols create one child directory per protocol.",
    )
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--slots", type=int, default=3000)
    parser.add_argument("--node-count", type=int, default=None)
    parser.add_argument("--area-size-m", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--beamwidth-deg", type=int, choices=sorted(BEAMWIDTH_TO_CELLS), default=None)
    parser.add_argument("--azimuth-cells", type=int, default=None)
    parser.add_argument("--elevation-cells", type=int, default=None)
    parser.add_argument("--communication-range", type=float, default=None)
    parser.add_argument("--sensing-range", type=float, default=None)
    parser.add_argument("--false-alarm-rate", type=float, default=None)
    parser.add_argument("--miss-detection-rate", type=float, default=None)
    parser.add_argument("--angular-cell-offset-std", type=float, default=None)
    parser.add_argument("--sensing-period-slots", type=int, default=None)
    parser.add_argument("--slot-metric-period", type=int, default=0)
    parser.add_argument(
        "--target-status-diagnostics",
        action="store_true",
        help="Classify selected beams with offline true topology; disabled by default because it is expensive.",
    )
    parser.add_argument("--mobility-model", default=None)
    parser.add_argument("--spatial-dimensions", type=int, choices=(2, 3), default=None)
    parser.add_argument("--seed", type=int, default=20364205)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.eval_episodes) <= 0:
        raise ValueError("--eval-episodes must be positive.")
    if int(args.slots) <= 0:
        raise ValueError("--slots must be positive.")

    base_cfg = override_config(load_config(args.config), args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protocol_outputs = []
    for protocol in args.protocols:
        protocol = str(protocol)
        run_output = output if len(args.protocols) == 1 else output / protocol
        manifest = run_protocol_eval(protocol, base_cfg, args, run_output)
        protocol_outputs.append({"protocol": protocol, "output": str(run_output), "manifest": manifest})

    index_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "protocol_baseline_eval_index",
        "config": str(args.config),
        "output": str(output),
        "protocol_outputs": [
            {"protocol": item["protocol"], "output": item["output"]} for item in protocol_outputs
        ],
    }
    (output / "protocol_baseline_eval_index.json").write_text(
        json.dumps(index_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not args.quiet:
        print(json.dumps(index_manifest, ensure_ascii=False, indent=2))


def override_config(config: SimulationConfig, args: argparse.Namespace) -> SimulationConfig:
    replacements: dict[str, Any] = {
        "slots_per_episode": int(args.slots),
        "episodes": int(args.eval_episodes),
        "seed": int(args.seed),
        "slot_metric_period": int(args.slot_metric_period),
    }
    if args.node_count is not None:
        replacements["n_nodes"] = int(args.node_count)
    if args.area_size_m is not None:
        replacements["area_size_m"] = tuple(float(value) for value in args.area_size_m)
    if args.beamwidth_deg is not None:
        replacements["azimuth_cells"], replacements["elevation_cells"] = BEAMWIDTH_TO_CELLS[int(args.beamwidth_deg)]
    if args.azimuth_cells is not None:
        replacements["azimuth_cells"] = int(args.azimuth_cells)
    if args.elevation_cells is not None:
        replacements["elevation_cells"] = int(args.elevation_cells)
    optional_fields = {
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
    spatial_dims = getattr(args, "spatial_dimensions", None)
    if spatial_dims is not None:
        mobility["spatial_dimensions"] = int(spatial_dims)
    replacements["mobility"] = mobility
    return replace(config, **replacements)


def run_protocol_eval(protocol: str, cfg: SimulationConfig, args: argparse.Namespace, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    offset = stable_protocol_offset(protocol)
    for episode in range(int(args.eval_episodes)):
        scenario_seed = int(args.seed) + episode
        policy_seed = scenario_seed + offset
        simulator = NeighborDiscoverySimulator(
            cfg,
            protocol,
            policy_seed,
            scenario_seed=scenario_seed,
            collect_target_status_metrics=bool(args.target_status_diagnostics),
        )
        row = with_metric_aliases(simulator.run_episode(episode).as_dict(), cfg.n_nodes)
        row.update(
            {
                "phase": "eval_stochastic",
                "eval_episode": int(episode),
                "protocol": protocol,
                "method": method_name(protocol),
                "scenario_seed": int(scenario_seed),
                "policy_seed": int(policy_seed),
            }
        )
        rows.append(row)
        # Persist episode-level progress so a resource timeout does not erase
        # all completed seeds for a long narrow-beam baseline run.
        write_rows(output / "eval_episode_metrics.csv", rows)
    write_rows(output / "eval_episode_metrics.csv", rows)
    write_rows(output / "resource_log.csv", resource_rows)
    manifest = build_manifest(protocol, cfg, args, output, rows)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_manifest(
    protocol: str,
    cfg: SimulationConfig,
    args: argparse.Namespace,
    output: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    beamwidth = 360.0 / max(1, int(cfg.azimuth_cells))
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "marl_transfer_evaluation",
        "method": method_name(protocol),
        "method_label": method_label(protocol),
        "checkpoint": "",
        "train_algorithm": "protocol_baseline",
        "train_network": method_name(protocol),
        "train_reward_version": "none",
        "eval_reward_version": "none",
        "config": str(args.config),
        "output": str(output),
        "eval_episodes": int(args.eval_episodes),
        "slots_per_episode": int(cfg.slots_per_episode),
        "node_count": int(cfg.n_nodes),
        "area_size_m": [float(value) for value in cfg.area_size_m],
        "area_diagonal_m": math.sqrt(sum(float(value) ** 2 for value in cfg.area_size_m)),
        "beam_count": int(cfg.n_beams),
        "beamwidth_deg": float(beamwidth),
        "azimuth_cells": int(cfg.azimuth_cells),
        "elevation_cells": int(cfg.elevation_cells),
        "communication_range_m": float(cfg.communication_range_m),
        "sensing_range_m": float(cfg.sensing_range_m),
        "communication_phy": {
            "model": cfg.communication_phy_model,
            "carrier_frequency_hz": cfg.communication_carrier_frequency_hz,
            "bandwidth_hz": cfg.communication_bandwidth_hz,
            "tx_power_w": cfg.communication_tx_power_w,
            "noise_figure_db": cfg.communication_noise_figure_db,
            "path_loss_exponent": cfg.communication_path_loss_exponent,
            "shadowing_std_db": cfg.communication_shadowing_std_db,
            "rician_k_db": cfg.communication_rician_k_db,
            "sinr_threshold_db": cfg.communication_sinr_threshold_db,
            "sidelobe_gain_db": cfg.communication_sidelobe_gain_db,
            "antenna_gain_mode": cfg.communication_antenna_gain_mode,
            "channel_seed_policy": "scenario_seed_only",
        },
        "sensing_seed_policy": "scenario_slot_node_beam_event",
        "shared_waveform_power_enabled": bool(cfg.shared_waveform_power_enabled),
        "env_protocol": protocol,
        "feature_flags": {
            "candidate_mask": False,
            "candidate_score": False,
            "topology_deficit": protocol not in {"uniform_random", "skyorbs_like_skip_scan"},
            "rule_residual": False,
        },
        "deterministic": protocol == "skyorbs_like_skip_scan",
        "stochastic": protocol != "skyorbs_like_skip_scan",
        "eval_both": False,
        "target_status_diagnostics": bool(args.target_status_diagnostics),
        "final_eval": rows[-1] if rows else {},
        "files": ["eval_episode_metrics.csv", "resource_log.csv", "manifest.json"],
    }


def method_name(protocol: str) -> str:
    if protocol == "skyorbs_like_skip_scan":
        return "skyorbs_like"
    return protocol


def method_label(protocol: str) -> str:
    labels = {
        "uniform_random": "Uniform random",
        "skyorbs_like_skip_scan": "SkyOrbs-like",
        "rl_no_isac": "Proxy RL no-ISAC",
        "improved_rl_no_isac": "Proxy improved no-ISAC",
        "improved_rl_isac": "Proxy improved ISAC",
        "trust_gated_isac_tables": "Trust-gated ISAC tables",
        "budgeted_collision_aware_isac": "Budgeted collision-aware ISAC",
        "wang2025_isac_tables": "Wang2025 ISAC with table exchange",
        "improved_rl_isac_tables": "Rule ISAC with table exchange",
        "position_ordered_isac_rendezvous": "Position-ordered ISAC rendezvous diagnostic",
    }
    return labels.get(protocol, protocol.replace("_", " "))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
