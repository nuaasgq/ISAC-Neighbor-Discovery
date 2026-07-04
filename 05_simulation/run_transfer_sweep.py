from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "05_simulation" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isac_nd_sim.config import SimulationConfig, load_config  # noqa: E402
from isac_nd_sim.runner import run_detailed  # noqa: E402


DEFAULT_PROTOCOLS = (
    "skyorbs_like_skip_scan",
    "uniform_random",
    "rl_no_isac",
    "improved_rl_no_isac",
    "improved_rl_isac",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run node-scale and beamwidth transfer sweeps.")
    parser.add_argument("--config", default=str(ROOT / "05_simulation" / "configs" / "paper_core_d1.yaml"))
    parser.add_argument("--trained-config", required=True, help="best_config.yaml produced by run_training.py.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--node-counts", default="10,20,50,100")
    parser.add_argument("--beamwidth-degs", default="3,5,10,15,30")
    parser.add_argument("--mobilities", default="gauss_markov")
    parser.add_argument("--seeds", default="20290704")
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--slots", type=int, default=200)
    parser.add_argument("--slot-metric-period", type=int, default=0)
    parser.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS))
    parser.add_argument("--area-scale", choices=("density", "fixed"), default="density")
    parser.add_argument("--range-mode", choices=("base", "singlehop"), default="base")
    parser.add_argument(
        "--communication-range-ratios",
        default="",
        help="Optional comma-separated Rc/area-diagonal ratios. When set, overrides --range-mode for communication range.",
    )
    parser.add_argument(
        "--sensing-to-comm-ratios",
        default="",
        help="Optional comma-separated Rs/Rc ratios. Use with --communication-range-ratios for range-relation sweeps.",
    )
    parser.add_argument("--false-alarm-rates", default="", help="Optional comma-separated ISAC false-alarm rates.")
    parser.add_argument("--miss-detection-rates", default="", help="Optional comma-separated ISAC miss-detection rates.")
    parser.add_argument("--angular-cell-offsets", default="", help="Optional comma-separated ISAC angular error std in beam cells.")
    parser.add_argument(
        "--isac-error-profiles",
        default="",
        help=(
            "Optional semicolon-separated Pfa:Pmd:sigma_cell triples. "
            "When set, this paired profile list overrides the individual ISAC error lists."
        ),
    )
    parser.add_argument("--train-node-count", type=int, default=10)
    parser.add_argument("--train-beamwidth-deg", type=float, default=10.0)
    parser.add_argument("--alignment-tolerance-cells", type=int, default=0)
    parser.add_argument("--name", default="transfer_sweep")
    return parser.parse_args()


def load_trained_parameters(path: str | Path) -> dict[str, float]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    params = doc.get("shared_policy_parameters", {})
    if not params:
        raise ValueError(f"No shared_policy_parameters found in {path}.")
    return {str(key): float(value) for key, value in params.items()}


def parse_csv_numbers(value: str, cast: type = int) -> list[Any]:
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def parse_optional_csv_numbers(value: str, cast: type = float) -> list[Any | None]:
    parsed = parse_csv_numbers(value, cast)
    return parsed if parsed else [None]


def parse_isac_error_profiles(value: str) -> list[tuple[float | None, float | None, float | None]] | None:
    if not value.strip():
        return None
    profiles: list[tuple[float | None, float | None, float | None]] = []
    for raw_profile in value.split(";"):
        raw_profile = raw_profile.strip()
        if not raw_profile:
            continue
        parts = [part.strip() for part in raw_profile.split(":")]
        if len(parts) != 3:
            raise ValueError(
                "--isac-error-profiles entries must be Pfa:Pmd:sigma_cell triples, "
                f"got {raw_profile!r}."
            )
        profiles.append(tuple(float(part) for part in parts))
    if not profiles:
        raise ValueError("--isac-error-profiles was set but contained no valid triples.")
    return profiles


def beam_cells_from_width(width_deg: float) -> tuple[int, int]:
    if width_deg <= 0:
        raise ValueError("beamwidth must be positive.")
    azimuth_cells = max(1, int(round(360.0 / width_deg)))
    elevation_cells = max(1, int(round(180.0 / width_deg)))
    return azimuth_cells, elevation_cells


def scaled_area(base: tuple[float, float, float], node_count: int, reference_nodes: int, mode: str) -> tuple[float, float, float]:
    if mode == "fixed":
        return base
    factor = (max(1, node_count) / max(1, reference_nodes)) ** (1.0 / 3.0)
    return tuple(float(value) * factor for value in base)


def configure_case(
    base: SimulationConfig,
    params: dict[str, float],
    seed: int,
    mobility: str,
    node_count: int,
    beamwidth_deg: float,
    args: argparse.Namespace,
    communication_range_ratio: float | None = None,
    sensing_to_comm_ratio: float | None = None,
    false_alarm_rate: float | None = None,
    miss_detection_rate: float | None = None,
    angular_cell_offset: float | None = None,
) -> SimulationConfig:
    azimuth_cells, elevation_cells = beam_cells_from_width(beamwidth_deg)
    mobility_cfg = dict(base.mobility)
    mobility_cfg["model"] = mobility
    area = scaled_area(base.area_size_m, int(node_count), int(args.train_node_count), args.area_scale)
    area_diagonal = math.sqrt(sum(float(value) ** 2 for value in area))
    if communication_range_ratio is not None:
        communication_range = area_diagonal * float(communication_range_ratio)
        sensing_range = communication_range * float(1.0 if sensing_to_comm_ratio is None else sensing_to_comm_ratio)
    elif args.range_mode == "singlehop":
        communication_range = area_diagonal * 1.05
        sensing_range = communication_range
    else:
        communication_range = base.communication_range_m
        sensing_range = base.sensing_range_m
    return replace(
        base,
        seed=int(seed),
        episodes=int(args.episodes_per_seed),
        slots_per_episode=int(args.slots),
        slot_metric_period=int(args.slot_metric_period),
        n_nodes=int(node_count),
        area_size_m=area,
        communication_range_m=float(communication_range),
        sensing_range_m=float(sensing_range),
        mobility=mobility_cfg,
        azimuth_cells=azimuth_cells,
        elevation_cells=elevation_cells,
        alignment_tolerance_cells=int(args.alignment_tolerance_cells),
        false_alarm_rate=float(base.false_alarm_rate if false_alarm_rate is None else false_alarm_rate),
        miss_detection_rate=float(base.miss_detection_rate if miss_detection_rate is None else miss_detection_rate),
        angular_cell_offset_std=float(base.angular_cell_offset_std if angular_cell_offset is None else angular_cell_offset),
        **params,
    )


def enrich_row(row: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    enriched.update(case)
    return enriched


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_keys = (
        "protocol",
        "mobility_model",
        "area_scale",
        "range_mode",
        "range_sweep_enabled",
        "node_count",
        "beamwidth_deg",
        "azimuth_cells",
        "elevation_cells",
        "beam_count",
        "communication_range_to_diagonal_ratio",
        "sensing_to_comm_range_ratio",
        "false_alarm_rate",
        "miss_detection_rate",
        "angular_cell_offset_std",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in group_keys), []).append(row)
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and key not in set(group_keys) | {"episode", "seed", "scenario_seed"}
        }
    )
    output: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        aggregate = dict(zip(group_keys, key))
        aggregate["n_episodes"] = len(items)
        for metric in numeric_keys:
            values = [float(row[metric]) for row in items if metric in row]
            if not values:
                continue
            aggregate[f"{metric}_mean"] = sum(values) / len(values)
            aggregate[f"{metric}_min"] = min(values)
            aggregate[f"{metric}_max"] = max(values)
            aggregate[f"{metric}_std"] = pstdev(values)
        output.append(aggregate)
    return output


def pstdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_transfer_sweep(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config)
    trained_path = Path(args.trained_config)
    output_dir = Path(args.output)
    base = load_config(config_path)
    params = load_trained_parameters(trained_path)
    protocols = [item.strip() for item in args.protocols.split(",") if item.strip()]
    node_counts = parse_csv_numbers(args.node_counts, int)
    beamwidths = parse_csv_numbers(args.beamwidth_degs, float)
    mobilities = [item.strip() for item in args.mobilities.split(",") if item.strip()]
    seeds = parse_csv_numbers(args.seeds, int)
    communication_range_ratios = parse_optional_csv_numbers(args.communication_range_ratios, float)
    sensing_to_comm_ratios = parse_optional_csv_numbers(args.sensing_to_comm_ratios, float)
    false_alarm_rates = parse_optional_csv_numbers(args.false_alarm_rates, float)
    miss_detection_rates = parse_optional_csv_numbers(args.miss_detection_rates, float)
    angular_cell_offsets = parse_optional_csv_numbers(args.angular_cell_offsets, float)
    paired_error_profiles = parse_isac_error_profiles(args.isac_error_profiles)
    if paired_error_profiles is None:
        error_profiles = [
            (false_alarm_rate, miss_detection_rate, angular_cell_offset)
            for false_alarm_rate in false_alarm_rates
            for miss_detection_rate in miss_detection_rates
            for angular_cell_offset in angular_cell_offsets
        ]
    else:
        error_profiles = paired_error_profiles

    output_dir.mkdir(parents=True, exist_ok=True)
    episode_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    case_count = 0
    for seed in seeds:
        for mobility in mobilities:
            for node_count in node_counts:
                for beamwidth in beamwidths:
                    for communication_range_ratio in communication_range_ratios:
                        for sensing_to_comm_ratio in sensing_to_comm_ratios:
                            for false_alarm_rate, miss_detection_rate, angular_cell_offset in error_profiles:
                                case_count += 1
                                cfg = configure_case(
                                    base,
                                    params,
                                    seed,
                                    mobility,
                                    node_count,
                                    beamwidth,
                                    args,
                                    communication_range_ratio,
                                    sensing_to_comm_ratio,
                                    false_alarm_rate,
                                    miss_detection_rate,
                                    angular_cell_offset,
                                )
                                azimuth_cells, elevation_cells = beam_cells_from_width(beamwidth)
                                area_diagonal = math.sqrt(sum(float(value) ** 2 for value in cfg.area_size_m))
                                case = {
                                    "case_id": case_count - 1,
                                    "base_seed": int(seed),
                                    "node_count": int(node_count),
                                    "beamwidth_deg": float(beamwidth),
                                    "azimuth_cells": azimuth_cells,
                                    "elevation_cells": elevation_cells,
                                    "beam_count": azimuth_cells * elevation_cells,
                                    "area_diagonal_m": float(area_diagonal),
                                    "communication_range_m": float(cfg.communication_range_m),
                                    "sensing_range_m": float(cfg.sensing_range_m),
                                    "communication_range_to_diagonal_ratio": float(
                                        cfg.communication_range_m / max(area_diagonal, 1e-9)
                                    ),
                                    "sensing_range_to_diagonal_ratio": float(cfg.sensing_range_m / max(area_diagonal, 1e-9)),
                                    "sensing_to_comm_range_ratio": float(cfg.sensing_range_m / max(cfg.communication_range_m, 1e-9)),
                                    "false_alarm_rate": float(cfg.false_alarm_rate),
                                    "miss_detection_rate": float(cfg.miss_detection_rate),
                                    "angular_cell_offset_std": float(cfg.angular_cell_offset_std),
                                    "area_scale": args.area_scale,
                                    "range_mode": args.range_mode,
                                    "range_sweep_enabled": communication_range_ratio is not None,
                                    "isac_error_sweep_enabled": any(
                                        value is not None
                                        for value in (false_alarm_rate, miss_detection_rate, angular_cell_offset)
                                    ),
                                    "train_node_count": int(args.train_node_count),
                                    "train_beamwidth_deg": float(args.train_beamwidth_deg),
                                }
                                rows, slots, edges = run_detailed(cfg, protocols)
                                episode_rows.extend(enrich_row(row, case) for row in rows)
                                slot_rows.extend(enrich_row(row, case) for row in slots)
                                edge_rows.extend(enrich_row(row, case) for row in edges)

    aggregate = aggregate_rows(episode_rows)
    write_rows(output_dir / "per_episode_summary.csv", episode_rows)
    write_rows(output_dir / "per_slot_metrics.csv", slot_rows)
    write_rows(output_dir / "discovered_edges.csv", edge_rows)
    write_rows(output_dir / "aggregate_metrics.csv", aggregate)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "name": args.name,
        "config": str(config_path),
        "trained_config": str(trained_path),
        "protocols": protocols,
        "node_counts": node_counts,
        "beamwidth_degs": beamwidths,
        "mobilities": mobilities,
        "seeds": seeds,
        "communication_range_ratios": communication_range_ratios,
        "sensing_to_comm_ratios": sensing_to_comm_ratios,
        "false_alarm_rates": false_alarm_rates,
        "miss_detection_rates": miss_detection_rates,
        "angular_cell_offsets": angular_cell_offsets,
        "isac_error_profiles": paired_error_profiles,
        "episodes_per_seed": args.episodes_per_seed,
        "slots": args.slots,
        "slot_metric_period": args.slot_metric_period,
        "area_scale": args.area_scale,
        "range_mode": args.range_mode,
        "train_node_count": args.train_node_count,
        "train_beamwidth_deg": args.train_beamwidth_deg,
        "case_count": case_count,
        "episode_rows": len(episode_rows),
        "slot_rows": len(slot_rows),
        "edge_rows": len(edge_rows),
        "aggregate_rows": len(aggregate),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Transfer Sweep",
                "",
                f"- Name: `{args.name}`",
                f"- Training scale: N={args.train_node_count}, beamwidth={args.train_beamwidth_deg:g} deg",
                f"- Test node counts: {', '.join(map(str, node_counts))}",
                f"- Test beamwidths: {', '.join(f'{v:g}' for v in beamwidths)} deg",
                f"- Area scaling: `{args.area_scale}`",
                f"- Range mode: `{args.range_mode}`",
                f"- Communication range ratios Rc/D: {', '.join('base' if v is None else f'{v:g}' for v in communication_range_ratios)}",
                f"- Sensing/communication range ratios Rs/Rc: {', '.join('base' if v is None else f'{v:g}' for v in sensing_to_comm_ratios)}",
                f"- False alarm rates: {', '.join('base' if v is None else f'{v:g}' for v in false_alarm_rates)}",
                f"- Miss detection rates: {', '.join('base' if v is None else f'{v:g}' for v in miss_detection_rates)}",
                f"- Angular cell offsets: {', '.join('base' if v is None else f'{v:g}' for v in angular_cell_offsets)}",
                f"- ISAC error profiles: {format_error_profiles(error_profiles)}",
                f"- Aggregate rows: {len(aggregate)}",
                "",
                "Generated by `05_simulation/run_transfer_sweep.py`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def format_error_profiles(profiles: Iterable[tuple[float | None, float | None, float | None]]) -> str:
    tokens = []
    for false_alarm_rate, miss_detection_rate, angular_cell_offset in profiles:
        if false_alarm_rate is None and miss_detection_rate is None and angular_cell_offset is None:
            tokens.append("base")
        else:
            values = [false_alarm_rate, miss_detection_rate, angular_cell_offset]
            tokens.append("(" + ", ".join("base" if value is None else f"{value:g}" for value in values) + ")")
    return ", ".join(tokens)


def main() -> None:
    print(json.dumps(run_transfer_sweep(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
