from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime
import json
from itertools import product
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

import yaml

from .config import SimulationConfig, load_config
from .runner import run_detailed


DEFAULT_GROUP_KEYS = (
    "protocol",
    "mobility_model",
    "node_count",
    "slots",
    "azimuth_cells",
    "elevation_cells",
)


@dataclass(frozen=True)
class SweepCase:
    case_id: int
    base_seed: int
    mobility: dict[str, Any]
    node_count: int
    slots_per_episode: int
    azimuth_cells: int
    elevation_cells: int
    episodes: int | None = None

    @property
    def mobility_model(self) -> str:
        return str(self.mobility.get("model", "gauss_markov"))

    @property
    def beam_cells(self) -> int:
        return self.azimuth_cells * self.elevation_cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dynamic rule-protocol simulation sweeps.")
    parser.add_argument(
        "--config",
        default="05_simulation/configs/dynamic_rule_sweep.yaml",
        help="Path to sweep YAML config.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Defaults to sweep.output_root/name_timestamp.",
    )
    return parser.parse_args()


def run_sweep_from_config(config_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_path = resolve_base_config_path(config_path, raw)
    base_config = load_config(base_path)
    sweep = raw.get("sweep", {})
    protocols = tuple(
        str(p)
        for p in raw.get("protocols", sweep.get("protocols", raw.get("baselines", base_config.baselines)))
    )
    cases = list(iter_sweep_cases(raw, base_config))
    if not cases:
        raise ValueError("Sweep config produced no cases.")

    run_dir = resolve_output_dir(config_path, raw, output_dir)
    rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for case in cases:
        case_config = case_to_config(base_config, case)
        case_rows, case_slot_rows, case_edge_rows = run_detailed(case_config, list(protocols))
        rows.extend(add_case_metadata(row, case) for row in case_rows)
        slot_rows.extend(add_case_metadata(row, case) for row in case_slot_rows)
        edge_rows.extend(add_case_metadata(row, case) for row in case_edge_rows)

    aggregate_rows = aggregate_metrics(rows)
    write_sweep_outputs(config_path, base_path, run_dir, rows, slot_rows, edge_rows, aggregate_rows, raw)
    return {
        "run_dir": str(run_dir),
        "case_count": len(cases),
        "episode_rows": len(rows),
        "slot_rows": len(slot_rows),
        "edge_rows": len(edge_rows),
        "aggregate_rows": len(aggregate_rows),
        "protocols": list(protocols),
    }


def resolve_base_config_path(sweep_config_path: Path, raw: dict[str, Any]) -> Path:
    base = raw.get("base_config", raw.get("base_config_path", "05_simulation/configs/mobile_smoke.yaml"))
    path = Path(str(base))
    if path.is_absolute():
        return path
    repo_relative = Path.cwd() / path
    if repo_relative.exists():
        return repo_relative
    return sweep_config_path.parent / path


def resolve_output_dir(config_path: Path, raw: dict[str, Any], output_dir: str | Path | None) -> Path:
    if output_dir:
        return Path(output_dir)
    name = str(raw.get("name", raw.get("experiment", {}).get("name", config_path.stem)))
    output_root = Path(str(raw.get("output_root", "05_simulation/results_raw/dynamic_rule_sweep")))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"{name}_{stamp}"


def iter_sweep_cases(raw: dict[str, Any], base_config: SimulationConfig) -> Iterable[SweepCase]:
    sweep = raw.get("sweep", raw)
    seeds = as_list(sweep.get("seeds", base_config.seed))
    mobilities = normalize_mobility_sweep(sweep.get("mobility", sweep.get("mobility_models", [base_config.mobility])))
    slots = as_list(sweep.get("slots", sweep.get("slots_per_episode", base_config.slots_per_episode)))
    node_counts = as_list(sweep.get("node_count", sweep.get("node_counts", base_config.n_nodes)))
    beam_cells = normalize_beam_cells(
        sweep.get(
            "beam_cells",
            [{"azimuth_cells": base_config.azimuth_cells, "elevation_cells": base_config.elevation_cells}],
        )
    )
    episodes = sweep.get("episodes", raw.get("episodes", None))

    case_id = 0
    for seed, mobility, slot_count, node_count, beam_cell in product(seeds, mobilities, slots, node_counts, beam_cells):
        yield SweepCase(
            case_id=case_id,
            base_seed=int(seed),
            mobility=mobility,
            node_count=int(node_count),
            slots_per_episode=int(slot_count),
            azimuth_cells=int(beam_cell["azimuth_cells"]),
            elevation_cells=int(beam_cell["elevation_cells"]),
            episodes=int(episodes) if episodes is not None else None,
        )
        case_id += 1


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def normalize_mobility_sweep(value: Any) -> list[dict[str, Any]]:
    entries = as_list(value)
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, str):
            normalized.append({"model": entry})
        elif isinstance(entry, dict):
            mobility = dict(entry)
            if "default_model" in mobility and "model" not in mobility:
                mobility["model"] = mobility["default_model"]
            normalized.append(mobility)
        else:
            raise TypeError(f"Unsupported mobility sweep entry: {entry!r}")
    return normalized


def normalize_beam_cells(value: Any) -> list[dict[str, int]]:
    cells: list[dict[str, int]] = []
    for entry in as_list(value):
        if isinstance(entry, dict):
            cells.append(
                {
                    "azimuth_cells": int(entry["azimuth_cells"]),
                    "elevation_cells": int(entry["elevation_cells"]),
                }
            )
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            cells.append({"azimuth_cells": int(entry[0]), "elevation_cells": int(entry[1])})
        elif isinstance(entry, str) and "x" in entry.lower():
            azimuth, elevation = entry.lower().split("x", maxsplit=1)
            cells.append({"azimuth_cells": int(azimuth), "elevation_cells": int(elevation)})
        else:
            raise TypeError(f"Unsupported beam_cells entry: {entry!r}")
    return cells


def case_to_config(base_config: SimulationConfig, case: SweepCase) -> SimulationConfig:
    mobility = dict(base_config.mobility)
    mobility.update(case.mobility)
    return replace(
        base_config,
        seed=case.base_seed,
        episodes=case.episodes or base_config.episodes,
        slots_per_episode=case.slots_per_episode,
        n_nodes=case.node_count,
        mobility=mobility,
        azimuth_cells=case.azimuth_cells,
        elevation_cells=case.elevation_cells,
    )


def add_case_metadata(row: dict[str, Any], case: SweepCase) -> dict[str, Any]:
    enriched = dict(row)
    enriched.update(
        {
            "case_id": case.case_id,
            "base_seed": case.base_seed,
            "node_count": case.node_count,
            "azimuth_cells": case.azimuth_cells,
            "elevation_cells": case.elevation_cells,
            "beam_cells": case.beam_cells,
            "sweep_mobility_model": case.mobility_model,
        }
    )
    return enriched


def aggregate_metrics(rows: list[dict[str, Any]], group_keys: tuple[str, ...] = DEFAULT_GROUP_KEYS) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(group_key) for group_key in group_keys)
        groups.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and key not in set(group_keys) | {"episode", "seed", "base_seed", "case_id"}
        }
    )
    for key, group_rows in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        aggregate: dict[str, Any] = dict(zip(group_keys, key))
        aggregate["n_episodes"] = len(group_rows)
        aggregate["case_ids"] = ";".join(str(row["case_id"]) for row in group_rows)
        for metric in numeric_keys:
            values = [float(row[metric]) for row in group_rows if metric in row]
            if not values:
                continue
            aggregate[f"{metric}_mean"] = mean(values)
            aggregate[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
            aggregate[f"{metric}_min"] = min(values)
            aggregate[f"{metric}_max"] = max(values)
        aggregate_rows.append(aggregate)
    return aggregate_rows


def write_sweep_outputs(
    sweep_config_path: Path,
    base_config_path: Path,
    output_dir: Path,
    episode_rows: list[dict[str, Any]],
    slot_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    raw_config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sweep_config.yaml").write_text(
        sweep_config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (output_dir / "base_config_path.txt").write_text(str(base_config_path) + "\n", encoding="utf-8")
    write_rows_csv(output_dir / "per_episode_summary.csv", episode_rows)
    write_rows_csv(output_dir / "per_slot_metrics.csv", slot_rows)
    write_rows_csv(output_dir / "discovered_edges.csv", edge_rows)
    write_rows_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregate_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_readme(output_dir, episode_rows, slot_rows, edge_rows, aggregate_rows, raw_config)


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(
    output_dir: Path,
    episode_rows: list[dict[str, Any]],
    slot_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    raw_config: dict[str, Any],
) -> None:
    name = str(raw_config.get("name", raw_config.get("experiment", {}).get("name", "dynamic_rule_sweep")))
    protocols = sorted({str(row["protocol"]) for row in episode_rows})
    text = "\n".join(
        [
            "# Dynamic Rule Sweep",
            "",
            f"- Name: `{name}`",
            f"- Protocols: {', '.join(protocols)}",
            f"- Episode rows: {len(episode_rows)}",
            f"- Slot rows: {len(slot_rows)}",
            f"- Edge rows: {len(edge_rows)}",
            f"- Aggregate rows: {len(aggregate_rows)}",
            "",
            "Generated by `isac_nd_sim.sweep`.",
        ]
    )
    (output_dir / "README.md").write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result = run_sweep_from_config(args.config, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
