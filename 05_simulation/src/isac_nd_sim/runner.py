from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime
import json
import shutil
from pathlib import Path

from .config import SimulationConfig, load_config
from .simulator import NeighborDiscoverySimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ISAC-assisted neighbor discovery simulations.")
    parser.add_argument("--config", default="05_simulation/configs/mobile_smoke.yaml", help="Path to YAML config.")
    parser.add_argument("--output", default="05_simulation/results_raw/smoke_dynamic", help="Output directory.")
    parser.add_argument("--episodes", type=int, default=None, help="Override episode count.")
    parser.add_argument("--slots", type=int, default=None, help="Override slots per episode.")
    parser.add_argument("--protocols", default=None, help="Comma-separated protocol list.")
    parser.add_argument("--mobility", default=None, help="Override mobility model.")
    return parser.parse_args()


def override_config(config: SimulationConfig, episodes: int | None, slots: int | None) -> SimulationConfig:
    values = config.__dict__.copy()
    if episodes is not None:
        values["episodes"] = episodes
    if slots is not None:
        values["slots_per_episode"] = slots
    return SimulationConfig(**values)


def override_mobility(config: SimulationConfig, mobility: str | None) -> SimulationConfig:
    if not mobility:
        return config
    mobility_cfg = dict(config.mobility)
    mobility_cfg["model"] = mobility
    return replace(config, mobility=mobility_cfg)


def run(config: SimulationConfig, protocols: list[str]) -> list[dict]:
    rows, _slot_rows, _edge_rows = run_detailed(config, protocols)
    return rows


def run_detailed(config: SimulationConfig, protocols: list[str]) -> tuple[list[dict], list[dict], list[dict]]:
    rows: list[dict] = []
    slot_rows: list[dict] = []
    edge_rows: list[dict] = []
    for protocol in protocols:
        for episode in range(config.episodes):
            seed = config.seed + 1009 * episode + stable_protocol_offset(protocol)
            simulator = NeighborDiscoverySimulator(config, protocol, seed)
            rows.append(with_metric_aliases(simulator.run_episode(episode).as_dict(), config.n_nodes))
            slot_rows.extend(simulator.per_slot_rows)
            for edge_row in simulator.edge_rows:
                edge_rows.append({"episode": episode, **edge_row})
    return rows, slot_rows, edge_rows


def stable_protocol_offset(protocol: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(protocol))


def with_metric_aliases(row: dict, n_nodes: int) -> dict:
    row = dict(row)
    row["mean_discovery_delay"] = row["mean_delay_censored"]
    row["p95_discovery_delay"] = row["p95_delay_censored"]
    row["finite_time_discovery_rate"] = row["discovery_rate"]
    row["lcc_ratio"] = row["largest_component_size"] / max(1, n_nodes)
    row.setdefault("lambda2", 0.0)
    return row


def run_from_config(
    config_path: str | Path,
    output_root: str | Path,
    protocols: list[str] | None = None,
    episodes: int | None = None,
    slots: int | None = None,
    mobility: str | None = None,
) -> dict:
    config_path = Path(config_path)
    config = override_mobility(override_config(load_config(config_path), episodes, slots), mobility)
    selected_protocols = protocols or list(config.baselines)
    rows, slot_rows, edge_rows = run_detailed(config, selected_protocols)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{config.name}_{stamp}"
    write_outputs(config_path, run_dir, rows, config, slot_rows, edge_rows)
    return {"run_dir": str(run_dir), "rows": rows}


def write_outputs(
    config_path: Path,
    output_dir: Path,
    rows: list[dict],
    config: SimulationConfig | None = None,
    slot_rows: list[dict] | None = None,
    edge_rows: list[dict] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    if rows:
        with (output_dir / "per_episode_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    write_seed_manifest(output_dir, rows)
    write_rows_csv(output_dir / "per_slot_metrics.csv", slot_rows or [])
    write_rows_csv(output_dir / "discovered_edges.csv", edge_rows or [])
    write_run_readme(output_dir, rows, config)
    aggregate = aggregate_rows(rows)
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_seed_manifest(output_dir: Path, rows: list[dict]) -> None:
    manifest = [
        {"protocol": row["protocol"], "episode": row["episode"], "seed": row["seed"]}
        for row in rows
    ]
    (output_dir / "seed_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_placeholder_csv(path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        write_placeholder_csv(path, ["empty"])
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_run_readme(output_dir: Path, rows: list[dict], config: SimulationConfig | None) -> None:
    protocols = sorted({str(row["protocol"]) for row in rows})
    mobility = config.mobility.get("model", "unknown") if config else "unknown"
    text = "\n".join(
        [
            "# Simulation Run",
            "",
            f"- Mobility model: `{mobility}`",
            f"- Protocols: {', '.join(protocols)}",
            f"- Episodes: {len(rows)} protocol-episode rows",
            "",
            "This directory is generated by `isac_nd_sim.runner`.",
        ]
    )
    (output_dir / "README.md").write_text(text + "\n", encoding="utf-8")


def aggregate_rows(rows: list[dict]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["protocol"]), []).append(row)
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and key not in {"episode", "seed", "slots"}
        }
    )
    aggregate: dict[str, dict[str, float]] = {}
    for protocol, protocol_rows in grouped.items():
        aggregate[protocol] = {}
        for key in numeric_keys:
            values = [float(row[key]) for row in protocol_rows]
            aggregate[protocol][f"{key}_mean"] = sum(values) / max(1, len(values))
    return aggregate


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = override_mobility(override_config(load_config(config_path), args.episodes, args.slots), args.mobility)
    protocols = args.protocols.split(",") if args.protocols else list(config.baselines)
    rows, slot_rows, edge_rows = run_detailed(config, protocols)
    write_outputs(config_path, Path(args.output), rows, config, slot_rows, edge_rows)
    print(
        json.dumps(
            {
                "output": args.output,
                "episode_rows": len(rows),
                "slot_rows": len(slot_rows),
                "edge_rows": len(edge_rows),
                "protocols": protocols,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
