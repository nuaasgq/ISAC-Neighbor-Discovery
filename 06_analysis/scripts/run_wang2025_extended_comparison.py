from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, pstdev
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_SRC = REPO_ROOT / "05_simulation" / "src"
if str(SIM_SRC) not in sys.path:
    sys.path.insert(0, str(SIM_SRC))

from isac_nd_sim.config import load_config  # noqa: E402
from isac_nd_sim.runner import run_detailed  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "05_simulation" / "configs" / "wang2025_reproduction_smoke.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "06_analysis" / "paper_tables" / "wang2025_extended_comparison_20260708"
DEFAULT_FIGURES = REPO_ROOT / "06_analysis" / "paper_figures" / "wang2025_extended_comparison_20260708"
PROTOCOLS = (
    "uniform_random",
    "wang2025_isac_no_collab",
    "wang2025_comm_tables",
    "wang2025_isac_tables",
    "improved_rl_isac",
)
PROTOCOL_LABELS = {
    "uniform_random": "Uniform Random",
    "wang2025_isac_no_collab": "Wang-like ISAC, no table exchange",
    "wang2025_comm_tables": "Wang-like + neighbor table",
    "wang2025_isac_tables": "Wang-like + sensing table",
    "improved_rl_isac": "Ours: topology-aware ISAC rule",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Wang2025-style PHY-aware FANET ND comparison matrix.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--slots", type=int, default=200)
    parser.add_argument("--node-counts", default="10,20,30,40,50")
    parser.add_argument("--rf-chains", default="1,3,6")
    parser.add_argument("--protocols", default=",".join(PROTOCOLS))
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    output = args.output
    figures = args.figures
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(args.config)
    node_counts = parse_int_list(args.node_counts)
    rf_chains = parse_int_list(args.rf_chains)
    protocols = [part.strip() for part in args.protocols.split(",") if part.strip()]

    episode_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    case_id = 0
    for node_count in node_counts:
        for rf_chain in rf_chains:
            cfg = replace(
                base_cfg,
                seed=base_cfg.seed + 100_003 * case_id,
                n_nodes=node_count,
                rf_chains=rf_chain,
                episodes=args.episodes,
                slots_per_episode=args.slots,
            )
            rows, case_slots, _edge_rows = run_detailed(cfg, protocols)
            for row in rows:
                row.update(
                    {
                        "case_id": case_id,
                        "node_count": node_count,
                        "rf_chains": rf_chain,
                        "beamwidth_az_deg": 360.0 / max(1, cfg.azimuth_cells),
                        "beamwidth_el_deg": 180.0 / max(1, cfg.elevation_cells),
                        "method_label": PROTOCOL_LABELS.get(str(row["protocol"]), str(row["protocol"])),
                    }
                )
                episode_rows.append(row)
            for row in case_slots:
                row.update({"case_id": case_id, "node_count": node_count, "rf_chains": rf_chain})
                slot_rows.append(row)
            completion_rows.extend(completion_from_slot_rows(case_slots, node_count, rf_chain, args.slots))
            case_id += 1

    aggregate_rows = aggregate_episode_rows(episode_rows)
    completion_aggregate_rows = aggregate_completion_rows(completion_rows)
    merged_rows = merge_completion_into_aggregate(aggregate_rows, completion_aggregate_rows)

    write_csv(output / "per_episode_summary.csv", episode_rows)
    write_csv(output / "per_slot_metrics.csv", slot_rows)
    write_csv(output / "completion_slots.csv", completion_rows)
    write_csv(output / "aggregate_metrics.csv", merged_rows)
    write_manifest(output, figures, args, node_counts, rf_chains, protocols, len(episode_rows), len(slot_rows))
    write_readme(output, figures, node_counts, rf_chains, protocols, len(episode_rows), len(slot_rows))

    if not args.skip_figures:
        maybe_write_figures(figures, merged_rows)

    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "figures": figures.as_posix(),
                "episode_rows": len(episode_rows),
                "slot_rows": len(slot_rows),
                "aggregate_rows": len(merged_rows),
                "node_counts": node_counts,
                "rf_chains": rf_chains,
                "protocols": protocols,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def completion_from_slot_rows(slot_rows: list[dict[str, Any]], node_count: int, rf_chain: int, slots: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in slot_rows:
        grouped.setdefault((str(row["protocol"]), int(row["episode"])), []).append(row)

    rows: list[dict[str, Any]] = []
    for (protocol, episode), group_rows in grouped.items():
        ordered = sorted(group_rows, key=lambda item: int(item["slot"]))
        completion_slot = slots
        completed = False
        for row in ordered:
            true_edges_seen = int(row.get("true_edges_seen", 0))
            discovered_edges = int(row.get("discovered_edges", 0))
            if true_edges_seen > 0 and discovered_edges >= true_edges_seen:
                completion_slot = int(row["slot"]) + 1
                completed = True
                break
        rows.append(
            {
                "protocol": protocol,
                "method_label": PROTOCOL_LABELS.get(protocol, protocol),
                "episode": episode,
                "node_count": node_count,
                "rf_chains": rf_chain,
                "completed": int(completed),
                "consumed_slots_censored": completion_slot,
            }
        )
    return rows


def aggregate_episode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_keys = ("protocol", "method_label", "node_count", "rf_chains")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)

    metrics = [
        "discovery_rate",
        "collision_penalized_discovery_rate",
        "mean_delay_censored",
        "p95_delay_censored",
        "empty_scan_ratio",
        "collision_count",
        "scan_actions",
        "discoveries_per_1000_scan_actions",
        "discoveries_per_joule",
        "mean_sensing_pd",
        "mean_sensing_snr_db",
        "sensing_detection_rate",
        "sensing_miss_ratio",
        "sensing_false_alarm_ratio",
        "lambda2",
        "lcc_ratio",
        "isolated_node_ratio",
        "largest_component_size",
    ]
    aggregate: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        out: dict[str, Any] = dict(zip(group_keys, key))
        out["n_episodes"] = len(group_rows)
        for metric in metrics:
            values = [float(row.get(metric, 0.0)) for row in group_rows]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        aggregate.append(out)
    return aggregate


def aggregate_completion_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_keys = ("protocol", "method_label", "node_count", "rf_chains")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    aggregate: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        values = [float(row["consumed_slots_censored"]) for row in group_rows]
        completed = [float(row["completed"]) for row in group_rows]
        aggregate.append(
            {
                **dict(zip(group_keys, key)),
                "consumed_slots_censored_mean": mean(values),
                "consumed_slots_censored_std": pstdev(values) if len(values) > 1 else 0.0,
                "completion_rate_mean": mean(completed),
            }
        )
    return aggregate


def merge_completion_into_aggregate(
    aggregate_rows: list[dict[str, Any]],
    completion_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    key_fields = ("protocol", "method_label", "node_count", "rf_chains")
    completion_by_key = {tuple(row[key] for key in key_fields): row for row in completion_rows}
    merged = []
    for row in aggregate_rows:
        merged_row = dict(row)
        completion = completion_by_key.get(tuple(row[key] for key in key_fields), {})
        for key in ("consumed_slots_censored_mean", "consumed_slots_censored_std", "completion_rate_mean"):
            merged_row[key] = completion.get(key, 0.0)
        merged.append(merged_row)
    return merged


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(
    output: Path,
    figures: Path,
    args: argparse.Namespace,
    node_counts: list[int],
    rf_chains: list[int],
    protocols: list[str],
    episode_rows: int,
    slot_rows: int,
) -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Wang2025-style MIMO-OTFS ISAC FANET neighbor-discovery extended comparison",
        "config": args.config.resolve().relative_to(REPO_ROOT).as_posix(),
        "output": output.resolve().relative_to(REPO_ROOT).as_posix(),
        "figures": figures.resolve().relative_to(REPO_ROOT).as_posix(),
        "episodes_per_case": args.episodes,
        "slots_per_episode": args.slots,
        "node_counts": node_counts,
        "rf_chains": rf_chains,
        "protocols": protocols,
        "episode_rows": episode_rows,
        "slot_rows": slot_rows,
        "paper_boundary": "This campaign uses a PHY-aware abstraction of MIMO-OTFS sensing, not a full waveform receiver.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_readme(
    output: Path,
    figures: Path,
    node_counts: list[int],
    rf_chains: list[int],
    protocols: list[str],
    episode_rows: int,
    slot_rows: int,
) -> None:
    lines = [
        "# Wang2025 Extended Comparison",
        "",
        "This campaign compares Wang2025-like rule mechanisms and the current topology-aware ISAC method in a shared PHY-aware FANET setting.",
        "",
        f"- Node counts: {node_counts}",
        f"- RF chains: {rf_chains}",
        f"- Protocols: {', '.join(protocols)}",
        f"- Episode rows: {episode_rows}",
        f"- Slot rows: {slot_rows}",
        f"- Figures: `{figures.relative_to(REPO_ROOT).as_posix()}`",
        "",
        "Files:",
        "",
        "- `per_episode_summary.csv`",
        "- `per_slot_metrics.csv`",
        "- `completion_slots.csv`",
        "- `aggregate_metrics.csv`",
        "- `manifest.json`",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_write_figures(figures: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )
    colors = {
        "uniform_random": "#6b7280",
        "wang2025_isac_no_collab": "#4f83cc",
        "wang2025_comm_tables": "#2ca25f",
        "wang2025_isac_tables": "#8856a7",
        "improved_rl_isac": "#d95f02",
    }

    plot_line_metric(
        rows,
        figures / "node_scaling_discovery_rate_rf3.png",
        metric="discovery_rate_mean",
        ylabel="Discovery rate",
        rf_filter=3,
        colors=colors,
    )
    plot_line_metric(
        rows,
        figures / "node_scaling_consumed_slots_rf3.png",
        metric="consumed_slots_censored_mean",
        ylabel="Consumed slots (censored)",
        rf_filter=3,
        colors=colors,
    )
    plot_line_metric(
        rows,
        figures / "node_scaling_empty_scan_rf3.png",
        metric="empty_scan_ratio_mean",
        ylabel="Empty scan ratio",
        rf_filter=3,
        colors=colors,
    )
    plot_line_metric(
        rows,
        figures / "node_scaling_lambda2_rf3.png",
        metric="lambda2_mean",
        ylabel="Algebraic connectivity",
        rf_filter=3,
        colors=colors,
    )
    plot_rf_metric(
        rows,
        figures / "rf_sensitivity_discovery_rate_n50.png",
        metric="discovery_rate_mean",
        ylabel="Discovery rate",
        node_filter=50,
        colors=colors,
    )
    plot_rf_metric(
        rows,
        figures / "rf_sensitivity_consumed_slots_n50.png",
        metric="consumed_slots_censored_mean",
        ylabel="Consumed slots (censored)",
        node_filter=50,
        colors=colors,
    )


def plot_line_metric(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    metric: str,
    ylabel: str,
    rf_filter: int,
    colors: dict[str, str],
) -> None:
    import matplotlib.pyplot as plt

    subset = [row for row in rows if int(row["rf_chains"]) == rf_filter]
    protocols = list(PROTOCOLS)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for protocol in protocols:
        points = sorted((row for row in subset if row["protocol"] == protocol), key=lambda item: int(item["node_count"]))
        if not points:
            continue
        ax.plot(
            [int(row["node_count"]) for row in points],
            [float(row[metric]) for row in points],
            marker="o",
            linewidth=1.8,
            label=PROTOCOL_LABELS.get(protocol, protocol),
            color=colors.get(protocol),
        )
    ax.set_xlabel("Number of UAV nodes")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. node count (RF={rf_filter})")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_rf_metric(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    metric: str,
    ylabel: str,
    node_filter: int,
    colors: dict[str, str],
) -> None:
    import matplotlib.pyplot as plt

    subset = [row for row in rows if int(row["node_count"]) == node_filter]
    protocols = list(PROTOCOLS)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for protocol in protocols:
        points = sorted((row for row in subset if row["protocol"] == protocol), key=lambda item: int(item["rf_chains"]))
        if not points:
            continue
        ax.plot(
            [int(row["rf_chains"]) for row in points],
            [float(row[metric]) for row in points],
            marker="o",
            linewidth=1.8,
            label=PROTOCOL_LABELS.get(protocol, protocol),
            color=colors.get(protocol),
        )
    ax.set_xlabel("Number of RF chains")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. RF chains (N={node_filter})")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
