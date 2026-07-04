from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


FIGSIZE = (6.4, 4.8)
DEFAULT_DPI = 300
PAPER_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#F0E442",
)
LINE_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
PROTOCOL_ORDER = (
    "skyorbs_like_skip_scan",
    "uniform_random",
    "rl_no_isac",
    "improved_rl_no_isac",
    "improved_rl_isac",
    "isac_only",
    "topology_only",
    "itap_nd",
    "sensing_assisted",
    "marl",
    "beam_sweep",
    "exhaustive_scan",
    "random",
)
TRAINING_HISTORY_FILENAMES = ("elite_history.csv", "training_history.csv")


@dataclass(frozen=True)
class TableSet:
    label: str
    path: Path
    protocol_rows: list[dict[str, str]]
    slot_rows: list[dict[str, str]]


@dataclass(frozen=True)
class TrainingSource:
    label: str
    path: Path
    rows: list[dict[str, str]]


@dataclass(frozen=True)
class SummaryFigureSpec:
    stem: str
    title: str
    ylabel: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class SlotFigureSpec:
    stem: str
    title: str
    ylabel: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class TrainingFigureSpec:
    stem: str
    title: str
    ylabel: str
    aliases: tuple[str, ...]
    plot_all_matches: bool = False


TEST_SUMMARY_FIGURES = (
    SummaryFigureSpec(
        "discovery_rate",
        "Neighbor discovery rate",
        "Discovery rate",
        ("discovery_rate", "discovery_rate_mean", "finite_time_discovery_rate", "finite_time_discovery_rate_mean"),
    ),
    SummaryFigureSpec(
        "empty_scan_ratio",
        "Empty-scan ratio",
        "Empty-scan ratio",
        ("empty_scan_ratio", "empty_scan_ratio_mean"),
    ),
    SummaryFigureSpec(
        "mean_discovery_delay",
        "Mean discovery delay",
        "Delay (slots)",
        ("mean_discovery_delay", "mean_delay_censored"),
    ),
    SummaryFigureSpec(
        "p90_discovery_delay",
        "P90 discovery delay",
        "Delay (slots)",
        ("p90_discovery_delay", "p90_delay_censored"),
    ),
    SummaryFigureSpec(
        "p95_discovery_delay",
        "P95 discovery delay",
        "Delay (slots)",
        ("p95_discovery_delay", "p95_delay_censored"),
    ),
    SummaryFigureSpec(
        "p99_discovery_delay",
        "P99 discovery delay",
        "Delay (slots)",
        ("p99_discovery_delay", "p99_delay_censored"),
    ),
    SummaryFigureSpec(
        "lcc_ratio",
        "Largest connected component ratio",
        "LCC ratio",
        ("lcc_ratio",),
    ),
    SummaryFigureSpec(
        "lambda2",
        "Algebraic connectivity",
        "lambda2",
        ("lambda2",),
    ),
    SummaryFigureSpec(
        "collision_count",
        "Collision count",
        "Collisions per episode",
        ("collision_count",),
    ),
    SummaryFigureSpec(
        "discovered_edges",
        "Discovered edges",
        "Edges per episode",
        ("discovered_edges",),
    ),
)

TEST_SLOT_FIGURES = (
    SlotFigureSpec(
        "slot_discovered_edges",
        "Discovered-edge trajectory",
        "Discovered edges",
        ("discovered_edges",),
    ),
    SlotFigureSpec(
        "slot_lcc_ratio",
        "Connectivity trajectory",
        "LCC ratio",
        ("lcc_ratio",),
    ),
)

TRAINING_FIGURES = (
    TrainingFigureSpec(
        "score_curve",
        "Training score",
        "Score",
        ("elite_score", "best_score", "score", "mean_score", "score_mean"),
    ),
    TrainingFigureSpec(
        "reward_curve",
        "Training reward",
        "Reward",
        ("mean_reward", "avg_reward", "episode_reward", "reward", "reward_mean"),
    ),
    TrainingFigureSpec(
        "discovery_rate_curve",
        "Training discovery rate",
        "Discovery rate",
        ("discovery_rate", "discovery_rate_mean", "finite_time_discovery_rate", "finite_time_discovery_rate_mean"),
    ),
    TrainingFigureSpec(
        "empty_scan_ratio_curve",
        "Training empty-scan ratio",
        "Empty-scan ratio",
        ("empty_scan_ratio", "empty_scan_ratio_mean"),
    ),
    TrainingFigureSpec(
        "delay_curve",
        "Training discovery delay",
        "Delay (slots)",
        ("mean_discovery_delay", "mean_discovery_delay_mean", "mean_delay_censored", "mean_delay_censored_mean", "p95_discovery_delay", "p95_discovery_delay_mean", "p95_delay_censored", "p95_delay_censored_mean"),
    ),
    TrainingFigureSpec(
        "collision_curve",
        "Training collisions",
        "Collisions",
        ("collision_count", "collision_count_mean", "collisions", "mean_collision_count", "collision_per_slot_mean"),
    ),
    TrainingFigureSpec(
        "connectivity_curve",
        "Training connectivity",
        "Connectivity",
        ("lcc_ratio", "lcc_ratio_mean", "lambda2", "lambda2_mean"),
        plot_all_matches=True,
    ),
    TrainingFigureSpec(
        "loss_curve",
        "Training loss",
        "Loss",
        ("loss", "actor_loss", "critic_loss", "policy_loss", "value_loss"),
        plot_all_matches=True,
    ),
)

X_COLUMN_ALIASES = (
    "generation",
    "episode",
    "epoch",
    "iteration",
    "iter",
    "step",
    "round",
    "slot",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render IEEE-style paper figures from analysis tables and optional training histories.",
    )
    parser.add_argument(
        "table_dirs",
        nargs="+",
        help="One or more analysis table directories containing protocol_summary.csv and slot_protocol_summary.csv.",
    )
    parser.add_argument(
        "--training-dir",
        "--train-dir",
        action="append",
        dest="training_dirs",
        default=[],
        help="Optional training output directory or history CSV. Can be passed more than once.",
    )
    parser.add_argument(
        "--output",
        default="06_analysis/paper_figures",
        help="Output directory for PNG figures, paper_figure_manifest.json, and README.md.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"PNG resolution. The figure canvas remains {FIGSIZE[0]}x{FIGSIZE[1]} inches.",
    )
    return parser.parse_args(argv)


def generate_paper_figures(
    table_dirs: Iterable[str | Path],
    output_dir: str | Path,
    training_dirs: Iterable[str | Path] | None = None,
    dpi: int = DEFAULT_DPI,
) -> dict:
    table_sets = discover_table_sets([Path(item) for item in table_dirs])
    if not table_sets:
        raise FileNotFoundError("No table directories with protocol_summary.csv were found.")

    training_sources = discover_training_sources([Path(item) for item in training_dirs or []])
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt = configure_matplotlib(dpi)
    figures: list[dict] = []
    for table_set in table_sets:
        figures.extend(write_test_figures(table_set, output_path, plt, dpi))
    figures.extend(write_training_figures(training_sources, output_path, plt, dpi))

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "style": {
            "figsize_inches": list(FIGSIZE),
            "aspect_ratio": "4:3",
            "font_family": "Times New Roman with serif fallback",
            "dpi": dpi,
            "palette": list(PAPER_COLORS),
        },
        "capability": {
            "training_figure_specs": len(TRAINING_FIGURES),
            "test_figure_specs_per_table_dir": len(TEST_SUMMARY_FIGURES) + len(TEST_SLOT_FIGURES),
            "maximum_figures_for_this_input": len(table_sets)
            * (len(TEST_SUMMARY_FIGURES) + len(TEST_SLOT_FIGURES))
            + len(TRAINING_FIGURES),
        },
        "inputs": {
            "table_dirs": [
                {"label": item.label, "path": str(item.path.resolve())}
                for item in table_sets
            ],
            "training_files": [
                {"label": item.label, "path": str(item.path.resolve())}
                for item in training_sources
            ],
        },
        "counts": {
            "generated": sum(1 for item in figures if item["status"] == "generated"),
            "skipped": sum(1 for item in figures if item["status"] == "skipped"),
            "total_manifest_entries": len(figures),
        },
        "figures": figures,
    }
    manifest_path = output_path / "paper_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output_path, manifest)
    return manifest


def discover_table_sets(paths: Sequence[Path]) -> list[TableSet]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.resolve()
        candidate_dirs: list[Path] = []
        if path.is_file() and path.name == "protocol_summary.csv":
            candidate_dirs = [path.parent]
        elif (path / "protocol_summary.csv").exists():
            candidate_dirs = [path]
        elif path.exists():
            candidate_dirs = [item.parent for item in path.rglob("protocol_summary.csv")]
        for candidate in candidate_dirs:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(resolved)

    table_sets: list[TableSet] = []
    used_labels: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: str(item)):
        protocol_path = candidate / "protocol_summary.csv"
        slot_path = candidate / "slot_protocol_summary.csv"
        if not protocol_path.exists():
            continue
        label = unique_label(infer_table_label(candidate), used_labels)
        table_sets.append(
            TableSet(
                label=label,
                path=candidate,
                protocol_rows=read_csv(protocol_path),
                slot_rows=read_csv(slot_path),
            )
        )
    return table_sets


def discover_training_sources(paths: Sequence[Path]) -> list[TrainingSource]:
    files: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.resolve()
        candidates: list[Path] = []
        if path.is_file() and path.name in TRAINING_HISTORY_FILENAMES:
            candidates = [path]
        elif path.is_dir():
            direct = [path / name for name in TRAINING_HISTORY_FILENAMES if (path / name).exists()]
            nested = [item for item in path.rglob("*.csv") if item.name in TRAINING_HISTORY_FILENAMES]
            candidates = direct + nested
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(resolved)

    used_labels: set[str] = set()
    sources: list[TrainingSource] = []
    for file_path in sorted(files, key=lambda item: str(item)):
        source_name = normalize_label(file_path.parent.name)
        if len(files) > 1 and file_path.stem not in {"training_history", "elite_history"}:
            source_name = normalize_label(f"{source_name}_{file_path.stem}")
        label = unique_label(source_name or "training", used_labels)
        sources.append(TrainingSource(label=label, path=file_path, rows=read_csv(file_path)))
    return sources


def write_test_figures(table_set: TableSet, output_dir: Path, plt, dpi: int) -> list[dict]:
    figures: list[dict] = []
    for spec in TEST_SUMMARY_FIGURES:
        filename = f"test_{table_set.label}_{spec.stem}.png"
        path = output_dir / filename
        rows = rows_with_metric(table_set.protocol_rows, spec.aliases)
        if not rows:
            figures.append(
                skipped_figure(
                    filename,
                    "test",
                    spec.stem,
                    f"missing metric aliases: {', '.join(spec.aliases)}",
                    table_set.label,
                )
            )
            continue
        figures.append(plot_summary_bar(path, rows, spec, table_set, plt, dpi))

    for spec in TEST_SLOT_FIGURES:
        filename = f"test_{table_set.label}_{spec.stem}.png"
        path = output_dir / filename
        series = collect_slot_series(table_set.slot_rows, spec.aliases)
        if not series:
            figures.append(
                skipped_figure(
                    filename,
                    "test",
                    spec.stem,
                    f"missing slot metric aliases: {', '.join(spec.aliases)}",
                    table_set.label,
                )
            )
            continue
        figures.append(plot_slot_lines(path, series, spec, table_set, plt, dpi))
    return figures


def write_training_figures(
    training_sources: Sequence[TrainingSource],
    output_dir: Path,
    plt,
    dpi: int,
) -> list[dict]:
    figures: list[dict] = []
    for spec in TRAINING_FIGURES:
        filename = f"train_{spec.stem}.png"
        path = output_dir / filename
        series = collect_training_series(training_sources, spec)
        if not series:
            reason = "no training history files" if not training_sources else f"missing metric aliases: {', '.join(spec.aliases)}"
            figures.append(skipped_figure(filename, "training", spec.stem, reason))
            continue
        figures.append(plot_training_lines(path, series, spec, plt, dpi))
    return figures


def plot_summary_bar(
    path: Path,
    rows: list[dict[str, str]],
    spec: SummaryFigureSpec,
    table_set: TableSet,
    plt,
    dpi: int,
) -> dict:
    ordered_rows = sorted(rows, key=lambda row: protocol_sort_key(row.get("protocol", "")))
    protocols = [row.get("protocol", "unknown") or "unknown" for row in ordered_rows]
    values = [first_metric_value(row, spec.aliases, "mean") for row in ordered_rows]
    ci_values = [first_metric_value(row, spec.aliases, "ci95") for row in ordered_rows]
    values_float = [float(value) for value in values if value is not None]
    yerr = [0.0 if value is None else float(value) for value in ci_values]
    colors = [color_for_index(index) for index, _ in enumerate(protocols)]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x_positions = list(range(len(protocols)))
    ax.bar(
        x_positions,
        values_float,
        yerr=yerr if any(value > 0 for value in yerr) else None,
        width=0.68,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
    )
    ax.set_title(spec.title)
    ax.set_ylabel(spec.ylabel)
    ax.set_xlabel("Protocol")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(protocols, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.28)
    ax.set_axisbelow(True)
    apply_y_margin(ax, values_float)
    fig.tight_layout(pad=0.8)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return generated_figure(path, "test", spec.stem, len(rows), table_set.label, dpi)


def plot_slot_lines(
    path: Path,
    series: dict[str, list[tuple[float, float]]],
    spec: SlotFigureSpec,
    table_set: TableSet,
    plt,
    dpi: int,
) -> dict:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for index, protocol in enumerate(sorted(series, key=protocol_sort_key)):
        points = sorted(series[protocol], key=lambda item: item[0])
        xs = [item[0] for item in points]
        ys = [item[1] for item in points]
        ax.plot(
            xs,
            ys,
            label=protocol,
            color=color_for_index(index),
            marker=LINE_MARKERS[index % len(LINE_MARKERS)],
            markevery=max(1, len(xs) // 8),
        )
    ax.set_title(spec.title)
    ax.set_xlabel("Slot")
    ax.set_ylabel(spec.ylabel)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.8)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    rows_used = sum(len(points) for points in series.values())
    return generated_figure(path, "test", spec.stem, rows_used, table_set.label, dpi)


def plot_training_lines(path: Path, series: list[dict], spec: TrainingFigureSpec, plt, dpi: int) -> dict:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for index, item in enumerate(series):
        ax.plot(
            item["x"],
            item["y"],
            label=item["label"],
            color=color_for_index(index),
            marker=LINE_MARKERS[index % len(LINE_MARKERS)],
            markevery=max(1, len(item["x"]) // 10),
        )
    ax.set_title(spec.title)
    ax.set_xlabel(series[0]["x_label"])
    ax.set_ylabel(spec.ylabel)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.8)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return generated_figure(path, "training", spec.stem, sum(len(item["x"]) for item in series), dpi=dpi)


def collect_slot_series(
    rows: Sequence[dict[str, str]],
    aliases: Sequence[str],
) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        slot = to_float(row.get("slot"))
        value = first_metric_value(row, aliases, "mean")
        if slot is None or value is None:
            continue
        protocol = row.get("protocol", "unknown") or "unknown"
        series.setdefault(protocol, []).append((slot, value))
    return series


def collect_training_series(
    sources: Sequence[TrainingSource],
    spec: TrainingFigureSpec,
) -> list[dict]:
    output: list[dict] = []
    for source in sources:
        x_label, x_values = collect_x_values(source.rows)
        matched_columns = columns_for_aliases(source.rows, spec.aliases)
        if not spec.plot_all_matches and matched_columns:
            matched_columns = matched_columns[:1]
        for column in matched_columns:
            points: list[tuple[float, float]] = []
            for index, row in enumerate(source.rows):
                value = to_float(row.get(column))
                if value is None:
                    continue
                x_value = x_values[index] if index < len(x_values) else float(index + 1)
                points.append((x_value, value))
            if not points:
                continue
            points.sort(key=lambda item: item[0])
            label = source.label if len(matched_columns) == 1 else f"{source.label} {column}"
            output.append(
                {
                    "label": label,
                    "x_label": x_label,
                    "x": [item[0] for item in points],
                    "y": [item[1] for item in points],
                }
            )
    return output


def rows_with_metric(
    rows: Sequence[dict[str, str]],
    aliases: Sequence[str],
) -> list[dict[str, str]]:
    return [row for row in rows if row.get("protocol") and first_metric_value(row, aliases, "mean") is not None]


def first_metric_value(row: dict[str, str], aliases: Sequence[str], statistic: str) -> float | None:
    suffixes = [f"_{statistic}"]
    if statistic == "mean":
        suffixes.append("")
    for alias in aliases:
        for suffix in suffixes:
            value = to_float(row.get(f"{alias}{suffix}"))
            if value is not None:
                return value
    return None


def collect_x_values(rows: Sequence[dict[str, str]]) -> tuple[str, list[float]]:
    for column in X_COLUMN_ALIASES:
        values = [to_float(row.get(column)) for row in rows]
        if any(value is not None for value in values):
            return column.capitalize(), [
                float(value) if value is not None else float(index + 1)
                for index, value in enumerate(values)
            ]
    return "Iteration", [float(index + 1) for index, _row in enumerate(rows)]


def columns_for_aliases(rows: Sequence[dict[str, str]], aliases: Sequence[str]) -> list[str]:
    if not rows:
        return []
    columns: list[str] = []
    for alias in aliases:
        if alias in rows[0] and any(to_float(row.get(alias)) is not None for row in rows):
            columns.append(alias)
    return columns


def configure_matplotlib(dpi: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.2,
            "grid.linewidth": 0.5,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )
    return plt


def apply_y_margin(ax, values: Sequence[float]) -> None:
    if not values:
        return
    min_value = min(values)
    max_value = max(values)
    if math.isclose(min_value, max_value):
        margin = abs(max_value) * 0.1 if max_value else 0.1
    else:
        margin = (max_value - min_value) * 0.12
    lower = min(0.0, min_value - margin) if min_value >= 0 else min_value - margin
    upper = max_value + margin
    ax.set_ylim(lower, upper)


def generated_figure(
    path: Path,
    category: str,
    metric: str,
    row_count: int,
    source_label: str | None = None,
    dpi: int = DEFAULT_DPI,
) -> dict:
    item = {
        "status": "generated",
        "category": category,
        "metric": metric,
        "filename": path.name,
        "path": str(path.resolve()),
        "figsize_inches": list(FIGSIZE),
        "pixel_size": [round(FIGSIZE[0] * dpi), round(FIGSIZE[1] * dpi)],
        "dpi": dpi,
        "data_rows": row_count,
    }
    if source_label is not None:
        item["source_label"] = source_label
    return item


def skipped_figure(
    filename: str,
    category: str,
    metric: str,
    reason: str,
    source_label: str | None = None,
) -> dict:
    item = {
        "status": "skipped",
        "category": category,
        "metric": metric,
        "filename": filename,
        "reason": reason,
        "figsize_inches": list(FIGSIZE),
    }
    if source_label is not None:
        item["source_label"] = source_label
    return item


def write_readme(output_dir: Path, manifest: dict) -> None:
    generated = [item for item in manifest["figures"] if item["status"] == "generated"]
    skipped = [item for item in manifest["figures"] if item["status"] == "skipped"]
    lines = [
        "# Paper Figure Outputs",
        "",
        f"Generated at: {manifest['generated_at_utc']}",
        "",
        "## Style",
        "",
        f"- Canvas: {FIGSIZE[0]} x {FIGSIZE[1]} inches (4:3)",
        f"- DPI: {manifest['style']['dpi']}",
        "- Font: Times New Roman, with serif fallback if unavailable",
        "- Palette: colorblind-friendly IEEE-style line/bar series",
        "",
        "## Inputs",
        "",
    ]
    for item in manifest["inputs"]["table_dirs"]:
        lines.append(f"- Analysis tables `{item['label']}`: `{item['path']}`")
    if manifest["inputs"]["training_files"]:
        for item in manifest["inputs"]["training_files"]:
            lines.append(f"- Training history `{item['label']}`: `{item['path']}`")
    else:
        lines.append("- Training history: not provided")

    lines.extend(
        [
            "",
            "## Generated Figures",
            "",
        ]
    )
    if generated:
        for item in generated:
            lines.append(f"- `{item['filename']}` ({item['category']}, {item['metric']})")
    else:
        lines.append("- None")

    lines.extend(["", "## Skipped Figure Specs", ""])
    if skipped:
        for item in skipped:
            lines.append(f"- `{item['filename']}`: {item['reason']}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Manifest",
            "",
            "Full machine-readable metadata is stored in `paper_figure_manifest.json`.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(output):
        return None
    return output


def infer_table_label(path: Path) -> str:
    raw = path.name
    if raw.lower() in {"table", "tables", "analysis_tables"}:
        raw = path.parent.name
    raw = re.sub(r"[_-]?tables?$", "", raw, flags=re.IGNORECASE)
    return normalize_label(raw) or "analysis"


def normalize_label(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "run"


def unique_label(label: str, used: set[str]) -> str:
    base = label
    index = 2
    while label in used:
        label = f"{base}_{index}"
        index += 1
    used.add(label)
    return label


def color_for_index(index: int) -> str:
    return PAPER_COLORS[index % len(PAPER_COLORS)]


def protocol_sort_key(protocol: str) -> tuple[int, str]:
    normalized = protocol.lower()
    for index, token in enumerate(PROTOCOL_ORDER):
        if token in normalized:
            return index, normalized
    return len(PROTOCOL_ORDER), normalized


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = generate_paper_figures(
        table_dirs=args.table_dirs,
        output_dir=args.output,
        training_dirs=args.training_dirs,
        dpi=args.dpi,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
