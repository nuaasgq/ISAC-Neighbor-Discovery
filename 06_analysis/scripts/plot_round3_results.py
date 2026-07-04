from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


FIGSIZE = (6.4, 4.8)
DPI = 300
PROPOSED = "improved_rl_isac"
NO_ISAC = "improved_rl_no_isac"
COLORS = {
    "uniform_random": "#D55E00",
    "improved_rl_no_isac": "#009E73",
    "improved_rl_isac": "#E69F00",
    "ablation_isac_no_candidate_set": "#56B4E9",
    "ablation_isac_no_beam_lock": "#CC79A7",
    "ablation_isac_no_topology": "#0072B2",
}
PROTOCOL_ORDER = (
    "uniform_random",
    "improved_rl_no_isac",
    "ablation_isac_no_topology",
    "ablation_isac_no_beam_lock",
    "ablation_isac_no_candidate_set",
    "improved_rl_isac",
)
METRICS = {
    "discovery": ("discovery_rate_mean", "Discovery rate", "higher"),
    "empty_scan": ("empty_scan_ratio_mean", "Empty-scan ratio", "lower"),
    "lambda2": ("lambda2_mean", "Algebraic connectivity", "higher"),
    "mean_delay": ("mean_discovery_delay_mean", "Mean delay (slots)", "lower"),
    "collision": ("collision_count_mean", "Collisions per episode", "lower"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot round3 robustness, range, and ablation figures.")
    parser.add_argument("sweep_dirs", nargs="+", help="Directories or aggregate_metrics.csv files.")
    parser.add_argument("--output", default="06_analysis/paper_figures/round3_robustness")
    parser.add_argument("--node-count", type=int, default=100)
    parser.add_argument("--beamwidth-deg", type=float, default=10.0)
    return parser.parse_args()


def load_frame(paths: Iterable[str | Path]):
    import pandas as pd
    from pandas.errors import PerformanceWarning
    import warnings

    warnings.filterwarnings("ignore", category=PerformanceWarning)

    frames = []
    for item in paths:
        path = Path(item)
        csv_path = path / "aggregate_metrics.csv" if path.is_dir() else path
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        frame = frame.assign(source=path.name if path.is_dir() else path.parent.name)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No aggregate_metrics.csv files were found.")
    df = pd.concat(frames, ignore_index=True)
    for column in df.columns:
        if column in {"protocol", "mobility_model", "area_scale", "range_mode", "source"}:
            continue
        try:
            df[column] = pd.to_numeric(df[column], errors="raise")
        except (TypeError, ValueError):
            pass
    for column in (
        "communication_range_to_diagonal_ratio",
        "sensing_to_comm_range_ratio",
        "false_alarm_rate",
        "miss_detection_rate",
        "angular_cell_offset_std",
        "beamwidth_deg",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").round(6)
    return df


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 1.7,
            "lines.markersize": 4.5,
        }
    )
    return plt


def generate_round3_figures(
    sweep_dirs: Iterable[str | Path],
    output_dir: str | Path,
    node_count: int = 100,
    beamwidth_deg: float = 10.0,
) -> dict:
    df = load_frame(sweep_dirs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()
    manifest: list[dict] = []

    for stem in ("discovery", "empty_scan", "lambda2"):
        manifest.append(
            save_range_gain_heatmap(
                df,
                stem,
                output / f"range_gain_{stem}_n{node_count}_b{beam_token(beamwidth_deg)}.png",
                plt,
                node_count,
                beamwidth_deg,
            )
        )
    for stem in ("discovery", "empty_scan"):
        manifest.append(
            save_error_gain_heatmap(
                df,
                stem,
                output / f"error_gain_{stem}_n{node_count}_b{beam_token(beamwidth_deg)}.png",
                plt,
                node_count,
                beamwidth_deg,
            )
        )
    for stem in ("discovery", "empty_scan", "lambda2", "collision"):
        manifest.append(
            save_ablation_bar(
                df,
                stem,
                output / f"ablation_{stem}_n{node_count}_b{beam_token(beamwidth_deg)}.png",
                plt,
                node_count,
                beamwidth_deg,
            )
        )
    manifest.append(
        save_range_protocol_curves(
            df,
            "discovery",
            output / f"range_protocol_discovery_n{node_count}_b{beam_token(beamwidth_deg)}.png",
            plt,
            node_count,
            beamwidth_deg,
        )
    )
    manifest.append(
        save_error_profile_curves(
            df,
            "discovery",
            output / f"error_profile_discovery_n{node_count}_b{beam_token(beamwidth_deg)}.png",
            plt,
            node_count,
            beamwidth_deg,
        )
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "style": {
            "figsize_inches": list(FIGSIZE),
            "aspect_ratio": "4:3",
            "font_family": "Times New Roman with serif fallback",
            "dpi": DPI,
            "palette": COLORS,
        },
        "inputs": [str(Path(path)) for path in sweep_dirs],
        "selection": {"node_count": node_count, "beamwidth_deg": beamwidth_deg},
        "counts": {
            "generated": sum(1 for item in manifest if item["status"] == "generated"),
            "skipped": sum(1 for item in manifest if item["status"] == "skipped"),
            "total": len(manifest),
        },
        "figures": manifest,
    }
    (output / "round3_figure_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output, payload)
    return payload


def save_range_gain_heatmap(df, metric_key: str, path: Path, plt, node_count: int, beamwidth_deg: float) -> dict:
    import numpy as np

    if not has_columns(df, ["communication_range_to_diagonal_ratio", "sensing_to_comm_range_ratio"]):
        return skipped(path, metric_key, "missing range-ratio columns")
    metric_col, ylabel, direction = METRICS[metric_key]
    if metric_col not in df:
        return skipped(path, metric_key, f"missing metric {metric_col}")
    subset = range_sweep_subset(select_common(df, node_count, beamwidth_deg))
    subset = zero_error_subset(subset)
    merged = proposed_vs_baseline(subset, metric_col)
    if merged.empty:
        return skipped(path, metric_key, "missing proposed or no-ISAC baseline rows")
    merged["gain"] = improvement(merged[f"{metric_col}_proposed"], merged[f"{metric_col}_baseline"], direction)
    pivot = merged.pivot_table(
        index="communication_range_to_diagonal_ratio",
        columns="sensing_to_comm_range_ratio",
        values="gain",
        aggfunc="mean",
    )
    if pivot.empty or pivot.shape[0] < 2:
        return skipped(path, metric_key, "insufficient Rc/D variation")
    pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    image = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="coolwarm")
    ax.set_title(f"ISAC gain: {ylabel}")
    ax.set_xlabel("Rs/Rc")
    ax.set_ylabel("Rc/D")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{value:g}" for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{value:g}" for value in pivot.index])
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(gain_label(direction, ylabel))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return generated(path, metric_key, int(merged.shape[0]))


def save_error_gain_heatmap(df, metric_key: str, path: Path, plt, node_count: int, beamwidth_deg: float) -> dict:
    import numpy as np

    if not has_columns(df, ["false_alarm_rate", "miss_detection_rate"]):
        return skipped(path, metric_key, "missing ISAC error columns")
    metric_col, ylabel, direction = METRICS[metric_key]
    if metric_col not in df:
        return skipped(path, metric_key, f"missing metric {metric_col}")
    subset = source_contains_subset(select_common(df, node_count, beamwidth_deg), "error_robustness")
    subset = error_sweep_subset(subset)
    if "angular_cell_offset_std" in subset:
        subset = subset[subset["angular_cell_offset_std"] == subset["angular_cell_offset_std"].min()]
    merged = proposed_vs_baseline(subset, metric_col)
    if merged.empty:
        return skipped(path, metric_key, "missing proposed or no-ISAC baseline rows")
    if merged["false_alarm_rate"].nunique() < 2 or merged["miss_detection_rate"].nunique() < 2:
        return skipped(path, metric_key, "insufficient Pfa/Pmd variation")
    merged["gain"] = improvement(merged[f"{metric_col}_proposed"], merged[f"{metric_col}_baseline"], direction)
    pivot = merged.pivot_table(index="miss_detection_rate", columns="false_alarm_rate", values="gain", aggfunc="mean")
    pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    image = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="coolwarm")
    ax.set_title(f"Sensing-error gain: {ylabel}")
    ax.set_xlabel("False-alarm probability")
    ax.set_ylabel("Miss-detection probability")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{value:g}" for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{value:g}" for value in pivot.index])
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(gain_label(direction, ylabel))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return generated(path, metric_key, int(merged.shape[0]))


def save_ablation_bar(df, metric_key: str, path: Path, plt, node_count: int, beamwidth_deg: float) -> dict:
    metric_col, ylabel, _direction = METRICS[metric_key]
    if metric_col not in df:
        return skipped(path, metric_key, f"missing metric {metric_col}")
    protocols = set(df["protocol"].astype(str))
    if not any(protocol.startswith("ablation_") for protocol in protocols):
        return skipped(path, metric_key, "missing ablation protocols")
    subset = source_contains_subset(select_common(df, node_count, beamwidth_deg), "ablation")
    subset = zero_error_subset(subset)
    subset = subset[subset["protocol"].isin(PROTOCOL_ORDER)]
    if subset.empty:
        return skipped(path, metric_key, "no rows after N/beam/error filtering")
    grouped = subset.groupby("protocol", as_index=False)[metric_col].mean()
    grouped["sort_key"] = grouped["protocol"].map(lambda value: protocol_sort_key(str(value)))
    grouped = grouped.sort_values(["sort_key", "protocol"])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(
        range(len(grouped)),
        grouped[metric_col],
        color=[COLORS.get(protocol, "#999999") for protocol in grouped["protocol"]],
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_title(f"Mechanism ablation: {ylabel}")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Protocol")
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels([label_protocol(str(item)) for item in grouped["protocol"]], rotation=22, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return generated(path, metric_key, int(grouped.shape[0]))


def save_range_protocol_curves(df, metric_key: str, path: Path, plt, node_count: int, beamwidth_deg: float) -> dict:
    if not has_columns(df, ["communication_range_to_diagonal_ratio", "sensing_to_comm_range_ratio"]):
        return skipped(path, metric_key, "missing range-ratio columns")
    metric_col, ylabel, _direction = METRICS[metric_key]
    if metric_col not in df:
        return skipped(path, metric_key, f"missing metric {metric_col}")
    subset = range_sweep_subset(select_common(df, node_count, beamwidth_deg))
    subset = zero_error_subset(subset)
    subset = subset[subset["protocol"].isin(["uniform_random", NO_ISAC, PROPOSED])]
    if subset.empty or subset["sensing_to_comm_range_ratio"].nunique() < 2:
        return skipped(path, metric_key, "insufficient Rs/Rc variation")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    rows_used = 0
    for protocol in sorted(subset["protocol"].unique(), key=protocol_sort_key):
        protocol_rows = subset[subset["protocol"] == protocol]
        for rc_ratio, rows in sorted(protocol_rows.groupby("communication_range_to_diagonal_ratio")):
            series = rows.groupby("sensing_to_comm_range_ratio", as_index=False)[metric_col].mean().sort_values(
                "sensing_to_comm_range_ratio"
            )
            if series.shape[0] < 2:
                continue
            rows_used += int(series.shape[0])
            ax.plot(
                series["sensing_to_comm_range_ratio"],
                series[metric_col],
                marker="o",
                color=COLORS.get(str(protocol)),
                linestyle="-" if str(protocol) == PROPOSED else "--",
                label=f"{label_protocol(str(protocol))}, Rc/D={rc_ratio:g}",
            )
    if rows_used == 0:
        plt.close(fig)
        return skipped(path, metric_key, "no plottable protocol curve")
    ax.set_title(f"Range sensitivity: {ylabel}")
    ax.set_xlabel("Rs/Rc")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return generated(path, metric_key, rows_used)


def save_error_profile_curves(df, metric_key: str, path: Path, plt, node_count: int, beamwidth_deg: float) -> dict:
    if not has_columns(df, ["false_alarm_rate", "miss_detection_rate", "angular_cell_offset_std"]):
        return skipped(path, metric_key, "missing ISAC error columns")
    metric_col, ylabel, _direction = METRICS[metric_key]
    if metric_col not in df:
        return skipped(path, metric_key, f"missing metric {metric_col}")
    subset = source_contains_subset(select_common(df, node_count, beamwidth_deg), "error_profiles")
    subset = error_sweep_subset(subset)
    subset = subset[subset["protocol"].isin([NO_ISAC, PROPOSED])]
    if subset.empty:
        return skipped(path, metric_key, "missing proposed/no-ISAC rows")
    subset = subset.copy()
    subset["profile_score"] = (
        subset["false_alarm_rate"].astype(float)
        + subset["miss_detection_rate"].astype(float)
        + subset["angular_cell_offset_std"].astype(float) / 10.0
    )
    if subset["profile_score"].nunique() < 2:
        return skipped(path, metric_key, "insufficient error-profile variation")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    rows_used = 0
    for protocol in sorted(subset["protocol"].unique(), key=protocol_sort_key):
        series = subset[subset["protocol"] == protocol].groupby("profile_score", as_index=False)[metric_col].mean()
        series = series.sort_values("profile_score")
        rows_used += int(series.shape[0])
        ax.plot(
            series["profile_score"],
            series[metric_col],
            marker="o",
            color=COLORS.get(str(protocol)),
            label=label_protocol(str(protocol)),
        )
    ax.set_title(f"ISAC error robustness: {ylabel}")
    ax.set_xlabel("Error-profile severity index")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return generated(path, metric_key, rows_used)


def proposed_vs_baseline(df, metric_col: str):
    base_keys = [
        "source",
        "mobility_model",
        "area_scale",
        "range_mode",
        "node_count",
        "beamwidth_deg",
        "communication_range_to_diagonal_ratio",
        "sensing_to_comm_range_ratio",
        "false_alarm_rate",
        "miss_detection_rate",
        "angular_cell_offset_std",
    ]
    keys = [key for key in base_keys if key in df.columns]
    proposed = df[df["protocol"] == PROPOSED][keys + [metric_col]]
    baseline = df[df["protocol"] == NO_ISAC][keys + [metric_col]]
    return proposed.merge(baseline, on=keys, suffixes=("_proposed", "_baseline"))


def select_common(df, node_count: int, beamwidth_deg: float):
    subset = df.copy()
    if "node_count" in subset:
        subset = subset[subset["node_count"] == node_count]
    if "beamwidth_deg" in subset:
        subset = subset[subset["beamwidth_deg"] == beamwidth_deg]
    return subset


def zero_error_subset(df):
    subset = df.copy()
    for column in ("false_alarm_rate", "miss_detection_rate", "angular_cell_offset_std"):
        if column in subset and subset[column].notna().any() and 0.0 in set(subset[column].astype(float)):
            subset = subset[subset[column].astype(float) == 0.0]
    return subset


def error_sweep_subset(df):
    subset = df.copy()
    if "source" in subset:
        error_sources = subset["source"].astype(str).str.contains("error", case=False, na=False)
        if error_sources.any():
            return subset[error_sources]
    nonzero_columns = [
        column
        for column in ("false_alarm_rate", "miss_detection_rate", "angular_cell_offset_std")
        if column in subset.columns
    ]
    if nonzero_columns:
        nonzero_mask = False
        for column in nonzero_columns:
            nonzero_mask = nonzero_mask | (subset[column].fillna(0).astype(float) > 0.0)
        if nonzero_mask.any():
            sources = set(subset.loc[nonzero_mask, "source"]) if "source" in subset else set()
            if sources and "source" in subset:
                return subset[subset["source"].isin(sources)]
    return subset


def range_sweep_subset(df):
    subset = source_contains_subset(df, "range")
    if not subset.empty:
        return subset
    return df


def source_contains_subset(df, token: str):
    if "source" not in df:
        return df
    mask = df["source"].astype(str).str.contains(token, case=False, na=False)
    if mask.any():
        return df[mask].copy()
    return df


def improvement(proposed, baseline, direction: str):
    if direction == "lower":
        return baseline - proposed
    return proposed - baseline


def gain_label(direction: str, ylabel: str) -> str:
    if direction == "lower":
        return f"Reduction vs {label_protocol(NO_ISAC)} ({ylabel})"
    return f"Gain vs {label_protocol(NO_ISAC)} ({ylabel})"


def has_columns(df, columns: list[str]) -> bool:
    return all(column in df.columns for column in columns)


def protocol_sort_key(protocol: str) -> tuple[int, str]:
    try:
        return PROTOCOL_ORDER.index(protocol), protocol
    except ValueError:
        return len(PROTOCOL_ORDER), protocol


def label_protocol(protocol: str) -> str:
    labels = {
        "uniform_random": "Random",
        "improved_rl_no_isac": "Improved-RL",
        "improved_rl_isac": "Improved-RL+ISAC",
        "ablation_isac_no_candidate_set": "No candidate set",
        "ablation_isac_no_beam_lock": "No beam lock",
        "ablation_isac_no_topology": "No topology",
    }
    return labels.get(protocol, protocol)


def beam_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def generated(path: Path, metric: str, rows_used: int) -> dict:
    return {
        "status": "generated",
        "metric": metric,
        "filename": path.name,
        "path": str(path.resolve()),
        "rows_used": rows_used,
        "figsize_inches": list(FIGSIZE),
        "pixel_size": [int(FIGSIZE[0] * DPI), int(FIGSIZE[1] * DPI)],
    }


def skipped(path: Path, metric: str, reason: str) -> dict:
    return {
        "status": "skipped",
        "metric": metric,
        "filename": path.name,
        "path": str(path.resolve()),
        "reason": reason,
        "figsize_inches": list(FIGSIZE),
    }


def write_readme(output: Path, payload: dict) -> None:
    lines = [
        "# Round3 Robustness Figures",
        "",
        f"Generated at: {payload['generated_at_utc']}",
        f"Generated figures: {payload['counts']['generated']}",
        "",
        "## Generated",
        "",
    ]
    generated_items = [item for item in payload["figures"] if item["status"] == "generated"]
    if generated_items:
        for item in generated_items:
            lines.append(f"- `{item['filename']}`")
    else:
        lines.append("- None")
    lines.extend(["", "## Skipped", ""])
    skipped_items = [item for item in payload["figures"] if item["status"] == "skipped"]
    if skipped_items:
        for item in skipped_items:
            lines.append(f"- `{item['filename']}`: {item['reason']}")
    else:
        lines.append("- None")
    lines.append("")
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = generate_round3_figures(args.sweep_dirs, args.output, args.node_count, args.beamwidth_deg)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
