from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "05_simulation" / "results_raw" / "overnight_20260709"
TABLE_DIR = ROOT / "06_analysis" / "paper_tables" / "marl" / "overnight_20260709_marl_isac_rebuild"
FIG_DIR = ROOT / "06_analysis" / "paper_figures" / "marl_overnight_20260709"
REPORT = ROOT / "06_analysis" / "overnight_marl_isac_rebuild_20260709.md"

PALETTE = {
    "Uniform random": "#8C8C8C",
    "Wang ISAC": "#4C78A8",
    "Wang ISAC+tables": "#72B7B2",
    "Rule ISAC": "#F58518",
    "Collision-aware ISAC": "#54A24B",
    "MARL": "#B279A2",
    "MARL+gate": "#9D755D",
    "BC-MARL": "#E45756",
    "BC-MARL+gate": "#FF9DA6",
    "MARL+tables": "#B07AA1",
    "MARL+tables+gate": "#D4A6C8",
}

METHOD_LABELS = {
    "baseline_uniform_random": "Uniform random",
    "baseline_wang2025_isac_no_collab": "Wang ISAC",
    "baseline_wang2025_isac_tables": "Wang ISAC+tables",
    "baseline_improved_rl_isac": "Rule ISAC",
    "baseline_collision_aware_isac": "Collision-aware ISAC",
    "notable_nogate": "MARL",
    "notable_gate": "MARL+gate",
    "bc_notable_nogate": "BC-MARL",
    "bc_notable_gate": "BC-MARL+gate",
    "table_nogate": "MARL+tables",
    "table_gate": "MARL+tables+gate",
}

METRICS = [
    "discovery_rate",
    "collision_count",
    "collision_penalized_discovery_rate",
    "lambda2",
    "empty_scan_ratio",
    "scan_actions_per_discovery_censored",
    "energy_per_discovery_censored_j",
    "access_gate_backoff_ratio",
    "access_gate_normal_ratio",
    "access_gate_aggressive_ratio",
]


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    training_rows = collect_training_rows()
    transfer_rows = collect_transfer_rows()
    write_csv(TABLE_DIR / "training_episode_rows.csv", training_rows)
    write_csv(TABLE_DIR / "transfer_episode_rows.csv", transfer_rows)
    training_summary = summarize_training(training_rows)
    transfer_summary = summarize_transfer(transfer_rows)
    write_csv(TABLE_DIR / "training_summary.csv", training_summary)
    write_csv(TABLE_DIR / "transfer_summary.csv", transfer_summary)
    plot_all(training_rows, transfer_summary)
    write_manifest(training_rows, transfer_rows)
    write_report(training_summary, transfer_summary)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.figsize": (8.0, 6.0),
            "figure.dpi": 220,
            "savefig.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def collect_training_rows() -> list[dict[str, str]]:
    runs = {
        "train_notable_nogate_ep60": "MARL",
        "train_notable_gate_ep60": "MARL+gate",
        "train_table_nogate_ep60": "MARL+tables",
        "train_table_gate_safe_ep80": "MARL+tables+gate",
        "train_bc_notable_nogate_ep120": "BC-MARL",
        "train_bc_notable_gate_ep120": "BC-MARL+gate",
    }
    rows: list[dict[str, str]] = []
    for run_name, label in runs.items():
        path = RAW / run_name / "episode_metrics.csv"
        if not path.exists():
            continue
        for row in read_csv(path):
            row = dict(row)
            row["run"] = run_name
            row["method"] = label
            rows.append(row)
    return rows


def collect_transfer_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    transfer_dir = RAW / "transfer_eval"
    if transfer_dir.exists():
        for run_dir in sorted(p for p in transfer_dir.iterdir() if p.is_dir()):
            metric_file = run_dir / "eval_episode_metrics.csv"
            if not metric_file.exists():
                continue
            name = run_dir.name
            match_n = re.search(r"N(\d+)", name)
            match_b = re.search(r"B(\d+)", name)
            if not match_n or not match_b:
                continue
            method_key = method_key_from_name(name)
            for row in read_csv(metric_file):
                if row.get("phase", "eval_stochastic") != "eval_stochastic":
                    continue
                row = dict(row)
                row["run"] = name
                row["method"] = METHOD_LABELS.get(method_key, method_key)
                row["method_key"] = method_key
                row["node_count"] = match_n.group(1)
                row["beamwidth_deg"] = match_b.group(1)
                row["source_type"] = "marl"
                rows.append(row)

    for root in [
        RAW / "baseline_protocols_B10_3000slot_3ep",
        RAW / "baseline_protocols_B15_3000slot_2ep",
    ]:
        if not root.exists():
            continue
        match_b = re.search(r"B(\d+)", root.name)
        if not match_b:
            continue
        beamwidth = match_b.group(1)
        for node_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("N")):
            node_count = node_dir.name[1:]
            for method_dir in sorted(p for p in node_dir.iterdir() if p.is_dir()):
                metric_file = method_dir / "eval_episode_metrics.csv"
                if not metric_file.exists():
                    continue
                method_key = "baseline_" + method_dir.name
                for row in read_csv(metric_file):
                    row = dict(row)
                    row["run"] = f"{root.name}/{node_dir.name}/{method_dir.name}"
                    row["method"] = METHOD_LABELS.get(method_key, method_key)
                    row["method_key"] = method_key
                    row["node_count"] = node_count
                    row["beamwidth_deg"] = beamwidth
                    row["source_type"] = "baseline"
                    rows.append(row)
    return rows


def method_key_from_name(name: str) -> str:
    for prefix in [
        "bc_notable_nogate",
        "bc_notable_gate",
        "notable_nogate",
        "notable_gate",
        "table_nogate",
        "table_gate",
    ]:
        if name.startswith(prefix):
            return prefix
    return name


def summarize_training(rows: list[dict[str, str]]) -> list[dict[str, str | float | int]]:
    out = []
    by_run: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_run.setdefault(row["run"], []).append(row)
    for run, group in sorted(by_run.items()):
        tail = sorted(group, key=lambda r: int(float(r.get("episode", 0))))[-10:]
        out.append(
            {
                "run": run,
                "method": group[0]["method"],
                "episodes": len(group),
                **{f"last10_{metric}": avg(tail, metric) for metric in [
                    "discovery_rate",
                    "episode_return_sum",
                    "collision_count",
                    "collision_penalized_discovery_rate",
                    "lambda2",
                    "expert_bc_loss",
                ]},
            }
        )
    return out


def summarize_transfer(rows: list[dict[str, str]]) -> list[dict[str, str | float | int]]:
    out = []
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["beamwidth_deg"], row["node_count"], row["method"])
        groups.setdefault(key, []).append(row)
    for (beamwidth, node_count, method), group in sorted(groups.items(), key=lambda x: (int(x[0][0]), int(x[0][1]), x[0][2])):
        item: dict[str, str | float | int] = {
            "beamwidth_deg": beamwidth,
            "node_count": node_count,
            "method": method,
            "episodes": len(group),
        }
        for metric in METRICS:
            item[f"{metric}_mean"] = avg(group, metric)
            item[f"{metric}_std"] = stdev(group, metric)
        out.append(item)
    return out


def plot_all(training_rows: list[dict[str, str]], transfer_summary: list[dict[str, str | float | int]]) -> None:
    plot_training_curve(training_rows, "discovery_rate", "Training Discovery Rate", "training_discovery_rate.png")
    plot_training_curve(training_rows, "episode_return_sum", "Training Episode Return", "training_episode_return.png")
    plot_training_curve(training_rows, "collision_count", "Training Collision Count", "training_collision_count.png")
    plot_training_curve(training_rows, "lambda2", "Training Algebraic Connectivity", "training_lambda2.png")
    plot_training_curve(training_rows, "collision_penalized_discovery_rate", "Training Collision-Penalized Discovery", "training_cpd.png")
    plot_training_curve(training_rows, "expert_bc_loss", "Expert BC Loss", "training_expert_bc_loss.png", only_methods=("BC-MARL", "BC-MARL+gate"))
    plot_training_curve(training_rows, "access_gate_backoff_ratio", "Access-Gate Backoff Ratio", "training_gate_backoff_ratio.png", only_methods=("MARL+gate", "MARL+tables+gate", "BC-MARL+gate"))
    plot_training_curve(training_rows, "access_gate_normal_ratio", "Access-Gate Normal Ratio", "training_gate_normal_ratio.png", only_methods=("MARL+gate", "MARL+tables+gate", "BC-MARL+gate"))

    for beamwidth in ("10", "15"):
        for node_count in ("50", "100"):
            plot_bar(transfer_summary, beamwidth, node_count, "discovery_rate", f"B={beamwidth}, N={node_count}: Discovery Rate", f"transfer_b{beamwidth}_n{node_count}_discovery.png")
            plot_bar(transfer_summary, beamwidth, node_count, "collision_count", f"B={beamwidth}, N={node_count}: Collision Count", f"transfer_b{beamwidth}_n{node_count}_collision.png", log_y=True)
            plot_bar(transfer_summary, beamwidth, node_count, "collision_penalized_discovery_rate", f"B={beamwidth}, N={node_count}: Collision-Penalized Discovery", f"transfer_b{beamwidth}_n{node_count}_cpd.png")
            plot_bar(transfer_summary, beamwidth, node_count, "lambda2", f"B={beamwidth}, N={node_count}: Algebraic Connectivity", f"transfer_b{beamwidth}_n{node_count}_lambda2.png")
    plot_beamwidth_transfer(transfer_summary)


def plot_training_curve(
    rows: list[dict[str, str]],
    metric: str,
    title: str,
    filename: str,
    only_methods: tuple[str, ...] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        method = row["method"]
        if only_methods is not None and method not in only_methods:
            continue
        if row.get(metric, "") == "":
            continue
        grouped.setdefault(method, []).append(row)
    for method, group in grouped.items():
        group = sorted(group, key=lambda r: int(float(r.get("episode", 0))))
        x = [int(float(r["episode"])) for r in group]
        y = [to_float(r.get(metric)) for r in group]
        ax.plot(x, y, label=method, color=PALETTE.get(method), linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)


def plot_bar(
    summary: list[dict[str, str | float | int]],
    beamwidth: str,
    node_count: str,
    metric: str,
    title: str,
    filename: str,
    *,
    log_y: bool = False,
) -> None:
    rows = [
        row
        for row in summary
        if str(row["beamwidth_deg"]) == beamwidth
        and str(row["node_count"]) == node_count
        and row["method"] in PALETTE
    ]
    if not rows:
        return
    rows.sort(key=lambda r: float(r[f"{metric}_mean"]), reverse=metric != "collision_count")
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [str(r["method"]) for r in rows]
    values = [float(r[f"{metric}_mean"]) for r in rows]
    errors = [float(r[f"{metric}_std"]) for r in rows]
    colors = [PALETTE.get(label, "#777777") for label in labels]
    ax.bar(range(len(values)), values, yerr=errors, color=colors, edgecolor="black", linewidth=0.4, capsize=2)
    ax.set_title(title)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    if log_y:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)


def plot_beamwidth_transfer(summary: list[dict[str, str | float | int]]) -> None:
    methods = ["BC-MARL", "Wang ISAC", "Collision-aware ISAC", "Uniform random"]
    for node_count in ("50", "100"):
        fig, ax = plt.subplots(figsize=(8, 6))
        for method in methods:
            rows = [
                row
                for row in summary
                if str(row["node_count"]) == node_count and row["method"] == method
            ]
            rows.sort(key=lambda r: int(str(r["beamwidth_deg"])))
            if not rows:
                continue
            x = [int(str(r["beamwidth_deg"])) for r in rows]
            y = [float(r["discovery_rate_mean"]) for r in rows]
            ax.plot(x, y, marker="o", label=method, color=PALETTE.get(method), linewidth=1.8)
        ax.set_title(f"Beamwidth Transfer, N={node_count}")
        ax.set_xlabel("Beamwidth (deg)")
        ax.set_ylabel("Discovery Rate")
        ax.set_xticks([10, 15])
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"beamwidth_transfer_n{node_count}.png")
        plt.close(fig)


def write_report(training_summary: list[dict[str, str | float | int]], transfer_summary: list[dict[str, str | float | int]]) -> None:
    def row(beamwidth: str, node_count: str, method: str) -> dict[str, str | float | int] | None:
        for item in transfer_summary:
            if str(item["beamwidth_deg"]) == beamwidth and str(item["node_count"]) == node_count and item["method"] == method:
                return item
        return None

    key_rows = [
        row("10", "100", "BC-MARL"),
        row("10", "100", "MARL"),
        row("10", "100", "Wang ISAC"),
        row("10", "100", "Collision-aware ISAC"),
        row("15", "100", "BC-MARL"),
        row("15", "100", "Wang ISAC"),
    ]
    lines = [
        "# Overnight MARL-ISAC Rebuild Report (2026-07-09)",
        "",
        "## Scope",
        "",
        "This bundle consolidates the single-RF, N=10-trained MARL/ISAC runs, N=50/N=100 transfer evaluations, B=10 and B=15 beamwidth checks, Wang-style baselines, rule ISAC baselines, and expert-assisted BC-MARL variants generated overnight.",
        "",
        "## Main Findings",
        "",
        "- ISAC sensing/candidate information is decisive: non-ISAC random and no-ISAC proxy protocols remain near zero discovery in the narrow-beam 3D setting.",
        "- Wang-style ISAC and collision-aware rule ISAC are currently stronger than trained MARL on the primary target discovery-rate metric.",
        "- Expert-assisted BC-MARL improves large-scale transfer discovery over the earlier trained MARL, but it creates excessive collisions and therefore poor collision-penalized discovery.",
        "- Table exchange is not yet a reliable win in the current implementation; it needs trust-gated fusion instead of unconditional boosting.",
        "",
        "## Selected Metrics",
        "",
        "| B | N | Method | Discovery | Collisions | CPD | Lambda2 |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for item in [r for r in key_rows if r is not None]:
        lines.append(
            f"| {item['beamwidth_deg']} | {item['node_count']} | {item['method']} | "
            f"{float(item['discovery_rate_mean']):.3f} | {float(item['collision_count_mean']):.1f} | "
            f"{float(item['collision_penalized_discovery_rate_mean']):.3f} | {float(item['lambda2_mean']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Paper-Readiness Judgment",
            "",
            "The current data are useful for a diagnostic paper section and for motivating the final method, but they are not yet sufficient to claim a high-level TWC/TCOM MARL method that beats Wang-style ISAC baselines. The strongest defensible claim is: ISAC-assisted empty-beam exclusion makes the problem tractable; naive MARL does not inherit that benefit under N=10 to N=100 transfer; rule-guided expert pretraining improves raw discovery but must be paired with stronger collision-constrained MARL.",
            "",
            "## Next Technical Move",
            "",
            "The next method should factor actions into two timescales: a rule/ISAC candidate beam executor and a learned constrained access controller trained with collision budget or Lagrangian penalty. The gate should learn when to throttle Tx, not overwrite candidate beams. Table exchange should be trust-gated by recency, collision evidence, and peer-table consistency.",
            "",
            "## Artifact Locations",
            "",
            f"- Tables: `{TABLE_DIR.relative_to(ROOT)}`",
            f"- Figures: `{FIG_DIR.relative_to(ROOT)}`",
            f"- Raw results: `{RAW.relative_to(ROOT)}`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(training_rows: list[dict[str, str]], transfer_rows: list[dict[str, str]]) -> None:
    manifest = {
        "created_at": "2026-07-09",
        "raw_root": str(RAW.relative_to(ROOT)),
        "table_dir": str(TABLE_DIR.relative_to(ROOT)),
        "figure_dir": str(FIG_DIR.relative_to(ROOT)),
        "training_rows": len(training_rows),
        "transfer_rows": len(transfer_rows),
        "figures": sorted(p.name for p in FIG_DIR.glob("*.png")),
    }
    (TABLE_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def avg(rows: list[dict[str, object]], key: str) -> float:
    vals = [to_float(row.get(key)) for row in rows]
    vals = [value for value in vals if math.isfinite(value)]
    return statistics.mean(vals) if vals else float("nan")


def stdev(rows: list[dict[str, object]], key: str) -> float:
    vals = [to_float(row.get(key)) for row in rows]
    vals = [value for value in vals if math.isfinite(value)]
    return statistics.stdev(vals) if len(vals) >= 2 else 0.0


def to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    main()
