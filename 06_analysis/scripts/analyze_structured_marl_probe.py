from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


FIGSIZE = (6.4, 4.8)
DPI = 300
VARIANT_ORDER = ("flat", "mask", "mask_score", "mask_score_topo_rule")
VARIANT_LABELS = {
    "flat": "Flat",
    "mask": "Mask",
    "mask_score": "Mask+score",
    "mask_score_topo_rule": "Full residual",
}
PHASE_LABELS = {
    "eval_deterministic": "Det.",
    "eval_stochastic": "Stoch.",
}
COLORS = {
    "flat": "#0072B2",
    "mask": "#56B4E9",
    "mask_score": "#009E73",
    "mask_score_topo_rule": "#E69F00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate structured MARL probe outputs.")
    parser.add_argument("--input-root", default=None, help="Probe root. Defaults to latest marker.")
    parser.add_argument("--output", default="06_analysis/paper_tables/structured_marl_probe")
    parser.add_argument("--figures", default="06_analysis/paper_figures/structured_marl_probe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = resolve_input_root(args.input_root)
    output_dir = Path(args.output)
    figure_dir = Path(args.figures)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    rows = load_probe_rows(input_root, pd)
    if rows.empty:
        raise FileNotFoundError(f"No training_history.csv files found under {input_root}.")
    rows.to_csv(output_dir / "structured_marl_probe_rows.csv", index=False)
    summary = summarize_eval(rows, pd)
    summary.to_csv(output_dir / "structured_marl_probe_eval_summary.csv", index=False)
    training = summarize_training(rows, pd)
    training.to_csv(output_dir / "structured_marl_probe_training_summary.csv", index=False)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "figure_dir": str(figure_dir),
        "rows": int(len(rows)),
        "runs": int(rows["run"].nunique()),
        "figures": write_figures(rows, summary, figure_dir),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output_dir, manifest, summary)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def resolve_input_root(raw: str | None) -> Path:
    if raw:
        return Path(raw)
    marker = Path("05_simulation/results_raw/pre11_marl_probe_latest.txt")
    if not marker.exists():
        raise FileNotFoundError("Missing latest marker and --input-root was not provided.")
    return Path(marker.read_text(encoding="utf-8-sig").strip())


def load_probe_rows(input_root: Path, pd):
    frames = []
    for history_path in sorted(input_root.rglob("training_history.csv")):
        run_dir = history_path.parent
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(history_path)
        frame["run"] = run_dir.name
        frame["probe_group"] = infer_probe_group(run_dir.name)
        if "env_protocol" not in frame.columns:
            frame["env_protocol"] = str(manifest.get("env_protocol", "isac_structured_marl"))
        frame["variant"] = infer_variant(run_dir.name, manifest)
        frame["train_seed"] = int(manifest.get("seed", 0))
        frame["candidate_mask"] = bool(manifest.get("candidate_mask", False))
        frame["candidate_score"] = bool(manifest.get("candidate_score", False))
        frame["topology_deficit"] = bool(manifest.get("topology_deficit", False))
        frame["rule_residual"] = bool(manifest.get("rule_residual", False))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    rows = pd.concat(frames, ignore_index=True)
    for column in rows.columns:
        if column in {"phase", "env_protocol", "expert_protocol", "run", "probe_group", "variant"}:
            continue
        try:
            rows[column] = pd.to_numeric(rows[column], errors="raise")
        except (TypeError, ValueError):
            pass
    rows["variant"] = pd.Categorical(rows["variant"], categories=VARIANT_ORDER, ordered=True)
    return rows.sort_values(["probe_group", "variant", "train_seed", "episode"]).reset_index(drop=True)


def infer_variant(run_name: str, manifest: dict) -> str:
    if run_name.startswith("mask_score_topo_rule"):
        return "mask_score_topo_rule"
    if run_name.startswith("mask_score"):
        return "mask_score"
    if run_name.startswith("mask"):
        return "mask"
    if run_name.startswith("flat"):
        return "flat"
    if manifest.get("rule_residual"):
        return "mask_score_topo_rule"
    if manifest.get("candidate_score"):
        return "mask_score"
    if manifest.get("candidate_mask"):
        return "mask"
    return "flat"


def infer_probe_group(run_name: str) -> str:
    if run_name.startswith("no_isac_env"):
        return "no_isac_env"
    if run_name.startswith("no_isac_label"):
        return "no_isac_label"
    if run_name.startswith("rl10"):
        return "rl10"
    return "core"


def summarize_eval(rows, pd):
    eval_rows = rows[rows["phase"].astype(str).str.startswith("eval_")].copy()
    metrics = [
        "env_discovery_rate",
        "env_discovered_edges",
        "env_lambda2",
        "env_empty_scan_ratio",
        "env_collision_count",
        "reward_mean",
    ]
    grouped = []
    for (probe_group, variant, phase), group in eval_rows.groupby(["probe_group", "variant", "phase"], observed=True):
        row = {
            "probe_group": probe_group,
            "variant": str(variant),
            "phase": phase,
            "eval_n": int(len(group)),
            "seed_n": int(group["train_seed"].nunique()),
            "nonzero_edges": int((group["env_discovered_edges"].fillna(0) > 0).sum()),
        }
        for metric in metrics:
            values = pd.to_numeric(group.get(metric), errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_ci95"] = 1.96 * row[f"{metric}_std"] / (len(values) ** 0.5) if len(values) > 1 else 0.0
        grouped.append(row)
    return pd.DataFrame(grouped)


def summarize_training(rows, pd):
    train_rows = rows[rows["phase"].isin(["bc", "rl"])].copy()
    metrics = ["loss", "bc_mode_loss", "bc_beam_loss", "value_loss", "reward_mean", "mode_accuracy", "active_beam_accuracy"]
    grouped = []
    for (probe_group, variant, phase, episode), group in train_rows.groupby(
        ["probe_group", "variant", "phase", "episode"], observed=True
    ):
        row = {
            "probe_group": probe_group,
            "variant": str(variant),
            "phase": phase,
            "episode": int(episode),
            "n": int(len(group)),
        }
        for metric in metrics:
            if metric in group:
                row[f"{metric}_mean"] = float(pd.to_numeric(group[metric], errors="coerce").mean())
        grouped.append(row)
    return pd.DataFrame(grouped)


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
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    return plt


def write_figures(rows, summary, figure_dir: Path) -> list[dict]:
    plt = setup_matplotlib()
    plot_rows = rows[rows["probe_group"] == "core"].copy()
    plot_summary = summary[summary["probe_group"] == "core"].copy()
    if plot_rows.empty:
        plot_rows = rows
        plot_summary = summary
    figures = []
    figures.append(plot_eval_grouped(plot_summary, "env_discovery_rate_mean", "env_discovery_rate_ci95", "Discovery rate", figure_dir / "marl_eval_discovery_rate.png", plt))
    figures.append(plot_eval_grouped(plot_summary, "env_lambda2_mean", "env_lambda2_ci95", r"$\lambda_2$", figure_dir / "marl_eval_lambda2.png", plt))
    figures.append(plot_eval_grouped(plot_summary, "env_empty_scan_ratio_mean", "env_empty_scan_ratio_ci95", "Empty-scan ratio", figure_dir / "marl_eval_empty_scan_ratio.png", plt))
    figures.append(plot_training_curve(plot_rows, "loss", "BC loss", figure_dir / "marl_bc_loss_curve.png", plt))
    figures.append(plot_training_curve(plot_rows, "reward_mean", "Mean reward", figure_dir / "marl_reward_curve.png", plt))
    figures.append(plot_efficiency_scatter(plot_summary, figure_dir / "marl_empty_collision_tradeoff.png", plt))
    return figures


def plot_eval_grouped(summary, metric: str, ci_metric: str, ylabel: str, path: Path, plt) -> dict:
    import numpy as np

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(VARIANT_ORDER), dtype=float)
    width = 0.34
    for offset, phase in [(-width / 2, "eval_deterministic"), (width / 2, "eval_stochastic")]:
        rows = summary[summary["phase"] == phase].set_index("variant")
        means = [float(rows.loc[v, metric]) if v in rows.index else 0.0 for v in VARIANT_ORDER]
        cis = [float(rows.loc[v, ci_metric]) if v in rows.index else 0.0 for v in VARIANT_ORDER]
        ax.bar(x + offset, means, width=width, yerr=cis, capsize=3, label=PHASE_LABELS[phase], color="#0072B2" if phase.endswith("deterministic") else "#E69F00", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER], rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"path": str(path), "type": "eval_grouped", "metric": metric}


def plot_training_curve(rows, metric: str, ylabel: str, path: Path, plt) -> dict:
    import pandas as pd

    subset = rows[(rows["phase"] == "bc") & rows[metric].notna()].copy()
    if subset.empty:
        return {"path": str(path), "status": "skipped", "reason": f"missing {metric}"}
    grouped = (
        subset.groupby(["variant", "episode"], observed=True)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["ci95"] = grouped.apply(lambda row: 1.96 * float(row["std"] or 0.0) / (float(row["count"]) ** 0.5), axis=1)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for variant in VARIANT_ORDER:
        curve = grouped[grouped["variant"].astype(str) == variant]
        if curve.empty:
            continue
        ax.plot(curve["episode"], curve["mean"], label=VARIANT_LABELS[variant], color=COLORS[variant])
        ax.fill_between(curve["episode"], curve["mean"] - curve["ci95"], curve["mean"] + curve["ci95"], color=COLORS[variant], alpha=0.12, linewidth=0)
    ax.set_xlabel("BC episode")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"path": str(path), "type": "training_curve", "metric": metric}


def plot_efficiency_scatter(summary, path: Path, plt) -> dict:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    subset = summary[summary["phase"] == "eval_stochastic"].copy()
    for variant in VARIANT_ORDER:
        rows = subset[subset["variant"].astype(str) == variant]
        if rows.empty:
            continue
        row = rows.iloc[0]
        ax.scatter(
            row["env_empty_scan_ratio_mean"],
            row["env_collision_count_mean"],
            s=80 + 260 * max(0.0, float(row["env_discovery_rate_mean"])),
            color=COLORS[variant],
            alpha=0.85,
            label=VARIANT_LABELS[variant],
            edgecolor="white",
            linewidth=0.8,
        )
    ax.set_xlabel("Empty-scan ratio")
    ax.set_ylabel("Collisions per episode")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"path": str(path), "type": "scatter", "metric": "empty_collision"}


def write_readme(output_dir: Path, manifest: dict, summary) -> None:
    readme = [
        "# Structured MARL Probe Aggregation",
        "",
        f"- Created: {manifest['created_at']}",
        f"- Input root: `{manifest['input_root']}`",
        f"- Runs loaded: {manifest['runs']}",
        f"- Rows loaded: {manifest['rows']}",
        "",
        "This aggregation is a method-probe summary, not a paper main-result replacement.",
        "It compares the same shared actor-critic path with flat actions, candidate masks, candidate scores, and full local rule residuals.",
    ]
    if not summary.empty:
        readme.extend(["", "## Eval Summary", "", "```csv", summary.to_csv(index=False).strip(), "```"])
    output_dir.joinpath("README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
