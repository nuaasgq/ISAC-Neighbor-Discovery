"""Post-hoc radio-state power sensitivity for round13.

This script reweights the already archived per-episode radio-state counts under
several plausible power profiles. It does not rerun discovery simulations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


FIGSIZE = (6.4, 4.8)
DPI = 300
SLOT_DURATION_COL = "slot_duration_ms"
PROTOCOL_ORDER = [
    "ablation_isac_one_slot_delay",
    "improved_rl_isac",
    "collision_aware_isac",
]
PROTOCOL_LABELS = {
    "ablation_isac_one_slot_delay": "One-slot delay",
    "improved_rl_isac": "Proposed",
    "collision_aware_isac": "Collision-aware",
}
COLORS = {
    "ablation_isac_one_slot_delay": "#0072B2",
    "improved_rl_isac": "#E69F00",
    "collision_aware_isac": "#D55E00",
}


@dataclass(frozen=True)
class PowerProfile:
    name: str
    tx: float
    rx: float
    sense: float
    idle: float
    piggyback: float
    note: str


POWER_PROFILES = [
    PowerProfile("default", 1.0, 0.6, 1.2, 0.05, 0.2, "Simulator default radio-state accounting."),
    PowerProfile("tx_x2", 2.0, 0.6, 1.2, 0.05, 0.2, "Transmit-heavy profile."),
    PowerProfile("rx_x2", 1.0, 1.2, 1.2, 0.05, 0.2, "Receive-heavy profile."),
    PowerProfile("sense_x2", 1.0, 0.6, 2.4, 0.05, 0.4, "Sensing and piggyback-sensing heavy profile."),
    PowerProfile("idle_x4", 1.0, 0.6, 1.2, 0.20, 0.2, "Higher idle/listening baseline profile."),
    PowerProfile("sense_half", 1.0, 0.6, 0.6, 0.05, 0.1, "Lower sensing-overhead profile."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze post-hoc radio-state power sensitivity.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("05_simulation/results_raw/round13_collision_energy_10seed/per_episode_summary.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("06_analysis/paper_tables/round13_energy_sensitivity"))
    parser.add_argument("--figures", type=Path, default=Path("06_analysis/paper_figures/round13_energy_sensitivity"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    episode = pd.read_csv(args.source)
    reweighted = build_reweighted_episode_table(episode)
    summary = build_summary(reweighted)
    paired = build_paired_delta(reweighted)
    reweighted.to_csv(args.output / "energy_sensitivity_per_episode.csv", index=False)
    summary.to_csv(args.output / "energy_sensitivity_summary.csv", index=False)
    paired.to_csv(args.output / "energy_sensitivity_paired_deltas.csv", index=False)
    figures = write_figures(summary, paired, args.figures)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(args.source),
        "output": str(args.output),
        "figures": figures,
        "power_profiles": [profile.__dict__ for profile in POWER_PROFILES],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(args.output, manifest, paired)
    print(json.dumps({"summary_rows": len(summary), "paired_rows": len(paired), "figures": figures}, indent=2))


def build_reweighted_episode_table(episode: pd.DataFrame) -> pd.DataFrame:
    rows = []
    required = [
        "tx_actions",
        "rx_actions",
        "sense_actions",
        "idle_actions",
        "piggyback_sense_actions",
        "discovered_edges",
        SLOT_DURATION_COL,
    ]
    missing = [col for col in required if col not in episode.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for profile in POWER_PROFILES:
        df = episode.copy()
        slot_s = df[SLOT_DURATION_COL].astype(float) / 1000.0
        df["power_profile"] = profile.name
        df["tx_power_w"] = profile.tx
        df["rx_power_w"] = profile.rx
        df["sense_power_w"] = profile.sense
        df["idle_power_w"] = profile.idle
        df["piggyback_sense_power_w"] = profile.piggyback
        df["energy_j_reweighted"] = slot_s * (
            df["tx_actions"] * profile.tx
            + df["rx_actions"] * profile.rx
            + df["sense_actions"] * profile.sense
            + df["idle_actions"] * profile.idle
            + df["piggyback_sense_actions"] * profile.piggyback
        )
        df["discoveries_per_joule_reweighted"] = df["discovered_edges"] / df["energy_j_reweighted"].clip(lower=1e-12)
        df["energy_per_discovery_censored_j_reweighted"] = df["energy_j_reweighted"] / df["discovered_edges"].clip(lower=1)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def build_summary(reweighted: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        reweighted[reweighted["protocol"].isin(PROTOCOL_ORDER)]
        .groupby(["power_profile", "beamwidth_deg", "protocol"], as_index=False)
        .agg(
            n=("discoveries_per_joule_reweighted", "count"),
            discovery_rate_mean=("discovery_rate", "mean"),
            discoveries_per_joule_mean=("discoveries_per_joule_reweighted", "mean"),
            discoveries_per_joule_std=("discoveries_per_joule_reweighted", "std"),
            energy_per_discovery_mean=("energy_per_discovery_censored_j_reweighted", "mean"),
            energy_per_discovery_std=("energy_per_discovery_censored_j_reweighted", "std"),
            energy_j_mean=("energy_j_reweighted", "mean"),
            energy_j_std=("energy_j_reweighted", "std"),
        )
        .fillna(0.0)
    )
    for metric in ("discoveries_per_joule", "energy_per_discovery", "energy_j"):
        grouped[f"{metric}_ci95"] = 1.96 * grouped[f"{metric}_std"] / grouped["n"].pow(0.5)
    return grouped.sort_values(["beamwidth_deg", "power_profile", "protocol"])


def build_paired_delta(reweighted: pd.DataFrame) -> pd.DataFrame:
    pair_cols = ["power_profile", "scenario_seed", "node_count", "beamwidth_deg", "area_scale", "mobility_model"]
    metrics = [
        "discoveries_per_joule_reweighted",
        "energy_per_discovery_censored_j_reweighted",
        "energy_j_reweighted",
    ]
    controls = ["improved_rl_isac", "ablation_isac_one_slot_delay"]
    rows = []
    for (power_profile, beamwidth), profile_df in reweighted.groupby(["power_profile", "beamwidth_deg"]):
        treatment = profile_df[profile_df["protocol"] == "collision_aware_isac"]
        for control in controls:
            control_df = profile_df[profile_df["protocol"] == control]
            merged = treatment.merge(control_df, on=pair_cols, suffixes=("_treatment", "_control"))
            for metric in metrics:
                delta = merged[f"{metric}_treatment"] - merged[f"{metric}_control"]
                rows.append(
                    {
                        "power_profile": str(power_profile),
                        "control": control,
                        "metric": metric.replace("_reweighted", ""),
                        "beamwidth_deg": float(beamwidth),
                        "n_pairs": int(delta.shape[0]),
                        "treatment_mean": float(merged[f"{metric}_treatment"].mean()),
                        "control_mean": float(merged[f"{metric}_control"].mean()),
                        "delta_mean": float(delta.mean()),
                        "delta_std": float(delta.std(ddof=1)) if delta.shape[0] > 1 else 0.0,
                        "delta_ci95": float(1.96 * delta.std(ddof=1) / (delta.shape[0] ** 0.5)) if delta.shape[0] > 1 else 0.0,
                        "positive_pairs": int((delta > 0).sum()),
                        "negative_pairs": int((delta < 0).sum()),
                        "zero_pairs": int((delta == 0).sum()),
                    }
                )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["beamwidth_deg", "control", "metric", "power_profile"])


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
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    return plt


def write_figures(summary: pd.DataFrame, paired: pd.DataFrame, figure_dir: Path) -> list[dict]:
    plt = setup_matplotlib()
    figures = [
        plot_discoveries_per_joule(summary, figure_dir / "energy_sensitivity_discoveries_per_joule.png", plt),
        plot_delta(paired, figure_dir / "energy_sensitivity_delta_vs_proposed.png", plt),
    ]
    return figures


def plot_discoveries_per_joule(summary: pd.DataFrame, path: Path, plt) -> dict:
    import numpy as np

    profiles = [profile.name for profile in POWER_PROFILES]
    x = np.arange(len(profiles), dtype=float)
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=False)
    for ax, beamwidth in zip(axes, [10.0, 15.0]):
        subset = summary[summary["beamwidth_deg"] == beamwidth]
        for idx, protocol in enumerate(PROTOCOL_ORDER):
            rows = subset[subset["protocol"] == protocol].set_index("power_profile")
            values = [float(rows.loc[p, "discoveries_per_joule_mean"]) if p in rows.index else 0.0 for p in profiles]
            ax.bar(x + (idx - 1) * width, values, width=width, color=COLORS[protocol], label=PROTOCOL_LABELS[protocol])
        ax.set_title(f"B={int(beamwidth)} deg")
        ax.set_xticks(x)
        ax.set_xticklabels(profiles, rotation=35, ha="right")
        ax.set_ylabel("Discoveries per joule")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"path": str(path), "type": "bar", "metric": "discoveries_per_joule_reweighted"}


def plot_delta(paired: pd.DataFrame, path: Path, plt) -> dict:
    import numpy as np

    profiles = [profile.name for profile in POWER_PROFILES]
    x = np.arange(len(profiles), dtype=float)
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    metric_rows = paired[(paired["metric"] == "discoveries_per_joule") & (paired["control"] == "improved_rl_isac")]
    for ax, beamwidth in zip(axes, [10.0, 15.0]):
        rows = metric_rows[metric_rows["beamwidth_deg"] == beamwidth].set_index("power_profile")
        values = [float(rows.loc[p, "delta_mean"]) if p in rows.index else 0.0 for p in profiles]
        errs = [float(rows.loc[p, "delta_ci95"]) if p in rows.index else 0.0 for p in profiles]
        ax.bar(x, values, yerr=errs, width=width, capsize=3, color="#D55E00", alpha=0.86)
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_title(f"B={int(beamwidth)} deg")
        ax.set_xticks(x)
        ax.set_xticklabels(profiles, rotation=35, ha="right")
        ax.set_ylabel("Delta vs proposed (disc./J)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"path": str(path), "type": "bar", "metric": "discoveries_per_joule_delta_vs_proposed"}


def write_readme(output: Path, manifest: dict, paired: pd.DataFrame) -> None:
    vs_proposed = paired[(paired["metric"] == "discoveries_per_joule") & (paired["control"] == "improved_rl_isac")]
    vs_delay = paired[(paired["metric"] == "discoveries_per_joule") & (paired["control"] == "ablation_isac_one_slot_delay")]
    positive_mean_vs_proposed = int((vs_proposed["delta_mean"] > 0).sum())
    total_vs_proposed = int(vs_proposed.shape[0])
    min_positive_vs_proposed = int(vs_proposed["positive_pairs"].min()) if not vs_proposed.empty else 0
    min_positive_vs_delay = int(vs_delay["positive_pairs"].min()) if not vs_delay.empty else 0
    boundary_rows = vs_proposed[vs_proposed["delta_mean"] <= 0]
    boundary_note = "No negative mean delta versus the proposed policy appears in the tested profiles."
    if not boundary_rows.empty:
        parts = []
        for _, row in boundary_rows.iterrows():
            parts.append(
                f"`{row['power_profile']}` at B={int(float(row['beamwidth_deg']))} deg "
                f"(mean delta {float(row['delta_mean']):.4f}, {int(row['positive_pairs'])}/10 positive)"
            )
        boundary_note = "Boundary case versus the proposed policy: " + "; ".join(parts) + "."
    text = [
        "# Round13 Radio-State Power Sensitivity",
        "",
        f"- Created: {manifest['created_at']}",
        f"- Source: `{manifest['source']}`",
        "",
        "This is a post-hoc reweighting of the round13 per-episode radio-state action counts.",
        "It does not rerun the neighbor-discovery simulator and does not change discovery or collision outcomes.",
        "",
        "## Main Result",
        "",
        f"Versus the one-slot-delay control, `collision_aware_isac` keeps positive discoveries-per-joule deltas in at least {min_positive_vs_delay}/10 paired seeds for every tested power profile and beamwidth.",
        f"Versus the proposed low-latency policy, the mean discoveries-per-joule delta is positive in {positive_mean_vs_proposed}/{total_vs_proposed} profile/beamwidth combinations; the weakest sign count is {min_positive_vs_proposed}/10 paired seeds.",
        boundary_note,
        "Thus the round13 energy result is useful diagnostic robustness evidence for TX-, sensing-, and idle-power variation, while RX-heavy platforms remain a boundary and no platform-calibrated energy optimality is claimed.",
        "",
        "## Power Profiles",
        "",
    ]
    for profile in POWER_PROFILES:
        text.append(
            f"- `{profile.name}`: tx={profile.tx}, rx={profile.rx}, sense={profile.sense}, "
            f"idle={profile.idle}, piggyback={profile.piggyback}. {profile.note}"
        )
    text.extend(
        [
            "",
            "## Outputs",
            "",
            "- `energy_sensitivity_per_episode.csv`",
            "- `energy_sensitivity_summary.csv`",
            "- `energy_sensitivity_paired_deltas.csv`",
            "- `manifest.json`",
        ]
    )
    output.joinpath("README.md").write_text("\n".join(text) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
