from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "06_analysis" / "paper_tables" / "paired_delta_summary"
BOOTSTRAP_REPS = 20000
RNG_SEED = 20260705


@dataclass(frozen=True)
class Comparison:
    block: str
    source: str
    treatment: str
    control: str
    filters: dict[str, object]
    pair_cols: tuple[str, ...]
    metrics: tuple[str, ...]
    note: str


MAIN_METRICS = (
    "discovery_rate",
    "empty_scan_ratio",
    "lambda2",
    "mean_discovery_delay",
    "collision_count",
)

EFFICIENCY_METRICS = (
    "discovery_rate",
    "collision_penalized_discovery_rate",
    "empty_scan_ratio",
    "lambda2",
    "collision_count",
)


COMPARISONS: tuple[Comparison, ...] = (
    Comparison(
        block="main_n100_b10_vs_enhanced_no_isac",
        source="06_analysis/paper_tables/round3_robustness/n100_density_multiseed/per_episode_summary.csv",
        treatment="improved_rl_isac",
        control="improved_rl_no_isac",
        filters={"node_count": 100, "beamwidth_deg": 10.0, "area_scale": "density", "mobility_model": "gauss_markov"},
        pair_cols=("scenario_seed", "node_count", "beamwidth_deg", "area_scale", "mobility_model"),
        metrics=MAIN_METRICS,
        note="Main N=100/B=10 density-scaled comparison against enhanced communication-only policy.",
    ),
    Comparison(
        block="main_n100_b10_vs_uniform_random",
        source="06_analysis/paper_tables/round3_robustness/n100_density_multiseed/per_episode_summary.csv",
        treatment="improved_rl_isac",
        control="uniform_random",
        filters={"node_count": 100, "beamwidth_deg": 10.0, "area_scale": "density", "mobility_model": "gauss_markov"},
        pair_cols=("scenario_seed", "node_count", "beamwidth_deg", "area_scale", "mobility_model"),
        metrics=MAIN_METRICS,
        note="Main N=100/B=10 density-scaled comparison against uniform random scanning.",
    ),
    Comparison(
        block="main_n100_b10_vs_skyorbs_like",
        source="06_analysis/paper_tables/round3_robustness/n100_density_multiseed/per_episode_summary.csv",
        treatment="improved_rl_isac",
        control="skyorbs_like_skip_scan",
        filters={"node_count": 100, "beamwidth_deg": 10.0, "area_scale": "density", "mobility_model": "gauss_markov"},
        pair_cols=("scenario_seed", "node_count", "beamwidth_deg", "area_scale", "mobility_model"),
        metrics=MAIN_METRICS,
        note="Reference comparison against the deterministic 3-D skip-scan baseline under this information boundary.",
    ),
    Comparison(
        block="main_n100_b15_vs_enhanced_no_isac",
        source="06_analysis/paper_tables/round3_robustness/n100_density_multiseed/per_episode_summary.csv",
        treatment="improved_rl_isac",
        control="improved_rl_no_isac",
        filters={"node_count": 100, "beamwidth_deg": 15.0, "area_scale": "density", "mobility_model": "gauss_markov"},
        pair_cols=("scenario_seed", "node_count", "beamwidth_deg", "area_scale", "mobility_model"),
        metrics=MAIN_METRICS,
        note="Main N=100/B=15 density-scaled comparison against enhanced communication-only policy.",
    ),
    Comparison(
        block="round10_extra_n100_b10_vs_enhanced_no_isac",
        source="06_analysis/paper_tables/round10_n100_b10_b15_extra_seeds/per_episode_summary.csv",
        treatment="improved_rl_isac",
        control="improved_rl_no_isac",
        filters={"node_count": 100, "beamwidth_deg": 10.0, "area_scale": "density", "mobility_model": "gauss_markov"},
        pair_cols=("scenario_seed", "node_count", "beamwidth_deg", "area_scale", "mobility_model"),
        metrics=MAIN_METRICS,
        note="Backup extra-seed N=100/B=10 comparison; used to check scenario-seed sensitivity, not to replace the main table.",
    ),
    Comparison(
        block="round10_extra_n100_b15_vs_enhanced_no_isac",
        source="06_analysis/paper_tables/round10_n100_b10_b15_extra_seeds/per_episode_summary.csv",
        treatment="improved_rl_isac",
        control="improved_rl_no_isac",
        filters={"node_count": 100, "beamwidth_deg": 15.0, "area_scale": "density", "mobility_model": "gauss_markov"},
        pair_cols=("scenario_seed", "node_count", "beamwidth_deg", "area_scale", "mobility_model"),
        metrics=MAIN_METRICS,
        note="Backup extra-seed N=100/B=15 comparison; used to check scenario-seed sensitivity, not to replace the main table.",
    ),
    Comparison(
        block="ablation_b10_candidate_set",
        source="06_analysis/paper_tables/round4_delay_ablation/per_episode_summary.csv",
        treatment="improved_rl_isac",
        control="ablation_isac_no_candidate_set",
        filters={"node_count": 100, "beamwidth_deg": 10.0, "area_scale": "density", "mobility_model": "gauss_markov"},
        pair_cols=("scenario_seed", "node_count", "beamwidth_deg", "area_scale", "mobility_model"),
        metrics=EFFICIENCY_METRICS,
        note="Mechanism check: full ISAC-assisted protocol minus candidate-set removal.",
    ),
    Comparison(
        block="ablation_b10_one_slot_boundary",
        source="06_analysis/paper_tables/round4_delay_ablation/per_episode_summary.csv",
        treatment="improved_rl_isac",
        control="ablation_isac_one_slot_delay",
        filters={"node_count": 100, "beamwidth_deg": 10.0, "area_scale": "density", "mobility_model": "gauss_markov"},
        pair_cols=("scenario_seed", "node_count", "beamwidth_deg", "area_scale", "mobility_model"),
        metrics=EFFICIENCY_METRICS,
        note="Implementation-boundary check: low-latency candidate use minus one-slot delayed use.",
    ),
    Comparison(
        block="ablation_b10_delay_vs_no_isac",
        source="06_analysis/paper_tables/round4_delay_ablation/per_episode_summary.csv",
        treatment="ablation_isac_one_slot_delay",
        control="improved_rl_no_isac",
        filters={"node_count": 100, "beamwidth_deg": 10.0, "area_scale": "density", "mobility_model": "gauss_markov"},
        pair_cols=("scenario_seed", "node_count", "beamwidth_deg", "area_scale", "mobility_model"),
        metrics=EFFICIENCY_METRICS,
        note="Implementation-boundary check: one-slot delayed ISAC remains above enhanced no-ISAC.",
    ),
)


def mobility_comparisons() -> list[Comparison]:
    blocks: list[Comparison] = []
    for beam in (10.0, 15.0):
        for mobility in ("gauss_markov", "random_walk", "random_direction", "random_waypoint"):
            blocks.append(
                Comparison(
                    block=f"mobility_b{int(beam)}_{mobility}_vs_no_isac",
                    source="06_analysis/paper_tables/round5_mobility_transfer/per_episode_summary.csv",
                    treatment="improved_rl_isac",
                    control="improved_rl_no_isac",
                    filters={
                        "node_count": 100,
                        "beamwidth_deg": beam,
                        "area_scale": "density",
                        "mobility_model": mobility,
                    },
                    pair_cols=("scenario_seed", "node_count", "beamwidth_deg", "area_scale", "mobility_model"),
                    metrics=EFFICIENCY_METRICS,
                    note="Mobility-boundary paired comparison against enhanced no-ISAC.",
                )
            )
    return blocks


def error_profile_comparisons() -> list[Comparison]:
    blocks: list[Comparison] = []
    profiles = (
        (0.0, 0.0, 0.0, "nominal"),
        (0.01, 0.05, 0.5, "mild"),
        (0.05, 0.15, 1.0, "moderate"),
        (0.10, 0.30, 1.5, "severe"),
    )
    for mobility in ("gauss_markov", "random_walk"):
        for pfa, pmd, offset, label in profiles:
            filters = {
                "node_count": 100,
                "beamwidth_deg": 15.0,
                "area_scale": "density",
                "mobility_model": mobility,
                "false_alarm_rate": pfa,
                "miss_detection_rate": pmd,
                "angular_cell_offset_std": offset,
            }
            blocks.append(
                Comparison(
                    block=f"error_b15_{mobility}_{label}_vs_no_isac",
                    source="06_analysis/paper_tables/round8_error_profiles_b15_gm_rw_600slot/per_episode_summary.csv",
                    treatment="improved_rl_isac",
                    control="improved_rl_no_isac",
                    filters=filters,
                    pair_cols=(
                        "scenario_seed",
                        "node_count",
                        "beamwidth_deg",
                        "area_scale",
                        "mobility_model",
                        "false_alarm_rate",
                        "miss_detection_rate",
                        "angular_cell_offset_std",
                    ),
                    metrics=EFFICIENCY_METRICS,
                    note="B=15 configured ISAC-error paired comparison against enhanced no-ISAC.",
                )
            )
    return blocks


def approx_equal_series(series: pd.Series, value: object) -> pd.Series:
    if isinstance(value, float):
        return np.isclose(series.astype(float), value)
    return series == value


def apply_filters(df: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, value in filters.items():
        if col not in df.columns:
            raise KeyError(f"Missing filter column {col}")
        mask &= approx_equal_series(df[col], value)
    return df.loc[mask].copy()


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 0:
        return (np.nan, np.nan)
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    draws = rng.choice(values, size=(BOOTSTRAP_REPS, len(values)), replace=True).mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]).astype(float))


def summarize_comparison(comp: Comparison, rng: np.random.Generator) -> list[dict[str, object]]:
    df = pd.read_csv(ROOT / comp.source)
    df = apply_filters(df, comp.filters)
    treatment = df[df["protocol"] == comp.treatment]
    control = df[df["protocol"] == comp.control]
    if treatment.empty or control.empty:
        raise ValueError(f"Empty treatment/control for {comp.block}: {comp.treatment} vs {comp.control}")

    rows: list[dict[str, object]] = []
    merge_cols = list(comp.pair_cols)
    for metric in comp.metrics:
        if metric not in treatment.columns or metric not in control.columns:
            continue
        left = treatment[merge_cols + [metric]].rename(columns={metric: "treatment_value"})
        right = control[merge_cols + [metric]].rename(columns={metric: "control_value"})
        paired = left.merge(right, on=merge_cols, how="inner")
        if paired.empty:
            raise ValueError(f"No paired rows for {comp.block} metric {metric}")
        paired["delta"] = paired["treatment_value"] - paired["control_value"]
        deltas = paired["delta"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(deltas, rng)
        rows.append(
            {
                "block": comp.block,
                "source": comp.source,
                "treatment": comp.treatment,
                "control": comp.control,
                "metric": metric,
                "n_pairs": int(len(deltas)),
                "treatment_mean": float(paired["treatment_value"].mean()),
                "control_mean": float(paired["control_value"].mean()),
                "delta_mean": float(deltas.mean()),
                "delta_std": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "n_positive": int((deltas > 0).sum()),
                "n_negative": int((deltas < 0).sum()),
                "n_zero": int((deltas == 0).sum()),
                "filters_json": json.dumps(comp.filters, sort_keys=True),
                "pair_cols": ",".join(comp.pair_cols),
                "note": comp.note,
            }
        )
    return rows


def build_summary() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows: list[dict[str, object]] = []
    for comp in (*COMPARISONS, *mobility_comparisons(), *error_profile_comparisons()):
        rows.extend(summarize_comparison(comp, rng))
    return pd.DataFrame(rows)


def write_readme(df: pd.DataFrame) -> None:
    metric_counts = df.groupby("metric")["block"].count().to_dict()
    text = (
        "# Paired Delta Summary\n\n"
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n"
        "This directory contains paired treatment-control deltas computed from archived "
        "`per_episode_summary.csv` files. Pairing uses identical scenario seeds and scenario "
        "parameters, so each delta compares protocols on the same simulated geometry/mobility draw.\n\n"
        "Files:\n\n"
        "- `paired_delta_summary.csv`: one row per comparison block and metric, including treatment/control means, "
        "mean paired delta, bootstrap percentile 95% CI over paired deltas, and seed-level sign counts.\n\n"
        "Interpretation notes:\n\n"
        "- Bootstrap intervals are descriptive because several key blocks have only three paired seeds.\n"
        "- Positive deltas are beneficial for discovery rate, collision-penalized discovery, and lambda2; "
        "negative deltas are beneficial for empty-scan ratio, delay, and collision count.\n"
        "- The SkyOrbs-like comparison is a deterministic 3-D skip-scan reference under this simulator's information boundary, "
        "not a strict reproduction of the full SkyOrbs protocol.\n\n"
        f"Rows: {len(df)}\n\n"
        f"Metric rows: {json.dumps(metric_counts, sort_keys=True)}\n"
    )
    (OUTPUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_summary()
    df.to_csv(OUTPUT_DIR / "paired_delta_summary.csv", index=False)
    write_readme(df)
    print(f"wrote {len(df)} rows to {OUTPUT_DIR / 'paired_delta_summary.csv'}")


if __name__ == "__main__":
    main()
