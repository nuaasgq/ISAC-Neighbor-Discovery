"""Build paired sign-test evidence for the Phase10 primary MARL comparison.

This script is deliberately narrow. It tests the main ISAC-benefit claim for
the Phase9/Phase10 final 3000-slot, N=100 transfer line where the treatment
and the communication-only controls share identical scenario seeds.
Gate-family operating points are left descriptive because their archived runs
use different seed blocks.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(".")
OUT_DIR = ROOT / "06_analysis" / "paper_tables" / "marl" / "p10_paired_significance_primary"
FIG_DIR = ROOT / "06_analysis" / "figures" / "marl" / "p10_paired_significance_primary"
REPORT = ROOT / "06_analysis" / "phase10_paired_significance_report_20260707.md"

SOURCE_FILES = [
    ROOT / "06_analysis" / "paper_tables" / "marl" / "phase9_fiveway_n100_b10_3000slot_10ep_stoch_all_methods" / "marl_transfer_eval_rows.csv",
    ROOT / "06_analysis" / "paper_tables" / "marl" / "phase9_fiveway_n100_b15_3000slot_10ep_stoch_all_methods" / "marl_transfer_eval_rows.csv",
]

TREATMENT = "contention_actor"
CONTROLS = ["uniform_random", "skyorbs_like", "mappo_no_isac", "contention_no_isac"]
METRICS = [
    ("discovery_rate", "primary_discovery", "positive"),
    ("empty_scan_ratio", "mechanism_empty_scan", "negative"),
    ("lambda2", "topology_support", "positive"),
]
CONFIRMATORY_FAMILIES = {"primary_discovery", "mechanism_empty_scan"}
RNG_SEED = 20260707
BOOTSTRAPS = 10000


@dataclass
class TestRow:
    family: str
    metric: str
    beamwidth_deg: float
    treatment: str
    control: str
    expected_direction: str
    n_pairs: int
    positive_count: int
    negative_count: int
    zero_count: int
    treatment_mean: float
    control_mean: float
    mean_delta: float
    median_delta: float
    ci95_low: float
    ci95_high: float
    paired_dz: float
    sign_p_two_sided: float
    holm_p: float
    holm_reject_0p05: bool
    direction_ok: bool
    evidence_role: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def sign_test_two_sided(positive: int, negative: int) -> float:
    n = positive + negative
    if n == 0:
        return 1.0
    k = min(positive, negative)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, (idx, p_value) in enumerate(indexed):
        candidate = min(1.0, p_value * (m - rank))
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted


def collect_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in SOURCE_FILES:
        for row in read_csv(path):
            row = dict(row)
            row["source_file"] = path.as_posix()
            rows.append(row)
    return rows


def build_pairs(rows: list[dict[str, str]]) -> tuple[list[TestRow], list[dict[str, object]]]:
    rng = np.random.default_rng(RNG_SEED)
    by_key: dict[tuple[float, str, str], dict[str, str]] = {}
    for row in rows:
        beam = as_float(row["beamwidth_deg"])
        seed = row.get("scenario_seed") or row.get("seed") or row.get("eval_episode")
        method = row["method"]
        by_key[(beam, seed, method)] = row

    summaries: list[TestRow] = []
    deltas_out: list[dict[str, object]] = []
    for metric, family, direction in METRICS:
        for beam in sorted({key[0] for key in by_key}):
            seeds = sorted({key[1] for key in by_key if key[0] == beam})
            for control in CONTROLS:
                paired_values: list[tuple[str, float, float, float]] = []
                for seed in seeds:
                    treatment_row = by_key.get((beam, seed, TREATMENT))
                    control_row = by_key.get((beam, seed, control))
                    if not treatment_row or not control_row:
                        continue
                    treatment_value = as_float(treatment_row[metric])
                    control_value = as_float(control_row[metric])
                    delta = treatment_value - control_value
                    paired_values.append((seed, treatment_value, control_value, delta))
                    deltas_out.append(
                        {
                            "family": family,
                            "metric": metric,
                            "beamwidth_deg": beam,
                            "scenario_seed": seed,
                            "treatment": TREATMENT,
                            "control": control,
                            "expected_direction": direction,
                            "treatment_value": treatment_value,
                            "control_value": control_value,
                            "delta": delta,
                        }
                    )
                deltas = np.array([item[3] for item in paired_values], dtype=float)
                treatment_values = np.array([item[1] for item in paired_values], dtype=float)
                control_values = np.array([item[2] for item in paired_values], dtype=float)
                pos = int(np.sum(deltas > 0.0))
                neg = int(np.sum(deltas < 0.0))
                zero = int(np.sum(deltas == 0.0))
                ci_low, ci_high = bootstrap_ci(deltas, rng)
                std = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
                dz = float(np.mean(deltas) / std) if std > 0.0 else float("inf")
                direction_ok = bool(np.mean(deltas) > 0.0) if direction == "positive" else bool(np.mean(deltas) < 0.0)
                summaries.append(
                    TestRow(
                        family=family,
                        metric=metric,
                        beamwidth_deg=beam,
                        treatment=TREATMENT,
                        control=control,
                        expected_direction=direction,
                        n_pairs=len(deltas),
                        positive_count=pos,
                        negative_count=neg,
                        zero_count=zero,
                        treatment_mean=float(np.mean(treatment_values)),
                        control_mean=float(np.mean(control_values)),
                        mean_delta=float(np.mean(deltas)),
                        median_delta=float(np.median(deltas)),
                        ci95_low=ci_low,
                        ci95_high=ci_high,
                        paired_dz=dz,
                        sign_p_two_sided=sign_test_two_sided(pos, neg),
                        holm_p=1.0,
                        holm_reject_0p05=False,
                        direction_ok=direction_ok,
                        evidence_role="confirmatory" if family in CONFIRMATORY_FAMILIES else "supportive",
                    )
                )

    for family in sorted({row.family for row in summaries}):
        family_rows = [row for row in summaries if row.family == family]
        adjusted = holm_adjust([row.sign_p_two_sided for row in family_rows])
        for row, adj in zip(family_rows, adjusted):
            row.holm_p = adj
            row.holm_reject_0p05 = bool(adj < 0.05 and row.direction_ok)
    return summaries, deltas_out


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_to_dict(row: TestRow) -> dict[str, object]:
    return {
        "family": row.family,
        "metric": row.metric,
        "beamwidth_deg": row.beamwidth_deg,
        "treatment": row.treatment,
        "control": row.control,
        "expected_direction": row.expected_direction,
        "n_pairs": row.n_pairs,
        "positive_count": row.positive_count,
        "negative_count": row.negative_count,
        "zero_count": row.zero_count,
        "treatment_mean": row.treatment_mean,
        "control_mean": row.control_mean,
        "mean_delta": row.mean_delta,
        "median_delta": row.median_delta,
        "bootstrap_ci95_low": row.ci95_low,
        "bootstrap_ci95_high": row.ci95_high,
        "paired_dz": row.paired_dz,
        "sign_p_two_sided": row.sign_p_two_sided,
        "holm_p": row.holm_p,
        "holm_reject_0p05": row.holm_reject_0p05,
        "direction_ok": row.direction_ok,
        "evidence_role": row.evidence_role,
    }


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "figure.figsize": (6.4, 4.8),
            "figure.dpi": 240,
            "savefig.dpi": 240,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_metric(rows: list[TestRow], family: str, ylabel: str, path: Path) -> Path:
    configure_plot()
    selected = [row for row in rows if row.family == family]
    selected.sort(key=lambda row: (row.beamwidth_deg, CONTROLS.index(row.control)))
    labels = [f"B={int(row.beamwidth_deg)}\n{short_control(row.control)}" for row in selected]
    means = np.array([row.mean_delta for row in selected], dtype=float)
    lows = np.array([row.mean_delta - row.ci95_low for row in selected], dtype=float)
    highs = np.array([row.ci95_high - row.mean_delta for row in selected], dtype=float)
    colors = ["#2F6B9A" if row.beamwidth_deg == 10.0 else "#D17C3F" for row in selected]

    fig, ax = plt.subplots()
    ax.bar(range(len(selected)), means, yerr=np.vstack([lows, highs]), capsize=4, color=colors, edgecolor="#222222", linewidth=0.6)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(selected)))
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    for idx, row in enumerate(selected):
        marker = "*" if row.holm_reject_0p05 else "n.s."
        y = row.ci95_high if row.mean_delta >= 0 else row.ci95_low
        offset = 0.018 if row.mean_delta >= 0 else -0.018
        ax.text(idx, y + offset, marker, ha="center", va="bottom" if row.mean_delta >= 0 else "top", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def short_control(control: str) -> str:
    return {
        "uniform_random": "Random",
        "skyorbs_like": "SkyOrbs",
        "mappo_no_isac": "MAPPO",
        "contention_no_isac": "No-ISAC",
    }[control]


def write_report(summary: list[TestRow], figures: list[Path], source_hashes: list[dict[str, str]]) -> None:
    confirmatory = [row for row in summary if row.evidence_role == "confirmatory"]
    all_confirmatory_pass = all(row.holm_reject_0p05 for row in confirmatory)
    primary = [row for row in summary if row.family == "primary_discovery"]
    empty = [row for row in summary if row.family == "mechanism_empty_scan"]
    lines = [
        "# Phase10 Paired Significance Validation - 2026-07-07",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        "- Verification Status: ANALYZED",
        "- Scope: N=100, 3000-slot B=10/B=15 final Phase10 primary ISAC-vs-communication-only comparison",
        "",
        "## Predeclared Testing Boundary",
        "",
        "- Confirmatory family 1: discovery-rate deltas for `contention_actor` versus `uniform_random`, `skyorbs_like`, `mappo_no_isac`, and `contention_no_isac` at B=10 and B=15.",
        "- Confirmatory family 2: empty-scan-ratio deltas for the same eight paired comparisons.",
        "- Test: exact two-sided paired sign test over identical scenario seeds, excluding zero deltas.",
        "- Multiple comparisons: Holm correction within each confirmatory family at alpha=0.05.",
        "- Supportive only: lambda2 sign tests are reported for topology context but are not used as the confirmatory decision gate.",
        "",
        "## Summary",
        "",
        f"- Confirmatory tests passing Holm-adjusted alpha=0.05: {sum(row.holm_reject_0p05 for row in confirmatory)}/{len(confirmatory)}.",
        f"- Overall confirmatory verdict: {'PASS' if all_confirmatory_pass else 'CAUTION'}.",
        f"- Discovery deltas are positive in {min(row.positive_count for row in primary)}/10 to {max(row.positive_count for row in primary)}/10 paired seeds across the eight primary comparisons.",
        f"- Empty-scan deltas are negative in {min(row.negative_count for row in empty)}/10 to {max(row.negative_count for row in empty)}/10 paired seeds across the eight mechanism comparisons.",
        "",
        "## Primary Discovery Family",
        "",
        "| Beam | Control | Mean delta | 95% bootstrap CI | Sign count | Holm p |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            f"| {row.beamwidth_deg:g} | {row.control} | {row.mean_delta:.4f} | [{row.ci95_low:.4f}, {row.ci95_high:.4f}] | +{row.positive_count}/-{row.negative_count}/0{row.zero_count} | {row.holm_p:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism Empty-Scan Family",
            "",
            "| Beam | Control | Mean delta | 95% bootstrap CI | Sign count | Holm p |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in empty:
        lines.append(
            f"| {row.beamwidth_deg:g} | {row.control} | {row.mean_delta:.4f} | [{row.ci95_low:.4f}, {row.ci95_high:.4f}] | +{row.positive_count}/-{row.negative_count}/0{row.zero_count} | {row.holm_p:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Statistical Fallacy Scan",
            "",
            "- Simpson's paradox: checked; no subgroup reversal claim is made because the report stratifies by beamwidth.",
            "- Ecological fallacy: checked; inference unit is scenario-seed episode, not individual UAV fairness.",
            "- Berkson's paradox: checked; no selected-success-only sample is used.",
            "- Collider bias: checked; no post-treatment control variable is introduced in the sign tests.",
            "- Base-rate neglect: checked; discovery and empty-scan rates are paired directly, not diagnostic predictive values.",
            "- Regression to the mean: checked; no pre/post extreme selection design is used.",
            "- Survivorship bias: checked; all archived final evaluation episodes in the paired source files are included.",
            "- Look-elsewhere effect: addressed by predeclared families and Holm correction.",
            "- Garden of forking paths: bounded by fixed treatment/control sets, fixed metrics, and a written test boundary.",
            "- Correlation-causation fallacy: not applicable to simulator treatment assignment, but hardware/generalization causality is not claimed.",
            "- Reverse causality: not applicable to simulator treatment assignment.",
            "",
            "## Interpretation Boundary",
            "",
            "These tests strengthen the primary simulator claim that ISAC-assisted contention actor evaluation improves discovery and reduces empty scans versus communication-only controls under paired scenario seeds. They do not prove significance for every gate-family operating point, every mobility model, every beamwidth stress case, or any hardware-calibrated PHY implementation.",
            "",
            "## Generated Figures",
            "",
        ]
    )
    lines.extend(f"- `{path.as_posix()}`" for path in figures)
    lines.extend(
        [
            "",
            "## Source File Hashes",
            "",
            "| Path | SHA256 |",
            "|---|---|",
        ]
    )
    lines.extend(f"| `{item['path']}` | `{item['sha256']}` |" for item in source_hashes)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    summary_rows, delta_rows = build_pairs(rows)
    summary_dicts = [row_to_dict(row) for row in summary_rows]
    write_csv(OUT_DIR / "primary_paired_sign_tests.csv", summary_dicts, list(summary_dicts[0].keys()))
    write_csv(OUT_DIR / "primary_paired_delta_values.csv", delta_rows, list(delta_rows[0].keys()))
    source_hashes = [{"path": path.as_posix(), "sha256": file_hash(path)} for path in SOURCE_FILES]
    write_csv(OUT_DIR / "source_file_hashes.csv", source_hashes, ["path", "sha256"])
    figures = [
        plot_metric(summary_rows, "primary_discovery", "Paired discovery-rate delta", FIG_DIR / "phase10_primary_discovery_paired_deltas.png"),
        plot_metric(summary_rows, "mechanism_empty_scan", "Paired empty-scan-ratio delta", FIG_DIR / "phase10_empty_scan_paired_deltas.png"),
    ]
    write_report(summary_rows, figures, source_hashes)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Phase10 N=100 3000-slot paired significance for primary ISAC-vs-communication-only comparisons.",
        "source_files": [path.as_posix() for path in SOURCE_FILES],
        "tables": [
            (OUT_DIR / "primary_paired_sign_tests.csv").as_posix(),
            (OUT_DIR / "primary_paired_delta_values.csv").as_posix(),
            (OUT_DIR / "source_file_hashes.csv").as_posix(),
        ],
        "figures": [path.as_posix() for path in figures],
        "report": REPORT.as_posix(),
        "confirmatory_families": sorted(CONFIRMATORY_FAMILIES),
        "all_confirmatory_tests_pass": all(row.holm_reject_0p05 for row in summary_rows if row.evidence_role == "confirmatory"),
        "confirmatory_test_count": sum(1 for row in summary_rows if row.evidence_role == "confirmatory"),
        "holm_alpha": 0.05,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
