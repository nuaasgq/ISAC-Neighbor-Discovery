from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "05_simulation" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isac_nd_sim.communication_phy import (  # noqa: E402
    close_in_path_loss_db,
    db_to_linear,
    main_lobe_gain_db,
    sample_rician_power,
    thermal_noise_power_w,
)
from isac_nd_sim.config import SimulationConfig, load_config  # noqa: E402
from isac_nd_sim.simulator import NeighborDiscoverySimulator  # noqa: E402


PARAMETER_SWEEPS: dict[str, tuple[float, ...]] = {
    "path_loss_exponent": (1.8, 2.0, 2.1, 2.2, 2.5, 3.0),
    "rician_k_db": (0.0, 5.0, 10.0, 13.3, 22.0),
    "shadowing_std_db": (0.0, 2.0, 4.0, 6.0, 8.0),
    "sidelobe_gain_db": (-30.0, -20.0, -10.0, -5.0, 0.0),
    "sinr_threshold_db": (-5.0, 0.0, 5.0, 10.0, 15.0),
    "tx_power_w": (0.1, 0.25, 0.5, 1.0, 2.0),
}

CONFIG_FIELDS = {
    "path_loss_exponent": "communication_path_loss_exponent",
    "rician_k_db": "communication_rician_k_db",
    "shadowing_std_db": "communication_shadowing_std_db",
    "sidelobe_gain_db": "communication_sidelobe_gain_db",
    "sinr_threshold_db": "communication_sinr_threshold_db",
    "tx_power_w": "communication_tx_power_w",
}

DISTANCES_M = (1000.0, 5000.0, 10000.0, 15000.0, 17320.508075688773)
JOINT_EXPONENTS = (2.0, 2.1, 2.2, 2.3, 2.4)
JOINT_THRESHOLDS_DB = (0.0, 2.5, 5.0, 7.5, 10.0)
JOINT_TX_POWERS_W = (0.25, 0.5, 1.0, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate communication-PHY sensitivity without protocol tuning.")
    parser.add_argument("--config", default="05_simulation/configs/twc_trainable_n10.yaml")
    parser.add_argument(
        "--output",
        default="05_simulation/results_raw/communication_phy_calibration_20260710",
    )
    parser.add_argument(
        "--paper-output",
        default="06_analysis/paper_tables/communication_phy_calibration_20260710",
    )
    parser.add_argument(
        "--figure-output",
        default="06_analysis/paper_figures/communication_phy_calibration_20260710",
    )
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--protocol-validation-episodes", type=int, default=5)
    parser.add_argument("--protocol-validation-slots", type=int, default=300)
    return parser.parse_args()


def stable_rng(seed: int, *parts: Any) -> np.random.Generator:
    text = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def sampled_link_power_w(
    cfg: SimulationConfig,
    distance_m: float,
    samples: int,
    rng: np.random.Generator,
    tx_gain_db: float,
    rx_gain_db: float,
) -> np.ndarray:
    base_loss_db = float(
        close_in_path_loss_db(
            distance_m,
            cfg.communication_carrier_frequency_hz,
            cfg.communication_path_loss_exponent,
            cfg.communication_reference_distance_m,
        )
    ) + float(cfg.communication_system_loss_db)
    if cfg.communication_shadowing_enabled:
        shadowing_db = rng.standard_normal(size=samples) * cfg.communication_shadowing_std_db
    else:
        shadowing_db = np.zeros(samples, dtype=float)
    if cfg.communication_fading_enabled:
        fading_power = sample_rician_power(rng, (samples,), cfg.communication_rician_k_db)
    else:
        fading_power = np.ones(samples, dtype=float)
    gain_linear = float(db_to_linear(tx_gain_db + rx_gain_db))
    return (
        cfg.communication_tx_power_w
        * gain_linear
        * fading_power
        * db_to_linear(-(base_loss_db + shadowing_db))
    )


def probability_with_wilson_interval(values: np.ndarray) -> tuple[float, float, float]:
    count = int(np.count_nonzero(values))
    total = int(values.size)
    if total <= 0:
        return 0.0, 0.0, 0.0
    rate = count / total
    z = 1.959963984540054
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return float(rate), float(max(0.0, center - radius)), float(min(1.0, center + radius))


def evaluate_operating_point(
    cfg: SimulationConfig,
    samples: int,
    seed: int,
    label: str,
) -> dict[str, float]:
    main_gain_db = main_lobe_gain_db(cfg)
    noise_w = thermal_noise_power_w(cfg)
    threshold = float(db_to_linear(cfg.communication_sinr_threshold_db))
    edge_distance = DISTANCES_M[-1]

    edge_power = sampled_link_power_w(
        cfg,
        edge_distance,
        samples,
        stable_rng(seed, label, "edge"),
        main_gain_db,
        main_gain_db,
    )
    edge_snr = edge_power / noise_w
    mid_power = sampled_link_power_w(
        cfg,
        10000.0,
        samples,
        stable_rng(seed, label, "mid"),
        main_gain_db,
        main_gain_db,
    )
    mid_snr = mid_power / noise_w

    equal_first = sampled_link_power_w(
        cfg,
        10000.0,
        samples,
        stable_rng(seed, label, "equal_1"),
        main_gain_db,
        main_gain_db,
    )
    equal_second = sampled_link_power_w(
        cfg,
        10000.0,
        samples,
        stable_rng(seed, label, "equal_2"),
        main_gain_db,
        main_gain_db,
    )
    equal_first_sinr = equal_first / (noise_w + equal_second)
    equal_second_sinr = equal_second / (noise_w + equal_first)
    equal_decode = np.maximum(equal_first_sinr, equal_second_sinr) >= threshold

    near_power = sampled_link_power_w(
        cfg,
        3000.0,
        samples,
        stable_rng(seed, label, "near"),
        main_gain_db,
        main_gain_db,
    )
    far_power = sampled_link_power_w(
        cfg,
        10000.0,
        samples,
        stable_rng(seed, label, "far"),
        main_gain_db,
        main_gain_db,
    )
    near_sinr = near_power / (noise_w + far_power)
    near_capture = (near_sinr >= threshold) & (near_power >= far_power)

    desired_power = sampled_link_power_w(
        cfg,
        10000.0,
        samples,
        stable_rng(seed, label, "desired"),
        main_gain_db,
        main_gain_db,
    )
    one_sided_interference = sampled_link_power_w(
        cfg,
        2000.0,
        samples,
        stable_rng(seed, label, "one_sided_interferer"),
        main_gain_db,
        cfg.communication_sidelobe_gain_db,
    )
    one_sided_sinr = desired_power / (noise_w + one_sided_interference)

    probability_samples = {
        "edge_coverage": edge_snr >= threshold,
        "mid_coverage": mid_snr >= threshold,
        "equal_power_decode": equal_decode,
        "near_far_capture": near_capture,
        "one_sided_interference_survival": one_sided_sinr >= threshold,
    }
    probability_metrics: dict[str, float] = {}
    for name, values in probability_samples.items():
        rate, low, high = probability_with_wilson_interval(values)
        probability_metrics[f"{name}_rate"] = rate
        probability_metrics[f"{name}_ci95_low"] = low
        probability_metrics[f"{name}_ci95_high"] = high
    return {
        **probability_metrics,
        "edge_snr_mean_db": float(10.0 * np.log10(np.maximum(edge_snr, 1e-300)).mean()),
        "edge_snr_p10_db": float(np.percentile(10.0 * np.log10(np.maximum(edge_snr, 1e-300)), 10)),
        "edge_snr_p50_db": float(np.percentile(10.0 * np.log10(np.maximum(edge_snr, 1e-300)), 50)),
    }


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.samples) <= 0:
        raise ValueError("--samples must be positive.")
    cfg = load_config(args.config)
    output = Path(args.output)
    paper_output = Path(args.paper_output)
    figure_output = Path(args.figure_output)
    for directory in (output, paper_output, figure_output):
        directory.mkdir(parents=True, exist_ok=True)

    baseline = evaluate_operating_point(cfg, args.samples, args.seed, "joint")
    oat_rows: list[dict[str, Any]] = []
    for parameter, values in PARAMETER_SWEEPS.items():
        field = CONFIG_FIELDS[parameter]
        for value in values:
            condition = replace(cfg, **{field: value})
            metrics = evaluate_operating_point(
                condition,
                args.samples,
                args.seed,
                f"oat:{parameter}",
            )
            oat_rows.append(
                {
                    "parameter": parameter,
                    "value": value,
                    "samples": int(args.samples),
                    **metrics,
                }
            )

    coverage_rows: list[dict[str, Any]] = []
    main_gain_db = main_lobe_gain_db(cfg)
    threshold = float(db_to_linear(cfg.communication_sinr_threshold_db))
    noise_w = thermal_noise_power_w(cfg)
    for distance_m in DISTANCES_M:
        power = sampled_link_power_w(
            cfg,
            distance_m,
            args.samples,
            stable_rng(args.seed, "coverage", distance_m),
            main_gain_db,
            main_gain_db,
        )
        snr = power / noise_w
        coverage_rate, coverage_low, coverage_high = probability_with_wilson_interval(snr >= threshold)
        coverage_rows.append(
            {
                "distance_m": distance_m,
                "coverage_rate": coverage_rate,
                "coverage_ci95_low": coverage_low,
                "coverage_ci95_high": coverage_high,
                "snr_mean_db": float(np.mean(10.0 * np.log10(np.maximum(snr, 1e-300)))),
                "snr_p10_db": float(np.percentile(10.0 * np.log10(np.maximum(snr, 1e-300)), 10)),
            }
        )

    joint_rows: list[dict[str, Any]] = []
    for exponent in JOINT_EXPONENTS:
        for threshold_db in JOINT_THRESHOLDS_DB:
            for tx_power_w in JOINT_TX_POWERS_W:
                condition = replace(
                    cfg,
                    communication_path_loss_exponent=exponent,
                    communication_sinr_threshold_db=threshold_db,
                    communication_tx_power_w=tx_power_w,
                )
                metrics = evaluate_operating_point(
                    condition,
                    args.samples,
                    args.seed,
                    "joint",
                )
                joint_rows.append(
                    {
                        "path_loss_exponent": exponent,
                        "sinr_threshold_db": threshold_db,
                        "tx_power_w": tx_power_w,
                        "samples": int(args.samples),
                        **metrics,
                    }
                )

    recommended_rows = rank_operating_points(joint_rows)
    recommended_profiles = select_recommended_profiles(recommended_rows)
    validation_rows, validation_summary = validate_profiles_in_protocol(
        cfg,
        recommended_profiles,
        episodes=int(getattr(args, "protocol_validation_episodes", 0)),
        slots=int(getattr(args, "protocol_validation_slots", 300)),
        seed=int(args.seed),
    )
    write_rows(output / "oat_metrics.csv", oat_rows)
    write_rows(output / "coverage_distance.csv", coverage_rows)
    write_rows(output / "joint_metrics.csv", joint_rows)
    write_rows(output / "recommended_operating_points.csv", recommended_rows)
    write_rows(output / "recommended_profiles.csv", recommended_profiles)
    write_rows(output / "protocol_validation_rows.csv", validation_rows)
    write_rows(output / "protocol_validation_summary.csv", validation_summary)
    for name in (
        "oat_metrics.csv",
        "coverage_distance.csv",
        "joint_metrics.csv",
        "recommended_operating_points.csv",
        "recommended_profiles.csv",
        "protocol_validation_rows.csv",
        "protocol_validation_summary.csv",
    ):
        shutil.copy2(output / name, paper_output / name)
    plot_outputs(oat_rows, coverage_rows, joint_rows, figure_output)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "communication_phy_sensitivity_calibration",
        "config": str(args.config),
        "samples_per_condition": int(args.samples),
        "seed": int(args.seed),
        "method": "vectorized_monte_carlo_common_random_number_design",
        "baseline": baseline,
        "parameter_sweeps": {key: list(values) for key, values in PARAMETER_SWEEPS.items()},
        "joint_sweep": {
            "path_loss_exponent": list(JOINT_EXPONENTS),
            "sinr_threshold_db": list(JOINT_THRESHOLDS_DB),
            "tx_power_w": list(JOINT_TX_POWERS_W),
        },
        "calibration_targets": {
            "edge_coverage_rate": [0.80, 0.98],
            "equal_power_decode_rate_max": 0.20,
            "near_far_capture_rate_min": 0.75,
        },
        "link_scenarios": {
            "edge_distance_m": DISTANCES_M[-1],
            "mid_distance_m": 10000.0,
            "equal_power_hello_distances_m": [10000.0, 10000.0],
            "near_far_hello_distances_m": [3000.0, 10000.0],
            "one_sided_interference_distances_m": {
                "desired": 10000.0,
                "interferer": 2000.0,
            },
        },
        "recommended": recommended_rows[:10],
        "recommended_profiles": recommended_profiles,
        "protocol_validation": {
            "protocol": "improved_rl_isac_tables",
            "episodes": int(getattr(args, "protocol_validation_episodes", 0)),
            "slots_per_episode": int(getattr(args, "protocol_validation_slots", 300)),
            "summary": validation_summary,
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "files": [
            "oat_metrics.csv",
            "coverage_distance.csv",
            "joint_metrics.csv",
            "recommended_operating_points.csv",
            "recommended_profiles.csv",
            "protocol_validation_rows.csv",
            "protocol_validation_summary.csv",
            "manifest.json",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (paper_output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def rank_operating_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for row in rows:
        edge = float(row["edge_coverage_rate"])
        equal = float(row["equal_power_decode_rate"])
        capture = float(row["near_far_capture_rate"])
        feasible = 0.80 <= edge <= 0.98 and equal <= 0.20 and capture >= 0.75
        target_distance = abs(edge - 0.90)
        score = 2.0 * target_distance + max(0.0, equal - 0.20) + max(0.0, 0.75 - capture)
        ranked.append({**row, "feasible": int(feasible), "calibration_score": score})
    return sorted(
        ranked,
        key=lambda row: (-int(row["feasible"]), float(row["calibration_score"]), -float(row["near_far_capture_rate"])),
    )


def select_recommended_profiles(ranked_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feasible = [row for row in ranked_rows if int(row["feasible"]) == 1]
    if not feasible:
        return []

    def nominal_distance(row: dict[str, Any]) -> float:
        return (
            2.0 * abs(float(row["path_loss_exponent"]) - 2.1)
            + abs(float(row["sinr_threshold_db"]) - 5.0) / 5.0
            + 0.5 * abs(float(np.log2(float(row["tx_power_w"]))))
        )

    selections = [
        ("nominal", min(feasible, key=lambda row: (nominal_distance(row), row["calibration_score"]))),
        ("balanced_stress", feasible[0]),
        ("low_power", min(feasible, key=lambda row: (row["tx_power_w"], row["calibration_score"]))),
        (
            "high_selectivity",
            min(feasible, key=lambda row: (row["equal_power_decode_rate"], row["calibration_score"])),
        ),
        (
            "highest_feasible_coverage",
            max(feasible, key=lambda row: (row["edge_coverage_rate"], -row["tx_power_w"])),
        ),
    ]
    return [{"profile": profile, **row} for profile, row in selections]


def validate_profiles_in_protocol(
    cfg: SimulationConfig,
    profiles: list[dict[str, Any]],
    episodes: int,
    slots: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if episodes <= 0 or not profiles:
        return [], []
    rows: list[dict[str, Any]] = []
    protocol = "improved_rl_isac_tables"
    for profile in profiles:
        profile_cfg = replace(
            cfg,
            slots_per_episode=slots,
            communication_path_loss_exponent=float(profile["path_loss_exponent"]),
            communication_sinr_threshold_db=float(profile["sinr_threshold_db"]),
            communication_tx_power_w=float(profile["tx_power_w"]),
        )
        for episode in range(episodes):
            scenario_seed = seed + 1009 * episode
            policy_seed = scenario_seed + sum((index + 1) * ord(char) for index, char in enumerate(protocol))
            simulator = NeighborDiscoverySimulator(
                profile_cfg,
                protocol,
                seed=policy_seed,
                scenario_seed=scenario_seed,
            )
            result = simulator.run_episode(episode).as_dict()
            attempts = int(result["handshake_attempts"])
            rows.append(
                {
                    "profile": profile["profile"],
                    "episode": episode,
                    "scenario_seed": scenario_seed,
                    "path_loss_exponent": profile["path_loss_exponent"],
                    "sinr_threshold_db": profile["sinr_threshold_db"],
                    "tx_power_w": profile["tx_power_w"],
                    "discovered_edges": result["discovered_edges"],
                    "discovery_rate": result["discovery_rate"],
                    "handshake_attempts": attempts,
                    "handshake_successes": result["handshake_successes"],
                    "handshake_success_rate": result["handshake_successes"] / max(1, attempts),
                    "interference_limited_failures": result["interference_limited_failures"],
                    "phy_outage_failures": result["phy_outage_failures"],
                    "mean_handshake_sinr_db": result["mean_handshake_sinr_db"],
                    "p10_handshake_sinr_db": result["p10_handshake_sinr_db"],
                }
            )
    summary: list[dict[str, Any]] = []
    for profile in profiles:
        selected = [row for row in rows if row["profile"] == profile["profile"]]
        summary.append(
            {
                "profile": profile["profile"],
                "episodes": len(selected),
                "path_loss_exponent": profile["path_loss_exponent"],
                "sinr_threshold_db": profile["sinr_threshold_db"],
                "tx_power_w": profile["tx_power_w"],
                "discovered_edges_mean": float(np.mean([row["discovered_edges"] for row in selected])),
                "discovery_rate_mean": float(np.mean([row["discovery_rate"] for row in selected])),
                "handshake_attempts_total": int(sum(row["handshake_attempts"] for row in selected)),
                "handshake_successes_total": int(sum(row["handshake_successes"] for row in selected)),
                "handshake_success_rate_pooled": sum(row["handshake_successes"] for row in selected)
                / max(1, sum(row["handshake_attempts"] for row in selected)),
                "interference_limited_failures_total": int(
                    sum(row["interference_limited_failures"] for row in selected)
                ),
                "phy_outage_failures_total": int(sum(row["phy_outage_failures"] for row in selected)),
            }
        )
    return rows, summary


def plot_outputs(
    oat_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    joint_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = ["#1f4e79", "#2e75b6", "#70ad47", "#c55a11"]
    metrics = (
        ("edge_coverage_rate", "Edge coverage"),
        ("equal_power_decode_rate", "Equal-power decode"),
        ("near_far_capture_rate", "Near-far capture"),
        ("one_sided_interference_survival_rate", "One-sided interference survival"),
    )
    labels = {
        "path_loss_exponent": "Path-loss exponent",
        "rician_k_db": "Rician K-factor (dB)",
        "shadowing_std_db": "Shadowing standard deviation (dB)",
        "sidelobe_gain_db": "Sidelobe gain (dBi)",
        "sinr_threshold_db": "SINR threshold (dB)",
        "tx_power_w": "Transmit power (W)",
    }
    for parameter in PARAMETER_SWEEPS:
        rows = sorted((row for row in oat_rows if row["parameter"] == parameter), key=lambda row: row["value"])
        fig, ax = plt.subplots(figsize=(8, 6))
        x = [float(row["value"]) for row in rows]
        for color, (metric, label) in zip(colors, metrics, strict=True):
            ax.plot(x, [float(row[metric]) for row in rows], marker="o", linewidth=2.0, color=color, label=label)
        ax.set_xlabel(labels[parameter])
        ax.set_ylabel("Probability")
        ax.set_ylim(-0.03, 1.03)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(output / f"sensitivity_{parameter}.png", dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        [float(row["distance_m"]) / 1000.0 for row in coverage_rows],
        [float(row["coverage_rate"]) for row in coverage_rows],
        marker="o",
        linewidth=2.2,
        color=colors[0],
    )
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Isolated-link coverage probability")
    ax.set_ylim(-0.03, 1.03)
    fig.tight_layout()
    fig.savefig(output / "coverage_vs_distance.png", dpi=220)
    plt.close(fig)

    selected_power = 1.0
    selected = [row for row in joint_rows if float(row["tx_power_w"]) == selected_power]
    matrix = np.full((len(JOINT_THRESHOLDS_DB), len(JOINT_EXPONENTS)), np.nan)
    for row in selected:
        y = JOINT_THRESHOLDS_DB.index(float(row["sinr_threshold_db"]))
        x = JOINT_EXPONENTS.index(float(row["path_loss_exponent"]))
        matrix[y, x] = float(row["edge_coverage_rate"])
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xticks(range(len(JOINT_EXPONENTS)), [str(value) for value in JOINT_EXPONENTS])
    ax.set_yticks(range(len(JOINT_THRESHOLDS_DB)), [str(value) for value in JOINT_THRESHOLDS_DB])
    ax.set_xlabel("Path-loss exponent")
    ax.set_ylabel("SINR threshold (dB)")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Edge coverage probability")
    fig.tight_layout()
    fig.savefig(output / "joint_exponent_threshold_edge_coverage_1w.png", dpi=220)
    plt.close(fig)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print(json.dumps(run_calibration(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
