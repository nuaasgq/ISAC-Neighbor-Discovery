from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SIMULATION = ROOT / "05_simulation"
SOURCE = SIMULATION / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

DEFAULT_RUN_ROOT = (
    SIMULATION / "results_raw" / "n2_b8_isac_measurement_aux_pilot_3seed"
)
DEFAULT_TABLE_DIR = (
    ROOT / "06_analysis" / "paper_tables" / "n2_b8_isac_measurement_aux_pilot_10k_20260712"
)
DEFAULT_FIGURE_DIR = (
    ROOT / "06_analysis" / "paper_figures" / "n2_b8_isac_measurement_aux_pilot_10k_20260712"
)
METHODS = (
    "learned_beam_no_isac",
    "learned_beam_direct_isac",
    "learned_beam_direct_isac_measurement_aux",
    "learned_beam_residual_isac_measurement_aux",
)
METHOD_LABELS = {
    "learned_beam_no_isac": "MARL, no ISAC",
    "learned_beam_direct_isac": "MARL + direct ISAC",
    "learned_beam_direct_isac_measurement_aux": "Direct ISAC + auxiliary",
    "learned_beam_residual_isac_measurement_aux": "Residual ISAC + auxiliary",
}
METHOD_COLORS = {
    "learned_beam_no_isac": "#5B6573",
    "learned_beam_direct_isac": "#2878B5",
    "learned_beam_direct_isac_measurement_aux": "#C43C39",
    "learned_beam_residual_isac_measurement_aux": "#3A923A",
}
EXPECTED_METHOD_CONTRACT = {
    "learned_beam_no_isac": ("none", 0.0, "structured_marl_no_isac"),
    "learned_beam_direct_isac": ("direct", 0.0, "improved_rl_isac_tables"),
    "learned_beam_direct_isac_measurement_aux": (
        "direct",
        0.1,
        "improved_rl_isac_tables",
    ),
    "learned_beam_residual_isac_measurement_aux": (
        "residual",
        0.1,
        "improved_rl_isac_tables",
    ),
}
T_95_DF2 = 4.302652729749


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and plot the N=2, B=8 local measurement auxiliary gate."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--probe-scenarios", type=int, default=32)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    return float(value) if value not in (None, "") else 0.0


def ratio_of_sums(rows: Iterable[dict[str, str]], numerator: str, denominator: str) -> float:
    rows = list(rows)
    denominator_sum = sum(number(row, denominator) for row in rows)
    if denominator_sum <= 0.0:
        return 0.0
    return sum(number(row, numerator) for row in rows) / denominator_sum


def seed_metric_row(method: str, seed: int, rows: list[dict[str, str]]) -> dict[str, Any]:
    stochastic = [row for row in rows if row.get("phase") == "eval_stochastic"]
    if len(stochastic) != 100:
        raise ValueError(f"{method}/seed_{seed}: expected 100 stochastic evaluations.")
    tx = sum(number(row, "tx_actions") for row in stochastic)
    rx = sum(number(row, "rx_actions") for row in stochastic)
    return {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "training_seed": seed,
        "eval_episodes": len(stochastic),
        "discovery_rate": np.mean([number(row, "discovery_rate") for row in stochastic]),
        "mean_delay_censored_slots": np.mean(
            [number(row, "mean_delay_censored") for row in stochastic]
        ),
        "bilateral_alignment_per_active_pair_slot": ratio_of_sums(
            stochastic, "bilaterally_aligned_pair_slots", "active_undiscovered_pair_slots"
        ),
        "aligned_opportunity_per_bilateral_alignment": ratio_of_sums(
            stochastic, "aligned_handshake_opportunities", "bilaterally_aligned_pair_slots"
        ),
        "success_per_aligned_opportunity": ratio_of_sums(
            stochastic, "handshake_successes", "aligned_handshake_opportunities"
        ),
        "tx_ratio": tx / (tx + rx),
    }


def summarize_seed_metrics(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metrics = (
        "discovery_rate",
        "mean_delay_censored_slots",
        "bilateral_alignment_per_active_pair_slot",
        "aligned_opportunity_per_bilateral_alignment",
        "success_per_aligned_opportunity",
        "tx_ratio",
    )
    for method in METHODS:
        group = [row for row in seed_rows if row["method"] == method]
        if len(group) != 3:
            raise ValueError(f"{method}: expected three independent training seeds.")
        row: dict[str, Any] = {"method": method, "method_label": METHOD_LABELS[method]}
        for metric in metrics:
            values = np.asarray([float(item[metric]) for item in group], dtype=float)
            mean_value = float(values.mean())
            sd = float(values.std(ddof=1))
            half_width = T_95_DF2 * sd / math.sqrt(len(values))
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci95_low"] = mean_value - half_width
            row[f"{metric}_ci95_high"] = mean_value + half_width
        output.append(row)
    return output


def paired_seed_deltas(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row["method"], row["training_seed"]): row for row in seed_rows}
    proposed = "learned_beam_direct_isac_measurement_aux"
    comparisons = (
        "learned_beam_no_isac",
        "learned_beam_direct_isac",
        "learned_beam_residual_isac_measurement_aux",
    )
    output: list[dict[str, Any]] = []
    for baseline in comparisons:
        for seed in sorted({int(row["training_seed"]) for row in seed_rows}):
            lhs = indexed[(proposed, seed)]
            rhs = indexed[(baseline, seed)]
            output.append(
                {
                    "proposed": proposed,
                    "baseline": baseline,
                    "training_seed": seed,
                    "discovery_rate_delta": lhs["discovery_rate"] - rhs["discovery_rate"],
                    "bilateral_alignment_delta": (
                        lhs["bilateral_alignment_per_active_pair_slot"]
                        - rhs["bilateral_alignment_per_active_pair_slot"]
                    ),
                    "delay_delta_slots": (
                        lhs["mean_delay_censored_slots"] - rhs["mean_delay_censored_slots"]
                    ),
                }
            )
    return output


def training_block_rows(
    method: str, seed: int, rows: list[dict[str, str]], block_size: int = 125
) -> list[dict[str, Any]]:
    if len(rows) != 625:
        raise ValueError(f"{method}/seed_{seed}: expected 625 training episodes.")
    output: list[dict[str, Any]] = []
    for start in range(0, len(rows), block_size):
        block = rows[start : start + block_size]
        tx = sum(number(row, "tx_actions") for row in block)
        rx = sum(number(row, "rx_actions") for row in block)
        output.append(
            {
                "method": method,
                "training_seed": seed,
                "episode_start": start + 1,
                "episode_end": start + len(block),
                "training_step_end": int(number(block[-1], "training_step")),
                "discovery_rate": np.mean([number(row, "discovery_rate") for row in block]),
                "bilateral_alignment_per_active_pair_slot": ratio_of_sums(
                    block, "bilaterally_aligned_pair_slots", "active_undiscovered_pair_slots"
                ),
                "aligned_opportunity_per_bilateral_alignment": ratio_of_sums(
                    block, "aligned_handshake_opportunities", "bilaterally_aligned_pair_slots"
                ),
                "tx_ratio": tx / (tx + rx),
                "episode_return_mean_per_agent": np.mean(
                    [number(row, "episode_return_mean_per_agent") for row in block]
                ),
                "measurement_prediction_aux_loss": np.mean(
                    [number(row, "measurement_prediction_aux_loss") for row in block]
                ),
                "measurement_prediction_sample_count": np.mean(
                    [number(row, "measurement_prediction_sample_count") for row in block]
                ),
                "max_rollout_replay_logprob_error": max(
                    number(row, "rollout_replay_logprob_max_abs_error") for row in block
                ),
            }
        )
    return output


def contract_audit(run_root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    eval_seed_sequences: dict[tuple[str, int], list[int]] = {}

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    seeds: set[int] = set()
    for method in METHODS:
        feature_set, aux_coef, protocol = EXPECTED_METHOD_CONTRACT[method]
        method_dirs = sorted((run_root / method).glob("seed_*"))
        check(f"{method}:three_seeds", len(method_dirs) == 3, f"found={len(method_dirs)}")
        for output in method_dirs:
            seed = int(output.name.removeprefix("seed_"))
            seeds.add(seed)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            eval_rows = read_csv(output / "eval_episode_metrics.csv")
            train_rows = read_csv(output / "episode_metrics.csv")
            stochastic = [row for row in eval_rows if row.get("phase") == "eval_stochastic"]
            eval_seed_sequences[(method, seed)] = [int(number(row, "scenario_seed")) for row in stochastic]
            prefix = f"{method}/seed_{seed}"
            common_expectations = {
                "node_count": 2,
                "beam_count": 8,
                "slots_per_episode": 16,
                "action_contract": "joint_role_beam",
                "network": "recurrent_contention_shared",
                "role_factorization": "beam_conditioned_antisymmetric",
                "decoupled_role_tower": True,
                "clean_ctde": True,
                "actor_global_state_access": False,
                "pair_derived_action_guidance_enabled": False,
                "rendezvous_observation_enabled": False,
                "single_rf_chain": True,
                "allow_standalone_sense": False,
                "expert_bc_weight": 0.0,
                "beam_isac_feedback_coef": 0.0,
            }
            for field, expected in common_expectations.items():
                check(
                    f"{prefix}:{field}",
                    manifest.get(field) == expected,
                    f"observed={manifest.get(field)!r}; expected={expected!r}",
                )
            check(
                f"{prefix}:measurement_feature_set",
                manifest.get("measurement_feature_set") == feature_set,
                f"observed={manifest.get('measurement_feature_set')!r}; expected={feature_set!r}",
            )
            check(
                f"{prefix}:aux_coefficient",
                float(manifest["measurement_prediction_aux"]["coefficient"]) == aux_coef,
                f"observed={manifest['measurement_prediction_aux']['coefficient']}; expected={aux_coef}",
            )
            check(
                f"{prefix}:protocol",
                manifest.get("env_protocol") == protocol,
                f"observed={manifest.get('env_protocol')!r}; expected={protocol!r}",
            )
            check(
                f"{prefix}:feature_flags_disabled",
                not any(bool(value) for value in manifest["feature_flags"].values()),
                json.dumps(manifest["feature_flags"], sort_keys=True),
            )
            check(
                f"{prefix}:uniform_exploration",
                float(manifest["stochastic_support"]["beam_uniform_mixture"]) == 0.1,
                json.dumps(manifest["stochastic_support"], sort_keys=True),
            )
            check(f"{prefix}:train_rows", len(train_rows) == 625, f"found={len(train_rows)}")
            check(f"{prefix}:eval_rows", len(eval_rows) == 200, f"found={len(eval_rows)}")
            check(
                f"{prefix}:replay_error",
                max(number(row, "rollout_replay_logprob_max_abs_error") for row in train_rows) == 0.0,
                "maximum must be zero",
            )
    check("matrix:three_common_seeds", len(seeds) == 3, f"seeds={sorted(seeds)}")
    for seed in sorted(seeds):
        reference = eval_seed_sequences[(METHODS[0], seed)]
        for method in METHODS[1:]:
            check(
                f"matrix:paired_eval_scenarios:{method}:seed_{seed}",
                eval_seed_sequences[(method, seed)] == reference,
                "stochastic scenario-seed sequence equals no-ISAC control",
            )
    return {
        "passed": all(item["passed"] for item in checks),
        "check_count": len(checks),
        "failed_count": sum(not item["passed"] for item in checks),
        "checks": checks,
    }


def clone_observation(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.copy() if hasattr(value, "copy") else copy.deepcopy(value)
        for key, value in observation.items()
    }


def set_measurement(observation: dict[str, Any], beam: int, occupied: bool) -> dict[str, Any]:
    changed = clone_observation(observation)
    for field in (
        "beam_target_count",
        "beam_target_count_variance",
        "beam_measurement_confidence",
        "beam_interaction_count",
        "beam_residual_target_count",
    ):
        changed[field][:] = 0.0
    changed["beam_age"][:] = 1.0
    changed["beam_target_count"][beam] = 1.0 if occupied else 0.0
    changed["beam_residual_target_count"][beam] = 1.0 if occupied else 0.0
    changed["beam_measurement_confidence"][beam] = 1.0
    changed["beam_age"][beam] = 0.0
    return changed


def load_policy(checkpoint_path: Path):
    import torch

    from isac_nd_sim.neural_recurrent_contention_actor_critic import (
        RecurrentContentionGraphActorCritic,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    feature_set = str(args.get("measurement_feature_set", "none"))
    policy = RecurrentContentionGraphActorCritic(
        8,
        hidden_dim=int(args.get("hidden_dim", 64)),
        device="cpu",
        use_candidate_mask=False,
        use_candidate_score=False,
        use_topology_deficit=False,
        use_rule_residual=False,
        measurement_feature_set=feature_set,
        use_measurement_prediction_head=float(args.get("measurement_prediction_aux_coef", 0.0)) > 0.0,
        role_probability_floor=float(args.get("role_probability_floor", 0.0)),
        beam_uniform_mixture=float(args.get("beam_uniform_mixture", 0.0)),
        disabled_modes=("sense", "idle"),
        action_contract=str(args.get("action_contract", "joint_role_beam")),
        azimuth_cells=8,
        elevation_cells=1,
        use_candidate_score_prior=False,
        use_bounded_score_residual=False,
        use_decoupled_role_tower=bool(args.get("decoupled_role_tower", False)),
        role_factorization=str(args.get("role_factorization", "independent")),
    )
    policy.model.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return policy


def evidence_response_probe(run_root: Path, scenarios: int) -> list[dict[str, Any]]:
    import torch

    from isac_nd_sim.config import load_config
    from isac_nd_sim.marl_env import MarlNeighborDiscoveryEnv

    cfg = load_config(SIMULATION / "configs" / "sanity_planar_n2_b45_ideal.yaml")
    output: list[dict[str, Any]] = []
    for method in METHODS:
        for seed_dir in sorted((run_root / method).glob("seed_*")):
            seed = int(seed_dir.name.removeprefix("seed_"))
            policy = load_policy(seed_dir / "final_model.pt")
            occupied_action: list[float] = []
            empty_action: list[float] = []
            occupied_head: list[float] = []
            empty_head: list[float] = []
            env_protocol = EXPECTED_METHOD_CONTRACT[method][2]
            env = MarlNeighborDiscoveryEnv(cfg, protocol=env_protocol)
            for scenario in range(scenarios):
                observations, _info = env.reset(seed=73000000 + scenario)
                for observation in observations:
                    for beam in range(8):
                        occupied = set_measurement(observation, beam, True)
                        empty = set_measurement(observation, beam, False)
                        with torch.no_grad():
                            _mode, occupied_logits, _value = policy.logits_value(occupied)
                            _mode, empty_logits, _value = policy.logits_value(empty)
                            occupied_action.append(
                                float(policy._beam_probabilities(occupied_logits)[beam])
                            )
                            empty_action.append(float(policy._beam_probabilities(empty_logits)[beam]))
                            if policy.use_measurement_prediction_head:
                                occupied_head.append(
                                    float(
                                        torch.sigmoid(
                                            policy.measurement_occupancy_logits([occupied])[0, beam]
                                        )
                                    )
                                )
                                empty_head.append(
                                    float(
                                        torch.sigmoid(
                                            policy.measurement_occupancy_logits([empty])[0, beam]
                                        )
                                    )
                                )
            output.append(
                {
                    "method": method,
                    "training_seed": seed,
                    "probe_samples_per_condition": len(occupied_action),
                    "occupied_beam_action_probability": np.mean(occupied_action),
                    "empty_beam_action_probability": np.mean(empty_action),
                    "action_probability_contrast": np.mean(occupied_action) - np.mean(empty_action),
                    "occupied_head_probability": (
                        np.mean(occupied_head) if occupied_head else ""
                    ),
                    "empty_head_probability": np.mean(empty_head) if empty_head else "",
                    "head_probability_contrast": (
                        np.mean(occupied_head) - np.mean(empty_head) if occupied_head else ""
                    ),
                }
            )
    return output


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    result = np.full(values.shape, np.nan, dtype=float)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    result[window - 1 :] = (cumulative[window:] - cumulative[:-window]) / window
    return result


def plot_training_curves(training: dict[tuple[str, int], list[dict[str, str]]], figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "savefig.dpi": 300,
        }
    )
    metric_specs = {
        "episode_return_mean_per_agent": ("Mean episode return per agent", "training_return.png"),
        "discovery_rate": ("Discovery rate", "training_discovery_rate.png"),
        "bilateral_ratio": ("Bilateral alignment per active pair-slot", "training_bilateral_alignment.png"),
        "opportunity_ratio": ("Opportunity per bilateral alignment", "training_role_conversion.png"),
        "tx_ratio": ("TX action ratio", "training_tx_ratio.png"),
        "measurement_prediction_aux_loss": (
            "Measurement prediction auxiliary loss",
            "training_measurement_aux_loss.png",
        ),
    }
    window = 50
    for metric, (ylabel, filename) in metric_specs.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        for method in METHODS:
            seed_curves: list[np.ndarray] = []
            steps: np.ndarray | None = None
            for (candidate_method, _seed), rows in training.items():
                if candidate_method != method:
                    continue
                steps = np.asarray([number(row, "training_step") for row in rows], dtype=float)
                if metric == "bilateral_ratio":
                    values = np.asarray(
                        [
                            number(row, "bilaterally_aligned_pair_slots")
                            / max(1.0, number(row, "active_undiscovered_pair_slots"))
                            for row in rows
                        ]
                    )
                elif metric == "opportunity_ratio":
                    values = np.asarray(
                        [
                            number(row, "aligned_handshake_opportunities")
                            / max(1.0, number(row, "bilaterally_aligned_pair_slots"))
                            for row in rows
                        ]
                    )
                elif metric == "tx_ratio":
                    values = np.asarray(
                        [
                            number(row, "tx_actions")
                            / max(1.0, number(row, "tx_actions") + number(row, "rx_actions"))
                            for row in rows
                        ]
                    )
                else:
                    values = np.asarray([number(row, metric) for row in rows], dtype=float)
                curve = rolling_mean(values, window)
                seed_curves.append(curve)
                ax.plot(steps, curve, color=METHOD_COLORS[method], alpha=0.18, linewidth=0.9)
            if steps is None:
                continue
            matrix = np.asarray(seed_curves)
            valid = np.arange(len(steps)) >= window - 1
            ax.plot(
                steps[valid],
                np.mean(matrix[:, valid], axis=0),
                color=METHOD_COLORS[method],
                linewidth=2.2,
                label=METHOD_LABELS[method],
            )
        ax.set_xlabel("Environment step")
        ax.set_ylabel(ylabel)
        ax.set_xlim(window * 16, 10000)
        if metric in {"discovery_rate", "bilateral_ratio", "opportunity_ratio", "tx_ratio"}:
            ax.set_ylim(0.0, 1.0)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(figure_dir / filename, bbox_inches="tight")
        plt.close(fig)


def gate_decision(
    seed_rows: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, Any]:
    no_isac = {
        int(row["training_seed"]): row
        for row in seed_rows
        if row["method"] == "learned_beam_no_isac"
    }
    def arm_criteria(method: str) -> dict[str, bool]:
        arm = [row for row in seed_rows if row["method"] == method]
        arm_probe = [row for row in probe_rows if row["method"] == method]
        arm_blocks = [row for row in blocks if row["method"] == method]
        final_by_seed = {
            int(row["training_seed"]): row
            for row in arm_blocks
            if int(row["episode_end"]) == 625
        }
        penultimate_by_seed = {
            int(row["training_seed"]): row
            for row in arm_blocks
            if int(row["episode_end"]) == 500
        }
        return {
            "action_contrast_positive_all_seeds": all(
                float(row["action_probability_contrast"]) > 0.0 for row in arm_probe
            ),
            "action_contrast_above_10pp_all_seeds": all(
                float(row["action_probability_contrast"]) > 0.10 for row in arm_probe
            ),
            "ba_above_no_isac_all_seeds": all(
                float(row["bilateral_alignment_per_active_pair_slot"])
                > float(no_isac[int(row["training_seed"])]["bilateral_alignment_per_active_pair_slot"])
                for row in arm
            ),
            "discovery_above_no_isac_all_seeds": all(
                float(row["discovery_rate"])
                > float(no_isac[int(row["training_seed"])]["discovery_rate"])
                for row in arm
            ),
            "tx_ratio_35_to_65_percent_all_seeds": all(
                0.35 <= float(row["tx_ratio"]) <= 0.65 for row in arm
            ),
            "final_training_discovery_at_least_60_percent_all_seeds": all(
                float(row["discovery_rate"]) >= 0.60 for row in final_by_seed.values()
            ),
            "final_training_not_down_more_than_5pp_all_seeds": all(
                float(final_by_seed[seed]["discovery_rate"])
                >= float(penultimate_by_seed[seed]["discovery_rate"]) - 0.05
                for seed in final_by_seed
            ),
            "replay_error_zero_all_final_blocks": all(
                float(row["max_rollout_replay_logprob_error"]) == 0.0
                for row in final_by_seed.values()
            ),
        }

    direct_criteria = arm_criteria("learned_beam_direct_isac_measurement_aux")
    residual_criteria = arm_criteria("learned_beam_residual_isac_measurement_aux")
    common_criteria = {
        "contract_audit_passed": bool(audit["passed"]),
    }
    primary_passed = all(common_criteria.values()) and all(direct_criteria.values())
    full_matrix_passed = primary_passed and all(residual_criteria.values())
    return {
        "primary_three_arm_formal_gate_passed": primary_passed,
        "full_four_arm_formal_gate_passed": full_matrix_passed,
        "common_criteria": common_criteria,
        "direct_auxiliary_criteria": direct_criteria,
        "residual_auxiliary_criteria": residual_criteria,
    }


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    table_dir = args.table_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    audit = contract_audit(run_root)
    seed_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    training: dict[tuple[str, int], list[dict[str, str]]] = {}
    for method in METHODS:
        for seed_dir in sorted((run_root / method).glob("seed_*")):
            seed = int(seed_dir.name.removeprefix("seed_"))
            eval_rows = read_csv(seed_dir / "eval_episode_metrics.csv")
            train_rows = read_csv(seed_dir / "episode_metrics.csv")
            seed_rows.append(seed_metric_row(method, seed, eval_rows))
            block_rows.extend(training_block_rows(method, seed, train_rows))
            training[(method, seed)] = train_rows

    aggregate_rows = summarize_seed_metrics(seed_rows)
    delta_rows = paired_seed_deltas(seed_rows)
    probe_rows = evidence_response_probe(run_root, int(args.probe_scenarios))
    decision = gate_decision(seed_rows, block_rows, probe_rows, audit)

    write_csv(table_dir / "eval_by_training_seed.csv", seed_rows)
    write_csv(table_dir / "eval_aggregate_across_training_seeds.csv", aggregate_rows)
    write_csv(table_dir / "paired_training_seed_deltas.csv", delta_rows)
    write_csv(table_dir / "training_125_episode_blocks.csv", block_rows)
    write_csv(table_dir / "evidence_response_probe.csv", probe_rows)
    (table_dir / "contract_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (table_dir / "formal_gate_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_training_curves(training, figure_dir)
    manifest = {
        "raw_run": str(run_root),
        "table_dir": str(table_dir),
        "figure_dir": str(figure_dir),
        "evaluation_phase": "eval_stochastic",
        "inference_unit": "independent_training_seed",
        "training_seeds": sorted({int(row["training_seed"]) for row in seed_rows}),
        "probe_scenarios": int(args.probe_scenarios),
        "probe_information_source": "synthetic_actor_visible_local_anonymous_measurements_only",
        "primary_three_arm_formal_gate_passed": decision[
            "primary_three_arm_formal_gate_passed"
        ],
        "full_four_arm_formal_gate_passed": decision["full_four_arm_formal_gate_passed"],
    }
    (table_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"audit": audit["passed"], "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
