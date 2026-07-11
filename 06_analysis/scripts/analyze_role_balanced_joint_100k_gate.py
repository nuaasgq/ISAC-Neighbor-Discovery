from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    ROOT
    / "05_simulation"
    / "results_raw"
    / "role_balanced_joint_100200_three_seed_final_20260711"
)
OUTPUT = (
    ROOT
    / "06_analysis"
    / "paper_tables"
    / "role_balanced_joint_100k_gate_20260712"
)
REPORT = ROOT / "06_analysis" / "role_balanced_joint_100k_gate_20260712.md"
SEEDS = (29260711, 29261711, 29262711)
ARMS = {
    "A": "Rule beam + random role",
    "B": "Learned beam + random role",
    "C": "Rule beam + learned role",
    "D": "Learned beam + learned role",
}
METRICS = (
    "discovery_rate",
    "mean_delay_censored",
    "neighbor_knowledge_recall",
    "aligned_handshake_opportunities",
    "step_reward_mean",
)
COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    fieldnames.extend(
        key for row in rows[1:] for key in row if key not in fieldnames
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.png")
    fig.savefig(OUTPUT / f"{stem}.pdf")
    plt.close(fig)


def rolling_mean(values: np.ndarray, width: int = 20) -> np.ndarray:
    result = np.full(values.shape, np.nan)
    if values.size >= width:
        result[width - 1 :] = np.convolve(values, np.ones(width) / width, mode="valid")
    return result


def validate_training(seed: int, rows: list[dict[str, str]]) -> None:
    if len(rows) != 334 or int(rows[-1]["training_step"]) != 100_200:
        raise ValueError(f"seed{seed}: expected 334 episodes and 100,200 steps")
    replay_error = max(float(row["rollout_replay_logprob_max_abs_error"]) for row in rows)
    if replay_error > 1e-7:
        raise ValueError(f"seed{seed}: replay error is {replay_error}")
    manifest = RUN_ROOT / f"seed{seed}" / "train" / "manifest.json"
    text = manifest.read_text(encoding="utf-8")
    required = (
        '"architecture_version": "joint_decoupled_role_recurrent_beam_mpnn_score_residual_v4"',
        '"reward_event_source": "dedicated_per_node_handshake_counters_v2"',
        '"coefficient": 0.01',
        '"execution_global_information": false',
        '"actor_global_state_access": false',
        '"tracked_worktree_dirty": false',
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise ValueError(f"seed{seed}: manifest contract mismatch: {missing}")


def load_data() -> tuple[dict[int, list[dict[str, str]]], list[dict[str, object]]]:
    training: dict[int, list[dict[str, str]]] = {}
    evaluation: list[dict[str, object]] = []
    for seed in SEEDS:
        rows = read_rows(RUN_ROOT / f"seed{seed}" / "train" / "episode_metrics.csv")
        validate_training(seed, rows)
        training[seed] = rows
        for arm in ARMS:
            eval_rows = read_rows(
                RUN_ROOT / f"seed{seed}" / f"eval_{arm}_dev20" / "eval_episode_metrics.csv"
            )
            if len(eval_rows) != 20:
                raise ValueError(f"seed{seed}/{arm}: expected 20 evaluation episodes")
            scenario_seeds = [int(row["scenario_seed"]) for row in eval_rows]
            if len(set(scenario_seeds)) != 20:
                raise ValueError(f"seed{seed}/{arm}: scenario seeds are not unique")
            for row in eval_rows:
                item: dict[str, object] = {
                    "training_seed": seed,
                    "arm": arm,
                    "scenario_seed": int(row["scenario_seed"]),
                    "tx_ratio": float(row["tx_actions"])
                    / (float(row["slots"]) * 10.0),
                }
                item.update({metric: float(row[metric]) for metric in METRICS})
                evaluation.append(item)
    reference = {}
    for seed in SEEDS:
        for arm in ARMS:
            reference[(seed, arm)] = {
                int(row["scenario_seed"])
                for row in evaluation
                if row["training_seed"] == seed and row["arm"] == arm
            }
    expected = set(range(33260711, 33260731))
    for key, scenario_seeds in reference.items():
        if scenario_seeds != expected:
            raise ValueError(f"{key}: common-random-number scenario set mismatch")
    return training, evaluation


def summarize(evaluation: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_seed: list[dict[str, object]] = []
    for seed in SEEDS:
        for arm in ARMS:
            rows = [
                row
                for row in evaluation
                if row["training_seed"] == seed and row["arm"] == arm
            ]
            summary: dict[str, object] = {
                "training_seed": seed,
                "arm": arm,
                "method": ARMS[arm],
                "evaluation_episodes": len(rows),
            }
            for metric in (*METRICS, "tx_ratio"):
                values = np.asarray([float(row[metric]) for row in rows])
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_sd_across_scenarios"] = float(values.std(ddof=1))
            per_seed.append(summary)

    aggregate: list[dict[str, object]] = []
    for arm in ARMS:
        selected = [row for row in per_seed if row["arm"] == arm]
        item: dict[str, object] = {"arm": arm, "method": ARMS[arm], "training_seeds": 3}
        for metric in (*METRICS, "tx_ratio"):
            values = np.asarray([float(row[f"{metric}_mean"]) for row in selected])
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_sd_across_training_seeds"] = float(values.std(ddof=1))
            item[f"{metric}_ci95_half_width"] = 4.303 * float(values.std(ddof=1)) / np.sqrt(3)
        aggregate.append(item)
    return per_seed, aggregate


def build_contrasts(per_seed: list[dict[str, object]]) -> list[dict[str, object]]:
    definitions = {
        "B-A (learned beam effect, random role)": ("B", "A"),
        "D-C (learned beam effect, learned role)": ("D", "C"),
        "C-A (learned role effect, rule beam)": ("C", "A"),
        "D-B (learned role effect, learned beam)": ("D", "B"),
        "D-B-C+A (interaction)": ("interaction", "interaction"),
    }
    output: list[dict[str, object]] = []
    for seed in SEEDS:
        values = {
            str(row["arm"]): float(row["discovery_rate_mean"])
            for row in per_seed
            if row["training_seed"] == seed
        }
        for name, (high, low) in definitions.items():
            delta = (
                values["D"] - values["B"] - values["C"] + values["A"]
                if high == "interaction"
                else values[high] - values[low]
            )
            output.append(
                {"training_seed": seed, "contrast": name, "discovery_rate_delta": delta}
            )
    for name in definitions:
        values = np.asarray(
            [float(row["discovery_rate_delta"]) for row in output if row["contrast"] == name]
        )
        output.append(
            {
                "training_seed": "mean",
                "contrast": name,
                "discovery_rate_delta": float(values.mean()),
                "sd_across_training_seeds": float(values.std(ddof=1)),
                "ci95_half_width": 4.303 * float(values.std(ddof=1)) / np.sqrt(3),
                "positive_training_seeds": int(np.sum(values > 0)),
            }
        )
    return output


def plot_training(training: dict[int, list[dict[str, str]]]) -> None:
    configure_plot()
    steps = np.asarray([float(row["training_step"]) for row in training[SEEDS[0]]])
    specifications = (
        ("discovery_rate", 100.0, "Discovery rate (%)", "training_discovery_vs_step", None),
        ("episode_return_mean_per_agent", 1.0, "Episode return per UAV", "training_return_vs_step", None),
        ("mean_policy_tx_probability", 100.0, "Mean policy TX probability (%)", "training_tx_probability_vs_step", 50.0),
    )
    for metric, scale, ylabel, stem, reference in specifications:
        curves = np.vstack(
            [rolling_mean(np.asarray([float(row[metric]) for row in training[seed]])) * scale for seed in SEEDS]
        )
        valid = ~np.all(np.isnan(curves), axis=0)
        plot_steps = steps[valid]
        curves = curves[:, valid]
        fig, ax = plt.subplots(figsize=(4, 3))
        for curve, seed, color in zip(curves, SEEDS, COLORS):
            ax.plot(plot_steps, curve, color=color, alpha=0.35, linewidth=0.8, label=f"Seed {seed}")
        mean = np.nanmean(curves, axis=0)
        sd = np.nanstd(curves, axis=0, ddof=1)
        ax.plot(plot_steps, mean, color="#222222", linewidth=1.6, label="Three-seed mean")
        ax.fill_between(plot_steps, mean - sd, mean + sd, color="#777777", alpha=0.16, linewidth=0)
        if reference is not None:
            ax.axhline(reference, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Training environment step")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.6)
        ax.legend(frameon=False, ncol=2, loc="best")
        fig.tight_layout()
        save_figure(fig, stem)


def plot_evaluation(per_seed: list[dict[str, object]], aggregate: list[dict[str, object]]) -> None:
    configure_plot()
    specifications = (
        ("discovery_rate", 100.0, "Discovery rate (%)", "evaluation_abcd_discovery"),
        ("mean_delay_censored", 1.0, "Censored mean delay (slots)", "evaluation_abcd_delay"),
        ("aligned_handshake_opportunities", 1.0, "Aligned opportunities per episode", "evaluation_abcd_alignment"),
        ("tx_ratio", 100.0, "TX action ratio (%)", "evaluation_abcd_tx_ratio"),
    )
    x = np.arange(len(ARMS))
    for metric, scale, ylabel, stem in specifications:
        means = np.asarray([float(row[f"{metric}_mean"]) for row in aggregate]) * scale
        sds = np.asarray([float(row[f"{metric}_sd_across_training_seeds"]) for row in aggregate]) * scale
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(x, means, yerr=sds, color=COLORS, width=0.66, capsize=3, edgecolor="#333333", linewidth=0.5)
        for index, arm in enumerate(ARMS):
            values = [
                float(row[f"{metric}_mean"]) * scale
                for row in per_seed
                if row["arm"] == arm
            ]
            ax.scatter(np.full(3, index) + np.asarray((-0.08, 0.0, 0.08)), values, color="#222222", s=13, zorder=3)
        ax.set_xticks(x, tuple(ARMS))
        ax.set_xlabel("Ablation arm")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.6)
        fig.tight_layout()
        save_figure(fig, stem)


def plot_contrasts(contrasts: list[dict[str, object]]) -> None:
    configure_plot()
    means = [row for row in contrasts if row["training_seed"] == "mean"]
    labels = ("B-A", "D-C", "C-A", "D-B", "Interaction")
    x = np.arange(len(means))
    values = np.asarray([100.0 * float(row["discovery_rate_delta"]) for row in means])
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.bar(x, values, color=COLORS[0], width=0.62, alpha=0.85)
    for index, row in enumerate(means):
        seed_values = [
            100.0 * float(item["discovery_rate_delta"])
            for item in contrasts
            if item["contrast"] == row["contrast"] and item["training_seed"] != "mean"
        ]
        ax.scatter(np.full(3, index) + np.asarray((-0.08, 0.0, 0.08)), seed_values, color="#222222", s=13, zorder=3)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Discovery-rate contrast (pp)")
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.6)
    fig.tight_layout()
    save_figure(fig, "evaluation_component_contrasts")


def write_report(
    training: dict[int, list[dict[str, str]]],
    aggregate: list[dict[str, object]],
    contrasts: list[dict[str, object]],
) -> None:
    by_arm = {str(row["arm"]): row for row in aggregate}
    mean_contrasts = {str(row["contrast"]): row for row in contrasts if row["training_seed"] == "mean"}
    role_delta = mean_contrasts["D-B (learned role effect, learned beam)"]
    interaction = mean_contrasts["D-B-C+A (interaction)"]
    final_training = []
    for seed in SEEDS:
        last20 = training[seed][-20:]
        final_training.append(
            (
                seed,
                100.0 * float(np.mean([float(row["discovery_rate"]) for row in last20])),
                100.0 * float(np.mean([float(row["tx_actions"]) / 3000.0 for row in last20])),
                float(np.mean([float(row["episode_return_mean_per_agent"]) for row in last20])),
            )
        )
    role_pass = float(role_delta["discovery_rate_delta"]) >= 0.03
    stable_role = int(role_delta["positive_training_seeds"]) >= 2
    tx_pass = all(0.35 <= float(by_arm[arm]["tx_ratio_mean"]) <= 0.65 for arm in ARMS)
    decision = "PASS" if role_pass and stable_role and tx_pass else "FAIL"
    table_lines = [
        "| Arm | Discovery (%) | Delay (slots) | TX ratio (%) | Aligned opportunities |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = by_arm[arm]
        table_lines.append(
            f"| {arm} | {100 * float(row['discovery_rate_mean']):.2f} +/- "
            f"{100 * float(row['discovery_rate_sd_across_training_seeds']):.2f} | "
            f"{float(row['mean_delay_censored_mean']):.2f} | "
            f"{100 * float(row['tx_ratio_mean']):.2f} | "
            f"{float(row['aligned_handshake_opportunities_mean']):.2f} |"
        )
    training_lines = [
        f"| {seed} | {discovery:.2f} | {tx:.2f} | {episode_return:.3f} |"
        for seed, discovery, tx, episode_return in final_training
    ]
    REPORT.write_text(
        f"""# Role-balanced joint MAPPO 100k gate

## Material Passport

- Origin: frozen three-seed training and common-random-number dev20 evaluation
- Date: 2026-07-12
- Verification status: ANALYZED
- Scope: N=10, planar 15-degree codebook, 300 slots, 334 episodes (100,200 environment steps)

## Contract

The decentralized actor uses only local ISAC residual-table observations, local candidate processing, local topology deficit, and received post-handshake tables. The centralized MPNN critic accesses global training state only. Standalone sensing, idle actions, handcrafted action recommendations, rule residual logits, and global execution guidance are disabled. The role-balance term is training-only and has coefficient 0.01.

## Training endpoint

| Training seed | Last-20 discovery (%) | Last-20 TX ratio (%) | Last-20 return/UAV |
|---:|---:|---:|---:|
{chr(10).join(training_lines)}

## Frozen dev20 ablation

Values are mean +/- SD across three independently trained policies. All arms use the same 20 scenario seeds.

{chr(10).join(table_lines)}

The learned-role effect with the learned beam policy is D-B = {100 * float(role_delta['discovery_rate_delta']):.2f} pp ({int(role_delta['positive_training_seeds'])}/3 training seeds positive). The factorial interaction is {100 * float(interaction['discovery_rate_delta']):.2f} pp. The predeclared paper-level role-learning gate requires mean D-B >= 3 pp, at least 2/3 positive seeds, and no arm-level TX collapse outside 35%-65%.

## Decision

**{decision}.** This gate is a development evaluation, not the untouched final holdout. A failure means that no additional transfer matrix or paper-level MARL superiority claim should be launched from this checkpoint. A pass permits one untouched holdout evaluation before broader experiments.

## Artifacts

- `per_training_seed_ablation.csv`
- `aggregate_ablation.csv`
- `factorial_contrasts.csv`
- Eight 4:3 Times New Roman figures in PNG and PDF formats
""",
        encoding="utf-8",
    )


def main() -> None:
    training, evaluation = load_data()
    per_seed, aggregate = summarize(evaluation)
    contrasts = build_contrasts(per_seed)
    write_rows(OUTPUT / "per_episode_ablation.csv", evaluation)
    write_rows(OUTPUT / "per_training_seed_ablation.csv", per_seed)
    write_rows(OUTPUT / "aggregate_ablation.csv", aggregate)
    write_rows(OUTPUT / "factorial_contrasts.csv", contrasts)
    plot_training(training)
    plot_evaluation(per_seed, aggregate)
    plot_contrasts(contrasts)
    write_report(training, aggregate, contrasts)
    print(REPORT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
