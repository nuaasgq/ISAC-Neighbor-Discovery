from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "05_simulation" / "results_raw"
OUTPUT = ROOT / "06_analysis" / "figures"
SERIES = {
    "Normalized VDN": RESULTS / "sanity_n2_vdn_direction_300_20260712" / "episode_metrics.csv",
    "Feed-forward QMIX": RESULTS / "sanity_n2_qmix_direction_300_20260712" / "episode_metrics.csv",
    "Recurrent MAPPO": RESULTS / "sanity_n2_rmappo_direction_1000_20260712" / "episode_metrics.csv",
}
COLORS = {
    "Normalized VDN": "#0072B2",
    "Feed-forward QMIX": "#D55E00",
    "Recurrent MAPPO": "#009E73",
}


def rolling_mean(values: np.ndarray, window: int = 50) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    for index in range(values.size):
        start = max(0, index + 1 - window)
        result[index] = (cumulative[index + 1] - cumulative[start]) / (index + 1 - start)
    return result


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "figure.figsize": (6.4, 4.8),
        }
    )
    figure, axis = plt.subplots()
    for label, path in SERIES.items():
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        episodes = np.asarray([int(row["episode"]) + 1 for row in rows])
        rates = np.asarray([float(row["discovery_rate"]) for row in rows])
        axis.plot(episodes, rolling_mean(rates), label=label, color=COLORS[label], linewidth=1.8)
    axis.axhline(0.195, color="#666666", linestyle="--", linewidth=1.2, label="Random beam (evaluation)")
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Discovery rate (50-episode moving average)")
    axis.set_xlim(left=1)
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.8)
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "two_node_optimizer_sanity_training_20260712.png", dpi=300)
    figure.savefig(OUTPUT / "two_node_optimizer_sanity_training_20260712.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
