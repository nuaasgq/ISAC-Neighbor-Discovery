from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report independent-role ablation progress.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", default="59262731")
    parser.add_argument("--episodes", type=int, default=1000)
    return parser.parse_args()


def count_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    seeds = tuple(int(part.strip()) for part in args.seeds.split(",") if part.strip())
    total = 0
    complete = 0
    for seed in seeds:
        output = args.run_root / "mappo_direct_isac_independent_role" / f"seed_{seed}"
        episodes = count_rows(output / "episode_metrics.csv")
        done = episodes == args.episodes and (output / "eval_episode_metrics.csv").is_file()
        total += episodes
        complete += int(done)
        state = "complete" if done else ("partial" if episodes else "missing")
        print(
            f"{state:8s} {100.0 * episodes / args.episodes:6.2f}% "
            f"episode={episodes:4d}/{args.episodes:4d} seed_{seed}"
        )
    print(
        f"complete={complete}/{len(seeds)} overall="
        f"{100.0 * total / (len(seeds) * args.episodes):.2f}%"
    )


if __name__ == "__main__":
    main()
