from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEEDS = (69260713, 69261722, 69262731)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report MATD3 reference matrix progress.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def count_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    seeds = tuple(int(part.strip()) for part in args.seeds.split(",") if part.strip())
    rows = []
    total = 0
    complete = 0
    for seed in seeds:
        output = args.run_root / "matd3_direct_isac_reference" / f"seed_{seed}"
        episodes = count_rows(output / "episode_metrics.csv")
        done = (output / "manifest.json").is_file() and episodes == args.episodes
        rows.append(
            {
                "seed": seed,
                "episodes": episodes,
                "expected_episodes": args.episodes,
                "percent": 100.0 * episodes / args.episodes,
                "complete": done,
            }
        )
        total += episodes
        complete += int(done)
    payload = {
        "complete": complete,
        "runs": len(seeds),
        "episodes": total,
        "expected_episodes": len(seeds) * args.episodes,
        "percent": 100.0 * total / (len(seeds) * args.episodes),
        "details": rows,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
        return
    print(
        f"complete={complete}/{len(seeds)} overall={payload['percent']:.2f}% "
        f"episode={total}/{payload['expected_episodes']}"
    )
    for row in rows:
        state = "complete" if row["complete"] else ("partial" if row["episodes"] else "missing")
        print(
            f"{state:8s} {row['percent']:6.2f}% episode={row['episodes']:4d}/{args.episodes:4d} "
            f"seed_{row['seed']}"
        )


if __name__ == "__main__":
    main()
