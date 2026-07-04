from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "05_simulation" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isac_nd_sim.config import load_config  # noqa: E402
from isac_nd_sim.runner import override_config, run, write_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small dynamic mobility smoke simulation.")
    parser.add_argument("--config", default=str(ROOT / "05_simulation" / "configs" / "mobile_smoke.yaml"))
    parser.add_argument("--output", default=str(ROOT / "05_simulation" / "results_raw" / "mobile_smoke"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--slots", type=int, default=None)
    parser.add_argument("--protocols", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = override_config(load_config(config_path), args.episodes, args.slots)
    protocols = args.protocols.split(",") if args.protocols else list(config.baselines)
    rows = run(config, protocols)
    write_outputs(config_path, Path(args.output), rows)
    print(f"wrote {len(rows)} episode summaries to {args.output}")


if __name__ == "__main__":
    main()
