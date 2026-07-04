from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "05_simulation" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isac_nd_sim.config import load_config  # noqa: E402
from isac_nd_sim.runner import override_config, override_mobility, run_detailed, write_outputs  # noqa: E402


DEFAULT_PROTOCOLS = (
    "skyorbs_like_skip_scan",
    "uniform_random",
    "rl_no_isac",
    "improved_rl_no_isac",
    "improved_rl_isac",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate five-way protocols with trained shared-policy parameters.")
    parser.add_argument("--config", default=str(ROOT / "05_simulation" / "configs" / "paper_core_d1.yaml"))
    parser.add_argument(
        "--trained-config",
        required=True,
        help="Path to training output best_config.yaml.",
    )
    parser.add_argument("--output", required=True, help="Output directory for runner artifacts.")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--slots", type=int, default=None)
    parser.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS))
    parser.add_argument("--mobility", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override base scenario seed.")
    return parser.parse_args()


def load_trained_parameters(path: str | Path) -> dict[str, float]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    params = doc.get("shared_policy_parameters", {})
    if not params:
        raise ValueError(f"No shared_policy_parameters found in {path}.")
    return {str(key): float(value) for key, value in params.items()}


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    trained_path = Path(args.trained_config)
    config = override_mobility(override_config(load_config(config_path), args.episodes, args.slots), args.mobility)
    if args.seed is not None:
        config = replace(config, seed=int(args.seed))
    params = load_trained_parameters(trained_path)
    config = replace(config, **params)
    protocols = [item.strip() for item in args.protocols.split(",") if item.strip()]
    rows, slot_rows, edge_rows = run_detailed(config, protocols)
    output_dir = Path(args.output)
    write_outputs(config_path, output_dir, rows, config, slot_rows, edge_rows)
    (output_dir / "trained_parameters.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "trained_config": str(trained_path),
                "episode_rows": len(rows),
                "slot_rows": len(slot_rows),
                "edge_rows": len(edge_rows),
                "protocols": protocols,
                "trained_parameters": params,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
