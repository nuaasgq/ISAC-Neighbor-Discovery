from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "06_analysis" / "scripts" / "analyze_n10_b15_static_ideal_paired_eval.py"
SMOKE = ROOT / "05_simulation" / "results_raw" / "n10_b15_static_ideal_paired_eval_smoke2"


def load_analysis():
    spec = importlib.util.spec_from_file_location("paired_eval_analysis", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_smoke_analysis_discovers_all_eight_methods() -> None:
    module = load_analysis()
    rows = module.load_evaluation_rows(SMOKE)
    counts = module.validate_paired_contract(rows, allow_incomplete=True)

    assert set(counts) == set(module.METHOD_LABELS)
    assert all(count == 2 for count in counts.values())


def test_smoke_timeline_contains_one_row_per_edge_and_episode() -> None:
    module = load_analysis()
    rows = module.load_timeline_rows(SMOKE)

    assert len(rows) == 8 * 2 * 45
    assert {row["method"] for row in rows} == set(module.METHOD_LABELS)
