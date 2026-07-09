from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

from isac_nd_sim.config import load_config
from isac_nd_sim.simulator import MODE_SENSE, NeighborDiscoverySimulator


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "06_analysis" / "scripts" / "run_wang2025_aligned_marl_matrix.py"


def _load_matrix_script():
    spec = importlib.util.spec_from_file_location("run_wang2025_aligned_marl_matrix", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_wang_aligned_matrix_uses_common_env_and_non_gated_marl() -> None:
    module = _load_matrix_script()

    assert module.COMMON_ENV_PROTOCOL == "wang2025_isac_tables"
    assert module.ACTION_POLICY_PROTOCOLS == (
        "uniform_trx_idle_random",
        "wang2025_isac_tables",
        "budgeted_collision_aware_isac",
    )
    assert module.METHOD_SPECS[0]["env_protocol"] == module.COMMON_ENV_PROTOCOL
    assert module.METHOD_SPECS[0]["network"] == "contention_shared"


def test_uniform_trx_idle_action_policy_has_no_standalone_sense() -> None:
    module = _load_matrix_script()
    cfg = replace(
        load_config("05_simulation/configs/wang2025_reproduction_smoke.yaml"),
        n_nodes=8,
        episodes=1,
        slots_per_episode=1,
    )
    sim = NeighborDiscoverySimulator(cfg, protocol=module.COMMON_ENV_PROTOCOL, seed=123)
    sim.reset()

    actions = module.uniform_trx_idle_actions(sim)

    assert len(actions) == cfg.n_nodes
    assert all(action.mode != MODE_SENSE for action in actions)
