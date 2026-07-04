from __future__ import annotations

from dataclasses import replace

from isac_nd_sim.config import load_config
from isac_nd_sim.runner import run


def test_simulator_smoke_with_dynamic_mobility() -> None:
    cfg = load_config("05_simulation/configs/mvp.yaml")
    cfg = replace(
        cfg,
        episodes=1,
        slots_per_episode=30,
        n_nodes=8,
        azimuth_cells=8,
        elevation_cells=4,
        communication_range_m=350.0,
        sensing_range_m=400.0,
        baselines=("uniform_random", "isac_only", "itap_nd"),
    )
    rows = run(cfg, list(cfg.baselines))
    assert len(rows) == 3
    for row in rows:
        assert row["mobility_model"] != "static"
        assert row["moved_distance_mean_m"] > 0.0
        assert 0.0 <= row["discovery_rate"] <= 1.0
        assert 0.0 <= row["empty_scan_ratio"] <= 1.0
