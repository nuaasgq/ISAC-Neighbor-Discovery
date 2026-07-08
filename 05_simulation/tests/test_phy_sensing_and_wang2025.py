from __future__ import annotations

from dataclasses import replace

from isac_nd_sim.config import load_config
from isac_nd_sim.phy_sensing import detection_probability, radar_snr_db
from isac_nd_sim.runner import run


def test_radar_snr_sensing_probability_decreases_with_distance() -> None:
    cfg = load_config("05_simulation/configs/mvp.yaml")
    cfg = replace(
        cfg,
        isac_sensing_model="radar_snr",
        isac_processing_gain_db=70.0,
        detection_midpoint_snr_db=-10.0,
        detection_slope_per_db=0.5,
    )

    near_snr = radar_snr_db(500.0, cfg)
    far_snr = radar_snr_db(5000.0, cfg)
    assert near_snr > far_snr
    assert detection_probability(500.0, cfg) > detection_probability(5000.0, cfg)


def test_default_config_keeps_constant_error_sensing_model() -> None:
    cfg = load_config("05_simulation/configs/mvp.yaml")
    assert cfg.isac_sensing_model == "constant_error"
    assert cfg.isac_waveform == "abstract"


def test_wang2025_isac_tables_runs_with_phy_sensing_metrics() -> None:
    cfg = load_config("05_simulation/configs/mvp.yaml")
    cfg = replace(
        cfg,
        episodes=1,
        slots_per_episode=20,
        n_nodes=8,
        azimuth_cells=12,
        elevation_cells=4,
        communication_range_m=900.0,
        sensing_range_m=900.0,
        false_alarm_rate=0.01,
        miss_detection_rate=0.05,
        angular_cell_offset_std=0.0,
        sensing_period_slots=1,
        slot_duration_s=0.005,
        isac_sensing_model="radar_snr",
        isac_waveform="mimo_otfs",
        isac_processing_gain_db=85.0,
        detection_midpoint_snr_db=-10.0,
        baselines=("uniform_random", "wang2025_isac_tables"),
    )
    rows = run(cfg, list(cfg.baselines))
    assert len(rows) == 2
    wang = next(row for row in rows if row["protocol"] == "wang2025_isac_tables")
    assert wang["piggyback_sense_actions"] > 0
    assert wang["sensing_observations"] > 0
    assert "mean_sensing_snr_db" in wang
    assert 0.0 <= wang["sensing_detection_rate"] <= 1.0
