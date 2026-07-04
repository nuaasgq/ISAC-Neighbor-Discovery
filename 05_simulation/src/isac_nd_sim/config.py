from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def first(value: Any) -> Any:
    return value[0] if isinstance(value, list) else value


@dataclass(frozen=True)
class SimulationConfig:
    name: str
    seed: int
    episodes: int
    slots_per_episode: int
    slot_metric_period: int
    slot_duration_s: float
    n_nodes: int
    area_size_m: tuple[float, float, float]
    communication_range_m: float
    sensing_range_m: float
    mobility: dict[str, Any]
    azimuth_cells: int
    elevation_cells: int
    alignment_tolerance_cells: int
    p_sense: float
    p_tx: float
    p_rx: float
    p_idle: float
    false_alarm_rate: float
    miss_detection_rate: float
    angular_cell_offset_std: float
    sensing_period_slots: int
    belief_update_rho: float
    exploration_floor: float
    softmax_beta: float
    target_degree: int
    alpha_occupancy: float
    beta_diversity: float
    gamma_degree_need: float
    eta_staleness: float
    confidence_decay: float
    piggyback_sensing_period_multiplier: float
    baselines: tuple[str, ...]

    @property
    def n_beams(self) -> int:
        return self.azimuth_cells * self.elevation_cells


def normalize_mobility(network_cfg: dict[str, Any], mobility_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    mobility = network_cfg.get("mobility", {})
    if isinstance(mobility, str):
        mobility = {"model": mobility}
    mobility = dict(mobility)
    if mobility_cfg:
        mobility.update(mobility_cfg)
    if "default_model" in mobility and "model" not in mobility:
        mobility["model"] = mobility["default_model"]
    alias_map = {
        "mean_speed_mps": "speed_mean_mps",
        "speed_min_mps": "min_speed_mps",
        "speed_max_mps": "max_speed_mps",
        "gauss_markov_alpha": "alpha",
        "direction_update_interval_slots": "direction_update_period_slots",
    }
    for old_key, new_key in alias_map.items():
        if old_key in mobility and new_key not in mobility:
            mobility[new_key] = mobility[old_key]
    mobility.setdefault("model", "gauss_markov")
    mobility.setdefault("speed_mean_mps", 15.0)
    mobility.setdefault("speed_std_mps", 3.0)
    mobility.setdefault("min_speed_mps", 0.0)
    mobility.setdefault("alpha", 0.85)
    mobility.setdefault("random_walk_step_std_m", 1.5)
    mobility.setdefault("direction_update_period_slots", 25)
    mobility.setdefault("waypoint_pause_slots", 0)
    mobility.setdefault("boundary", "reflect")
    return mobility


def load_config(path: str | Path) -> SimulationConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    experiment = raw["experiment"]
    network = raw["network"]
    beam = raw["beam"]
    slot = raw["slot"]
    isac_error = raw["isac_error"]
    protocol = raw["protocol"]

    return SimulationConfig(
        name=str(experiment["name"]),
        seed=int(experiment["seed"]),
        episodes=int(experiment["episodes"]),
        slots_per_episode=int(experiment["slots_per_episode"]),
        slot_metric_period=int(experiment.get("slot_metric_period", 1)),
        slot_duration_s=float(experiment["slot_duration_ms"]) / 1000.0,
        n_nodes=int(first(network.get("node_counts", network.get("node_count", 30)))),
        area_size_m=tuple(float(v) for v in network["area_size_m"]),
        communication_range_m=float(network["communication_range_m"]),
        sensing_range_m=float(network["sensing_range_m"]),
        mobility=normalize_mobility(network, raw.get("mobility")),
        azimuth_cells=int(first(beam["azimuth_cells"])),
        elevation_cells=int(first(beam["elevation_cells"])),
        alignment_tolerance_cells=int(beam["alignment_tolerance_cells"]),
        p_sense=float(slot["p_sense"]),
        p_tx=float(slot["p_tx"]),
        p_rx=float(slot["p_rx"]),
        p_idle=float(slot["p_idle"]),
        false_alarm_rate=float(first(isac_error["false_alarm_rates"])),
        miss_detection_rate=float(first(isac_error["miss_detection_rates"])),
        angular_cell_offset_std=float(first(isac_error["angular_cell_offsets"])),
        sensing_period_slots=int(first(isac_error["sensing_periods"])),
        belief_update_rho=float(isac_error["belief_update_rho"]),
        exploration_floor=float(protocol["exploration_floor"]),
        softmax_beta=float(protocol["softmax_beta"]),
        target_degree=int(protocol["target_degree"]),
        alpha_occupancy=float(protocol["alpha_occupancy"]),
        beta_diversity=float(protocol["beta_divity"])
        if "beta_divity" in protocol
        else float(protocol["beta_diversity"]),
        gamma_degree_need=float(protocol["gamma_degree_need"]),
        eta_staleness=float(protocol["eta_staleness"]),
        confidence_decay=float(protocol["confidence_decay"]),
        piggyback_sensing_period_multiplier=float(protocol.get("piggyback_sensing_period_multiplier", 1.0)),
        baselines=tuple(str(v) for v in raw["baselines"]),
    )
