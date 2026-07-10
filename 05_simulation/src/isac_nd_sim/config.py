from __future__ import annotations

from dataclasses import dataclass, replace
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
    rf_chains: int
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
    tx_power_w: float = 1.0
    rx_power_w: float = 0.6
    sense_power_w: float = 1.2
    idle_power_w: float = 0.05
    piggyback_sense_power_w: float = 0.2
    isac_sensing_model: str = "constant_error"
    isac_waveform: str = "abstract"
    carrier_frequency_hz: float = 30.0e9
    bandwidth_hz: float = 64.0e6
    radar_cross_section_m2: float = 1.0
    noise_psd_w_per_hz: float = 2.0e-21
    isac_tx_power_w: float = 1.0
    isac_processing_gain_db: float = 0.0
    isac_piggyback_loss_db: float = 0.0
    detection_midpoint_snr_db: float = -10.0
    detection_slope_per_db: float = 0.5
    min_detection_probability: float = 0.02
    max_detection_probability: float = 0.98
    sensing_footprint_radius_cells: int = 0
    sensing_position_error_std_m: float = 25.0
    sensing_report_ttl_slots: int = 100
    communication_phy_model: str = "ideal"
    communication_carrier_frequency_hz: float = 30.0e9
    communication_bandwidth_hz: float = 64.0e6
    communication_tx_power_w: float = 1.0
    communication_noise_figure_db: float = 7.0
    communication_path_loss_exponent: float = 2.0
    communication_reference_distance_m: float = 1.0
    communication_system_loss_db: float = 0.0
    communication_shadowing_std_db: float = 0.0
    communication_rician_k_db: float = 10.0
    communication_sinr_threshold_db: float = 5.0
    communication_antenna_efficiency: float = 0.70
    communication_sidelobe_gain_db: float = -10.0
    communication_fading_enabled: bool = True
    communication_shadowing_enabled: bool = True
    communication_antenna_gain_mode: str = "legacy_sector"
    communication_fixed_main_lobe_gain_db: float = 21.0
    shared_waveform_power_enabled: bool = False
    rendezvous_observation_enabled: bool = False

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
    energy = raw.get("energy", {})
    phy = raw.get("phy_sensing", {})
    comm_phy = raw.get("phy_communication", {})
    shared_phy = raw.get("phy_shared", {})
    shared_power_enabled = bool(shared_phy.get("enabled", False))
    shared_power_w = float(shared_phy.get("tx_power_w", energy.get("tx_power_w", 1.0)))
    energy_tx_power_w = shared_power_w if shared_power_enabled else float(energy.get("tx_power_w", 1.0))
    sensing_tx_power_w = (
        shared_power_w
        if shared_power_enabled
        else float(phy.get("tx_power_w", energy.get("tx_power_w", 1.0)))
    )
    communication_tx_power_w = (
        shared_power_w
        if shared_power_enabled
        else float(comm_phy.get("tx_power_w", energy.get("tx_power_w", 1.0)))
    )

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
        rf_chains=int(first(beam.get("rf_chains", 1))),
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
        tx_power_w=energy_tx_power_w,
        rx_power_w=float(energy.get("rx_power_w", 0.6)),
        sense_power_w=float(energy.get("sense_power_w", 1.2)),
        idle_power_w=float(energy.get("idle_power_w", 0.05)),
        piggyback_sense_power_w=float(energy.get("piggyback_sense_power_w", 0.2)),
        isac_sensing_model=str(phy.get("model", "constant_error")),
        isac_waveform=str(phy.get("waveform", "abstract")),
        carrier_frequency_hz=float(phy.get("carrier_frequency_hz", 30.0e9)),
        bandwidth_hz=float(phy.get("bandwidth_hz", 64.0e6)),
        radar_cross_section_m2=float(phy.get("radar_cross_section_m2", 1.0)),
        noise_psd_w_per_hz=float(phy.get("noise_psd_w_per_hz", 2.0e-21)),
        isac_tx_power_w=sensing_tx_power_w,
        isac_processing_gain_db=float(phy.get("processing_gain_db", 0.0)),
        isac_piggyback_loss_db=float(phy.get("piggyback_loss_db", 0.0)),
        detection_midpoint_snr_db=float(phy.get("detection_midpoint_snr_db", -10.0)),
        detection_slope_per_db=float(phy.get("detection_slope_per_db", 0.5)),
        min_detection_probability=float(phy.get("min_detection_probability", 0.02)),
        max_detection_probability=float(phy.get("max_detection_probability", 0.98)),
        sensing_footprint_radius_cells=int(phy.get("angular_footprint_radius_cells", 0)),
        sensing_position_error_std_m=float(phy.get("position_error_std_m", 25.0)),
        sensing_report_ttl_slots=int(phy.get("report_ttl_slots", 100)),
        communication_phy_model=str(comm_phy.get("model", "ideal")),
        communication_carrier_frequency_hz=float(
            comm_phy.get("carrier_frequency_hz", phy.get("carrier_frequency_hz", 30.0e9))
        ),
        communication_bandwidth_hz=float(
            comm_phy.get("bandwidth_hz", phy.get("bandwidth_hz", 64.0e6))
        ),
        communication_tx_power_w=communication_tx_power_w,
        communication_noise_figure_db=float(comm_phy.get("noise_figure_db", 7.0)),
        communication_path_loss_exponent=float(comm_phy.get("path_loss_exponent", 2.0)),
        communication_reference_distance_m=float(comm_phy.get("reference_distance_m", 1.0)),
        communication_system_loss_db=float(comm_phy.get("system_loss_db", 0.0)),
        communication_shadowing_std_db=float(comm_phy.get("shadowing_std_db", 0.0)),
        communication_rician_k_db=float(comm_phy.get("rician_k_db", 10.0)),
        communication_sinr_threshold_db=float(comm_phy.get("sinr_threshold_db", 5.0)),
        communication_antenna_efficiency=float(comm_phy.get("antenna_efficiency", 0.70)),
        communication_sidelobe_gain_db=float(comm_phy.get("sidelobe_gain_db", -10.0)),
        communication_fading_enabled=bool(comm_phy.get("fading_enabled", True)),
        communication_shadowing_enabled=bool(comm_phy.get("shadowing_enabled", True)),
        communication_antenna_gain_mode=str(comm_phy.get("antenna_gain_mode", "legacy_sector")),
        communication_fixed_main_lobe_gain_db=float(comm_phy.get("fixed_main_lobe_gain_db", 21.0)),
        shared_waveform_power_enabled=shared_power_enabled,
        rendezvous_observation_enabled=bool(protocol.get("rendezvous_observation_enabled", False)),
    )


def with_communication_tx_power(cfg: SimulationConfig, power_w: float) -> SimulationConfig:
    """Apply a communication power point without breaking a shared ISAC waveform budget."""

    power = float(power_w)
    updates: dict[str, float] = {"communication_tx_power_w": power}
    if cfg.shared_waveform_power_enabled:
        updates.update(tx_power_w=power, isac_tx_power_w=power)
    return replace(cfg, **updates)
