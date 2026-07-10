from __future__ import annotations

import math
from dataclasses import dataclass

from .config import SimulationConfig

LIGHT_SPEED_MPS = 299_792_458.0
SENSING_MEASUREMENT_MODES = ("ideal_count", "noisy_count", "binary_occupancy")


@dataclass(frozen=True)
class AnonymousSensingDetection:
    """One locally detected anonymous target; no neighbor identity is exposed."""

    detection_id: str
    position_m: tuple[float, float, float]
    snr_db: float
    detection_probability: float
    confidence: float


@dataclass(frozen=True)
class BeamSensingMeasurement:
    """Common physical-to-link measurement consumed by every protocol."""

    measurement_id: str
    node: int
    beam: int
    slot: int
    mode: str
    occupancy_value: float
    estimated_target_count: int
    count_variance: float
    confidence: float
    max_snr_db: float
    detections: tuple[AnonymousSensingDetection, ...]
    false_alarm_count: int = 0


@dataclass(frozen=True)
class SharedSensingReport:
    """Anonymous table entry with immutable provenance for TTL and deduplication."""

    detection_id: str
    position_m: tuple[float, float, float]
    confidence: float
    snr_db: float
    origin_node: int
    origin_slot: int


def radar_snr_linear(distance_m: float, cfg: SimulationConfig, *, piggyback: bool = False) -> float:
    """Monostatic radar-equation SNR used for PHY-aware ISAC abstraction.

    The base equation follows the MIMO-OTFS FANET neighbor-discovery model:
    SNR = lambda^2 sigma P_t / ((4 pi)^3 r^4 sigma_w^2).
    Processing/array/waveform gains are exposed as a calibrated dB term so the
    link-layer simulator can reproduce a target sensing operating point without
    reimplementing the full OTFS receiver.
    """

    distance = max(float(distance_m), 1e-3)
    carrier = max(float(cfg.carrier_frequency_hz), 1.0)
    wavelength = LIGHT_SPEED_MPS / carrier
    noise_power = max(float(cfg.noise_psd_w_per_hz) * float(cfg.bandwidth_hz), 1e-30)
    numerator = (wavelength**2) * max(float(cfg.radar_cross_section_m2), 1e-12) * max(float(cfg.isac_tx_power_w), 1e-12)
    denominator = ((4.0 * math.pi) ** 3) * (distance**4) * noise_power
    gain_db = float(cfg.isac_processing_gain_db)
    if piggyback:
        gain_db -= float(cfg.isac_piggyback_loss_db)
    return max(0.0, numerator / denominator * db_to_linear(gain_db))


def radar_snr_db(distance_m: float, cfg: SimulationConfig, *, piggyback: bool = False) -> float:
    snr = radar_snr_linear(distance_m, cfg, piggyback=piggyback)
    return linear_to_db(max(snr, 1e-30))


def detection_probability_from_snr_db(snr_db: float, cfg: SimulationConfig) -> float:
    low = float(cfg.min_detection_probability)
    high = float(cfg.max_detection_probability)
    if high < low:
        low, high = high, low
    midpoint = float(cfg.detection_midpoint_snr_db)
    slope = max(float(cfg.detection_slope_per_db), 1e-6)
    logistic = 1.0 / (1.0 + math.exp(-slope * (float(snr_db) - midpoint)))
    return float(low + (high - low) * logistic)


def detection_probability(distance_m: float, cfg: SimulationConfig, *, piggyback: bool = False) -> float:
    if cfg.isac_sensing_model != "radar_snr":
        return float(1.0 - cfg.miss_detection_rate)
    return detection_probability_from_snr_db(
        radar_snr_db(distance_m, cfg, piggyback=piggyback),
        cfg,
    )


def db_to_linear(value_db: float) -> float:
    return 10.0 ** (float(value_db) / 10.0)


def linear_to_db(value: float) -> float:
    return 10.0 * math.log10(max(float(value), 1e-30))
