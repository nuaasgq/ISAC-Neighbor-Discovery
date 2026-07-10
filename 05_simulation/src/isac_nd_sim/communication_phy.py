from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import SimulationConfig

LIGHT_SPEED_MPS = 299_792_458.0
BOLTZMANN_J_PER_K = 1.380_649e-23
REFERENCE_TEMPERATURE_K = 290.0


def db_to_linear(value_db: float | np.ndarray) -> float | np.ndarray:
    return np.power(10.0, np.asarray(value_db) / 10.0)


def linear_to_db(value: float | np.ndarray) -> float | np.ndarray:
    values = np.maximum(np.asarray(value), 1e-300)
    result = 10.0 * np.log10(values)
    return float(result) if result.ndim == 0 else result


def free_space_path_loss_db(distance_m: float | np.ndarray, carrier_frequency_hz: float) -> float | np.ndarray:
    distance = np.maximum(np.asarray(distance_m, dtype=float), 1e-6)
    wavelength = LIGHT_SPEED_MPS / max(float(carrier_frequency_hz), 1.0)
    result = 20.0 * np.log10(4.0 * math.pi * distance / wavelength)
    return float(result) if result.ndim == 0 else result


def close_in_path_loss_db(
    distance_m: float | np.ndarray,
    carrier_frequency_hz: float,
    path_loss_exponent: float,
    reference_distance_m: float = 1.0,
) -> float | np.ndarray:
    reference = max(float(reference_distance_m), 1e-3)
    distance = np.maximum(np.asarray(distance_m, dtype=float), reference)
    reference_loss = free_space_path_loss_db(reference, carrier_frequency_hz)
    result = reference_loss + 10.0 * max(float(path_loss_exponent), 0.0) * np.log10(distance / reference)
    return float(result) if np.asarray(result).ndim == 0 else result


def main_lobe_gain_db(cfg: SimulationConfig) -> float:
    """Ideal sectored-beam gain derived from one codebook cell's solid angle."""

    azimuth_width = 2.0 * math.pi / max(1, int(cfg.azimuth_cells))
    elevation_width = math.pi / max(1, int(cfg.elevation_cells))
    solid_angle = azimuth_width * 2.0 * math.sin(elevation_width / 2.0)
    directivity = 4.0 * math.pi / max(solid_angle, 1e-12)
    efficiency = float(np.clip(cfg.communication_antenna_efficiency, 1e-6, 1.0))
    return 10.0 * math.log10(max(efficiency * directivity, 1e-12))


def thermal_noise_power_w(cfg: SimulationConfig) -> float:
    bandwidth = max(float(cfg.communication_bandwidth_hz), 1.0)
    noise_factor = float(db_to_linear(cfg.communication_noise_figure_db))
    return BOLTZMANN_J_PER_K * REFERENCE_TEMPERATURE_K * bandwidth * noise_factor


def sample_rician_power(rng: np.random.Generator, shape: tuple[int, ...], k_db: float) -> np.ndarray:
    k_linear = float(db_to_linear(k_db))
    specular = math.sqrt(k_linear / (k_linear + 1.0))
    diffuse = math.sqrt(1.0 / (k_linear + 1.0))
    scatter = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / math.sqrt(2.0)
    channel = specular + diffuse * scatter
    return np.abs(channel) ** 2


@dataclass(frozen=True)
class HandshakePhyResult:
    success_matrix: np.ndarray
    forward_decoded_matrix: np.ndarray
    forward_sinr_db: np.ndarray
    ack_sinr_db: np.ndarray
    forward_interference_failures: np.ndarray
    forward_outage_failures: np.ndarray
    ack_interference_failures: np.ndarray
    ack_outage_failures: np.ndarray

    @property
    def sinr_samples_db(self) -> np.ndarray:
        forward = self.forward_sinr_db[np.isfinite(self.forward_sinr_db)]
        ack = self.ack_sinr_db[np.isfinite(self.ack_sinr_db)]
        if forward.size == 0:
            return ack
        if ack.size == 0:
            return forward
        return np.concatenate([forward, ack])


class CommunicationPhy:
    """Two-phase directional HELLO/ACK PHY with reciprocal slot fading."""

    def __init__(self, cfg: SimulationConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        if cfg.communication_phy_model not in {"ideal", "close_in_rician_sinr"}:
            raise ValueError(
                "communication PHY model must be 'ideal' or 'close_in_rician_sinr'."
            )
        if cfg.communication_bandwidth_hz <= 0.0 or cfg.communication_tx_power_w <= 0.0:
            raise ValueError("Communication bandwidth and transmit power must be positive.")
        if cfg.communication_reference_distance_m <= 0.0:
            raise ValueError("Communication reference distance must be positive.")
        self._shadowing_db = np.zeros((0, 0), dtype=float)
        self._fading_slot: int | None = None
        self._fading_power = np.zeros((0, 0), dtype=float)

    def reset(self, n_nodes: int) -> None:
        n_nodes = int(n_nodes)
        self._shadowing_db = np.zeros((n_nodes, n_nodes), dtype=float)
        if self.cfg.communication_shadowing_enabled and self.cfg.communication_shadowing_std_db > 0.0:
            upper = self.rng.normal(
                0.0,
                float(self.cfg.communication_shadowing_std_db),
                size=(n_nodes, n_nodes),
            )
            upper = np.triu(upper, k=1)
            self._shadowing_db = upper + upper.T
        self._fading_slot = None
        self._fading_power = np.ones((n_nodes, n_nodes), dtype=float)
        np.fill_diagonal(self._fading_power, 0.0)

    def channel_power(self, n_nodes: int, slot: int) -> np.ndarray:
        if self._fading_power.shape != (n_nodes, n_nodes):
            self.reset(n_nodes)
        if self._fading_slot == int(slot):
            return self._fading_power
        if self.cfg.communication_fading_enabled:
            upper = sample_rician_power(
                self.rng,
                (n_nodes, n_nodes),
                self.cfg.communication_rician_k_db,
            )
            upper = np.triu(upper, k=1)
            self._fading_power = upper + upper.T
        else:
            self._fading_power = np.ones((n_nodes, n_nodes), dtype=float)
            np.fill_diagonal(self._fading_power, 0.0)
        self._fading_slot = int(slot)
        return self._fading_power

    def path_loss_matrix_db(self, distance_m: np.ndarray) -> np.ndarray:
        loss = np.asarray(
            close_in_path_loss_db(
                distance_m,
                self.cfg.communication_carrier_frequency_hz,
                self.cfg.communication_path_loss_exponent,
                self.cfg.communication_reference_distance_m,
            ),
            dtype=float,
        )
        loss += float(self.cfg.communication_system_loss_db)
        if self._shadowing_db.shape == loss.shape:
            loss += self._shadowing_db
        np.fill_diagonal(loss, np.inf)
        return loss

    def received_power_matrix(
        self,
        selected_beams: np.ndarray,
        true_beams: np.ndarray,
        distance_m: np.ndarray,
        emitter_mask: np.ndarray,
        receiver_mask: np.ndarray,
        channel_power: np.ndarray,
    ) -> np.ndarray:
        main_gain = float(db_to_linear(main_lobe_gain_db(self.cfg)))
        side_gain = float(db_to_linear(self.cfg.communication_sidelobe_gain_db))
        path_gain = db_to_linear(-self.path_loss_matrix_db(distance_m))
        pointing = self._beam_match_matrix(selected_beams, true_beams)
        tx_gain = np.where(pointing, main_gain, side_gain)
        rx_gain = np.where(pointing.T, main_gain, side_gain)
        active_links = emitter_mask[:, None] & receiver_mask[None, :]
        power = (
            float(self.cfg.communication_tx_power_w)
            * path_gain
            * tx_gain
            * rx_gain
            * channel_power
            * active_links
        )
        np.fill_diagonal(power, 0.0)
        return power

    def _beam_match_matrix(self, selected_beams: np.ndarray, true_beams: np.ndarray) -> np.ndarray:
        azimuth_cells = max(1, int(self.cfg.azimuth_cells))
        tolerance = max(0, int(self.cfg.alignment_tolerance_cells))
        selected_az = (selected_beams % azimuth_cells)[:, None]
        selected_el = (selected_beams // azimuth_cells)[:, None]
        expected_az = true_beams % azimuth_cells
        expected_el = true_beams // azimuth_cells
        azimuth_delta = np.abs(selected_az - expected_az)
        azimuth_delta = np.minimum(azimuth_delta, azimuth_cells - azimuth_delta)
        elevation_delta = np.abs(selected_el - expected_el)
        return (azimuth_delta <= tolerance) & (elevation_delta <= tolerance)

    def resolve_handshake(
        self,
        candidate_matrix: np.ndarray,
        selected_beams: np.ndarray,
        true_beams: np.ndarray,
        distance_m: np.ndarray,
        tx_mask: np.ndarray,
        rx_mask: np.ndarray,
        slot: int,
    ) -> HandshakePhyResult:
        n_nodes = candidate_matrix.shape[0]
        channel_power = self.channel_power(n_nodes, slot)
        noise = thermal_noise_power_w(self.cfg)
        threshold = float(db_to_linear(self.cfg.communication_sinr_threshold_db))

        forward_power = self.received_power_matrix(
            selected_beams,
            true_beams,
            distance_m,
            tx_mask,
            rx_mask,
            channel_power,
        )
        forward_decoded, forward_sinr, forward_interference, forward_outage = self._decode_phase(
            candidate_matrix,
            forward_power,
            tx_mask,
            noise,
            threshold,
        )

        responders = forward_decoded.any(axis=0)
        ack_receivers = forward_decoded.any(axis=1)
        ack_candidate_matrix = forward_decoded.T
        ack_power = self.received_power_matrix(
            selected_beams,
            true_beams,
            distance_m,
            responders,
            ack_receivers,
            channel_power,
        )
        ack_decoded_transposed, ack_sinr_transposed, ack_interference_t, ack_outage_t = self._decode_phase(
            ack_candidate_matrix,
            ack_power,
            responders,
            noise,
            threshold,
        )
        success_matrix = forward_decoded & ack_decoded_transposed.T
        return HandshakePhyResult(
            success_matrix=success_matrix,
            forward_decoded_matrix=forward_decoded,
            forward_sinr_db=self._matrix_linear_to_db(forward_sinr),
            ack_sinr_db=self._matrix_linear_to_db(ack_sinr_transposed.T),
            forward_interference_failures=forward_interference,
            forward_outage_failures=forward_outage,
            ack_interference_failures=ack_interference_t.T,
            ack_outage_failures=ack_outage_t.T,
        )

    @staticmethod
    def _decode_phase(
        candidate_matrix: np.ndarray,
        received_power: np.ndarray,
        emitter_mask: np.ndarray,
        noise_power: float,
        threshold_linear: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        decoded = np.zeros_like(candidate_matrix, dtype=bool)
        sinr = np.full(candidate_matrix.shape, np.nan, dtype=float)
        interference_failures = np.zeros_like(candidate_matrix, dtype=bool)
        outage_failures = np.zeros_like(candidate_matrix, dtype=bool)
        for receiver in np.flatnonzero(candidate_matrix.any(axis=0)):
            candidate_sources = np.flatnonzero(candidate_matrix[:, receiver])
            aggregate = float(received_power[emitter_mask, receiver].sum())
            for source in candidate_sources:
                signal = float(received_power[source, receiver])
                interference = max(0.0, aggregate - signal)
                sinr[source, receiver] = signal / max(noise_power + interference, 1e-300)
            eligible = candidate_sources[sinr[candidate_sources, receiver] >= threshold_linear]
            if eligible.size:
                chosen = int(eligible[np.argmax(received_power[eligible, receiver])])
                decoded[chosen, receiver] = True
            for source in candidate_sources:
                if decoded[source, receiver]:
                    continue
                snr = float(received_power[source, receiver]) / max(noise_power, 1e-300)
                if snr >= threshold_linear:
                    interference_failures[source, receiver] = True
                else:
                    outage_failures[source, receiver] = True
        return decoded, sinr, interference_failures, outage_failures

    @staticmethod
    def _matrix_linear_to_db(values: np.ndarray) -> np.ndarray:
        result = np.full(values.shape, np.nan, dtype=float)
        valid = np.isfinite(values)
        result[valid] = linear_to_db(values[valid])
        return result
