from __future__ import annotations

from dataclasses import dataclass
from math import erf, isinf, sqrt
from typing import Iterable

import numpy as np

from .beam import beam_for_target, beam_matches, rotation_body_to_global, shift_beam
from .communication_phy import CommunicationPhy
from .config import SimulationConfig
from .mobility import NodeState, initialize_states, step_states
from .phy_sensing import detection_probability, radar_snr_db

MODE_SENSE = "sense"
MODE_TX = "tx"
MODE_RX = "rx"
MODE_IDLE = "idle"
ACCESS_BACKOFF = "backoff"
ACCESS_NORMAL = "normal"
ACCESS_AGGRESSIVE = "aggressive"

WANG2025_PROTOCOLS = (
    "wang2025_isac_no_collab",
    "wang2025_comm_tables",
    "wang2025_isac_tables",
)

ISAC_PROTOCOLS = (
    "improved_rl_isac",
    "improved_rl_isac_tables",
    "trust_gated_isac_tables",
    "collision_aware_isac",
    "budgeted_collision_aware_isac",
    "isac_structured_marl",
    "wang2025_isac_no_collab",
    "wang2025_comm_tables",
    "wang2025_isac_tables",
    "ablation_isac_one_slot_delay",
    "ablation_isac_no_candidate_set",
    "ablation_isac_no_beam_lock",
    "ablation_isac_no_topology",
)

WANG2025_COMM_TABLE_PROTOCOLS = ("wang2025_comm_tables", "wang2025_isac_tables")
WANG2025_SENSING_TABLE_PROTOCOLS = ("wang2025_isac_tables",)
OURS_TABLE_PROTOCOLS = ("improved_rl_isac_tables", "trust_gated_isac_tables")
TRUST_GATED_TABLE_PROTOCOLS = ("trust_gated_isac_tables",)
COMM_TABLE_PROTOCOLS = WANG2025_COMM_TABLE_PROTOCOLS + OURS_TABLE_PROTOCOLS
SENSING_TABLE_PROTOCOLS = WANG2025_SENSING_TABLE_PROTOCOLS + OURS_TABLE_PROTOCOLS
DELAYED_CANDIDATE_PROTOCOLS = ("ablation_isac_one_slot_delay",)

NO_ISAC_SENSING_PROTOCOLS = (
    "rl_no_isac",
    "basic_marl_no_isac",
    "improved_rl_no_isac",
    "structured_marl_no_isac",
)


@dataclass(frozen=True)
class Action:
    mode: str
    beam: int
    access_gate: str = ACCESS_NORMAL


@dataclass
class EpisodeResult:
    protocol: str
    episode: int
    seed: int
    scenario_seed: int
    slots: int
    mobility_model: str
    true_edges_seen: int
    discovered_edges: int
    discovery_rate: float
    mean_delay_censored: float
    p90_delay_censored: float
    p95_delay_censored: float
    p99_delay_censored: float
    empty_scan_ratio: float
    handshake_attempts: int
    handshake_successes: int
    forward_decode_failures: int
    ack_decode_failures: int
    interference_limited_failures: int
    phy_outage_failures: int
    mean_handshake_sinr_db: float
    p10_handshake_sinr_db: float
    collision_count: int
    empty_scan_count: int
    scan_actions: int
    tx_actions: int
    rx_actions: int
    sense_actions: int
    idle_actions: int
    piggyback_sense_actions: int
    sensing_observations: int
    sensing_target_observations: int
    sensing_detection_rate: float
    sensing_false_alarm_ratio: float
    sensing_miss_ratio: float
    mean_sensing_pd: float
    mean_sensing_snr_db: float
    discovery_per_scan_action: float
    discoveries_per_1000_scan_actions: float
    scan_actions_per_discovery_censored: float
    collisions_per_discovery_censored: float
    collision_normalized_efficiency: float
    collision_penalized_discovery_rate: float
    energy_j: float
    discoveries_per_joule: float
    energy_per_discovery_censored_j: float
    moved_distance_mean_m: float
    active_discovered_edges: int
    largest_component_size: int
    connected_components: int
    lcc_ratio: float
    isolated_node_ratio: float
    lambda2: float
    knowledge_lambda2: float

    def as_dict(self) -> dict[str, float | int | str]:
        return self.__dict__.copy()


class NeighborDiscoverySimulator:
    def __init__(self, config: SimulationConfig, protocol: str, seed: int, scenario_seed: int | None = None):
        self.cfg = config
        self.protocol = protocol
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.scenario_seed = seed if scenario_seed is None else scenario_seed
        self.mobility_rng = np.random.default_rng(self.scenario_seed)
        sensing_seed = np.random.SeedSequence([self.scenario_seed, self.seed, 0x15AC])
        self.sensing_rng = np.random.default_rng(sensing_seed)
        communication_seed = np.random.SeedSequence([self.scenario_seed, 0xC011])
        self.communication_rng = np.random.default_rng(communication_seed)
        if self.cfg.communication_phy_model != "ideal" and int(self.cfg.rf_chains) != 1:
            raise ValueError("SINR communication PHY currently requires exactly one RF chain.")
        self.communication_phy = CommunicationPhy(self.cfg, self.communication_rng)
        self.states: list[NodeState] = []
        self.initial_positions: np.ndarray | None = None
        self.belief = np.zeros((self.cfg.n_nodes, self.cfg.n_beams), dtype=float)
        self.age = np.zeros_like(self.belief)
        self.success_count = np.zeros_like(self.belief)
        self.fail_count = np.zeros_like(self.belief)
        self.collision_fail_count = np.zeros_like(self.belief)
        self.empty_beam_count = np.zeros_like(self.belief)
        self.last_positive_slot = np.full_like(self.belief, -10**9)
        self.wang_sensing_flag = np.zeros_like(self.belief)
        self.wang_node_num = np.full_like(self.belief, -1.0)
        self.wang_dis_num = np.zeros_like(self.belief)
        self.wang_snr_db = np.full_like(self.belief, -300.0)
        self.neighbor_records: list[dict[int, tuple[np.ndarray, int]]] = [
            {} for _ in range(self.cfg.n_nodes)
        ]
        self.sensing_report_position = np.full(
            (self.cfg.n_nodes, self.cfg.n_beams, 3), np.nan, dtype=float
        )
        self.sensing_report_confidence = np.zeros_like(self.belief)
        self.sensing_report_slot = np.full_like(self.belief, -10**9, dtype=int)
        self.discovered_edges: set[tuple[int, int]] = set()
        self.first_true_slot: dict[tuple[int, int], int] = {}
        self.discovery_slot: dict[tuple[int, int], int] = {}
        self.empty_scans = 0
        self.scan_actions = 0
        self.handshake_attempts = 0
        self.handshake_successes = 0
        self.forward_decode_failures = 0
        self.ack_decode_failures = 0
        self.interference_limited_failures = 0
        self.phy_outage_failures = 0
        self.handshake_sinr_samples_db: list[float] = []
        self.collision_count = 0
        self.node_empty_scans = np.zeros(self.cfg.n_nodes, dtype=int)
        self.node_scan_actions = np.zeros(self.cfg.n_nodes, dtype=int)
        self.node_collision_count = np.zeros(self.cfg.n_nodes, dtype=int)
        self.tx_actions = 0
        self.rx_actions = 0
        self.sense_actions = 0
        self.idle_actions = 0
        self.piggyback_sense_actions = 0
        self.sensing_observations = 0
        self.sensing_target_observations = 0
        self.sensing_detection_count = 0
        self.sensing_false_alarm_count = 0
        self.sensing_miss_count = 0
        self.sensing_pd_sum = 0.0
        self.sensing_snr_db_sum = 0.0
        self.sensing_snr_sample_count = 0
        self.last_sense_slot = np.full(self.cfg.n_nodes, -10**9, dtype=int)
        self.per_slot_rows: list[dict] = []
        self.edge_rows: list[dict] = []
        self._candidate_pool_cache: dict[tuple[int, int], np.ndarray] = {}
        self._pre_sensing_candidate_pool: dict[int, np.ndarray] = {}
        self._sensing_profile_cache: dict[tuple[float, bool], list[list[list[tuple[float, float]]]]] = {}
        self._beam_match_cache: dict[int, np.ndarray] = {}
        self._observable_beam_cache: dict[int, tuple[tuple[int, float], ...]] = {}
        self._angular_match_probability_cache: dict[tuple[int, int], float] = {}
        self._beam_matrix_cache: np.ndarray | None = None
        self._distance_matrix_cache: np.ndarray | None = None
        self._complete_edges_cache: set[tuple[int, int]] | None = None

    def reset(self) -> None:
        self.states = initialize_states(self.cfg.n_nodes, self.cfg.area_size_m, self.cfg.mobility, self.mobility_rng)
        self.initial_positions = np.asarray([s.position.copy() for s in self.states])
        self.belief.fill(0.5)
        self.age.fill(0.0)
        self.success_count.fill(0.0)
        self.fail_count.fill(0.0)
        self.collision_fail_count.fill(0.0)
        self.empty_beam_count.fill(0.0)
        self.last_positive_slot.fill(-10**9)
        self.wang_sensing_flag.fill(1.0)
        self.wang_node_num.fill(-1.0)
        self.wang_dis_num.fill(0.0)
        self.wang_snr_db.fill(-300.0)
        self.communication_phy.reset(self.cfg.n_nodes)
        self.neighbor_records = [{} for _ in range(self.cfg.n_nodes)]
        self.sensing_report_position.fill(np.nan)
        self.sensing_report_confidence.fill(0.0)
        self.sensing_report_slot.fill(-10**9)
        self.discovered_edges.clear()
        self.first_true_slot.clear()
        self.discovery_slot.clear()
        self.empty_scans = 0
        self.scan_actions = 0
        self.handshake_attempts = 0
        self.handshake_successes = 0
        self.forward_decode_failures = 0
        self.ack_decode_failures = 0
        self.interference_limited_failures = 0
        self.phy_outage_failures = 0
        self.handshake_sinr_samples_db = []
        self.collision_count = 0
        self.node_empty_scans.fill(0)
        self.node_scan_actions.fill(0)
        self.node_collision_count.fill(0)
        self.tx_actions = 0
        self.rx_actions = 0
        self.sense_actions = 0
        self.idle_actions = 0
        self.piggyback_sense_actions = 0
        self.sensing_observations = 0
        self.sensing_target_observations = 0
        self.sensing_detection_count = 0
        self.sensing_false_alarm_count = 0
        self.sensing_miss_count = 0
        self.sensing_pd_sum = 0.0
        self.sensing_snr_db_sum = 0.0
        self.sensing_snr_sample_count = 0
        self.last_sense_slot.fill(-10**9)
        self.per_slot_rows = []
        self.edge_rows = []
        self._candidate_pool_cache.clear()
        self._pre_sensing_candidate_pool.clear()
        self._sensing_profile_cache.clear()
        self._beam_match_cache.clear()
        self._observable_beam_cache.clear()
        self._angular_match_probability_cache.clear()
        self._beam_matrix_cache = None
        self._distance_matrix_cache = None
        self._complete_edges_cache = None

    def run_episode(self, episode: int) -> EpisodeResult:
        self.reset()
        for slot in range(self.cfg.slots_per_episode):
            self.invalidate_geometry_cache()
            true_comm_edges = self.true_edges(self.cfg.communication_range_m)
            for edge in true_comm_edges:
                self.first_true_slot.setdefault(edge, slot)
            self.age += 1.0
            self.belief *= self.cfg.confidence_decay
            self._candidate_pool_cache.clear()
            actions = self.select_actions(slot, true_comm_edges)
            self.snapshot_pre_sensing_candidates(slot)
            self.update_action_counts(actions, slot)
            self.update_empty_scan_counts(actions, true_comm_edges, slot)
            self.update_sensing(actions, slot)
            self._candidate_pool_cache.clear()
            new_edges = self.resolve_discoveries(slot, actions, true_comm_edges)
            if self.cfg.slot_metric_period > 0 and slot % self.cfg.slot_metric_period == 0:
                self.per_slot_rows.append(self.slot_metrics(episode, slot, true_comm_edges, new_edges))
            step_states(
                self.states,
                self.cfg.area_size_m,
                self.cfg.mobility,
                self.cfg.slot_duration_s,
                slot,
                self.mobility_rng,
            )
            self.invalidate_geometry_cache()
        return self.summarize(episode)

    def invalidate_geometry_cache(self) -> None:
        self._beam_matrix_cache = None
        self._distance_matrix_cache = None
        self._sensing_profile_cache.clear()

    def slot_metrics(
        self,
        episode: int,
        slot: int,
        true_comm_edges: set[tuple[int, int]],
        new_edges: list[tuple[int, int]],
    ) -> dict:
        active_edges = self.discovered_edges.intersection(true_comm_edges)
        components = connected_components(self.cfg.n_nodes, active_edges)
        largest = max((len(c) for c in components), default=0)
        isolated = sum(1 for c in components if len(c) == 1)
        return {
            "episode": episode,
            "protocol": self.protocol,
            "slot": slot,
            "mobility_model": str(self.cfg.mobility.get("model", "gauss_markov")),
            "true_edges": len(true_comm_edges),
            "true_edges_seen": len(self.first_true_slot),
            "discovered_edges": len(self.discovered_edges),
            "active_discovered_edges": len(active_edges),
            "new_edges": len(new_edges),
            "empty_scan_ratio": self.empty_scans / max(1, self.scan_actions),
            "handshake_attempts": self.handshake_attempts,
            "handshake_successes": self.handshake_successes,
            "forward_decode_failures": self.forward_decode_failures,
            "ack_decode_failures": self.ack_decode_failures,
            "interference_limited_failures": self.interference_limited_failures,
            "phy_outage_failures": self.phy_outage_failures,
            "mean_handshake_sinr_db": float(np.mean(self.handshake_sinr_samples_db))
            if self.handshake_sinr_samples_db
            else 0.0,
            "collision_count": self.collision_count,
            "empty_scan_count": self.empty_scans,
            "scan_actions": self.scan_actions,
            "tx_actions": self.tx_actions,
            "rx_actions": self.rx_actions,
            "sense_actions": self.sense_actions,
            "idle_actions": self.idle_actions,
            "piggyback_sense_actions": self.piggyback_sense_actions,
            "sensing_observations": self.sensing_observations,
            "sensing_target_observations": self.sensing_target_observations,
            "sensing_detection_rate": self.sensing_detection_count / max(1, self.sensing_target_observations),
            "sensing_false_alarm_ratio": self.sensing_false_alarm_count / max(1, self.sensing_observations),
            "sensing_miss_ratio": self.sensing_miss_count / max(1, self.sensing_target_observations),
            "mean_sensing_pd": self.sensing_pd_sum / max(1, self.sensing_target_observations),
            "mean_sensing_snr_db": self.sensing_snr_db_sum / max(1, self.sensing_snr_sample_count),
            "largest_component_size": largest,
            "connected_components": len(components),
            "lcc_ratio": largest / max(1, self.cfg.n_nodes),
            "isolated_node_ratio": isolated / max(1, self.cfg.n_nodes),
            "lambda2": algebraic_connectivity(self.cfg.n_nodes, active_edges),
            "knowledge_lambda2": algebraic_connectivity(self.cfg.n_nodes, self.discovered_edges),
        }

    def positions(self) -> np.ndarray:
        return np.asarray([state.position for state in self.states])

    def area_diagonal(self) -> float:
        return float(np.sqrt(sum(float(value) ** 2 for value in self.cfg.area_size_m)))

    def complete_edges(self) -> set[tuple[int, int]]:
        if self._complete_edges_cache is None:
            self._complete_edges_cache = {
                (i, j) for i in range(self.cfg.n_nodes) for j in range(i + 1, self.cfg.n_nodes)
            }
        return self._complete_edges_cache

    def distance_matrix(self) -> np.ndarray:
        if self._distance_matrix_cache is None:
            pos = self.positions()
            delta = pos[:, None, :] - pos[None, :, :]
            self._distance_matrix_cache = np.einsum("ijk,ijk->ij", delta, delta)
        return self._distance_matrix_cache

    def beam_matrix(self) -> np.ndarray:
        if self._beam_matrix_cache is not None:
            return self._beam_matrix_cache
        pos = self.positions()
        n_nodes = self.cfg.n_nodes
        beams = np.zeros((n_nodes, n_nodes), dtype=int)
        for source, state in enumerate(self.states):
            delta = pos - pos[source]
            norm = np.linalg.norm(delta, axis=1, keepdims=True)
            norm[norm <= 1e-12] = 1.0
            direction_global = delta / norm
            direction_global[source] = np.asarray([1.0, 0.0, 0.0])
            rotation = rotation_body_to_global(state.yaw, state.pitch, state.roll)
            direction_body = direction_global @ rotation
            direction_norm = np.linalg.norm(direction_body, axis=1)
            direction_norm[direction_norm <= 1e-12] = 1.0
            unit = direction_body / direction_norm[:, None]
            azimuth = np.arctan2(unit[:, 1], unit[:, 0])
            elevation = np.arcsin(np.clip(unit[:, 2], -1.0, 1.0))
            az_idx = np.floor((azimuth + np.pi) / (2.0 * np.pi) * self.cfg.azimuth_cells).astype(int)
            az_idx %= self.cfg.azimuth_cells
            el_idx = np.floor((elevation + np.pi / 2.0) / np.pi * self.cfg.elevation_cells).astype(int)
            el_idx = np.clip(el_idx, 0, self.cfg.elevation_cells - 1)
            beams[source] = el_idx * self.cfg.azimuth_cells + az_idx
        self._beam_matrix_cache = beams
        return beams

    def true_edges(self, radius: float) -> set[tuple[int, int]]:
        if self.cfg.n_nodes < 2:
            return set()
        if float(radius) >= self.area_diagonal():
            return self.complete_edges()
        dist2 = self.distance_matrix()
        mask = np.triu(dist2 <= float(radius) ** 2, k=1)
        rows, cols = np.nonzero(mask)
        return set(zip(rows.astype(int).tolist(), cols.astype(int).tolist()))

    def occupied_beams(self, radius: float) -> list[set[int]]:
        occupied = [set() for _ in range(self.cfg.n_nodes)]
        beams = self.beam_matrix()
        if float(radius) >= self.area_diagonal():
            for i in range(self.cfg.n_nodes):
                occupied[i].update(int(value) for value in np.delete(beams[i], i))
            return occupied
        dist2 = self.distance_matrix()
        within = dist2 <= float(radius) ** 2
        np.fill_diagonal(within, False)
        for i in range(self.cfg.n_nodes):
            occupied[i].update(int(value) for value in beams[i, within[i]])
        return occupied

    def beam_from_to(self, source: int, target: int) -> int:
        return beam_for_target(
            self.states[source],
            self.states[target].position,
            self.cfg.azimuth_cells,
            self.cfg.elevation_cells,
        )

    def select_actions(self, slot: int, true_comm_edges: set[tuple[int, int]]) -> list[Action]:
        actions: list[Action] = []
        if self.protocol == "oracle":
            oracle_occupied = self.occupied_beams(self.cfg.communication_range_m)
        else:
            oracle_occupied = [set() for _ in range(self.cfg.n_nodes)]
        for node in range(self.cfg.n_nodes):
            mode = self.select_mode(node, slot)
            beam = self.select_beam(node, slot, mode, oracle_occupied[node], true_comm_edges)
            actions.append(Action(mode, beam))
        return actions

    def select_mode(self, node: int, slot: int) -> str:
        if self.protocol == "deterministic_scan":
            return MODE_TX if (slot + node) % 2 == 0 else MODE_RX
        if self.protocol == "crt_oblivious_like":
            return MODE_TX if (slot + 2 * node) % 5 in (0, 1) else MODE_RX
        if self.protocol == "skyorbs_like_skip_scan":
            return MODE_TX if (slot // 2 + node) % 2 == 0 else MODE_RX
        if self.protocol in WANG2025_PROTOCOLS:
            p_tx = float(np.clip(self.cfg.p_tx, 0.05, 0.95))
            return MODE_TX if self.rng.random() < p_tx else MODE_RX
        if self.protocol in ("rl_no_isac", "basic_marl_no_isac"):
            return str(self.rng.choice([MODE_TX, MODE_RX, MODE_IDLE], p=[0.45, 0.45, 0.10]))
        if self.protocol in ("improved_rl_no_isac", "structured_marl_no_isac"):
            degree = sum(1 for edge in self.discovered_edges if node in edge)
            degree_need = max(0.0, self.cfg.target_degree - degree) / max(1, self.cfg.target_degree)
            idle_prob = 0.02 + 0.04 * (1.0 - degree_need)
            tx_prob = 0.49 + 0.04 * degree_need
            rx_prob = max(0.05, 1.0 - idle_prob - tx_prob)
            probs = np.asarray([tx_prob, rx_prob, idle_prob], dtype=float)
            probs = probs / probs.sum()
            return str(self.rng.choice([MODE_TX, MODE_RX, MODE_IDLE], p=probs))
        if self.protocol in ISAC_PROTOCOLS:
            degree = sum(1 for edge in self.discovered_edges if node in edge)
            degree_need = max(0.0, self.cfg.target_degree - degree) / max(1, self.cfg.target_degree)
            if self.protocol == "collision_aware_isac":
                probs = self.collision_aware_role_probabilities(node, slot, degree_need)
            elif self.protocol == "budgeted_collision_aware_isac":
                probs = self.budgeted_collision_aware_role_probabilities(node, slot, degree_need)
            else:
                idle_prob = 0.01 + 0.03 * (1.0 - degree_need)
                tx_prob = 0.50 + 0.05 * degree_need
                rx_prob = max(0.05, 1.0 - idle_prob - tx_prob)
                probs = np.asarray([tx_prob, rx_prob, idle_prob], dtype=float)
            probs = probs / probs.sum()
            return str(self.rng.choice([MODE_TX, MODE_RX, MODE_IDLE], p=probs))
        if self.protocol in ("isac_only", "topology_only", "itap_nd"):
            if self.should_sense(node, slot):
                return MODE_SENSE
            return MODE_TX if (slot + node) % 2 == 0 else MODE_RX
        probs = np.asarray([self.cfg.p_sense, self.cfg.p_tx, self.cfg.p_rx, self.cfg.p_idle], dtype=float)
        probs = probs / probs.sum()
        return str(self.rng.choice([MODE_SENSE, MODE_TX, MODE_RX, MODE_IDLE], p=probs))

    def should_sense(self, node: int, slot: int) -> bool:
        if self.protocol == "topology_only":
            return False
        period = max(1, self.cfg.sensing_period_slots)
        if self.protocol in ISAC_PROTOCOLS:
            period = max(1, 2 * period)
        staggered_periodic_sense = slot % period == node % period
        if self.protocol in ISAC_PROTOCOLS:
            staggered_periodic_sense = slot % period == (2 * node) % period
        stale_belief = float(np.mean(self.age[node])) >= 2.0 * period
        weak_belief = float(np.max(self.belief[node])) < 0.2
        if self.protocol in ISAC_PROTOCOLS:
            return bool(staggered_periodic_sense)
        return bool(staggered_periodic_sense or stale_belief or weak_belief)

    def collision_aware_role_probabilities(self, node: int, slot: int, degree_need: float) -> np.ndarray:
        candidate_pool = self.isac_candidate_pool(node, slot)
        candidate_pressure = min(1.0, len(candidate_pool) / max(1.0, float(self.cfg.target_degree)))
        failure_pressure = 0.0
        collision_pressure = 0.0
        if len(candidate_pool) > 0:
            success = self.success_count[node, candidate_pool]
            fail = self.fail_count[node, candidate_pool]
            collision_fail = self.collision_fail_count[node, candidate_pool]
            failure_pressure = float(np.mean(fail / np.maximum(1.0, success + fail)))
            failure_pressure = float(np.clip(failure_pressure, 0.0, 1.0))
            collision_pressure = float(np.mean(collision_fail / np.maximum(1.0, success + collision_fail)))
            collision_pressure = float(np.clip(collision_pressure, 0.0, 1.0))

        idle_prob = 0.01 + 0.03 * (1.0 - degree_need)
        tx_prob = 0.50 + 0.05 * degree_need
        tx_prob -= 0.12 * candidate_pressure + 0.12 * collision_pressure + 0.04 * failure_pressure
        tx_prob = float(np.clip(tx_prob, 0.28, 0.55))
        rx_prob = max(0.05, 1.0 - idle_prob - tx_prob)
        return np.asarray([tx_prob, rx_prob, idle_prob], dtype=float)

    def budgeted_collision_aware_role_probabilities(self, node: int, slot: int, degree_need: float) -> np.ndarray:
        """Density-adaptive ISAC access budget for collision-constrained discovery.

        The rule is intentionally local at execution time: it uses configured
        beam-codebook size, local discovered degree, and this node's public
        success/failure/collision memory. It is a rule expert for the later MARL
        access controller, not an oracle over true neighbors.
        """

        candidate_pool = self.isac_candidate_pool(node, slot)
        candidate_pressure = min(1.0, len(candidate_pool) / max(1.0, float(self.cfg.target_degree)))
        candidate_evidence = 0.0
        failure_pressure = 0.0
        collision_pressure = 0.0
        if len(candidate_pool) > 0:
            belief = self.belief[node, candidate_pool]
            success = self.success_count[node, candidate_pool]
            fail = self.fail_count[node, candidate_pool]
            collision_fail = self.collision_fail_count[node, candidate_pool]
            candidate_evidence = float(np.clip(np.mean(belief + 0.15 * np.log1p(success)), 0.0, 1.0))
            failure_pressure = float(np.mean(fail / np.maximum(1.0, success + fail)))
            failure_pressure = float(np.clip(failure_pressure, 0.0, 1.0))
            collision_pressure = float(np.mean(collision_fail / np.maximum(1.0, success + collision_fail)))
            collision_pressure = float(np.clip(collision_pressure, 0.0, 1.0))

        density = self.cfg.n_nodes / max(1.0, float(self.cfg.n_beams))
        contention_scale = np.sqrt(1.0 + 4.0 * density * max(1.0, float(self.cfg.target_degree)))
        base_tx = 0.48 / contention_scale
        tx_prob = (
            base_tx
            + 0.08 * float(degree_need)
            + 0.05 * candidate_evidence
            - 0.08 * candidate_pressure
            - 0.18 * collision_pressure
            - 0.08 * failure_pressure
        )
        if degree_need <= 0.0:
            tx_prob -= 0.08
        tx_prob = float(np.clip(tx_prob, 0.16, 0.45))
        idle_prob = 0.01 + 0.02 * (1.0 - float(degree_need)) + 0.04 * collision_pressure
        idle_prob = float(np.clip(idle_prob, 0.01, 0.09))
        rx_prob = max(0.05, 1.0 - idle_prob - tx_prob)
        return np.asarray([tx_prob, rx_prob, idle_prob], dtype=float)

    def select_beam(
        self,
        node: int,
        slot: int,
        mode: str,
        oracle_occupied: set[int],
        true_comm_edges: set[tuple[int, int]],
    ) -> int:
        if mode == MODE_IDLE:
            return 0
        if self.protocol == "uniform_random":
            return int(self.rng.integers(0, self.cfg.n_beams))
        if self.protocol == "deterministic_scan":
            return int((slot + node) % self.cfg.n_beams)
        if self.protocol == "crt_oblivious_like":
            return int(((slot + 1) * (2 * node + 1) + node) % self.cfg.n_beams)
        if self.protocol == "skyorbs_like_skip_scan":
            return self.skyorbs_like_beam(node, slot)
        if self.protocol in WANG2025_PROTOCOLS:
            return self.wang2025_table_beam(node)
        if self.protocol == "oracle" and oracle_occupied:
            return int(self.rng.choice(tuple(oracle_occupied)))
        if self.protocol in ISAC_PROTOCOLS and mode == MODE_SENSE:
            return self.isac_sensing_beam(node, slot)
        if self.protocol in ("rl_no_isac", "basic_marl_no_isac"):
            return self.memory_guided_beam(node, use_isac=False, topology=False)
        if self.protocol in ("improved_rl_no_isac", "structured_marl_no_isac"):
            return self.memory_guided_beam(node, use_isac=False, topology=True)
        if self.protocol in ISAC_PROTOCOLS:
            isac_beam = self.isac_candidate_cycle_beam(node, slot)
            if isac_beam is not None:
                return isac_beam
            return self.memory_guided_beam(
                node,
                use_isac=True,
                topology=self.protocol != "ablation_isac_no_topology",
            )

        weights = np.ones(self.cfg.n_beams, dtype=float) * self.cfg.exploration_floor
        if self.protocol in ("isac_only", "itap_nd", "oracle"):
            weights += self.cfg.alpha_occupancy * np.maximum(self.belief[node], 0.0)
        if self.protocol in ("topology_only", "itap_nd"):
            degree = sum(1 for edge in self.discovered_edges if node in edge)
            degree_need = max(0.0, self.cfg.target_degree - degree) / max(1, self.cfg.target_degree)
            diversity = 1.0 / (1.0 + self.success_count[node])
            staleness = self.age[node] / max(1.0, self.age[node].max())
            weights += self.cfg.beta_diversity * diversity
            weights += self.cfg.gamma_degree_need * degree_need
            weights += self.cfg.eta_staleness * staleness
        weights = np.maximum(weights, self.cfg.exploration_floor / self.cfg.n_beams)
        if self.protocol in ("isac_only", "itap_nd"):
            return self.sample_prioritized_beam(weights)
        if self.cfg.softmax_beta > 0:
            logits = self.cfg.softmax_beta * (weights - weights.max())
            probs = np.exp(logits)
        else:
            probs = weights
        probs = probs / probs.sum()
        return int(self.rng.choice(self.cfg.n_beams, p=probs))

    def skyorbs_like_beam(self, node: int, slot: int) -> int:
        """Discrete 3D skip-scan approximation of SkyOrbs-style scanning."""

        empty_pressure = self.empty_scans / max(1, self.scan_actions)
        dynamic_stride = max(1, int(round(np.sqrt(self.cfg.azimuth_cells)))) + int(2.0 * empty_pressure)
        ring_period = max(1, self.cfg.azimuth_cells // max(1, dynamic_stride))
        elevation = (node + slot // ring_period) % self.cfg.elevation_cells
        if (slot // max(1, self.cfg.elevation_cells)) % 2 == 1:
            elevation = self.cfg.elevation_cells - 1 - elevation
        azimuth = (node * dynamic_stride + slot * dynamic_stride) % self.cfg.azimuth_cells
        return int(elevation * self.cfg.azimuth_cells + azimuth)

    def wang2025_table_beam(self, node: int) -> int:
        """Rule-based ISAC table baseline inspired by Wang et al. 2025.

        Wang's neighbor-discovery process selects directions randomly from the
        sensing-table beam list whose Flag is 1. The table keeps Node_num
        (potential targets) and Dis_num (already interacted targets); a beam is
        closed when sensing finds no target or when Dis_num reaches Node_num.
        """

        active = self.wang2025_flagged_beams(node)
        if len(active) == 0:
            active = np.arange(self.cfg.n_beams, dtype=int)
        return int(self.rng.choice(active))

    def wang2025_flagged_beams(self, node: int) -> np.ndarray:
        return np.flatnonzero(self.wang_sensing_flag[int(node)] > 0.5).astype(int)

    def wang2025_positive_beams(self, node: int) -> np.ndarray:
        node = int(node)
        positive = (
            (self.wang_sensing_flag[node] > 0.5)
            & (self.wang_node_num[node] > 0.0)
            & (self.wang_dis_num[node] < self.wang_node_num[node])
        )
        return np.flatnonzero(positive).astype(int)

    def isac_sensing_beam(self, node: int, slot: int) -> int:
        period = max(1, self.cfg.sensing_period_slots)
        if self.protocol in ISAC_PROTOCOLS:
            period = max(1, 2 * period)
        sense_round = slot // period
        prime_stride = self.near_coprime_stride(self.cfg.n_beams)
        return int((node * prime_stride + sense_round * prime_stride) % self.cfg.n_beams)

    def isac_candidate_cycle_beam(self, node: int, slot: int) -> int | None:
        selected = self.isac_candidate_pool(node, slot)
        if len(selected) == 0:
            return None
        if self.rng.random() < self.cfg.exploration_floor:
            return None
        if self.protocol == "ablation_isac_no_beam_lock":
            success = self.success_count[node, selected]
            fail = self.fail_count[node, selected]
            score = self.belief[node, selected] + 0.15 * np.log1p(success) - 0.20 * np.log1p(fail)
            logits = self.cfg.softmax_beta * (score - np.max(score)) if self.cfg.softmax_beta > 0 else score
            probs = np.exp(logits)
            probs = probs / probs.sum()
            return int(self.rng.choice(selected, p=probs))
        lock_period = max(2, int(round(0.04 / max(self.cfg.slot_duration_s, 1e-6))))
        selected_index = int((slot // lock_period + node) % len(selected))
        return int(selected[selected_index])

    def snapshot_pre_sensing_candidates(self, slot: int) -> None:
        if self.protocol not in DELAYED_CANDIDATE_PROTOCOLS:
            self._pre_sensing_candidate_pool.clear()
            return
        self._pre_sensing_candidate_pool = {
            node: self.isac_candidate_pool(node, slot).copy() for node in range(self.cfg.n_nodes)
        }

    def handshake_candidate_pool(self, node: int, slot: int) -> np.ndarray:
        if self.protocol in DELAYED_CANDIDATE_PROTOCOLS:
            return self._pre_sensing_candidate_pool.get(int(node), np.asarray([], dtype=int))
        return self.isac_candidate_pool(node, slot)

    def isac_candidate_pool(self, node: int, slot: int) -> np.ndarray:
        cache_key = (int(slot), int(node))
        cached = self._candidate_pool_cache.get(cache_key)
        if cached is not None:
            return cached
        belief = self.belief[node]
        positive_ttl = max(50, min(300, int(round(0.50 / max(self.cfg.slot_duration_s, 1e-6)))))
        recent_positive = (slot - self.last_positive_slot[node]) <= positive_ttl
        candidate_mask = (belief >= 0.55) | (
            (self.success_count[node] > 0.05) & (self.empty_beam_count[node] < 1.0) & recent_positive
        )
        if not np.any(candidate_mask):
            empty = np.asarray([], dtype=int)
            self._candidate_pool_cache[cache_key] = empty
            return empty
        candidates = np.flatnonzero(candidate_mask)
        success = self.success_count[node, candidates]
        fail = self.fail_count[node, candidates]
        recency = np.maximum(0.0, slot - self.last_positive_slot[node, candidates])
        score = belief[candidates] + 0.35 * np.log1p(success) - 0.25 * np.log1p(fail) - 0.002 * recency
        order = np.lexsort((candidates, -score))
        max_candidates = max(1, min(len(candidates), max(2, self.cfg.target_degree)))
        selected = candidates[order[:max_candidates]]
        self._candidate_pool_cache[cache_key] = selected
        return selected

    def handshake_beam_ready(self, node: int, selected_beam: int, expected_beam: int, slot: int) -> bool:
        return beam_matches(selected_beam, expected_beam, self.cfg.azimuth_cells, self.cfg.alignment_tolerance_cells)

    def near_coprime_stride(self, modulus: int) -> int:
        stride = max(1, int(round(np.sqrt(modulus))))
        while np.gcd(stride, modulus) != 1:
            stride += 1
        return stride

    def memory_guided_beam(self, node: int, use_isac: bool, topology: bool) -> int:
        """Local-memory policy proxy used before trained MARL policies are wired in."""

        age = self.age[node] / max(1.0, self.age[node].max())
        success = self.success_count[node]
        fail = self.fail_count[node]
        if use_isac:
            belief = self.belief[node]
            non_rejected = np.flatnonzero((self.empty_beam_count[node] < 1.0) | (self.success_count[node] > 0.0))
            if len(non_rejected) > 0:
                oldest = self.age[node, non_rejected]
                max_age = float(np.max(oldest)) if len(oldest) else 0.0
                old_enough = non_rejected[oldest >= max_age]
                return int(self.rng.choice(old_enough))
        uncertainty = 1.0 / np.sqrt(1.0 + success + fail)
        weights = np.ones(self.cfg.n_beams, dtype=float) * self.cfg.exploration_floor
        weights += 0.25 * age
        weights += 0.40 * uncertainty
        weights += 0.15 * np.log1p(success)
        weights -= 0.25 * np.log1p(fail)
        if topology:
            degree = sum(1 for edge in self.discovered_edges if node in edge)
            degree_need = max(0.0, self.cfg.target_degree - degree) / max(1, self.cfg.target_degree)
            diversity = 1.0 / (1.0 + success)
            weights += self.cfg.beta_diversity * diversity
            weights += self.cfg.gamma_degree_need * degree_need
        if use_isac:
            belief = np.maximum(self.belief[node], 0.0)
            risk = belief * (1.0 - belief)
            weights += self.cfg.alpha_occupancy * belief
            weights += 0.25 * risk
        weights = np.maximum(weights, self.cfg.exploration_floor / self.cfg.n_beams)
        if self.cfg.softmax_beta > 0:
            logits = self.cfg.softmax_beta * (weights - weights.max())
            probs = np.exp(logits)
        else:
            probs = weights
        probs = probs / probs.sum()
        return int(self.rng.choice(self.cfg.n_beams, p=probs))

    def sample_prioritized_beam(self, weights: np.ndarray) -> int:
        if self.rng.random() < self.cfg.exploration_floor:
            return int(self.rng.integers(0, self.cfg.n_beams))
        candidate_count = min(self.cfg.n_beams, max(4, int(np.sqrt(self.cfg.n_beams))))
        candidate_indices = np.argpartition(weights, -candidate_count)[-candidate_count:]
        candidate_weights = weights[candidate_indices]
        if self.cfg.softmax_beta > 0:
            logits = self.cfg.softmax_beta * (candidate_weights - candidate_weights.max())
            probs = np.exp(logits)
        else:
            probs = candidate_weights
        probs = probs / probs.sum()
        return int(self.rng.choice(candidate_indices, p=probs))

    def active_beams_for_action(self, node: int, action: Action, slot: int | None = None) -> np.ndarray:
        """Beams activated by one node in one slot under a multi-RF abstraction."""

        if action.mode == MODE_IDLE:
            return np.asarray([], dtype=int)
        beams: list[int] = [int(action.beam)]
        rf_chains = max(1, int(self.cfg.rf_chains))
        if rf_chains <= 1:
            return np.asarray(beams, dtype=int)

        if self.protocol in ISAC_PROTOCOLS and slot is not None:
            for candidate in self.isac_candidate_pool(node, slot):
                candidate = int(candidate)
                if candidate not in beams:
                    beams.append(candidate)
                if len(beams) >= rf_chains:
                    return np.asarray(beams, dtype=int)

        stride = self.near_coprime_stride(self.cfg.n_beams)
        k = 1
        while len(beams) < rf_chains and len(beams) < self.cfg.n_beams:
            candidate = int((action.beam + k * stride) % self.cfg.n_beams)
            if candidate not in beams:
                beams.append(candidate)
            k += 1
        return np.asarray(beams, dtype=int)

    def update_sensing(self, actions: list[Action], slot: int | None = None) -> None:
        if self.protocol in NO_ISAC_SENSING_PROTOCOLS:
            return
        eligible_nodes: list[tuple[int, Action]] = []
        for node, action in enumerate(actions):
            piggyback_isac = self.action_has_piggyback_isac(action)
            if action.mode != MODE_SENSE and not piggyback_isac:
                continue
            if slot is not None:
                period = max(1, self.cfg.sensing_period_slots)
                if piggyback_isac:
                    period = max(1, int(round(period * self.cfg.piggyback_sensing_period_multiplier)))
                if slot - int(self.last_sense_slot[node]) < period:
                    continue
            eligible_nodes.append((node, action))
        if not eligible_nodes:
            return
        useful_sensing_range = self.cfg.sensing_range_m
        ideal_sensing = (
            self.cfg.isac_sensing_model == "constant_error"
            and self.cfg.false_alarm_rate <= 0.0
            and self.cfg.miss_detection_rate <= 0.0
            and self.cfg.angular_cell_offset_std <= 0.0
        )
        for node, action in eligible_nodes:
            if slot is not None:
                self.last_sense_slot[node] = slot
            observation = self.belief[node].copy()
            piggyback_isac = self.action_has_piggyback_isac(action)
            if piggyback_isac:
                sensed_beams = self.sensing_sectors_for_action(node, action, slot)
                for sensed_beam in sensed_beams:
                    if ideal_sensing:
                        observed_value = 1.0 if self.beam_has_sensing_target(node, int(sensed_beam), useful_sensing_range) else 0.0
                        self.record_ideal_sensing_observation(
                            node,
                            int(sensed_beam),
                            useful_sensing_range,
                            piggyback=True,
                            observed_value=observed_value,
                        )
                    else:
                        observed_value = self.sample_sensing_observation(
                            node,
                            int(sensed_beam),
                            useful_sensing_range,
                            piggyback=True,
                        )
                    observation[sensed_beam] = observed_value
                    if observed_value <= 0.0:
                        self.fail_count[node, sensed_beam] += 0.10
                        self.success_count[node, sensed_beam] *= 0.90
                        if self.success_count[node, sensed_beam] <= 0.05:
                            self.empty_beam_count[node, sensed_beam] += 1.0
                    else:
                        self.success_count[node, sensed_beam] += 0.25
                        self.fail_count[node, sensed_beam] *= 0.5
                        self.empty_beam_count[node, sensed_beam] = 0.0
                        if slot is not None:
                            self.last_positive_slot[node, sensed_beam] = slot
                    self.update_wang2025_sensing_table_from_observation(
                        node,
                        int(sensed_beam),
                        observed_value,
                        slot,
                        useful_sensing_range,
                        piggyback=True,
                    )
                    if observed_value > 0.0 and slot is not None:
                        self.record_sensing_position_estimate(
                            node, int(sensed_beam), useful_sensing_range, slot
                        )
                rho = self.cfg.belief_update_rho
                self.belief[node] = (1.0 - rho) * self.belief[node] + rho * observation
                self.age[node, sensed_beams] = 0.0
                continue
            sensed_beams = self.sensing_sectors_for_action(node, action, slot)
            for sensed_beam in sensed_beams:
                if ideal_sensing:
                    observed_value = 1.0 if self.beam_has_sensing_target(node, int(sensed_beam), useful_sensing_range) else 0.0
                    self.record_ideal_sensing_observation(
                        node,
                        int(sensed_beam),
                        useful_sensing_range,
                        piggyback=False,
                        observed_value=observed_value,
                    )
                else:
                    observed_value = self.sample_sensing_observation(
                        node,
                        int(sensed_beam),
                        useful_sensing_range,
                        piggyback=False,
                    )
                observation[sensed_beam] = observed_value
                self.update_wang2025_sensing_table_from_observation(
                    node,
                    int(sensed_beam),
                    observed_value,
                    slot,
                    useful_sensing_range,
                    piggyback=False,
                )
                if observed_value > 0.0 and slot is not None:
                    self.record_sensing_position_estimate(
                        node, int(sensed_beam), useful_sensing_range, slot
                    )
            rho = self.cfg.belief_update_rho
            self.belief[node] = (1.0 - rho) * self.belief[node] + rho * observation
            self.age[node, sensed_beams] = 0.0

    def update_wang2025_sensing_table_from_observation(
        self,
        node: int,
        beam: int,
        observed_value: float,
        slot: int | None,
        radius: float,
        *,
        piggyback: bool,
    ) -> None:
        if self.protocol not in WANG2025_PROTOCOLS:
            return
        node = int(node)
        beam = int(beam)
        if observed_value <= 0.0:
            self.wang_node_num[node, beam] = 0.0
            self.wang_dis_num[node, beam] = 0.0
            self.wang_sensing_flag[node, beam] = 0.0
            return

        sensed_targets = self.wang2025_sensing_target_count(node, beam, radius)
        estimated_targets = max(1, int(sensed_targets))
        self.wang_node_num[node, beam] = max(float(estimated_targets), self.wang_node_num[node, beam])
        self.wang_snr_db[node, beam] = max(
            self.wang_snr_db[node, beam],
            self.max_sensing_snr_db(node, beam, radius, piggyback),
        )
        if slot is not None:
            self.last_positive_slot[node, beam] = slot
        self.update_wang2025_flag_after_counts(node, beam)

    def wang2025_sensing_target_count(self, node: int, sensed_beam: int, radius: float) -> int:
        beams = self.beam_matrix()
        dist2 = self.distance_matrix()
        count = 0
        for target in range(self.cfg.n_nodes):
            if target == node or dist2[node, target] > float(radius) ** 2:
                continue
            if beam_matches(
                int(sensed_beam),
                int(beams[node, target]),
                self.cfg.azimuth_cells,
                self.cfg.alignment_tolerance_cells,
            ):
                count += 1
        return count

    def update_wang2025_flag_after_counts(self, node: int, beam: int) -> None:
        node_num = float(self.wang_node_num[node, beam])
        if node_num <= 0.0:
            self.wang_sensing_flag[node, beam] = 0.0
            return
        self.wang_sensing_flag[node, beam] = 1.0 if self.wang_dis_num[node, beam] < node_num else 0.0

    def mark_wang2025_interaction(self, node: int, beam: int, slot: int) -> None:
        if self.protocol not in WANG2025_PROTOCOLS:
            return
        node = int(node)
        beam = int(beam)
        self.wang_dis_num[node, beam] += 1.0
        if self.wang_node_num[node, beam] < 0.0:
            self.wang_node_num[node, beam] = self.wang_dis_num[node, beam]
        else:
            self.wang_node_num[node, beam] = max(self.wang_node_num[node, beam], self.wang_dis_num[node, beam])
        self.wang_snr_db[node, beam] = max(self.wang_snr_db[node, beam], -300.0)
        self.success_count[node, beam] += 0.25
        self.empty_beam_count[node, beam] = 0.0
        self.last_positive_slot[node, beam] = slot
        self.update_wang2025_flag_after_counts(node, beam)

    def boost_wang2025_candidate(self, node: int, beam: int, slot: int, *, target_count: int = 1) -> None:
        if self.protocol not in WANG2025_PROTOCOLS:
            return
        node = int(node)
        beam = int(beam)
        self.wang_node_num[node, beam] = max(self.wang_node_num[node, beam], float(max(1, int(target_count))))
        self.wang_sensing_flag[node, beam] = 1.0
        self.empty_beam_count[node, beam] = 0.0
        self.last_positive_slot[node, beam] = slot

    def action_has_piggyback_isac(self, action: Action) -> bool:
        return self.protocol in ISAC_PROTOCOLS and action.mode == MODE_TX

    def sensing_sectors_for_action(self, node: int, action: Action, slot: int | None) -> np.ndarray:
        sector_beams: list[int] = []
        for active_beam in self.active_beams_for_action(node, action, slot):
            sector_beams.extend(self.sensing_sector(int(active_beam)).astype(int).tolist())
        if not sector_beams:
            return np.asarray([], dtype=int)
        return np.asarray(sorted(set(sector_beams)), dtype=int)

    def beam_has_sensing_target(self, node: int, sensed_beam: int, radius: float) -> bool:
        beams = self.beam_matrix()
        dist2 = self.distance_matrix()
        for target in range(self.cfg.n_nodes):
            if target == node or dist2[node, target] > float(radius) ** 2:
                continue
            if beam_matches(sensed_beam, int(beams[node, target]), self.cfg.azimuth_cells, self.cfg.alignment_tolerance_cells):
                return True
        return False

    def sample_sensing_observation(self, node: int, sensed_beam: int, radius: float, *, piggyback: bool) -> float:
        true_sector_candidates: list[tuple[float, float]] = []
        profile = self.sensing_profile(radius, piggyback)
        for true_beam in self.matching_true_beams(int(sensed_beam)):
            true_sector_candidates.extend(profile[node][int(true_beam)])

        has_target_in_sector = bool(true_sector_candidates)
        detected = False
        max_pd = max((pd for pd, _snr_db in true_sector_candidates), default=0.0)
        max_snr_db = max((snr_db for _pd, snr_db in true_sector_candidates), default=-300.0)
        for true_beam, angular_match_probability in self.observable_true_beams(int(sensed_beam)):
            for pd, _snr_db in profile[node][int(true_beam)]:
                if self.sensing_rng.random() < pd * angular_match_probability:
                    detected = True
                    break
            if detected:
                break

        self.sensing_observations += 1
        if has_target_in_sector:
            self.sensing_target_observations += 1
            self.sensing_pd_sum += max_pd
            if max_snr_db > -300.0:
                self.sensing_snr_db_sum += max_snr_db
                self.sensing_snr_sample_count += 1
            if detected:
                self.sensing_detection_count += 1
                return 1.0
            self.sensing_miss_count += 1
        if self.sensing_rng.random() < self.cfg.false_alarm_rate:
            self.sensing_false_alarm_count += 1
            return 1.0
        return 0.0

    def record_ideal_sensing_observation(
        self,
        node: int,
        sensed_beam: int,
        radius: float,
        *,
        piggyback: bool,
        observed_value: float,
    ) -> None:
        """Record sensing statistics for deterministic/ideal observations."""

        self.sensing_observations += 1
        has_target = self.beam_has_sensing_target(node, sensed_beam, radius)
        if has_target:
            self.sensing_target_observations += 1
            self.sensing_pd_sum += 1.0
            max_snr_db = self.max_sensing_snr_db(node, sensed_beam, radius, piggyback)
            if max_snr_db > -300.0:
                self.sensing_snr_db_sum += max_snr_db
                self.sensing_snr_sample_count += 1
            if observed_value > 0.0:
                self.sensing_detection_count += 1
            else:
                self.sensing_miss_count += 1
        elif observed_value > 0.0:
            self.sensing_false_alarm_count += 1

    def record_sensing_position_estimate(self, node: int, sensed_beam: int, radius: float, slot: int) -> None:
        """Store an anonymous noisy position estimate produced by the sensing front end."""

        beams = self.beam_matrix()
        dist2 = self.distance_matrix()[node]
        candidates = [
            target
            for target in range(self.cfg.n_nodes)
            if target != node
            and dist2[target] <= float(radius) ** 2
            and beam_matches(
                int(sensed_beam),
                int(beams[node, target]),
                self.cfg.azimuth_cells,
                self.cfg.alignment_tolerance_cells,
            )
        ]
        if not candidates:
            # A false alarm can affect the local occupancy belief, but it does
            # not create an identifiable position report for table exchange.
            return
        target = min(candidates, key=lambda item: float(dist2[item]))
        error_std = max(0.0, float(self.cfg.sensing_position_error_std_m))
        estimate = self.states[target].position.copy()
        if error_std > 0.0:
            estimate += self.sensing_rng.normal(0.0, error_std, size=3)
        estimate = np.clip(estimate, np.zeros(3), np.asarray(self.cfg.area_size_m, dtype=float))
        self.sensing_report_position[node, sensed_beam] = estimate
        self.sensing_report_confidence[node, sensed_beam] = float(self.belief[node, sensed_beam])
        self.sensing_report_slot[node, sensed_beam] = int(slot)

    def max_sensing_snr_db(self, node: int, sensed_beam: int, radius: float, piggyback: bool) -> float:
        profile = self.sensing_profile(radius, piggyback)
        values: list[float] = []
        for true_beam in self.matching_true_beams(int(sensed_beam)):
            values.extend(snr_db for _pd, snr_db in profile[node][int(true_beam)])
        return max(values, default=-300.0)

    def sensing_profile(self, radius: float, piggyback: bool) -> list[list[list[tuple[float, float]]]]:
        key = (round(float(radius), 6), bool(piggyback))
        cached = self._sensing_profile_cache.get(key)
        if cached is not None:
            return cached

        beams = self.beam_matrix()
        dist2 = self.distance_matrix()
        profiles: list[list[list[tuple[float, float]]]] = [
            [[] for _beam in range(self.cfg.n_beams)] for _node in range(self.cfg.n_nodes)
        ]
        radius2 = float(radius) ** 2
        for node in range(self.cfg.n_nodes):
            for target in range(self.cfg.n_nodes):
                if target == node or dist2[node, target] > radius2:
                    continue
                distance = float(np.sqrt(dist2[node, target]))
                pd = detection_probability(distance, self.cfg, piggyback=piggyback)
                snr_db = (
                    radar_snr_db(distance, self.cfg, piggyback=piggyback)
                    if self.cfg.isac_sensing_model == "radar_snr"
                    else 0.0
                )
                profiles[node][int(beams[node, target])].append((pd, snr_db))
        self._sensing_profile_cache[key] = profiles
        return profiles

    def matching_true_beams(self, sensed_beam: int) -> np.ndarray:
        cached = self._beam_match_cache.get(int(sensed_beam))
        if cached is not None:
            return cached
        matches = [
            beam
            for beam in range(self.cfg.n_beams)
            if beam_matches(int(sensed_beam), beam, self.cfg.azimuth_cells, self.cfg.alignment_tolerance_cells)
        ]
        arr = np.asarray(matches, dtype=int)
        self._beam_match_cache[int(sensed_beam)] = arr
        return arr

    def observable_true_beams(self, sensed_beam: int) -> tuple[tuple[int, float], ...]:
        cached = self._observable_beam_cache.get(int(sensed_beam))
        if cached is not None:
            return cached
        observable = tuple(
            (true_beam, probability)
            for true_beam in range(self.cfg.n_beams)
            if (probability := self.angular_observation_match_probability(true_beam, int(sensed_beam))) > 1e-6
        )
        self._observable_beam_cache[int(sensed_beam)] = observable
        return observable

    def angular_observation_match_probability(self, true_beam: int, sensed_beam: int) -> float:
        key = (int(true_beam), int(sensed_beam))
        cached = self._angular_match_probability_cache.get(key)
        if cached is not None:
            return cached

        offset_std = float(self.cfg.angular_cell_offset_std)
        if offset_std <= 0.0:
            probability = (
                1.0
                if beam_matches(sensed_beam, true_beam, self.cfg.azimuth_cells, self.cfg.alignment_tolerance_cells)
                else 0.0
            )
            self._angular_match_probability_cache[key] = probability
            return probability

        max_shift = int(max(2, np.ceil(4.0 * offset_std + 1.0)))
        shift_probabilities = [
            (shift, self.rounded_normal_shift_probability(shift, max_shift, offset_std))
            for shift in range(-max_shift, max_shift + 1)
        ]
        probability = 0.0
        for az_shift, p_az in shift_probabilities:
            if p_az <= 0.0:
                continue
            for el_shift, p_el in shift_probabilities:
                if p_el <= 0.0:
                    continue
                observed_beam = shift_beam(
                    true_beam,
                    self.cfg.azimuth_cells,
                    self.cfg.elevation_cells,
                    az_shift,
                    el_shift,
                )
                if beam_matches(sensed_beam, observed_beam, self.cfg.azimuth_cells, self.cfg.alignment_tolerance_cells):
                    probability += p_az * p_el
        self._angular_match_probability_cache[key] = probability
        return probability

    @staticmethod
    def rounded_normal_shift_probability(shift: int, max_shift: int, std: float) -> float:
        lower = -float("inf") if shift == -max_shift else shift - 0.5
        upper = float("inf") if shift == max_shift else shift + 0.5

        def cdf(x: float) -> float:
            if isinf(x):
                return 1.0 if x > 0 else 0.0
            return 0.5 * (1.0 + erf(x / (std * sqrt(2.0))))

        return max(0.0, cdf(upper) - cdf(lower))

    def update_action_counts(self, actions: list[Action], slot: int) -> None:
        for node, action in enumerate(actions):
            if action.mode == MODE_TX:
                self.tx_actions += 1
            elif action.mode == MODE_RX:
                self.rx_actions += 1
            elif action.mode == MODE_SENSE:
                self.sense_actions += 1
            elif action.mode == MODE_IDLE:
                self.idle_actions += 1

            if not self.action_has_piggyback_isac(action):
                continue
            period = max(1, self.cfg.sensing_period_slots)
            period = max(1, int(round(period * self.cfg.piggyback_sensing_period_multiplier)))
            if slot - int(self.last_sense_slot[node]) >= period:
                self.piggyback_sense_actions += 1

    def sensing_sector(self, center_beam: int) -> np.ndarray:
        radius = max(0, int(self.cfg.sensing_footprint_radius_cells))
        center_el, center_az = divmod(center_beam, self.cfg.azimuth_cells)
        beams: list[int] = []
        for el_delta in range(-radius, radius + 1):
            el = center_el + el_delta
            if el < 0 or el >= self.cfg.elevation_cells:
                continue
            for az_delta in range(-radius, radius + 1):
                az = (center_az + az_delta) % self.cfg.azimuth_cells
                beams.append(el * self.cfg.azimuth_cells + az)
        return np.asarray(sorted(set(beams)), dtype=int)

    def update_empty_scan_counts(self, actions: list[Action], true_comm_edges: set[tuple[int, int]], slot: int) -> None:
        occupied = self.occupied_beams(self.cfg.communication_range_m)
        for node, action in enumerate(actions):
            if action.mode not in (MODE_TX, MODE_RX):
                continue
            active_beams = self.active_beams_for_action(node, action, slot)
            self.scan_actions += len(active_beams)
            self.node_scan_actions[node] += len(active_beams)
            for active_beam in active_beams:
                if not any(
                    beam_matches(int(active_beam), beam_idx, self.cfg.azimuth_cells, self.cfg.alignment_tolerance_cells)
                    for beam_idx in occupied[node]
                ):
                    self.empty_scans += 1
                    self.node_empty_scans[node] += 1

    def beam_has_comm_neighbor(self, node: int, selected_beam: int, true_comm_edges: set[tuple[int, int]]) -> bool:
        for edge in true_comm_edges:
            if node not in edge:
                continue
            other = edge[1] if edge[0] == node else edge[0]
            expected = self.beam_from_to(node, other)
            if beam_matches(selected_beam, expected, self.cfg.azimuth_cells, self.cfg.alignment_tolerance_cells):
                return True
        return False

    def resolve_discoveries(self, slot: int, actions: list[Action], true_comm_edges: set[tuple[int, int]]) -> list[tuple[int, int]]:
        new_edges: list[tuple[int, int]] = []
        n_nodes = self.cfg.n_nodes
        beams = self.beam_matrix()
        true_mask = np.zeros((n_nodes, n_nodes), dtype=bool)
        if len(true_comm_edges) == n_nodes * (n_nodes - 1) // 2:
            true_mask[:, :] = True
            np.fill_diagonal(true_mask, False)
        else:
            for node_a, node_b in true_comm_edges:
                true_mask[node_a, node_b] = True
                true_mask[node_b, node_a] = True

        ready_mask = np.zeros((n_nodes, self.cfg.n_beams), dtype=bool)
        modes = np.asarray([action.mode for action in actions], dtype=object)
        active_mask = (modes == MODE_TX) | (modes == MODE_RX)
        for node, action in enumerate(actions):
            if not active_mask[node]:
                continue
            active_beams = self.active_beams_for_action(node, action, slot)
            ready_mask[node, active_beams] = True

        node_idx = np.arange(n_nodes)[:, None]
        ready_forward = ready_mask[node_idx, beams]
        tx_mask = modes == MODE_TX
        rx_mask = modes == MODE_RX
        candidate_matrix = (
            true_mask
            & tx_mask[:, None]
            & rx_mask[None, :]
            & ready_forward
            & ready_forward.T
        )
        for node_a, node_b in self.discovered_edges:
            candidate_matrix[node_a, node_b] = False
            candidate_matrix[node_b, node_a] = False
        self.handshake_attempts += int(candidate_matrix.sum())

        if self.cfg.communication_phy_model == "ideal":
            tx_degree = candidate_matrix.sum(axis=1)
            rx_degree = candidate_matrix.sum(axis=0)
            collision_matrix = candidate_matrix & ((tx_degree[:, None] > 1) | (rx_degree[None, :] > 1))
            self._record_failed_handshakes(collision_matrix, collision_matrix, actions)
            success_matrix = candidate_matrix & (tx_degree[:, None] == 1) & (rx_degree[None, :] == 1)
        else:
            phy_result = self.communication_phy.resolve_handshake(
                candidate_matrix=candidate_matrix,
                selected_beams=np.asarray([action.beam for action in actions], dtype=int),
                true_beams=beams,
                distance_m=np.sqrt(np.maximum(self.distance_matrix(), 0.0)),
                tx_mask=tx_mask,
                rx_mask=rx_mask,
                slot=slot,
            )
            success_matrix = phy_result.success_matrix
            forward_failed = candidate_matrix & ~phy_result.forward_decoded_matrix
            ack_failed = phy_result.forward_decoded_matrix & ~success_matrix
            interference_failed = (
                phy_result.forward_interference_failures | phy_result.ack_interference_failures
            )
            outage_failed = phy_result.forward_outage_failures | phy_result.ack_outage_failures
            self.forward_decode_failures += int(forward_failed.sum())
            self.ack_decode_failures += int(ack_failed.sum())
            self.interference_limited_failures += int(interference_failed.sum())
            self.phy_outage_failures += int(outage_failed.sum())
            self.handshake_sinr_samples_db.extend(phy_result.sinr_samples_db.astype(float).tolist())
            self._record_failed_handshakes(
                candidate_matrix & ~success_matrix,
                interference_failed,
                actions,
            )

        self.handshake_successes += int(success_matrix.sum())
        for tx_node, rx_node in np.argwhere(success_matrix).astype(int).tolist():
            edge = canonical_edge(tx_node, rx_node)
            is_new = edge not in self.discovered_edges
            if not is_new:
                # Duplicate-response suppression is part of the common MAC
                # contract, so learned and rule baselines face the same reply rule.
                continue
            self.discovered_edges.add(edge)
            self.discovery_slot.setdefault(edge, slot)
            self.neighbor_records[tx_node][rx_node] = (self.states[rx_node].position.copy(), slot)
            self.neighbor_records[rx_node][tx_node] = (self.states[tx_node].position.copy(), slot)
            if is_new:
                first_slot = self.first_true_slot.get(edge, slot)
                self.edge_rows.append(
                    {
                        "protocol": self.protocol,
                        "edge_i": edge[0],
                        "edge_j": edge[1],
                        "first_true_slot": first_slot,
                        "discovery_slot": slot,
                        "delay_slots": slot - first_slot + 1,
                    }
                )
                new_edges.append(edge)
            self.success_count[tx_node, int(beams[tx_node, rx_node])] += 1.0
            self.success_count[rx_node, int(beams[rx_node, tx_node])] += 1.0
            self.mark_wang2025_interaction(tx_node, int(beams[tx_node, rx_node]), slot)
            self.mark_wang2025_interaction(rx_node, int(beams[rx_node, tx_node]), slot)
            if self.protocol in COMM_TABLE_PROTOCOLS:
                self.exchange_neighbor_and_sensing_tables(tx_node, rx_node, slot)
        return new_edges

    def _record_failed_handshakes(
        self,
        failed_matrix: np.ndarray,
        interference_matrix: np.ndarray,
        actions: list[Action],
    ) -> None:
        interference_count = int(interference_matrix.sum())
        self.collision_count += interference_count
        for tx_node, rx_node in np.argwhere(failed_matrix).astype(int).tolist():
            is_interference = bool(interference_matrix[tx_node, rx_node])
            for node in (tx_node, rx_node):
                beam = int(actions[node].beam)
                self.fail_count[node, beam] += 1.0
                if is_interference:
                    self.collision_fail_count[node, beam] += 1.0
                    self.node_collision_count[node] += 1

    def exchange_neighbor_and_sensing_tables(self, node_a: int, node_b: int, slot: int) -> None:
        include_sensing = self.protocol in SENSING_TABLE_PROTOCOLS
        self.merge_peer_tables(dst=node_a, src=node_b, slot=slot, include_sensing=include_sensing)
        self.merge_peer_tables(dst=node_b, src=node_a, slot=slot, include_sensing=include_sensing)

    def merge_peer_tables(self, dst: int, src: int, slot: int, *, include_sensing: bool) -> None:
        report_ttl = max(1, int(self.cfg.sensing_report_ttl_slots))
        shared_reports: list[tuple[np.ndarray, str]] = []
        for target, (position, report_slot) in self.neighbor_records[src].items():
            if target != dst and slot - int(report_slot) <= report_ttl:
                shared_reports.append((position.copy(), "comm"))

        if include_sensing:
            recent = (slot - self.sensing_report_slot[src]) <= report_ttl
            valid = recent & (self.sensing_report_confidence[src] > 0.0)
            for source_beam in np.flatnonzero(valid)[: max(1, 2 * self.cfg.target_degree)]:
                position = self.sensing_report_position[src, int(source_beam)]
                if np.isfinite(position).all():
                    shared_reports.append((position.copy(), "sense"))

        seen_beams: set[tuple[int, str]] = set()
        for position, support in shared_reports:
            beam = beam_for_target(
                self.states[dst], position, self.cfg.azimuth_cells, self.cfg.elevation_cells
            )
            report_key = (int(beam), support)
            if report_key in seen_beams:
                continue
            seen_beams.add(report_key)
            trust_weight = 1.0
            if self.protocol in TRUST_GATED_TABLE_PROTOCOLS:
                trust_weight = self.shared_table_trust_weight(dst, beam, support, slot)
                if trust_weight <= 0.0:
                    continue
            self.boost_shared_beam(dst, beam, slot, strength=trust_weight)

    def shared_table_trust_weight(self, dst: int, beam: int, support: str, slot: int) -> float:
        """Local confidence gate for peer-provided neighbor/sensing table hints."""

        success = float(self.success_count[dst, beam])
        fail = float(self.fail_count[dst, beam])
        collision = float(self.collision_fail_count[dst, beam])
        local_support = float(np.clip(self.belief[dst, beam] + 0.10 * np.log1p(success), 0.0, 1.0))
        failure_pressure = fail / max(1.0, success + fail)
        collision_pressure = collision / max(1.0, success + collision)
        positive_ttl = max(20, min(200, int(round(0.50 / max(self.cfg.slot_duration_s, 1e-6)))))
        age = slot - int(self.last_positive_slot[dst, beam])
        recency = 0.0 if age > positive_ttl else 1.0 - max(0.0, float(age)) / max(1.0, float(positive_ttl))

        base = 0.65 if support == "comm" else 0.35
        trust = base + 0.25 * local_support + 0.15 * recency - 0.45 * collision_pressure - 0.25 * failure_pressure
        if support == "comm" and trust >= 0.35:
            return float(np.clip(trust, 0.25, 0.90))
        if support == "sense" and trust >= 0.45:
            return float(np.clip(trust, 0.20, 0.75))
        return 0.0

    def boost_shared_beam(self, node: int, beam: int, slot: int, *, strength: float = 1.0) -> None:
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0.0:
            return
        rho = min(0.85, max(0.0, self.cfg.belief_update_rho)) * strength
        self.belief[node, beam] = max(self.belief[node, beam], (1.0 - rho) * self.belief[node, beam] + rho)
        self.success_count[node, beam] += 0.15 * strength
        self.fail_count[node, beam] *= max(0.0, 1.0 - 0.25 * strength)
        self.empty_beam_count[node, beam] = 0.0
        self.age[node, beam] = 0.0
        self.last_positive_slot[node, beam] = slot
        self.boost_wang2025_candidate(node, beam, slot)

    def summarize(self, episode: int) -> EpisodeResult:
        delays = []
        censored = float(self.cfg.slots_per_episode)
        for edge, first_slot in self.first_true_slot.items():
            if edge in self.discovery_slot:
                delays.append(float(self.discovery_slot[edge] - first_slot + 1))
            else:
                delays.append(float(max(1, self.cfg.slots_per_episode - first_slot)))
        if not delays:
            delays = [censored]
        current_true_edges = self.true_edges(self.cfg.communication_range_m)
        active_edges = self.discovered_edges.intersection(current_true_edges)
        components = connected_components(self.cfg.n_nodes, active_edges)
        largest = max((len(c) for c in components), default=0)
        isolated = sum(1 for c in components if len(c) == 1)
        discovered_count = len(self.discovered_edges)
        discovery_rate = discovered_count / max(1, len(self.first_true_slot))
        discovery_per_scan_action = discovered_count / max(1, self.scan_actions)
        discoveries_per_1000_scan_actions = 1000.0 * discovery_per_scan_action
        scan_actions_per_discovery = self.scan_actions / max(1, discovered_count)
        collisions_per_discovery = self.collision_count / max(1, discovered_count)
        collision_normalized_efficiency = discovered_count / max(1, discovered_count + self.collision_count)
        collision_penalized_discovery_rate = discovered_count / max(1, len(self.first_true_slot) + self.collision_count)
        energy_j = self.radio_energy_j()
        sensing_detection_rate = self.sensing_detection_count / max(1, self.sensing_target_observations)
        sensing_false_alarm_ratio = self.sensing_false_alarm_count / max(1, self.sensing_observations)
        sensing_miss_ratio = self.sensing_miss_count / max(1, self.sensing_target_observations)
        mean_sensing_pd = self.sensing_pd_sum / max(1, self.sensing_target_observations)
        mean_sensing_snr_db = self.sensing_snr_db_sum / max(1, self.sensing_snr_sample_count)
        mean_handshake_sinr_db = (
            float(np.mean(self.handshake_sinr_samples_db)) if self.handshake_sinr_samples_db else 0.0
        )
        p10_handshake_sinr_db = (
            float(np.percentile(self.handshake_sinr_samples_db, 10)) if self.handshake_sinr_samples_db else 0.0
        )
        discoveries_per_joule = discovered_count / max(energy_j, 1e-12)
        energy_per_discovery = energy_j / max(1, discovered_count)
        moved = 0.0
        if self.initial_positions is not None:
            moved = float(np.linalg.norm(self.positions() - self.initial_positions, axis=1).mean())
        return EpisodeResult(
            protocol=self.protocol,
            episode=episode,
            seed=self.seed,
            scenario_seed=self.scenario_seed,
            slots=self.cfg.slots_per_episode,
            mobility_model=str(self.cfg.mobility.get("model", "gauss_markov")),
            true_edges_seen=len(self.first_true_slot),
            discovered_edges=discovered_count,
            discovery_rate=discovery_rate,
            mean_delay_censored=float(np.mean(delays)),
            p90_delay_censored=float(np.percentile(delays, 90)),
            p95_delay_censored=float(np.percentile(delays, 95)),
            p99_delay_censored=float(np.percentile(delays, 99)),
            empty_scan_ratio=self.empty_scans / max(1, self.scan_actions),
            handshake_attempts=self.handshake_attempts,
            handshake_successes=self.handshake_successes,
            forward_decode_failures=self.forward_decode_failures,
            ack_decode_failures=self.ack_decode_failures,
            interference_limited_failures=self.interference_limited_failures,
            phy_outage_failures=self.phy_outage_failures,
            mean_handshake_sinr_db=mean_handshake_sinr_db,
            p10_handshake_sinr_db=p10_handshake_sinr_db,
            collision_count=self.collision_count,
            empty_scan_count=self.empty_scans,
            scan_actions=self.scan_actions,
            tx_actions=self.tx_actions,
            rx_actions=self.rx_actions,
            sense_actions=self.sense_actions,
            idle_actions=self.idle_actions,
            piggyback_sense_actions=self.piggyback_sense_actions,
            sensing_observations=self.sensing_observations,
            sensing_target_observations=self.sensing_target_observations,
            sensing_detection_rate=sensing_detection_rate,
            sensing_false_alarm_ratio=sensing_false_alarm_ratio,
            sensing_miss_ratio=sensing_miss_ratio,
            mean_sensing_pd=mean_sensing_pd,
            mean_sensing_snr_db=mean_sensing_snr_db,
            discovery_per_scan_action=discovery_per_scan_action,
            discoveries_per_1000_scan_actions=discoveries_per_1000_scan_actions,
            scan_actions_per_discovery_censored=scan_actions_per_discovery,
            collisions_per_discovery_censored=collisions_per_discovery,
            collision_normalized_efficiency=collision_normalized_efficiency,
            collision_penalized_discovery_rate=collision_penalized_discovery_rate,
            energy_j=energy_j,
            discoveries_per_joule=discoveries_per_joule,
            energy_per_discovery_censored_j=energy_per_discovery,
            moved_distance_mean_m=moved,
            active_discovered_edges=len(active_edges),
            largest_component_size=largest,
            connected_components=len(components),
            lcc_ratio=largest / max(1, self.cfg.n_nodes),
            isolated_node_ratio=isolated / max(1, self.cfg.n_nodes),
            lambda2=algebraic_connectivity(self.cfg.n_nodes, active_edges),
            knowledge_lambda2=algebraic_connectivity(self.cfg.n_nodes, self.discovered_edges),
        )

    def radio_energy_j(self) -> float:
        slot_s = float(self.cfg.slot_duration_s)
        return float(
            slot_s
            * (
                self.tx_actions * self.cfg.tx_power_w
                + self.rx_actions * self.cfg.rx_power_w
                + self.sense_actions * self.cfg.sense_power_w
                + self.idle_actions * self.cfg.idle_power_w
                + self.piggyback_sense_actions * self.cfg.piggyback_sense_power_w
            )
        )


def canonical_edge(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def connected_components(n_nodes: int, edges: Iterable[tuple[int, int]]) -> list[set[int]]:
    adjacency = [set() for _ in range(n_nodes)]
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    seen: set[int] = set()
    components: list[set[int]] = []
    for node in range(n_nodes):
        if node in seen:
            continue
        stack = [node]
        comp: set[int] = set()
        seen.add(node)
        while stack:
            cur = stack.pop()
            comp.add(cur)
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(comp)
    return components


def algebraic_connectivity(n_nodes: int, edges: Iterable[tuple[int, int]]) -> float:
    if n_nodes < 2:
        return 0.0
    laplacian = np.zeros((n_nodes, n_nodes), dtype=float)
    for i, j in edges:
        laplacian[i, i] += 1.0
        laplacian[j, j] += 1.0
        laplacian[i, j] -= 1.0
        laplacian[j, i] -= 1.0
    eigenvalues = np.linalg.eigvalsh(laplacian)
    return float(np.sort(eigenvalues)[1])
