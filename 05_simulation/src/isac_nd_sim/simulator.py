from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .beam import beam_for_target, beam_matches, shift_beam
from .config import SimulationConfig
from .mobility import NodeState, initialize_states, step_states

MODE_SENSE = "sense"
MODE_TX = "tx"
MODE_RX = "rx"
MODE_IDLE = "idle"


@dataclass(frozen=True)
class Action:
    mode: str
    beam: int


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
    collision_count: int
    moved_distance_mean_m: float
    largest_component_size: int
    connected_components: int
    lcc_ratio: float
    isolated_node_ratio: float
    lambda2: float

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
        self.states: list[NodeState] = []
        self.initial_positions: np.ndarray | None = None
        self.belief = np.zeros((self.cfg.n_nodes, self.cfg.n_beams), dtype=float)
        self.age = np.zeros_like(self.belief)
        self.success_count = np.zeros_like(self.belief)
        self.fail_count = np.zeros_like(self.belief)
        self.discovered_edges: set[tuple[int, int]] = set()
        self.first_true_slot: dict[tuple[int, int], int] = {}
        self.discovery_slot: dict[tuple[int, int], int] = {}
        self.empty_scans = 0
        self.scan_actions = 0
        self.collision_count = 0
        self.last_sense_slot = np.full(self.cfg.n_nodes, -10**9, dtype=int)
        self.per_slot_rows: list[dict] = []
        self.edge_rows: list[dict] = []

    def reset(self) -> None:
        self.states = initialize_states(self.cfg.n_nodes, self.cfg.area_size_m, self.cfg.mobility, self.mobility_rng)
        self.initial_positions = np.asarray([s.position.copy() for s in self.states])
        self.belief.fill(0.5)
        self.age.fill(0.0)
        self.success_count.fill(0.0)
        self.fail_count.fill(0.0)
        self.discovered_edges.clear()
        self.first_true_slot.clear()
        self.discovery_slot.clear()
        self.empty_scans = 0
        self.scan_actions = 0
        self.collision_count = 0
        self.last_sense_slot.fill(-10**9)
        self.per_slot_rows = []
        self.edge_rows = []

    def run_episode(self, episode: int) -> EpisodeResult:
        self.reset()
        for slot in range(self.cfg.slots_per_episode):
            true_comm_edges = self.true_edges(self.cfg.communication_range_m)
            for edge in true_comm_edges:
                self.first_true_slot.setdefault(edge, slot)
            self.age += 1.0
            self.belief *= self.cfg.confidence_decay
            actions = self.select_actions(slot, true_comm_edges)
            self.update_empty_scan_counts(actions, true_comm_edges)
            self.update_sensing(actions, slot)
            new_edges = self.resolve_discoveries(slot, actions, true_comm_edges)
            self.per_slot_rows.append(self.slot_metrics(episode, slot, true_comm_edges, new_edges))
            step_states(
                self.states,
                self.cfg.area_size_m,
                self.cfg.mobility,
                self.cfg.slot_duration_s,
                slot,
                self.mobility_rng,
            )
        return self.summarize(episode)

    def slot_metrics(
        self,
        episode: int,
        slot: int,
        true_comm_edges: set[tuple[int, int]],
        new_edges: list[tuple[int, int]],
    ) -> dict:
        components = connected_components(self.cfg.n_nodes, self.discovered_edges)
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
            "new_edges": len(new_edges),
            "empty_scan_ratio": self.empty_scans / max(1, self.scan_actions),
            "collision_count": self.collision_count,
            "largest_component_size": largest,
            "connected_components": len(components),
            "lcc_ratio": largest / max(1, self.cfg.n_nodes),
            "isolated_node_ratio": isolated / max(1, self.cfg.n_nodes),
            "lambda2": algebraic_connectivity(self.cfg.n_nodes, self.discovered_edges),
        }

    def positions(self) -> np.ndarray:
        return np.asarray([state.position for state in self.states])

    def true_edges(self, radius: float) -> set[tuple[int, int]]:
        pos = self.positions()
        edges: set[tuple[int, int]] = set()
        for i in range(self.cfg.n_nodes):
            for j in range(i + 1, self.cfg.n_nodes):
                if float(np.linalg.norm(pos[i] - pos[j])) <= radius:
                    edges.add((i, j))
        return edges

    def occupied_beams(self, radius: float) -> list[set[int]]:
        occupied = [set() for _ in range(self.cfg.n_nodes)]
        pos = self.positions()
        for i in range(self.cfg.n_nodes):
            for j in range(self.cfg.n_nodes):
                if i == j:
                    continue
                if float(np.linalg.norm(pos[i] - pos[j])) <= radius:
                    occupied[i].add(self.beam_from_to(i, j))
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
        if self.protocol in ("improved_rl_isac", "isac_structured_marl"):
            degree = sum(1 for edge in self.discovered_edges if node in edge)
            degree_need = max(0.0, self.cfg.target_degree - degree) / max(1, self.cfg.target_degree)
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
        if self.protocol in ("improved_rl_isac", "isac_structured_marl"):
            period = max(1, 2 * period)
        staggered_periodic_sense = slot % period == node % period
        if self.protocol in ("improved_rl_isac", "isac_structured_marl"):
            staggered_periodic_sense = slot % period == (2 * node) % period
        stale_belief = float(np.mean(self.age[node])) >= 2.0 * period
        weak_belief = float(np.max(self.belief[node])) < 0.2
        if self.protocol in ("improved_rl_isac", "isac_structured_marl"):
            return bool(staggered_periodic_sense)
        return bool(staggered_periodic_sense or stale_belief or weak_belief)

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
        if self.protocol == "oracle" and oracle_occupied:
            return int(self.rng.choice(tuple(oracle_occupied)))
        if self.protocol in ("improved_rl_isac", "isac_structured_marl") and mode == MODE_SENSE:
            return self.isac_sensing_beam(node, slot)
        if self.protocol in ("rl_no_isac", "basic_marl_no_isac"):
            return self.memory_guided_beam(node, use_isac=False, topology=False)
        if self.protocol in ("improved_rl_no_isac", "structured_marl_no_isac"):
            return self.memory_guided_beam(node, use_isac=False, topology=True)
        if self.protocol in ("improved_rl_isac", "isac_structured_marl"):
            return self.memory_guided_beam(node, use_isac=True, topology=True)

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

    def isac_sensing_beam(self, node: int, slot: int) -> int:
        period = max(1, self.cfg.sensing_period_slots)
        if self.protocol in ("improved_rl_isac", "isac_structured_marl"):
            period = max(1, 2 * period)
        sense_round = slot // period
        prime_stride = self.near_coprime_stride(self.cfg.n_beams)
        return int((node * prime_stride + sense_round * prime_stride) % self.cfg.n_beams)

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

    def update_sensing(self, actions: list[Action], slot: int | None = None) -> None:
        true_occupied = self.occupied_beams(self.cfg.sensing_range_m)
        for node, action in enumerate(actions):
            piggyback_isac = self.protocol in ("improved_rl_isac", "isac_structured_marl") and action.mode in (
                MODE_TX,
                MODE_RX,
            )
            if action.mode != MODE_SENSE and not piggyback_isac:
                continue
            if slot is not None:
                period = max(1, self.cfg.sensing_period_slots)
                if piggyback_isac:
                    period = max(1, int(round(period * self.cfg.piggyback_sensing_period_multiplier)))
                if slot - int(self.last_sense_slot[node]) < period:
                    continue
                self.last_sense_slot[node] = slot
            observation = self.belief[node].copy()
            sensed_beams = self.sensing_sector(action.beam)
            for sensed_beam in sensed_beams:
                observed_value = 0.0
                for beam_idx in true_occupied[node]:
                    if self.rng.random() < self.cfg.miss_detection_rate:
                        continue
                    offset_std = self.cfg.angular_cell_offset_std
                    az_shift = int(np.rint(self.rng.normal(0.0, offset_std)))
                    el_shift = int(np.rint(self.rng.normal(0.0, offset_std)))
                    observed = shift_beam(beam_idx, self.cfg.azimuth_cells, self.cfg.elevation_cells, az_shift, el_shift)
                    if beam_matches(sensed_beam, observed, self.cfg.azimuth_cells, self.cfg.alignment_tolerance_cells):
                        observed_value = 1.0
                        break
                if self.rng.random() < self.cfg.false_alarm_rate:
                    observed_value = 1.0
                observation[sensed_beam] = observed_value
            rho = self.cfg.belief_update_rho
            self.belief[node] = (1.0 - rho) * self.belief[node] + rho * observation
            self.age[node, sensed_beams] = 0.0

    def sensing_sector(self, center_beam: int) -> np.ndarray:
        radius = max(1, self.cfg.alignment_tolerance_cells + int(np.ceil(self.cfg.angular_cell_offset_std)))
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

    def update_empty_scan_counts(self, actions: list[Action], true_comm_edges: set[tuple[int, int]]) -> None:
        for node, action in enumerate(actions):
            if action.mode not in (MODE_TX, MODE_RX):
                continue
            self.scan_actions += 1
            if not self.beam_has_comm_neighbor(node, action.beam, true_comm_edges):
                self.empty_scans += 1

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
        for rx_node, rx_action in enumerate(actions):
            if rx_action.mode != MODE_RX:
                continue
            candidates: list[int] = []
            for tx_node, tx_action in enumerate(actions):
                if tx_node == rx_node or tx_action.mode != MODE_TX:
                    continue
                edge = canonical_edge(tx_node, rx_node)
                if edge not in true_comm_edges:
                    continue
                tx_expected = self.beam_from_to(tx_node, rx_node)
                rx_expected = self.beam_from_to(rx_node, tx_node)
                tx_ok = beam_matches(tx_action.beam, tx_expected, self.cfg.azimuth_cells, self.cfg.alignment_tolerance_cells)
                rx_ok = beam_matches(rx_action.beam, rx_expected, self.cfg.azimuth_cells, self.cfg.alignment_tolerance_cells)
                if tx_ok and rx_ok:
                    candidates.append(tx_node)
            if len(candidates) > 1:
                self.collision_count += len(candidates)
                for tx_node in candidates:
                    self.fail_count[tx_node, actions[tx_node].beam] += 1.0
                self.fail_count[rx_node, rx_action.beam] += 1.0
                continue
            if len(candidates) == 1:
                tx_node = candidates[0]
                edge = canonical_edge(tx_node, rx_node)
                is_new = edge not in self.discovered_edges
                self.discovered_edges.add(edge)
                self.discovery_slot.setdefault(edge, slot)
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
                self.success_count[tx_node, actions[tx_node].beam] += 1.0
                self.success_count[rx_node, rx_action.beam] += 1.0
        return new_edges

    def summarize(self, episode: int) -> EpisodeResult:
        delays = []
        censored = float(self.cfg.slots_per_episode)
        for edge, first_slot in self.first_true_slot.items():
            if edge in self.discovery_slot:
                delays.append(float(self.discovery_slot[edge] - first_slot + 1))
            else:
                delays.append(censored)
        if not delays:
            delays = [censored]
        components = connected_components(self.cfg.n_nodes, self.discovered_edges)
        largest = max((len(c) for c in components), default=0)
        isolated = sum(1 for c in components if len(c) == 1)
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
            discovered_edges=len(self.discovered_edges),
            discovery_rate=len(self.discovered_edges) / max(1, len(self.first_true_slot)),
            mean_delay_censored=float(np.mean(delays)),
            p90_delay_censored=float(np.percentile(delays, 90)),
            p95_delay_censored=float(np.percentile(delays, 95)),
            p99_delay_censored=float(np.percentile(delays, 99)),
            empty_scan_ratio=self.empty_scans / max(1, self.scan_actions),
            collision_count=self.collision_count,
            moved_distance_mean_m=moved,
            largest_component_size=largest,
            connected_components=len(components),
            lcc_ratio=largest / max(1, self.cfg.n_nodes),
            isolated_node_ratio=isolated / max(1, self.cfg.n_nodes),
            lambda2=algebraic_connectivity(self.cfg.n_nodes, self.discovered_edges),
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
