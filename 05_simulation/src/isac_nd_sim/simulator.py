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
    slots: int
    mobility_model: str
    true_edges_seen: int
    discovered_edges: int
    discovery_rate: float
    mean_delay_censored: float
    p95_delay_censored: float
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
    def __init__(self, config: SimulationConfig, protocol: str, seed: int):
        self.cfg = config
        self.protocol = protocol
        self.rng = np.random.default_rng(seed)
        self.seed = seed
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
        self.per_slot_rows: list[dict] = []
        self.edge_rows: list[dict] = []

    def reset(self) -> None:
        self.states = initialize_states(self.cfg.n_nodes, self.cfg.area_size_m, self.cfg.mobility, self.rng)
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
            self.update_sensing(actions)
            new_edges = self.resolve_discoveries(slot, actions, true_comm_edges)
            self.per_slot_rows.append(self.slot_metrics(episode, slot, true_comm_edges, new_edges))
            step_states(
                self.states,
                self.cfg.area_size_m,
                self.cfg.mobility,
                self.cfg.slot_duration_s,
                slot,
                self.rng,
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
        oracle_occupied = self.occupied_beams(self.cfg.communication_range_m)
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
        probs = np.asarray([self.cfg.p_sense, self.cfg.p_tx, self.cfg.p_rx, self.cfg.p_idle], dtype=float)
        probs = probs / probs.sum()
        return str(self.rng.choice([MODE_SENSE, MODE_TX, MODE_RX, MODE_IDLE], p=probs))

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
            step = max(1, int(np.sqrt(self.cfg.n_beams)))
            return int((node * step + slot * step) % self.cfg.n_beams)
        if self.protocol == "oracle" and oracle_occupied:
            return int(self.rng.choice(tuple(oracle_occupied)))

        weights = np.ones(self.cfg.n_beams, dtype=float) * self.cfg.exploration_floor
        if self.protocol in ("isac_only", "itap_nd", "oracle"):
            weights += np.maximum(self.belief[node], 0.0)
        if self.protocol in ("topology_only", "itap_nd"):
            degree = sum(1 for edge in self.discovered_edges if node in edge)
            degree_need = max(0.0, self.cfg.target_degree - degree) / max(1, self.cfg.target_degree)
            diversity = 1.0 / (1.0 + self.success_count[node])
            staleness = self.age[node] / max(1.0, self.age[node].max())
            weights += self.cfg.beta_diversity * diversity
            weights += self.cfg.gamma_degree_need * degree_need
            weights -= self.cfg.eta_staleness * staleness
        weights = np.maximum(weights, self.cfg.exploration_floor / self.cfg.n_beams)
        if self.cfg.softmax_beta > 0:
            logits = self.cfg.softmax_beta * (weights - weights.max())
            probs = np.exp(logits)
        else:
            probs = weights
        probs = probs / probs.sum()
        return int(self.rng.choice(self.cfg.n_beams, p=probs))

    def update_sensing(self, actions: list[Action]) -> None:
        true_occupied = self.occupied_beams(self.cfg.sensing_range_m)
        for node, action in enumerate(actions):
            if action.mode != MODE_SENSE:
                continue
            observation = np.zeros(self.cfg.n_beams, dtype=float)
            for beam_idx in true_occupied[node]:
                if self.rng.random() < self.cfg.miss_detection_rate:
                    continue
                offset_std = self.cfg.angular_cell_offset_std
                az_shift = int(np.rint(self.rng.normal(0.0, offset_std)))
                el_shift = int(np.rint(self.rng.normal(0.0, offset_std)))
                observed = shift_beam(beam_idx, self.cfg.azimuth_cells, self.cfg.elevation_cells, az_shift, el_shift)
                observation[observed] = 1.0
            false_alarms = self.rng.random(self.cfg.n_beams) < self.cfg.false_alarm_rate
            observation[false_alarms] = 1.0
            rho = self.cfg.belief_update_rho
            self.belief[node] = (1.0 - rho) * self.belief[node] + rho * observation
            self.age[node] = 0.0

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
            slots=self.cfg.slots_per_episode,
            mobility_model=str(self.cfg.mobility.get("model", "gauss_markov")),
            true_edges_seen=len(self.first_true_slot),
            discovered_edges=len(self.discovered_edges),
            discovery_rate=len(self.discovered_edges) / max(1, len(self.first_true_slot)),
            mean_delay_censored=float(np.mean(delays)),
            p95_delay_censored=float(np.percentile(delays, 95)),
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
