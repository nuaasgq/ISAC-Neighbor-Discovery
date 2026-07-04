from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import SimulationConfig
from .mobility import step_states
from .simulator import (
    Action,
    MODE_IDLE,
    MODE_RX,
    MODE_SENSE,
    MODE_TX,
    NeighborDiscoverySimulator,
    algebraic_connectivity,
    connected_components,
)

MODE_NAMES = (MODE_SENSE, MODE_TX, MODE_RX, MODE_IDLE)
MODE_TO_INDEX = {name: idx for idx, name in enumerate(MODE_NAMES)}


@dataclass(frozen=True)
class EnvStepResult:
    observations: list[dict[str, np.ndarray | int]]
    rewards: np.ndarray
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class MarlNeighborDiscoveryEnv:
    """Lightweight multi-agent RL facade over the slot-level simulator.

    The public observation and info contracts intentionally expose only local
    execution-time information. Global truth used by a future centralized critic
    is available through training_state() only.
    """

    def __init__(self, config: SimulationConfig, seed: int | None = None, protocol: str = "isac_structured_marl"):
        self.cfg = config
        self.seed = int(config.seed if seed is None else seed)
        self.protocol = str(protocol)
        self._sim = NeighborDiscoverySimulator(config, protocol=self.protocol, seed=self.seed)
        self._slot = 0
        self._last_actions: list[Action] = [Action(MODE_IDLE, 0) for _ in range(config.n_nodes)]

    @property
    def n_agents(self) -> int:
        return self.cfg.n_nodes

    @property
    def n_beams(self) -> int:
        return self.cfg.n_beams

    @property
    def modes(self) -> tuple[str, ...]:
        return MODE_NAMES

    def reset(self, seed: int | None = None) -> tuple[list[dict[str, np.ndarray | int]], dict[str, Any]]:
        if seed is not None:
            self.seed = int(seed)
        self._sim = NeighborDiscoverySimulator(self.cfg, protocol=self.protocol, seed=self.seed)
        self._sim.reset()
        self._slot = 0
        self._last_actions = [Action(MODE_IDLE, 0) for _ in range(self.n_agents)]
        return self._observations(), self._safe_info(new_edges_count=0)

    def step(
        self,
        actions: Sequence[Action | Mapping[str, Any] | Sequence[Any]] | Mapping[int | str, Any],
    ) -> tuple[list[dict[str, np.ndarray | int]], np.ndarray, bool, bool, dict[str, Any]]:
        if not self._sim.states:
            raise RuntimeError("Environment must be reset before step().")
        if self._slot >= self.cfg.slots_per_episode:
            raise RuntimeError("Episode already truncated. Call reset() before step().")

        parsed_actions = self._normalize_actions(actions)
        true_comm_edges = self._sim.true_edges(self.cfg.communication_range_m)
        for edge in true_comm_edges:
            self._sim.first_true_slot.setdefault(edge, self._slot)

        self._sim.age += 1.0
        self._sim.belief *= self.cfg.confidence_decay

        empty_by_node = np.zeros(self.n_agents, dtype=bool)
        for node, action in enumerate(parsed_actions):
            if action.mode in (MODE_TX, MODE_RX):
                empty_by_node[node] = not self._sim.beam_has_comm_neighbor(node, action.beam, true_comm_edges)

        fail_before = self._sim.fail_count.sum(axis=1).copy()
        success_before = self._sim.success_count.sum(axis=1).copy()

        self._sim._candidate_pool_cache.clear()
        self._sim.snapshot_pre_sensing_candidates(self._slot)
        self._sim.update_action_counts(parsed_actions, self._slot)
        self._sim.update_empty_scan_counts(parsed_actions, true_comm_edges)
        self._sim.update_sensing(parsed_actions, self._slot)
        self._sim._candidate_pool_cache.clear()
        new_edges = self._sim.resolve_discoveries(self._slot, parsed_actions, true_comm_edges)
        metric_row = self._sim.slot_metrics(episode=0, slot=self._slot, true_comm_edges=true_comm_edges, new_edges=new_edges)
        self._sim.per_slot_rows.append(metric_row)

        rewards = self._rewards(parsed_actions, new_edges, empty_by_node, fail_before, success_before)
        step_states(
            self._sim.states,
            self.cfg.area_size_m,
            self.cfg.mobility,
            self.cfg.slot_duration_s,
            self._slot,
            self._sim.mobility_rng,
        )
        self._last_actions = parsed_actions
        self._slot += 1

        terminated = False
        truncated = self._slot >= self.cfg.slots_per_episode
        return self._observations(), rewards, terminated, truncated, self._safe_info(new_edges_count=len(new_edges))

    def training_state(self) -> dict[str, Any]:
        """Return centralized-training state that may contain simulator truth.

        This method is deliberately separate from reset()/step() info to avoid
        accidental execution-time leakage into decentralized actors.
        """

        true_edges = self._sim.true_edges(self.cfg.communication_range_m)
        return {
            "slot": self._slot,
            "seed": self.seed,
            "positions": self._positions().astype(np.float32),
            "velocities": self._velocities().astype(np.float32),
            "attitudes": self._attitudes().astype(np.float32),
            "true_edges": _edge_array(true_edges),
            "discovered_edges": _edge_array(self._sim.discovered_edges),
            "true_adjacency": _adjacency(self.n_agents, true_edges),
            "discovered_adjacency": _adjacency(self.n_agents, self._sim.discovered_edges),
            "belief": self._sim.belief.astype(np.float32).copy(),
        }

    def _observations(self) -> list[dict[str, np.ndarray | int]]:
        return [self._observation_for(node) for node in range(self.n_agents)]

    def _observation_for(self, node: int) -> dict[str, np.ndarray | int]:
        state = self._sim.states[node]
        area = np.asarray(self.cfg.area_size_m, dtype=float)
        speed_scale = _speed_scale(self.cfg.mobility)
        degree = sum(1 for edge in self._sim.discovered_edges if node in edge)
        last_action = self._last_actions[node]
        mode_one_hot = np.zeros(len(MODE_NAMES), dtype=np.float32)
        mode_one_hot[MODE_TO_INDEX[last_action.mode]] = 1.0

        self_state = np.asarray(
            [
                *(state.position / np.maximum(area, 1e-9)),
                *(state.velocity / speed_scale),
                state.yaw / np.pi,
                state.pitch / (np.pi / 2.0),
                state.roll / np.pi,
            ],
            dtype=np.float32,
        )
        local_summary = np.asarray(
            [
                degree / max(1, self.n_agents - 1),
                self._slot / max(1, self.cfg.slots_per_episode),
                self._sim.empty_scans / max(1, self._sim.scan_actions),
                self._sim.collision_count / max(1, self._slot + 1),
            ],
            dtype=np.float32,
        )
        return {
            "agent_id": node,
            "self_state": self_state,
            "beam_belief": self._sim.belief[node].astype(np.float32).copy(),
            "beam_age": (self._sim.age[node] / max(1, self.cfg.slots_per_episode)).astype(np.float32).copy(),
            "beam_success": self._sim.success_count[node].astype(np.float32).copy(),
            "beam_fail": self._sim.fail_count[node].astype(np.float32).copy(),
            "last_mode": mode_one_hot,
            "last_beam": np.asarray([last_action.beam / max(1, self.n_beams - 1)], dtype=np.float32),
            "local_summary": local_summary,
        }

    def _normalize_actions(
        self,
        actions: Sequence[Action | Mapping[str, Any] | Sequence[Any]] | Mapping[int | str, Any],
    ) -> list[Action]:
        if isinstance(actions, Mapping):
            entries = []
            for node in range(self.n_agents):
                if node in actions:
                    entries.append(actions[node])
                elif str(node) in actions:
                    entries.append(actions[str(node)])
                else:
                    raise ValueError(f"Missing action for agent {node}.")
        else:
            entries = list(actions)
        if len(entries) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} actions, got {len(entries)}.")
        return [self._parse_action(entry) for entry in entries]

    def _parse_action(self, raw: Action | Mapping[str, Any] | Sequence[Any]) -> Action:
        if isinstance(raw, Action):
            mode, beam = raw.mode, raw.beam
        elif isinstance(raw, Mapping):
            if "mode" not in raw:
                raise ValueError("Action mapping must contain a 'mode' field.")
            mode = raw["mode"]
            beam = raw.get("beam", 0)
        elif isinstance(raw, (tuple, list)) and len(raw) >= 2:
            mode, beam = raw[0], raw[1]
        else:
            raise TypeError("Action must be Action, mapping, or (mode, beam) pair.")
        parsed_mode = _parse_mode(mode)
        parsed_beam = int(beam)
        if parsed_mode == MODE_IDLE:
            parsed_beam = 0
        if not 0 <= parsed_beam < self.n_beams:
            raise ValueError(f"Beam index {parsed_beam} outside [0, {self.n_beams}).")
        return Action(parsed_mode, parsed_beam)

    def _rewards(
        self,
        actions: list[Action],
        new_edges: list[tuple[int, int]],
        empty_by_node: np.ndarray,
        fail_before: np.ndarray,
        success_before: np.ndarray,
    ) -> np.ndarray:
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        for node, action in enumerate(actions):
            if action.mode == MODE_SENSE:
                rewards[node] -= 0.01
            elif action.mode == MODE_IDLE:
                rewards[node] -= 0.005
            elif empty_by_node[node]:
                rewards[node] -= 0.02

        fail_delta = self._sim.fail_count.sum(axis=1) - fail_before
        success_delta = self._sim.success_count.sum(axis=1) - success_before
        rewards -= (fail_delta > 0.0).astype(np.float32) * 0.05
        rewards += (success_delta > 0.0).astype(np.float32) * 0.02
        for i, j in new_edges:
            rewards[i] += 1.0
            rewards[j] += 1.0
        return rewards

    def _safe_info(self, new_edges_count: int) -> dict[str, Any]:
        components = connected_components(self.n_agents, self._sim.discovered_edges)
        largest = max((len(component) for component in components), default=0)
        isolated = sum(1 for component in components if len(component) == 1)
        return {
            "slot": self._slot,
            "seed": self.seed,
            "n_agents": self.n_agents,
            "n_beams": self.n_beams,
            "mode_names": MODE_NAMES,
            "new_edges_count": int(new_edges_count),
            "discovered_edges_count": len(self._sim.discovered_edges),
            "scan_actions": self._sim.scan_actions,
            "tx_actions": self._sim.tx_actions,
            "rx_actions": self._sim.rx_actions,
            "sense_actions": self._sim.sense_actions,
            "idle_actions": self._sim.idle_actions,
            "piggyback_sense_actions": self._sim.piggyback_sense_actions,
            "discovery_per_scan_action": len(self._sim.discovered_edges) / max(1, self._sim.scan_actions),
            "empty_scan_ratio": self._sim.empty_scans / max(1, self._sim.scan_actions),
            "collision_count": self._sim.collision_count,
            "largest_component_size": largest,
            "connected_components": len(components),
            "lcc_ratio": largest / max(1, self.n_agents),
            "isolated_node_ratio": isolated / max(1, self.n_agents),
            "lambda2": algebraic_connectivity(self.n_agents, self._sim.discovered_edges),
            "mobility_model": str(self.cfg.mobility.get("model", "gauss_markov")),
        }

    def _positions(self) -> np.ndarray:
        return np.asarray([state.position for state in self._sim.states])

    def _velocities(self) -> np.ndarray:
        return np.asarray([state.velocity for state in self._sim.states])

    def _attitudes(self) -> np.ndarray:
        return np.asarray([[state.yaw, state.pitch, state.roll] for state in self._sim.states])


def _parse_mode(mode: str | int | np.integer) -> str:
    if isinstance(mode, (int, np.integer)):
        idx = int(mode)
        if not 0 <= idx < len(MODE_NAMES):
            raise ValueError(f"Mode index {idx} outside [0, {len(MODE_NAMES)}).")
        return MODE_NAMES[idx]
    text = str(mode).lower()
    aliases = {"s": MODE_SENSE, "t": MODE_TX, "r": MODE_RX, "i": MODE_IDLE}
    text = aliases.get(text, text)
    if text not in MODE_TO_INDEX:
        raise ValueError(f"Unsupported action mode: {mode}")
    return text


def _speed_scale(mobility: Mapping[str, Any]) -> float:
    max_speed = float(mobility.get("max_speed_mps", 0.0) or 0.0)
    if max_speed > 0.0:
        return max_speed
    mean = float(mobility.get("speed_mean_mps", 15.0))
    std = float(mobility.get("speed_std_mps", 3.0))
    return max(1.0, mean + 3.0 * std)


def _edge_array(edges: set[tuple[int, int]]) -> np.ndarray:
    if not edges:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(sorted(edges), dtype=np.int64)


def _adjacency(n_agents: int, edges: set[tuple[int, int]]) -> np.ndarray:
    adjacency = np.zeros((n_agents, n_agents), dtype=np.float32)
    for i, j in edges:
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    return adjacency
