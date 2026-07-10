from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import SimulationConfig
from .mobility import step_states
from .simulator import (
    ACCESS_AGGRESSIVE,
    ACCESS_BACKOFF,
    ACCESS_NORMAL,
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
ACCESS_GATE_NAMES = (ACCESS_BACKOFF, ACCESS_NORMAL, ACCESS_AGGRESSIVE)
ACCESS_GATE_TO_INDEX = {name: idx for idx, name in enumerate(ACCESS_GATE_NAMES)}
REWARD_VERSIONS = ("legacy", "collision_topology", "discovery_first", "discovery_access_stable")
CANDIDATE_SOURCES = ("default", "wang_table")


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

    def __init__(
        self,
        config: SimulationConfig,
        seed: int | None = None,
        protocol: str = "isac_structured_marl",
        reward_version: str = "legacy",
        candidate_source: str = "default",
        collect_slot_metrics: bool = True,
        rich_info: bool = True,
    ):
        self.cfg = config
        self.seed = int(config.seed if seed is None else seed)
        self.protocol = str(protocol)
        self.reward_version = _parse_reward_version(reward_version)
        self.candidate_source = _parse_candidate_source(candidate_source)
        self.collect_slot_metrics = bool(collect_slot_metrics)
        self.rich_info = bool(rich_info)
        self._sim = NeighborDiscoverySimulator(config, protocol=self.protocol, seed=self.seed)
        self._slot = 0
        self._last_actions: list[Action] = [Action(MODE_IDLE, 0) for _ in range(config.n_nodes)]
        self._access_gate_counts = {name: 0 for name in ACCESS_GATE_NAMES}

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
        self._access_gate_counts = {name: 0 for name in ACCESS_GATE_NAMES}
        return self._observations(), self._info(new_edges_count=0)

    def step(
        self,
        actions: Sequence[Action | Mapping[str, Any] | Sequence[Any]] | Mapping[int | str, Any],
    ) -> tuple[list[dict[str, np.ndarray | int]], np.ndarray, bool, bool, dict[str, Any]]:
        if not self._sim.states:
            raise RuntimeError("Environment must be reset before step().")
        if self._slot >= self.cfg.slots_per_episode:
            raise RuntimeError("Episode already truncated. Call reset() before step().")

        parsed_actions = self._normalize_actions(actions)
        for action in parsed_actions:
            self._access_gate_counts[action.access_gate] += 1
        execution_actions = [self._apply_access_gate(node, action) for node, action in enumerate(parsed_actions)]
        true_comm_edges = self._sim.true_edges(self.cfg.communication_range_m)
        for edge in true_comm_edges:
            self._sim.first_true_slot.setdefault(edge, self._slot)

        self._sim.age += 1.0
        self._sim.belief *= self.cfg.confidence_decay

        empty_by_node = np.zeros(self.n_agents, dtype=bool)
        for node, action in enumerate(execution_actions):
            if action.mode in (MODE_TX, MODE_RX):
                empty_by_node[node] = not self._sim.beam_has_comm_neighbor(node, action.beam, true_comm_edges)

        fail_before = self._sim.fail_count.sum(axis=1).copy()
        success_before = self._sim.success_count.sum(axis=1).copy()
        collision_fail_before = self._sim.collision_fail_count.sum(axis=1).copy()
        topology_before = (
            self._topology_reward_snapshot()
            if self.reward_version in {"collision_topology", "discovery_first", "discovery_access_stable"}
            else None
        )

        self._sim._candidate_pool_cache.clear()
        self._sim.snapshot_pre_sensing_candidates(self._slot)
        self._sim.update_action_counts(execution_actions, self._slot)
        self._sim.update_empty_scan_counts(execution_actions, true_comm_edges, self._slot)
        self._sim.update_sensing(execution_actions, self._slot)
        self._sim._candidate_pool_cache.clear()
        new_edges = self._sim.resolve_discoveries(self._slot, execution_actions, true_comm_edges)
        if self.collect_slot_metrics:
            metric_row = self._sim.slot_metrics(
                episode=0,
                slot=self._slot,
                true_comm_edges=true_comm_edges,
                new_edges=new_edges,
            )
            self._sim.per_slot_rows.append(metric_row)

        rewards = self._rewards(
            parsed_actions,
            execution_actions,
            new_edges,
            empty_by_node,
            fail_before,
            success_before,
            collision_fail_before,
            topology_before,
        )
        step_states(
            self._sim.states,
            self.cfg.area_size_m,
            self.cfg.mobility,
            self.cfg.slot_duration_s,
            self._slot,
            self._sim.mobility_rng,
        )
        self._sim.invalidate_geometry_cache()
        self._last_actions = execution_actions
        self._slot += 1

        terminated = False
        truncated = self._slot >= self.cfg.slots_per_episode
        return self._observations(), rewards, terminated, truncated, self._info(new_edges_count=len(new_edges))

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
        topology_deficit = max(0.0, self.cfg.target_degree - degree) / max(1, self.cfg.target_degree)
        last_action = self._last_actions[node]
        mode_one_hot = np.zeros(len(MODE_NAMES), dtype=np.float32)
        mode_one_hot[MODE_TO_INDEX[last_action.mode]] = 1.0
        candidate = self._candidate_features_for(node)
        contention_state = self._contention_state_for(node, degree, topology_deficit, candidate)

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
                self._sim.node_empty_scans[node] / max(1, self._sim.node_scan_actions[node]),
                self._sim.node_collision_count[node] / max(1, self._slot + 1),
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
            "beam_collision": self._sim.collision_fail_count[node].astype(np.float32).copy(),
            "candidate_mask": candidate["mask"],
            "candidate_score": candidate["score"],
            "topology_deficit": np.asarray([topology_deficit], dtype=np.float32),
            "contention_state": contention_state,
            "rule_mode_logits": self._rule_mode_logits_for(node, topology_deficit, int(np.count_nonzero(candidate["mask"]))),
            "last_mode": mode_one_hot,
            "last_access_gate": self._last_access_gate_one_hot(last_action),
            "last_beam": np.asarray([last_action.beam / max(1, self.n_beams - 1)], dtype=np.float32),
            "local_summary": local_summary,
        }

    def _last_access_gate_one_hot(self, action: Action) -> np.ndarray:
        gate = getattr(action, "access_gate", ACCESS_NORMAL)
        gate = gate if gate in ACCESS_GATE_TO_INDEX else ACCESS_NORMAL
        one_hot = np.zeros(len(ACCESS_GATE_NAMES), dtype=np.float32)
        one_hot[ACCESS_GATE_TO_INDEX[gate]] = 1.0
        return one_hot

    def _contention_state_for(
        self,
        node: int,
        degree: int,
        topology_deficit: float,
        candidate: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Local contention/topology summary exposed to decentralized actors.

        The vector is computed from this node's public memory and aggregate
        episode counters. It does not use true neighbor positions or hidden
        adjacency. It lets policies learn role throttling and beam risk without
        requiring centralized execution.
        """

        success = self._sim.success_count[node]
        fail = self._sim.fail_count[node]
        collision = self._sim.collision_fail_count[node]
        empty = self._sim.empty_beam_count[node]
        total_feedback = float(success.sum() + fail.sum() + empty.sum() + 1.0)
        fail_total = float(fail.sum())
        collision_total = float(collision.sum())
        candidate_mask = candidate["mask"]
        candidate_score = candidate["score"]
        candidate_count = float(np.count_nonzero(candidate_mask))
        candidate_fraction = candidate_count / max(1.0, float(self.n_beams))
        masked_scores = candidate_score[candidate_mask > 0.5]
        candidate_score_mean = float(masked_scores.mean()) if masked_scores.size else 0.0
        candidate_score_max = float(masked_scores.max()) if masked_scores.size else 0.0
        return np.asarray(
            [
                degree / max(1, self.n_agents - 1),
                float(topology_deficit),
                float(success.sum()) / total_feedback,
                fail_total / total_feedback,
                collision_total / max(1.0, fail_total + collision_total),
                float(empty.sum()) / total_feedback,
                candidate_fraction,
                candidate_score_mean,
                candidate_score_max,
                float(self._last_actions[node].mode == "tx") - float(self._last_actions[node].mode == "rx"),
            ],
            dtype=np.float32,
        )

    def _candidate_features_for(self, node: int) -> dict[str, np.ndarray]:
        """Local beam proposal features for candidate-constrained policies.

        These features are derived only from local belief/memory variables
        exposed to the actor. They do not use true neighbor positions or
        undiscovered topology. Existing policies may ignore these optional keys.
        """

        if self.candidate_source == "wang_table":
            return self._wang_table_candidate_features_for(node)

        belief = self._sim.belief[node]
        success = np.log1p(self._sim.success_count[node])
        fail = np.log1p(self._sim.fail_count[node])
        age = self._sim.age[node] / max(1.0, float(self.cfg.slots_per_episode))
        positive_ttl = max(50, min(300, int(round(0.50 / max(self.cfg.slot_duration_s, 1e-6)))))
        recency_slots = np.maximum(0.0, self._slot - self._sim.last_positive_slot[node])
        bounded_recency = np.minimum(recency_slots, float(positive_ttl)) / max(1.0, float(positive_ttl))
        raw_score = belief + 0.35 * success - 0.25 * fail - 0.05 * bounded_recency + 0.02 * age
        score_min = float(raw_score.min()) if raw_score.size else 0.0
        score_span = float(raw_score.max() - score_min) if raw_score.size else 0.0
        if score_span > 1e-9:
            score = (raw_score - score_min) / score_span
        else:
            score = np.full_like(raw_score, 0.5, dtype=float)
        top_k = min(self.n_beams, max(4, int(np.ceil(np.sqrt(max(1, self.n_beams))))))
        top_indices = np.argpartition(raw_score, -top_k)[-top_k:]
        threshold = max(float(self.cfg.exploration_floor), float(np.quantile(belief, 0.90)) if self.n_beams > 1 else 0.0)
        mask = (belief >= threshold).astype(np.float32)
        mask[top_indices] = 1.0
        last_beam = int(self._last_actions[node].beam)
        if 0 <= last_beam < self.n_beams:
            mask[last_beam] = 1.0
        return {
            "mask": mask.astype(np.float32, copy=False),
            "score": score.astype(np.float32, copy=False),
        }

    def _wang_table_candidate_features_for(self, node: int) -> dict[str, np.ndarray]:
        """Expose Wang sensing-table candidates to the MARL actor.

        The hard mask is the Wang baseline's executable set, i.e., beams whose
        sensing-table Flag is 1. If all flags are closed, it falls back to all
        beams, matching ``wang2025_table_beam``.
        """

        node = int(node)
        active = (self._sim.wang_sensing_flag[node] > 0.5).astype(np.float32)
        mask = active.copy()
        if not np.any(mask > 0.5):
            mask = np.ones(self.n_beams, dtype=np.float32)

        node_num = self._sim.wang_node_num[node]
        dis_num = self._sim.wang_dis_num[node]
        open_gap = np.maximum(0.0, node_num - dis_num)
        open_norm = np.clip(open_gap / max(1.0, float(self.cfg.target_degree)), 0.0, 1.0)
        known_target = (node_num > 0.0).astype(np.float32)
        positive_ttl = max(50, min(300, int(round(0.50 / max(self.cfg.slot_duration_s, 1e-6)))))
        recency_slots = np.maximum(0.0, self._slot - self._sim.last_positive_slot[node])
        recent = 1.0 - np.minimum(recency_slots, float(positive_ttl)) / max(1.0, float(positive_ttl))
        snr_norm = np.clip((self._sim.wang_snr_db[node] + 20.0) / 50.0, 0.0, 1.0)
        raw_score = active * (0.35 + 0.30 * known_target + 0.20 * open_norm + 0.10 * recent + 0.05 * snr_norm)

        score = np.zeros(self.n_beams, dtype=np.float32)
        visible = raw_score[mask > 0.5]
        if visible.size:
            score_min = float(visible.min())
            score_span = float(visible.max() - score_min)
            if score_span > 1e-9:
                score[mask > 0.5] = ((visible - score_min) / score_span).astype(np.float32)
            else:
                score[mask > 0.5] = 0.5
        return {
            "mask": mask.astype(np.float32, copy=False),
            "score": score.astype(np.float32, copy=False),
        }

    def _rule_mode_logits_for(self, node: int, topology_deficit: float, candidate_count: int) -> np.ndarray:
        """Local coordination prior used as an optional neural residual input."""

        sparse_candidates = candidate_count <= max(2, int(np.ceil(np.sqrt(max(1, self.n_beams)))))
        tx_phase = (node + self._slot) % 2 == 0
        active_bias = 0.25 + 0.75 * float(topology_deficit)
        logits = np.asarray(
            [
                0.10 + 0.40 * float(topology_deficit) + (0.20 if sparse_candidates else 0.0),
                active_bias if tx_phase else 0.15 * float(topology_deficit),
                0.15 * float(topology_deficit) if tx_phase else active_bias,
                -0.60 - 0.40 * float(topology_deficit),
            ],
            dtype=np.float32,
        )
        return logits

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
            mode, beam, access_gate = raw.mode, raw.beam, raw.access_gate
        elif isinstance(raw, Mapping):
            if "mode" not in raw:
                raise ValueError("Action mapping must contain a 'mode' field.")
            mode = raw["mode"]
            beam = raw.get("beam", 0)
            access_gate = raw.get("access_gate", raw.get("gate", ACCESS_NORMAL))
        elif isinstance(raw, (tuple, list)) and len(raw) >= 2:
            mode, beam = raw[0], raw[1]
            access_gate = raw[2] if len(raw) >= 3 else ACCESS_NORMAL
        else:
            raise TypeError("Action must be Action, mapping, or (mode, beam) pair.")
        parsed_mode = _parse_mode(mode)
        parsed_gate = _parse_access_gate(access_gate)
        parsed_beam = int(beam)
        if parsed_mode == MODE_IDLE:
            parsed_beam = 0
        if not 0 <= parsed_beam < self.n_beams:
            raise ValueError(f"Beam index {parsed_beam} outside [0, {self.n_beams}).")
        return Action(parsed_mode, parsed_beam, parsed_gate)

    def _apply_access_gate(self, node: int, action: Action) -> Action:
        if action.access_gate == ACCESS_BACKOFF and action.mode == MODE_TX and self._has_collision_evidence(node, action.beam):
            return Action(MODE_RX, action.beam, action.access_gate)
        if action.access_gate == ACCESS_AGGRESSIVE and action.mode == MODE_RX and self._has_low_collision_evidence(node, action.beam):
            return Action(MODE_TX, action.beam, action.access_gate)
        return action

    def _has_collision_evidence(self, node: int, beam: int) -> bool:
        beam_collision = float(self._sim.collision_fail_count[node, beam])
        beam_fail = float(self._sim.fail_count[node, beam])
        episode_pressure = self._sim.node_collision_count[node] / max(1.0, float(self._slot + 1))
        return bool(beam_collision > 0.0 or (episode_pressure > 0.75 and beam_fail > 0.0))

    def _has_low_collision_evidence(self, node: int, beam: int) -> bool:
        beam_collision = float(self._sim.collision_fail_count[node, beam])
        beam_success = float(self._sim.success_count[node, beam])
        degree = sum(1 for edge in self._sim.discovered_edges if node in edge)
        needs_degree = degree < int(self.cfg.target_degree)
        return bool(needs_degree and beam_collision <= beam_success)

    def _rewards(
        self,
        sampled_actions: list[Action],
        actions: list[Action],
        new_edges: list[tuple[int, int]],
        empty_by_node: np.ndarray,
        fail_before: np.ndarray,
        success_before: np.ndarray,
        collision_fail_before: np.ndarray,
        topology_before: dict[str, Any] | None,
    ) -> np.ndarray:
        if self.reward_version in {"discovery_first", "discovery_access_stable"}:
            rewards = self._discovery_first_rewards(
                sampled_actions,
                actions,
                new_edges,
                empty_by_node,
                fail_before,
                success_before,
                collision_fail_before,
                topology_before,
            )
            if self.reward_version == "discovery_access_stable":
                rewards = self._access_stability_reward_shaping(rewards, actions, empty_by_node, topology_before)
            return rewards

        rewards = np.zeros(self.n_agents, dtype=np.float32)
        for node, action in enumerate(actions):
            if action.mode == MODE_SENSE:
                rewards[node] -= 0.01
            elif action.mode == MODE_IDLE:
                rewards[node] -= 0.005
            elif empty_by_node[node]:
                rewards[node] -= 0.02
            if sampled_actions[node].access_gate == ACCESS_AGGRESSIVE:
                rewards[node] -= 0.003

        fail_delta = self._sim.fail_count.sum(axis=1) - fail_before
        success_delta = self._sim.success_count.sum(axis=1) - success_before
        rewards -= (fail_delta > 0.0).astype(np.float32) * 0.05
        rewards += (success_delta > 0.0).astype(np.float32) * 0.02
        for i, j in new_edges:
            rewards[i] += 1.0
            rewards[j] += 1.0
        if self.reward_version == "collision_topology":
            rewards = self._collision_topology_reward_shaping(
                rewards,
                actions,
                new_edges,
                collision_fail_before,
                topology_before,
            )
        return rewards

    def _topology_reward_snapshot(self) -> dict[str, Any]:
        components = connected_components(self.n_agents, self._sim.discovered_edges)
        component_id: dict[int, int] = {}
        for index, component in enumerate(components):
            for node in component:
                component_id[int(node)] = index
        degree = np.zeros(self.n_agents, dtype=np.float32)
        for i, j in self._sim.discovered_edges:
            degree[i] += 1.0
            degree[j] += 1.0
        return {"component_id": component_id, "degree": degree}

    def _collision_topology_reward_shaping(
        self,
        rewards: np.ndarray,
        actions: list[Action],
        new_edges: list[tuple[int, int]],
        collision_fail_before: np.ndarray,
        topology_before: dict[str, Any] | None,
    ) -> np.ndarray:
        shaped = rewards.astype(np.float32, copy=True)
        collision_delta = self._sim.collision_fail_count.sum(axis=1) - collision_fail_before
        shaped -= np.minimum(collision_delta, 3.0).astype(np.float32) * 0.08

        tx_count = sum(1 for action in actions if action.mode == MODE_TX)
        rx_count = sum(1 for action in actions if action.mode == MODE_RX)
        role_imbalance = abs(tx_count - rx_count) / max(1, tx_count + rx_count)
        if role_imbalance > 0.0:
            for node, action in enumerate(actions):
                if action.mode in (MODE_TX, MODE_RX):
                    shaped[node] -= 0.02 * float(role_imbalance)

        if topology_before is None:
            return shaped
        component_id = topology_before["component_id"]
        degree_before = np.asarray(topology_before["degree"], dtype=np.float32)
        for i, j in new_edges:
            if component_id.get(i, i) != component_id.get(j, j):
                shaped[i] += 0.30
                shaped[j] += 0.30
            if degree_before[i] < float(self.cfg.target_degree):
                shaped[i] += 0.10
            if degree_before[j] < float(self.cfg.target_degree):
                shaped[j] += 0.10
        return shaped

    def _discovery_first_rewards(
        self,
        sampled_actions: list[Action],
        actions: list[Action],
        new_edges: list[tuple[int, int]],
        empty_by_node: np.ndarray,
        fail_before: np.ndarray,
        success_before: np.ndarray,
        collision_fail_before: np.ndarray,
        topology_before: dict[str, Any] | None,
    ) -> np.ndarray:
        """Reward early useful discoveries without making collision the target.

        Collision is represented as a light failed-access cost. The dominant
        signal is direct discovery, with extra credit for early discoveries and
        topology-improving links.
        """

        rewards = np.zeros(self.n_agents, dtype=np.float32)
        horizon = max(1.0, float(self.cfg.slots_per_episode - 1))
        progress = min(1.0, float(self._slot) / horizon)
        early_bonus = 1.0 - progress
        degree_before = (
            np.asarray(topology_before.get("degree", np.zeros(self.n_agents)), dtype=np.float32)
            if topology_before is not None
            else np.zeros(self.n_agents, dtype=np.float32)
        )
        component_id = topology_before.get("component_id", {}) if topology_before is not None else {}

        for node, action in enumerate(actions):
            if action.mode == MODE_SENSE:
                rewards[node] -= 0.03
            elif action.mode == MODE_IDLE:
                deficit = max(0.0, float(self.cfg.target_degree) - float(degree_before[node]))
                rewards[node] -= 0.006 + 0.004 * min(1.0, deficit / max(1.0, float(self.cfg.target_degree)))
            elif action.mode in (MODE_TX, MODE_RX):
                rewards[node] -= 0.002

            if sampled_actions[node].access_gate == ACCESS_AGGRESSIVE:
                rewards[node] -= 0.001

        fail_delta = np.maximum(0.0, self._sim.fail_count.sum(axis=1) - fail_before)
        success_delta = np.maximum(0.0, self._sim.success_count.sum(axis=1) - success_before)
        collision_delta = np.maximum(0.0, self._sim.collision_fail_count.sum(axis=1) - collision_fail_before)
        rewards -= np.minimum(fail_delta, 2.0).astype(np.float32) * 0.010
        rewards -= np.minimum(collision_delta, 2.0).astype(np.float32) * 0.015
        rewards += np.minimum(success_delta, 2.0).astype(np.float32) * 0.050

        edge_reward = 1.15 + 0.75 * early_bonus
        for i, j in new_edges:
            rewards[i] += edge_reward
            rewards[j] += edge_reward
            if component_id.get(i, i) != component_id.get(j, j):
                rewards[i] += 0.35
                rewards[j] += 0.35
            if degree_before[i] < float(self.cfg.target_degree):
                rewards[i] += 0.18
            if degree_before[j] < float(self.cfg.target_degree):
                rewards[j] += 0.18
            if degree_before[i] <= 0.0:
                rewards[i] += 0.12
            if degree_before[j] <= 0.0:
                rewards[j] += 0.12
        return rewards.astype(np.float32, copy=False)

    def _access_stability_reward_shaping(
        self,
        rewards: np.ndarray,
        actions: list[Action],
        empty_by_node: np.ndarray,
        topology_before: dict[str, Any] | None,
    ) -> np.ndarray:
        """Keep the learned access controller active without hiding the main objective.

        This shaping is deliberately smaller than discovery rewards. It prevents
        late-stage over-idle collapse and gives the beam head local credit when a
        TX/RX beam points at a viable neighbor direction before a full handshake
        happens.
        """

        shaped = rewards.astype(np.float32, copy=True)
        degree_before = (
            np.asarray(topology_before.get("degree", np.zeros(self.n_agents)), dtype=np.float32)
            if topology_before is not None
            else np.zeros(self.n_agents, dtype=np.float32)
        )
        tx_count = sum(1 for action in actions if action.mode == MODE_TX)
        idle_count = sum(1 for action in actions if action.mode == MODE_IDLE)
        tx_fraction = tx_count / max(1.0, float(self.n_agents))
        idle_fraction = idle_count / max(1.0, float(self.n_agents))
        low_tx_pressure = max(0.0, 0.35 - tx_fraction)
        high_tx_pressure = max(0.0, tx_fraction - 0.70)

        for node, action in enumerate(actions):
            deficit = max(0.0, float(self.cfg.target_degree) - float(degree_before[node]))
            deficit_ratio = min(1.0, deficit / max(1.0, float(self.cfg.target_degree)))
            if action.mode == MODE_IDLE:
                shaped[node] -= 0.018 + 0.018 * deficit_ratio
                shaped[node] -= 0.020 * idle_fraction
                shaped[node] -= 0.030 * low_tx_pressure
                continue

            if action.mode == MODE_TX:
                shaped[node] += 0.004 + 0.008 * low_tx_pressure
                shaped[node] -= 0.010 * high_tx_pressure
            elif action.mode == MODE_RX:
                shaped[node] -= 0.004 * low_tx_pressure

            if (
                self.candidate_source == "wang_table"
                and action.mode in (MODE_TX, MODE_RX)
                and 0 <= int(action.beam) < self.n_beams
            ):
                beam = int(action.beam)
                if self._sim.wang_sensing_flag[node, beam] > 0.5:
                    shaped[node] += 0.002
                if self._sim.wang_node_num[node, beam] > self._sim.wang_dis_num[node, beam]:
                    shaped[node] += 0.006

        return shaped.astype(np.float32, copy=False)

    def _safe_info(self, new_edges_count: int) -> dict[str, Any]:
        true_edges = self._sim.true_edges(self.cfg.communication_range_m)
        active_edges = self._sim.discovered_edges.intersection(true_edges)
        components = connected_components(self.n_agents, active_edges)
        largest = max((len(component) for component in components), default=0)
        isolated = sum(1 for component in components if len(component) == 1)
        return {
            "slot": self._slot,
            "seed": self.seed,
            "n_agents": self.n_agents,
            "n_beams": self.n_beams,
            "mode_names": MODE_NAMES,
            "access_gate_names": ACCESS_GATE_NAMES,
            "new_edges_count": int(new_edges_count),
            "discovered_edges_count": len(self._sim.discovered_edges),
            "active_discovered_edges_count": len(active_edges),
            "scan_actions": self._sim.scan_actions,
            "tx_actions": self._sim.tx_actions,
            "rx_actions": self._sim.rx_actions,
            "sense_actions": self._sim.sense_actions,
            "idle_actions": self._sim.idle_actions,
            "piggyback_sense_actions": self._sim.piggyback_sense_actions,
            "sensing_observations": self._sim.sensing_observations,
            "sensing_target_observations": self._sim.sensing_target_observations,
            "sensing_detection_rate": self._sim.sensing_detection_count / max(1, self._sim.sensing_target_observations),
            "sensing_false_alarm_ratio": self._sim.sensing_false_alarm_count / max(1, self._sim.sensing_observations),
            "sensing_miss_ratio": self._sim.sensing_miss_count / max(1, self._sim.sensing_target_observations),
            "mean_sensing_pd": self._sim.sensing_pd_sum / max(1, self._sim.sensing_target_observations),
            "mean_sensing_snr_db": self._sim.sensing_snr_db_sum / max(1, self._sim.sensing_snr_sample_count),
            "discovery_per_scan_action": len(self._sim.discovered_edges) / max(1, self._sim.scan_actions),
            "empty_scan_ratio": self._sim.empty_scans / max(1, self._sim.scan_actions),
            "hello_transmissions": self._sim.tx_actions,
            "handshake_attempts": self._sim.handshake_attempts,
            "role_compatible_pairs": self._sim.role_compatible_pairs,
            "aligned_handshake_opportunities": self._sim.aligned_handshake_opportunities,
            "forward_decodes": self._sim.forward_decodes,
            "handshake_successes": self._sim.handshake_successes,
            "forward_decode_failures": self._sim.forward_decode_failures,
            "ack_decode_failures": self._sim.ack_decode_failures,
            "interference_limited_failures": self._sim.interference_limited_failures,
            "phy_outage_failures": self._sim.phy_outage_failures,
            "mean_handshake_sinr_db": float(np.mean(self._sim.handshake_sinr_samples_db))
            if self._sim.handshake_sinr_samples_db
            else 0.0,
            "collision_count": self._sim.collision_count,
            "largest_component_size": largest,
            "connected_components": len(components),
            "lcc_ratio": largest / max(1, self.n_agents),
            "isolated_node_ratio": isolated / max(1, self.n_agents),
            "lambda2": algebraic_connectivity(self.n_agents, active_edges),
            "knowledge_lambda2": algebraic_connectivity(self.n_agents, self._sim.discovered_edges),
            "mobility_model": str(self.cfg.mobility.get("model", "gauss_markov")),
            "reward_version": self.reward_version,
            **self.access_gate_summary(),
        }

    def _minimal_info(self, new_edges_count: int) -> dict[str, Any]:
        return {
            "slot": self._slot,
            "seed": self.seed,
            "n_agents": self.n_agents,
            "n_beams": self.n_beams,
            "mode_names": MODE_NAMES,
            "access_gate_names": ACCESS_GATE_NAMES,
            "new_edges_count": int(new_edges_count),
            "reward_version": self.reward_version,
            **self.access_gate_summary(),
        }

    def _info(self, new_edges_count: int) -> dict[str, Any]:
        if self.rich_info:
            return self._safe_info(new_edges_count)
        return self._minimal_info(new_edges_count)

    def access_gate_summary(self) -> dict[str, int | float]:
        total = max(1, sum(int(value) for value in self._access_gate_counts.values()))
        out: dict[str, int | float] = {"access_gate_total": int(sum(self._access_gate_counts.values()))}
        for name in ACCESS_GATE_NAMES:
            count = int(self._access_gate_counts[name])
            out[f"access_gate_{name}_count"] = count
            out[f"access_gate_{name}_ratio"] = count / total
        return out

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


def _parse_access_gate(access_gate: str | int | np.integer) -> str:
    if isinstance(access_gate, (int, np.integer)):
        idx = int(access_gate)
        if not 0 <= idx < len(ACCESS_GATE_NAMES):
            raise ValueError(f"Access-gate index {idx} outside [0, {len(ACCESS_GATE_NAMES)}).")
        return ACCESS_GATE_NAMES[idx]
    text = str(access_gate).lower()
    aliases = {"b": ACCESS_BACKOFF, "n": ACCESS_NORMAL, "a": ACCESS_AGGRESSIVE}
    text = aliases.get(text, text)
    if text not in ACCESS_GATE_TO_INDEX:
        raise ValueError(f"Unsupported access gate: {access_gate}")
    return text


def _parse_reward_version(reward_version: str) -> str:
    text = str(reward_version).lower()
    if text not in REWARD_VERSIONS:
        raise ValueError(f"reward_version must be one of {REWARD_VERSIONS}.")
    return text


def _parse_candidate_source(candidate_source: str) -> str:
    text = str(candidate_source).lower()
    if text not in CANDIDATE_SOURCES:
        raise ValueError(f"candidate_source must be one of {CANDIDATE_SOURCES}.")
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
