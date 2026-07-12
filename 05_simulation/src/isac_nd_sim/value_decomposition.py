from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .beam import beam_center_direction_features
from .neural_contention_actor_critic import (
    ContentionFeatures,
    observations_to_batched_contention_tensors,
)
from .simulator import Action, MODE_RX, MODE_TX


VALUE_BASED_ALGORITHMS = ("idqn", "shared_idqn", "vdn", "qmix")
VALUE_ACTION_MODES = (MODE_TX, MODE_RX)
VALUE_ACTION_CONTRACTS = (
    "joint_role_beam",
    "beam_only_fixed_role",
    "beam_only_complementary_role",
)
BEAM_ONLY_ACTION_CONTRACTS = VALUE_ACTION_CONTRACTS[1:]


def is_beam_only_action_contract(action_contract: str) -> bool:
    return str(action_contract) in BEAM_ONLY_ACTION_CONTRACTS


def requires_global_training_state(algorithm: str) -> bool:
    if str(algorithm) not in VALUE_BASED_ALGORITHMS:
        raise ValueError(f"algorithm must be one of {VALUE_BASED_ALGORITHMS}.")
    return str(algorithm) == "qmix"


@dataclass
class JointTransition:
    observations: list[dict[str, Any]]
    action_indices: np.ndarray
    rewards: np.ndarray
    next_observations: list[dict[str, Any]]
    done: bool | np.ndarray
    central_state: np.ndarray
    next_central_state: np.ndarray


class JointReplayBuffer:
    def __init__(self, capacity: int):
        if int(capacity) <= 0:
            raise ValueError("Replay capacity must be positive.")
        self._rows: deque[JointTransition] = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self._rows)

    def append(self, transition: JointTransition) -> None:
        self._rows.append(transition)

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[JointTransition]:
        if int(batch_size) > len(self._rows):
            raise ValueError("Replay buffer does not contain enough transitions.")
        indices = rng.choice(len(self._rows), size=int(batch_size), replace=False)
        rows = list(self._rows)
        return [rows[int(index)] for index in indices]


@dataclass
class LocalTransition:
    observation: dict[str, Any]
    action_index: int
    reward: float
    next_observation: dict[str, Any]
    done: bool


class IndependentReplayBuffer:
    """Separate local replay memory and sampling stream for every UAV."""

    def __init__(self, n_agents: int, capacity: int, seed: int, *, reward_scope: str = "team"):
        if int(n_agents) <= 0 or int(capacity) <= 0:
            raise ValueError("Independent replay dimensions must be positive.")
        self.n_agents = int(n_agents)
        self.reward_scope = str(reward_scope)
        if self.reward_scope not in {"team", "local"}:
            raise ValueError("reward_scope must be 'team' or 'local'.")
        self._rows = [deque(maxlen=int(capacity)) for _node in range(self.n_agents)]
        self._rngs = [np.random.default_rng(int(seed) + 104729 * node) for node in range(self.n_agents)]

    def __len__(self) -> int:
        return min((len(rows) for rows in self._rows), default=0)

    def append(self, transition: JointTransition) -> None:
        team_reward = float(np.mean(transition.rewards))
        for node in range(self.n_agents):
            self._rows[node].append(
                LocalTransition(
                    observation=transition.observations[node],
                    action_index=int(transition.action_indices[node]),
                    reward=team_reward if self.reward_scope == "team" else float(transition.rewards[node]),
                    next_observation=transition.next_observations[node],
                    done=bool(transition.done),
                )
            )

    def sample(self, batch_size: int, _rng: np.random.Generator) -> list[JointTransition]:
        if int(batch_size) > len(self):
            raise ValueError("Independent replay buffers do not contain enough transitions.")
        per_agent_samples: list[list[LocalTransition]] = []
        for node, rows in enumerate(self._rows):
            indices = self._rngs[node].choice(len(rows), size=int(batch_size), replace=False)
            materialized = list(rows)
            per_agent_samples.append([materialized[int(index)] for index in indices])

        zeros = np.zeros(1, dtype=np.float32)
        joint_batch: list[JointTransition] = []
        for batch_index in range(int(batch_size)):
            local_rows = [per_agent_samples[node][batch_index] for node in range(self.n_agents)]
            joint_batch.append(
                JointTransition(
                    observations=[row.observation for row in local_rows],
                    action_indices=np.asarray([row.action_index for row in local_rows], dtype=np.int64),
                    rewards=np.asarray([row.reward for row in local_rows], dtype=np.float32),
                    next_observations=[row.next_observation for row in local_rows],
                    done=np.asarray([row.done for row in local_rows], dtype=np.float32),
                    central_state=zeros,
                    next_central_state=zeros,
                )
            )
        return joint_batch


def build_local_q_network(
    n_beams: int,
    hidden_dim: int,
    *,
    residual_features: bool = True,
    action_contract: str = "joint_role_beam",
    azimuth_cells: int | None = None,
    elevation_cells: int = 1,
):
    import torch
    import torch.nn as nn

    dims = ContentionFeatures()
    if str(action_contract) not in VALUE_ACTION_CONTRACTS:
        raise ValueError(f"action_contract must be one of {VALUE_ACTION_CONTRACTS}.")
    azimuth_cells = int(n_beams if azimuth_cells is None else azimuth_cells)
    elevation_cells = int(elevation_cells)
    if azimuth_cells * elevation_cells != int(n_beams):
        raise ValueError("azimuth_cells * elevation_cells must equal n_beams.")
    beam_dim = (dims.residual_beam_dim if residual_features else dims.beam_dim) + 3
    context_dim = 9 + 4 + 4 + 1 + 1 + dims.candidate_stats_dim + 2 * int(hidden_dim)

    class LocalActionQNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_beams = int(n_beams)
            self.hidden_dim = int(hidden_dim)
            self.residual_features = bool(residual_features)
            self.action_contract = str(action_contract)
            self.register_buffer(
                "beam_center_directions",
                torch.as_tensor(
                    beam_center_direction_features(azimuth_cells, elevation_cells),
                    dtype=torch.float32,
                ),
            )
            self.beam_encoder = nn.Sequential(
                nn.Linear(beam_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            self.contention_encoder = nn.Sequential(
                nn.Linear(dims.contention_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            self.context_encoder = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            if self.action_contract == "joint_role_beam":
                self.role_embedding = nn.Parameter(torch.empty(len(VALUE_ACTION_MODES), hidden_dim))
            self.action_advantage = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.state_value = nn.Linear(hidden_dim, 1)
            if self.action_contract == "joint_role_beam":
                nn.init.normal_(self.role_embedding, mean=0.0, std=0.05)

        def forward(self, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
            directions = self.beam_center_directions.unsqueeze(0).expand(
                tensors["beam_features"].shape[0], -1, -1
            )
            beam_tokens = self.beam_encoder(
                torch.cat([tensors["beam_features"], directions], dim=-1)
            )
            contention = self.contention_encoder(tensors["contention_state"])
            beam_context = beam_tokens.mean(dim=1)
            context_input = torch.cat(
                [
                    tensors["self_state"],
                    tensors["local_summary"],
                    tensors["last_mode"],
                    tensors["last_beam"],
                    tensors["topology_deficit"],
                    tensors["candidate_stats"],
                    beam_context,
                    contention,
                ],
                dim=1,
            )
            context = self.context_encoder(context_input)
            if is_beam_only_action_contract(self.action_contract):
                action_tokens = torch.tanh(beam_tokens + context[:, None, :])
                advantage = self.action_advantage(action_tokens).squeeze(-1)
                value = self.state_value(context)
                return value + advantage - advantage.mean(dim=1, keepdim=True)

            action_tokens = torch.tanh(
                beam_tokens[:, None, :, :]
                + context[:, None, None, :]
                + self.role_embedding[None, :, None, :]
            )
            advantage = self.action_advantage(action_tokens).squeeze(-1)
            value = self.state_value(context).unsqueeze(-1)
            q_values = value + advantage - advantage.mean(dim=(1, 2), keepdim=True)
            return q_values.reshape(q_values.shape[0], len(VALUE_ACTION_MODES) * self.n_beams)

    return LocalActionQNetwork()


class MonotonicQMix:
    def __new__(cls, n_agents: int, state_dim: int, embed_dim: int):
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional

        class Mixer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_agents = int(n_agents)
                self.embed_dim = int(embed_dim)
                self.hyper_w1 = nn.Linear(state_dim, self.n_agents * self.embed_dim)
                self.hyper_b1 = nn.Linear(state_dim, self.embed_dim)
                self.hyper_w2 = nn.Linear(state_dim, self.embed_dim)
                self.value = nn.Sequential(
                    nn.Linear(state_dim, self.embed_dim),
                    nn.SiLU(),
                    nn.Linear(self.embed_dim, 1),
                )

            def forward(self, agent_q: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
                batch = agent_q.shape[0]
                weights1 = functional.softplus(self.hyper_w1(state)).view(
                    batch, self.n_agents, self.embed_dim
                )
                bias1 = self.hyper_b1(state).view(batch, 1, self.embed_dim)
                hidden = functional.elu(torch.bmm(agent_q.unsqueeze(1), weights1) + bias1)
                weights2 = functional.softplus(self.hyper_w2(state)).view(batch, self.embed_dim, 1)
                return (torch.bmm(hidden, weights2).squeeze(-1).squeeze(-1) + self.value(state).squeeze(-1))

        return Mixer()


class ValueDecompositionLearner:
    def __init__(
        self,
        algorithm: str,
        n_agents: int,
        n_beams: int,
        state_dim: int,
        *,
        hidden_dim: int = 64,
        mixer_dim: int = 32,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gradient_clip: float = 10.0,
        reward_scope: str = "team",
        action_contract: str = "joint_role_beam",
        azimuth_cells: int | None = None,
        elevation_cells: int = 1,
        device: str = "cpu",
    ):
        import torch

        if str(algorithm) not in VALUE_BASED_ALGORITHMS:
            raise ValueError(f"algorithm must be one of {VALUE_BASED_ALGORITHMS}.")
        self.algorithm = str(algorithm)
        self.n_agents = int(n_agents)
        self.n_beams = int(n_beams)
        self.azimuth_cells = int(self.n_beams if azimuth_cells is None else azimuth_cells)
        self.elevation_cells = int(elevation_cells)
        if self.azimuth_cells * self.elevation_cells != self.n_beams:
            raise ValueError("azimuth_cells * elevation_cells must equal n_beams.")
        self.action_contract = str(action_contract)
        if self.action_contract not in VALUE_ACTION_CONTRACTS:
            raise ValueError(f"action_contract must be one of {VALUE_ACTION_CONTRACTS}.")
        self.n_actions = (
            self.n_beams
            if is_beam_only_action_contract(self.action_contract)
            else len(VALUE_ACTION_MODES) * self.n_beams
        )
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.mixer_dim = int(mixer_dim)
        self.gamma = float(gamma)
        self.gradient_clip = float(gradient_clip)
        self.reward_scope = str(reward_scope)
        if self.reward_scope not in {"team", "local"}:
            raise ValueError("reward_scope must be 'team' or 'local'.")
        self.device = torch.device(device)

        network_count = self.n_agents if self.algorithm == "idqn" else 1
        self.q_networks = torch.nn.ModuleList(
            [
                build_local_q_network(
                    self.n_beams,
                    self.hidden_dim,
                    residual_features=True,
                    action_contract=self.action_contract,
                    azimuth_cells=self.azimuth_cells,
                    elevation_cells=self.elevation_cells,
                )
                for _ in range(network_count)
            ]
        ).to(self.device)
        self.target_q_networks = torch.nn.ModuleList(
            [
                build_local_q_network(
                    self.n_beams,
                    self.hidden_dim,
                    residual_features=True,
                    action_contract=self.action_contract,
                    azimuth_cells=self.azimuth_cells,
                    elevation_cells=self.elevation_cells,
                )
                for _ in range(network_count)
            ]
        ).to(self.device)
        self.mixer = None
        self.target_mixer = None
        if self.algorithm == "qmix":
            self.mixer = MonotonicQMix(self.n_agents, self.state_dim, self.mixer_dim).to(self.device)
            self.target_mixer = MonotonicQMix(self.n_agents, self.state_dim, self.mixer_dim).to(self.device)

        if self.independent_parameters:
            self.optimizers = [
                torch.optim.Adam(network.parameters(), lr=float(learning_rate))
                for network in self.q_networks
            ]
        else:
            parameters = list(self.q_networks.parameters())
            if self.mixer is not None:
                parameters.extend(self.mixer.parameters())
            self.optimizers = [torch.optim.Adam(parameters, lr=float(learning_rate))]
        self.sync_targets()

    @property
    def independent_parameters(self) -> bool:
        return self.algorithm == "idqn"

    @property
    def centralized_training(self) -> bool:
        return self.algorithm in {"vdn", "qmix"}

    def sync_targets(self) -> None:
        self.target_q_networks.load_state_dict(self.q_networks.state_dict())
        if self.mixer is not None and self.target_mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def train(self) -> None:
        self.q_networks.train()
        if self.mixer is not None:
            self.mixer.train()

    def eval(self) -> None:
        self.q_networks.eval()
        if self.mixer is not None:
            self.mixer.eval()

    def q_values(self, observations: Sequence[dict[str, Any]], *, target: bool = False):
        import torch

        if len(observations) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} observations, got {len(observations)}.")
        networks = self.target_q_networks if target else self.q_networks
        if self.independent_parameters:
            rows = []
            for node, observation in enumerate(observations):
                tensors = observations_to_batched_contention_tensors(
                    [observation],
                    self.device,
                    torch,
                    self.n_beams,
                    use_residual_measurement_features=True,
                )
                rows.append(networks[node](tensors).squeeze(0))
            return torch.stack(rows, dim=0)
        tensors = observations_to_batched_contention_tensors(
            list(observations),
            self.device,
            torch,
            self.n_beams,
            use_residual_measurement_features=True,
        )
        return networks[0](tensors)

    def batched_q_values(
        self,
        observations: Sequence[Sequence[dict[str, Any]]],
        *,
        target: bool = False,
    ):
        import torch

        batch_size = len(observations)
        networks = self.target_q_networks if target else self.q_networks
        if self.independent_parameters:
            per_agent = []
            for node in range(self.n_agents):
                tensors = observations_to_batched_contention_tensors(
                    [row[node] for row in observations],
                    self.device,
                    torch,
                    self.n_beams,
                    use_residual_measurement_features=True,
                )
                per_agent.append(networks[node](tensors))
            return torch.stack(per_agent, dim=1)
        flat = [row[node] for row in observations for node in range(self.n_agents)]
        tensors = observations_to_batched_contention_tensors(
            flat,
            self.device,
            torch,
            self.n_beams,
            use_residual_measurement_features=True,
        )
        return networks[0](tensors).view(batch_size, self.n_agents, self.n_actions)

    def update(self, batch: Sequence[JointTransition]) -> dict[str, float]:
        import torch
        import torch.nn.functional as functional

        self.train()
        observations = [row.observations for row in batch]
        next_observations = [row.next_observations for row in batch]
        action_indices = torch.as_tensor(
            np.stack([row.action_indices for row in batch]),
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.as_tensor(
            np.stack([row.rewards for row in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        done_by_agent = torch.as_tensor(
            np.stack(
                [
                    np.full(self.n_agents, float(row.done), dtype=np.float32)
                    if np.isscalar(row.done)
                    else np.asarray(row.done, dtype=np.float32)
                    for row in batch
                ]
            ),
            dtype=torch.float32,
            device=self.device,
        )
        team_done = done_by_agent[:, 0]
        state = torch.as_tensor(
            np.stack([row.central_state for row in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        next_state = torch.as_tensor(
            np.stack([row.next_central_state for row in batch]),
            dtype=torch.float32,
            device=self.device,
        )

        q_all = self.batched_q_values(observations)
        chosen_q = q_all.gather(-1, action_indices.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_online = mask_invalid_actions(
                self.batched_q_values(next_observations), next_observations, self.n_beams
            )
            next_indices = next_online.argmax(dim=-1)
            next_target = self.batched_q_values(next_observations, target=True)
            next_chosen_q = next_target.gather(-1, next_indices.unsqueeze(-1)).squeeze(-1)

        if self.algorithm in {"idqn", "shared_idqn"}:
            learning_rewards = rewards
            if self.reward_scope == "team" and not self.independent_parameters:
                learning_rewards = rewards.mean(dim=-1, keepdim=True).expand_as(rewards)
            targets = learning_rewards + self.gamma * (1.0 - done_by_agent) * next_chosen_q
            prediction = chosen_q
        elif self.algorithm == "vdn":
            prediction = chosen_q.mean(dim=-1)
            targets = rewards.mean(dim=-1) + self.gamma * (1.0 - team_done) * next_chosen_q.mean(dim=-1)
        else:
            if self.mixer is None or self.target_mixer is None:
                raise RuntimeError("QMIX learner requires online and target mixers.")
            prediction = self.mixer(chosen_q, state)
            with torch.no_grad():
                targets = rewards.mean(dim=-1) + self.gamma * (1.0 - team_done) * self.target_mixer(
                    next_chosen_q, next_state
                )

        if self.independent_parameters:
            per_agent_losses = [
                functional.smooth_l1_loss(prediction[:, node], targets[:, node])
                for node in range(self.n_agents)
            ]
            for optimizer in self.optimizers:
                optimizer.zero_grad(set_to_none=True)
            torch.stack(per_agent_losses).sum().backward()
            gradient_norms = []
            for node, optimizer in enumerate(self.optimizers):
                gradient_norms.append(
                    torch.nn.utils.clip_grad_norm_(
                        self.q_networks[node].parameters(), self.gradient_clip
                    )
                )
                optimizer.step()
            loss = torch.stack(per_agent_losses).mean()
            grad_norm = torch.stack([value.detach() for value in gradient_norms]).mean()
        else:
            loss = functional.smooth_l1_loss(prediction, targets)
            optimizer = self.optimizers[0]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]],
                self.gradient_clip,
            )
            optimizer.step()
        return {
            "td_loss": float(loss.detach().cpu().item()),
            "q_mean": float(chosen_q.detach().mean().cpu().item()),
            "target_mean": float(targets.detach().mean().cpu().item()),
            "gradient_norm": float(grad_norm.detach().cpu().item()),
        }

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "n_agents": self.n_agents,
            "n_beams": self.n_beams,
            "azimuth_cells": self.azimuth_cells,
            "elevation_cells": self.elevation_cells,
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "mixer_dim": self.mixer_dim,
            "reward_scope": self.reward_scope,
            "action_contract": self.action_contract,
            "q_networks": self.q_networks.state_dict(),
            "target_q_networks": self.target_q_networks.state_dict(),
            "mixer": self.mixer.state_dict() if self.mixer is not None else None,
            "target_mixer": self.target_mixer.state_dict() if self.target_mixer is not None else None,
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_checkpoint_state(self, state: dict[str, Any], *, load_optimizers: bool = False) -> None:
        expected = {
            "algorithm": self.algorithm,
            "n_agents": self.n_agents,
            "n_beams": self.n_beams,
            "azimuth_cells": self.azimuth_cells,
            "elevation_cells": self.elevation_cells,
            "action_contract": self.action_contract,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"Checkpoint {key}={state.get(key)!r} does not match learner {value!r}."
                )
        self.q_networks.load_state_dict(state["q_networks"])
        self.target_q_networks.load_state_dict(state["target_q_networks"])
        if self.mixer is not None:
            self.mixer.load_state_dict(state["mixer"])
            self.target_mixer.load_state_dict(state["target_mixer"])
        if load_optimizers:
            optimizer_states = state.get("optimizers", [])
            if len(optimizer_states) != len(self.optimizers):
                raise ValueError("Checkpoint optimizer count does not match learner.")
            for optimizer, optimizer_state in zip(
                self.optimizers,
                optimizer_states,
                strict=True,
            ):
                optimizer.load_state_dict(optimizer_state)


def mask_invalid_actions(q_values, observations: Sequence[Sequence[dict[str, Any]]], n_beams: int):
    import torch

    masks = np.stack(
        [
            np.stack(
                [np.asarray(observation["candidate_mask"], dtype=np.float32) > 0.5 for observation in row]
            )
            for row in observations
        ]
    )
    empty_rows = ~masks.any(axis=-1)
    masks[empty_rows] = True
    action_mask = masks if q_values.shape[-1] == int(n_beams) else np.concatenate([masks, masks], axis=-1)
    valid = torch.as_tensor(action_mask, dtype=torch.bool, device=q_values.device)
    return q_values.masked_fill(~valid, -1.0e9)


def select_local_actions(
    q_values: np.ndarray,
    observations: Sequence[dict[str, Any]],
    rng: np.random.Generator,
    *,
    role_uniform_mixture: float,
    beam_uniform_mixture: float,
) -> tuple[list[Action], np.ndarray]:
    if q_values.ndim != 2:
        raise ValueError("q_values must have shape [agents, 2 * beams].")
    n_agents = len(observations)
    n_beams = q_values.shape[1] // len(VALUE_ACTION_MODES)
    if q_values.shape != (n_agents, len(VALUE_ACTION_MODES) * n_beams):
        raise ValueError("q_values and observation counts are inconsistent.")
    if not 0.0 <= float(role_uniform_mixture) <= 1.0:
        raise ValueError("role_uniform_mixture must be in [0, 1].")
    if not 0.0 <= float(beam_uniform_mixture) <= 1.0:
        raise ValueError("beam_uniform_mixture must be in [0, 1].")

    reshaped = q_values.reshape(n_agents, len(VALUE_ACTION_MODES), n_beams)
    actions: list[Action] = []
    indices = np.zeros(n_agents, dtype=np.int64)
    for node, observation in enumerate(observations):
        candidate = np.flatnonzero(np.asarray(observation["candidate_mask"], dtype=float) > 0.5)
        if candidate.size == 0:
            candidate = np.arange(n_beams, dtype=int)
        role_values = np.asarray(
            [float(np.max(reshaped[node, role, candidate])) for role in range(len(VALUE_ACTION_MODES))]
        )
        role = (
            int(rng.integers(len(VALUE_ACTION_MODES)))
            if rng.random() < role_uniform_mixture
            else int(np.argmax(role_values))
        )
        if rng.random() < beam_uniform_mixture:
            beam = int(rng.choice(candidate))
        else:
            beam = int(candidate[int(np.argmax(reshaped[node, role, candidate]))])
        actions.append(Action(VALUE_ACTION_MODES[role], beam))
        indices[node] = role * n_beams + beam
    return actions, indices


def select_beam_only_actions(
    q_values: np.ndarray,
    observations: Sequence[dict[str, Any]],
    role_rng: np.random.Generator,
    beam_gate_rng: np.random.Generator,
    beam_choice_rng: np.random.Generator,
    *,
    beam_uniform_mixture: float,
) -> tuple[list[Action], np.ndarray]:
    """Select only beams; TX/RX is an independent Bernoulli(0.5) protocol action."""

    if q_values.ndim != 2:
        raise ValueError("q_values must have shape [agents, beams].")
    n_agents, n_beams = q_values.shape
    if len(observations) != n_agents:
        raise ValueError("q_values and observation counts are inconsistent.")
    if not 0.0 <= float(beam_uniform_mixture) <= 1.0:
        raise ValueError("beam_uniform_mixture must be in [0, 1].")

    actions: list[Action] = []
    beam_indices = np.zeros(n_agents, dtype=np.int64)
    for node, observation in enumerate(observations):
        candidate = np.flatnonzero(np.asarray(observation["candidate_mask"], dtype=float) > 0.5)
        if candidate.size == 0:
            candidate = np.arange(n_beams, dtype=int)
        mode = MODE_TX if role_rng.random() < 0.5 else MODE_RX
        random_gate = beam_gate_rng.random()
        random_quantile = beam_choice_rng.random()
        if random_gate < beam_uniform_mixture:
            random_index = min(int(random_quantile * candidate.size), candidate.size - 1)
            beam = int(candidate[random_index])
        else:
            beam = int(candidate[int(np.argmax(q_values[node, candidate]))])
        actions.append(Action(mode, beam))
        beam_indices[node] = beam
    return actions, beam_indices


def select_beam_only_complementary_actions(
    q_values: np.ndarray,
    observations: Sequence[dict[str, Any]],
    beam_gate_rng: np.random.Generator,
    beam_choice_rng: np.random.Generator,
    *,
    beam_uniform_mixture: float,
) -> tuple[list[Action], np.ndarray]:
    """Diagnostic selector with alternating deterministic roles and learned beams.

    This contract removes TX/RX coordination from the two-node optimizer sanity
    check. It is not a deployable neighbor-discovery policy.
    """

    if q_values.ndim != 2:
        raise ValueError("q_values must have shape [agents, beams].")
    n_agents, n_beams = q_values.shape
    if len(observations) != n_agents:
        raise ValueError("q_values and observation counts are inconsistent.")
    if not 0.0 <= float(beam_uniform_mixture) <= 1.0:
        raise ValueError("beam_uniform_mixture must be in [0, 1].")

    actions: list[Action] = []
    beam_indices = np.zeros(n_agents, dtype=np.int64)
    for node, observation in enumerate(observations):
        candidate = np.flatnonzero(np.asarray(observation["candidate_mask"], dtype=float) > 0.5)
        if candidate.size == 0:
            candidate = np.arange(n_beams, dtype=int)
        random_gate = beam_gate_rng.random()
        random_quantile = beam_choice_rng.random()
        if random_gate < beam_uniform_mixture:
            random_index = min(int(random_quantile * candidate.size), candidate.size - 1)
            beam = int(candidate[random_index])
        else:
            beam = int(candidate[int(np.argmax(q_values[node, candidate]))])
        actions.append(Action(MODE_TX if node % 2 == 0 else MODE_RX, beam))
        beam_indices[node] = beam
    return actions, beam_indices


def select_candidate_score_actions(
    observations: Sequence[dict[str, Any]],
    role_rng: np.random.Generator,
    beam_choice_rng: np.random.Generator,
    *,
    selection: str,
    complementary_roles: bool = False,
) -> tuple[list[Action], np.ndarray]:
    """Non-learning beam controls driven only by the exposed candidate score."""

    if selection not in ("argmax", "proportional"):
        raise ValueError("selection must be 'argmax' or 'proportional'.")
    n_agents = len(observations)
    n_beams = len(observations[0]["candidate_mask"])
    actions: list[Action] = []
    beam_indices = np.zeros(n_agents, dtype=np.int64)
    for node, observation in enumerate(observations):
        candidate = np.flatnonzero(np.asarray(observation["candidate_mask"], dtype=float) > 0.5)
        if candidate.size == 0:
            candidate = np.arange(n_beams, dtype=int)
        scores = np.maximum(
            0.0,
            np.asarray(observation.get("candidate_score", np.zeros(n_beams)), dtype=float)[candidate],
        )
        random_quantile = beam_choice_rng.random()
        if complementary_roles:
            mode = MODE_TX if node % 2 == 0 else MODE_RX
        else:
            mode = MODE_TX if role_rng.random() < 0.5 else MODE_RX
        if selection == "argmax":
            beam = int(candidate[int(np.argmax(scores))])
        elif float(scores.sum()) <= 0.0:
            random_index = min(int(random_quantile * candidate.size), candidate.size - 1)
            beam = int(candidate[random_index])
        else:
            cumulative = np.cumsum(scores / scores.sum())
            beam = int(candidate[min(int(np.searchsorted(cumulative, random_quantile)), candidate.size - 1)])
        actions.append(Action(mode, beam))
        beam_indices[node] = beam
    return actions, beam_indices
