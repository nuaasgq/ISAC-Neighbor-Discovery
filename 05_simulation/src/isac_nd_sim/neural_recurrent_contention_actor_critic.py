from __future__ import annotations

from typing import Sequence

import numpy as np

from .beam import beam_center_direction_features
from .marl_env import MODE_NAMES
from .neural_contention_actor_critic import (
    ContentionFeatures,
    ContentionGraphActorCritic,
    _temperature_scaled_logits,
    observations_to_batched_contention_tensors,
    resolve_measurement_feature_set,
)
from .neural_shared_actor_critic import PolicyStep
from .simulator import Action


class RecurrentContentionGraphActorCritic(ContentionGraphActorCritic):
    """Local recurrent beam policy for the fixed Bernoulli-role contract."""

    def __init__(
        self,
        n_beams: int,
        hidden_dim: int = 96,
        device: str = "cpu",
        use_candidate_mask: bool = False,
        use_candidate_score: bool = False,
        use_topology_deficit: bool = False,
        use_rule_residual: bool = False,
        rule_residual_scale: float = 1.0,
        use_contention_mode_prior: bool = False,
        use_rendezvous_adapter: bool = False,
        use_residual_measurement_features: bool = False,
        measurement_feature_set: str | None = None,
        use_measurement_prediction_head: bool = False,
        role_probability_floor: float = 0.0,
        beam_uniform_mixture: float = 0.0,
        use_access_gate: bool = False,
        access_gate_variant: str = "legacy",
        disabled_modes: Sequence[str] | None = None,
        action_contract: str = "beam_only_fixed_role",
        azimuth_cells: int | None = None,
        elevation_cells: int = 1,
        use_candidate_score_prior: bool = False,
        candidate_score_prior_power: float = 1.0,
        use_bounded_score_residual: bool = False,
        score_residual_max_logit: float = 2.0,
        use_decoupled_role_tower: bool = False,
        role_factorization: str = "independent",
    ):
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover - depends on optional torch
            raise RuntimeError("PyTorch is required for RecurrentContentionGraphActorCritic.") from exc

        if str(action_contract) not in (
            "beam_only_fixed_role",
            "beam_only_complementary_role",
            "joint_role_beam",
        ):
            raise ValueError("Unsupported recurrent action contract.")
        if use_rule_residual:
            raise ValueError("The recurrent fixed-role actor does not support rule residuals.")
        if use_contention_mode_prior:
            raise ValueError("The recurrent fixed-role actor does not support contention mode priors.")
        if use_rendezvous_adapter:
            raise ValueError("The recurrent fixed-role actor does not support rendezvous adapters.")
        if use_access_gate:
            raise ValueError("Beam-only action contracts do not support a learned access gate.")
        if str(action_contract) != "joint_role_beam" and float(role_probability_floor) != 0.0:
            raise ValueError("Beam-only action contracts use nonlearned roles, not a role floor.")
        if not 0.0 <= float(role_probability_floor) < 0.5:
            raise ValueError("role_probability_floor must be in [0, 0.5).")
        if not 0.0 <= float(beam_uniform_mixture) <= 1.0:
            raise ValueError("beam_uniform_mixture must be in [0, 1].")

        self.torch = torch
        self.nn = nn
        self.n_beams = int(n_beams)
        self.azimuth_cells = int(azimuth_cells or n_beams)
        self.elevation_cells = int(elevation_cells)
        if self.azimuth_cells <= 0 or self.elevation_cells <= 0:
            raise ValueError("Beam-grid dimensions must be positive.")
        if self.azimuth_cells * self.elevation_cells != self.n_beams:
            raise ValueError("azimuth_cells * elevation_cells must equal n_beams.")
        self.hidden_dim = int(hidden_dim)
        self.device = torch.device(device)
        self.use_candidate_mask = bool(use_candidate_mask)
        self.use_candidate_score = bool(use_candidate_score)
        self.use_candidate_score_prior = bool(use_candidate_score_prior)
        if self.use_candidate_score_prior and not self.use_candidate_score:
            raise ValueError("candidate-score prior requires use_candidate_score=True.")
        self.candidate_score_prior_power = float(candidate_score_prior_power)
        if self.candidate_score_prior_power <= 0.0:
            raise ValueError("candidate_score_prior_power must be positive.")
        self.use_bounded_score_residual = bool(use_bounded_score_residual)
        if self.use_bounded_score_residual and not self.use_candidate_score_prior:
            raise ValueError("bounded score residual requires the candidate-score prior.")
        self.score_residual_max_logit = float(score_residual_max_logit)
        if self.score_residual_max_logit <= 0.0:
            raise ValueError("score_residual_max_logit must be positive.")
        self.use_topology_deficit = bool(use_topology_deficit)
        self.use_rule_residual = False
        self.rule_residual_scale = float(rule_residual_scale)
        self.use_contention_mode_prior = False
        self.use_rendezvous_adapter = False
        self.measurement_feature_set = resolve_measurement_feature_set(
            measurement_feature_set,
            bool(use_residual_measurement_features),
        )
        self.use_residual_measurement_features = self.measurement_feature_set == "residual"
        self.use_measurement_prediction_head = bool(use_measurement_prediction_head)
        if self.use_measurement_prediction_head and self.measurement_feature_set == "none":
            raise ValueError("The measurement prediction head requires direct or residual measurements.")
        self.role_probability_floor = float(role_probability_floor)
        self.beam_uniform_mixture = float(beam_uniform_mixture)
        self.use_access_gate = False
        self.access_gate_variant = str(access_gate_variant)
        self.supports_access_gate_action = False
        self.action_contract = str(action_contract)
        self.role_factorization = str(role_factorization)
        if self.role_factorization not in {
            "independent",
            "beam_conditioned",
            "beam_conditioned_antisymmetric",
        }:
            raise ValueError("Unsupported role_factorization.")
        if self.role_factorization != "independent" and self.action_contract != "joint_role_beam":
            raise ValueError("Beam-conditioned roles require joint_role_beam.")
        self.learned_mode_head_present = self.action_contract == "joint_role_beam"
        self.use_decoupled_role_tower = bool(use_decoupled_role_tower)
        if self.use_decoupled_role_tower and not self.learned_mode_head_present:
            raise ValueError("decoupled role tower requires joint_role_beam.")
        if self.role_factorization != "independent" and not self.use_decoupled_role_tower:
            raise ValueError("Beam-conditioned roles require the decoupled role tower.")
        self.disabled_mode_indices = tuple(
            MODE_NAMES.index(mode) for mode in (disabled_modes or ()) if mode in MODE_NAMES
        )
        self.model = _RecurrentContentionGraphActorCriticModule(
            self.n_beams,
            self.hidden_dim,
            self.measurement_feature_set,
            self.use_measurement_prediction_head,
            self.azimuth_cells,
            self.elevation_cells,
            self.use_candidate_score_prior,
            self.candidate_score_prior_power,
            self.use_bounded_score_residual,
            self.score_residual_max_logit,
            self.learned_mode_head_present,
            self.use_decoupled_role_tower,
            self.role_factorization,
        ).to(self.device)
        self._recurrent_state = None

    @property
    def recurrent_state(self):
        return self._recurrent_state

    def clone_recurrent_state(self):
        if self._recurrent_state is None:
            return None
        return self._recurrent_state.detach().clone()

    def restore_recurrent_state(self, state) -> None:
        if state is None:
            self._recurrent_state = None
            return
        if state.ndim != 2 or state.shape[1] != self.hidden_dim:
            raise ValueError("Recurrent state must have shape [n_agents, hidden_dim].")
        self._recurrent_state = state.detach().clone().to(self.device)

    def reset_recurrent_state(self, n_agents: int) -> None:
        if int(n_agents) < 0:
            raise ValueError("n_agents must be non-negative.")
        self._recurrent_state = self._zero_state(int(n_agents))

    def logits_value(self, observation: dict, hard_mask: bool = False):
        """Evaluate one observation from zero state without changing online state."""

        mode_logits, beam_logits, values, _next_state = self._step_from_state(
            [observation], self._zero_state(1), hard_mask=hard_mask
        )
        mode_logits = self._marginal_mode_logits(mode_logits, beam_logits)
        return mode_logits.squeeze(0), beam_logits.squeeze(0), values.squeeze(0)

    def batched_logits_value(self, observations: Sequence[dict], hard_mask: bool = False):
        """Evaluate one independent batch step from zero state."""

        mode_logits, beam_logits, values, _next_state = self._step_from_state(
            observations, self._zero_state(len(observations)), hard_mask=hard_mask
        )
        return self._marginal_mode_logits(mode_logits, beam_logits), beam_logits, values

    def advance_recurrent_logits(self, observations: Sequence[dict], hard_mask: bool = True):
        """Advance online recurrent state and return logits for external evaluation."""

        if self._recurrent_state is None:
            self.reset_recurrent_state(len(observations))
        if self._recurrent_state.shape[0] != len(observations):
            raise ValueError("Agent count changed; call reset_recurrent_state before the next episode.")
        mode_logits, beam_logits, values, next_state = self._step_from_state(
            observations,
            self._recurrent_state,
            hard_mask=hard_mask,
        )
        self._recurrent_state = next_state.detach()
        return self._marginal_mode_logits(mode_logits, beam_logits), beam_logits, values

    def act(
        self,
        observations: Sequence[dict],
        deterministic: bool = False,
        mode_temperature: float = 1.0,
        beam_temperature: float = 1.0,
        gate_temperature: float = 1.0,
        role_rng: np.random.Generator | None = None,
    ) -> PolicyStep:
        del gate_temperature
        torch = self.torch
        from torch.distributions import Categorical

        if not observations:
            empty = torch.empty(0, device=self.device)
            return PolicyStep(actions=[], log_probs=empty, values=empty, entropies=empty)
        if self.action_contract == "beam_only_fixed_role" and role_rng is None:
            raise ValueError("beam_only_fixed_role requires an explicit role_rng.")
        if self._recurrent_state is None:
            self.reset_recurrent_state(len(observations))
        if self._recurrent_state.shape[0] != len(observations):
            raise ValueError("Agent count changed; call reset_recurrent_state before the next episode.")

        mode_logits, beam_logits, value, next_state = self._step_from_state(
            observations,
            self._recurrent_state,
            hard_mask=True,
        )
        self._recurrent_state = next_state.detach()
        sample_beam_logits = _temperature_scaled_logits(beam_logits, beam_temperature)
        beam_dist = self._beam_distribution(sample_beam_logits)
        if deterministic:
            beam_idx = torch.argmax(beam_logits, dim=-1)
        else:
            beam_idx = beam_dist.sample()
        selected_mode_logits = self._selected_mode_logits(mode_logits, beam_idx)
        sample_mode_logits = _temperature_scaled_logits(selected_mode_logits, mode_temperature)
        mode_dist = Categorical(logits=sample_mode_logits)
        if self.action_contract == "beam_only_fixed_role":
            mode_idx = torch.as_tensor(
                [
                    MODE_NAMES.index("tx") if role_rng.random() < 0.5 else MODE_NAMES.index("rx")
                    for _ in observations
                ],
                dtype=torch.long,
                device=self.device,
            )
        elif self.action_contract == "beam_only_complementary_role":
            mode_idx = torch.as_tensor(
                [
                    MODE_NAMES.index("tx") if node % 2 == 0 else MODE_NAMES.index("rx")
                    for node in range(len(observations))
                ],
                dtype=torch.long,
                device=self.device,
            )
        elif deterministic:
            mode_idx = torch.argmax(selected_mode_logits, dim=-1)
        else:
            mode_idx = mode_dist.sample()
        active_beam_mask = mode_idx != MODE_NAMES.index("idle")
        beam_idx = torch.where(active_beam_mask, beam_idx, torch.zeros_like(beam_idx))
        actions = [
            Action(MODE_NAMES[int(mode.item())], int(beam.item()))
            for mode, beam in zip(mode_idx, beam_idx, strict=True)
        ]
        raw_beam_log_prob = beam_dist.log_prob(beam_idx)
        beam_log_prob = torch.where(
            active_beam_mask,
            raw_beam_log_prob,
            torch.zeros_like(raw_beam_log_prob),
        )
        beam_entropy = beam_dist.entropy()
        if self.action_contract != "joint_role_beam":
            mode_log_prob = torch.zeros_like(beam_log_prob)
            mode_entropy = torch.zeros_like(beam_entropy)
        else:
            mode_log_prob = mode_dist.log_prob(mode_idx)
            mode_entropy = self._conditional_mode_entropy(mode_logits, beam_logits)
        return PolicyStep(
            actions=actions,
            log_probs=mode_log_prob + beam_log_prob,
            values=value.squeeze(-1),
            entropies=mode_entropy + beam_entropy,
            mode_log_probs=mode_log_prob,
            beam_log_probs=beam_log_prob,
            gate_log_probs=torch.zeros_like(beam_log_prob),
            mode_entropies=mode_entropy,
            beam_entropies=beam_entropy,
            gate_entropies=torch.zeros_like(beam_entropy),
            active_beam_mask=active_beam_mask,
        )

    def evaluate_action_sequence(
        self,
        observations_by_step: Sequence[Sequence[dict]],
        actions_by_step: Sequence[Sequence[Action]],
    ) -> dict[str, object]:
        """Recompute a trajectory from zero state while retaining the BPTT graph."""

        torch = self.torch
        from torch.distributions import Categorical

        if len(observations_by_step) != len(actions_by_step):
            raise ValueError("Observation and action sequences must have the same length.")
        if not observations_by_step:
            empty = torch.empty((0, 0), device=self.device)
            return {
                "log_probs": empty,
                "mode_log_probs": empty,
                "beam_log_probs": empty,
                "gate_log_probs": empty,
                "values": empty,
                "entropies": empty,
                "mode_entropies": empty,
                "beam_entropies": empty,
                "gate_entropies": empty,
                "active_beam_mask": empty.bool(),
                "mode_tx_probabilities": empty,
            }

        n_agents = len(observations_by_step[0])
        recurrent_state = self._zero_state(n_agents)
        rows = {
            "log_probs": [],
            "mode_log_probs": [],
            "beam_log_probs": [],
            "gate_log_probs": [],
            "values": [],
            "entropies": [],
            "mode_entropies": [],
            "beam_entropies": [],
            "gate_entropies": [],
            "active_beam_mask": [],
            "mode_tx_probabilities": [],
        }
        for observations, actions in zip(observations_by_step, actions_by_step, strict=True):
            if len(observations) != n_agents or len(actions) != n_agents:
                raise ValueError("Every sequence step must contain the same number of agents and actions.")
            mode_logits, beam_logits, values, recurrent_state = self._step_from_state(
                observations,
                recurrent_state,
                hard_mask=True,
            )
            beam_dist = self._beam_distribution(beam_logits)
            action_beams = torch.as_tensor(
                [int(action.beam) for action in actions],
                dtype=torch.long,
                device=self.device,
            )
            active_beam_mask = torch.as_tensor(
                [action.mode != "idle" for action in actions],
                dtype=torch.bool,
                device=self.device,
            )
            raw_beam_log_prob = beam_dist.log_prob(action_beams)
            beam_log_prob = torch.where(active_beam_mask, raw_beam_log_prob, torch.zeros_like(raw_beam_log_prob))
            beam_entropy = beam_dist.entropy()
            if self.action_contract != "joint_role_beam":
                if self.action_contract == "beam_only_complementary_role":
                    expected_modes = ["tx" if node % 2 == 0 else "rx" for node in range(n_agents)]
                    actual_modes = [action.mode for action in actions]
                    if actual_modes != expected_modes:
                        raise ValueError("Stored actions violate the complementary-role diagnostic contract.")
                mode_log_prob = torch.zeros_like(beam_log_prob)
                mode_entropy = torch.zeros_like(beam_entropy)
            else:
                selected_mode_logits = self._selected_mode_logits(mode_logits, action_beams)
                mode_dist = Categorical(logits=selected_mode_logits)
                action_modes = torch.as_tensor(
                    [MODE_NAMES.index(action.mode) for action in actions],
                    dtype=torch.long,
                    device=self.device,
                )
                mode_log_prob = mode_dist.log_prob(action_modes)
                mode_entropy = self._conditional_mode_entropy(mode_logits, beam_logits)
            rows["log_probs"].append(mode_log_prob + beam_log_prob)
            rows["mode_log_probs"].append(mode_log_prob)
            rows["beam_log_probs"].append(beam_log_prob)
            rows["gate_log_probs"].append(torch.zeros_like(beam_log_prob))
            rows["values"].append(values.squeeze(-1))
            rows["entropies"].append(mode_entropy + beam_entropy)
            rows["mode_entropies"].append(mode_entropy)
            rows["beam_entropies"].append(beam_entropy)
            rows["gate_entropies"].append(torch.zeros_like(beam_entropy))
            rows["active_beam_mask"].append(active_beam_mask)
            marginal_mode_logits = self._marginal_mode_logits(mode_logits, beam_logits)
            rows["mode_tx_probabilities"].append(
                torch.softmax(marginal_mode_logits, dim=-1)[:, MODE_NAMES.index("tx")]
            )
        return {key: torch.stack(value) for key, value in rows.items()}

    def measurement_occupancy_logits(self, observations: Sequence[dict]):
        if not self.use_measurement_prediction_head:
            raise RuntimeError("The measurement prediction head is disabled.")
        tensors = observations_to_batched_contention_tensors(
            observations,
            self.device,
            self.torch,
            self.n_beams,
            measurement_feature_set=self.measurement_feature_set,
        )
        tensors = self._prepare_tensors(tensors)
        beam_tokens, _beam_inputs = self.model.encode_beam_tokens(tensors)
        return self.model.measurement_occupancy_head(beam_tokens).squeeze(-1)

    def _selected_mode_logits(self, mode_logits, beam_indices):
        if mode_logits.ndim == 2:
            return mode_logits
        if mode_logits.ndim != 3:
            raise ValueError("Mode logits must have shape [agents,modes] or [agents,beams,modes].")
        rows = self.torch.arange(mode_logits.shape[0], device=mode_logits.device)
        return mode_logits[rows, beam_indices]

    def _marginal_mode_logits(self, mode_logits, beam_logits):
        if mode_logits.ndim == 2:
            return mode_logits
        joint_log_probability = self.torch.log(
            self._beam_probabilities(beam_logits).clamp_min(1.0e-12)
        ).unsqueeze(-1) + self.torch.log_softmax(mode_logits, dim=-1)
        return self.torch.logsumexp(joint_log_probability, dim=1)

    def _conditional_mode_entropy(self, mode_logits, beam_logits):
        from torch.distributions import Categorical

        if mode_logits.ndim == 2:
            return Categorical(logits=mode_logits).entropy()
        per_beam_entropy = Categorical(logits=mode_logits).entropy()
        beam_probability = self._beam_probabilities(beam_logits)
        return (beam_probability * per_beam_entropy).sum(dim=-1)

    def _beam_probabilities(self, beam_logits):
        probabilities = self.torch.softmax(beam_logits, dim=-1)
        if self.beam_uniform_mixture <= 0.0:
            return probabilities
        valid = beam_logits > -1.0e8
        has_valid = valid.any(dim=-1, keepdim=True)
        valid = self.torch.where(has_valid, valid, self.torch.ones_like(valid))
        uniform = valid.to(probabilities.dtype) / valid.sum(dim=-1, keepdim=True).clamp_min(1)
        return (
            (1.0 - self.beam_uniform_mixture) * probabilities
            + self.beam_uniform_mixture * uniform
        )

    def _beam_distribution(self, beam_logits):
        from torch.distributions import Categorical

        return Categorical(probs=self._beam_probabilities(beam_logits))

    def _step_from_state(self, observations: Sequence[dict], recurrent_state, *, hard_mask: bool):
        if not observations:
            raise ValueError("A recurrent policy step requires at least one observation.")
        tensors = observations_to_batched_contention_tensors(
            observations,
            self.device,
            self.torch,
            self.n_beams,
            measurement_feature_set=self.measurement_feature_set,
        )
        tensors = self._prepare_tensors(tensors)
        mode_logits, beam_logits, value, next_state = self.model(tensors, recurrent_state)
        if hard_mask and self.use_candidate_mask and "candidate_mask" in tensors:
            mask = tensors["candidate_mask"] > 0.5
            has_candidate = mask.any(dim=-1, keepdim=True)
            beam_logits = self.torch.where(
                has_candidate,
                beam_logits.masked_fill(~mask, -1.0e9),
                beam_logits,
            )
        mode_logits = self._mask_disabled_modes(mode_logits)
        mode_logits, beam_logits = self._regularize_stochastic_support(mode_logits, beam_logits, tensors)
        return mode_logits, beam_logits, value, next_state

    def _zero_state(self, n_agents: int):
        parameter = next(self.model.parameters())
        return self.torch.zeros(
            (int(n_agents), self.hidden_dim),
            dtype=parameter.dtype,
            device=self.device,
        )


class _RecurrentContentionGraphActorCriticModule:
    def __new__(
        cls,
        n_beams: int,
        hidden_dim: int,
        measurement_feature_set: str,
        use_measurement_prediction_head: bool,
        azimuth_cells: int,
        elevation_cells: int,
        use_candidate_score_prior: bool,
        candidate_score_prior_power: float,
        use_bounded_score_residual: bool,
        score_residual_max_logit: float,
        use_learned_mode_head: bool,
        use_decoupled_role_tower: bool,
        role_factorization: str,
    ):
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional

        dims = ContentionFeatures()
        measurement_dims = {
            "none": dims.beam_dim,
            "direct": dims.direct_measurement_beam_dim,
            "residual": dims.residual_beam_dim,
        }
        beam_dim = measurement_dims[measurement_feature_set] + 3

        class BeamGridResidualBlock(nn.Module):
            def __init__(self, channels: int):
                super().__init__()
                self.azimuth_conv = nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode="circular",
                )
                self.elevation_conv = nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                )
                self.norm = nn.LayerNorm(channels)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                batch, elevations, azimuths, channels = inputs.shape
                azimuth_rows = inputs.reshape(batch * elevations, azimuths, channels).transpose(1, 2)
                azimuth_rows = functional.silu(self.azimuth_conv(azimuth_rows)).transpose(1, 2)
                azimuth_rows = azimuth_rows.reshape(batch, elevations, azimuths, channels)
                elevation_columns = azimuth_rows.permute(0, 2, 3, 1).reshape(
                    batch * azimuths, channels, elevations
                )
                elevation_columns = self.elevation_conv(elevation_columns)
                outputs = elevation_columns.reshape(batch, azimuths, channels, elevations).permute(0, 3, 1, 2)
                return functional.silu(self.norm(outputs + inputs))

        class Module(nn.Module):
            def __init__(self, n_beams: int, hidden_dim: int):
                super().__init__()
                self.n_beams = int(n_beams)
                self.hidden_dim = int(hidden_dim)
                self.beam_linear = nn.Linear(beam_dim, hidden_dim)
                self.beam_linear_norm = nn.LayerNorm(hidden_dim)
                self.azimuth_cells = int(azimuth_cells)
                self.elevation_cells = int(elevation_cells)
                self.register_buffer(
                    "beam_center_directions",
                    torch.as_tensor(
                        beam_center_direction_features(self.azimuth_cells, self.elevation_cells),
                        dtype=torch.float32,
                    ),
                    persistent=False,
                )
                self.beam_convolution = BeamGridResidualBlock(hidden_dim)
                self.contention_encoder = nn.Sequential(
                    nn.Linear(dims.contention_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
                context_dim = 9 + 4 + 4 + 1 + 1 + dims.candidate_stats_dim + 2 * hidden_dim
                self.context_encoder = nn.Sequential(
                    nn.Linear(context_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
                self.recurrent_cell = nn.GRUCell(hidden_dim, hidden_dim)
                self.beam_query = nn.Linear(hidden_dim, hidden_dim)
                self.beam_bias = nn.Linear(hidden_dim, 1)
                self.measurement_occupancy_head = (
                    nn.Linear(hidden_dim, 1) if use_measurement_prediction_head else None
                )
                self.value_head = nn.Linear(hidden_dim, 1)
                self.role_encoder = None
                self.role_beam_encoder = None
                self.role_direction_axis = None
                self.role_direction_raw_scale = None
                if use_learned_mode_head and use_decoupled_role_tower:
                    role_input_dim = dims.contention_dim + 4 + 4 + 1 + dims.candidate_stats_dim
                    self.role_encoder = nn.Sequential(
                        nn.Linear(role_input_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.SiLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.SiLU(),
                    )
                    if role_factorization == "beam_conditioned":
                        self.role_beam_encoder = nn.Sequential(
                            nn.Linear(beam_dim, hidden_dim),
                            nn.LayerNorm(hidden_dim),
                            nn.SiLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.SiLU(),
                        )
                    elif role_factorization == "beam_conditioned_antisymmetric":
                        self.role_direction_axis = nn.Parameter(torch.empty(3))
                        nn.init.normal_(self.role_direction_axis, mean=0.0, std=1.0)
                        direction_mask = [1.0, 1.0, 0.0] if self.elevation_cells == 1 else [1.0, 1.0, 1.0]
                        self.register_buffer(
                            "role_direction_mask",
                            torch.tensor(direction_mask, dtype=torch.float32),
                            persistent=False,
                        )
                        self.role_direction_raw_scale = nn.Parameter(
                            torch.tensor(float(np.log(np.expm1(2.0))), dtype=torch.float32)
                        )
                self.mode_head = nn.Linear(hidden_dim, len(MODE_NAMES)) if use_learned_mode_head else None
                if self.mode_head is not None:
                    nn.init.zeros_(self.mode_head.weight)
                    nn.init.zeros_(self.mode_head.bias)
                self.use_candidate_score_prior = bool(use_candidate_score_prior)
                self.use_bounded_score_residual = bool(use_bounded_score_residual)
                self.score_residual_max_logit = float(score_residual_max_logit)
                if self.use_candidate_score_prior:
                    self.candidate_prior_raw_strength = nn.Parameter(
                        torch.tensor(
                            float(np.log(np.expm1(float(candidate_score_prior_power)))),
                            dtype=torch.float32,
                        )
                    )
                    nn.init.zeros_(self.beam_query.weight)
                    nn.init.zeros_(self.beam_query.bias)
                    nn.init.zeros_(self.beam_bias.weight)
                    nn.init.zeros_(self.beam_bias.bias)
                if self.use_bounded_score_residual:
                    initial_fraction = 0.1 / self.score_residual_max_logit
                    initial_fraction = float(np.clip(initial_fraction, 1.0e-4, 1.0 - 1.0e-4))
                    self.score_residual_raw_gate = nn.Parameter(
                        torch.tensor(float(np.log(initial_fraction / (1.0 - initial_fraction))))
                    )

            def encode_beam_tokens(self, tensors: dict[str, torch.Tensor]):
                directions = self.beam_center_directions.unsqueeze(0).expand(
                    tensors["beam_features"].shape[0], -1, -1
                )
                beam_inputs = torch.cat([tensors["beam_features"], directions], dim=-1)
                beam_tokens = functional.silu(self.beam_linear_norm(self.beam_linear(beam_inputs)))
                beam_grid = beam_tokens.reshape(
                    beam_tokens.shape[0], self.elevation_cells, self.azimuth_cells, self.hidden_dim
                )
                beam_tokens = self.beam_convolution(beam_grid).reshape(
                    beam_tokens.shape[0], self.n_beams, self.hidden_dim
                )
                return beam_tokens, beam_inputs

            def forward(self, tensors: dict[str, torch.Tensor], recurrent_state: torch.Tensor):
                beam_tokens, beam_inputs = self.encode_beam_tokens(tensors)
                beam_context = beam_tokens.mean(dim=1)
                contention = self.contention_encoder(tensors["contention_state"])
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
                    dim=-1,
                )
                current_context = self.context_encoder(context_input)
                next_state = self.recurrent_cell(current_context, recurrent_state)
                query = self.beam_query(next_state)
                residual_logits = (beam_tokens * query.unsqueeze(1)).sum(dim=-1) / np.sqrt(
                    float(self.hidden_dim)
                )
                residual_logits = residual_logits + self.beam_bias(beam_tokens).squeeze(-1)
                if self.use_bounded_score_residual:
                    residual_scale = self.score_residual_max_logit * torch.sigmoid(
                        self.score_residual_raw_gate
                    )
                    residual_logits = residual_scale * torch.tanh(residual_logits)
                beam_logits = residual_logits
                if self.use_candidate_score_prior:
                    beam_logits = beam_logits + self._candidate_prior_logits(tensors)
                value = self.value_head(next_state)
                if self.mode_head is not None:
                    if self.role_encoder is not None:
                        role_input = torch.cat(
                            [
                                tensors["contention_state"],
                                tensors["local_summary"],
                                tensors["last_mode"],
                                tensors["topology_deficit"],
                                tensors["candidate_stats"],
                            ],
                            dim=-1,
                        )
                        role_hidden = self.role_encoder(role_input)
                    else:
                        role_hidden = next_state
                    if self.role_direction_axis is not None:
                        global_directions = self._global_beam_directions(tensors["self_state"])
                        masked_axis = self.role_direction_axis * self.role_direction_mask
                        direction_axis = masked_axis / masked_axis.norm().clamp_min(1.0e-6)
                        role_scale = functional.softplus(self.role_direction_raw_scale)
                        role_score = role_scale * (global_directions @ direction_axis)
                        mode_logits = torch.zeros(
                            (*role_score.shape, len(MODE_NAMES)),
                            dtype=role_score.dtype,
                            device=role_score.device,
                        )
                        mode_logits[..., MODE_NAMES.index("tx")] = role_score
                        mode_logits[..., MODE_NAMES.index("rx")] = -role_score
                    elif self.role_beam_encoder is not None:
                        role_beam_hidden = self.role_beam_encoder(beam_inputs)
                        mode_logits = self.mode_head(
                            functional.silu(role_hidden.unsqueeze(1) + role_beam_hidden)
                        )
                    else:
                        mode_logits = self.mode_head(role_hidden)
                else:
                    mode_logits = torch.zeros(
                        (next_state.shape[0], len(MODE_NAMES)),
                        dtype=next_state.dtype,
                        device=next_state.device,
                    )
                return mode_logits, beam_logits, value, next_state

            def _global_beam_directions(self, self_state: torch.Tensor) -> torch.Tensor:
                body = self.beam_center_directions.unsqueeze(0).expand(self_state.shape[0], -1, -1)
                yaw = self_state[:, 6] * np.pi
                pitch = self_state[:, 7] * (np.pi / 2.0)
                roll = self_state[:, 8] * np.pi
                cy, sy = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
                cp, sp = torch.cos(pitch)[:, None], torch.sin(pitch)[:, None]
                cr, sr = torch.cos(roll)[:, None], torch.sin(roll)[:, None]
                x1 = body[..., 0]
                y1 = cr * body[..., 1] - sr * body[..., 2]
                z1 = sr * body[..., 1] + cr * body[..., 2]
                x2 = cp * x1 + sp * z1
                y2 = y1
                z2 = -sp * x1 + cp * z1
                return torch.stack(
                    [cy * x2 - sy * y2, sy * x2 + cy * y2, z2],
                    dim=-1,
                )

            def _candidate_prior_logits(self, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
                scores = torch.clamp(tensors["candidate_score"], min=0.0)
                candidate_mask = tensors.get("candidate_mask", torch.ones_like(scores)) > 0.5
                has_candidate = candidate_mask.any(dim=-1, keepdim=True)
                candidate_mask = torch.where(has_candidate, candidate_mask, torch.ones_like(candidate_mask))
                scores = scores * candidate_mask.to(scores.dtype)
                score_sum = scores.sum(dim=-1, keepdim=True)
                uniform = candidate_mask.to(scores.dtype) / candidate_mask.sum(dim=-1, keepdim=True).clamp_min(1)
                probabilities = torch.where(score_sum > 0.0, scores / score_sum.clamp_min(1.0e-12), uniform)
                strength = functional.softplus(self.candidate_prior_raw_strength)
                return strength * torch.log(probabilities.clamp_min(1.0e-12))

        return Module(n_beams, hidden_dim)
