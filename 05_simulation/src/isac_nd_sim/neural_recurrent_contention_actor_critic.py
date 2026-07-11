from __future__ import annotations

from typing import Sequence

import numpy as np

from .marl_env import MODE_NAMES
from .neural_contention_actor_critic import (
    ContentionFeatures,
    ContentionGraphActorCritic,
    _temperature_scaled_logits,
    observations_to_batched_contention_tensors,
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
    ):
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover - depends on optional torch
            raise RuntimeError("PyTorch is required for RecurrentContentionGraphActorCritic.") from exc

        if str(action_contract) not in ("beam_only_fixed_role", "joint_role_beam"):
            raise ValueError("Unsupported recurrent action contract.")
        if use_rule_residual:
            raise ValueError("The recurrent fixed-role actor does not support rule residuals.")
        if use_contention_mode_prior:
            raise ValueError("The recurrent fixed-role actor does not support contention mode priors.")
        if use_rendezvous_adapter:
            raise ValueError("The recurrent fixed-role actor does not support rendezvous adapters.")
        if use_access_gate:
            raise ValueError("beam_only_fixed_role does not support a learned access gate.")
        if str(action_contract) == "beam_only_fixed_role" and float(role_probability_floor) != 0.0:
            raise ValueError("beam_only_fixed_role uses fixed Bernoulli(0.5) roles, not a role floor.")
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
        self.use_residual_measurement_features = bool(use_residual_measurement_features)
        self.role_probability_floor = float(role_probability_floor)
        self.beam_uniform_mixture = float(beam_uniform_mixture)
        self.use_access_gate = False
        self.access_gate_variant = str(access_gate_variant)
        self.supports_access_gate_action = False
        self.action_contract = str(action_contract)
        self.learned_mode_head_present = self.action_contract == "joint_role_beam"
        self.disabled_mode_indices = tuple(
            MODE_NAMES.index(mode) for mode in (disabled_modes or ()) if mode in MODE_NAMES
        )
        self.model = _RecurrentContentionGraphActorCriticModule(
            self.n_beams,
            self.hidden_dim,
            self.use_residual_measurement_features,
            self.azimuth_cells,
            self.elevation_cells,
            self.use_candidate_score_prior,
            self.candidate_score_prior_power,
            self.use_bounded_score_residual,
            self.score_residual_max_logit,
            self.learned_mode_head_present,
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
        return mode_logits.squeeze(0), beam_logits.squeeze(0), values.squeeze(0)

    def batched_logits_value(self, observations: Sequence[dict], hard_mask: bool = False):
        """Evaluate one independent batch step from zero state."""

        mode_logits, beam_logits, values, _next_state = self._step_from_state(
            observations, self._zero_state(len(observations)), hard_mask=hard_mask
        )
        return mode_logits, beam_logits, values

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
        return mode_logits, beam_logits, values

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
        sample_mode_logits = _temperature_scaled_logits(mode_logits, mode_temperature)
        beam_dist = Categorical(logits=sample_beam_logits)
        mode_dist = Categorical(logits=sample_mode_logits)
        if deterministic:
            beam_idx = torch.argmax(beam_logits, dim=-1)
        else:
            beam_idx = beam_dist.sample()
        if self.action_contract == "beam_only_fixed_role":
            mode_idx = torch.as_tensor(
                [
                    MODE_NAMES.index("tx") if role_rng.random() < 0.5 else MODE_NAMES.index("rx")
                    for _ in observations
                ],
                dtype=torch.long,
                device=self.device,
            )
        elif deterministic:
            mode_idx = torch.argmax(mode_logits, dim=-1)
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
        if self.action_contract == "beam_only_fixed_role":
            mode_log_prob = torch.zeros_like(beam_log_prob)
            mode_entropy = torch.zeros_like(beam_entropy)
        else:
            mode_log_prob = mode_dist.log_prob(mode_idx)
            mode_entropy = mode_dist.entropy()
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
        }
        for observations, actions in zip(observations_by_step, actions_by_step, strict=True):
            if len(observations) != n_agents or len(actions) != n_agents:
                raise ValueError("Every sequence step must contain the same number of agents and actions.")
            mode_logits, beam_logits, values, recurrent_state = self._step_from_state(
                observations,
                recurrent_state,
                hard_mask=True,
            )
            beam_dist = Categorical(logits=beam_logits)
            mode_dist = Categorical(logits=mode_logits)
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
            if self.action_contract == "beam_only_fixed_role":
                mode_log_prob = torch.zeros_like(beam_log_prob)
                mode_entropy = torch.zeros_like(beam_entropy)
            else:
                action_modes = torch.as_tensor(
                    [MODE_NAMES.index(action.mode) for action in actions],
                    dtype=torch.long,
                    device=self.device,
                )
                mode_log_prob = mode_dist.log_prob(action_modes)
                mode_entropy = mode_dist.entropy()
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
        return {key: torch.stack(value) for key, value in rows.items()}

    def _step_from_state(self, observations: Sequence[dict], recurrent_state, *, hard_mask: bool):
        if not observations:
            raise ValueError("A recurrent policy step requires at least one observation.")
        tensors = observations_to_batched_contention_tensors(
            observations,
            self.device,
            self.torch,
            self.n_beams,
            use_residual_measurement_features=self.use_residual_measurement_features,
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
        use_residual_measurement_features: bool,
        azimuth_cells: int,
        elevation_cells: int,
        use_candidate_score_prior: bool,
        candidate_score_prior_power: float,
        use_bounded_score_residual: bool,
        score_residual_max_logit: float,
        use_learned_mode_head: bool,
    ):
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional

        dims = ContentionFeatures()
        beam_dim = dims.residual_beam_dim if use_residual_measurement_features else dims.beam_dim

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
                self.value_head = nn.Linear(hidden_dim, 1)
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

            def forward(self, tensors: dict[str, torch.Tensor], recurrent_state: torch.Tensor):
                beam_tokens = functional.silu(self.beam_linear_norm(self.beam_linear(tensors["beam_features"])))
                beam_grid = beam_tokens.reshape(
                    beam_tokens.shape[0], self.elevation_cells, self.azimuth_cells, self.hidden_dim
                )
                beam_tokens = self.beam_convolution(beam_grid).reshape(
                    beam_tokens.shape[0], self.n_beams, self.hidden_dim
                )
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
                mode_logits = (
                    self.mode_head(next_state)
                    if self.mode_head is not None
                    else torch.zeros(
                        (next_state.shape[0], len(MODE_NAMES)),
                        dtype=next_state.dtype,
                        device=next_state.device,
                    )
                )
                return mode_logits, beam_logits, value, next_state

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
