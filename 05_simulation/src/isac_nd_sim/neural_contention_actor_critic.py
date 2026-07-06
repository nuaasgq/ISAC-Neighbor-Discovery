from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .marl_env import MODE_NAMES
from .neural_shared_actor_critic import PolicyStep
from .simulator import Action


@dataclass(frozen=True)
class ContentionFeatures:
    contention_dim: int = 10
    beam_dim: int = 8
    candidate_stats_dim: int = 4


class ContentionGraphActorCritic:
    """Contention/topology-aware shared actor for decentralized execution.

    This network keeps the parameter sharing and variable-codebook transfer
    properties of the shared actor, but exposes a separate local contention
    pathway. Beam tokens include local collision pressure, while mode logits
    receive a learned residual from local contention/topology state.
    """

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
        use_access_gate: bool = False,
        access_gate_variant: str = "legacy",
    ):
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover - depends on optional torch
            raise RuntimeError("PyTorch is required for ContentionGraphActorCritic.") from exc

        self.torch = torch
        self.nn = nn
        self.n_beams = int(n_beams)
        self.hidden_dim = int(hidden_dim)
        self.device = torch.device(device)
        self.use_candidate_mask = bool(use_candidate_mask)
        self.use_candidate_score = bool(use_candidate_score)
        self.use_topology_deficit = bool(use_topology_deficit)
        self.use_rule_residual = bool(use_rule_residual)
        self.rule_residual_scale = float(rule_residual_scale)
        self.use_access_gate = bool(use_access_gate)
        self.access_gate_variant = str(access_gate_variant)
        self.model = _ContentionGraphActorCriticModule(
            self.n_beams,
            self.hidden_dim,
            self.use_access_gate,
            self.access_gate_variant,
        ).to(self.device)

    def parameters(self):
        return self.model.parameters()

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def logits_value(self, observation: dict, hard_mask: bool = False):
        torch = self.torch
        tensors = observation_to_contention_tensors(observation, self.device, torch, self.n_beams)
        tensors = self._prepare_tensors(tensors)
        mode_logits, beam_logits, value = self.model(tensors)
        mode_logits, beam_logits = self._apply_residuals(tensors, mode_logits, beam_logits)
        if hard_mask and self.use_candidate_mask and "candidate_mask" in tensors:
            mask = tensors["candidate_mask"] > 0.5
            if bool(mask.any().item()):
                beam_logits = beam_logits.masked_fill(~mask, -1.0e9)
        return mode_logits, beam_logits, value

    def batched_logits_value(self, observations: Sequence[dict], hard_mask: bool = False):
        torch = self.torch
        tensors = observations_to_batched_contention_tensors(observations, self.device, torch, self.n_beams)
        tensors = self._prepare_tensors(tensors)
        mode_logits, beam_logits, value = self.model(tensors)
        mode_logits, beam_logits = self._apply_residuals(tensors, mode_logits, beam_logits)
        if hard_mask and self.use_candidate_mask and "candidate_mask" in tensors:
            mask = tensors["candidate_mask"] > 0.5
            has_candidate = mask.any(dim=-1, keepdim=True)
            beam_logits = torch.where(
                has_candidate,
                beam_logits.masked_fill(~mask, -1.0e9),
                beam_logits,
            )
        return mode_logits, beam_logits, value

    def _prepare_tensors(self, tensors: dict):
        torch = self.torch
        if not self.use_candidate_score and "candidate_score" in tensors:
            tensors = dict(tensors)
            tensors["candidate_score"] = torch.zeros_like(tensors["candidate_score"])
            tensors["beam_features"] = tensors["beam_features"].clone()
            tensors["beam_features"][..., 4] = 0.0
        if not self.use_topology_deficit and "topology_deficit" in tensors:
            tensors = dict(tensors)
            tensors["topology_deficit"] = torch.zeros_like(tensors["topology_deficit"])
        return tensors

    def _apply_residuals(self, tensors: dict, mode_logits, beam_logits):
        mode_logits = mode_logits + self._contention_mode_prior(tensors)
        if self.use_rule_residual:
            if "rule_mode_logits" in tensors:
                mode_logits = mode_logits + self.rule_residual_scale * tensors["rule_mode_logits"]
            if "candidate_score" in tensors:
                # ISAC candidate confidence attracts access, while local
                # collision pressure suppresses repeatedly colliding beams.
                beam_logits = beam_logits + self.rule_residual_scale * (
                    tensors["candidate_score"] - 0.85 * tensors["beam_collision_norm"]
                )
        return mode_logits, beam_logits

    def _contention_mode_prior(self, tensors: dict):
        torch = self.torch
        state = tensors["contention_state"]
        topology_need = state[..., 1]
        fail_pressure = state[..., 3]
        collision_pressure = state[..., 4]
        candidate_fraction = state[..., 6]
        last_role_delta = state[..., 9]
        prior = torch.zeros_like(tensors["rule_mode_logits"]) if "rule_mode_logits" in tensors else torch.zeros(
            (*state.shape[:-1], len(MODE_NAMES)),
            dtype=state.dtype,
            device=state.device,
        )
        sparse_candidates = torch.clamp(1.0 - 8.0 * candidate_fraction, min=0.0, max=1.0)
        contention = torch.clamp(0.65 * collision_pressure + 0.35 * fail_pressure, min=0.0, max=1.0)
        prior[..., MODE_NAMES.index("sense")] = 0.20 * topology_need + 0.35 * sparse_candidates + 0.35 * contention
        prior[..., MODE_NAMES.index("tx")] = 0.45 * topology_need - 0.95 * contention - 0.25 * torch.clamp(last_role_delta, min=0.0)
        prior[..., MODE_NAMES.index("rx")] = 0.45 * topology_need - 0.70 * contention + 0.15 * torch.clamp(last_role_delta, min=0.0)
        prior[..., MODE_NAMES.index("idle")] = 0.60 * contention - 0.35 * topology_need
        return prior

    def act(self, observations: Sequence[dict], deterministic: bool = False) -> PolicyStep:
        torch = self.torch
        from torch.distributions import Categorical

        if not observations:
            empty = torch.empty(0, device=self.device)
            return PolicyStep(actions=[], log_probs=empty, values=empty, entropies=empty)

        mode_logits, beam_logits, value = self.batched_logits_value(observations, hard_mask=True)
        mode_dist = Categorical(logits=mode_logits)
        beam_dist = Categorical(logits=beam_logits)
        if deterministic:
            mode_idx = torch.argmax(mode_logits, dim=-1)
            sampled_beam_idx = torch.argmax(beam_logits, dim=-1)
        else:
            mode_idx = mode_dist.sample()
            sampled_beam_idx = beam_dist.sample()
        idle_idx = MODE_NAMES.index("idle")
        beam_idx = torch.where(
            mode_idx == idle_idx,
            torch.zeros_like(sampled_beam_idx),
            sampled_beam_idx,
        )
        actions = [
            Action(MODE_NAMES[int(mode_item.item())], int(beam_item.item()))
            for mode_item, beam_item in zip(mode_idx, beam_idx, strict=True)
        ]
        beam_log_prob = torch.where(
            mode_idx == idle_idx,
            torch.zeros_like(value.squeeze(-1)),
            beam_dist.log_prob(sampled_beam_idx),
        )
        return PolicyStep(
            actions=actions,
            log_probs=mode_dist.log_prob(mode_idx) + beam_log_prob,
            values=value.squeeze(-1),
            entropies=mode_dist.entropy() + beam_dist.entropy(),
        )


class _ContentionGraphActorCriticModule:
    def __new__(
        cls,
        n_beams: int,
        hidden_dim: int,
        use_access_gate: bool = False,
        access_gate_variant: str = "legacy",
    ):
        import torch
        import torch.nn as nn

        dims = ContentionFeatures()
        gate_variant = str(access_gate_variant)
        if gate_variant not in {"legacy", "adaptive", "topology_preserving", "balanced_topology"}:
            raise ValueError(
                "access_gate_variant must be 'legacy', 'adaptive', 'topology_preserving', or 'balanced_topology'."
            )

        class Module(nn.Module):
            def __init__(self, n_beams: int, hidden_dim: int, use_access_gate: bool):
                super().__init__()
                self.n_beams = int(n_beams)
                self.hidden_dim = int(hidden_dim)
                self.use_access_gate = bool(use_access_gate)
                self.access_gate_variant = gate_variant
                self.beam_encoder = nn.Sequential(
                    nn.Linear(dims.beam_dim, hidden_dim),
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
                context_dim = 9 + 4 + 4 + 1 + 1 + dims.candidate_stats_dim + 2 * hidden_dim
                self.context_encoder = nn.Sequential(
                    nn.Linear(context_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
                self.mode_head = nn.Linear(hidden_dim, len(MODE_NAMES))
                self.contention_mode_residual = nn.Linear(hidden_dim, len(MODE_NAMES))
                self.beam_query = nn.Linear(hidden_dim, hidden_dim)
                self.contention_beam_gate = nn.Linear(hidden_dim, hidden_dim)
                self.beam_bias = nn.Linear(hidden_dim, 1)
                self.value_head = nn.Linear(hidden_dim, 1)
                if self.use_access_gate:
                    self.access_gate_head = nn.Sequential(
                        nn.Linear(2 * hidden_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.SiLU(),
                        nn.Linear(hidden_dim, 1),
                    )
                    nn.init.zeros_(self.access_gate_head[-1].weight)
                    nn.init.zeros_(self.access_gate_head[-1].bias)

            def forward(self, tensors: dict[str, torch.Tensor]):
                beam_features = tensors["beam_features"]
                beam_tokens = self.beam_encoder(beam_features)
                contention = self.contention_encoder(tensors["contention_state"])
                if beam_tokens.dim() == 2:
                    beam_context = beam_tokens.mean(dim=0)
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
                        dim=0,
                    )
                    query = self.beam_query(self.context_encoder(context_input))
                else:
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
                    query = self.beam_query(self.context_encoder(context_input))
                context = self.context_encoder(context_input)
                mode_logits = self.mode_head(context) + self.contention_mode_residual(contention)
                if self.use_access_gate:
                    mode_logits = self.apply_access_gate(mode_logits, context, contention, tensors)
                gated_query = query * (1.0 + torch.tanh(self.contention_beam_gate(contention)))
                query_dim = 0 if beam_tokens.dim() == 2 else 1
                beam_logits = (beam_tokens * gated_query.unsqueeze(query_dim)).sum(dim=-1) / np.sqrt(float(self.hidden_dim))
                beam_logits = beam_logits + self.beam_bias(beam_tokens).squeeze(-1)
                value = self.value_head(context)
                return mode_logits, beam_logits, value

            def apply_access_gate(
                self,
                mode_logits: torch.Tensor,
                context: torch.Tensor,
                contention: torch.Tensor,
                tensors: dict[str, torch.Tensor],
            ) -> torch.Tensor:
                """Learn a decentralized active-access gate from public local state."""

                state = tensors["contention_state"]
                topology_need = state[..., 1]
                fail_pressure = state[..., 3]
                collision_pressure = state[..., 4]
                candidate_fraction = state[..., 6]
                candidate_score_max = state[..., 8]
                learned_logit = self.access_gate_head(torch.cat([context, contention], dim=-1)).squeeze(-1)
                if self.access_gate_variant == "adaptive":
                    collision_guard = collision_pressure * (0.55 + 0.45 * candidate_fraction)
                    rule_logit = (
                        0.95 * topology_need
                        + 0.55 * candidate_score_max
                        - 1.35 * collision_guard
                        - 0.55 * fail_pressure
                        - 0.15 * candidate_fraction
                    )
                elif self.access_gate_variant == "topology_preserving":
                    topology_evidence = torch.clamp(topology_need * candidate_score_max, min=0.0, max=1.0)
                    collision_relief = 1.0 - 0.45 * topology_evidence
                    collision_guard = collision_pressure * torch.clamp(collision_relief, min=0.45, max=1.0)
                    rule_logit = (
                        0.85 * topology_need
                        + 0.50 * candidate_score_max
                        + 0.35 * topology_evidence
                        - 0.95 * collision_guard
                        - 0.42 * fail_pressure
                        - 0.10 * candidate_fraction
                    )
                elif self.access_gate_variant == "balanced_topology":
                    topology_evidence = torch.clamp(topology_need * candidate_score_max, min=0.0, max=1.0)
                    crowding_guard = collision_pressure * (0.62 + 0.38 * candidate_fraction)
                    rule_logit = (
                        0.82 * topology_need
                        + 0.44 * candidate_score_max
                        + 0.18 * topology_evidence
                        - 1.18 * crowding_guard
                        - 0.46 * fail_pressure
                        - 0.14 * candidate_fraction
                    )
                else:
                    rule_logit = (
                        0.70 * topology_need
                        + 0.35 * candidate_score_max
                        - 1.10 * collision_pressure
                        - 0.45 * fail_pressure
                        - 0.25 * candidate_fraction
                    )
                gate = torch.tanh(learned_logit + rule_logit)
                adjusted = mode_logits.clone()
                tx_idx = MODE_NAMES.index("tx")
                rx_idx = MODE_NAMES.index("rx")
                sense_idx = MODE_NAMES.index("sense")
                idle_idx = MODE_NAMES.index("idle")
                if self.access_gate_variant == "adaptive":
                    active_gate = torch.clamp(gate, min=0.0)
                    throttle_gate = torch.clamp(-gate, min=0.0)
                    adjusted[..., tx_idx] = adjusted[..., tx_idx] + 0.65 * active_gate - 0.30 * throttle_gate
                    adjusted[..., rx_idx] = adjusted[..., rx_idx] + 0.80 * active_gate - 0.15 * throttle_gate
                    adjusted[..., sense_idx] = adjusted[..., sense_idx] - 0.35 * active_gate + 0.40 * throttle_gate
                    adjusted[..., idle_idx] = adjusted[..., idle_idx] - 0.60 * active_gate + 0.45 * throttle_gate
                elif self.access_gate_variant == "topology_preserving":
                    topology_evidence = torch.clamp(topology_need * candidate_score_max, min=0.0, max=1.0)
                    active_gate = torch.clamp(gate, min=0.0)
                    throttle_gate = torch.clamp(-gate, min=0.0)
                    adjusted[..., tx_idx] = (
                        adjusted[..., tx_idx] + 0.72 * active_gate + 0.20 * topology_evidence - 0.18 * throttle_gate
                    )
                    adjusted[..., rx_idx] = (
                        adjusted[..., rx_idx] + 0.86 * active_gate + 0.24 * topology_evidence - 0.10 * throttle_gate
                    )
                    adjusted[..., sense_idx] = (
                        adjusted[..., sense_idx] - 0.28 * active_gate + 0.25 * throttle_gate - 0.08 * topology_evidence
                    )
                    adjusted[..., idle_idx] = (
                        adjusted[..., idle_idx] - 0.55 * active_gate + 0.25 * throttle_gate - 0.18 * topology_evidence
                    )
                elif self.access_gate_variant == "balanced_topology":
                    topology_evidence = torch.clamp(topology_need * candidate_score_max, min=0.0, max=1.0)
                    active_gate = torch.clamp(gate, min=0.0)
                    throttle_gate = torch.clamp(-gate, min=0.0)
                    adjusted[..., tx_idx] = (
                        adjusted[..., tx_idx] + 0.62 * active_gate + 0.10 * topology_evidence - 0.26 * throttle_gate
                    )
                    adjusted[..., rx_idx] = (
                        adjusted[..., rx_idx] + 0.74 * active_gate + 0.14 * topology_evidence - 0.18 * throttle_gate
                    )
                    adjusted[..., sense_idx] = (
                        adjusted[..., sense_idx] - 0.26 * active_gate + 0.36 * throttle_gate - 0.05 * topology_evidence
                    )
                    adjusted[..., idle_idx] = (
                        adjusted[..., idle_idx] - 0.48 * active_gate + 0.36 * throttle_gate - 0.08 * topology_evidence
                    )
                else:
                    adjusted[..., tx_idx] = adjusted[..., tx_idx] + 0.85 * gate
                    adjusted[..., rx_idx] = adjusted[..., rx_idx] + 0.65 * gate
                    adjusted[..., sense_idx] = adjusted[..., sense_idx] - 0.45 * gate
                    adjusted[..., idle_idx] = adjusted[..., idle_idx] - 0.75 * gate
                return adjusted

        return Module(n_beams, hidden_dim, use_access_gate)


class GatedContentionGraphActorCritic(ContentionGraphActorCritic):
    """Contention actor with a learned decentralized active-access gate."""

    def __init__(self, *args, **kwargs):
        kwargs["use_access_gate"] = True
        super().__init__(*args, **kwargs)


class AdaptiveGatedContentionGraphActorCritic(ContentionGraphActorCritic):
    """Gated contention actor with a collision-adaptive decentralized access rule."""

    def __init__(self, *args, **kwargs):
        kwargs["use_access_gate"] = True
        kwargs["access_gate_variant"] = "adaptive"
        super().__init__(*args, **kwargs)


class TopologyAdaptiveGatedContentionGraphActorCritic(ContentionGraphActorCritic):
    """Adaptive gated actor that preserves active access under strong topology evidence."""

    def __init__(self, *args, **kwargs):
        kwargs["use_access_gate"] = True
        kwargs["access_gate_variant"] = "topology_preserving"
        super().__init__(*args, **kwargs)


class BalancedTopologyGatedContentionGraphActorCritic(ContentionGraphActorCritic):
    """Adaptive gated actor balancing collision suppression with topology growth evidence."""

    def __init__(self, *args, **kwargs):
        kwargs["use_access_gate"] = True
        kwargs["access_gate_variant"] = "balanced_topology"
        super().__init__(*args, **kwargs)


def observation_to_contention_tensors(observation: dict, device, torch_module, n_beams: int) -> dict:
    observed_beams = len(observation["beam_belief"])
    candidate_score = np.asarray(observation.get("candidate_score", np.zeros(observed_beams)), dtype=np.float32)
    beam_collision = np.log1p(np.asarray(observation.get("beam_collision", np.zeros(observed_beams)), dtype=np.float32))
    collision_norm = _normalize_vector(beam_collision)
    beam_belief = np.asarray(observation["beam_belief"], dtype=np.float32)
    uncertainty = beam_belief * (1.0 - beam_belief)
    beam_features = np.stack(
        [
            beam_belief,
            np.asarray(observation["beam_age"], dtype=np.float32),
            np.log1p(np.asarray(observation["beam_success"], dtype=np.float32)),
            np.log1p(np.asarray(observation["beam_fail"], dtype=np.float32)),
            candidate_score,
            collision_norm,
            uncertainty.astype(np.float32, copy=False),
            np.asarray(observation.get("candidate_mask", np.zeros(observed_beams)), dtype=np.float32),
        ],
        axis=1,
    )
    candidate_mask = np.asarray(observation.get("candidate_mask", np.zeros(observed_beams)), dtype=np.float32)
    candidate_stats = np.asarray(
        [
            float(np.mean(candidate_mask)) if candidate_mask.size else 0.0,
            float(np.mean(candidate_score)) if candidate_score.size else 0.0,
            float(np.max(candidate_score)) if candidate_score.size else 0.0,
            float(np.mean(collision_norm)) if collision_norm.size else 0.0,
        ],
        dtype=np.float32,
    )
    tensors = {
        "self_state": torch_module.as_tensor(observation["self_state"], dtype=torch_module.float32, device=device),
        "local_summary": torch_module.as_tensor(observation["local_summary"], dtype=torch_module.float32, device=device),
        "last_mode": torch_module.as_tensor(observation["last_mode"], dtype=torch_module.float32, device=device),
        "last_beam": torch_module.as_tensor(observation["last_beam"], dtype=torch_module.float32, device=device),
        "topology_deficit": torch_module.as_tensor(
            observation.get("topology_deficit", np.zeros(1, dtype=np.float32)),
            dtype=torch_module.float32,
            device=device,
        ),
        "contention_state": torch_module.as_tensor(
            observation.get("contention_state", np.zeros(ContentionFeatures.contention_dim, dtype=np.float32)),
            dtype=torch_module.float32,
            device=device,
        ),
        "candidate_stats": torch_module.as_tensor(candidate_stats, dtype=torch_module.float32, device=device),
        "beam_features": torch_module.as_tensor(beam_features, dtype=torch_module.float32, device=device),
        "candidate_score": torch_module.as_tensor(candidate_score, dtype=torch_module.float32, device=device),
        "beam_collision_norm": torch_module.as_tensor(collision_norm, dtype=torch_module.float32, device=device),
    }
    if "candidate_mask" in observation:
        tensors["candidate_mask"] = torch_module.as_tensor(candidate_mask, dtype=torch_module.float32, device=device)
    if "rule_mode_logits" in observation:
        tensors["rule_mode_logits"] = torch_module.as_tensor(
            observation["rule_mode_logits"], dtype=torch_module.float32, device=device
        )
    return tensors


def observations_to_batched_contention_tensors(observations: Sequence[dict], device, torch_module, n_beams: int) -> dict:
    tensors = [observation_to_contention_tensors(observation, device, torch_module, n_beams) for observation in observations]
    keys = set().union(*(tensor.keys() for tensor in tensors))
    batched = {}
    for key in keys:
        if all(key in tensor for tensor in tensors):
            batched[key] = torch_module.stack([tensor[key] for tensor in tensors], dim=0)
    return batched


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32, copy=False)
    span = float(values.max() - values.min())
    if span <= 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - float(values.min())) / span).astype(np.float32, copy=False)
