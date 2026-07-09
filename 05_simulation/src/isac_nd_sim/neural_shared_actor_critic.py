from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .marl_env import MODE_NAMES
from .simulator import Action


@dataclass(frozen=True)
class PolicyStep:
    actions: list[Action]
    log_probs: object
    values: object
    entropies: object


class SharedBeamActorCritic:
    """Torch-backed shared actor-critic for decentralized UAV execution.

    The network consumes only the public per-agent observation dictionary from
    MarlNeighborDiscoveryEnv. Centralized truth is not used by the actor.
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
        disabled_modes: Sequence[str] | None = None,
    ):
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover - depends on optional torch
            raise RuntimeError("PyTorch is required for SharedBeamActorCritic.") from exc

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
        self.disabled_mode_indices = tuple(MODE_NAMES.index(mode) for mode in (disabled_modes or ()) if mode in MODE_NAMES)
        self.model = _SharedBeamActorCriticModule(self.n_beams, self.hidden_dim).to(self.device)

    def parameters(self):
        return self.model.parameters()

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def logits_value(self, observation: dict, hard_mask: bool = False):
        torch = self.torch
        tensors = observation_to_tensors(observation, self.device, torch)
        tensors = self._prepare_tensors(tensors)
        mode_logits, beam_logits, value = self.model(tensors)
        if self.use_rule_residual:
            if "rule_mode_logits" in tensors:
                mode_logits = mode_logits + self.rule_residual_scale * tensors["rule_mode_logits"]
            if "candidate_score" in tensors:
                beam_logits = beam_logits + self.rule_residual_scale * tensors["candidate_score"]
        if hard_mask and self.use_candidate_mask and "candidate_mask" in tensors:
            mask = tensors["candidate_mask"] > 0.5
            if bool(mask.any().item()):
                beam_logits = beam_logits.masked_fill(~mask, -1.0e9)
        mode_logits = self._mask_disabled_modes(mode_logits)
        return mode_logits, beam_logits, value

    def batched_logits_value(self, observations: Sequence[dict], hard_mask: bool = False):
        torch = self.torch
        tensors = observations_to_batched_tensors(observations, self.device, torch)
        tensors = self._prepare_tensors(tensors)
        mode_logits, beam_logits, value = self.model(tensors)
        if self.use_rule_residual:
            if "rule_mode_logits" in tensors:
                mode_logits = mode_logits + self.rule_residual_scale * tensors["rule_mode_logits"]
            if "candidate_score" in tensors:
                beam_logits = beam_logits + self.rule_residual_scale * tensors["candidate_score"]
        if hard_mask and self.use_candidate_mask and "candidate_mask" in tensors:
            mask = tensors["candidate_mask"] > 0.5
            has_candidate = mask.any(dim=-1, keepdim=True)
            beam_logits = torch.where(
                has_candidate,
                beam_logits.masked_fill(~mask, -1.0e9),
                beam_logits,
            )
        mode_logits = self._mask_disabled_modes(mode_logits)
        return mode_logits, beam_logits, value

    def _mask_disabled_modes(self, mode_logits):
        if not self.disabled_mode_indices:
            return mode_logits
        masked = mode_logits.clone()
        for mode_index in self.disabled_mode_indices:
            masked[..., mode_index] = -1.0e9
        return masked

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
        log_probs = mode_dist.log_prob(mode_idx) + beam_log_prob
        values = value.squeeze(-1)
        entropies = mode_dist.entropy() + beam_dist.entropy()
        return PolicyStep(
            actions=actions,
            log_probs=log_probs,
            values=values,
            entropies=entropies,
        )


class _SharedBeamActorCriticModule:
    def __new__(cls, n_beams: int, hidden_dim: int):
        import torch
        import torch.nn as nn

        class Module(nn.Module):
            def __init__(self, n_beams: int, hidden_dim: int):
                super().__init__()
                self.n_beams = int(n_beams)
                self.hidden_dim = int(hidden_dim)
                self.beam_encoder = nn.Sequential(
                    nn.Linear(5, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                )
                context_dim = 9 + 4 + 4 + 1 + 1 + hidden_dim
                self.context_encoder = nn.Sequential(
                    nn.Linear(context_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                )
                self.mode_head = nn.Linear(hidden_dim, len(MODE_NAMES))
                self.beam_query = nn.Linear(hidden_dim, hidden_dim)
                self.beam_bias = nn.Linear(hidden_dim, 1)
                self.value_head = nn.Linear(hidden_dim, 1)

            def forward(self, tensors: dict[str, torch.Tensor]):
                beam_features = tensors["beam_features"]
                beam_tokens = self.beam_encoder(beam_features)
                if beam_tokens.dim() == 2:
                    beam_context = beam_tokens.mean(dim=0)
                    context_input = torch.cat(
                        [
                            tensors["self_state"],
                            tensors["local_summary"],
                            tensors["last_mode"],
                            tensors["last_beam"],
                            tensors["topology_deficit"],
                            beam_context,
                        ],
                        dim=0,
                    )
                else:
                    beam_context = beam_tokens.mean(dim=1)
                    context_input = torch.cat(
                        [
                            tensors["self_state"],
                            tensors["local_summary"],
                            tensors["last_mode"],
                            tensors["last_beam"],
                            tensors["topology_deficit"],
                            beam_context,
                        ],
                        dim=1,
                    )
                context = self.context_encoder(context_input)
                mode_logits = self.mode_head(context)
                query = self.beam_query(context)
                query_dim = 0 if beam_tokens.dim() == 2 else 1
                beam_logits = (beam_tokens * query.unsqueeze(query_dim)).sum(dim=-1) / np.sqrt(float(self.hidden_dim))
                beam_logits = beam_logits + self.beam_bias(beam_tokens).squeeze(-1)
                value = self.value_head(context)
                return mode_logits, beam_logits, value

        return Module(n_beams, hidden_dim)


def observation_to_tensors(observation: dict, device, torch_module) -> dict:
    n_beams = len(observation["beam_belief"])
    candidate_score = np.asarray(observation.get("candidate_score", np.zeros(n_beams)), dtype=np.float32)
    beam_features = np.stack(
        [
            np.asarray(observation["beam_belief"], dtype=np.float32),
            np.asarray(observation["beam_age"], dtype=np.float32),
            np.log1p(np.asarray(observation["beam_success"], dtype=np.float32)),
            np.log1p(np.asarray(observation["beam_fail"], dtype=np.float32)),
            candidate_score,
        ],
        axis=1,
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
        "beam_features": torch_module.as_tensor(beam_features, dtype=torch_module.float32, device=device),
    }
    tensors["candidate_score"] = torch_module.as_tensor(candidate_score, dtype=torch_module.float32, device=device)
    if "candidate_mask" in observation:
        tensors["candidate_mask"] = torch_module.as_tensor(
            observation["candidate_mask"], dtype=torch_module.float32, device=device
        )
    if "rule_mode_logits" in observation:
        tensors["rule_mode_logits"] = torch_module.as_tensor(
            observation["rule_mode_logits"], dtype=torch_module.float32, device=device
        )
    return tensors


def observations_to_batched_tensors(observations: Sequence[dict], device, torch_module) -> dict:
    tensors = [observation_to_tensors(observation, device, torch_module) for observation in observations]
    keys = set().union(*(tensor.keys() for tensor in tensors))
    batched = {}
    for key in keys:
        if all(key in tensor for tensor in tensors):
            batched[key] = torch_module.stack([tensor[key] for tensor in tensors], dim=0)
    return batched
