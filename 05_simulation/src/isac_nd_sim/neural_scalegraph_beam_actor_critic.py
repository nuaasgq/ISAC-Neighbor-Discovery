from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .marl_env import MODE_NAMES
from .neural_shared_actor_critic import PolicyStep
from .simulator import Action


class ScaleGraphBeamActorCritic:
    """Scale-invariant beam-set actor-critic for decentralized execution.

    The actor still consumes only per-agent public observations, but it replaces
    mean beam pooling with query-based set attention over candidate beam tokens.
    This keeps the policy independent of the number of UAVs and compatible with
    different beam codebooks at transfer time.
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
        pool_tokens: int = 4,
        disabled_modes: Sequence[str] | None = None,
    ):
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover - depends on optional torch
            raise RuntimeError("PyTorch is required for ScaleGraphBeamActorCritic.") from exc

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
        self.model = _ScaleGraphBeamActorCriticModule(self.n_beams, self.hidden_dim, int(pool_tokens)).to(self.device)

    def parameters(self):
        return self.model.parameters()

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def logits_value(self, observation: dict, hard_mask: bool = False):
        torch = self.torch
        tensors = observation_to_scalegraph_tensors(observation, self.device, torch, self.n_beams)
        tensors = self._prepare_tensors(tensors)
        mode_logits, beam_logits, value = self.model(tensors, use_candidate_pool=self.use_candidate_mask)
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
            tensors["beam_features"][:, 4] = 0.0
        if not self.use_candidate_mask and "candidate_mask" in tensors:
            tensors = dict(tensors)
            tensors["beam_features"] = tensors["beam_features"].clone()
            tensors["beam_features"][:, 5] = 0.0
            tensors["candidate_stats"] = tensors["candidate_stats"].clone()
            tensors["candidate_stats"][0] = 0.0
        if not self.use_topology_deficit and "topology_deficit" in tensors:
            tensors = dict(tensors)
            tensors["topology_deficit"] = torch.zeros_like(tensors["topology_deficit"])
        return tensors

    def act(
        self,
        observations: Sequence[dict],
        deterministic: bool = False,
        mode_temperature: float = 1.0,
        beam_temperature: float = 1.0,
        gate_temperature: float = 1.0,
    ) -> PolicyStep:
        torch = self.torch
        from torch.distributions import Categorical

        actions: list[Action] = []
        log_probs = []
        values = []
        entropies = []
        for observation in observations:
            mode_logits, beam_logits, value = self.logits_value(observation, hard_mask=True)
            sample_mode_logits = _temperature_scaled_logits(mode_logits, mode_temperature)
            sample_beam_logits = _temperature_scaled_logits(beam_logits, beam_temperature)
            mode_dist = Categorical(logits=sample_mode_logits)
            beam_dist = Categorical(logits=sample_beam_logits)
            if deterministic:
                mode_idx = int(torch.argmax(mode_logits).item())
                beam_idx = int(torch.argmax(beam_logits).item())
            else:
                mode_idx = int(mode_dist.sample().item())
                beam_idx = int(beam_dist.sample().item())
            mode = MODE_NAMES[mode_idx]
            sampled_beam_idx = beam_idx
            if mode == "idle":
                beam_idx = 0
            actions.append(Action(mode, beam_idx))
            mode_tensor = torch.tensor(mode_idx, dtype=torch.long, device=self.device)
            beam_tensor = torch.tensor(sampled_beam_idx, dtype=torch.long, device=self.device)
            beam_log_prob = torch.zeros((), dtype=value.dtype, device=self.device)
            if mode != "idle":
                beam_log_prob = beam_dist.log_prob(beam_tensor)
            log_probs.append(mode_dist.log_prob(mode_tensor) + beam_log_prob)
            values.append(value.squeeze(-1))
            entropies.append(mode_dist.entropy() + beam_dist.entropy())
        return PolicyStep(
            actions=actions,
            log_probs=torch.stack(log_probs),
            values=torch.stack(values),
            entropies=torch.stack(entropies),
        )


def _temperature_scaled_logits(logits, temperature: float):
    value = max(float(temperature), 1.0e-6)
    if abs(value - 1.0) <= 1.0e-9:
        return logits
    return logits / value


class _ScaleGraphBeamActorCriticModule:
    def __new__(cls, n_beams: int, hidden_dim: int, pool_tokens: int):
        import torch
        import torch.nn as nn

        class Module(nn.Module):
            def __init__(self, n_beams: int, hidden_dim: int, pool_tokens: int):
                super().__init__()
                self.n_beams = int(n_beams)
                self.hidden_dim = int(hidden_dim)
                self.pool_tokens = int(max(1, pool_tokens))
                attention_heads = _divisible_heads(hidden_dim)
                self.beam_encoder = nn.Sequential(
                    nn.Linear(9, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
                self.pool_queries = nn.Parameter(torch.randn(self.pool_tokens, hidden_dim) * 0.02)
                self.pool_attention = nn.MultiheadAttention(hidden_dim, attention_heads, batch_first=False)
                context_dim = 9 + 4 + 4 + 1 + 1 + 4 + self.pool_tokens * hidden_dim
                self.context_encoder = nn.Sequential(
                    nn.Linear(context_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
                self.mode_head = nn.Linear(hidden_dim, len(MODE_NAMES))
                self.beam_query = nn.Linear(hidden_dim, hidden_dim)
                self.beam_bias = nn.Linear(hidden_dim, 1)
                self.value_head = nn.Linear(hidden_dim, 1)

            def forward(self, tensors: dict[str, torch.Tensor], use_candidate_pool: bool = False):
                beam_features = tensors["beam_features"]
                beam_tokens = self.beam_encoder(beam_features)
                key_padding_mask = None
                if use_candidate_pool and "candidate_mask" in tensors:
                    mask = tensors["candidate_mask"] > 0.5
                    if bool(mask.any().item()):
                        key_padding_mask = (~mask).unsqueeze(0)
                queries = self.pool_queries.unsqueeze(1)
                keys = beam_tokens.unsqueeze(1)
                pooled, _weights = self.pool_attention(queries, keys, keys, key_padding_mask=key_padding_mask)
                beam_context = pooled.transpose(0, 1).reshape(-1)
                context_input = torch.cat(
                    [
                        tensors["self_state"],
                        tensors["local_summary"],
                        tensors["last_mode"],
                        tensors["last_beam"],
                        tensors["topology_deficit"],
                        tensors["candidate_stats"],
                        beam_context,
                    ],
                    dim=0,
                )
                context = self.context_encoder(context_input)
                mode_logits = self.mode_head(context)
                query = self.beam_query(context)
                beam_logits = (beam_tokens * query.unsqueeze(0)).sum(dim=1) / np.sqrt(float(self.hidden_dim))
                beam_logits = beam_logits + self.beam_bias(beam_tokens).squeeze(-1)
                value = self.value_head(context)
                return mode_logits, beam_logits, value

        return Module(n_beams, hidden_dim, pool_tokens)


def _divisible_heads(hidden_dim: int) -> int:
    heads = min(4, max(1, int(hidden_dim)))
    while heads > 1 and int(hidden_dim) % heads != 0:
        heads -= 1
    return heads


def observation_to_scalegraph_tensors(observation: dict, device, torch_module, n_beams: int) -> dict:
    beam_belief = np.asarray(observation["beam_belief"], dtype=np.float32)
    n_observed_beams = len(beam_belief)
    candidate_score = np.asarray(observation.get("candidate_score", np.zeros(n_observed_beams)), dtype=np.float32)
    candidate_mask = np.asarray(observation.get("candidate_mask", np.zeros(n_observed_beams)), dtype=np.float32)
    indices = np.arange(n_observed_beams, dtype=np.float32)
    phase = 2.0 * np.pi * indices / max(1.0, float(n_beams))
    uncertainty = beam_belief * (1.0 - beam_belief)
    beam_features = np.stack(
        [
            beam_belief,
            np.asarray(observation["beam_age"], dtype=np.float32),
            np.log1p(np.asarray(observation["beam_success"], dtype=np.float32)),
            np.log1p(np.asarray(observation["beam_fail"], dtype=np.float32)),
            candidate_score,
            candidate_mask,
            uncertainty.astype(np.float32, copy=False),
            np.sin(phase).astype(np.float32, copy=False),
            np.cos(phase).astype(np.float32, copy=False),
        ],
        axis=1,
    )
    candidate_stats = np.asarray(
        [
            float(np.mean(candidate_mask)) if candidate_mask.size else 0.0,
            float(np.mean(candidate_score)) if candidate_score.size else 0.0,
            float(np.max(candidate_score)) if candidate_score.size else 0.0,
            float(np.mean(uncertainty)) if uncertainty.size else 0.0,
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
        "beam_features": torch_module.as_tensor(beam_features, dtype=torch_module.float32, device=device),
        "candidate_score": torch_module.as_tensor(candidate_score, dtype=torch_module.float32, device=device),
        "candidate_mask": torch_module.as_tensor(candidate_mask, dtype=torch_module.float32, device=device),
        "candidate_stats": torch_module.as_tensor(candidate_stats, dtype=torch_module.float32, device=device),
    }
    if "rule_mode_logits" in observation:
        tensors["rule_mode_logits"] = torch_module.as_tensor(
            observation["rule_mode_logits"], dtype=torch_module.float32, device=device
        )
    return tensors
