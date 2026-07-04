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

    def __init__(self, n_beams: int, hidden_dim: int = 96, device: str = "cpu", use_candidate_mask: bool = False):
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
        self.model = _SharedBeamActorCriticModule(self.n_beams, self.hidden_dim).to(self.device)

    def parameters(self):
        return self.model.parameters()

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def act(self, observations: Sequence[dict], deterministic: bool = False) -> PolicyStep:
        torch = self.torch
        from torch.distributions import Categorical

        actions: list[Action] = []
        log_probs = []
        values = []
        entropies = []
        for observation in observations:
            tensors = observation_to_tensors(observation, self.device, torch)
            mode_logits, beam_logits, value = self.model(tensors)
            if self.use_candidate_mask and "candidate_mask" in tensors:
                mask = tensors["candidate_mask"] > 0.5
                if bool(mask.any().item()):
                    beam_logits = beam_logits.masked_fill(~mask, -1.0e9)
            mode_dist = Categorical(logits=mode_logits)
            beam_dist = Categorical(logits=beam_logits)
            if deterministic:
                mode_idx = int(torch.argmax(mode_logits).item())
                beam_idx = int(torch.argmax(beam_logits).item())
            else:
                mode_idx = int(mode_dist.sample().item())
                beam_idx = int(beam_dist.sample().item())
            mode = MODE_NAMES[mode_idx]
            if mode == "idle":
                beam_idx = 0
            actions.append(Action(mode, beam_idx))
            mode_tensor = torch.tensor(mode_idx, dtype=torch.long, device=self.device)
            beam_tensor = torch.tensor(beam_idx, dtype=torch.long, device=self.device)
            log_probs.append(mode_dist.log_prob(mode_tensor) + beam_dist.log_prob(beam_tensor))
            values.append(value.squeeze(-1))
            entropies.append(mode_dist.entropy() + beam_dist.entropy())
        return PolicyStep(
            actions=actions,
            log_probs=torch.stack(log_probs),
            values=torch.stack(values),
            entropies=torch.stack(entropies),
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
                    nn.Linear(4, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                )
                context_dim = 9 + 4 + 4 + 1 + hidden_dim
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
                beam_context = beam_tokens.mean(dim=0)
                context_input = torch.cat(
                    [
                        tensors["self_state"],
                        tensors["local_summary"],
                        tensors["last_mode"],
                        tensors["last_beam"],
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

        return Module(n_beams, hidden_dim)


def observation_to_tensors(observation: dict, device, torch_module) -> dict:
    beam_features = np.stack(
        [
            np.asarray(observation["beam_belief"], dtype=np.float32),
            np.asarray(observation["beam_age"], dtype=np.float32),
            np.log1p(np.asarray(observation["beam_success"], dtype=np.float32)),
            np.log1p(np.asarray(observation["beam_fail"], dtype=np.float32)),
        ],
        axis=1,
    )
    tensors = {
        "self_state": torch_module.as_tensor(observation["self_state"], dtype=torch_module.float32, device=device),
        "local_summary": torch_module.as_tensor(observation["local_summary"], dtype=torch_module.float32, device=device),
        "last_mode": torch_module.as_tensor(observation["last_mode"], dtype=torch_module.float32, device=device),
        "last_beam": torch_module.as_tensor(observation["last_beam"], dtype=torch_module.float32, device=device),
        "beam_features": torch_module.as_tensor(beam_features, dtype=torch_module.float32, device=device),
    }
    if "candidate_mask" in observation:
        tensors["candidate_mask"] = torch_module.as_tensor(
            observation["candidate_mask"], dtype=torch_module.float32, device=device
        )
    return tensors
