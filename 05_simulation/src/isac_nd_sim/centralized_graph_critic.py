from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn


class _SharedMessagePassing(nn.Module):
    """One lightweight message-passing update, reused for both graph hops."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.source_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.target_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.message_activation = nn.SiLU()
        self.update = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_norm = nn.LayerNorm(hidden_dim)
        self.update_activation = nn.SiLU()

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        global_embeddings: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        # edge_features[:, target, source] describes the source -> target edge.
        source_terms = self.source_projection(node_embeddings).unsqueeze(1)
        target_terms = self.target_projection(node_embeddings).unsqueeze(2)
        messages = self.message_activation(source_terms + target_terms + edge_embeddings)

        weights = edge_mask.unsqueeze(-1).to(dtype=messages.dtype)
        neighbor_count = weights.sum(dim=2).clamp_min(1.0)
        aggregated = (messages * weights).sum(dim=2) / neighbor_count

        global_context = global_embeddings.unsqueeze(1).expand_as(node_embeddings)
        delta = self.update(torch.cat((node_embeddings, aggregated, global_context), dim=-1))
        return self.update_activation(self.update_norm(node_embeddings + delta))


class CentralizedGraphCritic(nn.Module):
    """Two-hop graph critic producing one centralized-training value per agent.

    Required graph tensors are ``node_features [T, N, F]``,
    ``edge_features [T, N, N, E]``, and ``global_features [T, G]``.
    The first and second node axes of ``edge_features`` denote target and source,
    respectively. An optional boolean/numeric ``edge_mask [T, N, N]`` can mark
    valid edges; without it, every non-self edge is valid. Self edges are always
    excluded. The two message-passing hops share the same parameters.
    """

    output_per_agent = True
    num_message_passing_steps = 2

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.node_feature_dim = _positive_dimension("node_feature_dim", node_feature_dim)
        self.edge_feature_dim = _positive_dimension("edge_feature_dim", edge_feature_dim)
        self.global_feature_dim = _positive_dimension("global_feature_dim", global_feature_dim)
        self.hidden_dim = _positive_dimension("hidden_dim", hidden_dim)

        self.node_encoder = nn.Linear(self.node_feature_dim, self.hidden_dim)
        self.edge_encoder = nn.Linear(self.edge_feature_dim, self.hidden_dim)
        self.global_encoder = nn.Linear(self.global_feature_dim, self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.input_activation = nn.SiLU()

        self.message_passing = _SharedMessagePassing(self.hidden_dim)
        self.value_head = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, graph_tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
        node_features, edge_features, global_features, edge_mask = self._validated_inputs(graph_tensors)

        global_embeddings = self.global_encoder(global_features)
        node_embeddings = self.input_activation(
            self.input_norm(self.node_encoder(node_features) + global_embeddings.unsqueeze(1))
        )
        edge_embeddings = self.edge_encoder(edge_features)

        for _ in range(self.num_message_passing_steps):
            node_embeddings = self.message_passing(
                node_embeddings,
                edge_embeddings,
                global_embeddings,
                edge_mask,
            )

        global_context = global_embeddings.unsqueeze(1).expand_as(node_embeddings)
        values = self.value_head(torch.cat((node_embeddings, global_context), dim=-1)).squeeze(-1)
        if not bool(torch.isfinite(values).all().item()):
            raise FloatingPointError("CentralizedGraphCritic produced non-finite values.")
        return values

    def _validated_inputs(
        self,
        graph_tensors: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(graph_tensors, Mapping):
            raise TypeError("graph_tensors must be a mapping of tensor names to torch.Tensor values.")

        required = ("node_features", "edge_features", "global_features")
        missing = [name for name in required if name not in graph_tensors]
        if missing:
            raise KeyError(f"graph_tensors is missing required keys: {', '.join(missing)}")

        node_features = _require_tensor(graph_tensors["node_features"], "node_features", 3)
        edge_features = _require_tensor(graph_tensors["edge_features"], "edge_features", 4)
        global_features = _require_tensor(graph_tensors["global_features"], "global_features", 2)

        time_steps, n_agents, node_dim = node_features.shape
        if time_steps <= 0 or n_agents <= 0:
            raise ValueError(
                "node_features must contain at least one time step and one agent; "
                f"received shape {tuple(node_features.shape)}."
            )
        expected_edge_shape = (time_steps, n_agents, n_agents, self.edge_feature_dim)
        expected_global_shape = (time_steps, self.global_feature_dim)
        if node_dim != self.node_feature_dim:
            raise ValueError(
                f"node_features last dimension must be {self.node_feature_dim}, got {node_dim}."
            )
        if tuple(edge_features.shape) != expected_edge_shape:
            raise ValueError(
                f"edge_features must have shape {expected_edge_shape}, got {tuple(edge_features.shape)}."
            )
        if tuple(global_features.shape) != expected_global_shape:
            raise ValueError(
                f"global_features must have shape {expected_global_shape}, got {tuple(global_features.shape)}."
            )

        feature_tensors = (node_features, edge_features, global_features)
        reference_device = node_features.device
        reference_dtype = node_features.dtype
        for name, tensor in zip(required, feature_tensors, strict=True):
            if not tensor.is_floating_point():
                raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}.")
            if tensor.device != reference_device:
                raise ValueError(
                    f"All graph feature tensors must be on {reference_device}; "
                    f"{name} is on {tensor.device}."
                )
            if tensor.dtype != reference_dtype:
                raise TypeError(
                    f"All graph feature tensors must use dtype {reference_dtype}; "
                    f"{name} uses {tensor.dtype}."
                )
            _require_finite(tensor, name)

        parameter = next(self.parameters())
        if parameter.device != reference_device:
            raise ValueError(
                f"Graph tensors are on {reference_device}, but the critic is on {parameter.device}. "
                "Move the critic and inputs to the same device."
            )
        if parameter.dtype != reference_dtype:
            raise TypeError(
                f"Graph tensors use {reference_dtype}, but the critic parameters use {parameter.dtype}. "
                "Convert the critic or inputs to the same dtype."
            )

        off_diagonal = ~torch.eye(n_agents, dtype=torch.bool, device=reference_device).unsqueeze(0)
        raw_edge_mask = graph_tensors.get("edge_mask")
        if raw_edge_mask is None:
            edge_mask = off_diagonal.expand(time_steps, -1, -1)
        else:
            edge_mask = self._validated_edge_mask(raw_edge_mask, time_steps, n_agents, reference_device)
            edge_mask = edge_mask & off_diagonal
        return node_features, edge_features, global_features, edge_mask

    @staticmethod
    def _validated_edge_mask(
        edge_mask: torch.Tensor,
        time_steps: int,
        n_agents: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(edge_mask, torch.Tensor):
            raise TypeError(f"edge_mask must be a torch.Tensor, got {type(edge_mask).__name__}.")
        expected_shape = (time_steps, n_agents, n_agents)
        if tuple(edge_mask.shape) != expected_shape:
            raise ValueError(f"edge_mask must have shape {expected_shape}, got {tuple(edge_mask.shape)}.")
        if edge_mask.device != device:
            raise ValueError(f"edge_mask must be on {device}, got {edge_mask.device}.")
        if edge_mask.dtype == torch.bool:
            return edge_mask
        if edge_mask.is_floating_point():
            _require_finite(edge_mask, "edge_mask")
        elif edge_mask.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError(f"edge_mask must have boolean or numeric dtype, got {edge_mask.dtype}.")
        return edge_mask != 0


def _positive_dimension(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer, got {value!r}.")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _require_tensor(value: object, name: str, expected_rank: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if value.ndim != expected_rank:
        raise ValueError(f"{name} must have rank {expected_rank}, got shape {tuple(value.shape)}.")
    return value


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} contains NaN or infinite values.")
