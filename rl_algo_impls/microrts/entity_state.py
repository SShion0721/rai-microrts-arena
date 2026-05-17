from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

NO_UNIT_TYPE_IDX = 6
IN_BOUNDS_IDX = 58


@dataclass
class EntityStateBatch:
    nodes: torch.Tensor  # Float[B, S, C]
    positions: torch.Tensor  # Float[B, S, 2], y/x in [-1, 1]
    key_padding_mask: torch.Tensor  # Bool[B, S]
    keep_mask: torch.Tensor  # Bool[B, H*W]
    n_entities: torch.Tensor  # Long[B]
    edge_index: Optional[torch.Tensor] = None  # Long[2, E], padded-node indices
    edge_attr: Optional[torch.Tensor] = None  # Float[E, F]

    def to(self, device: torch.device) -> "EntityStateBatch":
        return EntityStateBatch(
            nodes=self.nodes.to(device),
            positions=self.positions.to(device),
            key_padding_mask=self.key_padding_mask.to(device),
            keep_mask=self.keep_mask.to(device),
            n_entities=self.n_entities.to(device),
            edge_index=self.edge_index.to(device) if self.edge_index is not None else None,
            edge_attr=self.edge_attr.to(device) if self.edge_attr is not None else None,
        )


def empty_spaces_mask(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.shape[-1] <= max(NO_UNIT_TYPE_IDX, IN_BOUNDS_IDX):
        return torch.zeros(tokens.shape[:-1], dtype=torch.bool, device=tokens.device)
    return tokens[..., (NO_UNIT_TYPE_IDX, IN_BOUNDS_IDX)].bool().all(dim=-1)


def extract_entity_state_batch(
    obs: torch.Tensor,
    add_position_features: bool = True,
    edge_radius: Optional[float] = None,
) -> EntityStateBatch:
    """Extract non-empty microRTS cells into padded entity tensors.

    Args:
        obs: Observation tensor in CHW or BCHW layout.
        add_position_features: Append normalized y/x features to each node.
        edge_radius: Optional normalized radius in grid cells for relation edges.
            When unset, no edges are materialized; Transformer encoders can still
            use the padded nodes directly.
    """

    if obs.ndim == 3:
        obs = obs.unsqueeze(0)
    if obs.ndim != 4:
        raise ValueError(f"Expected CHW or BCHW observation, got shape {tuple(obs.shape)}")

    batch_size, channels, height, width = obs.shape
    flattened = obs.flatten(2).permute(0, 2, 1).float()
    keep_mask = ~empty_spaces_mask(flattened)
    n_entities = keep_mask.sum(dim=1)
    max_entities = int(n_entities.max().item()) if batch_size else 0

    nodes = torch.zeros(
        batch_size,
        max_entities,
        channels + (2 if add_position_features else 0),
        dtype=flattened.dtype,
        device=flattened.device,
    )
    positions = torch.zeros(
        batch_size, max_entities, 2, dtype=flattened.dtype, device=flattened.device
    )
    key_padding_mask = torch.ones(
        batch_size, max_entities, dtype=torch.bool, device=flattened.device
    )

    y_pos = torch.arange(height, device=flattened.device).repeat_interleave(width)
    x_pos = torch.arange(width, device=flattened.device).repeat(height)
    y_norm = _normalize_position(y_pos.float(), height)
    x_norm = _normalize_position(x_pos.float(), width)
    all_positions = torch.stack((y_norm, x_norm), dim=-1)

    for batch_idx in range(batch_size):
        selected = keep_mask[batch_idx]
        count = int(n_entities[batch_idx].item())
        if count == 0:
            continue
        selected_nodes = flattened[batch_idx, selected]
        selected_positions = all_positions[selected]
        if add_position_features:
            selected_nodes = torch.cat((selected_nodes, selected_positions), dim=-1)
        nodes[batch_idx, :count] = selected_nodes
        positions[batch_idx, :count] = selected_positions
        key_padding_mask[batch_idx, :count] = False

    edge_index, edge_attr = _build_radius_edges(positions, n_entities, edge_radius)
    return EntityStateBatch(
        nodes=nodes,
        positions=positions,
        key_padding_mask=key_padding_mask,
        keep_mask=keep_mask,
        n_entities=n_entities,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


def _normalize_position(position: torch.Tensor, size: int) -> torch.Tensor:
    if size <= 1:
        return torch.zeros_like(position)
    return position / (size - 1) * 2 - 1


def _build_radius_edges(
    positions: torch.Tensor,
    n_entities: torch.Tensor,
    edge_radius: Optional[float],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if edge_radius is None:
        return None, None

    edge_indices = []
    edge_attrs = []
    max_entities = positions.shape[1]
    for batch_idx, count_tensor in enumerate(n_entities):
        count = int(count_tensor.item())
        for src in range(count):
            for dst in range(count):
                if src == dst:
                    continue
                delta = positions[batch_idx, dst] - positions[batch_idx, src]
                distance = torch.linalg.vector_norm(delta, ord=2)
                if distance <= edge_radius:
                    edge_indices.append(
                        (
                            batch_idx * max_entities + src,
                            batch_idx * max_entities + dst,
                        )
                    )
                    edge_attrs.append(
                        torch.cat(
                            (
                                torch.tensor(
                                    [float(batch_idx)],
                                    dtype=positions.dtype,
                                    device=positions.device,
                                ),
                                delta,
                                distance.unsqueeze(0),
                            )
                        )
                    )

    if not edge_indices:
        return (
            torch.empty(2, 0, dtype=torch.long, device=positions.device),
            torch.empty(0, 4, dtype=positions.dtype, device=positions.device),
        )
    return (
        torch.tensor(edge_indices, dtype=torch.long, device=positions.device).T,
        torch.stack(edge_attrs),
    )
