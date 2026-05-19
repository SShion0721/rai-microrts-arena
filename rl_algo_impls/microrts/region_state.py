from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch

from rl_algo_impls.microrts.entity_state import (
    IN_BOUNDS_IDX,
    OWNER_END_IDX,
    OWNER_START_IDX,
    empty_spaces_mask,
)

RESOURCE_NONZERO_IDX = 2
HP_IDX = 0
REGION_TYPES: Tuple[str, ...] = (
    "resource",
    "friendly",
    "enemy",
    "frontline",
    "open",
)


@dataclass
class RegionStateBatch:
    tokens: torch.Tensor  # Float[B, R, F]
    positions: torch.Tensor  # Float[B, R, 2], y/x in [-1, 1]
    key_padding_mask: torch.Tensor  # Bool[B, R]
    region_names: Sequence[str] = REGION_TYPES

    def to(self, device: torch.device) -> "RegionStateBatch":
        return RegionStateBatch(
            tokens=self.tokens.to(device),
            positions=self.positions.to(device),
            key_padding_mask=self.key_padding_mask.to(device),
            region_names=self.region_names,
        )


def extract_region_state_batch(obs: torch.Tensor) -> RegionStateBatch:
    """Build a small fixed set of heuristic strategic region tokens.

    The tokens are intentionally simple and deterministic. They give the policy a
    stable region-level interface without committing to a learned segmentation model.
    """

    if obs.ndim == 3:
        obs = obs.unsqueeze(0)
    if obs.ndim != 4:
        raise ValueError(
            f"Expected CHW or BCHW observation, got shape {tuple(obs.shape)}"
        )

    batch_size, channels, height, width = obs.shape
    dtype = obs.dtype
    device = obs.device
    flat = obs.flatten(2)
    cell_features = flat.permute(0, 2, 1)

    in_bounds = (
        flat[:, IN_BOUNDS_IDX].bool()
        if channels > IN_BOUNDS_IDX
        else torch.ones(batch_size, height * width, dtype=torch.bool, device=device)
    )
    empty = empty_spaces_mask(cell_features)
    resource = (
        flat[:, RESOURCE_NONZERO_IDX] > 0
        if channels > RESOURCE_NONZERO_IDX
        else torch.zeros_like(in_bounds)
    ) & in_bounds

    owner_planes = (
        flat[:, OWNER_START_IDX:OWNER_END_IDX]
        if channels >= OWNER_END_IDX
        else torch.zeros(batch_size, 3, height * width, dtype=dtype, device=device)
    )
    friendly = owner_planes[:, 1].bool() & ~empty & in_bounds
    enemy = owner_planes[:, 2].bool() & ~empty & in_bounds
    frontline = _frontline_mask(friendly, enemy, height, width) & in_bounds
    open_area = in_bounds & empty & ~resource

    masks = (resource, friendly, enemy, frontline, open_area)
    y_pos, x_pos = _normalized_grid(height, width, device, dtype)
    tokens = []
    positions = []
    key_padding = []
    for region_idx, mask in enumerate(masks):
        summary, center, is_empty = _summarize_region(
            obs,
            mask,
            y_pos,
            x_pos,
            region_idx,
            len(REGION_TYPES),
        )
        tokens.append(summary)
        positions.append(center)
        key_padding.append(is_empty)

    return RegionStateBatch(
        tokens=torch.stack(tokens, dim=1),
        positions=torch.stack(positions, dim=1),
        key_padding_mask=torch.stack(key_padding, dim=1),
    )


def _normalized_grid(
    height: int, width: int, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(height, device=device, dtype=dtype).repeat_interleave(width)
    x = torch.arange(width, device=device, dtype=dtype).repeat(height)
    if height > 1:
        y = y / (height - 1) * 2 - 1
    else:
        y = torch.zeros_like(y)
    if width > 1:
        x = x / (width - 1) * 2 - 1
    else:
        x = torch.zeros_like(x)
    return y, x


def _frontline_mask(
    friendly: torch.Tensor, enemy: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    if not friendly.any() or not enemy.any():
        return torch.zeros_like(friendly)
    friendly_grid = friendly.view(-1, 1, height, width).float()
    enemy_grid = enemy.view(-1, 1, height, width).float()
    friendly_near = torch.nn.functional.max_pool2d(
        friendly_grid, kernel_size=5, stride=1, padding=2
    ).bool()
    enemy_near = torch.nn.functional.max_pool2d(
        enemy_grid, kernel_size=5, stride=1, padding=2
    ).bool()
    return (friendly_near & enemy_near).view(friendly.shape)


def _summarize_region(
    obs: torch.Tensor,
    mask: torch.Tensor,
    y_pos: torch.Tensor,
    x_pos: torch.Tensor,
    region_idx: int,
    num_region_types: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, channels, height, width = obs.shape
    flat = obs.flatten(2)
    weights = mask.to(dtype=obs.dtype)
    count = weights.sum(dim=1).clamp_min(1)
    is_empty = weights.sum(dim=1) == 0

    center_y = (weights * y_pos).sum(dim=1) / count
    center_x = (weights * x_pos).sum(dim=1) / count
    hp_mean = (
        (weights * flat[:, HP_IDX]).sum(dim=1) / count
        if channels > HP_IDX
        else torch.zeros(batch_size, dtype=obs.dtype, device=obs.device)
    )
    resource_mean = (
        (weights * flat[:, RESOURCE_NONZERO_IDX]).sum(dim=1) / count
        if channels > RESOURCE_NONZERO_IDX
        else torch.zeros(batch_size, dtype=obs.dtype, device=obs.device)
    )
    coverage = weights.sum(dim=1) / max(height * width, 1)

    type_one_hot = torch.zeros(
        batch_size, num_region_types, dtype=obs.dtype, device=obs.device
    )
    type_one_hot[:, region_idx] = 1
    dense_features = torch.stack(
        (center_y, center_x, coverage, hp_mean, resource_mean), dim=1
    )
    center = torch.stack((center_y, center_x), dim=1)
    token = torch.cat((type_one_hot, dense_features), dim=1)
    token = torch.where(is_empty.view(-1, 1), torch.zeros_like(token), token)
    center = torch.where(is_empty.view(-1, 1), torch.zeros_like(center), center)
    return token, center, is_empty
