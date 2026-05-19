from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn


class StrategicMemory(nn.Module, ABC):
    """Low-frequency strategic state interface for RTS policies."""

    output_dim: int

    @abstractmethod
    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor: ...

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
        episode_starts: Optional[torch.Tensor] = None,
        update_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...


class IdentityStrategicMemory(StrategicMemory):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.output_dim = input_dim

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.output_dim, device=device)

    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
        episode_starts: Optional[torch.Tensor] = None,
        update_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del memory_state, episode_starts, update_mask
        return x, x


class GRUStrategicMemory(StrategicMemory):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.output_dim = hidden_dim
        self.cell = nn.GRUCell(input_dim, hidden_dim)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.output_dim, device=device)

    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
        episode_starts: Optional[torch.Tensor] = None,
        update_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if memory_state is None:
            memory_state = self.initial_state(x.size(0), x.device)
        if episode_starts is not None:
            reset = episode_starts.to(device=x.device, dtype=torch.bool).view(-1, 1)
            memory_state = torch.where(
                reset, torch.zeros_like(memory_state), memory_state
            )

        next_state = self.cell(x, memory_state)
        if update_mask is not None:
            update = update_mask.to(device=x.device, dtype=torch.bool).view(-1, 1)
            next_state = torch.where(update, next_state, memory_state)
        return next_state, next_state


def build_strategic_memory(
    kind: str,
    input_dim: int,
    hidden_dim: Optional[int] = None,
) -> StrategicMemory:
    if kind in ("none", "identity"):
        return IdentityStrategicMemory(input_dim)
    if kind == "gru":
        return GRUStrategicMemory(input_dim, hidden_dim or input_dim)
    if kind == "mamba_ssm":
        try:
            import mamba_ssm  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "memory_kwargs.kind='mamba_ssm' requires the optional mamba_ssm package"
            ) from exc
        raise NotImplementedError(
            "mamba_ssm is dependency-gated; wire a concrete block after the GRU memory "
            "interface has been validated."
        )
    raise ValueError(f"Unsupported strategic memory kind: {kind}")
