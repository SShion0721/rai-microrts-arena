from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MacroWorldModelOutput:
    resource_phase: torch.Tensor
    army_composition: torch.Tensor
    region_control: torch.Tensor
    engagement_window: torch.Tensor


class MacroWorldModel(nn.Module):
    """Latent macro predictor for future planning experiments.

    This intentionally predicts strategic variables rather than primitive actions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        resource_dim: int,
        army_dim: int,
        region_dim: int,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.resource_head = nn.Linear(hidden_dim, resource_dim)
        self.army_head = nn.Linear(hidden_dim, army_dim)
        self.region_head = nn.Linear(hidden_dim, region_dim)
        self.engagement_head = nn.Linear(hidden_dim, 1)

    def forward(self, strategic_state: torch.Tensor) -> MacroWorldModelOutput:
        x = self.trunk(strategic_state)
        return MacroWorldModelOutput(
            resource_phase=self.resource_head(x),
            army_composition=self.army_head(x),
            region_control=self.region_head(x),
            engagement_window=torch.sigmoid(self.engagement_head(x)).squeeze(-1),
        )

    def loss(
        self,
        output: MacroWorldModelOutput,
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return (
            nn.functional.mse_loss(output.resource_phase, targets["resource_phase"])
            + nn.functional.mse_loss(
                output.army_composition, targets["army_composition"]
            )
            + nn.functional.mse_loss(output.region_control, targets["region_control"])
            + nn.functional.binary_cross_entropy(
                output.engagement_window, targets["engagement_window"].float()
            )
        )
