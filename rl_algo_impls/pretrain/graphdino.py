from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_algo_impls.shared.module.utils import mlp


@dataclass(frozen=True)
class GraphDINOConfig:
    input_dim: int
    embed_dim: int = 128
    projection_dim: int = 256
    hidden_dim: int = 256
    layers: int = 3
    attention_heads: int = 4
    teacher_momentum: float = 0.996
    temperature: float = 0.1
    entity_reconstruction_weight: float = 0.0
    region_reconstruction_weight: float = 0.0
    temporal_contrastive_weight: float = 0.0


class GraphDINOEncoder(nn.Module):
    """Transformer encoder for padded entity-state batches.

    It is graph-ready: edge features can be added by a future message-passing block,
    while the current implementation already gives GraphDINO/MAE a stable interface.
    """

    def __init__(self, config: GraphDINOConfig) -> None:
        super().__init__()
        self.config = config
        self.node_encoder = mlp(
            [config.input_dim, config.hidden_dim, config.embed_dim],
            nn.GELU,
            output_activation=nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.hidden_dim,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.layers)
        self.projector = mlp(
            [config.embed_dim, config.hidden_dim, config.projection_dim],
            nn.GELU,
        )
        self.reconstruction_head = mlp(
            [config.embed_dim, config.hidden_dim, config.input_dim],
            nn.GELU,
        )

    def forward(
        self, nodes: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.node_encoder(nodes)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        pooled = masked_mean(x, key_padding_mask)
        return x, self.projector(pooled)

    def reconstruct(self, encoded_nodes: torch.Tensor) -> torch.Tensor:
        return self.reconstruction_head(encoded_nodes)


class GraphDINOLoss(nn.Module):
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, student_projection: torch.Tensor, teacher_projection: torch.Tensor
    ) -> torch.Tensor:
        student_logp = F.log_softmax(student_projection / self.temperature, dim=-1)
        teacher_p = F.softmax(teacher_projection.detach() / self.temperature, dim=-1)
        return -(teacher_p * student_logp).sum(dim=-1).mean()


class GraphDINOPair(nn.Module):
    """Student/teacher pair with EMA teacher updates."""

    def __init__(self, config: GraphDINOConfig) -> None:
        super().__init__()
        self.config = config
        self.student = GraphDINOEncoder(config)
        self.teacher = GraphDINOEncoder(config)
        self.teacher.load_state_dict(self.student.state_dict())
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.loss = GraphDINOLoss(config.temperature)

    def forward(
        self,
        student_nodes: torch.Tensor,
        teacher_nodes: torch.Tensor,
        student_key_padding_mask: Optional[torch.Tensor] = None,
        teacher_key_padding_mask: Optional[torch.Tensor] = None,
        reconstruction_targets: Optional[torch.Tensor] = None,
        reconstruction_mask: Optional[torch.Tensor] = None,
        next_student_projection: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        student_encoded, student_projection = self.student(
            student_nodes, student_key_padding_mask
        )
        with torch.no_grad():
            _, teacher_projection = self.teacher(
                teacher_nodes, teacher_key_padding_mask
            )
        loss = self.loss(student_projection, teacher_projection)
        if (
            reconstruction_targets is not None
            and reconstruction_mask is not None
            and self.config.entity_reconstruction_weight > 0
        ):
            reconstructed = self.student.reconstruct(student_encoded)
            mask = reconstruction_mask.unsqueeze(-1).to(dtype=torch.bool)
            if mask.any():
                loss = loss + self.config.entity_reconstruction_weight * F.mse_loss(
                    reconstructed[mask.expand_as(reconstructed)],
                    reconstruction_targets[mask.expand_as(reconstruction_targets)],
                )
        if (
            next_student_projection is not None
            and self.config.temporal_contrastive_weight > 0
        ):
            loss = loss + self.config.temporal_contrastive_weight * (
                1
                - F.cosine_similarity(
                    student_projection, next_student_projection.detach(), dim=-1
                ).mean()
            )
        return loss

    @torch.no_grad()
    def update_teacher(self) -> None:
        momentum = self.config.teacher_momentum
        for teacher_parameter, student_parameter in zip(
            self.teacher.parameters(), self.student.parameters()
        ):
            teacher_parameter.mul_(momentum).add_(
                student_parameter.detach(), alpha=1.0 - momentum
            )


def masked_mean(
    x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    if key_padding_mask is None:
        return x.mean(dim=1)
    valid = (~key_padding_mask).unsqueeze(-1).to(dtype=x.dtype)
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
