from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box, MultiDiscrete, Space
from gymnasium.spaces import Dict as DictSpace

from rl_algo_impls.microrts.entity_state import (
    EntityStateBatch,
    extract_entity_state_batch,
)
from rl_algo_impls.microrts.region_state import REGION_TYPES, extract_region_state_batch
from rl_algo_impls.shared.actor import pi_forward
from rl_algo_impls.shared.actor.gridnet import GridnetDistribution, ValueDependentMask
from rl_algo_impls.shared.actor.gridnet_decoder import Transpose
from rl_algo_impls.shared.module.channelwise_activation import ChannelwiseActivation
from rl_algo_impls.shared.module.utils import layer_init, mlp
from rl_algo_impls.shared.policy.actor_critic_network.grid2entity_transformer import (
    TransformerEncoderBackbone,
)
from rl_algo_impls.shared.policy.actor_critic_network.network import (
    ACNForward,
    ActorCriticNetwork,
)
from rl_algo_impls.shared.policy.actor_critic_network.squeeze_unet import (
    SqueezeUnetBackbone,
)
from rl_algo_impls.shared.policy.memory import build_strategic_memory
from rl_algo_impls.shared.policy.policy import ACTIVATION
from rl_algo_impls.shared.tensor_utils import TensorOrDict


class HierarchicalHybridEntityGridNetwork(ActorCriticNetwork):
    """SquNet tactical actor with entity graph, region tokens, and strategic memory."""

    uses_memory = True

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        action_plane_space: Space,
        init_layers_orthogonal: bool = True,
        cnn_layers_init_orthogonal: Optional[bool] = None,
        channels_per_level: Optional[List[int]] = None,
        strides_per_level: Optional[List[Union[int, List[int]]]] = None,
        deconv_strides_per_level: Optional[List[Union[int, List[int]]]] = None,
        encoder_residual_blocks_per_level: Optional[List[int]] = None,
        decoder_residual_blocks_per_level: Optional[List[int]] = None,
        increment_kernel_size_on_down_conv: bool = False,
        encoder_embed_dim: int = 128,
        encoder_attention_heads: int = 4,
        encoder_feed_forward_dim: int = 256,
        encoder_layers: int = 2,
        hidden_critic_dims: Optional[List[int]] = None,
        num_additional_critics: int = 0,
        additional_critic_activation_functions: Optional[List[str]] = None,
        output_activation_fn: str = "identity",
        subaction_mask: Optional[Dict[int, Dict[int, int]]] = None,
        normalization: str = "layer",
        actor_head_kernel_size: int = 3,
        value_output_gain: float = 1.0,
        memory_kwargs: Optional[Dict[str, Any]] = None,
        region_tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        hierarchical_action_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        del region_tokenizer_kwargs
        assert isinstance(observation_space, Box)
        assert isinstance(action_plane_space, MultiDiscrete)

        if cnn_layers_init_orthogonal is None:
            cnn_layers_init_orthogonal = False
        if channels_per_level is None:
            channels_per_level = [64, 128, 256]
        if strides_per_level is None:
            strides_per_level = [2] * (len(channels_per_level) - 1)
        if encoder_residual_blocks_per_level is None:
            encoder_residual_blocks_per_level = [1] * len(channels_per_level)
        if decoder_residual_blocks_per_level is None:
            decoder_residual_blocks_per_level = encoder_residual_blocks_per_level[:-1]
        if hidden_critic_dims is None:
            hidden_critic_dims = []
        if num_additional_critics and not additional_critic_activation_functions:
            additional_critic_activation_functions = [
                "identity"
            ] * num_additional_critics

        memory_kwargs = memory_kwargs or {}
        hierarchical_action_kwargs = hierarchical_action_kwargs or {}

        self.range_size = np.max(observation_space.high) - np.min(observation_space.low)
        self.action_vec = action_plane_space.nvec
        self.subaction_mask = (
            ValueDependentMask.from_reference_index_to_index_to_value(subaction_mask)
            if subaction_mask
            else None
        )
        self.grid_channels = channels_per_level[0]
        self.encoder_embed_dim = encoder_embed_dim
        self.entity_edge_radius = memory_kwargs.get("entity_edge_radius", 2.0)
        input_channels = observation_space.shape[0]  # type: ignore

        if isinstance(action_space, DictSpace):
            pick_position_space = action_space["pick_position"]
            assert isinstance(pick_position_space, MultiDiscrete)
            self.pick_vec = pick_position_space.nvec
        elif isinstance(action_space, MultiDiscrete):
            self.pick_vec = None
        else:
            raise ValueError(
                f"Unsupported action space {action_space.__class__.__name__}"
            )

        self.grid_backbone = SqueezeUnetBackbone(
            input_channels,
            channels_per_level,
            strides_per_level,
            encoder_residual_blocks_per_level,
            decoder_residual_blocks_per_level,
            deconv_strides_per_level=deconv_strides_per_level,
            init_layers_orthogonal=cnn_layers_init_orthogonal,
            increment_kernel_size_on_down_conv=increment_kernel_size_on_down_conv,
            normalization=normalization,
        )
        self.entity_embedding = mlp(
            [input_channels + 2, encoder_feed_forward_dim, encoder_embed_dim],
            nn.GELU,
            output_activation=nn.GELU(),
            init_layers_orthogonal=init_layers_orthogonal,
            final_normalization=normalization,
        )
        self.entity_encoder = TransformerEncoderBackbone(
            encoder_embed_dim,
            encoder_attention_heads,
            encoder_feed_forward_dim,
            encoder_layers,
            normalization=normalization,
        )
        self.edge_embedding = mlp(
            [9, encoder_feed_forward_dim, encoder_embed_dim],
            nn.GELU,
            output_activation=nn.GELU(),
            init_layers_orthogonal=init_layers_orthogonal,
            final_normalization=normalization,
        )
        self.region_embedding = mlp(
            [len(REGION_TYPES) + 5, encoder_feed_forward_dim, encoder_embed_dim],
            nn.GELU,
            output_activation=nn.GELU(),
            init_layers_orthogonal=init_layers_orthogonal,
            final_normalization=normalization,
        )
        self.region_encoder = TransformerEncoderBackbone(
            encoder_embed_dim,
            encoder_attention_heads,
            encoder_feed_forward_dim,
            max(1, min(encoder_layers, 2)),
            normalization=normalization,
        )

        strategic_input_dim = encoder_embed_dim * 2
        memory_kind = memory_kwargs.get("kind", "gru")
        memory_hidden_dim = memory_kwargs.get("hidden_dim", strategic_input_dim)
        self.strategic_memory = build_strategic_memory(
            memory_kind, strategic_input_dim, memory_hidden_dim
        )
        self.film = mlp(
            [
                self.strategic_memory.output_dim,
                encoder_feed_forward_dim,
                self.grid_channels * 2,
            ],
            nn.GELU,
            init_layers_orthogonal=init_layers_orthogonal,
        )

        actor_head_padding = (actor_head_kernel_size - 1) // 2
        self.actor_head = nn.Sequential(
            layer_init(
                nn.Conv2d(
                    self.grid_channels,
                    self.action_vec.sum()
                    + (len(self.pick_vec) if self.pick_vec else 0),
                    kernel_size=actor_head_kernel_size,
                    padding=actor_head_padding,
                ),
                init_layers_orthogonal=init_layers_orthogonal,
                std=0.01,
            ),
            Transpose((0, 2, 3, 1)),
        )

        strategy_latent_dim = hierarchical_action_kwargs.get("strategy_latent_dim", 8)
        num_groups = hierarchical_action_kwargs.get("num_groups", 8)
        self.strategy_head = mlp(
            [
                self.strategic_memory.output_dim,
                encoder_feed_forward_dim,
                strategy_latent_dim,
            ],
            nn.GELU,
            init_layers_orthogonal=init_layers_orthogonal,
        )
        self.group_assignment_head = mlp(
            [encoder_embed_dim, encoder_feed_forward_dim, num_groups],
            nn.GELU,
            init_layers_orthogonal=init_layers_orthogonal,
        )
        self.region_intent_head = mlp(
            [
                self.strategic_memory.output_dim,
                encoder_feed_forward_dim,
                len(REGION_TYPES),
            ],
            nn.GELU,
            init_layers_orthogonal=init_layers_orthogonal,
        )

        output_activations = [
            ACTIVATION[act_fn_name]()
            for act_fn_name in [output_activation_fn]
            + (additional_critic_activation_functions or [])
        ]
        critic_input_dim = self.grid_channels * 2 + self.strategic_memory.output_dim
        self.critic_head = mlp(
            [critic_input_dim, *hidden_critic_dims, len(output_activations)],
            nn.GELU,
            output_activation=ChannelwiseActivation(output_activations),
            init_layers_orthogonal=init_layers_orthogonal,
            final_layer_gain=value_output_gain,
        )

    def initial_memory_state(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return self.strategic_memory.initial_state(batch_size, device)

    def _preprocess(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        return obs.float() / self.range_size

    def _distribution_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[TensorOrDict] = None,
        action_masks: Optional[TensorOrDict] = None,
    ) -> ACNForward:
        return self._distribution_and_value_with_memory(
            obs, action=action, action_masks=action_masks
        )

    def _distribution_and_value_with_memory(
        self,
        obs: torch.Tensor,
        action: Optional[TensorOrDict] = None,
        action_masks: Optional[TensorOrDict] = None,
        memory_state: Optional[torch.Tensor] = None,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> ACNForward:
        assert (
            action_masks is not None
        ), f"No mask case unhandled in {type(self).__name__}"
        grid_features, strategic_context, next_memory_state, _ = self._features(
            obs, memory_state=memory_state, episode_starts=episode_starts
        )
        logits = self.actor_head(grid_features)
        pi = GridnetDistribution(
            int(np.prod(grid_features.shape[-2:])),
            self.action_vec,
            logits,
            action_masks,
            subaction_mask=self.subaction_mask,
        )
        v = self._value_from_features(grid_features, strategic_context)
        return ACNForward(pi_forward(pi, action), v, next_memory_state)

    def value(
        self,
        obs: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        grid_features, strategic_context, _, _ = self._features(
            obs, memory_state=memory_state, episode_starts=episode_starts
        )
        return self._value_from_features(grid_features, strategic_context)

    def auxiliary_predictions(
        self,
        obs: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        _, strategic_context, _, entity_context = self._features(
            obs, memory_state=memory_state, episode_starts=episode_starts
        )
        return {
            "strategy_latent": self.strategy_head(strategic_context),
            "group_logits": self.group_assignment_head(entity_context),
            "region_intent_logits": self.region_intent_head(strategic_context),
        }

    def reset_noise(self, batch_size: Optional[int] = None) -> None:
        pass

    def freeze(
        self,
        freeze_policy_head: bool,
        freeze_value_head: bool,
        freeze_backbone: bool = True,
    ) -> None:
        backbone_modules = (
            self.grid_backbone,
            self.entity_embedding,
            self.entity_encoder,
            self.edge_embedding,
            self.region_embedding,
            self.region_encoder,
            self.strategic_memory,
            self.film,
        )
        for module in backbone_modules:
            for parameter in module.parameters():
                parameter.requires_grad = not freeze_backbone
        for module in (
            self.actor_head,
            self.strategy_head,
            self.group_assignment_head,
            self.region_intent_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = not freeze_policy_head
        for parameter in self.critic_head.parameters():
            parameter.requires_grad = not freeze_value_head

    def _features(
        self,
        obs: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        processed = self._preprocess(obs)
        grid_features = self.grid_backbone(processed)
        entity_context = self._entity_context(processed)
        region_context = self._region_context(processed)
        strategic_input = torch.cat((entity_context, region_context), dim=-1)
        strategic_context, next_memory_state = self.strategic_memory(
            strategic_input,
            memory_state=memory_state,
            episode_starts=episode_starts,
        )
        gamma_beta = self.film(strategic_context).view(
            processed.size(0), 2, self.grid_channels
        )
        gamma = gamma_beta[:, 0, :, None, None]
        beta = gamma_beta[:, 1, :, None, None]
        return (
            grid_features * (1 + gamma) + beta,
            strategic_context,
            next_memory_state,
            entity_context,
        )

    def _entity_context(self, obs: torch.Tensor) -> torch.Tensor:
        entity_batch = extract_entity_state_batch(
            obs, add_position_features=True, edge_radius=self.entity_edge_radius
        )
        if entity_batch.nodes.size(1) == 0:
            return torch.zeros(
                obs.size(0),
                self.encoder_embed_dim,
                dtype=obs.dtype,
                device=obs.device,
            )
        x = self.entity_embedding(entity_batch.nodes)
        x = self.entity_encoder(x, key_padding_mask=entity_batch.key_padding_mask)
        valid = (~entity_batch.key_padding_mask).unsqueeze(-1).to(dtype=x.dtype)
        entity_context = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return entity_context + self._edge_context(entity_batch, obs.size(0))

    def _edge_context(
        self, entity_batch: EntityStateBatch, batch_size: int
    ) -> torch.Tensor:
        if entity_batch.edge_attr is None or entity_batch.edge_attr.numel() == 0:
            return torch.zeros(
                batch_size,
                self.encoder_embed_dim,
                dtype=entity_batch.nodes.dtype,
                device=entity_batch.nodes.device,
            )
        batch_idx = entity_batch.edge_attr[:, 0].long()
        edge_embeddings = self.edge_embedding(entity_batch.edge_attr)
        edge_context = torch.zeros(
            batch_size,
            self.encoder_embed_dim,
            dtype=edge_embeddings.dtype,
            device=edge_embeddings.device,
        )
        edge_context.index_add_(0, batch_idx, edge_embeddings)
        counts = torch.bincount(batch_idx, minlength=batch_size).to(
            edge_embeddings.dtype
        )
        return edge_context / counts.unsqueeze(1).clamp_min(1)

    def _region_context(self, obs: torch.Tensor) -> torch.Tensor:
        region_batch = extract_region_state_batch(obs)
        if region_batch.key_padding_mask.all(dim=1).all():
            return torch.zeros(
                obs.size(0),
                self.encoder_embed_dim,
                dtype=obs.dtype,
                device=obs.device,
            )
        region_batch.key_padding_mask = region_batch.key_padding_mask.clone()
        all_empty = region_batch.key_padding_mask.all(dim=1)
        if all_empty.any():
            region_batch.key_padding_mask[all_empty, 0] = False
        x = self.region_embedding(region_batch.tokens)
        x = self.region_encoder(x, key_padding_mask=region_batch.key_padding_mask)
        valid = (~region_batch.key_padding_mask).unsqueeze(-1).to(dtype=x.dtype)
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    def _value_from_features(
        self, grid_features: torch.Tensor, strategic_context: torch.Tensor
    ) -> torch.Tensor:
        grid_avg = torch.mean(grid_features, dim=(-2, -1))
        grid_max = torch.amax(grid_features, dim=(-2, -1))
        v = self.critic_head(torch.cat((grid_avg, grid_max, strategic_context), dim=-1))
        if v.shape[-1] == 1:
            v = v.squeeze(-1)
        return v
