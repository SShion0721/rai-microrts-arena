import os
from abc import ABC, abstractmethod
from typing import Any, NamedTuple, Optional, Sequence

import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete, Space

from rl_algo_impls.shared.actor import PiForward
from rl_algo_impls.shared.tensor_utils import TensorOrDict

MemoryState = Any


class ACNForward(NamedTuple):
    pi_forward: PiForward
    v: torch.Tensor
    next_memory_state: Optional[MemoryState] = None


class ActorCriticNetwork(nn.Module, ABC):
    uses_memory: bool = False

    def forward(
        self,
        obs: torch.Tensor,
        action: TensorOrDict,
        action_masks: Optional[TensorOrDict] = None,
        memory_state: Optional[MemoryState] = None,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> ACNForward:
        return self.distribution_and_value(
            obs,
            action=action,
            action_masks=action_masks,
            memory_state=memory_state,
            episode_starts=episode_starts,
        )

    def distribution_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[TensorOrDict] = None,
        action_masks: Optional[TensorOrDict] = None,
        memory_state: Optional[MemoryState] = None,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> ACNForward:
        if memory_state is not None or episode_starts is not None:
            return self._distribution_and_value_with_memory(
                obs,
                action=action,
                action_masks=action_masks,
                memory_state=memory_state,
                episode_starts=episode_starts,
            )
        return self._distribution_and_value(
            obs, action=action, action_masks=action_masks
        )

    @abstractmethod
    def _distribution_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[TensorOrDict] = None,
        action_masks: Optional[TensorOrDict] = None,
    ) -> ACNForward: ...

    def _distribution_and_value_with_memory(
        self,
        obs: torch.Tensor,
        action: Optional[TensorOrDict] = None,
        action_masks: Optional[TensorOrDict] = None,
        memory_state: Optional[MemoryState] = None,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> ACNForward:
        return self._distribution_and_value(
            obs, action=action, action_masks=action_masks
        )

    def initial_memory_state(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> Optional[MemoryState]:
        return None

    @abstractmethod
    def value(self, obs: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def reset_noise(self, batch_size: Optional[int] = None) -> None: ...

    @abstractmethod
    def freeze(
        self,
        freeze_policy_head: bool,
        freeze_value_head: bool,
        freeze_backbone: bool = True,
    ) -> None: ...

    def unfreeze(self):
        self.freeze(False, False, freeze_backbone=False)


def default_hidden_sizes(obs_space: Space) -> Sequence[int]:
    if isinstance(obs_space, Box):
        if len(obs_space.shape) == 3:  # type: ignore
            # By default feature extractor to output has no hidden layers
            return []
        elif len(obs_space.shape) == 1:  # type: ignore
            return [64, 64]
        else:
            raise ValueError(f"Unsupported observation space: {obs_space}")
    elif isinstance(obs_space, Discrete):
        return [64]
    else:
        raise ValueError(f"Unsupported observation space: {obs_space}")
