from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Sequence

import numpy as np


class OptionName(str, Enum):
    HARVEST = "harvest"
    BUILD_BARRACKS = "build_barracks"
    TRAIN_WORKER = "train_worker"
    TRAIN_COMBAT = "train_combat"
    RALLY = "rally"
    RUSH = "rush"
    DEFEND = "defend"
    EXPAND = "expand"
    HARASS = "harass"
    RETREAT = "retreat"
    PRIMITIVE_FALLBACK = "primitive_fallback"


@dataclass(frozen=True)
class OptionAction:
    name: OptionName
    target_position: Optional[int] = None
    unit_position: Optional[int] = None
    unit_type: Optional[int] = None
    priority: float = 1.0


MICRO_RTS_OPTIONS: Sequence[OptionName] = (
    OptionName.HARVEST,
    OptionName.BUILD_BARRACKS,
    OptionName.TRAIN_WORKER,
    OptionName.TRAIN_COMBAT,
    OptionName.RALLY,
    OptionName.RUSH,
    OptionName.DEFEND,
    OptionName.EXPAND,
    OptionName.HARASS,
    OptionName.RETREAT,
    OptionName.PRIMITIVE_FALLBACK,
)


class OptionLibrary:
    """Programmatic option surface that safely falls back to primitive actions.

    This is intentionally conservative: it creates a stable high-level action API
    now, while later patches can specialize each option with domain heuristics or
    learned pointer heads.
    """

    def __init__(self, action_vec: Iterable[int]) -> None:
        self.action_vec = tuple(int(v) for v in action_vec)
        self.option_to_index: Dict[OptionName, int] = {
            option: idx for idx, option in enumerate(MICRO_RTS_OPTIONS)
        }

    def encode(self, option: OptionAction) -> int:
        return self.option_to_index[option.name]

    def legal_fallback_action(self, action_masks: np.ndarray) -> np.ndarray:
        """Return the lexicographically first independently legal primitive action."""

        masks = np.asarray(action_masks, dtype=np.bool_)
        if masks.ndim != 2:
            raise ValueError(f"Expected [positions, action_dim] mask, got {masks.shape}")
        action = np.zeros((masks.shape[0], len(self.action_vec)), dtype=np.int64)
        offset = 0
        for subaction_idx, choices in enumerate(self.action_vec):
            submask = masks[:, offset : offset + choices]
            valid_any = submask.any(axis=1)
            first_valid = np.argmax(submask, axis=1)
            action[:, subaction_idx] = np.where(valid_any, first_valid, 0)
            offset += choices
        return action

    def to_primitive(
        self, option: OptionAction, action_masks: np.ndarray, fallback: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if option.name == OptionName.PRIMITIVE_FALLBACK and fallback is not None:
            return fallback
        return self.legal_fallback_action(action_masks)
