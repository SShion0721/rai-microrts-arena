from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

import numpy as np


@dataclass(frozen=True)
class ReplayMetadata:
    """Metadata that makes an offline microRTS replay auditable."""

    map_path: str = ""
    map_hash: str = ""
    learner: str = ""
    opponent: str = ""
    policy_id: str = ""
    winner: Optional[int] = None
    game_length: Optional[int] = None
    phase_tags: List[str] = field(default_factory=list)
    event_tags: List[str] = field(default_factory=list)
    has_action_masks: Optional[bool] = None
    seed: Optional[int] = None
    source: str = ""
    schema_version: int = 2
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OfflineTransition:
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    done: bool
    action_mask: Optional[np.ndarray] = None
    next_obs: Optional[np.ndarray] = None
    info: Dict[str, Any] = field(default_factory=dict)


class OfflineReplay:
    """A small, explicit npz-backed trajectory format for BC/pretraining.

    The format intentionally keeps observations, masks and actions as arrays so it can
    be consumed by ACBC, GraphDINO/MAE, or Decision Transformer data loaders without
    replaying Java environments.
    """

    def __init__(
        self,
        metadata: Optional[ReplayMetadata] = None,
        transitions: Optional[Iterable[OfflineTransition]] = None,
    ) -> None:
        self.metadata = metadata or ReplayMetadata()
        self.transitions: List[OfflineTransition] = list(transitions or [])

    def append(self, transition: OfflineTransition) -> None:
        self.transitions.append(transition)

    def extend(self, transitions: Iterable[OfflineTransition]) -> None:
        self.transitions.extend(transitions)

    def __len__(self) -> int:
        return len(self.transitions)

    def save_npz(self, path: Union[str, Path]) -> None:
        if not self.transitions:
            raise ValueError("Cannot save an empty OfflineReplay")

        obs = np.stack([t.obs for t in self.transitions])
        actions = np.stack([t.action for t in self.transitions])
        rewards = np.stack([np.asarray(t.reward) for t in self.transitions])
        dones = np.asarray([t.done for t in self.transitions], dtype=np.bool_)
        infos = np.asarray(
            [json.dumps(t.info, sort_keys=True) for t in self.transitions]
        )

        payload: Dict[str, Any] = {
            "metadata": np.asarray(json.dumps(asdict(self.metadata), sort_keys=True)),
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "infos": infos,
        }

        if all(t.action_mask is not None for t in self.transitions):
            payload["action_masks"] = np.stack(
                [np.asarray(t.action_mask) for t in self.transitions]
            )
        if all(t.next_obs is not None for t in self.transitions):
            payload["next_obs"] = np.stack(
                [np.asarray(t.next_obs) for t in self.transitions]
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "OfflineReplay":
        data = np.load(path, allow_pickle=False)
        metadata = ReplayMetadata(**json.loads(str(data["metadata"])))
        action_masks = data["action_masks"] if "action_masks" in data else None
        next_obs = data["next_obs"] if "next_obs" in data else None
        infos = [json.loads(str(info)) for info in data["infos"]]

        transitions = []
        for idx in range(len(data["obs"])):
            transitions.append(
                OfflineTransition(
                    obs=data["obs"][idx],
                    action=data["actions"][idx],
                    reward=data["rewards"][idx],
                    done=bool(data["dones"][idx]),
                    action_mask=action_masks[idx] if action_masks is not None else None,
                    next_obs=next_obs[idx] if next_obs is not None else None,
                    info=infos[idx],
                )
            )
        return cls(metadata=metadata, transitions=transitions)

    def summary(self) -> Mapping[str, Any]:
        rewards = np.stack([np.asarray(t.reward) for t in self.transitions])
        return {
            "num_transitions": len(self.transitions),
            "obs_shape": (
                tuple(self.transitions[0].obs.shape) if self.transitions else ()
            ),
            "action_shape": (
                tuple(self.transitions[0].action.shape) if self.transitions else ()
            ),
            "reward_mean": rewards.mean(axis=0).tolist() if len(rewards) else [],
            "done_count": int(sum(t.done for t in self.transitions)),
            "metadata": asdict(self.metadata),
        }
