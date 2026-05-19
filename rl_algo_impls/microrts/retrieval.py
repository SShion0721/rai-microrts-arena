from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class RetrievalKey:
    map_topology: np.ndarray
    resource_phase: np.ndarray
    army_composition: np.ndarray
    region_control: np.ndarray

    def vector(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(self.map_topology, dtype=np.float32).ravel(),
                np.asarray(self.resource_phase, dtype=np.float32).ravel(),
                np.asarray(self.army_composition, dtype=np.float32).ravel(),
                np.asarray(self.region_control, dtype=np.float32).ravel(),
            ]
        )


@dataclass
class RetrievalEntry:
    key: RetrievalKey
    strategic_summary: np.ndarray
    score: float = 0.0


class EpisodicRetrievalStore:
    """Small evaluation-time retrieval store for strategic summaries, not actions."""

    def __init__(self, entries: Sequence[RetrievalEntry] = ()) -> None:
        self.entries: List[RetrievalEntry] = list(entries)

    def add(self, entry: RetrievalEntry) -> None:
        self.entries.append(entry)

    def query(self, key: RetrievalKey, k: int = 1) -> List[RetrievalEntry]:
        if not self.entries or k <= 0:
            return []
        query = key.vector()
        scored = []
        for entry in self.entries:
            candidate = entry.key.vector()
            denom = np.linalg.norm(query) * np.linalg.norm(candidate)
            similarity = 0.0 if denom == 0 else float(query @ candidate / denom)
            scored.append((similarity, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:k]]
