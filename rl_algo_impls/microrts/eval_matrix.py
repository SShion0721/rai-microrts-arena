from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


DEFAULT_PUBLIC_MAPS: Sequence[str] = (
    "maps/8x8/basesWorkers8x8A.xml",
    "maps/16x16/basesWorkers16x16A.xml",
    "maps/16x16/TwoBasesBarracks16x16.xml",
    "maps/DoubleGame24x24.xml",
    "maps/BWDistantResources32x32.xml",
    "maps/BroodWar/(4)BloodBath.scmB.xml",
    "maps/GardenOfWar64x64.xml",
)

DEFAULT_OPPONENTS: Sequence[str] = (
    "WorkerRush",
    "LightRush",
    "CoacAI",
    "Mayari",
    "TMA",
)


@dataclass(frozen=True)
class EvaluationSpec:
    maps: Sequence[str] = DEFAULT_PUBLIC_MAPS
    opponents: Sequence[str] = DEFAULT_OPPONENTS
    games_per_side: int = 50
    time_budget_ms: int = 100
    deterministic: bool = True

    @property
    def total_games(self) -> int:
        return len(self.maps) * len(self.opponents) * self.games_per_side * 2


@dataclass(frozen=True)
class WinRateEstimate:
    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games

    def wilson_interval(self, z: float = 1.96) -> Tuple[float, float]:
        n = self.games
        if n == 0:
            return (0.0, 0.0)
        p = self.score_rate
        denom = 1 + z**2 / n
        centre = p + z**2 / (2 * n)
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
        return ((centre - margin) / denom, (centre + margin) / denom)


def build_eval_jobs(spec: EvaluationSpec) -> List[dict]:
    jobs = []
    for map_path in spec.maps:
        for opponent in spec.opponents:
            for player_id in (0, 1):
                jobs.append(
                    {
                        "map_path": map_path,
                        "opponent": opponent,
                        "player_id": player_id,
                        "games": spec.games_per_side,
                        "time_budget_ms": spec.time_budget_ms,
                        "deterministic": spec.deterministic,
                    }
                )
    return jobs


def aggregate_estimates(results: Iterable[WinRateEstimate]) -> WinRateEstimate:
    wins = draws = losses = 0
    for result in results:
        wins += result.wins
        draws += result.draws
        losses += result.losses
    return WinRateEstimate(wins=wins, draws=draws, losses=losses)
