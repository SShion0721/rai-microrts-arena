from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    include_checkpoints: bool = False

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


@dataclass(frozen=True)
class EvaluationMetrics:
    win_rate: WinRateEstimate
    inference_ms_per_step: Optional[float] = None
    env_steps_per_second: Optional[float] = None
    gpu_utilization: Optional[float] = None
    policy_lag: Optional[float] = None
    invalid_action_rate: Optional[float] = None
    extras: Dict[str, float] = field(default_factory=dict)

    def dashboard_row(self) -> Dict[str, float]:
        low, high = self.win_rate.wilson_interval()
        row: Dict[str, float] = {
            "games": float(self.win_rate.games),
            "score_rate": self.win_rate.score_rate,
            "wilson_low": low,
            "wilson_high": high,
        }
        for key, value in {
            "inference_ms_per_step": self.inference_ms_per_step,
            "env_steps_per_second": self.env_steps_per_second,
            "gpu_utilization": self.gpu_utilization,
            "policy_lag": self.policy_lag,
            "invalid_action_rate": self.invalid_action_rate,
        }.items():
            if value is not None:
                row[key] = value
        row.update(self.extras)
        return row


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
