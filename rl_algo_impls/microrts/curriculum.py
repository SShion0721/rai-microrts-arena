from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    maps: Sequence[str]
    sampling_weight: float = 1.0
    min_score_rate: Optional[float] = None
    max_steps: Optional[int] = None


@dataclass
class MapCurriculum:
    stages: List[CurriculumStage] = field(default_factory=list)

    @classmethod
    def default_microrts(cls) -> "MapCurriculum":
        return cls(
            stages=[
                CurriculumStage(
                    "micro_8_12_16",
                    (
                        "maps/8x8/basesWorkers8x8A.xml",
                        "maps/12x12/basesWorkers12x12.xml",
                        "maps/16x16/basesWorkers16x16A.xml",
                        "maps/16x16/TwoBasesBarracks16x16.xml",
                    ),
                    min_score_rate=0.65,
                ),
                CurriculumStage(
                    "mid_24_32",
                    (
                        "maps/DoubleGame24x24.xml",
                        "maps/BWDistantResources32x32.xml",
                        "maps/chambers32x32.xml",
                    ),
                    min_score_rate=0.55,
                ),
                CurriculumStage(
                    "large_64",
                    (
                        "maps/BroodWar/(4)BloodBath.scmB.xml",
                        "maps/GardenOfWar64x64.xml",
                    ),
                    min_score_rate=0.50,
                ),
            ]
        )

    def stage_for_score(self, score_by_stage: Dict[str, float]) -> CurriculumStage:
        if not self.stages:
            raise ValueError("MapCurriculum requires at least one stage")
        current = self.stages[0]
        for stage in self.stages:
            current = stage
            if stage.min_score_rate is None:
                continue
            if score_by_stage.get(stage.name, 0.0) < stage.min_score_rate:
                return stage
        return current
