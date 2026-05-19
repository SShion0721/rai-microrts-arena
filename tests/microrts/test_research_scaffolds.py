import numpy as np
import torch

from rl_algo_impls.microrts.curriculum import MapCurriculum
from rl_algo_impls.microrts.retrieval import (
    EpisodicRetrievalStore,
    RetrievalEntry,
    RetrievalKey,
)
from rl_algo_impls.microrts.world_model import MacroWorldModel


def test_default_curriculum_advances_by_score():
    curriculum = MapCurriculum.default_microrts()

    assert curriculum.stage_for_score({}).name == "micro_8_12_16"
    assert (
        curriculum.stage_for_score(
            {"micro_8_12_16": 0.8, "mid_24_32": 0.6, "large_64": 0.4}
        ).name
        == "large_64"
    )


def test_retrieval_store_returns_strategic_summary_not_actions():
    key = RetrievalKey(
        map_topology=np.ones(2),
        resource_phase=np.zeros(2),
        army_composition=np.ones(3),
        region_control=np.zeros(1),
    )
    entry = RetrievalEntry(key=key, strategic_summary=np.array([1.0, 2.0]))
    store = EpisodicRetrievalStore([entry])

    results = store.query(key)

    assert results[0].strategic_summary.tolist() == [1.0, 2.0]


def test_macro_world_model_predicts_macro_variables():
    model = MacroWorldModel(
        input_dim=6, hidden_dim=8, resource_dim=2, army_dim=3, region_dim=4
    )

    output = model(torch.zeros(5, 6))

    assert output.resource_phase.shape == (5, 2)
    assert output.army_composition.shape == (5, 3)
    assert output.region_control.shape == (5, 4)
    assert output.engagement_window.shape == (5,)
