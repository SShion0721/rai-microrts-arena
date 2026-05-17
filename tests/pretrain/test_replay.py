import numpy as np

from rl_algo_impls.pretrain.replay import (
    OfflineReplay,
    OfflineTransition,
    ReplayMetadata,
)


def test_offline_replay_npz_roundtrip(tmp_path):
    replay = OfflineReplay(
        metadata=ReplayMetadata(map_path="maps/8x8/basesWorkers8x8A.xml", opponent="Mayari")
    )
    replay.append(
        OfflineTransition(
            obs=np.zeros((3, 4, 4), dtype=np.float32),
            action=np.zeros((16, 7), dtype=np.int64),
            action_mask=np.ones((16, 37), dtype=np.bool_),
            reward=np.array([1.0, 0.0], dtype=np.float32),
            done=False,
            info={"bot": "Mayari"},
        )
    )

    path = tmp_path / "demo.npz"
    replay.save_npz(path)
    loaded = OfflineReplay.load_npz(path)

    assert loaded.metadata.opponent == "Mayari"
    assert len(loaded) == 1
    np.testing.assert_array_equal(loaded.transitions[0].action, replay.transitions[0].action)
    assert loaded.summary()["done_count"] == 0
