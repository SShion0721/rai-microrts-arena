import numpy as np

from rl_algo_impls.microrts.vec_env.microrts_interface import MicroRTSInterface
from rl_algo_impls.microrts.vec_env.microrts_space_transform import (
    MicroRTSSpaceTransform,
)


class DummyMicroRTSInterface(MicroRTSInterface):
    DEBUG_VERIFY = False

    metadata = {"render.modes": []}

    def __init__(self):
        self._heights = [2, 4]
        self._widths = [2, 4]
        self._utt = {"unitTypes": [{}, {}]}

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def num_envs(self):
        return 2

    @property
    def heights(self):
        return self._heights

    @property
    def widths(self):
        return self._widths

    @property
    def utt(self):
        return self._utt

    @property
    def partial_obs(self):
        return False

    def terrain(self, env_idx):
        return np.zeros((self._heights[env_idx], self._widths[env_idx]))

    def terrain_md5(self, env_idx):
        return None

    def resources(self, env_idx):
        return np.zeros(2)

    def close(self, **kwargs):
        pass

    def add_listener(self, listener):
        pass

    def remove_listener(self, listener):
        pass

    def debug_matrix_obs(self, env_idx):
        return None

    def debug_matrix_mask(self, env_idx):
        return None


def test_to_microrts_action_uses_only_source_units_and_unpads_locations():
    transform = MicroRTSSpaceTransform(
        DummyMicroRTSInterface(), valid_sizes=[4], fixed_size=True
    )
    action_dim = transform._action_plane_dim
    masks = [
        np.zeros((2, action_dim + 2), dtype=np.int8),
        np.zeros((1, action_dim + 2), dtype=np.int8),
    ]
    masks[0][:, :2] = np.array([[0, 1], [1, 0]], dtype=np.int8)
    masks[1][:, :2] = np.array([[2, 3]], dtype=np.int8)
    transform._update_action_mask(masks)

    actions = np.zeros((2, 16, 7), dtype=np.int32)
    actions[0, 6] = [1, 2, 3, 0, 1, 0, 4]
    actions[0, 9] = [5, 3, 2, 1, 0, 1, 8]
    actions[1, 11] = [4, 1, 1, 2, 3, 1, 7]

    assert transform._to_microrts_action(actions) == [
        [[1, 1, 2, 3, 0, 1, 0, 4], [2, 5, 3, 2, 1, 0, 1, 8]],
        [[11, 4, 1, 1, 2, 3, 1, 7]],
    ]
