import torch

from rl_algo_impls.microrts.region_state import REGION_TYPES, extract_region_state_batch


def test_extract_region_state_batch_builds_fixed_tokens():
    obs = torch.zeros(1, 60, 4, 4)
    obs[:, 58, :, :] = 1
    obs[:, 6, :, :] = 1
    obs[:, 2, 0, 0] = 1
    obs[:, 6, 1, 1] = 0
    obs[:, 4, 1, 1] = 1
    obs[:, 7, 1, 1] = 1
    obs[:, 6, 2, 2] = 0
    obs[:, 5, 2, 2] = 1
    obs[:, 8, 2, 2] = 1

    batch = extract_region_state_batch(obs)

    assert batch.tokens.shape == (1, len(REGION_TYPES), len(REGION_TYPES) + 5)
    assert batch.positions.shape == (1, len(REGION_TYPES), 2)
    assert batch.key_padding_mask.shape == (1, len(REGION_TYPES))
    assert not batch.key_padding_mask[0, REGION_TYPES.index("resource")]
    assert not batch.key_padding_mask[0, REGION_TYPES.index("frontline")]


def test_extract_region_state_batch_handles_empty_observation():
    obs = torch.zeros(2, 60, 3, 3)
    obs[:, 58, :, :] = 1
    obs[:, 6, :, :] = 1

    batch = extract_region_state_batch(obs)

    assert batch.tokens.shape == (2, len(REGION_TYPES), len(REGION_TYPES) + 5)
    assert batch.key_padding_mask[:, :-1].all()
    assert not batch.key_padding_mask[:, -1].any()
