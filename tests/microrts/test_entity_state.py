import torch

from rl_algo_impls.microrts.entity_state import extract_entity_state_batch


def test_extract_entity_state_batch_filters_empty_cells():
    obs = torch.zeros(1, 60, 4, 4)
    obs[:, 58, :, :] = 1
    obs[:, 6, :, :] = 1
    obs[:, 6, 1, 2] = 0
    obs[:, 1, 1, 2] = 1

    batch = extract_entity_state_batch(obs, edge_radius=2.0)

    assert batch.nodes.shape[:2] == (1, 1)
    assert batch.n_entities.tolist() == [1]
    assert not batch.key_padding_mask[0, 0]
    assert batch.nodes.shape[-1] == 62
