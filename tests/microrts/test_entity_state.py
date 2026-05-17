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
    assert batch.edge_index is not None
    assert batch.edge_attr is not None
    assert batch.edge_index.shape == (2, 0)
    assert batch.edge_attr.shape == (0, 4)


def test_extract_entity_state_batch_builds_vectorized_radius_edges():
    obs = torch.zeros(2, 60, 3, 3)
    obs[:, 58, :, :] = 1
    obs[:, 6, :, :] = 1

    for batch_idx, y, x in [
        (0, 0, 0),
        (0, 0, 1),
        (0, 2, 2),
        (1, 1, 1),
        (1, 2, 1),
    ]:
        obs[batch_idx, 6, y, x] = 0
        obs[batch_idx, 1, y, x] = 1

    batch = extract_entity_state_batch(obs, edge_radius=1.01)

    assert batch.n_entities.tolist() == [3, 2]
    assert batch.edge_index is not None
    assert batch.edge_attr is not None
    torch.testing.assert_close(
        batch.edge_index,
        torch.tensor([[0, 1, 3, 4], [1, 0, 4, 3]], dtype=torch.long),
    )
    torch.testing.assert_close(
        batch.edge_attr,
        torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, -1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, -1.0, 0.0, 1.0],
            ]
        ),
    )
