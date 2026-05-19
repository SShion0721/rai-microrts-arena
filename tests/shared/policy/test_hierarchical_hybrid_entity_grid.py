import gymnasium.spaces
import numpy as np
import torch

from rl_algo_impls.shared.policy.actor_critic_network.hierarchical_hybrid_entity_grid import (
    HierarchicalHybridEntityGridNetwork,
)


def _obs(batch_size=2):
    obs = torch.zeros(batch_size, 60, 4, 4)
    obs[:, 58, :, :] = 1
    obs[:, 6, :, :] = 1
    obs[:, 2, 0, 0] = 1
    obs[:, 6, 1, 1] = 0
    obs[:, 4, 1, 1] = 1
    obs[:, 7, 1, 1] = 1
    obs[:, 6, 2, 2] = 0
    obs[:, 5, 2, 2] = 1
    obs[:, 8, 2, 2] = 1
    return obs


def test_hierarchical_hybrid_entity_grid_memory_and_aux_shapes():
    action_vec = np.array([6, 4, 4, 4, 4, 8, 49])
    obs_space = gymnasium.spaces.Box(low=0, high=1, shape=(60, 4, 4), dtype=np.float32)
    action_space = gymnasium.spaces.MultiDiscrete(np.tile(action_vec, 16))
    action_plane_space = gymnasium.spaces.MultiDiscrete(action_vec)
    network = HierarchicalHybridEntityGridNetwork(
        obs_space,
        action_space,
        action_plane_space,
        channels_per_level=[8, 16],
        encoder_residual_blocks_per_level=[1, 1],
        decoder_residual_blocks_per_level=[1],
        encoder_embed_dim=16,
        encoder_feed_forward_dim=32,
        encoder_attention_heads=4,
        encoder_layers=1,
        normalization="layer",
        memory_kwargs={"kind": "gru", "hidden_dim": 32},
        hierarchical_action_kwargs={"strategy_latent_dim": 5, "num_groups": 3},
    )
    obs = _obs()
    masks = torch.ones(2, 16, int(action_vec.sum()), dtype=torch.bool)
    memory = network.initial_memory_state(2, torch.device("cpu"))

    forward = network.distribution_and_value(
        obs,
        action_masks=masks,
        memory_state=memory,
        episode_starts=torch.tensor([True, False]),
    )
    aux = network.auxiliary_predictions(obs, memory_state=forward.next_memory_state)

    assert forward.v.shape == (2,)
    assert forward.next_memory_state.shape == (2, 32)
    assert forward.pi_forward.pi.sample().shape == (2, 16, len(action_vec))
    assert aux["strategy_latent"].shape == (2, 5)
    assert aux["group_logits"].shape == (2, 3)
    assert aux["region_intent_logits"].shape == (2, 5)
