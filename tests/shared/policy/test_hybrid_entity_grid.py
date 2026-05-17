import gymnasium.spaces
import numpy as np
import torch

from rl_algo_impls.shared.policy.actor_critic_network.hybrid_entity_grid import (
    HybridEntityGridActorCriticNetwork,
)


def test_hybrid_entity_grid_forward_shapes():
    action_vec = np.array([6, 4, 4, 4, 4, 8, 49])
    obs_space = gymnasium.spaces.Box(low=0, high=1, shape=(60, 4, 4), dtype=np.float32)
    action_space = gymnasium.spaces.MultiDiscrete(np.tile(action_vec, 16))
    action_plane_space = gymnasium.spaces.MultiDiscrete(action_vec)
    network = HybridEntityGridActorCriticNetwork(
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
    )
    obs = torch.zeros(2, 60, 4, 4)
    obs[:, 58, :, :] = 1
    obs[:, 6, :, :] = 1
    obs[:, 6, 1, 1] = 0
    masks = torch.ones(2, 16, int(action_vec.sum()), dtype=torch.bool)

    forward = network.distribution_and_value(obs, action_masks=masks)

    assert forward.v.shape == (2,)
    assert forward.pi_forward.pi.sample().shape == (2, 16, len(action_vec))
