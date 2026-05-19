from types import SimpleNamespace

import gymnasium.spaces
import numpy as np
import torch

from rl_algo_impls.rollout.ppo_rollout import PPORollout


def test_ppo_rollout_exposes_sequence_metadata():
    n_steps, n_envs = 4, 2
    obs = np.zeros((n_steps, n_envs, 3), dtype=np.float32)
    actions = np.zeros((n_steps, n_envs, 1), dtype=np.int64)
    rewards = np.ones((n_steps, n_envs), dtype=np.float32)
    episode_starts = np.zeros((n_steps, n_envs), dtype=np.bool_)
    values = np.zeros((n_steps, n_envs), dtype=np.float32)
    logprobs = np.zeros((n_steps, n_envs), dtype=np.float32)
    memory_states = np.zeros((n_steps, n_envs, 5), dtype=np.float32)
    rollout_view = SimpleNamespace(latest_checkpoint_policy=None)
    config = SimpleNamespace(algo_hyperparams={})

    rollout = PPORollout(
        config=config,
        rollout_view=rollout_view,
        next_episode_starts=np.zeros(n_envs, dtype=np.bool_),
        next_values=np.zeros(n_envs, dtype=np.float32),
        obs=obs,
        actions=actions,
        rewards=rewards,
        episode_starts=episode_starts,
        values=values,
        logprobs=logprobs,
        action_masks=None,
        gamma=0.99,
        gae_lambda=0.95,
        action_plane_space=gymnasium.spaces.MultiDiscrete([2]),
        memory_states=memory_states,
    )

    batch = rollout.batch(torch.device("cpu"))
    sequence_batches = list(
        rollout.sequence_minibatches(
            chunk_length=2,
            chunks_per_minibatch=2,
            device=torch.device("cpu"),
            shuffle=False,
        )
    )

    assert batch.episode_starts.shape == (n_steps * n_envs,)
    assert batch.memory_states.shape == (n_steps * n_envs, 5)
    assert batch.sequence_mask.shape == (n_steps * n_envs,)
    assert len(sequence_batches) == 2
    assert sequence_batches[0].obs.shape[0] == 4
