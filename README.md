# rai-microrts-arena

Improved deep reinforcement learning for **microRTS** — forked from
[sgoodfriend/rl-algo-impls](https://github.com/sgoodfriend/rl-algo-impls),
the IEEE CoG 2024 competition winner ([arXiv:2402.08112](https://arxiv.org/abs/2402.08112)).

Named after the **RAISocketAI** agent.

## Improvements over upstream

| # | Feature | What | Where |
|---|---------|------|-------|
| 1 | `torch.compile` | JIT-compile policy, 1.2-2x speedup | `runner/running_utils.py` |
| 2 | Mixed precision AMP | bfloat16/float16 with log_prob safety | `shared/autocast.py` |
| 3 | League training | ELO-based prioritized opponent selection | `wrappers/league_training_wrapper.py` |
| 4 | Value normalization | Per-minibatch value target normalization | `ppo/ppo.py` |
| 5 | Adaptive entropy | Auto-adjust ent_coef toward target | `ppo/ppo.py` |
| 6 | Clip schedule | Cosine/linear/exponential clip annealing | `ppo/ppo.py` |
| 7 | Gradient checkpointing | 50-70% GPU memory saving | `shared/policy/actor_critic.py` |
| 8 | Self-play fix | Team assignment in checkpoint_policy | `wrappers/self_play_wrapper.py` |
| 9 | ACBC AMP | Mixed precision for BC pretraining | `acbc/acbc.py` |
| 10 | Determinism fix | Default `False` for torch.compile compat | `runner/running_utils.py` |

## Algorithms

- **PPO** / **APPO** / **DPPO** — Proximal Policy Optimization + distributed variants
- **ACBC** — Actor-Critic Behavior Cloning for pretraining

## Network architectures

Squnet (Squeeze-UNet with SE attention), DoubleCone, SACUS (split Actor-Critic),
Grid2Seq/Grid2Entity Transformers, GridNet, UNet.

## Quick start

```sh
git clone https://github.com/SShion0721/rai-microrts-arena.git
cd rai-microrts-arena
# Requires: Python 3.10+, PyTorch 2.x, Java SDK
pip install -e ".[microrts]"

# Basic training
python train.py --algo ppo --env Microrts-squnet-map16-selfplay --seed 1
```

## Example configs

See `rai_microrts/configs/` for ready-to-use YAML snippets:
`squnet_compile.yaml` · `squnet_amp.yaml` · `squnet_league.yaml` · `squnet_full.yaml`

## Project structure

```
rl_algo_impls/
├── microrts/        # MicroRTS env + Java engine
├── ppo/             # PPO / APPO / DPPO
├── acbc/            # Behavior cloning
├── rollout/         # Data collection
├── runner/          # Training orchestration
├── shared/          # Networks, modules, wrappers, callbacks
│   ├── policy/actor_critic_network/  # All architectures
│   ├── actor/        # Distributions (MaskedCategorical, Gridnet)
│   ├── encoder/      # CNN / IMPALA / GridNet encoders
│   ├── module/       # SE, normalization, pooling
│   └── vec_env/      # Vectorized env factory
├── wrappers/         # Normalize, self-play, league, action-mask
├── hyperparams/      # microRTS training configs
└── utils/            # Device, interpolation, timing
```

## Upstream

Original by [Scott Goodfriend](https://github.com/sgoodfriend).

```bibtex
@misc{goodfriend2024competition,
      title={A Competition Winning Deep Reinforcement Learning Agent in microRTS},
      author={Scott Goodfriend},
      year={2024},
      eprint={2402.08112},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License

[MIT](LICENSE)
