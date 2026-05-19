# rai-microrts-arena

Improved deep RL for **microRTS** — forked from the IEEE CoG 2024 competition winner
[sgoodfriend/rl-algo-impls](https://github.com/sgoodfriend/rl-algo-impls).

Named after the **RAISocketAI** agent.

## Improvements over upstream

| # | Feature | What | File |
|---|---------|------|------|
| 1 | torch.compile | JIT-compile, 1.2-2x speedup | `runner/running_utils.py` |
| 2 | Mixed precision AMP | bfloat16/float16 + log_prob safety | `shared/autocast.py` |
| 3 | League training | ELO-based opponent selection (PFSP) | `wrappers/league_training_wrapper.py` |
| 4 | Value normalization | Per-minibatch value target norm | `ppo/ppo.py` |
| 5 | Adaptive entropy | Auto-adjust ent_coef to target | `ppo/ppo.py` |
| 6 | Clip schedule | Cosine/linear/exponential annealing | `ppo/ppo.py` |
| 7 | Gradient checkpointing | 50-70% GPU mem saved | `shared/policy/actor_critic.py` |
| 8 | Self-play fix | Team assignment bug | `wrappers/self_play_wrapper.py` |
| 9 | ACBC AMP | Mixed precision for BC pretraining | `acbc/acbc.py` |
| 10 | Determinism fix | Default False for compile compat | `runner/running_utils.py` |
| 11 | APPO AMP | Mixed precision for async PPO | `ppo/appo.py` |
| 12 | League env config | `league_kwargs` in env params | `runner/env_hyperparams.py` |
| 13 | Memory-aware policy API | Optional rollout memory state for recurrent policies | `shared/policy/actor_critic.py` |
| 14 | Entity-region hybrid | Entity graph + heuristic region tokens + GridNet head | `shared/policy/actor_critic_network/hierarchical_hybrid_entity_grid.py` |
| 15 | League roles | Main/exploiter/map-specialist population metadata | `wrappers/league_training_wrapper.py` |

## Quick start

```sh
git clone https://github.com/SShion0721/rai-microrts-arena.git
cd rai-microrts-arena
pip install -e ".[microrts]"

python train.py --algo ppo --env Microrts-squnet-map16-selfplay --seed 1
```

Current strongest research candidate:

```sh
python train.py --algo ppo --env Microrts-hierarchical-hybrid-memory-map16-selfplay --seed 1
```

Keep `Microrts-squnet-map16-selfplay` as the proven baseline, and compare it with
`Microrts-hybrid-entity-grid-map16-selfplay` plus the hierarchical memory run before
moving to 32/64 maps.

## Configs

Pre-made YAML in `rai_microrts/configs/`:

| File | Enables |
|------|---------|
| `squnet_compile.yaml` | torch.compile only |
| `squnet_amp.yaml` | Mixed precision only |
| `squnet_league.yaml` | ELO league training |
| `squnet_full.yaml` | All features combined |
| `hierarchical_hybrid_memory_league.yaml` | Strongest research candidate: entity graph + region tokens + GRU strategic memory + league roles |

## Algo params reference

```yaml
algo_hyperparams:
  autocast_loss: true               # mixed precision
  autocast_amp_dtype: "bfloat16"    # or "float16"
  normalize_value_targets: true     # value norm
  adaptive_entropy: true            # auto ent_coef
  target_entropy: 1.5
  gradient_checkpointing: true      # mem saving
  clip_range_schedule: "cosine"     # clip annealing
  clip_range_schedule_min: 0.1
```

## Upstream

Original: [sgoodfriend/rl-algo-impls](https://github.com/sgoodfriend/rl-algo-impls)

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

MIT licensed.
