# rai-microrts-arena

**Improved deep reinforcement learning for microRTS** — a competitive agent built on
[sgoodfriend/rl-algo-impls](https://github.com/sgoodfriend/rl-algo-impls), the IEEE
CoG 2024 competition winner ([arXiv:2402.08112](https://arxiv.org/abs/2402.08112)).

This directory documents the improvements made in this fork. All training code lives
in the parent `rl_algo_impls/` package.

---

## Improvements

| # | Feature | What it does | Where |
|---|---------|-------------|-------|
| 1 | **torch.compile** | JIT-compile policy for 1.2-2x training speedup | `runner/running_utils.py` `train.py` |
| 2 | **Mixed precision AMP** | bfloat16/float16 with numerical safety guards | `shared/autocast.py` `ppo/ppo.py` |
| 3 | **League training** | ELO-based prioritized opponent selection (PFSP) | `wrappers/league_training_wrapper.py` |
| 4 | **Value normalization** | Per-minibatch value target normalization | `ppo/ppo.py` |
| 5 | **Adaptive entropy** | Auto-adjust ent_coef toward target entropy | `ppo/ppo.py` |
| 6 | **Clip range scheduling** | Cosine/linear/exponential annealing of PPO clip | `ppo/ppo.py` |
| 7 | **Gradient checkpointing** | Trade compute for memory (50-70% GPU mem saved) | `shared/policy/actor_critic.py` `ppo/ppo.py` |
| 8 | **AMP in ACBC** | Mixed precision for behavior cloning pretraining | `acbc/acbc.py` |
| 9 | **Self-play fix** | Fixed team assignment in checkpoint_policy | `wrappers/self_play_wrapper.py` |
| 10 | **Determinism fix** | Changed default to `False` for torch.compile compat | `runner/running_utils.py` |

### Architecture inherited from upstream

- 8 network architectures (Squnet, DoubleCone, SACUS, UNet, Grid2Seq/Entity Transformers, ...)
- Multi-head critic with curriculum reward scheduling
- Self-play with rolling policy pool
- Transfer learning (multi-map + size-adaptive padding)
- Action masking (MaskedCategorical + subaction masks)
- APPO/DPPO distributed training (Ray + Accelerate)
- KL-adaptive learning rate
- Gymnasium 0.29+ API

---

## Quick start

```sh
git clone https://github.com/SShion0721/rai-microrts-arena.git
cd rai-microrts-arena

# Install dependencies (requires Java SDK for microRTS)
pip install -e ".[microrts]"

# Basic training
python train.py --algo ppo --env Microrts-squnet-map16-selfplay --seed 1
```

## Configs

Pre-made YAML snippets in `configs/`:

| File | Enables |
|------|---------|
| `squnet_compile.yaml` | torch.compile only |
| `squnet_amp.yaml` | Mixed precision only |
| `squnet_league.yaml` | ELO-based league training |
| `squnet_full.yaml` | All features combined |

Merge these into the main hyperparams YAML (`rl_algo_impls/hyperparams/ppo-Microrts.yml`)
or pass them via `--config-overrides`.

---

## What we changed

### New files
- `rl_algo_impls/wrappers/league_training_wrapper.py` — ELO-rated opponent selection

### Modified files (key changes)

**AMP system** — `rl_algo_impls/shared/autocast.py`:
- Added `resolve_amp_dtype()` for hardware-aware dtype selection
- Added `safe_amp_forward()` — wraps forward pass, forces log_prob/entropy → float32 when using float16

**PPO algorithm** — `rl_algo_impls/ppo/ppo.py`:
- New params: `autocast_amp_dtype`, `clip_range_schedule`/`clip_range_schedule_min`,
  `normalize_value_targets`, `adaptive_entropy`/`target_entropy`, `gradient_checkpointing`
- `_compute_clip_range()` method for schedule-based annealing
- Value target normalization in minibatch training loop
- Adaptive entropy coefficient adjustment

**Policy network** — `rl_algo_impls/shared/policy/actor_critic.py`:
- Added `forward_checkpoint()` for gradient checkpointing

**Training entry** — `rl_algo_impls/runner/train.py` + `running_utils.py`:
- `compile_policy()` — applies `torch.compile` to whole policy
- `set_device_optimizations()` — fixed `use_deterministic_algorithms` default

**Self-play** — `rl_algo_impls/wrappers/self_play_wrapper.py`:
- Fixed `checkpoint_policy` team assignment: old policy always goes to player-1 slot

**APPO + ACBC** — `rl_algo_impls/ppo/appo.py`, `rl_algo_impls/acbc/acbc.py`:
- Added `autocast_loss` + `autocast_amp_dtype` support

**Environment config** — `rl_algo_impls/runner/env_hyperparams.py`:
- Added `league_kwargs` field

**Env factories** — `rl_algo_impls/microrts/vec_env/microrts.py`, `lux/vec_env/lux.py`:
- League training wrapper integration

---

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

## License

MIT
