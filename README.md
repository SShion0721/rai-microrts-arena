# rai-microrts-arena

Fork of [sgoodfriend/rl-algo-impls](https://github.com/sgoodfriend/rl-algo-impls) — a
competitive deep reinforcement learning framework for microRTS and Lux AI Season 2,
with comprehensive improvements for training efficiency and agent performance.

The name pays tribute to the **RAISocketAI** agent — IEEE CoG 2024 competition winner.

## Why this fork

This fork builds upon the IEEE CoG 2024 competition-winning RAISocketAI agent
([arXiv:2402.08112](https://arxiv.org/abs/2402.08112)) with:

- **JIT compilation**: `torch.compile` integration for 1.2-2x training speedup
- **Mixed precision training**: Enhanced `torch.amp` with bfloat16/float16 support
  and numerical safety guards for PPO log-prob calculations
- **League training**: ELO-based prioritized opponent selection (PFSP-style) on top
  of the existing self-play wrapper
- **PPO improvements**: Value function normalization, adaptive entropy coefficient,
  clip range scheduling (cosine/linear/exponential)
- **Memory optimizations**: Gradient checkpointing support for large models
- **Bug fixes**: Self-play team assignment, deterministic algorithms default,
  and other stability fixes

## Algorithms

- **PPO** (Proximal Policy Optimization) — with AMP, compile, checkpointing,
  adaptive entropy, value target normalization
- **APPO** (Asynchronous PPO) — Ray-based distributed rollout generation
- **DPPO** (Distributed PPO) — Multi-GPU via HuggingFace Accelerate
- **A2C** (Advantage Actor-Critic)
- **ACBC** (Actor-Critic Behavior Cloning) — BC pretraining for RL policies

## Network Architectures

| Architecture | Description |
|---|---|
| Squeeze-UNet (Squnet) | U-Net with Squeeze-Excitation attention — competition winner |
| DoubleCone | Strided encoder-decoder with SE residual blocks |
| SACUS | Split Actor-Critic U-Shaped — separate paths for policy/value |
| Grid2Seq Transformer | Sequence-based transformer attention over grid |
| Grid2Entity Transformer | Entity-based transformer attention over grid |
| GridNet | Classic Conv-TransposeConv for microRTS baselines |
| UNet | Standard U-Net with skip connections |

## Environments

- **microRTS** — small real-time strategy game (Java engine via JPype)
- **Lux AI Season 2** — turn-based resource management (Kaggle competition)
- Classic, Box2D, MuJoCo, and Atari via **Gymnasium**

## Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.x
- CUDA 12.x (for GPU training)
- Java SDK (for microRTS)

### Installation

```sh
git clone https://github.com/SShion0721/rai-microrts-arena.git
cd rai-microrts-arena

# Option A: Poetry (recommended)
poetry install -E microrts

# Option B: pip
pip install -e ".[microrts]"
```

### Training

```sh
# Basic microRTS training with Squnet + self-play
python train.py --algo ppo --env Microrts-squnet-map16-selfplay --seed 1

# With torch.compile and mixed precision
# (set torch_compile: true, autocast_loss: true in device_hyperparams)
python train.py --algo ppo --env Microrts-squnet-map16-selfplay

# League training with ELO-based opponent selection
python train.py --algo ppo --env Microrts-squnet-map16-league
```

### Key Configuration (YAML)

```yaml
device_hyperparams:
  torch_compile: true           # Enable JIT compilation
  compile_mode: "reduce-overhead"
  set_float32_matmul_precision: high

algo_hyperparams:
  autocast_loss: true           # Enable mixed precision
  autocast_amp_dtype: "bfloat16"  # or "float16"
  normalize_value_targets: true # Stabilize value learning
  adaptive_entropy: true        # Auto-adjust exploration
  target_entropy: 1.5
  gradient_checkpointing: true  # Reduce GPU memory
  clip_range_schedule: "cosine" # Progressive clip annealing

env_hyperparams:
  league_kwargs:                # ELO-based opponent selection
    priority_mode: "closest"
    target_win_rate_min: 0.3
    target_win_rate_max: 0.7
    exploration_rate: 0.1
```

## Project Structure

```
rl_algo_impls/
├── acbc/            # Behavior cloning algorithm
├── a2c/             # A2C algorithm
├── dqn/             # DQN (legacy)
├── hyperparams/     # YAML configs for all envs
├── loss/            # Custom loss functions
├── lux/             # Lux AI S2 integration
├── microrts/        # MicroRTS integration (forked env, Java engine)
├── ppo/             # PPO/APPO/DPPO algorithms
├── rollout/         # Rollout generation infrastructure
├── runner/          # Training orchestration and config
├── shared/
│   ├── actor/       # Policy distributions (MaskedCategorical, Gridnet)
│   ├── encoder/     # IMPALA-CNN, GridNet encoders
│   ├── module/      # Building blocks (SE, normalization, pooling)
│   ├── policy/
│   │   └── actor_critic_network/  # All network architectures
│   └── vec_env/    # Vectorized env creation
├── utils/           # Utilities (device, interpolate, etc.)
├── wrappers/        # Env wrappers (normalize, self-play, league, etc.)
│
├── main.py          # Entry point
├── train.py         # Training script
├── enjoy.py         # Evaluation script
└── selfplay_enjoy.py # Self-play evaluation
```

## Features inherited from upstream

- 8 competitive network architectures
- Multi-head value function (3 critics + curriculum reward scheduling)
- Self-play with rolling policy pool and periodic swapping
- Behavior cloning pretraining (ACBC) from reference AI rollouts
- Transfer learning (multi-map training with size-adaptive padding)
- Action masking (MaskedCategorical with subaction masks)
- Observation/reward normalization wrappers
- Gymnasium 0.29+ API compliance
- KL-adaptive learning rate with guard mechanisms
- Distributed training via Ray (APPO) and Accelerate (DPPO)

## Acknowledgements

Original repository by [Scott Goodfriend](https://github.com/sgoodfriend), IEEE CoG
2024 competition winner. This fork adds the improvements described in the *MicroRTS-Py
Comprehensive Improvement Plan* (2024-2026 research report).

If you use the original microRTS agent work, please cite:

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

```bibtex
@misc{rai-microrts-arena,
      title={rai-microrts-arena: Improved deep RL for microRTS},
      author={SShion0721},
      publisher={GitHub},
      howpublished={\url{https://github.com/SShion0721/rai-microrts-arena}}
}
```

## License

[MIT](LICENSE)
