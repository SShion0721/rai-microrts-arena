# rai-microrts-arena

面向 **microRTS** 的深度强化学习训练仓库，基于
[sgoodfriend/rl-algo-impls](https://github.com/sgoodfriend/rl-algo-impls)
和 RAISocketAI 竞赛代理继续改造。当前目标不是只复现 PPO，而是把
imitation、self-play、league、实体表示、hybrid 网络和后续 option/预训练路线放进同一个可迭代框架里。

## 快速开始

推荐直接使用已有环境：

```powershell
micromamba activate mamba_env
python train.py --algo ppo --env Microrts-squnet-map16-selfplay --seed 1
```

也可以从头安装：

```powershell
python -m pip install -e ".[microrts]"
python -m pytest -q
```

常用命令：

```powershell
# 经典 SquNet 自博弈训练
python train.py --algo ppo --env Microrts-squnet-map16-selfplay --seed 1

# 新增的实体-网格混合网络实验
python train.py --algo ppo --env Microrts-hybrid-entity-grid-map16-selfplay --seed 1

# 当前最强研究候选：实体图 + region token + GRU strategic memory
python train.py --algo ppo --env Microrts-hierarchical-hybrid-memory-map16-selfplay --seed 1

# ACBC 行为克隆预训练，参考 Mayari 等脚本 bot
python train.py --algo acbc --env Microrts-squnet-d16-128-iMayari --seed 1

# 本地评估已保存模型
python enjoy.py --algo ppo --env Microrts-squnet-map16-selfplay --seed 1 --n-episodes 10

# 打印固定 microRTS 评测矩阵
python scripts/microrts_eval_matrix.py
```

## 代码结构

```text
rl_algo_impls/
  acbc/                         # Actor-Critic Behavior Cloning
  ppo/                          # PPO / APPO / DPPO
  rollout/                      # rollout 收集、reference rollout、PPO batch
  runner/                       # CLI 后的训练/评估编排
  hyperparams/                  # YAML 实验配置
  microrts/
    vec_env/                    # Java microRTS 环境、obs/action 转换
    options/                    # 程序化 option / macro-action API
    entity_state.py             # 从 plane obs 抽取非空实体
    eval_matrix.py              # 固定评测矩阵和胜率置信区间
  pretrain/
    replay.py                   # 离线轨迹 npz schema
    graphdino.py                # GraphDINO/MAE 预训练骨架
  shared/
    actor/                      # MaskedCategorical、GridnetDistribution
    policy/actor_critic_network # 所有 actor-critic 网络
    callbacks/                  # reward/LR/KL/self-play schedule
  wrappers/                     # self-play、league、normalization、action mask
```

## 当前网络

所有 on-policy 算法默认使用 `ActorCritic`。真正的网络由
`policy_hyperparams.actor_head_style` 选择，入口在
`rl_algo_impls/shared/policy/actor_critic.py`。

| `actor_head_style` | 主要用途 | 说明 |
|---|---|---|
| `gridnet` | 基础 microRTS actor | CNN 编码后对每个地图格输出 MultiDiscrete action logits。 |
| `unet` | 早期空间网络 | 保留高分辨率空间特征，适合小图实验。 |
| `double_cone` | RAISocketAI 经典强基线 | 多层残差块加下采样/上采样，大容量但大图推理较重。 |
| `squeeze_unet` | 当前主力 SquNet | U-Net 风格，速度更好，适合 8/12/16/32/64 分图训练。 |
| `sacus` | Split Actor-Critic U-Net | actor 用高分辨率 decoder，critic 用 bottleneck，更偏全局价值估计。 |
| `grid2seq_transformer` | 全格点 Transformer | 把所有格子作为 token，长程建模强但大图 token 多。 |
| `grid2entity_transformer` | 实体 Transformer | 只保留非空实体 token，适合稀疏地图和对象关系。 |
| `hybrid_entity_grid` | 新增研究线 | SquNet 网格 actor + 实体 Transformer 全局上下文，通过 FiLM 注入网格特征。 |
| `hierarchical_hybrid_entity_grid` | 当前最强研究候选 | SquNet tactical grid + 实体图 + region token + GRU strategic memory，仍使用 GridNet primitive head 和合法动作 mask。 |

## 当前最强实验

严格说有两个答案：

- **最稳的已验证基线**：`Microrts-squnet-map16-selfplay`。它仍是所有新模块必须打平或超过的 baseline。
- **最值得投入算力的最强候选**：`Microrts-hierarchical-hybrid-memory-map16-selfplay`。它把本轮重构里已经接通的 entity graph、region token、GRU strategic memory 和 SquNet/GridNet primitive head 放到同一条训练线上，风险比直接上 Mamba/world model 低。

建议固定三组对照一起跑：`Microrts-squnet-map16-selfplay`、`Microrts-hybrid-entity-grid-map16-selfplay`、`Microrts-hierarchical-hybrid-memory-map16-selfplay`。只有 hierarchical 在 8/16/32 eval matrix 上稳定不劣化，再进入 32/64 迁移。

### 动作头和 mask

microRTS 的动作被拆成多个子动作：动作类型、移动方向、采集方向、返回方向、生产方向、生产单位类型、攻击目标等。`GridnetDistribution` 会对每个地图格、每个子动作建立 masked categorical。

关键点：

- `mask_actions: true` 会使用环境给出的合法动作 mask。
- `subaction_mask` 表达子动作之间的依赖。例如非生产动作不应学习生产类型。
- 当前 primitive actor 仍是近似因子化的，因此新增 `microrts/options/` 作为后续两级 actor 的接口：先选 harvest/build/rush/defend 等宏动作，再落到合法 primitive action。

### Hybrid Entity-Grid

`hybrid_entity_grid` 的数据流：

1. 原始 plane observation 进入 `SqueezeUnetBackbone`，得到高分辨率局部网格特征。
2. `entity_state.py` 从同一 observation 中抽取非空格子，构成 `EntityStateBatch`。
3. 实体 token 加位置特征后进入 Transformer encoder，得到全局 entity context。
4. context 经过 FiLM 生成 `gamma/beta`，调制网格特征。
5. actor 仍使用 Gridnet logits，critic 使用 `grid_avg + grid_max + entity_context`。

这样保留了 GridNet 的合法动作和空间输出，同时给大图/稀疏图一个更便宜的全局通信通道。

## 训练流程

训练入口：

```text
train.py
  -> rl_algo_impls.train.train()
  -> runner.train.train()
  -> load_hyperparams(algo, env)
  -> make env / rollout generator / policy / algo / evaluator
  -> algo.learn(...)
```

配置加载规则：

- `--algo ppo --env Microrts-squnet-map16-selfplay`
  会优先读取 `rl_algo_impls/hyperparams/ppo-Microrts.yml` 中同名配置。
- 如果没有专门的 `*-Microrts.yml`，再回退到 `ppo.yml`。
- YAML 大量使用 anchor，例如 `_Microrts-squnet`、`_Microrts-squnet-map16`，具体实验用 `<<:` 继承并覆盖少量字段。

PPO 每轮大致做这些事：

1. rollout worker 运行 `n_envs * n_steps` 步，保存 obs、action、mask、logprob、value、reward heads。
2. 用 GAE 根据 `gamma`、`gae_lambda` 计算 advantage 和 return。
3. 如果有 `multi_reward_weights`，把多 reward head 的 advantage 加权成 PPO policy loss。
4. 按 `batch_size` 切 minibatch，重复 `n_epochs` 轮 PPO 更新。
5. policy loss 使用 `clip_range`，value loss 可使用 `clip_range_vf`。
6. entropy loss 由 `ent_coef` 控制；teacher imitation 由 `teacher_kl_loss_coef` 控制。
7. evaluator 按 `eval_hyperparams.step_freq` 周期评估并保存 best/latest。

ACBC 用于 imitation 预训练：

- `rollout_type: reference` 从参考 bot 产生示范动作。
- loss 主要是参考动作负 logprob，加 critic value loss。
- `scale_loss_by_num_actions: true` 会按单位/动作数量缩放，避免多单位局面主导梯度。
- 预训练 checkpoint 可在 PPO 配置中通过 `load_path` 或 `load_run_path` 接入继续 fine-tune。

## 重要超参数

### 顶层字段

| 字段 | 作用 | 常见取值 |
|---|---|---|
| `n_timesteps` | 总训练步数 | 小实验 `1e6` 到主训练 `100e6-500e6`。 |
| `process_mode` | 同步或异步训练 | 常用 `sync`，分布式可用 `async`。 |
| `device` | 训练设备 | `auto`、`cuda`、`cpu`。 |
| `additional_keys_to_log` | 额外日志 | microRTS 常记录 `microrts_stats`、`results`、`action_mask_stats`。 |

### `env_hyperparams`

| 字段 | 作用 |
|---|---|
| `n_envs` | 并行环境数。越大 rollout 吞吐越高，但显存/CPU/Java 负载更高。 |
| `env_type` | `microrts` 为常规环境，`microrts_bots` 用于参考 bot/ACBC。 |
| `mask_actions` | 是否启用合法动作 mask，microRTS 训练一般必须打开。 |
| `map_paths` | 训练地图列表。多图训练可提升泛化，但学习更慢。 |
| `valid_sizes` | 允许的地图尺寸，用于 padding 和模型选择。 |
| `bots` | 训练时混入的脚本 bot 数量，例如 `coacAI: 3`、`mayari: 3`。 |
| `self_play_kwargs` | 历史策略池、自博弈保存和替换策略。 |
| `league_kwargs` | league/PFSP/PBT 对手选择配置。 |
| `make_kwargs.max_steps` | 单局最大 game step，地图越大通常越高。 |
| `make_kwargs.reward_weight` | Java 环境原始 reward 分量权重。 |

典型 self-play 默认值：

```yaml
self_play_kwargs:
  num_old_policies: 12
  save_steps: 300000
  swap_steps: 6000
  swap_window_size: 4
  window: 33
```

### `policy_hyperparams`

| 字段 | 作用 |
|---|---|
| `actor_head_style` | 网络架构选择，见上面的网络表。 |
| `channels_per_level` | U-Net/SquNet 每层通道数。更大更强也更慢。 |
| `strides_per_level` | 下采样倍率，决定感受野和计算量。 |
| `encoder_residual_blocks_per_level` | encoder 每层残差块数。 |
| `decoder_residual_blocks_per_level` | decoder 每层残差块数。 |
| `normalization` | `layer` 等 normalization，Transformer/hybrid 通常需要。 |
| `encoder_embed_dim` | Transformer/entity context 维度。 |
| `encoder_attention_heads` | 注意力头数，需整除 `encoder_embed_dim`。 |
| `encoder_layers` | Transformer 层数。 |
| `actor_head_kernel_size` | actor 输出卷积核大小，`3` 可看一点邻域。 |
| `subaction_mask` | 子动作依赖约束，microRTS 强烈建议保留。 |
| `load_path` / `load_run_path` | 从本地或 wandb checkpoint 初始化。 |

当前新增 hybrid / hierarchical 配置：

```yaml
Microrts-hybrid-entity-grid-map16-selfplay:
  <<: *microrts-squnet-map16-selfplay
  policy_hyperparams:
    <<: *microrts-squnet-map16-policy-defaults
    actor_head_style: hybrid_entity_grid
    normalization: layer
    encoder_embed_dim: 128
    encoder_attention_heads: 4
    encoder_feed_forward_dim: 256
    encoder_layers: 2
    actor_head_kernel_size: 3

Microrts-hierarchical-hybrid-memory-map16-selfplay:
  <<: *microrts-squnet-map16-selfplay
  policy_hyperparams:
    <<: *microrts-squnet-map16-policy-defaults
    actor_head_style: hierarchical_hybrid_entity_grid
    normalization: layer
    encoder_embed_dim: 128
    encoder_attention_heads: 4
    encoder_feed_forward_dim: 256
    encoder_layers: 2
    actor_head_kernel_size: 3
    memory_kwargs:
      kind: gru
      hidden_dim: 256
      entity_edge_radius: 2.0
    region_tokenizer_kwargs:
      kind: heuristic
    hierarchical_action_kwargs:
      strategy_latent_dim: 8
      num_groups: 8
```

### `rollout_hyperparams`

| 字段 | 作用 |
|---|---|
| `n_steps` | 每个 env 每次 rollout 的长度。microRTS 常用 `512`。 |
| `gamma` | 折扣因子。大图/长期任务常用 `0.999` 到 `0.9999`。 |
| `gae_lambda` | GAE 平滑系数，常用 `0.95` 或 `0.99`。 |
| `include_logp` | 是否保存旧策略 logprob，PPO 需要。 |
| `full_batch_off_accelerator` | 大图/小 batch 场景减少 accelerator 占用。 |

如果 reward 是多头，`gamma` 和 `gae_lambda` 可以是数组，例如 SquNet 默认：

```yaml
gamma: [0.99, 0.999, 0.999]
gae_lambda: [0.95, 0.99, 0.99]
```

### `algo_hyperparams`

| 字段 | 作用 |
|---|---|
| `batch_size` | PPO minibatch 大小。小图可大 batch，大图通常要降。 |
| `n_epochs` | 每批 rollout 重复训练次数。过高容易 stale policy。 |
| `learning_rate` | 学习率。PPO 主线常在 `1e-5` 到 `2.5e-4`。 |
| `clip_range` | PPO policy clip，常用 `0.1` 或 `0.2`。 |
| `clip_range_vf` | value clip；设 `null` 表示不 clip value。 |
| `vf_coef` | value loss 权重，多 value head 时可为数组。 |
| `ent_coef` | entropy 权重，控制探索。早期高、后期低。 |
| `max_grad_norm` | 梯度裁剪，常用 `0.5`。 |
| `multi_reward_weights` | 多 reward head 的 policy advantage 权重。 |
| `vf_weights` | 多 value head 的 value loss 权重。 |
| `teacher_kl_loss_coef` | PPO fine-tune 时保持接近教师策略。 |
| `gradient_accumulation` | 大图小显存时累积梯度。 |
| `gradient_checkpointing` | 省显存，代价是训练更慢。 |
| `normalize_value_targets` | 对 value target 归一化，奖励尺度不稳时有用。 |
| `adaptive_entropy` / `target_entropy` | 自动调 `ent_coef` 到目标熵。 |
| `autocast_loss` / `autocast_amp_dtype` | AMP 训练，常用 `bfloat16`。 |
| `scale_loss_by_num_actions` | 多单位局面按动作数缩放 loss，ACBC 常用。 |

典型 SquNet map16 PPO：

```yaml
env_hyperparams:
  n_envs: 24
  map_paths:
    - maps/16x16/basesWorkers16x16A.xml
    - maps/16x16/TwoBasesBarracks16x16.xml
    - maps/16x16/melee16x16Mixed12.xml
  bots:
    coacAI: 3
    mayari: 3
rollout_hyperparams:
  n_steps: 512
  gamma: [0.99, 0.999, 0.999]
  gae_lambda: [0.95, 0.99, 0.99]
algo_hyperparams:
  batch_size: 6144
  n_epochs: 2
  clip_range: 0.1
  ent_coef: 0.01
```

## Reward schedule 和课程学习

`hyperparam_transitions_kwargs` 用来把训练分成几个阶段。SquNet self-play 的典型模式是：

```yaml
phases:
  - multi_reward_weights: [0.8, 0.01, 0.19]
    vf_coef: [0.5, 0.1, 0.2]
    ent_coef: 0.01
    learning_rate: 1e-4
  - multi_reward_weights: [0, 0.99, 0.01]
    vf_coef: [0, 0.5, 0.1]
    ent_coef: 0.001
    learning_rate: 5e-5
durations: [0.5, 0.3, 0.2]
```

解释：

- 前半段重 shaped reward，让模型先学会采矿、造兵、攻击等基本行为。
- 中间过渡阶段逐步降低 shaped reward。
- 后期以 win/loss sparse reward 为主，减少 reward hacking。
- `durations` 长度必须是 `2 * len(phases) - 1`，因为包含阶段和阶段间插值。

## 推荐实验路线

1. **环境 smoke test**

   ```powershell
   micromamba activate mamba_env
   python -m pytest -q tests/microrts tests/pretrain tests/wrappers
   ```

2. **小步确认入口和配置**

   ```powershell
   python train.py --algo ppo --env Microrts-hierarchical-hybrid-memory-map16-selfplay --seed 1
   ```

   第一次不建议直接跑满 `200e6`。可以临时复制一个 debug 配置，把 `n_timesteps` 降到 `1e6`，确认 rollout、loss、eval、checkpoint 都正常。

3. **ACBC 预训练**

   先跑 `acbc-Microrts.yml` 里的 `Microrts-squnet-d16-128-iMayari` 系列，让策略学 Mayari/Coac/LightRush 的基本行为。

4. **PPO fine-tune**

   用 `load_path` 或 `load_run_path` 接入 ACBC checkpoint，早期保留较高 `teacher_kl_loss_coef`，后期降到 0。

5. **Self-play / league**

   开启 `self_play_kwargs` 后，模型会周期性保存历史策略并替换对手。`LeagueTrainingWrapper` 现在支持 `priority_mode: pfsp`、matchup table 和 PBT 元数据，适合后续把 main agent、exploiter、league exploiter 分开管理。

6. **大图迁移**

   推荐按 16 -> 32 -> 64 迁移。大图通常降低 `batch_size`、提高 `gamma`，必要时启用 `gradient_accumulation` 和较轻网络。

## 评估

固定评估矩阵在 `rl_algo_impls/microrts/eval_matrix.py`：

- 地图覆盖 8、16、24、32、64 和 BroodWar map。
- 对手覆盖 WorkerRush、LightRush、CoacAI、Mayari、TMA。
- 每个地图/对手都应双边开局，记录胜/平/负、score rate、Wilson 置信区间、平均推理耗时和超时率。

打印评测任务：

```powershell
python scripts/microrts_eval_matrix.py
```

## 调参经验

- 训练不出基本行为：提高 shaped reward 比重，增大 `ent_coef`，先用 ACBC。
- 会采矿但赢不了：后期提高 win/loss reward，降低 shaped reward。
- value loss 很不稳：尝试 `normalize_value_targets: true` 或降低 `vf_coef`。
- policy KL 暴涨：降低 `learning_rate` 或 `clip_range`，减少 `n_epochs`。
- 大图太慢：减少 `channels_per_level`、降低 Transformer `encoder_layers`，或用 SquNet 而不是 DoubleCone。
- 大图不会长程决策：尝试 `hierarchical_hybrid_entity_grid`、`hybrid_entity_grid` 或 `grid2entity_transformer`，并从 16/32 图 checkpoint 迁移。
- 自博弈过拟合单一风格：增加历史策略池、混入 bot、使用 PFSP/league。

## 新增模块状态

| 模块 | 状态 | 下一步 |
|---|---|---|
| `pretrain/replay.py` | 可保存/读取离线轨迹 | 接入 rollout collector 和 ACBC dataset。 |
| `pretrain/graphdino.py` | 有 student/teacher 预训练骨架 | 增加 masked entity reconstruction 和真实数据 loader。 |
| `microrts/entity_state.py` | 可抽取实体 token | 增加更丰富关系边和单位语义特征。 |
| `hybrid_entity_grid.py` | 已接入 `actor_head_style` | 跑 ablation，比 SquNet/Grid2Entity 胜率和耗时。 |
| `hierarchical_hybrid_entity_grid.py` | 已接入 memory/entity/region 强候选 | 跑 3 seed eval matrix，确认相对 SquNet/Hybrid 的收益和延迟。 |
| `policy/memory/strategic_memory.py` | GRU strategic memory 已接入 | 先验证 recurrent PPO/logprob 一致性，再评估 Mamba。 |
| `microrts/region_state.py` | heuristic region token 已接入 | 做 empty-map/大图 token ablation，后续学习化 tokenizer。 |
| `microrts/options/` | 有 option API 和合法 fallback | 增加 option-to-primitive 规则与学习式 pointer head。 |
| `league_training_wrapper.py` | 有 PFSP/PBT 元数据 | 增加真正的 population mutation callback。 |

## 参考

- RAISocketAI 论文：[A Competition Winning Deep Reinforcement Learning Agent in microRTS](https://arxiv.org/abs/2402.08112)
- microRTS/Gym-microRTS：[Gym-microRTS](https://gaigresearch.github.io/2021/06/15/huang2021gym/)
- Invalid action masking：[arXiv:2006.14171](https://arxiv.org/abs/2006.14171)
- AlphaStar league/PBT 思路：[DeepMind AlphaStar](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)

## License

[MIT](LICENSE)
