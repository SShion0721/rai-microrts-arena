from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import List, Sequence

import gymnasium.spaces
import numpy as np
import torch

from rl_algo_impls.shared.policy.actor_critic_network.double_cone import (
    DoubleConeActorCritic,
)
from rl_algo_impls.shared.policy.actor_critic_network.grid2entity_transformer import (
    Grid2EntityTransformerNetwork,
)
from rl_algo_impls.shared.policy.actor_critic_network.hybrid_entity_grid import (
    HybridEntityGridActorCriticNetwork,
)
from rl_algo_impls.shared.policy.actor_critic_network.hierarchical_hybrid_entity_grid import (
    HierarchicalHybridEntityGridNetwork,
)
from rl_algo_impls.shared.policy.actor_critic_network.network import ActorCriticNetwork
from rl_algo_impls.shared.policy.actor_critic_network.squeeze_unet import (
    SqueezeUnetActorCriticNetwork,
)

ACTION_VEC = np.array([6, 4, 4, 4, 4, 8, 49], dtype=np.int64)
SUPPORTED_STYLES = (
    "squeeze_unet",
    "double_cone",
    "grid2entity_transformer",
    "hybrid_entity_grid",
    "hierarchical_hybrid_entity_grid",
)


@dataclass(frozen=True)
class ProfileResult:
    style: str
    batch_size: int
    map_size: int
    entities: int
    action_mode: str
    device: str
    dtype: str
    num_parameters: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile microRTS policy network forward latency without starting the "
            "Java environment. The script uses synthetic BCHW observations and "
            "all-legal GridNet masks so architecture overhead can be compared "
            "quickly."
        )
    )
    parser.add_argument(
        "--styles",
        default=",".join(SUPPORTED_STYLES),
        help=f"Comma-separated styles from: {', '.join(SUPPORTED_STYLES)}",
    )
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--channels", type=int, default=60)
    parser.add_argument("--entities", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cpu or cuda.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "float16", "bfloat16"),
    )
    parser.add_argument(
        "--action-mode",
        default="sample",
        choices=("distribution", "sample", "mode", "entropy"),
        help=(
            "distribution only builds logits/distribution/value; sample/mode/entropy "
            "also executes the corresponding distribution method."
        ),
    )
    parser.add_argument(
        "--channels-per-level",
        default=["64,128,256"],
        nargs="+",
        help=(
            "SquNet/Hybrid channel sizes. Accepts comma-separated values "
            "or space-separated values, for example 64,128,256 or 64 128 256."
        ),
    )
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--encoder-embed-dim", type=int, default=128)
    parser.add_argument("--encoder-attention-heads", type=int, default=4)
    parser.add_argument("--encoder-feed-forward-dim", type=int, default=256)
    parser.add_argument("--double-cone-channels", type=int, default=128)
    parser.add_argument("--double-cone-pooled-channels", type=int, default=512)
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Compile each network before profiling. Useful for deployment checks.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of a Markdown table.",
    )
    return parser.parse_args()


def parse_ints(raw: Sequence[str]) -> List[int]:
    values = [
        int(part.strip()) for chunk in raw for part in chunk.split(",") if part.strip()
    ]
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def build_spaces(map_size: int, channels: int):
    obs_space = gymnasium.spaces.Box(
        low=0,
        high=1,
        shape=(channels, map_size, map_size),
        dtype=np.float32,
    )
    action_space = gymnasium.spaces.MultiDiscrete(
        np.tile(ACTION_VEC, map_size * map_size)
    )
    action_plane_space = gymnasium.spaces.MultiDiscrete(ACTION_VEC)
    return obs_space, action_space, action_plane_space


def build_network(style: str, args: argparse.Namespace) -> ActorCriticNetwork:
    obs_space, action_space, action_plane_space = build_spaces(
        args.map_size, args.channels
    )
    channels_per_level = parse_ints(args.channels_per_level)
    strides_per_level = [2] * (len(channels_per_level) - 1)
    encoder_blocks = [1] * len(channels_per_level)
    decoder_blocks = encoder_blocks[:-1]

    if style == "squeeze_unet":
        return SqueezeUnetActorCriticNetwork(
            obs_space,
            action_space,
            action_plane_space,
            channels_per_level=channels_per_level,
            strides_per_level=strides_per_level,
            encoder_residual_blocks_per_level=encoder_blocks,
            decoder_residual_blocks_per_level=decoder_blocks,
            normalization="layer",
        )
    if style == "double_cone":
        return DoubleConeActorCritic(
            obs_space,
            action_space,
            action_plane_space,
            backbone_channels=args.double_cone_channels,
            pooled_channels=args.double_cone_pooled_channels,
        )
    if style == "grid2entity_transformer":
        return Grid2EntityTransformerNetwork(
            obs_space,
            action_space,
            action_plane_space,
            encoder_embed_dim=args.encoder_embed_dim,
            encoder_attention_heads=args.encoder_attention_heads,
            encoder_feed_forward_dim=args.encoder_feed_forward_dim,
            encoder_layers=args.encoder_layers,
            normalization="layer",
        )
    if style == "hybrid_entity_grid":
        return HybridEntityGridActorCriticNetwork(
            obs_space,
            action_space,
            action_plane_space,
            channels_per_level=channels_per_level,
            strides_per_level=strides_per_level,
            encoder_residual_blocks_per_level=encoder_blocks,
            decoder_residual_blocks_per_level=decoder_blocks,
            encoder_embed_dim=args.encoder_embed_dim,
            encoder_attention_heads=args.encoder_attention_heads,
            encoder_feed_forward_dim=args.encoder_feed_forward_dim,
            encoder_layers=args.encoder_layers,
            normalization="layer",
        )
    if style == "hierarchical_hybrid_entity_grid":
        return HierarchicalHybridEntityGridNetwork(
            obs_space,
            action_space,
            action_plane_space,
            channels_per_level=channels_per_level,
            strides_per_level=strides_per_level,
            encoder_residual_blocks_per_level=encoder_blocks,
            decoder_residual_blocks_per_level=decoder_blocks,
            encoder_embed_dim=args.encoder_embed_dim,
            encoder_attention_heads=args.encoder_attention_heads,
            encoder_feed_forward_dim=args.encoder_feed_forward_dim,
            encoder_layers=args.encoder_layers,
            normalization="layer",
            memory_kwargs={
                "kind": "gru",
                "hidden_dim": args.encoder_embed_dim * 2,
                "entity_edge_radius": 2.0,
            },
            hierarchical_action_kwargs={
                "strategy_latent_dim": 8,
                "num_groups": 8,
            },
        )
    raise ValueError(f"Unsupported style {style}")


def synthetic_obs(
    batch_size: int,
    channels: int,
    map_size: int,
    entities: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    obs = torch.zeros(
        batch_size, channels, map_size, map_size, device=device, dtype=dtype
    )
    obs[:, 58, :, :] = 1
    obs[:, 6, :, :] = 1

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    cells = map_size * map_size
    n_entities = min(entities, cells)
    for batch_idx in range(batch_size):
        flat_positions = torch.randperm(cells, generator=generator, device=device)[
            :n_entities
        ]
        y = flat_positions // map_size
        x = flat_positions % map_size
        obs[batch_idx, 6, y, x] = 0
        obs[batch_idx, 1, y, x] = 1
    return obs


def action_masks(batch_size: int, map_size: int, device: torch.device) -> torch.Tensor:
    return torch.ones(
        batch_size,
        map_size * map_size,
        int(ACTION_VEC.sum()),
        dtype=torch.bool,
        device=device,
    )


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_once(
    network: ActorCriticNetwork,
    obs: torch.Tensor,
    masks: torch.Tensor,
    action_mode: str,
) -> None:
    forward = network.distribution_and_value(obs, action_masks=masks)
    if action_mode == "sample":
        forward.pi_forward.pi.sample()
    elif action_mode == "mode":
        getattr(forward.pi_forward.pi, "mode")
    elif action_mode == "entropy":
        forward.pi_forward.pi.entropy()


def profile_style(style: str, args: argparse.Namespace) -> ProfileResult:
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    network = build_network(style, args).to(device=device)
    network.eval()
    if args.torch_compile:
        network = torch.compile(network)  # type: ignore[assignment]

    obs = synthetic_obs(
        args.batch_size,
        args.channels,
        args.map_size,
        args.entities,
        device,
        dtype,
        args.seed,
    )
    masks = action_masks(args.batch_size, args.map_size, device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            run_once(network, obs, masks, args.action_mode)
        sync_if_needed(device)

        timings = []
        for _ in range(args.iters):
            start = time.perf_counter()
            run_once(network, obs, masks, args.action_mode)
            sync_if_needed(device)
            timings.append((time.perf_counter() - start) * 1000)

    sorted_timings = sorted(timings)
    p95_idx = min(len(sorted_timings) - 1, int(round(0.95 * (len(sorted_timings) - 1))))
    return ProfileResult(
        style=style,
        batch_size=args.batch_size,
        map_size=args.map_size,
        entities=min(args.entities, args.map_size * args.map_size),
        action_mode=args.action_mode,
        device=str(device),
        dtype=args.dtype,
        num_parameters=sum(p.numel() for p in network.parameters()),
        mean_ms=mean(timings),
        p50_ms=median(timings),
        p95_ms=sorted_timings[p95_idx],
        min_ms=min(timings),
        max_ms=max(timings),
    )


def selected_styles(raw: str) -> List[str]:
    styles = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(styles) - set(SUPPORTED_STYLES))
    if unknown:
        raise ValueError(f"Unsupported styles: {', '.join(unknown)}")
    return styles


def print_table(results: Sequence[ProfileResult]) -> None:
    print(
        "| style | params | mean ms | p50 ms | p95 ms | min ms | max ms | "
        "batch | map | entities | mode | device |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for result in results:
        print(
            f"| {result.style} | {result.num_parameters:,} | "
            f"{result.mean_ms:.3f} | {result.p50_ms:.3f} | {result.p95_ms:.3f} | "
            f"{result.min_ms:.3f} | {result.max_ms:.3f} | {result.batch_size} | "
            f"{result.map_size} | {result.entities} | {result.action_mode} | "
            f"{result.device} |"
        )


def main() -> None:
    args = parse_args()
    styles = selected_styles(args.styles)
    results = [profile_style(style, args) for style in styles]
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        print_table(results)


if __name__ == "__main__":
    main()
