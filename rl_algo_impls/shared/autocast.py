import logging
import warnings
from contextlib import contextmanager

import torch

AMP_DTYPE_DEFAULT = "bfloat16"


def resolve_amp_dtype(dtype: str, device: torch.device) -> torch.dtype:
    """Resolve AMP dtype string to torch.dtype with hardware compatibility checks.

    Args:
        dtype: "bfloat16" or "float16"
        device: target torch device

    Returns:
        torch.bfloat16 or torch.float16

    Warns if the requested dtype is not supported and falls back.
    """
    if dtype == "bfloat16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        elif device.type == "cpu" and hasattr(torch, "bfloat16"):
            return torch.bfloat16
        else:
            warnings.warn(
                "bfloat16 not supported on this device, falling back to float16"
            )
            return torch.float16
    elif dtype == "float16":
        if device.type == "cuda":
            return torch.float16
        elif device.type == "cpu":
            warnings.warn("float16 AMP not meaningful on CPU, autocast disabled")
            return None  # type: ignore
        else:
            return torch.float16
    else:
        logging.warning(f"Unknown AMP dtype {dtype}, disabling autocast")
        return None  # type: ignore


@contextmanager
def maybe_autocast(
    autocast_enabled: bool,
    device: torch.device,
    amp_dtype: str = AMP_DTYPE_DEFAULT,
):
    """Context manager for automatic mixed precision (AMP).

    Args:
        autocast_enabled: Whether to enable autocast
        device: Target device
        amp_dtype: "bfloat16" (default, recommended) or "float16"
                   float16 requires care: log_prob and entropy should stay in float32
                   to avoid numerical overflow (float16 min ~ -65504). This context
                   manager does NOT enforce that; the caller (e.g. PPO) must handle
                   precision-sensitive operations manually when using float16.
    """
    if autocast_enabled and device.type == "cuda":
        dtype = resolve_amp_dtype(amp_dtype, device)
        if dtype is not None:
            with torch.autocast(device.type, dtype=dtype):
                yield
            return
    yield


def safe_amp_forward(
    autocast_enabled: bool,
    device: torch.device,
    amp_dtype: str,
    policy,
    obs,
    actions,
    action_masks,
    memory_state=None,
    episode_starts=None,
):
    """Forward pass with AMP-safe log_prob and entropy for PPO.

    When using float16 autocast, forces log_prob and entropy back to float32
    to prevent numerical issues (log_prob can go below float16 min of ~-65504).
    bfloat16 is naturally safe (same dynamic range as float32).
    """
    with maybe_autocast(autocast_enabled, device, amp_dtype=amp_dtype):
        new_logprobs, entropy, new_values = policy(
            obs,
            actions,
            action_masks=action_masks,
            memory_state=memory_state,
            episode_starts=episode_starts,
        )
    if amp_dtype == "float16":
        new_logprobs = new_logprobs.float()
        entropy = entropy.float()
    return new_logprobs, entropy, new_values
