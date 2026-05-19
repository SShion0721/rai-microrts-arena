import argparse
import importlib.util
import sys
from pathlib import Path


def _load_profile_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "profile_microrts_policy.py"
    )
    spec = importlib.util.spec_from_file_location(
        "profile_microrts_policy", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_ints_accepts_powershell_friendly_values():
    module = _load_profile_module()

    assert module.parse_ints(["8,16"]) == [8, 16]
    assert module.parse_ints(["8", "16"]) == [8, 16]


def test_profile_style_runs_minimal_hybrid_forward_on_cpu():
    module = _load_profile_module()
    args = argparse.Namespace(
        map_size=4,
        batch_size=1,
        channels=60,
        entities=2,
        warmup=0,
        iters=1,
        seed=1,
        device="cpu",
        dtype="float32",
        action_mode="distribution",
        channels_per_level=["8", "16"],
        encoder_layers=1,
        encoder_embed_dim=16,
        encoder_attention_heads=4,
        encoder_feed_forward_dim=32,
        double_cone_channels=8,
        double_cone_pooled_channels=16,
        torch_compile=False,
    )

    result = module.profile_style("hybrid_entity_grid", args)

    assert result.style == "hybrid_entity_grid"
    assert result.map_size == 4
    assert result.batch_size == 1
    assert result.num_parameters > 0
    assert result.mean_ms >= 0


def test_profile_style_runs_minimal_hierarchical_forward_on_cpu():
    module = _load_profile_module()
    args = argparse.Namespace(
        map_size=4,
        batch_size=1,
        channels=60,
        entities=2,
        warmup=0,
        iters=1,
        seed=1,
        device="cpu",
        dtype="float32",
        action_mode="distribution",
        channels_per_level=["8", "16"],
        encoder_layers=1,
        encoder_embed_dim=16,
        encoder_attention_heads=4,
        encoder_feed_forward_dim=32,
        double_cone_channels=8,
        double_cone_pooled_channels=16,
        torch_compile=False,
    )

    result = module.profile_style("hierarchical_hybrid_entity_grid", args)

    assert result.style == "hierarchical_hybrid_entity_grid"
    assert result.map_size == 4
    assert result.batch_size == 1
    assert result.num_parameters > 0
    assert result.mean_ms >= 0
