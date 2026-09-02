from __future__ import annotations

import torch

from wm3d.models.native_world_model import (
    MultiViewTokenFuser,
    NativeContextRGBImageDecoder,
    NativeOriginalV7ContextRGBImageDecoder,
    NativeRGBDecoder,
    NativeWorldModelConfig,
    PerViewAppearanceDynamics,
)
from wm3d.training.distributed_runtime import _materialize_meta_shards


def test_native_transformer_decoder_materializes_from_meta() -> None:
    with torch.device("meta"):
        layer = torch.nn.TransformerDecoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
            norm_first=True,
        )

    _materialize_meta_shards(layer, torch.device("cpu"))

    state = layer.state_dict()
    assert state
    assert all(not value.is_meta for value in state.values())
    assert all(bool(torch.isfinite(value).all()) for value in state.values())
    assert layer.self_attn.in_proj_weight.abs().sum().item() > 0.0
    assert layer.multihead_attn.in_proj_weight.abs().sum().item() > 0.0


def test_original_v7_auxiliary_projection_stays_zero_after_meta_materialization() -> None:
    config = NativeWorldModelConfig(
        token_dim=16,
        state_hidden=32,
        view_hidden=16,
        view_heads=4,
        num_views=2,
        rgb_context_enabled=True,
        rgb_original_v7_context=True,
    )
    with torch.device("meta"):
        fuser = MultiViewTokenFuser(config)

    _materialize_meta_shards(fuser, torch.device("cpu"))

    assert fuser.output_projection.weight.count_nonzero().item() == 0
    assert fuser.residual_gate.shape == (1,)
    assert torch.equal(fuser.residual_gate, torch.ones_like(fuser.residual_gate))


def test_legacy_multiview_gate_stays_zero_after_meta_materialization() -> None:
    config = NativeWorldModelConfig(
        token_dim=16,
        state_hidden=32,
        view_hidden=16,
        view_heads=4,
        num_views=2,
        rgb_original_v7_context=False,
    )
    with torch.device("meta"):
        fuser = MultiViewTokenFuser(config)

    _materialize_meta_shards(fuser, torch.device("cpu"))

    assert fuser.gate.weight.count_nonzero().item() == 0


def test_original_v7_motion_head_keeps_closed_initialization_after_meta() -> None:
    config = NativeWorldModelConfig(
        P=16,
        K=8,
        token_dim=32,
        task_dim=32,
        state_hidden=32,
        rgb_hidden=64,
        rgb_size=256,
        rgb_context_enabled=True,
        rgb_original_v7_context=True,
    )
    with torch.device("meta"):
        decoder = NativeOriginalV7ContextRGBImageDecoder(config)

    _materialize_meta_shards(decoder, torch.device("cpu"))

    assert decoder.motion_head.weight.count_nonzero().item() == 0
    assert torch.equal(
        decoder.motion_head.bias,
        torch.full_like(decoder.motion_head.bias, -4.0),
    )


def test_context_motion_head_keeps_closed_initialization_after_meta() -> None:
    config = NativeWorldModelConfig(
        P=16,
        K=8,
        token_dim=32,
        task_dim=32,
        state_hidden=32,
        rgb_hidden=64,
        rgb_size=256,
        rgb_context_enabled=True,
    )
    with torch.device("meta"):
        decoder = NativeContextRGBImageDecoder(config)

    _materialize_meta_shards(decoder, torch.device("cpu"))

    assert decoder.motion_head.weight.count_nonzero().item() == 0
    assert torch.equal(
        decoder.motion_head.bias,
        torch.full_like(decoder.motion_head.bias, -4.0),
    )


def test_flow_aligned_appearance_output_stays_zero_after_meta() -> None:
    config = NativeWorldModelConfig(
        P=16,
        K=8,
        token_dim=32,
        state_hidden=32,
        num_views=2,
        appearance_enabled=True,
        appearance_P=16,
        appearance_hidden=32,
        appearance_layers=1,
        appearance_heads=4,
        appearance_flow_aligned_detail=True,
        activation_checkpointing=False,
    )
    with torch.device("meta"):
        dynamics = PerViewAppearanceDynamics(config)

    _materialize_meta_shards(dynamics, torch.device("cpu"))

    assert dynamics.output.weight.count_nonzero().item() == 0


def test_original_v7_rgb_view_buffer_stays_zero_after_meta() -> None:
    config = NativeWorldModelConfig(
        P=16,
        K=8,
        token_dim=32,
        task_dim=32,
        state_hidden=32,
        rgb_hidden=64,
        rgb_size=256,
        rgb_decode_indices=(0, 1),
        rgb_context_enabled=True,
        rgb_original_v7_context=True,
        activation_checkpointing=False,
    )
    with torch.device("meta"):
        decoder = NativeRGBDecoder(config)

    _materialize_meta_shards(decoder, torch.device("cpu"))

    assert decoder.view_embed.count_nonzero().item() == 0
    assert bool(torch.isfinite(decoder.view_embed).all())
