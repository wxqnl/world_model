from __future__ import annotations

import torch

from wm3d.models.native_world_model import (
    MultiViewTokenFuser,
    NativeWorldModelConfig,
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
