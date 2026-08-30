from __future__ import annotations

import torch

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
