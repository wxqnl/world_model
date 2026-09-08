from __future__ import annotations

import torch
from torch import nn

from wm3d.data.foundation_bridge import FoundationBridgeConfig
from wm3d.data.foundation_input import (
    FoundationContextStore,
    FoundationInputAdapter,
)


class _VGGT(nn.Module):
    def forward(self, images, view_mask, **kwargs):
        assert images.shape[:3] == (2, 3, 3)
        assert view_mask.shape == (2, 3, 3)
        for name in ("geometry_row_mask", "appearance_row_mask", "rgb_row_mask"):
            assert kwargs[name].shape == (2, 3)
            assert kwargs[name].dtype == torch.bool
            assert not kwargs[name].any()
        tokens = torch.ones(2, 3, 3, 144, 2048)
        tokens[:, 1].mul_(2)
        tokens[:, 2].mul_(3)
        return {"view_tokens": tokens}


def _batch() -> dict[str, object]:
    batch_size, context, future, groups, substeps, action_dim = 2, 2, 4, 2, 2, 3
    length = context + future
    action = torch.randn(batch_size, future, groups, substeps, action_dim)
    action_mask = torch.zeros_like(action, dtype=torch.bool)
    action_mask[:, :, 0] = True
    history = torch.randn(batch_size, context, groups, substeps, action_dim)
    history_mask = torch.zeros_like(history, dtype=torch.bool)
    history_mask[:, :, 0] = True
    view_mask = torch.ones(batch_size, length, 3, dtype=torch.bool)
    view_mask[1, :, 2] = False
    fine_dt = torch.tensor([0.05, 0.10]).view(1, 1, 1, substeps)
    return {
        "direct_rgb_uint8": torch.randint(
            0, 256, (batch_size, length, 3, 3, 16, 16), dtype=torch.uint8
        ),
        "direct_view_mask": view_mask,
        "future_factual_fine_action_values": action,
        "future_factual_fine_action_mask": action_mask,
        "future_factual_fine_action_dt": torch.zeros(
            batch_size, future, groups, substeps
        ) + fine_dt,
        "future_factual_fine_sample_mask": action_mask.any(dim=-1),
        "future_world_boundaries_dt": torch.arange(
            future + 1, dtype=torch.float32
        ).mul(0.2).expand(batch_size, -1),
        "history_fine_action_values": history,
        "history_fine_action_mask": history_mask,
        "history_fine_action_dt": torch.zeros(
            batch_size, context, groups, substeps
        ) + fine_dt,
        "history_fine_sample_mask": history_mask.any(dim=-1),
        "history_world_boundaries_dt": torch.tensor(
            [-0.4, -0.2, 0.0]
        ).expand(batch_size, -1),
        "current_state_values": torch.randn(batch_size, groups, 2),
        "current_state_mask": torch.tensor(
            [[[1, 1], [0, 0]], [[1, 1], [0, 0]]], dtype=torch.bool
        ),
        "task_text": ["pick", "place"],
        "source_id": torch.zeros(batch_size, dtype=torch.long),
        "sample_index": torch.tensor([0, 3], dtype=torch.long),
        "action_group_ids": torch.tensor([[1, 0], [1, 0]]),
        "action_group_mask": torch.tensor([[1, 0], [1, 0]], dtype=torch.bool),
        "action_semantic_ids": torch.ones(batch_size, groups, action_dim, dtype=torch.long),
        "state_semantic_ids": torch.ones(batch_size, groups, 2, dtype=torch.long),
        "embodiment_ids": torch.ones(batch_size, dtype=torch.long),
        "action_normalization_offset": torch.zeros(batch_size, groups, action_dim),
        "action_normalization_scale": torch.ones(batch_size, groups, action_dim),
        "state_normalization_offset": torch.zeros(batch_size, groups, 2),
        "state_normalization_scale": torch.ones(batch_size, groups, 2),
    }


def test_input_adapter_binds_current_sampled_k8_and_final_vggt() -> None:
    bridge = FoundationBridgeConfig(
        context_frames=2,
        future_frames=4,
        video_height=48,
        video_width=32,
        max_action_groups=2,
        max_action_dim=3,
        max_state_dim=2,
    )
    store = FoundationContextStore(
        ["pick", "place"],
        torch.randn(2, 5, 32),
        torch.ones(2, 5, dtype=torch.bool),
    )
    adapter = FoundationInputAdapter(
        bridge,
        context_store=store,
        vggt_encoder=_VGGT(),
        device="cpu",
    )
    output = adapter.materialize(_batch())
    assert output["video"].shape == (2, 3, 5, 48, 32)
    assert output["wan_context"].shape == (2, 5, 32)
    assert output["vggt_current_tokens"].shape == (2, 3, 144, 2048)
    assert not torch.is_inference(output["vggt_current_tokens"])
    assert output["vggt_rollout_target_tokens"].shape == (2, 3, 64, 2048)
    assert not torch.is_inference(output["vggt_rollout_target_tokens"])
    assert output["vggt_rollout_target_horizon"].tolist() == [0, 3]
    assert not torch.is_inference(output["vggt_rollout_target_horizon"])
    assert output["vggt_future_tokens"].shape == (2, 3, 144, 2048)
    assert torch.equal(
        output["vggt_future_tokens"],
        torch.full_like(output["vggt_future_tokens"], 3),
    )
    assert adapter.metrics["calls"] == 1
    assert adapter.metrics["vggt_frames"] == 6


class _IndependentVGGT(nn.Module):
    def __init__(self):
        super().__init__()
        self.rows = 0

    def forward(self, images, view_mask, **kwargs):
        self.rows += images.shape[0] * images.shape[1]
        value = images.mean(dim=(-1, -2, -3)).masked_fill(~view_mask, 0.0)
        # Cross-view dependence must be kept in each cache entry.
        value = value + value.mean(dim=-1, keepdim=True)
        return {"view_tokens": value[..., None, None].expand(*value.shape, 144, 2048).clone()}


def test_frozen_native_cache_preserves_values_masks_and_bounded_storage() -> None:
    bridge = FoundationBridgeConfig(
        context_frames=2, future_frames=4, video_height=48, video_width=32,
        max_action_groups=2, max_action_dim=3, max_state_dim=2,
    )
    store = FoundationContextStore(["pick", "place"], torch.randn(2, 5, 32),
                                   torch.ones(2, 5, dtype=torch.bool))
    encoder = _IndependentVGGT()
    adapter = FoundationInputAdapter(bridge, context_store=store, vggt_encoder=encoder,
                                    device="cpu", vggt_cache_bytes=64 * 1024**2)
    batch = _batch()
    batch["direct_frame_keys"] = torch.arange(12).reshape(2, 6)
    adapter.vggt_cache_limit = 0
    reference = adapter.materialize(batch)
    adapter.vggt_cache_limit = 64 * 1024**2
    first = adapter.materialize(batch)
    computed = encoder.rows
    hit = adapter.materialize(batch)
    assert encoder.rows == computed
    assert adapter.vggt_cache_hits > 0
    for key, expected in reference.items():
        if isinstance(expected, torch.Tensor):
            assert torch.equal(first[key], expected), key
            assert torch.equal(hit[key], expected), key
    assert not torch.is_inference(hit["vggt_current_tokens"])
    # The same physical frame with a different camera subset is not a cache hit.
    batch["direct_view_mask"][0, :, 2] = False
    changed = adapter.materialize(batch)
    assert encoder.rows > computed
    adapter.vggt_cache_limit = 0
    changed_reference = adapter.materialize(batch)
    assert torch.equal(changed["vggt_current_tokens"], changed_reference["vggt_current_tokens"])
    adapter._vggt_cache.clear()
    adapter.vggt_cache_bytes = 0
    adapter.vggt_cache_limit = 1
    uncached = adapter.materialize(batch)
    assert adapter.vggt_cache_bytes == 0
    assert not adapter._vggt_cache
    assert torch.equal(uncached["vggt_future_tokens"], changed_reference["vggt_future_tokens"])
