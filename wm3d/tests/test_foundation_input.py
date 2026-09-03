from __future__ import annotations

import torch
from torch import nn

from wm3d.data.foundation_bridge import FoundationBridgeConfig
from wm3d.data.foundation_input import (
    FoundationContextStore,
    FoundationInputAdapter,
)


class _VGGT(nn.Module):
    def forward(self, images, view_mask, **_kwargs):
        assert images.shape[:3] == (2, 1, 3)
        assert view_mask.shape == (2, 1, 3)
        return {"view_tokens": torch.ones(2, 1, 3, 144, 2048)}


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


def test_input_adapter_binds_wan_context_and_final_future_vggt() -> None:
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
    assert output["vggt_future_tokens"].shape == (2, 3, 144, 2048)
    assert adapter.metrics["calls"] == 1
