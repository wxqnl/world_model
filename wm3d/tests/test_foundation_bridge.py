from __future__ import annotations

import pytest
import torch

from wm3d.data.foundation_bridge import (
    FoundationBridgeConfig,
    FoundationBridgeError,
    build_foundation_sample,
    pack_grouped_future_action,
)


def _config() -> FoundationBridgeConfig:
    return FoundationBridgeConfig(
        context_frames=2,
        future_frames=4,
        video_height=48,
        video_width=32,
        max_action_groups=3,
        max_action_dim=2,
        max_state_dim=2,
    )


def _batch() -> dict[str, object]:
    cfg = _config()
    batch_size, substeps = 2, 2
    action = torch.arange(
        batch_size * cfg.future_frames * 3 * substeps * 2,
        dtype=torch.float32,
    ).reshape(batch_size, cfg.future_frames, 3, substeps, 2)
    action_mask = torch.zeros_like(action, dtype=torch.bool)
    action_mask[:, :, 0, :, :] = True
    action_mask[:, :, 2, :, 0] = True
    sample_mask = action_mask.any(dim=-1)
    fine_dt = torch.tensor([0.05, 0.10], dtype=torch.float32).view(1, 1, 1, 2)
    state_mask = torch.tensor(
        [[[1, 1], [0, 0], [1, 0]], [[1, 1], [0, 0], [1, 0]]],
        dtype=torch.bool,
    )
    return {
        "direct_rgb_uint8": torch.randint(
            0, 256, (batch_size, 6, 3, 3, 24, 24), dtype=torch.uint8
        ),
        "direct_view_mask": torch.ones(batch_size, 6, 3, dtype=torch.bool),
        "future_factual_fine_action_values": action,
        "future_factual_fine_action_mask": action_mask,
        "future_factual_fine_action_dt": torch.full(
            (batch_size, cfg.future_frames, 3, substeps), 0.0
        ) + fine_dt,
        "future_factual_fine_sample_mask": sample_mask,
        "future_world_boundaries_dt": torch.arange(
            cfg.future_frames + 1, dtype=torch.float32
        ).mul(0.2).expand(batch_size, -1),
        "history_fine_action_values": action[:, : cfg.context_frames],
        "history_fine_action_mask": action_mask[:, : cfg.context_frames],
        "history_fine_action_dt": torch.full(
            (batch_size, cfg.context_frames, 3, substeps), 0.0
        ) + fine_dt,
        "history_fine_sample_mask": sample_mask[:, : cfg.context_frames],
        "history_world_boundaries_dt": torch.tensor(
            [-0.4, -0.2, 0.0], dtype=torch.float32
        ).expand(batch_size, -1),
        "current_state_values": torch.arange(12, dtype=torch.float32).reshape(2, 3, 2),
        "current_state_mask": state_mask,
        "task_text": ["pick the cup", "move the cup"],
        "source_id": torch.zeros(batch_size, dtype=torch.long),
        "action_group_ids": torch.tensor([[1, 0, 12], [1, 0, 12]]),
        "action_group_mask": torch.tensor([[1, 0, 1], [1, 0, 1]], dtype=torch.bool),
        "action_semantic_ids": torch.ones(batch_size, 3, 2, dtype=torch.long),
        "state_semantic_ids": torch.ones(batch_size, 3, 2, dtype=torch.long),
        "embodiment_ids": torch.ones(batch_size, dtype=torch.long),
        "action_normalization_offset": torch.zeros(batch_size, 3, 2),
        "action_normalization_scale": torch.ones(batch_size, 3, 2),
        "state_normalization_offset": torch.zeros(batch_size, 3, 2),
        "state_normalization_scale": torch.ones(batch_size, 3, 2),
    }


def test_foundation_bridge_preserves_native_action_cadence_and_masks() -> None:
    source = _batch()
    result = build_foundation_sample(source, _config())
    assert result["video"].shape == (2, 3, 5, 48, 32)
    assert result["action"].shape == (2, 8, 6)
    assert result["action_element_mask"].shape == (2, 8, 6)
    assert not result["action_is_pad"].any()
    assert result["action_dim_is_pad"].tolist() == [
        [False, False, True, True, False, True],
        [False, False, True, True, False, True],
    ]
    expected = source["future_factual_fine_action_values"].permute(
        0, 1, 3, 2, 4
    ).reshape(2, 8, 6)
    mask = result["action_element_mask"]
    assert torch.equal(result["action"][mask], expected[mask])
    assert torch.equal(
        result["action"][~mask], torch.zeros_like(result["action"])[~mask]
    )
    assert torch.allclose(
        result["action_time_s"][0, :, 0],
        torch.tensor([0.05, 0.10, 0.25, 0.30, 0.45, 0.50, 0.65, 0.70]),
    )
    assert result["proprio"].shape == (2, 1, 6)
    assert result["history_action"].shape == (2, 4, 6)
    assert result["history_action_element_mask"].shape == (2, 4, 6)


def test_foundation_bridge_keeps_sparse_commands_in_their_world_intervals() -> None:
    cfg = FoundationBridgeConfig(max_action_groups=3, max_action_dim=2)
    # Reproduce the real 31-command failure in K8 with four slots/interval.
    # Two extra capacity slots are unused everywhere and may be trimmed.
    values = torch.arange(2 * 8 * 3 * 6 * 2, dtype=torch.float32).reshape(2, 8, 3, 6, 2)
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask[:, :, 0, :4] = True
    mask[0, 2, :, 3] = False
    mask[1, 0, :, 0] = False
    dt = torch.arange(6, dtype=torch.float32).mul(0.04).add(0.01)
    dt = dt.view(1, 1, 1, 6).expand(2, 8, 3, 6)
    boundaries = torch.arange(9, dtype=torch.float32).mul(0.2).expand(2, -1)
    source = {
        "future_factual_fine_action_values": values,
        "future_factual_fine_action_mask": mask,
        "future_factual_fine_action_dt": dt,
        "future_factual_fine_sample_mask": mask.any(dim=-1),
        "future_world_boundaries_dt": boundaries,
    }
    result = pack_grouped_future_action(source, cfg)
    assert result["action"].shape == (2, 32, 6)
    assert (~result["action_is_pad"]).sum(dim=1).tolist() == [31, 31]
    assert result["action_is_pad"][0, 2 * 4 + 3]
    assert result["action_is_pad"][1, 0]
    expected = values[:, :, :, :4].permute(0, 1, 3, 2, 4).reshape(2, 32, 6)
    expected_mask = mask[:, :, :, :4].permute(0, 1, 3, 2, 4).reshape_as(expected)
    expected_time = (boundaries[:, :-1, None, None] + dt)[..., None].expand_as(values)
    expected_time = expected_time[:, :, :, :4].permute(0, 1, 3, 2, 4).reshape_as(expected)
    assert torch.equal(result["action_element_mask"], expected_mask)
    assert torch.equal(result["action"], expected.masked_fill(~expected_mask, 0))
    assert torch.equal(result["action_time_s"], expected_time.masked_fill(~expected_mask, 0))
    # The command immediately after a hole still belongs to the next interval.
    assert result["action_time_s"][0, 3 * 4, 0] == expected_time[0, 3 * 4, 0]

    # Exercise the next production boundary, not only the packing helper.
    from wm3d.models.foundation_world_model import ActionTimeConditioner

    conditioner = ActionTimeConditioner(hidden_dim=12, fourier_dim=8)
    conditioner.mlp[-1].weight.data.fill_(0.1)
    update = conditioner(result["action_time_s"], result["action_element_mask"])
    assert torch.isfinite(update).all()
    assert update[result["action_is_pad"]].count_nonzero() == 0
    assert update[~result["action_is_pad"]].count_nonzero() > 0


def test_foundation_bridge_rejects_cross_source_microbatch() -> None:
    batch = _batch()
    batch["source_id"] = torch.tensor([0, 1])
    with pytest.raises(FoundationBridgeError, match="one source"):
        build_foundation_sample(batch, _config())


def test_foundation_bridge_rejects_command_before_future_start() -> None:
    batch = _batch()
    batch["future_factual_fine_action_dt"][0, 0, :, 0] = -0.01
    with pytest.raises(FoundationBridgeError, match="non-negative"):
        build_foundation_sample(batch, _config())


def test_foundation_bridge_rejects_nonfactorizable_element_mask() -> None:
    batch = _batch()
    mask = batch["future_factual_fine_action_mask"]
    sample_mask = batch["future_factual_fine_sample_mask"]
    mask[0, 0, 0, 0, 1] = False
    sample_mask[0, 0, 0, 0] = True
    with pytest.raises(FoundationBridgeError, match="time/dimension masks"):
        build_foundation_sample(batch, _config())
