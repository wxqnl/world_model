from __future__ import annotations

import torch
from torch import nn

from wm3d.models.foundation_world_model import (
    ActionTimeConditioner,
    BoundVGGTTeacher,
    FoundationWorldModel,
    FoundationWorldModelConfig,
    GroupedRobotConditioner,
    Native3DRolloutHead,
    Native3DStateConditioner,
    ResidualContextFusion,
)


def _config() -> FoundationWorldModelConfig:
    return FoundationWorldModelConfig(
        wsa_text_dim=32,
        context_fusion_heads=4,
        robot_conditioning_hidden=8,
        action_time_fourier_dim=8,
        max_action_groups=2,
        max_action_dim=3,
        max_state_dim=2,
        max_group_id=16,
        max_embodiments=16,
        max_action_semantic_id=16,
        max_state_semantic_id=16,
    )


def test_context_fusion_starts_as_exact_identity_then_uses_extra_tokens() -> None:
    fusion = ResidualContextFusion(32, 16, 4)
    base = torch.randn(2, 5, 32)
    extra = torch.randn(2, 4, 16)
    base_mask = torch.ones(2, 5, dtype=torch.bool)
    extra_mask = torch.ones(2, 4, dtype=torch.bool)
    initial = fusion(base, base_mask, extra, extra_mask)
    assert torch.equal(initial, base)
    fusion.gate.data.fill_(0.1)
    changed = fusion(base, base_mask, extra, extra_mask)
    assert not torch.equal(changed, base)


def test_action_time_conditioner_preserves_zero_init_and_receives_gradient() -> None:
    conditioner = ActionTimeConditioner(hidden_dim=12, fourier_dim=8)
    time_s = torch.arange(8, dtype=torch.float32)[None, :, None].mul(0.05)
    time_s = time_s.expand(2, -1, 6)
    mask = torch.ones_like(time_s, dtype=torch.bool)
    output = conditioner(time_s, mask)
    assert torch.equal(output, torch.zeros_like(output))
    output.sum().backward()
    assert conditioner.mlp[-1].weight.grad is not None
    assert bool((conditioner.mlp[-1].weight.grad != 0).any())


def test_grouped_robot_conditioner_preserves_semantics_and_history_mask() -> None:
    config = _config()
    module = GroupedRobotConditioner(config)
    batch_size, history = 2, 4
    action_valid = torch.tensor(
        [[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool
    )
    state_valid = torch.tensor(
        [[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.bool
    )
    history_mask = action_valid[:, None].expand(-1, history, -1).clone()
    batch = {
        "action_group_ids": torch.tensor([[1, 0], [1, 0]]),
        "action_group_mask": torch.tensor([[1, 0], [1, 0]], dtype=torch.bool),
        "action_semantic_ids": torch.ones(batch_size, 2, 3, dtype=torch.long),
        "state_semantic_ids": torch.ones(batch_size, 2, 2, dtype=torch.long),
        "embodiment_ids": torch.ones(batch_size, dtype=torch.long),
        "action_dim_is_pad": ~action_valid,
        "proprio_dim_is_pad": ~state_valid,
        "action_normalization_offset": torch.zeros(batch_size, 2, 3),
        "action_normalization_scale": torch.ones(batch_size, 2, 3),
        "state_normalization_offset": torch.zeros(batch_size, 2, 2),
        "state_normalization_scale": torch.ones(batch_size, 2, 2),
        "history_action": torch.randn(batch_size, history, 6),
        "history_action_element_mask": history_mask,
        "history_action_time_s": torch.tensor(
            [-0.20, -0.15, -0.10, -0.05]
        )[None, :, None].expand(batch_size, -1, 6),
    }
    tokens, mask = module(batch)
    expected_length = 1 + config.action_width + config.state_width + history
    assert tokens.shape == (batch_size, expected_length, config.wsa_text_dim)
    assert mask.shape == tokens.shape[:2]
    assert mask[:, -history:].all()
    assert not bool(mask[:, 1 + 3].any())


def test_vggt_teacher_upsamples_spatial_grid_without_gradients() -> None:
    teacher = BoundVGGTTeacher(output_tokens_per_view=16, layers=2)
    source = torch.randn(2, 3, 4, 8, requires_grad=True)
    teacher.bind(source)
    outputs = teacher(torch.empty(0))
    teacher.clear()
    assert len(outputs) == 2
    assert outputs[0].shape == (2, 48, 8)
    assert not outputs[0].requires_grad


def test_native_3d_state_is_exact_wsa_identity_then_uses_each_view() -> None:
    module = Native3DStateConditioner(
        token_dim=8,
        hidden_dim=12,
        num_views=3,
        tokens_per_view=4,
    )
    tokens = torch.randn(2, 3, 4, 8)
    view_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    initial = module(tokens, view_mask)
    assert torch.equal(initial, torch.zeros_like(initial))
    module.gate.data.fill_(0.1)
    changed = module(tokens, view_mask).reshape(2, 3, 4, 12)
    assert bool((changed[:, :2] != 0).any())
    assert torch.equal(changed[1, 2], torch.zeros_like(changed[1, 2]))


def test_native_3d_rollout_decodes_all_horizons_with_one_sampled_target() -> None:
    head = Native3DRolloutHead(
        hidden_dim=12,
        token_dim=8,
        num_views=3,
        context_tokens_per_view=4,
        rollout_tokens_per_view=4,
        horizons=3,
        action_dim=6,
    )
    shared = torch.randn(2, 12, 12, requires_grad=True)
    target = torch.randn(2, 3, 4, 8)
    current = torch.randn(2, 3, 4, 8)
    action = torch.randn(2, 6, 6, requires_grad=True)
    action_time = torch.arange(6, dtype=torch.float32)[None, :, None].expand_as(
        action
    )
    action_mask = torch.ones_like(action, dtype=torch.bool)
    prediction = head(shared, current, action, action_time, action_mask)
    assert torch.equal(prediction, current[:, None].expand_as(prediction))
    loss, metrics = head.loss(
        prediction,
        target,
        torch.tensor([0, 2]),
        torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
        current,
    )
    assert prediction.shape == (2, 3, 3, 4, 8)
    assert torch.isfinite(loss)
    assert set(metrics) == {
        "loss_native_3d_rollout_delta_smooth_l1",
        "native_3d_rollout_direction",
        "native_3d_rollout_amplitude_ratio",
    }
    loss.backward()
    assert head.output.weight.grad is not None
    assert bool((head.output.weight.grad != 0).any())
    torch.optim.SGD(head.parameters(), lr=0.01).step()
    shared.grad = None
    action.grad = None
    prediction = head(shared, current, action, action_time, action_mask)
    loss, _metrics = head.loss(
        prediction,
        target,
        torch.tensor([0, 2]),
        torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
        current,
    )
    loss.backward()
    assert shared.grad is not None
    assert bool((shared.grad != 0).any())
    assert action.grad is not None
    assert bool((action.grad != 0).any())


class _Backend(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.sample = None
        self.calls = 0

    def forward(
        self,
        sample,
        vggt_target,
        current_native_3d,
        current_native_3d_view_mask,
        rollout_target,
        rollout_horizon,
        rollout_view_mask,
    ):
        del current_native_3d_view_mask, rollout_horizon, rollout_view_mask
        self.calls += 1
        self.sample = sample
        loss = (
            sample["context"].float().square().mean()
            + vggt_target.float().square().mean()
            + current_native_3d.float().square().mean()
            + rollout_target.float().square().mean()
        ) * self.scale
        return {"total": loss, "loss_video": loss.detach()}

    def infer_action(self, **kwargs):
        self.sample = kwargs
        return torch.full(
            kwargs["action_time_s"].shape,
            0.5,
            dtype=torch.float32,
        )


def test_foundation_model_keeps_wsa_context_and_robot_extensions_bounded() -> None:
    config = _config()
    backend = _Backend()
    model = FoundationWorldModel(config, backend=backend)
    batch_size, history = 2, 4
    action_valid = torch.tensor(
        [[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool
    )
    state_valid = torch.tensor(
        [[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.bool
    )
    batch = {
        "wan_context": torch.randn(batch_size, 5, 32),
        "wan_context_mask": torch.ones(batch_size, 5, dtype=torch.bool),
        "action_group_ids": torch.tensor([[1, 0], [1, 0]]),
        "action_group_mask": torch.tensor([[1, 0], [1, 0]], dtype=torch.bool),
        "action_semantic_ids": torch.ones(batch_size, 2, 3, dtype=torch.long),
        "state_semantic_ids": torch.ones(batch_size, 2, 2, dtype=torch.long),
        "embodiment_ids": torch.ones(batch_size, dtype=torch.long),
        "action_normalization_offset": torch.zeros(batch_size, 2, 3),
        "action_normalization_scale": torch.ones(batch_size, 2, 3),
        "state_normalization_offset": torch.zeros(batch_size, 2, 2),
        "state_normalization_scale": torch.ones(batch_size, 2, 2),
        "history_action": torch.randn(batch_size, history, 6),
        "history_action_element_mask": action_valid[:, None].expand(-1, history, -1),
        "history_action_time_s": torch.tensor(
            [-0.20, -0.15, -0.10, -0.05]
        )[None, :, None].expand(batch_size, -1, 6),
        "video": torch.randn(batch_size, 3, 9, 48, 32),
        "image_is_pad": torch.zeros(batch_size, 9, dtype=torch.bool),
        "action": torch.randn(batch_size, 8, 6),
        "action_is_pad": torch.zeros(batch_size, 8, dtype=torch.bool),
        "action_dim_is_pad": ~action_valid,
        "action_time_s": torch.arange(8, dtype=torch.float32)[None, :, None]
        .mul(0.05)
        .expand(batch_size, -1, 6),
        "action_element_mask": action_valid[:, None].expand(-1, 8, -1),
        "proprio": torch.randn(batch_size, 1, 4),
        "proprio_dim_is_pad": ~state_valid,
        "future_3d_images": torch.randn(batch_size, 3, 3, 24, 24),
        "future_3d_img_masks": torch.ones(batch_size, 3, dtype=torch.bool),
        "vggt_future_tokens": torch.randn(batch_size, 3, 4, 8),
        "vggt_current_tokens": torch.randn(batch_size, 3, 4, 8),
        "vggt_current_view_mask": torch.ones(batch_size, 3, dtype=torch.bool),
        "vggt_rollout_target_tokens": torch.randn(batch_size, 3, 4, 8),
        "vggt_rollout_target_horizon": torch.tensor([0, 3]),
        "vggt_rollout_target_view_mask": torch.ones(
            batch_size, 3, dtype=torch.bool
        ),
    }
    output = model(batch)
    output["total"].backward()
    assert backend.sample is not None
    assert backend.calls == 1
    assert backend.sample["context"].shape == (batch_size, 5, 32)
    assert model.robot_fusion.gate.grad is not None
    assert backend.scale.grad is not None


def test_foundation_action_serving_reuses_training_context_and_denormalizes() -> None:
    config = _config()
    backend = _Backend()
    model = FoundationWorldModel(config, backend=backend)
    action_valid = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    state_valid = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    history = 4
    batch = {
        "wan_context": torch.randn(1, 5, 32),
        "wan_context_mask": torch.ones(1, 5, dtype=torch.bool),
        "action_group_ids": torch.tensor([[1, 0]]),
        "action_group_mask": torch.tensor([[1, 0]], dtype=torch.bool),
        "action_semantic_ids": torch.ones(1, 2, 3, dtype=torch.long),
        "state_semantic_ids": torch.ones(1, 2, 2, dtype=torch.long),
        "embodiment_ids": torch.ones(1, dtype=torch.long),
        "action_normalization_offset": torch.ones(1, 2, 3),
        "action_normalization_scale": torch.full((1, 2, 3), 2.0),
        "state_normalization_offset": torch.zeros(1, 2, 2),
        "state_normalization_scale": torch.ones(1, 2, 2),
        "history_action": torch.randn(1, history, 6),
        "history_action_element_mask": action_valid[:, None].expand(-1, history, -1),
        "history_action_time_s": torch.tensor(
            [-0.20, -0.15, -0.10, -0.05]
        )[None, :, None].expand(1, -1, 6),
        "video": torch.randn(1, 3, 9, 48, 32),
        "action_dim_is_pad": ~action_valid,
        "action_time_s": torch.arange(8, dtype=torch.float32)[None, :, None]
        .mul(0.05)
        .expand(1, -1, 6),
        "action_element_mask": action_valid[:, None].expand(-1, 8, -1),
        "proprio": torch.randn(1, 1, 4),
        "proprio_dim_is_pad": ~state_valid,
        "vggt_current_tokens": torch.randn(1, 3, 4, 8),
        "vggt_current_view_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    result = model.predict_action(batch, num_inference_steps=4, seed=7)
    assert backend.sample is not None
    assert torch.equal(backend.sample["input_image"], batch["video"][:, :, 0])
    assert backend.sample["proprio"].shape == (1, 4)
    assert backend.sample["num_inference_steps"] == 4
    assert backend.sample["seed"] == 7
    assert result["action_normalized"].shape == (1, 8, 6)
    assert result["action_physical"].shape == (1, 8, 2, 3)
    assert torch.equal(result["action_physical"][..., 0, :], torch.full((1, 8, 3), 2.0))
    assert torch.equal(result["action_physical"][..., 1, :], torch.zeros(1, 8, 3))
