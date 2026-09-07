from __future__ import annotations

from types import SimpleNamespace

import pytest
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
    WSABackendAdapter,
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


def test_action_time_conditioner_ignores_masked_slots_after_learning() -> None:
    conditioner = ActionTimeConditioner(hidden_dim=12, fourier_dim=8)
    conditioner.mlp[-1].weight.data.fill_(0.1)
    times = torch.arange(4, dtype=torch.float32)[None, :, None].mul(0.05)
    times = times.expand(1, -1, 3).clone().requires_grad_()
    mask = torch.ones_like(times, dtype=torch.bool)
    mask[:, 1] = False
    mask[..., -1] = False
    poisoned = times.masked_fill(~mask, float("nan"))
    update = conditioner(poisoned, mask)
    valid = mask.any(dim=-1)
    expected = conditioner(times[:, valid[0]], mask[:, valid[0]])
    torch.testing.assert_close(update[:, valid[0]], expected)
    assert update[~valid].count_nonzero() == 0
    update.square().sum().backward()
    assert torch.isfinite(times.grad).all()
    assert times.grad[~mask].count_nonzero() == 0
    assert times.grad[mask].count_nonzero() > 0


def test_upstream_action_loss_excludes_time_padding_without_rescaling_samples() -> None:
    import os
    from pathlib import Path

    from wm3d.models.foundation_world_model import _load_upstream_symbols

    source = Path(os.environ.get("WM3D_WSA_SOURCE_ROOT", "/data/Minko/third_party/WSA"))
    if not source.is_dir():
        pytest.skip("official WSA source is not installed")
    _load_upstream_symbols(source)
    from lerobot.policies.WSA_Large.core.models.wan22.wsa_large import (
        masked_action_time_mean,
    )

    prediction = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    prediction = prediction.clone().requires_grad_()
    token_loss = prediction.square().mean(dim=-1)
    pad = torch.tensor([[False, True, False, False], [True, False, True, False]])
    actual = masked_action_time_mean(token_loss, pad)
    expected = torch.stack([token_loss[i, ~pad[i]].mean() for i in range(2)])
    torch.testing.assert_close(actual, expected)
    assert torch.equal(masked_action_time_mean(token_loss, None), token_loss.mean(dim=1))
    actual.mean().backward()
    assert prediction.grad[pad].count_nonzero() == 0
    assert prediction.grad[~pad].count_nonzero() > 0


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


def test_native_3d_rollout_ignores_extra_masked_command_capacity() -> None:
    torch.manual_seed(73)
    head = Native3DRolloutHead(
        hidden_dim=12, token_dim=8, num_views=3,
        context_tokens_per_view=4, rollout_tokens_per_view=4,
        horizons=3, action_dim=6,
    )
    with torch.no_grad():
        head.output.weight.normal_(std=0.1)
    shared = torch.randn(1, 12, 12)
    current = torch.randn(1, 3, 4, 8)
    action = torch.randn(1, 3, 2, 6)
    time_s = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2, 1).expand_as(action)
    mask = torch.ones_like(action, dtype=torch.bool)
    mask[:, 1, 1] = False
    baseline = head(shared, current, action.flatten(1, 2), time_s.flatten(1, 2), mask.flatten(1, 2))

    padded_action = torch.full((1, 3, 4, 6), 1234.0)
    padded_time = torch.full_like(padded_action, 9876.0)
    padded_mask = torch.zeros_like(padded_action, dtype=torch.bool)
    padded_action[:, :, :2] = action
    padded_time[:, :, :2] = time_s
    padded_mask[:, :, :2] = mask
    padded_action.requires_grad_()
    prediction = head(
        shared, current, padded_action.flatten(1, 2),
        padded_time.flatten(1, 2), padded_mask.flatten(1, 2),
    )
    torch.testing.assert_close(prediction, baseline, atol=1e-6, rtol=1e-5)
    prediction.square().mean().backward()
    gradient = padded_action.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient[padded_mask].count_nonzero() > 0
    assert gradient[~padded_mask].count_nonzero() == 0


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("known", [False, True])
def test_world_sampler_distinguishes_joint_and_known_action(
    fail: bool, known: bool
) -> None:
    # A protocol spy checks the production adapter, not learned video quality.
    # Real checkpoint/sample quality is measured by foundation-demo separately.
    seen = []

    class Sampler(nn.Module):
        train_action_scheduler = SimpleNamespace(num_train_timesteps=1000)

        def _predict_joint_noise(self, **kwargs):
            seen.append(kwargs)
            if fail:
                raise RuntimeError("sampling failure")
            return torch.zeros(1), torch.zeros(1)

        def infer_joint(self, **kwargs):
            assert kwargs["action"] is None
            assert kwargs["test_action_with_infer_action"] is False
            for time in (1000.0, 500.0, 50.0):
                self._predict_joint_noise(
                    latents_action=torch.randn(1, 4, 6),
                    timestep_action=torch.tensor([time]),
                    timestep_video=torch.tensor([time]),
                    gt_action=None,
                )
            return {"video": ["frame"], "action": torch.ones(4, 6)}

    adapter = WSABackendAdapter.__new__(WSABackendAdapter)
    nn.Module.__init__(adapter)
    adapter.wsa_model = Sampler()
    adapter.action_time = ActionTimeConditioner(hidden_dim=12, fourier_dim=8)
    adapter.native_3d_state = Native3DStateConditioner(
        token_dim=8, hidden_dim=12, num_views=3, tokens_per_view=4
    )
    adapter._active_time_update = None
    adapter._active_native_3d_update = None
    candidate = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    mask = torch.ones_like(candidate, dtype=torch.bool)
    mask[..., -1] = False
    args = dict(
        input_image=torch.zeros(1, 3, 32, 32),
        context=torch.zeros(1, 2, 32),
        context_mask=torch.ones(1, 2, dtype=torch.bool),
        proprio=torch.zeros(1, 4),
        current_native_3d=torch.zeros(1, 3, 4, 8),
        current_native_3d_view_mask=torch.ones(1, 3, dtype=torch.bool),
        factual_action=candidate if known else None,
        action_time_s=torch.arange(4, dtype=torch.float32)[None, :, None]
        .mul(0.05).expand_as(candidate),
        action_element_mask=mask,
        num_video_frames=9,
        num_inference_steps=3,
        seed=7,
    )
    if fail:
        with pytest.raises(RuntimeError, match="sampling failure"):
            adapter.infer_world(**args)
    else:
        result = adapter.infer_world(**args)
        assert "action" not in result
        if known:
            assert "action_normalized" not in result
            assert result["conditioning"]["mode"] == "action_dit_clean_candidate_t0"
            assert result["conditioning"]["conditioned_steps"] == 3
        else:
            assert result["conditioning"]["mode"] == "joint_from_observation_no_candidate"
            assert torch.equal(
                result["action_normalized"], torch.ones(1, 4, 6).masked_fill(~mask, 0)
            )
    assert len(seen) == (1 if fail else 3)
    for call in seen:
        if known:
            assert torch.equal(call["latents_action"], candidate.masked_fill(~mask, 0))
            assert torch.equal(call["timestep_action"], torch.zeros(1))
        else:
            assert call["timestep_action"].item() > 0
            assert not torch.equal(call["latents_action"], candidate.masked_fill(~mask, 0))
        assert call["gt_action"] is None
    assert "_predict_joint_noise" not in vars(adapter.wsa_model)
    assert adapter._active_time_update is None
    assert adapter._active_native_3d_update is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("joint", [False, True])
def test_action_padding_follows_official_training_flow_without_changing_valid_inputs(
    joint: bool, dtype: torch.dtype,
) -> None:
    import os
    from pathlib import Path

    from wm3d.models.foundation_world_model import _load_upstream_symbols

    source = Path(os.environ.get("WM3D_WSA_SOURCE_ROOT", "/data/Minko/third_party/WSA"))
    if not source.is_dir():
        pytest.skip("official WSA source is not installed")
    _load_upstream_symbols(source)
    from lerobot.policies.WSA_Large.core.models.wan22.schedulers.scheduler_continuous import (
        WanContinuousFlowMatchScheduler,
    )

    scheduler = WanContinuousFlowMatchScheduler()
    mask = torch.ones(1, 8, 42, dtype=torch.bool)
    mask[..., 7:] = False
    mask[:, 3] = False
    initial_noise = torch.randn(mask.shape).to(dtype=dtype)
    seen = []

    class Sampler(nn.Module):
        train_action_scheduler = scheduler

        def _predict_action_noise(self, **kwargs):
            value = kwargs["latents_action"]
            seen.append(value.clone())
            # This protocol test checks sampler wiring, not learned capability.
            return (value * 0.3).masked_fill(~mask, 1000.0)

        def _predict_joint_noise(self, **kwargs):
            return torch.tensor([3.0]), self._predict_action_noise(**kwargs)

    adapter = WSABackendAdapter.__new__(WSABackendAdapter)
    nn.Module.__init__(adapter)
    adapter.wsa_model = Sampler()
    name = "_predict_joint_noise" if joint else "_predict_action_noise"
    times, deltas = scheduler.build_inference_schedule(
        20, torch.device("cpu"), dtype
    )
    rng_before = torch.get_rng_state()
    latent = initial_noise.clone()
    with adapter._masked_action_sampling(mask, joint=joint):
        for t, delta in zip(times, deltas):
            # Arbitrary feedback in padding must never reach the expert.
            if len(seen):
                latent = latent.masked_fill(~mask, -1234.0)
            before = latent.clone()
            result = getattr(adapter.wsa_model, name)(
                latents_action=latent, timestep_action=t[None]
            )
            velocity = result[1] if joint else result
            expected = scheduler.add_noise(
                torch.zeros_like(latent), initial_noise, t[None]
            )
            assert torch.equal(seen[-1][mask], before[mask])
            assert torch.equal(seen[-1][~mask], expected[~mask])
            assert torch.equal(velocity[mask], (before * 0.3)[mask])
            assert torch.equal(velocity[~mask], initial_noise[~mask])
            if joint:
                assert torch.equal(result[0], torch.tensor([3.0]))
            latent = scheduler.step(velocity, delta, latent)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert name not in vars(adapter.wsa_model)
    # All-valid WSA requests bypass the adapter bit-for-bit.
    original = getattr(adapter.wsa_model, name)
    with adapter._masked_action_sampling(torch.ones_like(mask), joint=joint):
        assert getattr(adapter.wsa_model, name) == original
    assert name not in vars(adapter.wsa_model)


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

    def infer_world(self, **kwargs):
        self.sample = kwargs
        result = {
            "video": [torch.tensor(index) for index in range(kwargs["num_video_frames"])],
            "conditioning": {"mode": "joint" if kwargs["factual_action"] is None else "candidate"},
        }
        if kwargs["factual_action"] is None:
            result["action_normalized"] = torch.full(kwargs["action_time_s"].shape, 0.5)
        return result


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
    assert result["action_normalized"][..., 3:].count_nonzero() == 0
    assert result["action_physical"].shape == (1, 8, 2, 3)
    assert torch.equal(result["action_physical"][..., 0, :], torch.full((1, 8, 3), 2.0))
    assert torch.equal(result["action_physical"][..., 1, :], torch.zeros(1, 8, 3))
    # Joint prediction also works with no future action or target video supplied.
    observed = dict(batch, video=batch["video"][:, :, :1])
    joint = model.predict_joint(observed, num_inference_steps=4, seed=7)
    assert backend.sample["factual_action"] is None
    assert backend.sample["input_image"].shape == (1, 3, 48, 32)
    assert len(joint["video"]) == config.K + 1
    assert torch.equal(joint["action_physical"], result["action_physical"])
    poisoned = dict(batch, action=torch.full((1, 8, 6), float("nan")))
    poisoned["video"] = batch["video"].clone()
    poisoned["video"][:, :, 1:] = float("nan")
    changed = model.predict_joint(poisoned, num_inference_steps=4, seed=7)
    assert torch.equal(changed["action_normalized"], joint["action_normalized"])
    assert backend.sample["factual_action"] is None


def test_foundation_world_serving_separates_policy_from_candidate() -> None:
    config = _config()
    backend = _Backend()
    model = FoundationWorldModel(config, backend=backend)
    action_valid = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    state_valid = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    action = torch.randn(1, 8, 6)
    batch = {
        "wan_context": torch.randn(1, 5, 32),
        "wan_context_mask": torch.ones(1, 5, dtype=torch.bool),
        "action_group_ids": torch.tensor([[1, 0]]),
        "action_group_mask": torch.tensor([[1, 0]], dtype=torch.bool),
        "action_semantic_ids": torch.ones(1, 2, 3, dtype=torch.long),
        "state_semantic_ids": torch.ones(1, 2, 2, dtype=torch.long),
        "embodiment_ids": torch.ones(1, dtype=torch.long),
        "action_normalization_offset": torch.zeros(1, 2, 3),
        "action_normalization_scale": torch.ones(1, 2, 3),
        "state_normalization_offset": torch.zeros(1, 2, 2),
        "state_normalization_scale": torch.ones(1, 2, 2),
        "history_action": torch.randn(1, 4, 6),
        "history_action_element_mask": action_valid[:, None].expand(-1, 4, -1),
        "history_action_time_s": torch.tensor(
            [-0.20, -0.15, -0.10, -0.05]
        )[None, :, None].expand(1, -1, 6),
        "video": torch.randn(1, 3, 9, 48, 32),
        "action": action,
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
    policy_before = model.predict_action(batch, num_inference_steps=4, seed=11)
    result = model.predict_world(
        batch, num_inference_steps=4, seed=11, zero_future_action=True
    )
    assert len(result["video"]) == 9
    assert backend.sample is not None
    assert backend.sample["num_inference_steps"] == 4
    assert backend.sample["seed"] == 11
    assert torch.equal(backend.sample["input_image"], batch["video"][:, :, 0])
    assert torch.equal(backend.sample["factual_action"], torch.zeros_like(action))
    assert torch.equal(result["factual_action_normalized"], torch.zeros_like(action))
    assert "action_normalized" not in result
    observed_only = dict(batch, video=batch["video"][:, :, :1])
    world = model.predict_world(observed_only, num_inference_steps=4, seed=11)
    assert len(world["video"]) == config.K + 1
    changed = dict(batch, action=action + 10, video=batch["video"].clone())
    changed["video"][:, :, 1:] = 0
    policy_after = model.predict_action(changed, num_inference_steps=4, seed=11)
    assert torch.equal(
        policy_before["action_normalized"], policy_after["action_normalized"]
    )
