from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from wm3d.models.latent_motion_renderer import (
    NativeLatentFlowRGBDecoder,
    warp_with_pixel_flow,
)
from wm3d.training.native_objective import (
    compute_native_objective,
    objective_config_from_mapping,
)
from wm3d.training.runtime_contract import validate_runtime_profile


def _renderer_config() -> SimpleNamespace:
    return SimpleNamespace(
        appearance_P=4,
        P=4,
        K=2,
        token_dim=16,
        state_hidden=12,
        task_dim=8,
        num_views=2,
        rgb_latent_channels=4,
        rgb_latent_grid=4,
        rgb_latent_hidden=48,
        rgb_flow_max_pixels=32.0,
        rgb_size=32,
        rgb_decode_indices=(0, 1),
    )


def _renderer_inputs(cfg: SimpleNamespace) -> dict[str, torch.Tensor]:
    torch.manual_seed(73)
    batch = 1
    return {
        "appearance_tokens": torch.randn(
            batch, cfg.K, cfg.num_views, cfg.appearance_P, cfg.token_dim
        ),
        "appearance_context_tokens": torch.randn(
            batch, cfg.num_views, cfg.appearance_P, cfg.token_dim
        ),
        "geometry_tokens": torch.randn(
            batch, cfg.K, cfg.P, cfg.token_dim
        ),
        "factual_action_summary": torch.randn(
            batch, cfg.K, cfg.state_hidden
        ),
        "task_embedding": torch.randn(batch, cfg.task_dim),
        "future_times_s": torch.tensor([[0.2, 0.6]]),
        "context_latent": torch.randn(
            batch,
            cfg.num_views,
            cfg.rgb_latent_channels,
            cfg.rgb_latent_grid,
            cfg.rgb_latent_grid,
        ),
    }


def test_backward_pixel_flow_uses_target_to_source_sign() -> None:
    source = torch.zeros(1, 1, 5, 5)
    source[0, 0, 2, 1] = 1.0
    flow = torch.zeros(1, 2, 5, 5)
    flow[:, 0] = -1.0

    warped, valid = warp_with_pixel_flow(
        source, flow, image_height=5, image_width=5
    )

    assert warped[0, 0, 2, 2].item() == pytest.approx(1.0, abs=1.0e-6)
    assert warped[0, 0, 2, 1].item() == pytest.approx(0.0, abs=1.0e-6)
    assert not bool(valid[0, 0, 2, 0])


def test_out_of_bounds_transport_is_forced_to_synthesis() -> None:
    cfg = _renderer_config()
    decoder = NativeLatentFlowRGBDecoder(cfg).eval()
    with torch.no_grad():
        decoder.flow_head.weight.zero_()
        decoder.flow_head.bias.fill_(10.0)
        decoder.synthesis_head.weight.zero_()
        decoder.synthesis_head.bias.fill_(7.0)

    inputs = _renderer_inputs(cfg)
    output = decoder(**inputs, frame_indices=(0,))

    assert not bool(output["rgb_warp_valid"].any())
    torch.testing.assert_close(
        output["rgb_latent"], output["rgb_synthesis_latent"]
    )
    torch.testing.assert_close(
        output["rgb_latent"], torch.full_like(output["rgb_latent"], 7.0)
    )
    unaligned = inputs["context_latent"][:, None]
    assert not torch.allclose(output["rgb_latent"], unaligned)


def test_teacher_latent_objective_supervises_all_renderer_branches() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = yaml.safe_load(
        (root / "configs/objective/stage0_rgb_teacher_latent_flow.yaml").read_text()
    )
    config = objective_config_from_mapping(profile["objective"])
    torch.manual_seed(79)
    shape = (1, 2, 1, 4, 4, 4)
    flow_shape = (1, 2, 1, 2, 4, 4)
    mask_shape = (1, 2, 1, 1, 4, 4)
    prediction = torch.randn(shape, requires_grad=True)
    flow = torch.randn(flow_shape, requires_grad=True)
    disocclusion_logit = torch.zeros(mask_shape, requires_grad=True)
    warped = torch.randn(shape, requires_grad=True)
    synthesis = torch.randn(shape, requires_grad=True)
    disocclusion_target = torch.zeros(mask_shape)
    disocclusion_target[..., ::2, 1::2] = 1.0
    output = {
        "rgb_renderer_teacher_only": torch.ones(()),
        "appearance_teacher_ratio": torch.ones(()),
        "rgb_latent": prediction,
        "rgb_flow_pixels": flow,
        "rgb_disocclusion_logit": disocclusion_logit,
        "rgb_warped_latent": warped,
        "rgb_synthesis_latent": synthesis,
        "rgb_warp_valid": torch.ones(mask_shape, dtype=torch.bool),
    }
    batch = {
        "target_tokens": torch.zeros(1, 2, 1, 4),
        "target_rgb_latent": torch.randn(shape),
        "context_rgb_latent": torch.randn(1, 1, 4, 4, 4),
        "target_rgb": torch.rand(1, 2, 1, 3, 8, 8),
        "context_rgb": torch.rand(1, 1, 3, 8, 8),
        "target_rgb_mask": torch.ones(1, 2, 1, 1, 1, 1, dtype=torch.bool),
        "context_rgb_mask": torch.ones(1, 1, dtype=torch.bool),
        "rgb_flow_target_pixels": torch.zeros(flow_shape),
        "rgb_disocclusion_target": disocclusion_target,
    }

    losses = compute_native_objective(output=output, batch=batch, config=config)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()

    for value in (prediction, flow, disocclusion_logit, warped, synthesis):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


def test_teacher_runtime_profile_is_sealed_and_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime = yaml.safe_load(
        (root / "configs/runtime/h100_8_fsdp2_rgb_teacher_latent_flow500.yaml").read_text()
    )
    validate_runtime_profile(runtime)
    train = runtime["train"]
    assert train["rgb_teacher_renderer_only"] is True
    assert train["rgb_flow_teacher"]["input_size"] == 256
    assert train["global_batch_size"] == 8
    assert train["checkpoint_steps"] == [1, 20, 100, 250, 500]
