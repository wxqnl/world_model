from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from wm3d.encoders.native_vggt import NativeVGGTConfig
from wm3d.models.direct_vggt_teacher import (
    DirectVGGTTeacherAdapter,
    DirectVGGTTeacherConfig,
)


class _FakeNativeVGGT(torch.nn.Module):
    def __init__(
        self,
        *,
        maximum_rows: int | None = None,
        include_appearance: bool = True,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.maximum_rows = maximum_rows
        self.include_appearance = include_appearance
        self.calls: list[int] = []
        self.role_masks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    @torch.inference_mode()
    def forward(
        self,
        images: torch.Tensor,
        view_mask: torch.Tensor,
        *,
        geometry_row_mask: torch.Tensor | None = None,
        appearance_row_mask: torch.Tensor | None = None,
        rgb_row_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _batch, rows, views = images.shape[:3]
        self.calls.append(rows)
        default = torch.ones((1, rows), dtype=torch.bool, device=images.device)
        geometry_rows = default if geometry_row_mask is None else geometry_row_mask
        appearance_rows = (
            default if appearance_row_mask is None else appearance_row_mask
        )
        rgb_rows = default if rgb_row_mask is None else rgb_row_mask
        self.role_masks.append(
            (
                geometry_rows.detach().cpu().clone(),
                appearance_rows.detach().cpu().clone(),
                rgb_rows.detach().cpu().clone(),
            )
        )
        if self.maximum_rows is not None and rows > self.maximum_rows:
            raise torch.OutOfMemoryError("fixture chunk is too large")
        identity = images.mean(dim=(3, 4, 5)) * self.weight
        geometry = identity[..., None, None].expand(1, rows, views, 4, 2048)
        appearance = identity[..., None, None].expand(1, rows, views, 16, 2048)
        confidence = view_mask[..., None].float().expand(1, rows, views, 4)
        depth = identity[..., None].expand(1, rows, views, 4).add(1.0)
        point = identity[..., None, None].expand(1, rows, views, 4, 3)
        camera = identity[..., None].expand(1, rows, views, 9)
        rgb = F.interpolate(
            images.reshape(rows * views, 3, 56, 56),
            size=(8, 8),
            mode="bilinear",
            align_corners=False,
        ).reshape(1, rows, views, 3, 8, 8)
        geometry_weight = geometry_rows[..., None, None].float()
        depth = depth * geometry_weight
        confidence = confidence * geometry_weight
        point = point * geometry_weight[..., None]
        camera = camera * geometry_rows[..., None, None].float()
        appearance = appearance * appearance_rows[..., None, None, None].float()
        rgb = rgb * rgb_rows[..., None, None, None, None].float()
        result = {
            "view_tokens": geometry.to(torch.bfloat16),
            "view_mask": view_mask.bool(),
            "rgb": rgb.mul(255).round().clamp(0, 255).to(torch.uint8),
            "depth": depth.to(torch.float16),
            "point": point.to(torch.float16),
            "geometry_confidence": confidence.to(torch.float16),
            "camera_pose": camera.float(),
        }
        if self.include_appearance:
            result["appearance_tokens"] = appearance.to(torch.bfloat16)
        return result


def _adapter(
    backend: _FakeNativeVGGT, *, chunk_rows: int = 3
) -> DirectVGGTTeacherAdapter:
    encoder = NativeVGGTConfig(
        model_revision="fixture",
        token_grid=2,
        appearance_token_grid=4,
        appearance_feature_layer=4,
        input_rgb_size=56,
        target_rgb_size=8,
        token_dim=2048,
        max_views=3,
        dtype="bf16",
    )
    return DirectVGGTTeacherAdapter(
        DirectVGGTTeacherConfig(
            encoder=encoder,
            context_frames=2,
            future_frames=2,
            appearance_context_frames=1,
            appearance_enabled=True,
            rgb_decode_indices=(0, 1),
            encode_chunk_rows=chunk_rows,
            minimum_chunk_rows=1,
        ),
        device="cpu",
        encoder=backend,
    )


def _appearance_disabled_adapter(
    backend: _FakeNativeVGGT,
) -> DirectVGGTTeacherAdapter:
    encoder = NativeVGGTConfig(
        model_revision="fixture",
        token_grid=2,
        appearance_token_grid=0,
        appearance_feature_layer=-1,
        input_rgb_size=56,
        target_rgb_size=8,
        token_dim=2048,
        max_views=3,
        dtype="bf16",
    )
    return DirectVGGTTeacherAdapter(
        DirectVGGTTeacherConfig(
            encoder=encoder,
            context_frames=2,
            future_frames=2,
            appearance_context_frames=0,
            appearance_enabled=False,
            rgb_decode_indices=(0, 1),
            encode_chunk_rows=3,
            minimum_chunk_rows=1,
        ),
        device="cpu",
        encoder=backend,
    )


def _raw_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(91)
    images = torch.randint(
        0,
        256,
        (2, 4, 3, 3, 56, 56),
        dtype=torch.uint8,
        generator=generator,
    )
    mask = torch.ones(2, 4, 3, dtype=torch.bool)
    mask[0, 2, 2] = False
    mask[1, 3, 1] = False
    return {
        "direct_rgb_uint8": images,
        "direct_view_mask": mask,
        "source_id": torch.tensor([0, 0]),
    }


def test_direct_teacher_materializes_the_unchanged_world_model_abi() -> None:
    backend = _FakeNativeVGGT()
    adapter = _adapter(backend)
    batch = _raw_batch()
    result = adapter.materialize(batch)

    assert "direct_rgb_uint8" not in result
    assert "direct_view_mask" not in result
    assert result["world_tokens"].shape == (2, 2, 3, 4, 2048)
    assert result["appearance_context_tokens"].shape == (2, 1, 3, 16, 2048)
    assert result["target_appearance_tokens"].shape == (2, 2, 3, 16, 2048)
    assert result["target_tokens"].shape == (2, 2, 4, 2048)
    assert result["target_depth"].shape == (2, 2, 3, 4)
    assert result["target_point"].shape == (2, 2, 3, 4, 3)
    assert result["target_camera_pose"].shape == (2, 2, 3, 9)
    assert result["context_rgb"].shape == (2, 3, 3, 8, 8)
    assert result["context_rgb_mask"].shape == (2, 3)
    torch.testing.assert_close(
        result["context_rgb_mask"],
        torch.tensor([[True, False, False], [True, False, False]]),
    )
    assert result["target_rgb"].shape == (2, 2, 3, 3, 8, 8)
    assert result["target_rgb_mask"].shape == (2, 2, 3, 1, 1, 1)
    assert bool(result["target_rgb_mask"][:, :, 0].all())
    assert not bool(result["target_rgb_mask"][:, :, 1:].any())
    expected_anchor_identity = (
        batch["direct_rgb_uint8"][:, 2:, 0]
        .float()
        .div(255.0)
        .mean(dim=(2, 3, 4))
    )
    torch.testing.assert_close(
        result["target_tokens"][:, :, 0, 0].float(),
        expected_anchor_identity.to(torch.bfloat16).float(),
    )
    expected_context = F.interpolate(
        batch["direct_rgb_uint8"][:, 1].reshape(6, 3, 56, 56).float().div(255.0),
        size=(8, 8),
        mode="bilinear",
        align_corners=False,
    ).reshape(2, 3, 3, 8, 8)
    expected_context = expected_context.mul(255).round().div(255)
    torch.testing.assert_close(result["context_rgb"], expected_context)
    assert not bool(result["target_depth_mask"][0, 0, 2].any())
    assert not bool(result["target_camera_pose_mask"][1, 1, 1])
    assert backend.calls == [3, 3, 2]
    geometry_rows = torch.cat([entry[0] for entry in backend.role_masks], dim=1)
    appearance_rows = torch.cat([entry[1] for entry in backend.role_masks], dim=1)
    rgb_rows = torch.cat([entry[2] for entry in backend.role_masks], dim=1)
    torch.testing.assert_close(
        geometry_rows,
        torch.tensor([[False, False, True, True, False, False, True, True]]),
    )
    torch.testing.assert_close(
        appearance_rows,
        torch.tensor([[False, True, True, True, False, True, True, True]]),
    )
    torch.testing.assert_close(rgb_rows, torch.ones_like(rgb_rows))
    assert adapter.metrics["encoded_rows"] == 8
    assert adapter.metrics["encode_calls"] == 3
    assert adapter.metrics["geometry_head_rows"] == 4
    assert adapter.metrics["appearance_pool_rows"] == 6
    assert adapter.metrics["rgb_resize_rows"] == 8
    assert all(not parameter.requires_grad for parameter in adapter.parameters())
    assert not torch.is_inference(result["world_tokens"])


def test_direct_teacher_skips_appearance_when_the_model_disables_it() -> None:
    backend = _FakeNativeVGGT(include_appearance=False)
    adapter = _appearance_disabled_adapter(backend)
    result = adapter.materialize(_raw_batch())

    assert "appearance_context_tokens" not in result
    assert "appearance_context_mask" not in result
    assert "target_appearance_tokens" not in result
    assert "target_appearance_mask" not in result
    assert result["world_tokens"].shape == (2, 2, 3, 4, 2048)
    assert result["target_rgb"].shape == (2, 2, 3, 3, 8, 8)
    appearance_rows = torch.cat(
        [entry[1] for entry in backend.role_masks], dim=1
    )
    assert not bool(appearance_rows.any())
    assert adapter.metrics["appearance_pool_rows"] == 0


def test_direct_teacher_uses_latest_available_context_per_view() -> None:
    batch = _raw_batch()
    batch["direct_view_mask"][0, 1, 2] = False
    batch["direct_view_mask"][1, :2, 1] = False
    result = _adapter(_FakeNativeVGGT()).materialize(batch)

    assert not bool(result["context_rgb_mask"][0, 2])
    assert not bool(result["context_rgb_mask"][1, 1])
    assert not bool(result["target_rgb_mask"][1, :, 1].any())
    expected_fallback = F.interpolate(
        batch["direct_rgb_uint8"][0, 0, 2].float().div(255.0)[None],
        size=(8, 8),
        mode="bilinear",
        align_corners=False,
    )[0]
    expected_fallback = expected_fallback.mul(255).round().div(255)
    torch.testing.assert_close(result["context_rgb"][0, 2], expected_fallback)
    assert result["context_rgb"][1, 1].count_nonzero() == 0


def test_direct_teacher_halves_only_the_encoder_chunk_after_oom() -> None:
    backend = _FakeNativeVGGT(maximum_rows=2)
    adapter = _adapter(backend, chunk_rows=4)
    result = adapter.materialize(_raw_batch())

    assert result["target_tokens"].shape == (2, 2, 4, 2048)
    assert adapter.metrics["oom_backoffs"] == 1
    assert adapter.metrics["effective_chunk_rows"] == 2
    assert backend.calls[0] == 4
    assert backend.calls[1:] == [2, 2, 2, 2]


def test_direct_teacher_deduplicates_exact_frame_rows_without_abi_drift() -> None:
    deduplicated_batch = _raw_batch()
    deduplicated_batch["direct_rgb_uint8"][1, 2:] = deduplicated_batch[
        "direct_rgb_uint8"
    ][0, 2:]
    deduplicated_batch["direct_view_mask"][1, 2:] = deduplicated_batch[
        "direct_view_mask"
    ][0, 2:]
    baseline_batch = {name: value.clone() for name, value in deduplicated_batch.items()}
    baseline_backend = _FakeNativeVGGT()
    baseline = _adapter(baseline_backend).materialize(baseline_batch)

    deduplicated_batch["direct_frame_keys"] = torch.tensor(
        [[10, 11, 12, 13], [14, 15, 12, 13]],
        dtype=torch.int64,
    )
    deduplicated_backend = _FakeNativeVGGT()
    adapter = _adapter(deduplicated_backend)
    result = adapter.materialize(deduplicated_batch)

    assert set(result) == set(baseline)
    for name in result:
        torch.testing.assert_close(result[name], baseline[name])
    assert "direct_frame_keys" not in result
    assert deduplicated_backend.calls == [3, 3]
    assert adapter.metrics["input_rows"] == 8
    assert adapter.metrics["encoded_rows"] == 6
    assert adapter.metrics["deduplicated_rows"] == 2
    assert adapter.metrics["deduplication_ratio"] == 0.25


def test_direct_teacher_rejects_duplicate_keys_with_different_views() -> None:
    batch = _raw_batch()
    batch["direct_frame_keys"] = torch.tensor(
        [[10, 11, 12, 13], [14, 15, 16, 13]],
        dtype=torch.int64,
    )
    with pytest.raises(
        ValueError,
        match="duplicate direct frame keys changed view availability",
    ):
        _adapter(_FakeNativeVGGT()).materialize(batch)
