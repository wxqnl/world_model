from __future__ import annotations

from dataclasses import replace

from pathlib import Path
import torch
from scripts.data.run_cache_worker import _strict_encoder


from wm3d.encoders.native_vggt import NativeVGGTConfig, NativeVGGTEncoder


class _RecordingEncoder(torch.nn.Module):
    def __init__(self, grid: int = 2, dim: int = 2048, appearance_grid: int = 0):
        super().__init__()
        self.grid = grid
        self.dim = dim
        self.appearance_grid = appearance_grid
        self.calls: list[torch.Tensor] = []

    def forward(self, images: torch.Tensor):
        self.calls.append(images.detach().clone())
        batch, views = images.shape[:2]
        patches = self.grid * self.grid
        identity = images.mean(dim=(2, 3, 4))
        tokens = identity[..., None, None].expand(batch, views, patches, self.dim)
        scalar = identity[..., None, None, None].expand(batch, views, 2, 2, 1)
        vector = identity[..., None, None, None].expand(batch, views, 2, 2, 3)
        pose = identity[..., None].expand(batch, views, 9)
        result = {
            "pooled": tokens,
            "depth": scalar,
            "depth_conf": scalar.abs().add(1),
            "world_points": vector,
            "world_points_conf": scalar.abs().add(1),
            "pose_enc": pose,
        }
        if self.appearance_grid:
            appearance_patches = self.appearance_grid * self.appearance_grid
            result["appearance_pooled"] = identity[..., None, None].expand(
                batch, views, appearance_patches, self.dim
            )
        return result


def _config() -> NativeVGGTConfig:
    return NativeVGGTConfig(
        model_revision="fixture",
        token_grid=2,
        input_rgb_size=14,
        input_preprocess="aspect_pad_white",
        target_rgb_size=8,
        token_dim=2048,
        max_views=3,
        dtype="bf16",
    )


def test_time_is_folded_into_batch_and_missing_views_never_enter_geometry() -> None:
    backend = _RecordingEncoder()
    encoder = NativeVGGTEncoder(_config(), device="cpu", encoder=backend)
    images = torch.zeros(1, 3, 3, 3, 14, 14)
    images[:, 0] = 0.1
    images[:, 1] = 0.5
    images[:, 2] = 0.9
    mask = torch.tensor([[[True, True, False], [True, False, False], [True, True, False]]])
    output = encoder(images, mask)

    # Two availability patterns produce two calls.  The repeated two-camera
    # pattern folds its two timestamps into the backend batch, not sequence.
    assert sorted(tuple(call.shape[:2]) for call in backend.calls) == [(1, 1), (2, 2)]
    assert output["view_tokens"].shape == (1, 3, 3, 4, 2048)
    assert output["view_tokens"][:, :, 2].eq(0).all()
    assert output["geometry_confidence"][:, :, 2].eq(0).all()


def test_changing_a_future_frame_cannot_change_an_earlier_cached_token() -> None:
    backend = _RecordingEncoder()
    encoder = NativeVGGTEncoder(_config(), device="cpu", encoder=backend)
    images = torch.rand(1, 3, 2, 3, 14, 14)
    mask = torch.ones(1, 3, 2, dtype=torch.bool)
    first = encoder(images, mask)["view_tokens"]
    changed = images.clone()
    changed[:, 2] = 1.0 - changed[:, 2]
    second = encoder(changed, mask)["view_tokens"]
    torch.testing.assert_close(first[:, :2], second[:, :2], rtol=0, atol=0)
    assert not torch.allclose(first[:, 2], second[:, 2])


def test_dual_grid_preserves_per_view_appearance_tokens() -> None:
    backend = _RecordingEncoder(appearance_grid=4)
    config = replace(
        _config(),
        appearance_token_grid=4,
        input_rgb_size=56,
    )
    encoder = NativeVGGTEncoder(config, device="cpu", encoder=backend)
    images = torch.rand(1, 2, 3, 3, 56, 56)
    mask = torch.tensor([[[True, True, False], [True, False, True]]])

    output = encoder(images, mask)

    assert output["view_tokens"].shape == (1, 2, 3, 4, 2048)
    assert output["appearance_tokens"].shape == (1, 2, 3, 16, 2048)
    assert output["appearance_tokens"][:, 0, 2].eq(0).all()
    assert output["appearance_tokens"][:, 1, 1].eq(0).all()


def test_existing_encoder_contract_defaults_to_geometry_only() -> None:
    config = _strict_encoder(
        Path("configs/encoder/vggt_native_p64.yaml")
    )
    assert config.token_grid == 8
    assert config.appearance_token_grid == 0


def test_dual_path_encoder_contracts_keep_geometry_and_appearance_grids() -> None:
    one_b = _strict_encoder(
        Path("configs/encoder/vggt_native_p64_appearance_p256.yaml")
    )
    five_b = _strict_encoder(
        Path("configs/encoder/vggt_native_p144_appearance_p256.yaml")
    )

    assert (one_b.token_grid, one_b.appearance_token_grid) == (8, 16)
    assert (five_b.token_grid, five_b.appearance_token_grid) == (12, 16)
