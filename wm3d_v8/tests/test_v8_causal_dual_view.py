from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch


class FutureMixingEncoder(torch.nn.Module):
    """Small deterministic encoder whose output depends on every input frame."""

    def __init__(self) -> None:
        super().__init__()
        self.return_depth = True
        self.return_depth_conf = True
        self.return_geom_extra = True
        self.call_lengths: list[int] = []

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        assert images.ndim == 5 and images.shape[0] == 1
        _, frames, _, height, width = images.shape
        self.call_lengths.append(int(frames))
        frame_signal = images.float().mean(dim=(2, 3, 4))
        window_signal = frame_signal.mean(dim=1, keepdim=True)
        mixed = frame_signal + window_signal
        pooled = mixed[:, :, None, None].repeat(1, 1, 4, 6)
        depth = mixed[:, :, None, None].repeat(1, 1, height, width)
        depth_conf = torch.ones_like(depth) * 0.75
        points = torch.stack((depth, depth * 2.0, depth * 3.0), dim=-1)
        point_conf = torch.ones_like(depth) * 0.5
        pose = mixed[:, :, None].repeat(1, 1, 9)
        return {
            "pooled": pooled,
            "depth": depth,
            "depth_conf": depth_conf,
            "world_points": points,
            "world_points_conf": point_conf,
            "pose_enc": pose,
        }


class IdentityCodec:
    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.float()

    @staticmethod
    def quantize(latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = torch.full(
            (*latent.shape[:-2], 1, 1),
            1.0 / 127.0,
            dtype=torch.float32,
            device=latent.device,
        )
        codes = torch.round(latent / scale).clamp(-127, 127).to(torch.int8)
        return codes, scale.to(torch.float16)


def _module():
    return importlib.import_module("wm3d_v3.data.v8_causal_dual_view")


def _frames(future_value: float) -> torch.Tensor:
    observed = torch.zeros((3, 3, 8, 8), dtype=torch.float32)
    future = torch.full((2, 3, 8, 8), future_value, dtype=torch.float32)
    return torch.cat((observed, future), dim=0)


def test_context_encoding_is_invariant_to_future_pixels() -> None:
    """Catches a single full-window forward being reused for context inputs."""

    module = _module()
    encoder_a = FutureMixingEncoder()
    encoder_b = FutureMixingEncoder()

    output_a = module.encode_causal_dual_view(
        _frames(0.0), encoder=encoder_a, codec=IdentityCodec(), T=3, k=2
    )
    output_b = module.encode_causal_dual_view(
        _frames(1.0), encoder=encoder_b, codec=IdentityCodec(), T=3, k=2
    )

    np.testing.assert_array_equal(
        output_a["context_codes"], output_b["context_codes"]
    )
    np.testing.assert_array_equal(
        output_a["context_scale"], output_b["context_scale"]
    )
    assert not np.array_equal(output_a["future_codes"], output_b["future_codes"])
    assert encoder_a.call_lengths == [3, 5]
    assert encoder_b.call_lengths == [3, 5]


def test_target_forward_observed_outputs_are_discarded() -> None:
    """Catches retaining T+K target tokens instead of the final K targets."""

    module = _module()
    output = module.encode_causal_dual_view(
        _frames(0.5), encoder=FutureMixingEncoder(), codec=IdentityCodec(), T=3, k=2
    )

    assert output["context_codes"].shape == (3, 4, 6)
    assert output["future_codes"].shape == (2, 4, 6)
    assert output["future_depth_patch"].shape == (2, 8, 8)
    assert output["future_point_patch"].shape == (2, 8, 8, 3)
    assert output["future_pose_enc"].shape == (2, 9)


def test_archive_validator_rejects_legacy_schema(tmp_path) -> None:
    """Catches a legacy full-window cache being accepted by causal mode."""

    module = _module()
    path = tmp_path / "legacy.npz"
    np.savez(
        path,
        schema=np.asarray("wm3d_v7_compact_geom_v3"),
        pooled=np.zeros((5, 4, 6), dtype=np.float16),
    )

    with np.load(path, allow_pickle=False) as archive:
        with pytest.raises(ValueError, match="causal dual-view schema"):
            module.validate_causal_dual_view_archive(
                archive, T=3, k=2, paired_views=False
            )


def test_archive_validator_rejects_future_leakage_flag(tmp_path) -> None:
    """Catches metadata that admits future-conditioned context."""

    module = _module()
    payload = module.causal_dual_view_metadata(T=3, k=2)
    payload["context_future_leakage"] = np.asarray(True)
    payload.update(
        {
            "context_codes": np.zeros((3, 4, 6), dtype=np.int8),
            "context_scale": np.ones((3, 1, 1), dtype=np.float16),
            "future_codes": np.zeros((2, 4, 6), dtype=np.int8),
            "future_scale": np.ones((2, 1, 1), dtype=np.float16),
            "future_depth_patch": np.zeros((2, 8, 8), dtype=np.float16),
            "future_depth_conf_patch": np.ones((2, 8, 8), dtype=np.float16),
            "future_point_patch": np.zeros((2, 8, 8, 3), dtype=np.float16),
            "future_point_conf_patch": np.ones((2, 8, 8), dtype=np.float16),
            "future_pose_enc": np.zeros((2, 9), dtype=np.float16),
        }
    )
    path = tmp_path / "leaking.npz"
    np.savez(path, **payload)

    with np.load(path, allow_pickle=False) as archive:
        with pytest.raises(ValueError, match="context_future_leakage"):
            module.validate_causal_dual_view_archive(
                archive, T=3, k=2, paired_views=False
            )


def test_archive_validator_accepts_exact_contract(tmp_path) -> None:
    """Catches validators that reject a well-formed causal target-only payload."""

    module = _module()
    payload = module.causal_dual_view_metadata(T=3, k=2)
    payload.update(
        {
            "context_codes": np.zeros((3, 4, 6), dtype=np.int8),
            "context_scale": np.ones((3, 1, 1), dtype=np.float16),
            "future_codes": np.zeros((2, 4, 6), dtype=np.int8),
            "future_scale": np.ones((2, 1, 1), dtype=np.float16),
            "future_depth_patch": np.zeros((2, 8, 8), dtype=np.float16),
            "future_depth_conf_patch": np.ones((2, 8, 8), dtype=np.float16),
            "future_point_patch": np.zeros((2, 8, 8, 3), dtype=np.float16),
            "future_point_conf_patch": np.ones((2, 8, 8), dtype=np.float16),
            "future_pose_enc": np.zeros((2, 9), dtype=np.float16),
        }
    )
    path = tmp_path / "valid.npz"
    np.savez(path, **payload)

    with np.load(path, allow_pickle=False) as archive:
        summary = module.validate_causal_dual_view_archive(
            archive, T=3, k=2, paired_views=False
        )

    assert summary == {
        "compact": False,
        "windows": 1,
        "context_frames": 3,
        "future_frames": 2,
        "token_count": 4,
        "latent_dim": 6,
    }
