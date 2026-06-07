"""Geometry decoder: pooled VGGT tokens -> depth, optionally point + pose."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResizeConvUp(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(min(8, c_out), c_out), nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(min(8, c_out), c_out), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.block(x)


class GeomDecoder(nn.Module):
    """[B, T, 64, 2048] -> depth [B, T, 224, 224] and optional extra geometry."""

    def __init__(
        self,
        token_dim=2048,
        token_grid=8,
        hidden=384,
        out_hw=256,
        crop_hw=224,
        enable_extra=True,
        upsample_mode="transpose",
    ):
        super().__init__()
        if upsample_mode not in {"transpose", "resize_conv"}:
            raise ValueError(f"unsupported geom upsample_mode: {upsample_mode}")
        self.token_grid = token_grid
        self.out_hw = out_hw
        self.crop_hw = crop_hw
        self.enable_extra = enable_extra
        self.upsample_mode = upsample_mode
        H = hidden
        # n_ups so token_grid * 2**n_ups == out_hw. 8→256: 5; 16→256: 4.
        n_ups = 0
        g = token_grid
        while g < out_hw:
            g *= 2; n_ups += 1
        assert g == out_hw, f"out_hw {out_hw} not power-of-2 multiple of token_grid {token_grid}"
        self.stem = nn.Sequential(nn.Conv2d(token_dim, H, 1), nn.GroupNorm(8, H), nn.GELU())
        def up(c_in, c_out):
            return nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out, 4, 2, 1), nn.GroupNorm(min(8, c_out), c_out), nn.GELU(),
                nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(min(8, c_out), c_out), nn.GELU(),
            )
        def resize_conv_up(c_in, c_out):
            return _ResizeConvUp(c_in, c_out)
        # Channel schedule (legacy n_ups=5 unchanged; n_ups=4 drops the last H//8→H//8 stage).
        if n_ups == 5:
            chans = [H, H, H // 2, H // 4, H // 8, H // 8]
        elif n_ups == 4:
            chans = [H, H, H // 2, H // 4, H // 8]
        else:
            chans = [H] + [max(H >> i, H // 8) for i in range(n_ups)]
        up_builder = resize_conv_up if upsample_mode == "resize_conv" else up
        ups = [up_builder(chans[i], chans[i + 1]) for i in range(n_ups)]
        self.ups = nn.ModuleList(ups)
        # Remap legacy v3 ckpt keys (up1..up5) -> ups.0..ups.4 on load.
        self._register_load_state_dict_pre_hook(self._remap_legacy_keys)
        c_last = chans[-1]
        self.depth_head = nn.Conv2d(c_last, 1, 1)
        if enable_extra:
            self.point_head = nn.Conv2d(c_last, 3, 1)
            self.pose_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(H, H), nn.GELU(),
                nn.Linear(H, 9),
            )
        else:
            self.point_head = None
            self.pose_head = None

    @staticmethod
    def _remap_legacy_keys(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for i in range(1, 6):
            old = f"{prefix}up{i}."
            new = f"{prefix}ups.{i - 1}."
            for k in list(state_dict.keys()):
                if k.startswith(old):
                    state_dict[new + k[len(old):]] = state_dict.pop(k)

    def forward(self, tok: torch.Tensor) -> dict[str, torch.Tensor]:
        B, T, P, D = tok.shape
        G = self.token_grid
        x = tok.reshape(B * T, G * G, D).transpose(1, 2).reshape(B * T, D, G, G)
        x = self.stem(x)
        pose = self.pose_head(x).view(B, T, 9) if self.pose_head is not None else None
        for u in self.ups:
            x = u(x)
        # crop out_hw → crop_hw centered
        off = (self.out_hw - self.crop_hw) // 2
        x = x[:, :, off:off + self.crop_hw, off:off + self.crop_hw]
        depth = F.softplus(self.depth_head(x)).squeeze(1).view(B, T, self.crop_hw, self.crop_hw)
        out = {"depth": depth}
        if self.point_head is not None and pose is not None:
            out["point"] = self.point_head(x).permute(0, 2, 3, 1).contiguous().view(
                B, T, self.crop_hw, self.crop_hw, 3
            )
            out["pose"] = pose
        return out
