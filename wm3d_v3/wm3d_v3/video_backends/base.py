"""Common interface for Hunyuan/Wan/tau0-style video renderers.

The world model owns structured future prediction. Video backends consume those
controls and render high-fidelity RGB. Keeping this boundary explicit prevents
the renderer from becoming the project state representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import torch


@dataclass
class VideoConditionBundle:
    """Structured controls for a video renderer.

    Tensors follow the wm3d_v3 convention:

    - RGB videos: `[B, T, 3, H, W]` in `[0, 1]`.
    - context RGB: `[B, 3, H, W]` in `[0, 1]`.
    - depth: `[B, T, H, W]`.
    - masks/hints: `[B, T, 1, H, W]` in `[0, 1]`.
    - tokens: `[B, T, P, D]`.
    """

    context_rgb: torch.Tensor
    action_cond: torch.Tensor | None = None
    task_emb: torch.Tensor | None = None
    task_text: list[str] | None = None
    pred_tokens: torch.Tensor | None = None
    depth: torch.Tensor | None = None
    motion_hint: torch.Tensor | None = None
    contact_hint: torch.Tensor | None = None
    rough_rgb: torch.Tensor | None = None
    extra: Mapping[str, torch.Tensor] | None = None

    def to(self, device: torch.device | str) -> "VideoConditionBundle":
        """Move tensor fields to a device, preserving metadata fields."""

        def move(x):
            return x.to(device) if isinstance(x, torch.Tensor) else x

        extra = None
        if self.extra is not None:
            extra = {k: move(v) for k, v in self.extra.items()}
        return VideoConditionBundle(
            context_rgb=move(self.context_rgb),
            action_cond=move(self.action_cond),
            task_emb=move(self.task_emb),
            task_text=self.task_text,
            pred_tokens=move(self.pred_tokens),
            depth=move(self.depth),
            motion_hint=move(self.motion_hint),
            contact_hint=move(self.contact_hint),
            rough_rgb=move(self.rough_rgb),
            extra=extra,
        )


@dataclass
class VideoBackendOutput:
    """Renderer output container."""

    rgb: torch.Tensor
    latents: torch.Tensor | None = None
    metadata: dict | None = None


class VideoBackend(Protocol):
    """Minimal protocol implemented by concrete video renderers."""

    name: str

    def generate(
        self,
        bundle: VideoConditionBundle,
        *,
        num_frames: int,
        height: int,
        width: int,
        seed: int | None = None,
        **kwargs,
    ) -> VideoBackendOutput:
        """Render a future RGB clip from structured world-model controls."""

