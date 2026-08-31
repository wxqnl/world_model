"""Online observation tokenization for benchmark adapters."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from wm3d_v3.benchmarks.adapter import TokenizedObservation
from wm3d_v3.encoders.vggt_encoder import VGGTEncoder
from wm3d_v3.stage1.action_window_geometry import VGGT_MODEL_REVISION
from wm3d_v3.policy.task_condition import TaskConditionProvider


def _as_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB frame, got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def resize_frames(frames: Iterable[np.ndarray], size: int) -> torch.Tensor:
    """Convert HWC uint8 RGB frames to `[T,3,size,size]` float in `[0,1]`."""
    arr = np.stack([_as_rgb_uint8(frame) for frame in frames], axis=0)
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-2:] != (size, size):
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return x.contiguous()


def _optional_batch_tensor(value: torch.Tensor | np.ndarray | None, *, min_ndim: int = 2) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = value.detach().float().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
    while tensor.ndim < min_ndim:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[0] != 1:
        tensor = tensor.unsqueeze(0)
    return tensor.contiguous()


def _pad_trim(frames: list[np.ndarray], T: int) -> list[np.ndarray]:
    if not frames:
        raise ValueError("at least one frame is required")
    if len(frames) < T:
        return [frames[0]] * (T - len(frames)) + frames
    if len(frames) > T:
        return frames[-T:]
    return frames


def _concat_camera_frames(frames_by_camera: Mapping[str, list[np.ndarray]], camera_names: list[str], T: int) -> list[np.ndarray]:
    streams = [_pad_trim(list(frames_by_camera[name]), T) for name in camera_names]
    out: list[np.ndarray] = []
    for t in range(T):
        out.append(np.concatenate([_as_rgb_uint8(stream[t]) for stream in streams], axis=1))
    return out


class OnlineObservationTokenizer:
    """Turn a rolling RGB observation window into WM3D policy inputs.

    The first version intentionally mirrors the cache builder: one camera stream,
    VGGT patch tokens pooled to the model grid, and one Qwen task embedding per
    episode. It is slow but explicit, and can be optimized after the external
    benchmark loop is proven.
    """

    def __init__(
        self,
        *,
        T: int,
        token_grid: int,
        task_cache_dir: str | Path | None = None,
        device: str = "cuda:0",
        image_size: int = 224,
        context_rgb_size: int = 256,
        qwen_device: str | None = None,
        allow_zero_task_fallback: bool = False,
        camera_fusion: str = "mean",
        anchor_camera_key: str | None = None,
        wrist_camera_key: str | None = None,
    ) -> None:
        self.T = int(T)
        self.image_size = int(image_size)
        self.context_rgb_size = int(context_rgb_size)
        if camera_fusion not in {"mean", "concat", "separate"}:
            raise ValueError(
                f"camera_fusion must be 'mean', 'concat', or 'separate', got {camera_fusion!r}"
            )
        self.camera_fusion = str(camera_fusion)
        self.anchor_camera_key = anchor_camera_key
        self.wrist_camera_key = wrist_camera_key
        self.vggt = VGGTEncoder(
            device=device,
            token_grid=token_grid,
            return_depth=False,
            model_revision=VGGT_MODEL_REVISION,
            local_files_only=True,
        )
        self.task_provider = TaskConditionProvider(
            cache_dir=task_cache_dir,
            device=qwen_device or device,
            allow_zero_fallback=allow_zero_task_fallback,
        )

    @torch.inference_mode()
    def tokenize(
        self,
        frames: list[np.ndarray] | Mapping[str, list[np.ndarray]],
        task_text: str,
        *,
        lowdim_state: torch.Tensor | np.ndarray | None = None,
        object_state: torch.Tensor | np.ndarray | None = None,
        plan_state: torch.Tensor | np.ndarray | None = None,
        action_history: torch.Tensor | np.ndarray | None = None,
        progress_state: torch.Tensor | np.ndarray | None = None,
    ) -> TokenizedObservation:
        if isinstance(frames, Mapping):
            if not frames:
                raise ValueError("frames mapping must contain at least one camera")
            camera_names = sorted(str(key) for key in frames.keys())
            wrist_tokens = None
            view_mask = None
            if self.camera_fusion == "separate":
                anchor_name = self.anchor_camera_key or camera_names[0]
                wrist_name = self.wrist_camera_key or (
                    camera_names[1] if len(camera_names) > 1 else None
                )
                if wrist_name is None or anchor_name == wrist_name:
                    raise ValueError("camera_fusion=separate requires distinct anchor and wrist cameras")
                missing = [name for name in (anchor_name, wrist_name) if name not in frames]
                if missing:
                    raise ValueError(f"missing separate camera streams: {missing}")
                anchor_frames = _pad_trim(list(frames[anchor_name]), self.T)
                wrist_frames = _pad_trim(list(frames[wrist_name]), self.T)
                anchor_out = self.vggt(resize_frames(anchor_frames, self.image_size).unsqueeze(0))
                wrist_out = self.vggt(resize_frames(wrist_frames, self.image_size).unsqueeze(0))
                context_tokens = anchor_out["pooled"].float()
                wrist_tokens = wrist_out["pooled"].float()
                view_mask = torch.ones(
                    context_tokens.shape[0], self.T, 2, dtype=torch.bool
                )
                concat_frames = _concat_camera_frames(
                    frames, [anchor_name, wrist_name], self.T
                )
                context_rgb = resize_frames(
                    [concat_frames[-1]], self.context_rgb_size
                )[0].unsqueeze(0)
                camera_names = [anchor_name, wrist_name]
            elif self.camera_fusion == "concat":
                context_frames = _concat_camera_frames(frames, camera_names, self.T)
                vggt_in = resize_frames(context_frames, self.image_size)
                out = self.vggt(vggt_in.unsqueeze(0))
                context_tokens = out["pooled"].float()
                context_rgb = resize_frames([context_frames[-1]], self.context_rgb_size)[0].unsqueeze(0)
            else:
                pooled = []
                context_frames = None
                for camera_name in camera_names:
                    camera_frames = _pad_trim(list(frames[camera_name]), self.T)
                    vggt_in = resize_frames(camera_frames, self.image_size)
                    out = self.vggt(vggt_in.unsqueeze(0))
                    pooled.append(out["pooled"].float())
                    if context_frames is None:
                        context_frames = camera_frames
                context_tokens = torch.stack(pooled, dim=0).mean(dim=0)
                if context_frames is None:
                    raise ValueError("no context frames after camera fusion")
                context_rgb = resize_frames([context_frames[-1]], self.context_rgb_size)[0].unsqueeze(0)
            metadata = {
                "task_text": task_text,
                "num_frames": self.T,
                "cameras": camera_names,
                "camera_fusion": self.camera_fusion,
            }
        else:
            frames = _pad_trim(list(frames), self.T)
            vggt_in = resize_frames(frames, self.image_size)
            out = self.vggt(vggt_in.unsqueeze(0))
            context_tokens = out["pooled"].float()
            wrist_tokens = None
            view_mask = None
            context_rgb = resize_frames([frames[-1]], self.context_rgb_size)[0].unsqueeze(0)
            metadata = {"task_text": task_text, "num_frames": len(frames), "cameras": ["default"]}
        task_emb = self.task_provider.embed(task_text).reshape(1, -1)
        return TokenizedObservation(
            context_tokens=context_tokens.cpu(),
            task_emb=task_emb.float().cpu(),
            wrist_tokens=wrist_tokens.cpu() if wrist_tokens is not None else None,
            view_mask=view_mask.cpu() if view_mask is not None else None,
            context_rgb=context_rgb.float().cpu(),
            lowdim_state=_optional_batch_tensor(lowdim_state),
            object_state=_optional_batch_tensor(object_state),
            plan_state=_optional_batch_tensor(plan_state),
            action_history=_optional_batch_tensor(action_history, min_ndim=3),
            progress_state=_optional_batch_tensor(progress_state),
            metadata=metadata,
        )


class FrameWindow:
    """Fixed-length frame buffer used by online adapters."""

    def __init__(self, T: int) -> None:
        self.frames: deque[np.ndarray] = deque(maxlen=int(T))

    def reset(self, first_frame: np.ndarray) -> None:
        self.frames.clear()
        for _ in range(self.frames.maxlen or 1):
            self.frames.append(_as_rgb_uint8(first_frame))

    def append(self, frame: np.ndarray) -> None:
        self.frames.append(_as_rgb_uint8(frame))

    def list(self) -> list[np.ndarray]:
        return list(self.frames)
