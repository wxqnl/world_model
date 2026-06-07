"""Control adapter that injects wm3d controls into Hunyuan DiT image tokens.

The adapter is intentionally isolated from the core world-model stages. It is
zero initialized, so installing it on a frozen Hunyuan transformer is an exact
no-op until a control checkpoint is trained and loaded.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


CONTROL_CHECKPOINT_KIND = "hunyuan_dit_control_adapter_v1"


def _norm_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _Conv3dBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_norm_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


@dataclass
class HunyuanDiTControlConfig:
    token_dim: int = 2048
    token_grid: int = 8
    latent_channels: int = 16
    hidden: int = 192
    dit_hidden: int = 3072
    action_dim: int = 7
    task_dim: int = 2048
    double_blocks: int = 20
    single_blocks: int = 40
    use_depth: bool = True
    use_motion: bool = True
    use_contact: bool = True
    use_rough: bool = True
    use_context: bool = True
    use_action: bool = True
    use_task: bool = True
    use_point: bool = True
    use_pose: bool = True
    point_dim: int = 3
    pose_dim: int = 9
    cfg_branch: str = "conditional_only"


@dataclass
class HunyuanDiTControlState:
    features: torch.Tensor
    scale: float = 1.0
    img_token_len: int | None = None

    @property
    def batch_size(self) -> int:
        return int(self.features.shape[0])


class HunyuanDiTControlAdapter(nn.Module):
    """Encode wm3d controls into residuals for Hunyuan DiT image tokens."""

    def __init__(self, cfg: HunyuanDiTControlConfig | None = None):
        super().__init__()
        self.cfg = cfg or HunyuanDiTControlConfig()
        h = self.cfg.hidden

        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.token_dim),
            nn.Linear(self.cfg.token_dim, h),
            nn.SiLU(inplace=True),
            nn.Linear(h, h),
        )
        self.depth_proj = nn.Sequential(nn.Conv3d(1, h, 3, padding=1), nn.GroupNorm(_norm_groups(h), h), nn.SiLU(inplace=True))
        self.motion_proj = nn.Sequential(nn.Conv3d(1, h, 3, padding=1), nn.GroupNorm(_norm_groups(h), h), nn.SiLU(inplace=True))
        self.contact_proj = nn.Sequential(nn.Conv3d(1, h, 3, padding=1), nn.GroupNorm(_norm_groups(h), h), nn.SiLU(inplace=True))
        self.rough_proj = nn.Sequential(nn.Conv3d(3, h, 3, padding=1), nn.GroupNorm(_norm_groups(h), h), nn.SiLU(inplace=True))
        self.context_proj = nn.Sequential(nn.Conv2d(3, h, 3, padding=1), nn.GroupNorm(_norm_groups(h), h), nn.SiLU(inplace=True))
        self.action_proj = nn.Sequential(nn.Linear(self.cfg.action_dim, h), nn.SiLU(inplace=True), nn.Linear(h, h))
        self.task_proj = nn.Sequential(nn.LayerNorm(self.cfg.task_dim), nn.Linear(self.cfg.task_dim, h), nn.SiLU(inplace=True), nn.Linear(h, h))
        self.point_proj = nn.Sequential(nn.Conv3d(self.cfg.point_dim, h, 3, padding=1), nn.GroupNorm(_norm_groups(h), h), nn.SiLU(inplace=True))
        self.pose_proj = nn.Sequential(nn.Linear(self.cfg.pose_dim, h), nn.SiLU(inplace=True), nn.Linear(h, h))
        self.fuse = nn.Sequential(_Conv3dBlock(h), _Conv3dBlock(h))

        self.double_projections = nn.ModuleList([nn.Linear(h, self.cfg.dit_hidden) for _ in range(self.cfg.double_blocks)])
        self.single_projections = nn.ModuleList([nn.Linear(h, self.cfg.dit_hidden) for _ in range(self.cfg.single_blocks)])
        self.double_gates = nn.Parameter(torch.zeros(self.cfg.double_blocks))
        self.single_gates = nn.Parameter(torch.zeros(self.cfg.single_blocks))
        self.latent_projection = nn.Conv3d(h, self.cfg.latent_channels, kernel_size=1)
        self._control_state: HunyuanDiTControlState | None = None
        self.zero_init_output()

    @property
    def control_state(self) -> HunyuanDiTControlState | None:
        return self._control_state

    def zero_init_output(self) -> None:
        """Make all residual branches an exact no-op at initialization."""
        for layer in [*self.double_projections, *self.single_projections]:
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.ones_(self.double_gates)
        nn.init.ones_(self.single_gates)
        nn.init.zeros_(self.latent_projection.weight)
        if self.latent_projection.bias is not None:
            nn.init.zeros_(self.latent_projection.bias)

    @staticmethod
    def _grid_size(patches: int) -> int:
        grid = int(math.isqrt(patches))
        if grid * grid != patches:
            raise ValueError(f"P must be a square token grid, got P={patches}")
        return grid

    def _param_dtype_device(self) -> tuple[torch.dtype, torch.device]:
        p = next(self.parameters())
        return p.dtype, p.device

    @staticmethod
    def _resize_video(x: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        return F.interpolate(x, size=(t, h, w), mode="trilinear", align_corners=False)

    @staticmethod
    def _hint_to_bcthw(x: torch.Tensor, *, batch: int, horizon: int, name: str) -> torch.Tensor:
        if x.ndim == 4:
            x = x[:, :, None]
        if x.ndim != 5:
            raise ValueError(f"{name} must be [B,T,H,W] or [B,T,C,H,W], got {tuple(x.shape)}")
        if x.shape[0] != batch or x.shape[1] != horizon:
            raise ValueError(f"{name} leading dims must be {(batch, horizon)}, got {tuple(x.shape[:2])}")
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        return x

    @staticmethod
    def _rgb_to_bcthw(x: torch.Tensor, *, batch: int, horizon: int, name: str) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"{name} must be [B,T,3,H,W], got {tuple(x.shape)}")
        if x.shape[0] != batch or x.shape[1] != horizon:
            raise ValueError(f"{name} leading dims must be {(batch, horizon)}, got {tuple(x.shape[:2])}")
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        if x.shape[1] > 3:
            x = x[:, :3]
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1, -1)
        if x.shape[1] != 3:
            raise ValueError(f"{name} must have 1 or 3 channels, got {x.shape[1]}")
        return x

    def build_control_state(
        self,
        pred_tokens: torch.Tensor,
        depth: torch.Tensor | None = None,
        *,
        motion_hint: torch.Tensor | None = None,
        contact_hint: torch.Tensor | None = None,
        rough_rgb: torch.Tensor | None = None,
        context_rgb: torch.Tensor | None = None,
        action_cond: torch.Tensor | None = None,
        task_emb: torch.Tensor | None = None,
        point: torch.Tensor | None = None,
        pose_geom: torch.Tensor | None = None,
        scale: float = 1.0,
        img_token_len: int | None = None,
    ) -> HunyuanDiTControlState:
        if pred_tokens.ndim != 4:
            raise ValueError(f"pred_tokens must be [B,T,P,D], got {tuple(pred_tokens.shape)}")
        batch, horizon, patches, dim = pred_tokens.shape
        if dim != self.cfg.token_dim:
            raise ValueError(f"expected token dim {self.cfg.token_dim}, got {dim}")
        grid = self._grid_size(patches)
        dtype, device = self._param_dtype_device()
        pred_tokens = pred_tokens.to(device=device, dtype=dtype)

        tok = self.token_proj(pred_tokens)
        feat = tok.reshape(batch, horizon, grid, grid, self.cfg.hidden).permute(0, 4, 1, 2, 3).contiguous()

        if self.cfg.use_depth:
            if depth is None:
                depth_v = feat.new_zeros(batch, 1, horizon, grid, grid)
            else:
                depth_v = self._hint_to_bcthw(depth.to(device=device, dtype=dtype), batch=batch, horizon=horizon, name="depth")
                depth_v = self._resize_video(depth_v, horizon, grid, grid)
            feat = feat + self.depth_proj(depth_v)

        if self.cfg.use_motion and motion_hint is not None:
            motion_v = self._hint_to_bcthw(motion_hint.to(device=device, dtype=dtype), batch=batch, horizon=horizon, name="motion_hint")
            feat = feat + self.motion_proj(self._resize_video(motion_v, horizon, grid, grid))

        if self.cfg.use_contact and contact_hint is not None:
            contact_v = self._hint_to_bcthw(contact_hint.to(device=device, dtype=dtype), batch=batch, horizon=horizon, name="contact_hint")
            feat = feat + self.contact_proj(self._resize_video(contact_v, horizon, grid, grid))

        if self.cfg.use_rough and rough_rgb is not None:
            rough_v = self._rgb_to_bcthw(rough_rgb.to(device=device, dtype=dtype), batch=batch, horizon=horizon, name="rough_rgb")
            feat = feat + self.rough_proj(self._resize_video(rough_v, horizon, grid, grid))

        if self.cfg.use_context and context_rgb is not None:
            if context_rgb.ndim != 4 or context_rgb.shape[0] != batch:
                raise ValueError(f"context_rgb must be [B,3,H,W], got {tuple(context_rgb.shape)}")
            ctx = context_rgb.to(device=device, dtype=dtype)
            if ctx.shape[1] == 1:
                ctx = ctx.expand(-1, 3, -1, -1)
            if ctx.shape[1] != 3:
                raise ValueError(f"context_rgb must have 1 or 3 channels, got {ctx.shape[1]}")
            ctx = F.interpolate(ctx, size=(grid, grid), mode="bilinear", align_corners=False)
            ctx = self.context_proj(ctx)[:, :, None]
            feat = feat + ctx.expand(-1, -1, horizon, -1, -1)

        if self.cfg.use_action:
            if action_cond is None:
                action_cond = pred_tokens.new_zeros(batch, horizon, self.cfg.action_dim)
            action_cond = action_cond.to(device=device, dtype=dtype)
            if action_cond.ndim == 2:
                action_cond = action_cond[:, None].expand(-1, horizon, -1)
            if action_cond.shape != (batch, horizon, self.cfg.action_dim):
                raise ValueError(f"action_cond must be [B,T,{self.cfg.action_dim}], got {tuple(action_cond.shape)}")
            action = self.action_proj(action_cond).permute(0, 2, 1)[:, :, :, None, None]
            feat = feat + action

        if self.cfg.use_task:
            if task_emb is None:
                task_emb = pred_tokens.new_zeros(batch, self.cfg.task_dim)
            task_emb = task_emb.to(device=device, dtype=dtype)
            if task_emb.shape != (batch, self.cfg.task_dim):
                raise ValueError(f"task_emb must be [B,{self.cfg.task_dim}], got {tuple(task_emb.shape)}")
            task = self.task_proj(task_emb)[:, :, None, None, None]
            feat = feat + task

        if self.cfg.use_point and point is not None:
            pts = point.to(device=device, dtype=dtype)
            if pts.ndim == 5 and pts.shape[-1] == self.cfg.point_dim:
                pts = pts.permute(0, 4, 1, 2, 3).contiguous()  # [B,T,H,W,3] -> [B,3,T,H,W]
            elif not (pts.ndim == 5 and pts.shape[1] == self.cfg.point_dim):
                raise ValueError(f"point must be [B,T,H,W,{self.cfg.point_dim}] or [B,{self.cfg.point_dim},T,H,W], got {tuple(point.shape)}")
            feat = feat + self.point_proj(self._resize_video(pts, horizon, grid, grid))

        if self.cfg.use_pose and pose_geom is not None:
            pose_g = pose_geom.to(device=device, dtype=dtype)
            if pose_g.ndim == 2:
                pose_g = pose_g[:, None].expand(-1, horizon, -1)
            if pose_g.shape != (batch, horizon, self.cfg.pose_dim):
                raise ValueError(f"pose_geom must be [B,T,{self.cfg.pose_dim}], got {tuple(pose_geom.shape)}")
            pose_feat = self.pose_proj(pose_g).permute(0, 2, 1)[:, :, :, None, None]
            feat = feat + pose_feat

        state = HunyuanDiTControlState(features=self.fuse(feat), scale=float(scale), img_token_len=img_token_len)
        return state

    def prepare_controls(self, *args: Any, **kwargs: Any) -> HunyuanDiTControlState:
        state = self.build_control_state(*args, **kwargs)
        self._control_state = state
        return state

    def set_control_state(self, state: HunyuanDiTControlState) -> None:
        self._control_state = state

    def clear_control_state(self) -> None:
        self._control_state = None

    @staticmethod
    def _latent_thw(latent_shape: Any, state: HunyuanDiTControlState) -> tuple[int, int, int]:
        if latent_shape is None:
            return int(state.features.shape[2]), int(state.features.shape[3]), int(state.features.shape[4])
        if isinstance(latent_shape, torch.Tensor):
            latent_shape = tuple(latent_shape.shape)
        latent_shape = tuple(latent_shape)
        if len(latent_shape) >= 5:
            return int(latent_shape[-3]), int(latent_shape[-2]), int(latent_shape[-1])
        if len(latent_shape) == 4:
            return int(latent_shape[-3]), int(latent_shape[-2]), int(latent_shape[-1])
        if len(latent_shape) == 3:
            return int(latent_shape[0]), int(latent_shape[1]), int(latent_shape[2])
        raise ValueError(f"latent_shape must have 3, 4, or 5 dims, got {latent_shape}")

    def _sequence_features(self, token_len: int, latent_shape: Any) -> torch.Tensor:
        state = self._control_state
        if state is None:
            raise RuntimeError("control state is not prepared; call prepare_controls/build_control_state first")
        t, h, w = self._latent_thw(latent_shape, state)
        feat = self._resize_video(state.features, t, h, w)
        seq = feat.flatten(2).transpose(1, 2).contiguous()
        if seq.shape[1] != token_len:
            seq = F.interpolate(seq.transpose(1, 2), size=token_len, mode="linear", align_corners=False).transpose(1, 2)
        return seq

    @staticmethod
    def _require_channels(x: torch.Tensor, channels: int, *, name: str) -> torch.Tensor:
        if x.shape[-1] != channels:
            raise RuntimeError(
                f"{name} residual hidden dim {x.shape[-1]} does not match Hunyuan DiT hidden dim {channels}; "
                "train/load the control adapter with cfg.dit_hidden set to the target transformer hidden_size"
            )
        return x

    def _expand_for_cfg(self, residual: torch.Tensor, batch_size: int) -> torch.Tensor:
        state = self._control_state
        if state is None:
            raise RuntimeError("control state is not prepared")
        control_batch = state.batch_size
        if batch_size == control_batch:
            return residual
        if batch_size == 2 * control_batch and self.cfg.cfg_branch == "conditional_only":
            # HunyuanVideoPipeline concatenates classifier-free batches as
            # [negative/unconditional, prompt/conditional]. Keep wm3d control
            # on the conditional half so text guidance remains meaningful.
            zeros = residual.new_zeros(control_batch, *residual.shape[1:])
            return torch.cat([zeros, residual], dim=0)
        if control_batch == 1:
            return residual.expand(batch_size, *residual.shape[1:])
        raise ValueError(
            f"cannot map control batch {control_batch} to transformer batch {batch_size}; "
            "supported cases are equal batch or CFG batch 2x controls"
        )

    def double_residual(
        self,
        layer_idx: int,
        img_tokens: torch.Tensor,
        latent_shape: Any,
        batch_size: int,
    ) -> torch.Tensor:
        if self._control_state is None:
            return torch.zeros_like(img_tokens)
        if layer_idx < 0 or layer_idx >= len(self.double_projections):
            return torch.zeros_like(img_tokens)
        seq = self._sequence_features(int(img_tokens.shape[1]), latent_shape)
        residual = self.double_projections[layer_idx](seq) * self.double_gates[layer_idx].to(dtype=seq.dtype)
        residual = self._require_channels(residual, int(img_tokens.shape[-1]), name="double_block") * float(self._control_state.scale)
        residual = self._expand_for_cfg(residual, int(batch_size))
        return residual.to(device=img_tokens.device, dtype=img_tokens.dtype)

    def single_residual(
        self,
        layer_idx: int,
        x_tokens: torch.Tensor,
        img_token_len: int,
        latent_shape: Any,
        batch_size: int,
    ) -> torch.Tensor:
        if self._control_state is None:
            return torch.zeros_like(x_tokens)
        if layer_idx < 0 or layer_idx >= len(self.single_projections):
            return torch.zeros_like(x_tokens)
        img_token_len = max(0, min(int(img_token_len), int(x_tokens.shape[1])))
        if img_token_len == 0:
            return torch.zeros_like(x_tokens)
        seq = self._sequence_features(img_token_len, latent_shape)
        img_residual = self.single_projections[layer_idx](seq) * self.single_gates[layer_idx].to(dtype=seq.dtype)
        img_residual = self._require_channels(img_residual, int(x_tokens.shape[-1]), name="single_block") * float(self._control_state.scale)
        img_residual = self._expand_for_cfg(img_residual, int(batch_size))
        out = torch.zeros_like(x_tokens)
        out[:, :img_token_len] = img_residual.to(device=x_tokens.device, dtype=x_tokens.dtype)
        return out

    def latent_residual(self, target_latents: torch.Tensor | tuple[int, ...]) -> torch.Tensor:
        """Training helper that maps the same control state to latent velocity shape."""
        if self._control_state is None:
            raise RuntimeError("control state is not prepared")
        shape = tuple(target_latents.shape) if isinstance(target_latents, torch.Tensor) else tuple(target_latents)
        if len(shape) != 5:
            raise ValueError(f"target_latents must be [B,C,T,H,W], got {shape}")
        t, h, w = int(shape[2]), int(shape[3]), int(shape[4])
        feat = self._resize_video(self._control_state.features, t, h, w)
        residual = self.latent_projection(feat) * float(self._control_state.scale)
        residual = self._expand_for_cfg(residual, int(shape[0]))
        if isinstance(target_latents, torch.Tensor):
            residual = residual.to(device=target_latents.device, dtype=target_latents.dtype)
        return residual


class HunyuanDiTControlInjector:
    """Install temporary forward hooks on Hunyuan double/single DiT blocks."""

    def __init__(self, transformer: nn.Module, adapter: HunyuanDiTControlAdapter):
        self.transformer = transformer
        self.adapter = adapter
        self._handles: list[Any] = []
        self._img_token_len: int | None = None
        self._latent_shape: Any = None

    @staticmethod
    def _iter_blocks(owner: Any, name: str) -> list[nn.Module]:
        blocks = getattr(owner, name, None)
        if blocks is None:
            return []
        if isinstance(blocks, nn.ModuleList):
            return list(blocks)
        if isinstance(blocks, nn.Sequential):
            return list(blocks)
        if isinstance(blocks, (list, tuple)):
            return [b for b in blocks if isinstance(b, nn.Module)]
        return []

    @staticmethod
    def _extract_tensor(output: Any) -> tuple[torch.Tensor | None, Callable[[torch.Tensor], Any]]:
        if isinstance(output, torch.Tensor):
            return output, lambda x: x
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return output[0], lambda x: (x, *output[1:])
        if isinstance(output, list) and output and isinstance(output[0], torch.Tensor):
            return output[0], lambda x: [x, *output[1:]]
        return None, lambda _x: output

    def _make_double_hook(self, layer_idx: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            tensor, rebuild = self._extract_tensor(output)
            if tensor is None or self.adapter.control_state is None:
                return output
            self._img_token_len = int(tensor.shape[1])
            residual = self.adapter.double_residual(layer_idx, tensor, self._latent_shape, int(tensor.shape[0]))
            return rebuild(tensor + residual)
        return hook

    def _make_single_hook(self, layer_idx: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            tensor, rebuild = self._extract_tensor(output)
            if tensor is None or self.adapter.control_state is None:
                return output
            # Hunyuan HYVideoDiffusionTransformer enters single_blocks as
            # torch.cat((img, txt), 1), so the image stream is the prefix.
            img_len = self._img_token_len or self.adapter.control_state.img_token_len or int(tensor.shape[1])
            residual = self.adapter.single_residual(layer_idx, tensor, img_len, self._latent_shape, int(tensor.shape[0]))
            return rebuild(tensor + residual)
        return hook

    def install(self) -> None:
        if self._handles:
            return
        for idx, block in enumerate(self._iter_blocks(self.transformer, "double_blocks")):
            self._handles.append(block.register_forward_hook(self._make_double_hook(idx)))
        for idx, block in enumerate(self._iter_blocks(self.transformer, "single_blocks")):
            self._handles.append(block.register_forward_hook(self._make_single_hook(idx)))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._img_token_len = None
        self._latent_shape = None

    @contextmanager
    def use_controls(self, *args: Any, latent_shape: Any = None, **kwargs: Any):
        self.install()
        self._latent_shape = latent_shape
        try:
            self.adapter.prepare_controls(*args, **kwargs)
            yield self
        finally:
            self.adapter.clear_control_state()
            self.remove()

    @contextmanager
    def use_control_state(self, state: HunyuanDiTControlState, *, latent_shape: Any = None):
        self.install()
        self._latent_shape = latent_shape
        try:
            self.adapter.set_control_state(state)
            yield self
        finally:
            self.adapter.clear_control_state()
            self.remove()


def _cfg_from_payload(value: Any, *, ckpt_path: Path) -> HunyuanDiTControlConfig:
    if isinstance(value, HunyuanDiTControlConfig):
        return value
    if hasattr(value, "__dict__") and not isinstance(value, dict):
        value = vars(value)
    if not isinstance(value, dict):
        raise RuntimeError(f"{ckpt_path} cfg must be a dict, got {type(value).__name__}")
    try:
        return HunyuanDiTControlConfig(**value)
    except TypeError as exc:
        raise RuntimeError(f"{ckpt_path} cfg does not match HunyuanDiTControlConfig: {exc}") from exc


def load_hunyuan_dit_control_checkpoint(path: str | Path, device: str | torch.device | None = None) -> tuple[HunyuanDiTControlAdapter, dict[str, Any]]:
    ckpt_path = Path(path)
    payload = torch.load(ckpt_path, map_location=device or "cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{ckpt_path} checkpoint must be a dict")
    if "kind" not in payload:
        raise RuntimeError(f"{ckpt_path} checkpoint is missing kind={CONTROL_CHECKPOINT_KIND}")
    actual_kind = payload.get("kind")
    if actual_kind != CONTROL_CHECKPOINT_KIND:
        raise RuntimeError(f"{ckpt_path} kind must be {CONTROL_CHECKPOINT_KIND}, got {actual_kind!r}")
    if "model" not in payload or "cfg" not in payload:
        raise RuntimeError(f"{ckpt_path} must contain model and cfg")
    cfg = _cfg_from_payload(payload["cfg"], ckpt_path=ckpt_path)
    adapter = HunyuanDiTControlAdapter(cfg)
    adapter.load_state_dict(payload["model"], strict=True)
    if device is not None:
        adapter = adapter.to(device)
    adapter.eval()
    return adapter, payload


def save_hunyuan_dit_control_checkpoint(
    path: str | Path,
    adapter: HunyuanDiTControlAdapter,
    *,
    metrics: dict[str, Any] | None = None,
    wm_ckpt: str | Path | None = None,
    step: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = adapter.module if hasattr(adapter, "module") else adapter
    if not isinstance(target, HunyuanDiTControlAdapter):
        raise TypeError(f"expected HunyuanDiTControlAdapter, got {type(target).__name__}")
    payload: dict[str, Any] = {
        "kind": CONTROL_CHECKPOINT_KIND,
        "model": target.state_dict(),
        "cfg": asdict(target.cfg),
        "metrics": metrics or {},
        "wm_ckpt": str(wm_ckpt) if wm_ckpt is not None else None,
        "step": step,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, Path(path))
    return payload


__all__ = [
    "CONTROL_CHECKPOINT_KIND",
    "HunyuanDiTControlAdapter",
    "HunyuanDiTControlConfig",
    "HunyuanDiTControlInjector",
    "HunyuanDiTControlState",
    "load_hunyuan_dit_control_checkpoint",
    "save_hunyuan_dit_control_checkpoint",
]
