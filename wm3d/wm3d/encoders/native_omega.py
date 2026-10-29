"""Frozen VGGT-Omega feature-only adapter for the native trajectory path.

The raw RGB contract stays untouched (518 square, original aspect/white pad).
Only this teacher resizes to the official 512 variant. No depth/point output is
invented: callers requesting geometry or RGB heads are rejected.
"""
from __future__ import annotations
from collections import defaultdict
import inspect
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

SCHEMA = "wm3d_frozen_vggt_omega_512_v1"

def validate_teacher(value):
    expected = {
        "schema": SCHEMA, "model_name": "facebook/VGGT-Omega",
        "variant": "vggt_omega_1b_512.pt", "checkpoint_bytes": 4576706117,
        "input_size": 512, "raw_input_size": 518, "patch_size": 16,
        "prefix_tokens": 17, "feature_layer": 23, "token_grid": 8,
        "token_dim": 2048, "dtype": "bf16",
        "resize": "bicubic_antialias_clamp",
        "time_isolation": "fold_time_into_batch",
        "missing_view_policy": "exclude_from_geometry",
    }
    if set(value) != set(expected) | {"source_root", "checkpoint_path"}:
        raise ValueError("Omega teacher fields differ")
    if any(value[k] != v for k, v in expected.items()):
        raise ValueError("Omega teacher feature contract differs")
    if not all(Path(value[k]).is_absolute() for k in ("source_root", "checkpoint_path")):
        raise ValueError("Omega asset paths must be absolute")

class NativeOmegaEncoder(torch.nn.Module):
    def __init__(self, config, teacher, *, device):
        super().__init__()
        config.validate()
        validate_teacher(teacher)
        if (config.input_rgb_size != 518 or config.token_grid != 8
                or config.token_dim != 2048 or config.appearance_token_grid):
            raise ValueError("Omega requires the unchanged native 518/P64/2048 raw ABI")
        self.config = config
        self.teacher = dict(teacher)
        self.device = torch.device(device)
        root = Path(teacher["source_root"]).resolve(strict=True)
        checkpoint = Path(teacher["checkpoint_path"]).resolve(strict=True)
        stat_before = checkpoint.stat()
        if stat_before.st_size != teacher["checkpoint_bytes"] or checkpoint.name != teacher["variant"]:
            raise ValueError("Omega checkpoint variant/size mismatch")
        for name, module in tuple(sys.modules.items()):
            if name == "vggt_omega" or name.startswith("vggt_omega."):
                path = getattr(module, "__file__", None)
                if path:
                    Path(path).resolve(strict=True).relative_to(root)
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(root))
        from vggt_omega.models.vggt_omega import VGGTOmega
        Path(inspect.getsourcefile(VGGTOmega)).resolve(strict=True).relative_to(root)
        self.model = VGGTOmega(autocast=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if len(state) != 1411 or sum(t.numel() for t in state.values()) != 1144059977:
            raise ValueError("Omega complete official state layout differs")
        self.model.load_state_dict(state, strict=True)
        after = checkpoint.stat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(stat_before, k) != getattr(after, k) for k in fields):
            raise RuntimeError("Omega checkpoint changed during load")
        self.model.to(self.device).eval().requires_grad_(False)
        self.load_evidence = dict(tensors=len(state),
            elements=sum(t.numel() for t in state.values()), strict=True,
            checkpoint=str(checkpoint), source_root=str(root))
        del state

    def train(self, mode=True):
        super().train(False)
        self.model.eval()
        return self

    @torch.inference_mode()
    def forward(self, images, view_mask, *, geometry_row_mask=None,
                appearance_row_mask=None, rgb_row_mask=None):
        if images.ndim != 6:
            raise ValueError("Omega images must be [B,T,V,3,H,W]")
        b, t, v, c, h, w = images.shape
        if (c, h, w) != (3, 518, 518) or not 0 < v <= self.config.max_views:
            raise ValueError("Omega raw RGB ABI differs")
        if view_mask.dtype != torch.bool or tuple(view_mask.shape) != (b,t,v):
            raise ValueError("Omega view mask must be boolean [B,T,V]")
        if not bool(view_mask.any(-1).all()):
            raise ValueError("Omega requires a real view at every time")
        if not bool(torch.isfinite(images).all()) or float(images.min()) < 0 or float(images.max()) > 1:
            raise ValueError("Omega RGB must be finite and in [0,1]")
        for mask in (geometry_row_mask, appearance_row_mask, rgb_row_mask):
            if mask is None or tuple(mask.shape) != (b,t) or bool(mask.any()):
                raise ValueError("Omega trajectory adapter exposes features only, not geometry/RGB heads")
        flat = images.reshape(b*t,v,3,h,w).to(self.device)
        masks = view_mask.reshape(b*t,v).to(self.device)
        partitions = defaultdict(list)
        for row, pattern in enumerate(masks.cpu().tolist()):
            partitions[tuple(pattern)].append(row)
        output = torch.zeros((b*t,v,64,2048), device=self.device, dtype=torch.bfloat16)
        for pattern, indices in sorted(partitions.items()):
            rows = torch.tensor(indices, device=self.device)
            active = torch.tensor([i for i,x in enumerate(pattern) if x], device=self.device)
            selected = flat.index_select(0,rows).index_select(1,active)
            n, nv = selected.shape[:2]
            selected = F.interpolate(selected.flatten(0,1).float(), size=(512,512),
                mode="bicubic", align_corners=False, antialias=True).clamp_(0,1)
            selected = selected.reshape(n,nv,3,512,512)
            with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                enabled=self.device.type == "cuda"):
                features, prefix = self.model.aggregator(selected)
            patches = features[23]
            if int(prefix) != 17 or patches is None or tuple(patches.shape) != (n,nv,1041,2048):
                raise RuntimeError("Official Omega final-layer/prefix/patch ABI differs")
            patches = patches[:,:,17:,:].permute(0,1,3,2).reshape(n*nv,2048,32,32)
            pooled = F.adaptive_avg_pool2d(patches.float(), (8,8))
            pooled = pooled.reshape(n,nv,2048,64).permute(0,1,3,2).to(torch.bfloat16)
            if not bool(torch.isfinite(pooled).all()):
                raise FloatingPointError("Omega features contain NaN/Inf")
            for local, global_view in enumerate(active.tolist()):
                output[rows,global_view] = pooled[:,local]
        return {"view_tokens": output.reshape(b,t,v,64,2048),
                "view_mask": view_mask.to(self.device)}
