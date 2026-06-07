"""Decode a trained Hunyuan VAE latent adapter demo.

This is intentionally a VAE latent adapter demo. It does not run the Hunyuan
DiT or video generator.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from PIL import Image

from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.window_dataset import OXEWindowDataset
from wm3d_v3.eval.make_demo_gif import window_config_from_cfg
from wm3d_v3.eval.run_eval import build_model
from wm3d_v3.models.hunyuan_latent_adapter import HunyuanLatentAdapter, HunyuanLatentAdapterConfig
from wm3d_v3.training.train import encode_hunyuan_latents, load_hunyuan_vae, target_video_from_batch


@dataclass
class HunyuanDemoCheckpoint:
    source_format: str
    world_model_state: dict[str, Any]
    adapter_state: dict[str, Any]
    adapter_cfg: HunyuanLatentAdapterConfig


@torch.no_grad()
def decode_hunyuan_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    z = latents / float(vae.config.scaling_factor)
    out = vae.decode(z.to(dtype=vae.dtype), return_dict=False)[0]
    return out.div(2.0).add(0.5).clamp(0.0, 1.0).float()


def _require_mapping(value: Any, *, name: str, ckpt_path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{ckpt_path} has invalid {name}: expected a dict, got {type(value).__name__}")
    return value


def _adapter_cfg_from_payload(value: Any, *, ckpt_path: Path, key: str) -> HunyuanLatentAdapterConfig:
    if isinstance(value, HunyuanLatentAdapterConfig):
        return value
    if hasattr(value, "__dict__") and not isinstance(value, dict):
        value = vars(value)
    cfg = _require_mapping(value, name=key, ckpt_path=ckpt_path)
    try:
        return HunyuanLatentAdapterConfig(**cfg)
    except TypeError as exc:
        raise RuntimeError(
            f"{ckpt_path} {key} does not match HunyuanLatentAdapterConfig; "
            "expected a standalone scripts/train_hunyuan_latent_adapter.py checkpoint "
            "or a trainer checkpoint with hunyuan_adapter_cfg. "
            f"Original error: {exc}"
        ) from exc


def resolve_hunyuan_demo_checkpoint(
    ckpt: dict[str, Any],
    *,
    ckpt_path: Path,
    wm_ckpt_path: Path | None,
    load_checkpoint: Callable[[Path], dict[str, Any]],
) -> HunyuanDemoCheckpoint:
    """Resolve supported latent-demo checkpoint formats."""
    ckpt = _require_mapping(ckpt, name="checkpoint", ckpt_path=ckpt_path)
    if "hunyuan_adapter" in ckpt:
        if "model" not in ckpt:
            raise RuntimeError(f"{ckpt_path} contains hunyuan_adapter but no world model state at key model")
        if "hunyuan_adapter_cfg" not in ckpt:
            raise RuntimeError(f"{ckpt_path} contains hunyuan_adapter but no hunyuan_adapter_cfg")
        return HunyuanDemoCheckpoint(
            source_format="trainer_embedded_adapter",
            world_model_state=_require_mapping(ckpt["model"], name="model", ckpt_path=ckpt_path),
            adapter_state=_require_mapping(ckpt["hunyuan_adapter"], name="hunyuan_adapter", ckpt_path=ckpt_path),
            adapter_cfg=_adapter_cfg_from_payload(ckpt["hunyuan_adapter_cfg"], ckpt_path=ckpt_path, key="hunyuan_adapter_cfg"),
        )

    if "model" in ckpt and "cfg" in ckpt:
        if wm_ckpt_path is None:
            raise RuntimeError(
                f"{ckpt_path} looks like a standalone Hunyuan latent adapter checkpoint with keys model and cfg; "
                "pass --wm_ckpt pointing to the frozen world model checkpoint."
            )
        wm_ckpt = _require_mapping(load_checkpoint(wm_ckpt_path), name="wm_ckpt", ckpt_path=wm_ckpt_path)
        if "model" not in wm_ckpt:
            raise RuntimeError(f"--wm_ckpt {wm_ckpt_path} has no model state")
        return HunyuanDemoCheckpoint(
            source_format="standalone_adapter",
            world_model_state=_require_mapping(wm_ckpt["model"], name="wm_ckpt.model", ckpt_path=wm_ckpt_path),
            adapter_state=_require_mapping(ckpt["model"], name="model", ckpt_path=ckpt_path),
            adapter_cfg=_adapter_cfg_from_payload(ckpt["cfg"], ckpt_path=ckpt_path, key="cfg"),
        )

    keys = ", ".join(sorted(str(k) for k in ckpt.keys()))
    raise RuntimeError(
        f"Unsupported Hunyuan latent demo checkpoint {ckpt_path}; keys=[{keys}]. "
        "Expected either a trainer checkpoint with keys model, hunyuan_adapter, hunyuan_adapter_cfg, "
        "or a standalone adapter checkpoint with keys model and cfg plus --wm_ckpt. "
        "This demo only decodes VAE latents and does not run a Hunyuan DiT or flow video generator."
    )


def resize_frame(frame: torch.Tensor, hw: tuple[int, int]) -> np.ndarray:
    arr = frame.permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy()
    arr = (arr * 255).round().astype(np.uint8)
    if arr.shape[:2] != hw:
        arr = np.array(Image.fromarray(arr).resize((hw[1], hw[0]), Image.BILINEAR))
    return arr


def save_demo(path: Path, target: torch.Tensor, pred: torch.Tensor, rough: torch.Tensor | None) -> None:
    """Save target, Hunyuan adapter decode, and optional rough RGB as a GIF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    target = target[0]
    pred = pred[0]
    rough0 = rough[0] if rough is not None else None
    t = min(target.shape[1], pred.shape[1])
    hw = tuple(target.shape[-2:])
    frames = []
    for i in range(t):
        row = [
            resize_frame(target[:, i], hw),
            resize_frame(pred[:, i], hw),
        ]
        if rough0 is not None:
            row.append(resize_frame(rough0[:, i], hw))
        frames.append(np.concatenate(row, axis=1))
    imageio.mimsave(path, frames, duration=0.18)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--wm_ckpt", type=Path, default=None, help="world model ckpt for standalone adapter checkpoints")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_clips", type=int, default=2)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.cfg.read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    records = read_manifest(cfg["data"]["manifest"])
    ds = OXEWindowDataset(records, window_config_from_cfg(cfg))
    g = torch.Generator().manual_seed(cfg["data"]["seed"])
    perm = torch.randperm(len(ds), generator=g).tolist()
    n_val = max(1, int(len(ds) * cfg["data"]["val_frac"]))

    def load_ckpt(path: Path) -> dict[str, Any]:
        return torch.load(path, map_location=device, weights_only=False)

    payload = resolve_hunyuan_demo_checkpoint(
        load_ckpt(args.ckpt),
        ckpt_path=args.ckpt,
        wm_ckpt_path=args.wm_ckpt,
        load_checkpoint=load_ckpt,
    )
    print(f"hunyuan_latent_demo_checkpoint format={payload.source_format} ckpt={args.ckpt} wm_ckpt={args.wm_ckpt}")

    model = build_model(cfg).to(device).eval()
    model.load_state_dict(payload.world_model_state)
    adapter = HunyuanLatentAdapter(payload.adapter_cfg).to(device).eval()
    adapter.load_state_dict(payload.adapter_state)
    vae = load_hunyuan_vae(cfg["train"], device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str]] = set()
    made = 0
    for vi in perm[:n_val]:
        smp = ds[vi]
        key = (smp["dataset"], smp["clip_id"])
        if key in seen:
            continue
        seen.add(key)

        s = smp["s_in"].unsqueeze(0).to(device)
        c = smp["c"].unsqueeze(0).to(device)
        action_tgt = smp["action_tgt"].unsqueeze(0).to(device)
        action_tgt_norm = smp["action_tgt_norm"].unsqueeze(0).to(device)
        context_rgb = smp["rgb_in"][-1].permute(2, 0, 1).unsqueeze(0).to(device)
        rgb_tgt_p = smp["rgb_tgt"].permute(0, 3, 1, 2).unsqueeze(0).to(device)
        action_cond = make_action_condition(action_tgt, action_tgt_norm)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = model(s, c, action_cond=action_cond, context_rgb=context_rgb, pixel=True, bridging=False)
            target_video = target_video_from_batch(context_rgb, rgb_tgt_p)
            rough_rgb = out.get("rgb")
            pred_latents = adapter(
                out["pred_tokens"],
                out["depth"],
                context_rgb=context_rgb,
                motion_hint=out.get("motion_hint"),
                rough_rgb=rough_rgb,
                action_cond=action_cond,
                task_emb=c,
                target_latents=None,
            )
        rough_video = None
        if rough_rgb is not None:
            rough_video = torch.cat([context_rgb[:, None], rough_rgb.float()], dim=1).permute(0, 2, 1, 3, 4)
        if bool(cfg["train"].get("hunyuan_residual_from_rough", False)):
            if rough_video is None:
                raise RuntimeError(
                    "hunyuan_residual_from_rough=true requires the world model to output rough RGB; "
                    "use a world model ckpt and config with pixel or context pixel renderer enabled."
                )
            rough_latents = encode_hunyuan_latents(vae, rough_video.float())
            pred_latents = rough_latents.to(dtype=pred_latents.dtype) + float(
                cfg["train"].get("hunyuan_residual_scale", 1.0)
            ) * pred_latents
        pred_video = decode_hunyuan_latents(vae, pred_latents.float())

        dataset = smp["dataset"]
        clip_id = smp["clip_id"].replace("/", "__")
        out_path = args.out_dir / f"{made:02d}_{dataset}_{clip_id}_hunyuan_latent.gif"
        save_demo(out_path, target_video.float(), pred_video, rough_video.float() if rough_video is not None else None)
        print(f"wrote {out_path} pred={tuple(pred_video.shape)} target={tuple(target_video.shape)}")
        made += 1
        if made >= args.n_clips:
            break


if __name__ == "__main__":
    main()
