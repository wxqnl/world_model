"""Ablate inputs of the context residual pixel decoder on a validation split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.window_dataset import OXEWindowDataset, WindowConfig
from wm3d_v3.models.context_residual_pixel_decoder import (
    ContextResidualPixelDecoder,
    ContextResidualPixelDecoderConfig,
)


def window_config_from_cfg(cfg: dict) -> WindowConfig:
    data = cfg["data"]
    action_stats = data.get("action_stats")
    return WindowConfig(
        T=data["T"],
        k=data["k"],
        stride=data["stride"],
        cache_root=Path(data["cache_root"]),
        tokens_subdir=data.get("tokens_subdir", "vggt_p256"),
        action_stats=Path(action_stats) if action_stats else None,
    )


def batch_to_device(batch: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    tokens = batch["s_tgt"].to(device, non_blocking=True)
    context = batch["rgb_in"][:, -1].to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
    target = batch["rgb_tgt"].to(device, non_blocking=True).permute(0, 1, 4, 2, 3).contiguous()
    action = batch["action_tgt"].to(device, non_blocking=True)
    action_norm = batch["action_tgt_norm"].to(device, non_blocking=True)
    action_cond = make_action_condition(action, action_norm)
    task = batch["c"].to(device, non_blocking=True)
    return tokens, context, target, action_cond, task


def edge_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    tgt_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    tgt_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)


@torch.no_grad()
def metrics_for(pred: torch.Tensor, target: torch.Tensor, context: torch.Tensor, lpips_fn) -> dict[str, float]:
    bsz, horizon, channels, height, width = pred.shape
    ref = context[:, None].expand_as(target)
    motion = (target - ref).abs().mean(dim=2, keepdim=True)
    motion_mask = (motion > 0.03).float()
    motion_denom = (motion_mask.sum(dim=(1, 2, 3, 4)) * channels).clamp_min(1.0)
    motion_l1 = ((pred - target).abs() * motion_mask).sum(dim=(1, 2, 3, 4)) / motion_denom
    with torch.autocast(device_type="cuda", enabled=False):
        lp = lpips_fn(
            pred.flatten(0, 1).float() * 2.0 - 1.0,
            target.flatten(0, 1).float() * 2.0 - 1.0,
        ).view(bsz, horizon).mean(dim=1)
    return {
        "l1": float((pred - target).abs().mean(dim=(1, 2, 3, 4)).sum().item()),
        "motion_l1": float(motion_l1.sum().item()),
        "lpips": float(lp.sum().item()),
        "edge_l1": float(edge_l1(pred, target).item() * bsz),
        "motion_frac": float(motion_mask.mean(dim=(1, 2, 3, 4)).sum().item()),
        "count": float(bsz),
    }


def add_metrics(dst: dict[str, dict[str, float]], name: str, vals: dict[str, float]) -> None:
    cur = dst.setdefault(name, {})
    for key, value in vals.items():
        cur[key] = cur.get(key, 0.0) + value


def normalize_metrics(raw: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for name, vals in raw.items():
        count = max(1.0, vals.get("count", 0.0))
        out[name] = {key: value / count for key, value in vals.items() if key != "count"}
        out[name]["count"] = count
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_batches", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.cfg.read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    records = read_manifest(cfg["data"]["manifest"])
    ds = OXEWindowDataset(records, window_config_from_cfg(cfg))
    gen = torch.Generator().manual_seed(cfg["data"]["seed"])
    perm = torch.randperm(len(ds), generator=gen).tolist()
    n_val = max(1, int(len(ds) * cfg["data"]["val_frac"]))
    val = Subset(ds, perm[:n_val])
    loader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["train"]["num_workers"] > 0,
    )

    model = ContextResidualPixelDecoder(ContextResidualPixelDecoderConfig(**cfg["model"])).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    raw: dict[str, dict[str, float]] = {}
    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        tokens, context, target, action_cond, task = batch_to_device(batch, device)
        variants: dict[str, torch.Tensor] = {
            "context_repeat": context[:, None].expand_as(target),
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            variants["normal"] = model(tokens, context, action_cond=action_cond, task_emb=task)
            variants["zero_tokens"] = model(torch.zeros_like(tokens), context, action_cond=action_cond, task_emb=task)
            if tokens.shape[0] > 1:
                variants["shuffle_tokens"] = model(tokens.roll(1, dims=0), context, action_cond=action_cond, task_emb=task)
            variants["zero_context"] = model(tokens, torch.zeros_like(context), action_cond=action_cond, task_emb=task)
            variants["zero_action"] = model(tokens, context, action_cond=torch.zeros_like(action_cond), task_emb=task)
            variants["zero_task"] = model(tokens, context, action_cond=action_cond, task_emb=torch.zeros_like(task))
        for name, pred in variants.items():
            add_metrics(raw, name, metrics_for(pred.float().clamp(0, 1), target, context, lpips_fn))
        if (bi + 1) % 25 == 0:
            partial = normalize_metrics(raw)
            msg = " ".join(
                f"{name}:L1={vals['l1']:.4f}/M={vals['motion_l1']:.4f}"
                for name, vals in sorted(partial.items())
            )
            print(f"[{bi+1}/{len(loader)}] {msg}", flush=True)

    report = {
        "cfg": str(args.cfg),
        "ckpt": str(args.ckpt),
        "ckpt_epoch": ckpt.get("epoch"),
        "ckpt_val_total": ckpt.get("val_total"),
        "val_windows": len(val),
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "metrics": normalize_metrics(raw),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["metrics"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
