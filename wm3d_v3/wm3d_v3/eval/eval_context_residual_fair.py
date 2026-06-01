"""Evaluate a context residual decoder under its configured input ablations."""
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
from wm3d_v3.training.train_context_residual_pixel import apply_train_ablations


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


def edge_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    tgt_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    tgt_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)


def batch_to_device(batch: dict, device: torch.device):
    tokens = batch["s_tgt"].to(device, non_blocking=True)
    context = batch["rgb_in"][:, -1].to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
    target = batch["rgb_tgt"].to(device, non_blocking=True).permute(0, 1, 4, 2, 3).contiguous()
    action = batch["action_tgt"].to(device, non_blocking=True)
    action_norm = batch["action_tgt_norm"].to(device, non_blocking=True)
    action_cond = make_action_condition(action, action_norm)
    task = batch["c"].to(device, non_blocking=True)
    return tokens, context, target, action_cond, task


@torch.no_grad()
def batch_metrics(pred: torch.Tensor, target: torch.Tensor, context: torch.Tensor, lpips_fn) -> dict[str, float]:
    bsz, horizon, channels, _, _ = pred.shape
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
        "context_repeat_l1": float((ref - target).abs().mean(dim=(1, 2, 3, 4)).sum().item()),
        "count": float(bsz),
    }


def normalize(metrics: dict[str, float]) -> dict[str, float]:
    count = max(1.0, metrics["count"])
    return {k: v / count for k, v in metrics.items() if k != "count"} | {"count": count}


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=16)
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

    accum: dict[str, float] = {}
    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        tokens, context, target, action_cond, task = batch_to_device(batch, device)
        eval_context = context
        tokens, context, action_cond, task = apply_train_ablations(tokens, context, action_cond, task, cfg)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(tokens, context, action_cond=action_cond, task_emb=task)
        vals = batch_metrics(pred.float().clamp(0, 1), target, eval_context, lpips_fn)
        for key, value in vals.items():
            accum[key] = accum.get(key, 0.0) + value
        if (bi + 1) % 50 == 0:
            cur = normalize(accum)
            print(
                f"[{bi+1}/{len(loader)}] L1={cur['l1']:.4f} "
                f"motion={cur['motion_l1']:.4f} lpips={cur['lpips']:.4f}",
                flush=True,
            )

    report = {
        "cfg": str(args.cfg),
        "ckpt": str(args.ckpt),
        "ckpt_epoch": ckpt.get("epoch"),
        "ckpt_val_total": ckpt.get("val_total"),
        "val_windows": len(val),
        "input_ablations": {
            "zero_tokens": bool(cfg["train"].get("zero_tokens", False)),
            "zero_context": bool(cfg["train"].get("zero_context", False)),
            "zero_action": bool(cfg["train"].get("zero_action", False)),
            "zero_task": bool(cfg["train"].get("zero_task", False)),
        },
        "metrics": normalize(accum),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
