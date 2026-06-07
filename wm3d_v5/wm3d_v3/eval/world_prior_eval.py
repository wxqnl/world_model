"""Evaluate text-conditioned world-prior generation on cached validation windows.

This complements action-conditioned rollout metrics. It checks the 3D-native
prior modes that matter for the current architecture:

- text -> world
- text + RGB/context tokens -> world
- text + action -> world rollout
- text + context + action -> world rollout
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.eval.run_eval import build_dataset_for_split, build_model
from wm3d_v3.losses import _normalize_depth


def _accum(dst: dict[str, float], key: str, value: torch.Tensor) -> None:
    dst[key] += float(value.detach().float().sum().cpu())


def _mode_subset(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    all_metrics = metrics.get("ALL")
    return all_metrics if isinstance(all_metrics, dict) else {}


def context_rgb_for_world_prior(batch: dict[str, Any], cfg: dict[str, Any], device: torch.device | str, *, pixel: bool) -> torch.Tensor | None:
    if not bool(pixel):
        return None
    return batch["rgb_in"][:, -1].to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max_batches", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8, help="Euler flow sampling steps")
    ap.add_argument("--pixel", action="store_true", help="also decode prior_rgb and RGB L1 metrics")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.cfg.read_text())
    if not cfg.get("model", {}).get("enable_world_prior", False):
        raise RuntimeError("world_prior_eval requires model.enable_world_prior=true")
    if not cfg.get("data", {}).get("require_task_emb", False):
        raise RuntimeError("world_prior_eval requires data.require_task_emb=true")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    records = read_manifest(cfg["data"]["manifest"])
    val = build_dataset_for_split(records, cfg, split="val")
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else int(cfg["train"].get("batch_size_per_gpu", 1))
    loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])

    by_ds: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    cnt: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    required_keys = ("prior_future_tokens", "prior_depth", "prior_hunyuan_tokens", "prior_hunyuan_depth")

    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        s = batch["s_in"].to(device, non_blocking=True)
        c = batch["c"].to(device, non_blocking=True)
        s_tgt = batch["s_tgt"].to(device, non_blocking=True)
        depth_tgt = batch["depth_tgt"].to(device, non_blocking=True)
        action_tgt = batch["action_tgt"].to(device, non_blocking=True)
        action_tgt_norm = batch["action_tgt_norm"].to(device, non_blocking=True)
        action_cond = make_action_condition(action_tgt, action_tgt_norm)
        context_rgb = context_rgb_for_world_prior(batch, cfg, device, pixel=args.pixel)
        rgb_tgt = None
        if args.pixel:
            rgb_tgt = batch["rgb_tgt"].to(device, non_blocking=True).permute(0, 1, 4, 2, 3).contiguous()

        modes = {
            "text_only": (None, None, None),
            "text_context": (s, None, context_rgb),
            "text_action": (None, action_cond, None),
            "full": (s, action_cond, context_rgb),
        }
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            outs = {
                name: model.generate_world_prior(
                    c,
                    context_tokens=ctx,
                    action_cond=act,
                    context_rgb=rgb_ctx,
                    steps=args.steps,
                    pixel=args.pixel,
                )
                for name, (ctx, act, rgb_ctx) in modes.items()
            }

        depth_tn = _normalize_depth(depth_tgt.float())
        for mode, out in outs.items():
            missing = [key for key in required_keys if key not in out]
            if missing:
                raise RuntimeError(f"{mode} missing required prior outputs: {missing}")
            pred = out["prior_future_tokens"].float()
            depth = _normalize_depth(out["prior_depth"].float())
            token_mse = F.mse_loss(pred, s_tgt.float(), reduction="none").mean(dim=(1, 2, 3))
            token_cos = F.cosine_similarity(pred.flatten(-2), s_tgt.float().flatten(-2), dim=-1).mean(dim=-1)
            depth_l1 = (depth - depth_tn).abs().mean(dim=(1, 2, 3))
            init_mse = None
            if "prior_initial_tokens" in out:
                init_mse = F.mse_loss(
                    out["prior_initial_tokens"].float(),
                    s[:, -1].float(),
                    reduction="none",
                ).mean(dim=(1, 2))
            rgb_l1 = None
            if args.pixel:
                if "prior_rgb" not in out:
                    raise RuntimeError(f"{mode} pixel eval requested but prior_rgb missing")
                if rgb_tgt is None:
                    raise RuntimeError("internal RGB target setup failed")
                rgb_l1 = (out["prior_rgb"].float() - rgb_tgt.float()).abs().mean(dim=(1, 2, 3, 4))

            for i in range(s.shape[0]):
                datasets = (batch["dataset"][i], "ALL")
                for d in datasets:
                    m = by_ds[d][mode]
                    m["prior_token_mse"] += float(token_mse[i])
                    m["prior_token_cos"] += float(token_cos[i])
                    m["prior_depth_l1"] += float(depth_l1[i])
                    m["prior_token_abs_mean"] += float(pred[i].abs().mean())
                    m["prior_depth_mean"] += float(out["prior_depth"][i].float().mean())
                    m["prior_hunyuan_token_abs_mean"] += float(out["prior_hunyuan_tokens"][i].float().abs().mean())
                    m["prior_hunyuan_depth_mean"] += float(out["prior_hunyuan_depth"][i].float().mean())
                    if init_mse is not None:
                        m["prior_init_mse"] += float(init_mse[i])
                    if rgb_l1 is not None:
                        m["prior_rgb_l1"] += float(rgb_l1[i])
                    cnt[d][mode] += 1
        if (bi + 1) % 10 == 0:
            all_full = by_ds["ALL"]["full"]
            n = max(1, cnt["ALL"]["full"])
            token_mse = all_full.get("prior_token_mse", 0.0) / n
            depth_l1 = all_full.get("prior_depth_l1", 0.0) / n
            print(
                f"[{bi+1}/{len(loader)}] full prior_token_mse {token_mse:.4f} "
                f"depth_l1 {depth_l1:.4f}"
            )

    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for d, modes in by_ds.items():
        metrics[d] = {}
        for mode, vals in modes.items():
            n = max(1, cnt[d][mode])
            metrics[d][mode] = {k: v / n for k, v in vals.items()}

    report = {
        "cfg": str(args.cfg),
        "ckpt": str(args.ckpt),
        "mode": {
            "device": str(device),
            "flow_steps": args.steps,
            "pixel": bool(args.pixel),
            "require_task_emb": bool(cfg["data"].get("require_task_emb", False)),
        },
        "counts": {d: dict(v) for d, v in cnt.items()},
        "metrics": metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    print(json.dumps(_mode_subset(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
