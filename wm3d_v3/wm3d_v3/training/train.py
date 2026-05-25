"""v3 joint training. DDP-aware, bf16, gradient checkpointing optional."""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
import lpips
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.window_dataset import OXEWindowDataset, WindowConfig
from wm3d_v3.losses import LossWeights, compute_losses
from wm3d_v3.models.action_stream import ActionConfig
from wm3d_v3.models.dual_stream import DualConfig
from wm3d_v3.models.joint_model import JointConfig, JointWorldModel
from wm3d_v3.models.state_stream import StateConfig


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


def build_model(cfg: dict) -> JointWorldModel:
    sc = StateConfig(**cfg["model"]["state"])
    ac = ActionConfig(**cfg["model"]["action"])
    dc = DualConfig(state=sc, action=ac,
                    xattn_layers_state=tuple(cfg["model"]["xattn_layers_state"]),
                    xattn_n_heads=cfg["model"]["xattn_n_heads"])
    jc = JointConfig(
        dual=dc,
        action_proj_hidden=cfg["model"]["action_proj_hidden"],
        action_proj_layers=cfg["model"]["action_proj_layers"],
        geom_hidden=cfg["model"]["geom_hidden"],
        pixel_hidden=cfg["model"]["pixel_hidden"],
        pixel_n_res=cfg["model"]["pixel_n_res"],
        enable_pixel=cfg["model"].get("enable_pixel", True),
        enable_bridging=cfg["model"].get("enable_bridging", True),
    )
    return JointWorldModel(jc)


def build_datasets(cfg: dict, overfit_ids=None):
    records = read_manifest(cfg["data"]["manifest"])
    if overfit_ids:
        records = [r for r in records if r.clip_id in overfit_ids]
        if not records:
            raise RuntimeError(f"no records matched overfit ids: {overfit_ids}")
    wcfg = WindowConfig(T=cfg["data"]["T"], k=cfg["data"]["k"],
                        stride=cfg["data"]["stride"],
                        cache_root=Path(cfg["data"]["cache_root"]))
    ds = OXEWindowDataset(records, wcfg)
    n = len(ds)
    if n == 0:
        raise RuntimeError("OXEWindowDataset empty — caches missing?")
    g = torch.Generator().manual_seed(cfg["data"]["seed"])
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(n * cfg["data"]["val_frac"]))
    return Subset(ds, perm[n_val:]), Subset(ds, perm[:n_val])


def batch_to_device(batch: dict, device: torch.device, k: int) -> tuple:
    s = batch["s_in"].to(device, non_blocking=True)
    c = batch["c"].to(device, non_blocking=True)
    rgb_tgt_p = batch["rgb_tgt"].to(device, non_blocking=True).permute(0, 1, 4, 2, 3)
    # rgb_tgt: [B, k, 256, 256, 3]; permute -> [B, k, 3, 256, 256]
    tgt = {
        "s_tgt": batch["s_tgt"].to(device, non_blocking=True),
        "depth_tgt": batch["depth_tgt"].to(device, non_blocking=True),
        "action_tgt": batch["action_tgt"].to(device, non_blocking=True),
        "rgb_tgt_p": rgb_tgt_p,
    }
    return s, c, tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--disable_pixel_until", type=int, default=0,
                    help="train first N epochs without L_rgb (stage 1)")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.cfg.read_text())
    rank, world, local = setup_ddp()
    device = torch.device(f"cuda:{local}")

    overfit_ids = cfg.get("overfit_clip_ids") if args.overfit else None
    tr_ds, val_ds = build_datasets(cfg, overfit_ids=overfit_ids)
    bs = cfg["train"]["batch_size_per_gpu"]; nw = cfg["train"]["num_workers"]
    if world > 1:
        tr_s = DistributedSampler(tr_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        tr_loader = DataLoader(tr_ds, batch_size=bs, sampler=tr_s, num_workers=nw,
                                pin_memory=True, drop_last=True)
        val_s = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=bs, sampler=val_s, num_workers=nw, pin_memory=True)
    else:
        tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    model = build_model(cfg).to(device)
    if rank == 0:
        n_p = model.num_trainable_params()
        print(f"[rank0] JointWorldModel: {n_p/1e6:.1f}M; train_windows={len(tr_ds)} val_windows={len(val_ds)}")
    if world > 1:
        model = DDP(model, device_ids=[local], find_unused_parameters=True)

    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                             weight_decay=cfg["train"]["weight_decay"], betas=(0.9, 0.95))
    warmup = int(cfg["train"]["warmup_steps"])
    total_steps = max(1, len(tr_loader) * cfg["train"]["epochs"])
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    weights = LossWeights(**cfg["loss"])
    out_root = Path(cfg["out"]["root"])
    ckpt_dir = out_root / cfg["out"]["ckpt_dir"]
    if rank == 0:
        (out_root / cfg["out"]["tb_dir"]).mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tb = SummaryWriter(out_root / cfg["out"]["tb_dir"])

    start_epoch = 0; step = 0; best_val = float("inf")
    if args.resume is not None and args.resume.exists():
        sd = torch.load(args.resume, map_location=device, weights_only=False)
        (model.module if world > 1 else model).load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"]); sched.load_state_dict(sd["sched"])
        start_epoch = sd["epoch"] + 1; step = sd["step"]; best_val = sd.get("best_val", best_val)
        if rank == 0:
            print(f"[rank0] resumed from {args.resume} at epoch {start_epoch}")

    k = cfg["data"]["k"]
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        if world > 1:
            tr_s.set_epoch(epoch)
        do_pixel = epoch >= args.disable_pixel_until
        model.train()
        for batch in tr_loader:
            s, c, tgt = batch_to_device(batch, device, k)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = (model.module if world > 1 else model)(s, c, pixel=do_pixel, bridging=False) \
                    if not isinstance(model, DDP) else model(s, c, pixel=do_pixel, bridging=False)
                losses = compute_losses(out, tgt, weights, lpips_fn if do_pixel else None)
            opt.zero_grad(set_to_none=True)
            losses["L_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step(); sched.step()
            if rank == 0 and step % cfg["train"]["log_every"] == 0:
                for k_, v in losses.items():
                    tb.add_scalar(f"train/{k_}", float(v.detach().float()), step)
                tb.add_scalar("lr", sched.get_last_lr()[0], step)
                tb.add_scalar("stage_pixel", float(do_pixel), step)
            step += 1
        # Val
        model.eval()
        agg = {}; nb = 0
        with torch.no_grad():
            for batch in val_loader:
                s, c, tgt = batch_to_device(batch, device, k)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(s, c, pixel=do_pixel, bridging=False)
                    losses = compute_losses(out, tgt, weights, lpips_fn if do_pixel else None)
                for kk, v in losses.items():
                    agg[kk] = agg.get(kk, 0.0) + float(v.detach().float())
                nb += 1
        if world > 1:
            keys = sorted(agg.keys())
            v = torch.tensor([agg[kk] for kk in keys] + [float(nb)], device=device)
            dist.all_reduce(v)
            tot_nb = float(v[-1].item())
            for i, kk in enumerate(keys):
                agg[kk] = float(v[i].item()) / max(1.0, tot_nb)
            nb = 1
        if rank == 0:
            for kk, vv in agg.items():
                tb.add_scalar(f"val/{kk}", vv / max(1, nb), step)
            val_total = agg["L_total"] / max(1, nb)
            ckpt = {"model": (model.module if world > 1 else model).state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "step": step, "val_total": val_total,
                    "best_val": best_val, "cfg": cfg}
            if (epoch + 1) % cfg["train"]["ckpt_every_epochs"] == 0:
                torch.save(ckpt, ckpt_dir / f"epoch_{epoch:03d}.pt")
            if val_total < best_val:
                best_val = val_total
                ckpt["best_val"] = best_val
                torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"[rank0] epoch {epoch}: val_total {val_total:.4f} (best {best_val:.4f}, pixel={do_pixel})")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
