"""v3.5: scale up PixelDecoder only.

Loads v3 best.pt (dual + geom + action + small pixel decoder).
Builds a v3 JointWorldModel with a BIGGER PixelDecoder config.
Loads non-pixel weights, freezes them; new PixelDecoder is randomly initialized.
Trains only the new PixelDecoder using the original v3 L1+LPIPS+token losses
(though backbone is frozen, the token MSE is still computed for reporting).
"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path

import lpips
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.window_dataset import OXEWindowDataset, WindowConfig
from wm3d_v3.losses import LossWeights, compute_losses
from wm3d_v3.models.state_stream import StateConfig
from wm3d_v3.models.action_stream import ActionConfig
from wm3d_v3.models.dual_stream import DualConfig
from wm3d_v3.models.joint_model import JointConfig, JointWorldModel


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank(); world = dist.get_world_size()
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
        enable_pixel=True,
        enable_bridging=cfg["model"].get("enable_bridging", True),
    )
    return JointWorldModel(jc)


def load_v3_nonpixel(model: JointWorldModel, v3_ckpt: Path):
    sd = torch.load(v3_ckpt, map_location="cpu", weights_only=False)
    v3_state = sd["model"] if "model" in sd else sd
    own = set(model.state_dict().keys())
    # Take everything EXCEPT the pixel decoder (since shape may differ)
    to_load = {k: v for k, v in v3_state.items()
                if k in own and not k.startswith("pixel.")
                and model.state_dict()[k].shape == v.shape}
    res = model.load_state_dict(to_load, strict=False)
    return res, len(to_load)


def build_datasets(cfg: dict, max_records: int = 0):
    records = read_manifest(cfg["data"]["manifest"])
    if max_records:
        records = records[:max_records]
    wcfg = WindowConfig(T=cfg["data"]["T"], k=cfg["data"]["k"],
                        stride=cfg["data"]["stride"],
                        cache_root=Path(cfg["data"]["cache_root"]))
    ds = OXEWindowDataset(records, wcfg)
    n = len(ds)
    if n == 0:
        raise RuntimeError("Dataset empty")
    g = torch.Generator().manual_seed(cfg["data"]["seed"])
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(n * cfg["data"]["val_frac"]))
    return Subset(ds, perm[n_val:]), Subset(ds, perm[:n_val])


def batch_to_device(batch, device):
    s = batch["s_in"].to(device, non_blocking=True)
    c = batch["c"].to(device, non_blocking=True)
    rgb_tgt_p = batch["rgb_tgt"].to(device, non_blocking=True).permute(0, 1, 4, 2, 3)
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
    ap.add_argument("--v3_ckpt", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--reset_optim", action="store_true")
    ap.add_argument("--print_every", type=int, default=0)
    ap.add_argument("--max_records", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.cfg.read_text())
    rank, world, local = setup_ddp()
    device = torch.device(f"cuda:{local}")

    tr_ds, val_ds = build_datasets(cfg, max_records=args.max_records)
    bs = cfg["train"]["batch_size_per_gpu"]; nw = cfg["train"]["num_workers"]
    if world > 1:
        tr_s = DistributedSampler(tr_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        tr_loader = DataLoader(tr_ds, batch_size=bs, sampler=tr_s, num_workers=nw, pin_memory=True, drop_last=True)
        val_s = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=bs, sampler=val_s, num_workers=nw, pin_memory=True)
    else:
        tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    model = build_model(cfg).to(device)
    res, n_loaded = load_v3_nonpixel(model, args.v3_ckpt)
    # Freeze everything except pixel decoder
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("pixel.")
    if rank == 0:
        n_p_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_p_total = sum(p.numel() for p in model.parameters())
        print(f"[rank0] v3.5 total={n_p_total/1e6:.1f}M trainable(pixel only)={n_p_trainable/1e6:.1f}M; "
              f"v3 keys loaded={n_loaded}; missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}; "
              f"train_windows={len(tr_ds)} val_windows={len(val_ds)}")

    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    if world > 1:
        model = DDP(model, device_ids=[local], find_unused_parameters=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg["train"]["lr"],
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
        if args.reset_optim:
            if rank == 0:
                print(f"[rank0] resumed weights only from {args.resume} (optim RESET)")
        else:
            opt.load_state_dict(sd["opt"]); sched.load_state_dict(sd["sched"])
            start_epoch = sd["epoch"] + 1; step = sd["step"]; best_val = sd.get("best_val", best_val)
            if rank == 0:
                print(f"[rank0] resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        if world > 1:
            tr_s.set_epoch(epoch)
        model.train()
        for batch in tr_loader:
            s, c, tgt = batch_to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = (model.module if world > 1 else model)(s, c, pixel=True, bridging=False)
                losses = compute_losses(out, tgt, weights, lpips_fn)
            loss = losses["L_total"]
            if not torch.isfinite(loss):
                if rank == 0:
                    print(f"[rank0] WARN skip non-finite loss at step {step}", flush=True)
                opt.zero_grad(set_to_none=True); sched.step(); step += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable, cfg["train"]["grad_clip"])
            if not torch.isfinite(gn):
                if rank == 0:
                    print(f"[rank0] WARN skip non-finite grad_norm at step {step}", flush=True)
                opt.zero_grad(set_to_none=True); sched.step(); step += 1
                continue
            opt.step(); sched.step()
            if rank == 0 and step % cfg["train"]["log_every"] == 0:
                for k_, v in losses.items():
                    tb.add_scalar(f"train/{k_}", float(v.detach().float()), step)
                tb.add_scalar("lr", sched.get_last_lr()[0], step)
            if rank == 0 and args.print_every and step % args.print_every == 0:
                rgb_l1 = float(losses.get("L_rgb_l1", torch.tensor(0.)).detach().float())
                lpv = float(losses.get("L_rgb_lpips", torch.tensor(0.)).detach().float())
                print(f"[rank0] step {step} (ep {epoch}) L_total={float(loss.detach().float()):.4f} "
                      f"rgb_L1={rgb_l1:.4f} lpips={lpv:.4f} lr={sched.get_last_lr()[0]:.2e}", flush=True)
            step += 1
        # val
        model.eval()
        agg = {}; nb = 0
        with torch.no_grad():
            for batch in val_loader:
                s, c, tgt = batch_to_device(batch, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(s, c, pixel=True, bridging=False)
                    losses = compute_losses(out, tgt, weights, lpips_fn)
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
            rgb_l1 = agg.get("L_rgb_l1", 0.) / max(1, nb)
            lpv = agg.get("L_rgb_lpips", 0.) / max(1, nb)
            print(f"[rank0] epoch {epoch}: val_total {val_total:.4f} rgb_L1 {rgb_l1:.4f} lpips {lpv:.4f} (best {best_val:.4f})")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
