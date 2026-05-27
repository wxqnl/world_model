"""v4 training: train DiffusionHead on top of frozen v3 backbone.

Loads:
- v3 best.pt to initialize dual/geom/action (frozen)
- SD-1.5 VAE (frozen)
Trains:
- DiffusionHead only (eps-prediction MSE in latent space)
DDP-aware, bf16, cosine LR, 4×H100.
"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path

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
from wm3d_v3.models.state_stream import StateConfig
from wm3d_v3.models.action_stream import ActionConfig
from wm3d_v3.models.dual_stream import DualConfig

from wm3d_v4.models.diffusion_head import DiffusionHeadConfig
from wm3d_v4.models.joint_v4 import JointV4, JointV4Config
from wm3d_v4.models.vae_wrapper import VAEWrapper
from wm3d_v4.schedulers import CosineSchedule


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


def build_model(cfg: dict) -> JointV4:
    sc = StateConfig(**cfg["model"]["state"])
    ac = ActionConfig(**cfg["model"]["action"])
    dc = DualConfig(state=sc, action=ac,
                    xattn_layers_state=tuple(cfg["model"]["xattn_layers_state"]),
                    xattn_n_heads=cfg["model"]["xattn_n_heads"])
    diff_cfg = DiffusionHeadConfig(
        latent_channels=4, latent_size=32, patch_size=cfg["diff"].get("patch_size", 2),
        hidden=cfg["diff"]["hidden"], n_layers=cfg["diff"]["n_layers"],
        n_heads=cfg["diff"]["n_heads"], mlp_ratio=cfg["diff"].get("mlp_ratio", 4.0),
        cond_dim=cfg["model"]["state"]["D"], cond_seq_len=cfg["model"]["state"]["P"],
        timestep_dim=cfg["diff"].get("timestep_dim", 256),
        dropout=cfg["diff"].get("dropout", 0.0),
    )
    jc = JointV4Config(
        dual=dc, diff=diff_cfg,
        action_proj_hidden=cfg["model"]["action_proj_hidden"],
        action_proj_layers=cfg["model"]["action_proj_layers"],
        geom_hidden=cfg["model"]["geom_hidden"],
        freeze_v3=cfg["train"].get("freeze_v3", True),
    )
    return JointV4(jc)


def load_v3_into_v4(v4: JointV4, v3_ckpt_path: Path) -> None:
    """Copy v3 dual / action_proj / geom params from a v3 best.pt checkpoint."""
    sd = torch.load(v3_ckpt_path, map_location="cpu", weights_only=False)
    v3_state = sd["model"] if "model" in sd else sd
    own_keys = set(v4.state_dict().keys())
    loaded = {}
    for k, v in v3_state.items():
        if k.startswith("dual.") or k.startswith("action_proj.") or k.startswith("geom."):
            if k in own_keys:
                loaded[k] = v
    res = v4.load_state_dict(loaded, strict=False)
    return res, len(loaded)


def build_datasets(cfg: dict, max_records: int = 0):
    records = read_manifest(cfg["data"]["manifest"])
    if max_records and len(records) > max_records:
        records = records[:max_records]
    wcfg = WindowConfig(T=cfg["data"]["T"], k=cfg["data"]["k"],
                        stride=cfg["data"]["stride"],
                        cache_root=Path(cfg["data"]["cache_root"]))
    ds = OXEWindowDataset(records, wcfg)
    n = len(ds)
    if n == 0:
        raise RuntimeError("Dataset empty — caches missing?")
    g = torch.Generator().manual_seed(cfg["data"]["seed"])
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(n * cfg["data"]["val_frac"]))
    return Subset(ds, perm[n_val:]), Subset(ds, perm[:n_val])


def to_device(batch, device, k):
    s = batch["s_in"].to(device, non_blocking=True)
    c = batch["c"].to(device, non_blocking=True)
    rgb_tgt = batch["rgb_tgt"].to(device, non_blocking=True).permute(0, 1, 4, 2, 3)  # [B,k,3,256,256]
    return s, c, rgb_tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--v3_ckpt", type=Path, required=True,
                    help="v3 best.pt to bootstrap dual/geom/action")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--max_records", type=int, default=0,
                    help="cap manifest records (0=all), useful for smoke")
    ap.add_argument("--print_every", type=int, default=0,
                    help="print stdout step log every N steps (0=off)")
    ap.add_argument("--reset_optim", action="store_true",
                    help="on --resume, load model weights only, recreate fresh optimizer/scheduler/step state")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.cfg.read_text())
    rank, world, local = setup_ddp()
    device = torch.device(f"cuda:{local}")

    tr_ds, val_ds = build_datasets(cfg, max_records=args.max_records)
    bs = cfg["train"]["batch_size_per_gpu"]
    nw = cfg["train"]["num_workers"]
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
    # Load v3 weights
    res, n_loaded = load_v3_into_v4(model, args.v3_ckpt)
    if rank == 0:
        n_p = model.num_trainable_params()
        print(f"[rank0] JointV4: trainable={n_p/1e6:.1f}M; v3 keys loaded={n_loaded}; "
              f"missing(after-load)={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}; "
              f"train_windows={len(tr_ds)} val_windows={len(val_ds)}")

    vae = VAEWrapper(pretrained=cfg["model"].get("vae_pretrained", "stabilityai/sd-vae-ft-mse")).to(device).eval()
    schedule = CosineSchedule(num_train_timesteps=cfg["diff"].get("num_train_timesteps", 1000), device=device)

    if world > 1:
        model = DDP(model, device_ids=[local], find_unused_parameters=False)

    # Only diff head params have requires_grad=True (others are frozen via JointV4)
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
                print(f"[rank0] resumed weights only from {args.resume} (optim/sched/step RESET)")
        else:
            opt.load_state_dict(sd["opt"]); sched.load_state_dict(sd["sched"])
            start_epoch = sd["epoch"] + 1; step = sd["step"]; best_val = sd.get("best_val", best_val)
            if rank == 0:
                print(f"[rank0] resumed from {args.resume} at epoch {start_epoch}")

    k = cfg["data"]["k"]
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        if world > 1:
            tr_s.set_epoch(epoch)
        model.train()
        for batch in tr_loader:
            s, c, rgb_tgt = to_device(batch, device, k)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = (model.module if world > 1 else model).forward_train(s, c, rgb_tgt, vae, schedule)
            eps_pred = out["eps_pred"].float()
            eps_target = out["eps_target"].float()
            loss = F.mse_loss(eps_pred, eps_target)
            if not torch.isfinite(loss):
                if rank == 0:
                    print(f"[rank0] WARN skip non-finite loss at step {step} (loss={loss.item()})", flush=True)
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable, cfg["train"]["grad_clip"])
            if not torch.isfinite(gn):
                if rank == 0:
                    print(f"[rank0] WARN skip non-finite grad_norm at step {step} (gn={float(gn)})", flush=True)
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                continue
            opt.step(); sched.step()
            if rank == 0 and step % cfg["train"]["log_every"] == 0:
                tb.add_scalar("train/eps_mse", loss.item(), step)
                tb.add_scalar("lr", sched.get_last_lr()[0], step)
            if rank == 0 and args.print_every and step % args.print_every == 0:
                print(f"[rank0] step {step} (ep {epoch}) eps_mse {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
            step += 1
        # Val
        model.eval()
        agg = 0.0; nb = 0
        with torch.no_grad():
            for batch in val_loader:
                s, c, rgb_tgt = to_device(batch, device, k)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = (model.module if world > 1 else model).forward_train(s, c, rgb_tgt, vae, schedule)
                eps_pred = out["eps_pred"].float()
                eps_target = out["eps_target"].float()
                v = F.mse_loss(eps_pred, eps_target).item()
                agg += v; nb += 1
        if world > 1:
            t = torch.tensor([agg, float(nb)], device=device)
            dist.all_reduce(t)
            agg = float(t[0]); nb = int(t[1])
        if rank == 0:
            val_loss = agg / max(1, nb)
            tb.add_scalar("val/eps_mse", val_loss, step)
            ckpt = {"model": (model.module if world > 1 else model).state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "step": step, "val": val_loss,
                    "best_val": best_val, "cfg": cfg}
            if (epoch + 1) % cfg["train"]["ckpt_every_epochs"] == 0:
                torch.save(ckpt, ckpt_dir / f"epoch_{epoch:03d}.pt")
            if val_loss < best_val:
                best_val = val_loss
                ckpt["best_val"] = best_val
                torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"[rank0] epoch {epoch}: val_eps_mse {val_loss:.4f} (best {best_val:.4f})")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
