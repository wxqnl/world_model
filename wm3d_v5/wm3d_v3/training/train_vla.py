"""VLA stage-A fine-tune: warm-start from v3 best.pt, phased freezing."""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.window_dataset import OXEWindowDataset, WindowConfig
from wm3d_v3.losses import VLALossWeights, compute_losses_vla
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
        enable_aux_idm=cfg["model"].get("enable_aux_idm", True),
        aux_idm_hidden=cfg["model"].get("aux_idm_hidden", 1024),
        aux_idm_layers=cfg["model"].get("aux_idm_layers", 3),
    )
    return JointWorldModel(jc)


def build_datasets(cfg: dict):
    records = read_manifest(cfg["data"]["manifest"])
    wcfg = WindowConfig(T=cfg["data"]["T"], k=cfg["data"]["k"],
                        stride=cfg["data"]["stride"],
                        cache_root=Path(cfg["data"]["cache_root"]),
                        action_stats=Path(cfg["data"]["action_stats"]))
    ds = OXEWindowDataset(records, wcfg)
    n = len(ds)
    g = torch.Generator().manual_seed(cfg["data"]["seed"])
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(n * cfg["data"]["val_frac"]))
    return Subset(ds, perm[n_val:]), Subset(ds, perm[:n_val])


def apply_freeze(model: torch.nn.Module, prefixes: list[str]) -> tuple[int, int]:
    """Set requires_grad on each param based on name prefix match.
    Returns (n_frozen_params, n_trainable_params).
    """
    n_frozen = 0
    n_train = 0
    for name, p in model.named_parameters():
        if any(name.startswith(pfx) for pfx in prefixes):
            p.requires_grad = False
            n_frozen += p.numel()
        else:
            p.requires_grad = True
            n_train += p.numel()
    return n_frozen, n_train


def load_warm_start(model: JointWorldModel, ckpt_path: Path, rank: int) -> None:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd["model"]
    drop = [k for k in state if k.startswith("action_proj.")]
    for k in drop:
        del state[k]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if rank == 0:
        print(f"[warm-start] loaded {ckpt_path}; dropped {len(drop)} old action_proj keys")
        print(f"  missing (fresh init): {sorted(missing)[:6]}{'...' if len(missing)>6 else ''}")
        print(f"  unexpected (ignored): {sorted(unexpected)[:6]}{'...' if len(unexpected)>6 else ''}")


def batch_to_device(batch: dict, device: torch.device) -> tuple:
    s = batch["s_in"].to(device, non_blocking=True)
    c = batch["c"].to(device, non_blocking=True)
    tgt = {
        "action_tgt": batch["action_tgt"].to(device, non_blocking=True),
        "action_tgt_norm": batch["action_tgt_norm"].to(device, non_blocking=True),
    }
    return s, c, tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--max_batches_per_epoch", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.cfg.read_text())
    rank, world, local = setup_ddp()
    device = torch.device(f"cuda:{local}")

    tr_ds, val_ds = build_datasets(cfg)
    bs = cfg["train"]["batch_size_per_gpu"]
    nw = cfg["train"]["num_workers"]
    if world > 1:
        tr_s = DistributedSampler(tr_ds, num_replicas=world, rank=rank,
                                   shuffle=True, drop_last=True)
        tr_loader = DataLoader(tr_ds, batch_size=bs, sampler=tr_s,
                                num_workers=nw, pin_memory=True, drop_last=True)
        val_s = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=bs, sampler=val_s,
                                 num_workers=nw, pin_memory=True)
    else:
        tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True,
                                num_workers=nw, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                 num_workers=nw, pin_memory=True)

    model = build_model(cfg).to(device)

    stats = np.load(cfg["data"]["action_stats"])
    mean = torch.from_numpy(stats["mean"][:6].astype(np.float32)).to(device)
    std = torch.from_numpy(stats["std"][:6].astype(np.float32)).to(device)
    pos_rate = float(stats["pos_rate"][0])
    model.load_action_stats(mean, std)
    if pos_rate > 0:
        cfg["loss"]["grip_pos_weight"] = float(min(5.0, max(0.5, (1.0 - pos_rate) / pos_rate)))
    if rank == 0:
        print(f"action_stats: pos_rate={pos_rate:.4f} -> grip_pos_weight={cfg['loss']['grip_pos_weight']:.3f}")

    if args.resume is None and "warm_start" in cfg:
        load_warm_start(model, Path(cfg["warm_start"]), rank)

    cut = int(cfg["train"]["freeze_phase1_until_epoch"])
    phase1_pfx = cfg["train"]["freeze_phase1_prefixes"]
    phase2_pfx = cfg["train"]["freeze_phase2_prefixes"]

    start_epoch = 0; step = 0; best_val = float("inf")

    # Apply initial freeze based on starting epoch
    init_pfx = phase1_pfx if start_epoch < cut else phase2_pfx
    n_frozen, n_train = apply_freeze(model, init_pfx)
    if rank == 0:
        print(f"[freeze] phase={'A.1' if start_epoch<cut else 'A.2'} "
              f"frozen={n_frozen/1e6:.1f}M trainable={n_train/1e6:.1f}M")

    if world > 1:
        model = DDP(model, device_ids=[local], find_unused_parameters=True)

    weights_cfg = cfg["loss"]
    weights = VLALossWeights(**weights_cfg)

    def make_opt_sched(stp: int):
        target = model.module if world > 1 else model
        lr = cfg["train"]["lr"]
        op = torch.optim.AdamW([p for p in target.parameters() if p.requires_grad],
                                lr=lr,
                                weight_decay=cfg["train"]["weight_decay"],
                                betas=(0.9, 0.95))
        # LambdaLR with last_epoch>=0 requires initial_lr in each param group.
        for pg in op.param_groups:
            pg["initial_lr"] = lr
        warmup = int(cfg["train"]["warmup_steps"])
        total_steps = max(1, len(tr_loader) * cfg["train"]["epochs"])
        def lr_lambda(s):
            if s < warmup:
                return (s + 1) / warmup
            prog = (s - warmup) / max(1, total_steps - warmup)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))
        sch = torch.optim.lr_scheduler.LambdaLR(op, lr_lambda, last_epoch=stp - 1)
        return op, sch

    opt, sched = make_opt_sched(step)

    out_root = Path(cfg["out"]["root"])
    ckpt_dir = out_root / cfg["out"]["ckpt_dir"]
    if rank == 0:
        (out_root / cfg["out"]["tb_dir"]).mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tb = SummaryWriter(out_root / cfg["out"]["tb_dir"])

    if args.resume is not None and args.resume.exists():
        sd = torch.load(args.resume, map_location=device, weights_only=False)
        (model.module if world > 1 else model).load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"]); sched.load_state_dict(sd["sched"])
        start_epoch = sd["epoch"] + 1; step = sd["step"]; best_val = sd.get("best_val", best_val)
        if rank == 0:
            print(f"[resume] from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        prev_phase = "A.1" if (epoch - 1) < cut else "A.2"
        this_phase = "A.1" if epoch < cut else "A.2"
        if this_phase != prev_phase and epoch != start_epoch:
            target = model.module if world > 1 else model
            n_frozen, n_train = apply_freeze(target, phase2_pfx)
            opt, sched = make_opt_sched(step)
            if rank == 0:
                print(f"[freeze] phase=A.2 frozen={n_frozen/1e6:.1f}M trainable={n_train/1e6:.1f}M")
        if world > 1:
            tr_s.set_epoch(epoch)
        model.train()
        bi = 0
        for batch in tr_loader:
            if args.max_batches_per_epoch and bi >= args.max_batches_per_epoch:
                break
            s, c, tgt = batch_to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(s, c, pixel=False, bridging=False, aux_idm=True)
                losses = compute_losses_vla(out, tgt, weights)
            opt.zero_grad(set_to_none=True)
            losses["L_total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg["train"]["grad_clip"])
            opt.step(); sched.step()
            if rank == 0 and step % cfg["train"]["log_every"] == 0:
                for kk, v in losses.items():
                    tb.add_scalar(f"train/{kk}", float(v.detach()), step)
                tb.add_scalar("lr", sched.get_last_lr()[0], step)
                tb.add_scalar("grad_norm", float(grad_norm), step)
                print(f"  ep{epoch} step{step} L_total={float(losses['L_total'].detach()):.4f} "
                      f"L_pose={float(losses['L_pose']):.4f} L_grip={float(losses['L_grip']):.4f} "
                      f"L_aux_pose={float(losses['L_aux_pose']):.4f} gn={float(grad_norm):.2f}")
            step += 1
            bi += 1
        model.eval()
        agg = {}; nb = 0
        with torch.no_grad():
            for batch in val_loader:
                s, c, tgt = batch_to_device(batch, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(s, c, pixel=False, bridging=False, aux_idm=True)
                    losses = compute_losses_vla(out, tgt, weights)
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
            print(f"[rank0] epoch {epoch}: val_total {val_total:.4f} (best {best_val:.4f}, phase={this_phase})")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
