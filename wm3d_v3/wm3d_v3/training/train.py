"""v3 joint training. DDP-aware, bf16, gradient checkpointing optional."""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
import lpips
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.splits import episode_split, load_clip_split_file, random_window_indices
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
        enable_geom_extra=cfg["model"].get("enable_geom_extra", True),
        pixel_hidden=cfg["model"]["pixel_hidden"],
        pixel_n_res=cfg["model"]["pixel_n_res"],
        enable_pixel=cfg["model"].get("enable_pixel", True),
        enable_context_pixel=cfg["model"].get("enable_context_pixel", False),
        context_pixel_hidden=cfg["model"].get("context_pixel_hidden", 384),
        context_pixel_action_dim=cfg["model"].get("context_pixel_action_dim", 7),
        context_pixel_task_dim=cfg["model"].get("context_pixel_task_dim"),
        context_pixel_residual_scale=cfg["model"].get("context_pixel_residual_scale", 0.75),
        context_pixel_use_action=cfg["model"].get("context_pixel_use_action", True),
        context_pixel_use_task=cfg["model"].get("context_pixel_use_task", True),
        context_pixel_predict_motion=cfg["model"].get("context_pixel_predict_motion", False),
        context_pixel_motion_blend_gain=cfg["model"].get("context_pixel_motion_blend_gain", 0.0),
        enable_control_head=cfg["model"].get("enable_control_head", False),
        control_hidden=cfg["model"].get("control_hidden", 128),
        control_output_size=cfg["model"].get("control_output_size", 256),
        control_fuse_size=cfg["model"].get("control_fuse_size", 64),
        control_refine_channels=cfg["model"].get("control_refine_channels", 16),
        control_use_refine=cfg["model"].get("control_use_refine", True),
        control_action_dim=cfg["model"].get("control_action_dim", 7),
        control_task_dim=cfg["model"].get("control_task_dim"),
        control_use_context=cfg["model"].get("control_use_context", True),
        control_use_action=cfg["model"].get("control_use_action", True),
        control_use_task=cfg["model"].get("control_use_task", True),
        enable_progress_head=cfg["model"].get("enable_progress_head", False),
        progress_hidden=cfg["model"].get("progress_hidden", 256),
        progress_layers=cfg["model"].get("progress_layers", 2),
        progress_heads=cfg["model"].get("progress_heads", 4),
        progress_action_dim=cfg["model"].get("progress_action_dim", 7),
        progress_task_dim=cfg["model"].get("progress_task_dim"),
        progress_max_horizon=cfg["model"].get("progress_max_horizon", 32),
        progress_use_action=cfg["model"].get("progress_use_action", True),
        progress_use_task=cfg["model"].get("progress_use_task", True),
        enable_bridging=cfg["model"].get("enable_bridging", True),
    )
    return JointWorldModel(jc)


def _data_split_cfg(data_cfg: dict) -> dict:
    split_cfg = data_cfg.get("split") or {}
    if isinstance(split_cfg, (str, Path)):
        return {"file": str(split_cfg)}
    if not isinstance(split_cfg, dict):
        raise ValueError("data.split must be a mapping or split-file path")
    return dict(split_cfg)


def _split_value(data_cfg: dict, split_cfg: dict, key: str, default=None):
    return split_cfg.get(key, data_cfg.get(key, default))


def _explicit_clip_ids(data_cfg: dict, split_cfg: dict) -> tuple[list[str] | None, list[str] | None]:
    train_ids = split_cfg.get("train_clip_ids")
    val_ids = split_cfg.get("val_clip_ids")
    split_file = data_cfg.get("split_file") or split_cfg.get("file") or split_cfg.get("path")
    if split_file:
        file_split = load_clip_split_file(split_file)
        train_ids = train_ids if train_ids is not None else file_split["train_clip_ids"]
        val_ids = val_ids if val_ids is not None else file_split["val_clip_ids"]
    return train_ids, val_ids


def _window_config(data_cfg: dict) -> WindowConfig:
    return WindowConfig(T=data_cfg["T"], k=data_cfg["k"],
                        stride=data_cfg["stride"],
                        cache_root=Path(data_cfg["cache_root"]),
                        tokens_subdir=data_cfg.get("tokens_subdir", "vggt_pooled"),
                        action_stats=Path(data_cfg["action_stats"])
                        if data_cfg.get("action_stats") else None)


def build_datasets(cfg: dict, overfit_ids=None):
    records = read_manifest(cfg["data"]["manifest"])
    if overfit_ids:
        records = [r for r in records if r.clip_id in overfit_ids]
        if not records:
            raise RuntimeError(f"no records matched overfit ids: {overfit_ids}")
    data_cfg = cfg["data"]
    split_cfg = _data_split_cfg(data_cfg)
    has_episode_split_keys = (
        data_cfg.get("split_file")
        or any(k in split_cfg for k in ("file", "path", "train_clip_ids", "val_clip_ids", "heldout_dataset"))
    )
    mode = split_cfg.get("mode", "episode" if has_episode_split_keys else "random_window")
    val_frac = float(_split_value(data_cfg, split_cfg, "val_frac", 0.0))
    seed = int(_split_value(data_cfg, split_cfg, "seed", 0))
    wcfg = _window_config(data_cfg)

    if mode == "episode":
        train_ids, val_ids = _explicit_clip_ids(data_cfg, split_cfg)
        clip_split = episode_split(
            records,
            val_frac=val_frac,
            seed=seed,
            train_clip_ids=train_ids,
            val_clip_ids=val_ids,
            heldout_dataset=_split_value(data_cfg, split_cfg, "heldout_dataset"),
        )
        train_records = [r for r in records if r.clip_id in clip_split.train_clip_ids]
        val_records = [r for r in records if r.clip_id in clip_split.val_clip_ids]
        tr_ds = OXEWindowDataset(train_records, wcfg)
        val_ds = OXEWindowDataset(val_records, wcfg)
        if len(tr_ds) == 0:
            raise RuntimeError("episode train split empty — caches missing?")
        if len(val_ds) == 0:
            raise RuntimeError("episode val split empty — caches missing?")
        return tr_ds, val_ds

    if mode != "random_window":
        raise ValueError(f"unsupported data.split.mode: {mode}")

    ds = OXEWindowDataset(records, wcfg)
    n = len(ds)
    if n == 0:
        raise RuntimeError("OXEWindowDataset empty — caches missing?")
    train_idx, val_idx = random_window_indices(n, val_frac=val_frac, seed=seed)
    return Subset(ds, train_idx), Subset(ds, val_idx)


def batch_to_device(batch: dict, device: torch.device, k: int) -> tuple:
    s = batch["s_in"].to(device, non_blocking=True)
    c = batch["c"].to(device, non_blocking=True)
    context_rgb = batch["rgb_in"][:, -1].to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
    rgb_tgt_p = batch["rgb_tgt"].to(device, non_blocking=True).permute(0, 1, 4, 2, 3)
    # rgb_tgt: [B, k, 256, 256, 3]; permute -> [B, k, 3, 256, 256]
    tgt = {
        "s_tgt": batch["s_tgt"].to(device, non_blocking=True),
        "depth_tgt": batch["depth_tgt"].to(device, non_blocking=True),
        "rgb_tgt_p": rgb_tgt_p,
        "rgb_ref_p": context_rgb,
    }
    action_tgt = batch["action_tgt"].to(device, non_blocking=True)
    action_tgt_norm = batch["action_tgt_norm"].to(device, non_blocking=True)
    tgt["action_tgt"] = action_tgt
    tgt["action_tgt_norm"] = action_tgt_norm
    action_cond = make_action_condition(action_tgt, action_tgt_norm)
    return s, c, action_cond, context_rgb, tgt


def load_compatible_state_dict(model: torch.nn.Module, state: dict, strict: bool):
    if strict:
        return model.load_state_dict(state, strict=True)
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key in current and current[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    result = model.load_state_dict(compatible, strict=False)
    return SimpleNamespace(
        missing_keys=result.missing_keys,
        unexpected_keys=result.unexpected_keys,
        skipped_keys=skipped,
    )


def load_action_stats_if_available(model, cfg: dict, rank: int, device: torch.device) -> None:
    stats_path = cfg["data"].get("action_stats")
    if not stats_path:
        return
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"action_stats not found: {path}")
    stats = np.load(path)
    target = model.module if isinstance(model, DDP) else model
    mean = torch.as_tensor(stats["mean"][:6], device=device)
    std = torch.as_tensor(stats["std"][:6], device=device)
    target.load_action_stats(mean, std)
    if rank == 0:
        print(f"[rank0] loaded action_stats from {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--disable_pixel_until", type=int, default=0,
                    help="train first N epochs without L_rgb (stage 1)")
    ap.add_argument("--reset_optim", action="store_true",
                    help="on --resume, load model weights only; recreate fresh optimizer/scheduler/step")
    ap.add_argument("--print_every", type=int, default=0,
                    help="print stdout step log every N steps (0=off)")
    ap.add_argument("--strict_resume", action="store_true",
                    help="strict state_dict load on resume (default: allow mismatches)")
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
        model = DDP(
            model,
            device_ids=[local],
            find_unused_parameters=cfg["train"].get("find_unused_parameters", False),
        )

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
        target = (model.module if world > 1 else model)
        load_res = load_compatible_state_dict(target, sd["model"], strict=args.strict_resume)
        if args.reset_optim:
            if rank == 0:
                miss = len(getattr(load_res, "missing_keys", []) or [])
                un = len(getattr(load_res, "unexpected_keys", []) or [])
                skip = len(getattr(load_res, "skipped_keys", []) or [])
                print(f"[rank0] resumed weights only from {args.resume} (optim RESET) — "
                      f"missing={miss} unexpected={un} skipped={skip}")
        else:
            opt.load_state_dict(sd["opt"]); sched.load_state_dict(sd["sched"])
            start_epoch = sd["epoch"] + 1; step = sd["step"]; best_val = sd.get("best_val", best_val)
            if rank == 0:
                print(f"[rank0] resumed from {args.resume} at epoch {start_epoch}")
    load_action_stats_if_available(model, cfg, rank, device)

    k = cfg["data"]["k"]
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        if world > 1:
            tr_s.set_epoch(epoch)
        do_pixel = epoch >= args.disable_pixel_until
        model.train()
        for batch in tr_loader:
            s, c, action_cond, context_rgb, tgt = batch_to_device(batch, device, k)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(s, c, action_cond=action_cond, context_rgb=context_rgb,
                            pixel=do_pixel, bridging=False) \
                    if isinstance(model, DDP) else model(s, c, action_cond=action_cond,
                                                         context_rgb=context_rgb,
                                                         pixel=do_pixel, bridging=False)
                losses = compute_losses(out, tgt, weights, lpips_fn if do_pixel else None)
            loss = losses["L_total"]
            if not torch.isfinite(loss):
                if rank == 0:
                    print(f"[rank0] WARN skip non-finite loss at step {step}", flush=True)
                opt.zero_grad(set_to_none=True); sched.step(); step += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            if not torch.isfinite(gn):
                if rank == 0:
                    print(f"[rank0] WARN skip non-finite grad_norm at step {step}", flush=True)
                opt.zero_grad(set_to_none=True); sched.step(); step += 1
                continue
            opt.step(); sched.step()
            if rank == 0 and args.print_every and step % args.print_every == 0:
                rgb_l1 = float(losses.get("L_rgb_l1", torch.tensor(0.)).detach().float())
                lpv = float(losses.get("L_rgb_lpips", torch.tensor(0.)).detach().float())
                print(f"[rank0] step {step} (ep {epoch}) L_total={float(loss.detach().float()):.4f} "
                      f"rgb_L1={rgb_l1:.4f} lpips={lpv:.4f} lr={sched.get_last_lr()[0]:.2e}", flush=True)
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
                s, c, action_cond, context_rgb, tgt = batch_to_device(batch, device, k)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(s, c, action_cond=action_cond, context_rgb=context_rgb,
                                pixel=do_pixel, bridging=False)
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
