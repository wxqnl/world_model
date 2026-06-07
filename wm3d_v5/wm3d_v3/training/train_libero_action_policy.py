"""Train the direct WM3D action policy head on LIBERO expert token cache."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler

from wm3d_v3.eval.run_eval import build_model
from wm3d_v3.training.train_libero_success_p0 import LiberoExpertCacheDataset, _load_action_stats


class InMemoryDataset(Dataset):
    def __init__(self, base: Dataset) -> None:
        self.samples = [base[idx] for idx in range(len(base))]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def _setup_distributed() -> tuple[int, int, int, bool]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0, False
    backend = os.environ.get("WM3D_DDP_BACKEND", "nccl")
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world, local_rank, True


def _all_reduce_gradients(model: torch.nn.Module, world: int) -> None:
    if world <= 1:
        return
    backend = dist.get_backend()
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if backend == "gloo" and grad.is_cuda:
            cpu_grad = grad.float().cpu()
            dist.all_reduce(cpu_grad, op=dist.ReduceOp.SUM)
            cpu_grad.div_(world)
            grad.copy_(cpu_grad.to(device=grad.device, dtype=grad.dtype))
        else:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            grad.div_(world)


def _broadcast_model_state(model: torch.nn.Module, world: int) -> None:
    if world <= 1:
        return
    for tensor in model.state_dict().values():
        dist.broadcast(tensor, src=0)


def _freeze(model: torch.nn.Module, prefixes: list[str]) -> tuple[int, int]:
    trainable_prefixes = tuple(prefixes)
    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        enabled = name.startswith(trainable_prefixes)
        param.requires_grad = enabled
        if enabled:
            trainable += param.numel()
        else:
            frozen += param.numel()
    return frozen, trainable


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {
        "s": batch["s_in"].to(device, non_blocking=True),
        "c": batch["c"].to(device, non_blocking=True),
        "context_rgb": batch["context_rgb"].to(device, non_blocking=True),
        "action_tgt": batch["action_tgt"].to(device, non_blocking=True),
        "action_tgt_norm": batch["action_tgt_norm"].to(device, non_blocking=True),
        "proposer_weight": batch["proposer_weight"].to(device, non_blocking=True),
    }
    if "lowdim_state" in batch:
        out["lowdim_state"] = batch["lowdim_state"].to(device, non_blocking=True)
    if "object_state" in batch:
        out["object_state"] = batch["object_state"].to(device, non_blocking=True)
    if "plan_state" in batch:
        out["plan_state"] = batch["plan_state"].to(device, non_blocking=True)
    if "action_history" in batch:
        out["action_history"] = batch["action_history"].to(device, non_blocking=True)
    if "progress_tgt" in batch:
        out["progress_state"] = batch["progress_tgt"].to(device, non_blocking=True).float().reshape(-1, 1)
    return out


def _weighted_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(device=values.device, dtype=values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1e-6)


def _policy_forward(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    target = model.module if hasattr(model, "module") else model
    if getattr(target, "action_policy", None) is None:
        raise RuntimeError("enable_action_policy is required for direct policy training")
    return target.action_policy(
        batch["s"],
        task_emb=batch["c"],
        lowdim_state=batch.get("lowdim_state"),
        object_state=batch.get("object_state"),
        plan_state=batch.get("plan_state"),
        action_history=batch.get("action_history"),
        progress_state=batch.get("progress_state"),
    )


def _load_init_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> tuple[list[str], list[str], list[str]]:
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key in current and current[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return list(missing), list(unexpected), skipped


def _policy_losses(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict) -> dict[str, torch.Tensor]:
    if "policy_action_cond" not in out:
        raise RuntimeError("model output has no policy_action_cond; enable_action_policy is required")
    pose_pred = out["policy_pose_norm"].float()
    pose_tgt = batch["action_tgt_norm"].float()
    weight = batch["proposer_weight"].float()
    if float(weight.sum().detach().cpu()) <= 0:
        weight = torch.ones_like(weight)

    pose_err = F.smooth_l1_loss(
        pose_pred,
        pose_tgt,
        beta=float(cfg.get("huber_delta", 1.0)),
        reduction="none",
    ).mean(dim=(1, 2))
    first_pose_err = F.smooth_l1_loss(
        pose_pred[:, 0],
        pose_tgt[:, 0],
        beta=float(cfg.get("huber_delta", 1.0)),
        reduction="none",
    ).mean(dim=1)
    raw_pose_pred = out["policy_action_cond"].float()[..., :6]
    raw_delta = raw_pose_pred[:, 1:] - raw_pose_pred[:, :-1]
    tgt_delta = pose_tgt[:, 1:] - pose_tgt[:, :-1]
    delta_err = F.smooth_l1_loss(
        raw_delta,
        tgt_delta,
        beta=float(cfg.get("huber_delta", 1.0)),
        reduction="none",
    ).mean(dim=(1, 2))

    grip_tgt = (batch["action_tgt"][..., 6] > 0.5).float()
    grip_bce = F.binary_cross_entropy_with_logits(out["policy_gripper_logit"].float(), grip_tgt, reduction="none")
    transition_weight = float(cfg.get("grip_transition_weight", 0.0))
    if transition_weight > 0:
        if "action_history" in batch and batch["action_history"].shape[1] > 0:
            prev_grip = (batch["action_history"][:, -1, 6].float() > 0.5).float()
        else:
            prev_grip = grip_tgt[:, 0]
        prev_tgt = torch.cat([prev_grip[:, None], grip_tgt[:, :-1]], dim=1)
        grip_transition = (grip_tgt != prev_tgt).float()
        grip_step_weight = 1.0 + transition_weight * grip_transition
        grip_err = (grip_bce * grip_step_weight).sum(dim=1) / grip_step_weight.sum(dim=1).clamp_min(1e-6)
    else:
        grip_transition = torch.zeros_like(grip_tgt)
        grip_err = grip_bce.mean(dim=1)
    first_grip_err = F.binary_cross_entropy_with_logits(out["policy_gripper_logit"].float()[:, 0], grip_tgt[:, 0], reduction="none")

    L_pose = _weighted_mean(pose_err, weight)
    L_first_pose = _weighted_mean(first_pose_err, weight)
    L_delta = _weighted_mean(delta_err, weight)
    L_grip = _weighted_mean(grip_err, weight)
    L_first_grip = _weighted_mean(first_grip_err, weight)
    L_total = (
        float(cfg.get("pose_weight", 1.0)) * L_pose
        + float(cfg.get("first_pose_weight", 2.0)) * L_first_pose
        + float(cfg.get("delta_weight", 0.2)) * L_delta
        + float(cfg.get("grip_weight", 0.3)) * L_grip
        + float(cfg.get("first_grip_weight", 0.5)) * L_first_grip
    )
    grip_prob = torch.sigmoid(out["policy_gripper_logit"].float())
    grip_acc = ((grip_prob > 0.5) == (grip_tgt > 0.5)).float().mean(dim=1)
    transition_mask = grip_transition > 0.5
    if bool(transition_mask.any().detach().cpu()):
        transition_acc = ((grip_prob > 0.5) == (grip_tgt > 0.5)).float()[transition_mask].mean()
    else:
        transition_acc = grip_prob.new_zeros(())
    return {
        "L_total": L_total,
        "L_policy_pose": L_pose.detach(),
        "L_policy_first_pose": L_first_pose.detach(),
        "L_policy_delta": L_delta.detach(),
        "L_policy_grip": L_grip.detach(),
        "L_policy_first_grip": L_first_grip.detach(),
        "policy_pose_l1": _weighted_mean((pose_pred - pose_tgt).abs().mean(dim=(1, 2)), weight).detach(),
        "policy_first_pose_l1": _weighted_mean((pose_pred[:, 0] - pose_tgt[:, 0]).abs().mean(dim=1), weight).detach(),
        "policy_grip_acc": _weighted_mean(grip_acc, weight).detach(),
        "policy_grip_transition_acc": transition_acc.detach(),
        "policy_grip_transition_rate": grip_transition.float().mean().detach(),
        "policy_grip_prob_mean": grip_prob.mean().detach(),
    }


@torch.no_grad()
def _evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, loss_cfg: dict) -> dict[str, float]:
    model.eval()
    agg: dict[str, float] = {}
    n = 0
    for batch in loader:
        b = _batch_to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = _policy_forward(model, b)
            losses = _policy_losses(out, b, loss_cfg)
        for key, value in losses.items():
            agg[key] = agg.get(key, 0.0) + float(value.detach().float())
        n += 1
    model.train()
    return {key: val / max(1, n) for key, val in sorted(agg.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--print_every", type=int, default=25)
    args = ap.parse_args()

    rank, world, local_rank, distributed = _setup_distributed()
    cfg = yaml.safe_load(args.cfg.read_text())
    base_cfg = yaml.safe_load(Path(cfg["base_cfg"]).read_text())
    if distributed and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(cfg["train"].get("device", "cuda:0"))

    manifest_cfg = cfg["data"]["manifest"]
    manifests = [Path(item) for item in manifest_cfg] if isinstance(manifest_cfg, list) else Path(manifest_cfg)
    ds = LiberoExpertCacheDataset(manifests, plan_state_dim=int(cfg["data"].get("plan_state_dim", 8)))
    if bool(cfg["data"].get("preload", False)):
        if rank == 0:
            print(json.dumps({"preload": True, "windows": len(ds)}), flush=True)
        ds = InMemoryDataset(ds)
    val_frac = float(cfg["data"].get("val_frac", 0.1))
    n_val = max(1, int(len(ds) * val_frac))
    n_train = max(1, len(ds) - n_val)
    train_ds, val_ds = random_split(
        ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(int(cfg["data"].get("seed", 0))),
    )
    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        if distributed
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=True,
    )

    model = build_model(base_cfg)
    init_ckpt = Path(cfg["train"]["init_ckpt"])
    missing: list[str] = []
    unexpected: list[str] = []
    skipped: list[str] = []
    if rank == 0:
        sd = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        missing, unexpected, skipped = _load_init_state(model, sd["model"])
        bad_missing = [key for key in missing if not key.startswith("action_policy.")]
        bad_skipped = [key for key in skipped if not key.startswith("action_policy.")]
        if bad_missing or unexpected or bad_skipped:
            raise RuntimeError({"bad_missing": bad_missing, "bad_skipped": bad_skipped[:20], "unexpected": unexpected[:20]})
    _broadcast_model_state(model, world)
    model = model.to(device)
    _load_action_stats(model, Path(cfg["data"]["action_stats"]), device)
    frozen, trainable = _freeze(model, list(cfg["train"].get("trainable_prefixes", ["action_policy."])))
    if rank == 0:
        print(json.dumps({
            "loaded": str(init_ckpt),
            "missing_action_policy_keys": len([key for key in missing if key.startswith("action_policy.")]),
            "skipped_action_policy_keys": len([key for key in skipped if key.startswith("action_policy.")]),
            "train_windows": n_train,
            "val_windows": n_val,
            "frozen_M": frozen / 1e6,
            "trainable_M": trainable / 1e6,
            "world": world,
            "backend": dist.get_backend() if distributed else "none",
        }, sort_keys=True), flush=True)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["train"].get("lr", 1e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.02)),
        betas=(0.9, 0.95),
    )
    max_steps = int(args.max_steps or cfg["train"].get("max_steps", 1000))
    warmup = int(cfg["train"].get("warmup_steps", 50))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, max_steps - warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    out_root = Path(cfg["out"]["root"])
    ckpt_dir = out_root / cfg["out"].get("ckpt_dir", "ckpt")
    keep_eval_ckpts = bool(cfg["train"].get("keep_eval_ckpts", False))
    eval_ckpt_dir = out_root / str(cfg["train"].get("eval_ckpt_dir", cfg["out"].get("ckpt_dir", "ckpt")))
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if keep_eval_ckpts:
            eval_ckpt_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    best_val = float("inf")
    metrics: dict[str, float] = {}
    step = 0
    epoch = 0
    model.train()
    while step < max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            b = _batch_to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = _policy_forward(model, b)
                losses = _policy_losses(out, b, cfg["loss"])
            opt.zero_grad(set_to_none=True)
            losses["L_total"].backward()
            _all_reduce_gradients(model, world)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
            opt.step()
            sched.step()
            if rank == 0 and args.print_every and step % args.print_every == 0:
                print(
                    f"[libero-action-policy] step {step} "
                    f"L_total={float(losses['L_total'].detach().float()):.4f} "
                    f"pose_l1={float(losses['policy_pose_l1']):.4f} "
                    f"first_l1={float(losses['policy_first_pose_l1']):.4f} "
                    f"grip_acc={float(losses['policy_grip_acc']):.3f} "
                    f"lr={sched.get_last_lr()[0]:.2e}",
                    flush=True,
                )
            step += 1
            if step % int(cfg["train"].get("eval_every", 250)) == 0 or step >= max_steps:
                metrics = _evaluate(model, val_loader, device, cfg["loss"])
                if rank == 0:
                    out_root.mkdir(parents=True, exist_ok=True)
                    (out_root / "metrics_latest.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
                    val_total = float(metrics["L_total"])
                    print(json.dumps({"step": step, "val": metrics}, sort_keys=True), flush=True)
                    ckpt = {
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "sched": sched.state_dict(),
                        "step": step,
                        "cfg": cfg,
                        "base_cfg": base_cfg,
                        "metrics": metrics,
                    }
                    torch.save(ckpt, ckpt_dir / "latest.pt")
                    if keep_eval_ckpts:
                        torch.save(ckpt, eval_ckpt_dir / f"eval_step_{step:06d}.pt")
                    if val_total <= best_val:
                        best_val = val_total
                        torch.save(ckpt, ckpt_dir / "best.pt")
                if distributed:
                    dist.barrier()
            if step >= max_steps:
                break
        epoch += 1

    if rank == 0:
        (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
        print(json.dumps({"metrics": metrics, "ckpt": str(ckpt_dir / "best.pt"), "step": step}, indent=2, sort_keys=True), flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
