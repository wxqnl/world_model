"""Train the direct WM3D action policy head on LIBERO expert token cache."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Subset, random_split
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


def _episode_group_key(row: dict[str, Any]) -> str:
    if row.get("source_json"):
        return "source_json:" + str(row["source_json"])
    if row.get("hdf5_path") and row.get("demo_id"):
        return "hdf5:" + str(row["hdf5_path"]) + "::" + str(row["demo_id"])
    if row.get("source_jsonl") and row.get("demo_id"):
        return "source_jsonl:" + str(row["source_jsonl"]) + "::" + str(row["demo_id"])
    return "cache:" + str(row.get("cache_path", ""))


def _split_dataset(
    ds: LiberoExpertCacheDataset,
    *,
    val_frac: float,
    seed: int,
    split_mode: str,
) -> tuple[Dataset, Dataset, dict[str, Any]]:
    n_total = len(ds)
    n_val = max(1, int(n_total * val_frac))
    n_train = max(1, n_total - n_val)
    split_mode = str(split_mode or "random").lower()
    if split_mode in {"random", "window"}:
        train_ds, val_ds = random_split(
            ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
        return train_ds, val_ds, {
            "val_split": split_mode,
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "train_groups": None,
            "val_groups": None,
        }
    if split_mode not in {"episode", "cache_path"}:
        raise ValueError("data.val_split must be one of: random, window, episode, cache_path")

    groups: dict[str, list[int]] = {}
    for idx, row in enumerate(ds.rows):
        if split_mode == "cache_path":
            key = "cache:" + str(row.get("cache_path", ""))
        else:
            key = _episode_group_key(row)
        groups.setdefault(key, []).append(idx)

    keys = list(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    val_keys: set[str] = set()
    val_count = 0
    for key in keys:
        if val_count >= n_val and val_keys:
            break
        val_keys.add(key)
        val_count += len(groups[key])

    val_idx = sorted(idx for key in val_keys for idx in groups[key])
    train_idx = sorted(idx for key in keys if key not in val_keys for idx in groups[key])
    if not train_idx or not val_idx:
        train_ds, val_ds = random_split(
            ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
        return train_ds, val_ds, {
            "val_split": split_mode,
            "fallback": "random",
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "train_groups": None,
            "val_groups": None,
        }

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    return train_ds, val_ds, {
        "val_split": split_mode,
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "train_groups": len(keys) - len(val_keys),
        "val_groups": len(val_keys),
    }


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


def _normalise_param_name(name: str) -> str:
    while name.startswith("module."):
        name = name[len("module.") :]
    return name


def _prefix_matches(name: str, prefix: str) -> bool:
    prefix = str(prefix).strip()
    if not prefix:
        return False
    prefix = prefix[:-1] if prefix.endswith(".") else prefix
    return name == prefix or name.startswith(prefix + ".")


def _lr_multiplier_for_name(name: str, lr_multipliers: dict[str, float]) -> float:
    clean_name = _normalise_param_name(name)
    best_prefix = ""
    best_mult = 1.0
    for prefix, mult in lr_multipliers.items():
        if _prefix_matches(clean_name, prefix) and len(str(prefix)) > len(best_prefix):
            best_prefix = str(prefix)
            best_mult = float(mult)
    return best_mult


def _build_optimizer_param_groups(model: torch.nn.Module, train_cfg: dict, rank: int) -> list[dict[str, Any]]:
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.02))
    raw_multipliers = train_cfg.get("lr_multipliers") or {}
    if not isinstance(raw_multipliers, dict):
        raise ValueError("train.lr_multipliers must be a mapping of prefix -> multiplier")
    lr_multipliers = {str(k): float(v) for k, v in raw_multipliers.items()}
    grouped: dict[float, dict[str, Any]] = {}
    group_counts: dict[float, int] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        mult = _lr_multiplier_for_name(name, lr_multipliers)
        if mult <= 0:
            raise ValueError(f"lr multiplier must be positive for {name}, got {mult}")
        group = grouped.setdefault(
            mult,
            {
                "params": [],
                "lr": lr * mult,
                "weight_decay": weight_decay,
                "lr_multiplier": mult,
            },
        )
        group["params"].append(param)
        group_counts[mult] = group_counts.get(mult, 0) + param.numel()
    if not grouped:
        raise RuntimeError("optimizer has no trainable parameters after trainable_prefixes filtering")
    if rank == 0:
        if lr_multipliers:
            summary = ", ".join(
                f"mult={mult:g} lr={lr * mult:.3g} params={group_counts[mult] / 1e6:.2f}M"
                for mult in sorted(group_counts)
            )
            print(f"[rank0] optimizer lr groups: {summary}", flush=True)
        else:
            total = sum(group_counts.values())
            print(f"[rank0] optimizer single lr={lr:.3g} params={total / 1e6:.2f}M", flush=True)
    return [grouped[mult] for mult in sorted(grouped)]


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


def _axis_weights(cfg: dict, key: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    raw = cfg.get(key)
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 6:
        raise ValueError(f"loss.{key} must be a list of 6 numeric weights")
    weights = torch.tensor([float(v) for v in raw], device=device, dtype=dtype).view(1, 1, 6)
    return weights / weights.mean().clamp_min(1e-6)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _denormalize_pose_norm(pose_norm: torch.Tensor, model: torch.nn.Module | None) -> torch.Tensor | None:
    if model is None:
        return None
    action_proj = getattr(_unwrap_model(model), "action_proj", None)
    if action_proj is None or not hasattr(action_proj, "mean") or not hasattr(action_proj, "std"):
        return None
    mean = action_proj.mean.detach().to(device=pose_norm.device, dtype=pose_norm.dtype)
    std = action_proj.std.detach().to(device=pose_norm.device, dtype=pose_norm.dtype).clamp_min(1e-6)
    return pose_norm * std + mean


def _policy_forward(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    mode: str = "action_policy",
) -> dict[str, torch.Tensor]:
    target = model.module if hasattr(model, "module") else model
    if getattr(target, "action_policy", None) is None:
        raise RuntimeError("enable_action_policy is required for policy training")
    policy_kwargs = {
        "lowdim_state": batch.get("lowdim_state"),
        "object_state": batch.get("object_state"),
        "plan_state": batch.get("plan_state"),
        "action_history": batch.get("action_history"),
        "progress_state": batch.get("progress_state"),
        "context_rgb": batch.get("context_rgb"),
    }
    if mode == "action_policy":
        return target.action_policy(batch["s"], task_emb=batch["c"], **policy_kwargs)
    if mode == "act_policy":
        return target.act_policy(batch["s"], batch["c"], **policy_kwargs)
    raise ValueError(f"unknown train.policy_forward_mode={mode!r}; expected action_policy or act_policy")


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


def _policy_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: dict,
    *,
    model: torch.nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    if "policy_action_cond" not in out:
        raise RuntimeError("model output has no policy_action_cond; enable_action_policy is required")
    pose_pred = out["policy_pose_norm"].float()
    pose_tgt = batch["action_tgt_norm"].float()
    weight = batch["proposer_weight"].float()
    if float(weight.sum().detach().cpu()) <= 0:
        weight = torch.ones_like(weight)

    pose_axis_weight = _axis_weights(cfg, "pose_axis_weights", pose_pred.device, pose_pred.dtype)
    pose_elem = F.smooth_l1_loss(
        pose_pred,
        pose_tgt,
        beta=float(cfg.get("huber_delta", 1.0)),
        reduction="none",
    )
    if pose_axis_weight is not None:
        pose_elem = pose_elem * pose_axis_weight
    pose_err = pose_elem.mean(dim=(1, 2))
    first_pose_err = pose_elem[:, 0].mean(dim=1)
    norm_delta = pose_pred[:, 1:] - pose_pred[:, :-1]
    tgt_delta = pose_tgt[:, 1:] - pose_tgt[:, :-1]
    delta_err = F.smooth_l1_loss(
        norm_delta,
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
    pose_raw_pred = _denormalize_pose_norm(pose_pred, model)
    L_raw_pose = pose_pred.new_zeros(())
    L_raw_first_pose = pose_pred.new_zeros(())
    L_raw_delta = pose_pred.new_zeros(())
    raw_loss_metrics: dict[str, torch.Tensor] = {}
    if pose_raw_pred is not None:
        pose_raw_tgt = batch["action_tgt"].float()[..., :6]
        raw_axis_key = "raw_pose_axis_weights" if cfg.get("raw_pose_axis_weights") is not None else "pose_axis_weights"
        raw_axis_weight = _axis_weights(cfg, raw_axis_key, pose_raw_pred.device, pose_raw_pred.dtype)
        raw_huber_delta = float(cfg.get("raw_huber_delta", cfg.get("huber_delta", 1.0)))
        raw_elem = F.smooth_l1_loss(
            pose_raw_pred.float(),
            pose_raw_tgt,
            beta=raw_huber_delta,
            reduction="none",
        )
        if raw_axis_weight is not None:
            raw_elem = raw_elem * raw_axis_weight
        raw_pose_err = raw_elem.mean(dim=(1, 2))
        raw_first_pose_err = raw_elem[:, 0].mean(dim=1)
        raw_delta_pred = pose_raw_pred[:, 1:] - pose_raw_pred[:, :-1]
        raw_delta_tgt = pose_raw_tgt[:, 1:] - pose_raw_tgt[:, :-1]
        raw_delta_elem = F.smooth_l1_loss(
            raw_delta_pred.float(),
            raw_delta_tgt,
            beta=raw_huber_delta,
            reduction="none",
        )
        if raw_axis_weight is not None:
            raw_delta_elem = raw_delta_elem * raw_axis_weight
        raw_delta_err = raw_delta_elem.mean(dim=(1, 2))
        L_raw_pose = _weighted_mean(raw_pose_err, weight)
        L_raw_first_pose = _weighted_mean(raw_first_pose_err, weight)
        L_raw_delta = _weighted_mean(raw_delta_err, weight)
        raw_loss_metrics = {
            "L_policy_raw_pose": L_raw_pose.detach(),
            "L_policy_raw_first_pose": L_raw_first_pose.detach(),
            "L_policy_raw_delta": L_raw_delta.detach(),
        }
    L_total = (
        float(cfg.get("pose_weight", 1.0)) * L_pose
        + float(cfg.get("first_pose_weight", 2.0)) * L_first_pose
        + float(cfg.get("delta_weight", 0.2)) * L_delta
        + float(cfg.get("raw_pose_weight", 0.0)) * L_raw_pose
        + float(cfg.get("raw_first_pose_weight", 0.0)) * L_raw_first_pose
        + float(cfg.get("raw_delta_weight", 0.0)) * L_raw_delta
        + float(cfg.get("grip_weight", 0.3)) * L_grip
        + float(cfg.get("first_grip_weight", 0.5)) * L_first_grip
    )
    grip_prob = torch.sigmoid(out["policy_gripper_logit"].float())
    grip_acc = ((grip_prob > 0.5) == (grip_tgt > 0.5)).float().mean(dim=1)
    first_grip_acc = ((grip_prob[:, 0] > 0.5) == (grip_tgt[:, 0] > 0.5)).float()
    transition_mask = grip_transition > 0.5
    if bool(transition_mask.any().detach().cpu()):
        transition_acc = ((grip_prob > 0.5) == (grip_tgt > 0.5)).float()[transition_mask].mean()
    else:
        transition_acc = grip_prob.new_zeros(())
    metrics = {
        "L_total": L_total,
        "L_policy_pose": L_pose.detach(),
        "L_policy_first_pose": L_first_pose.detach(),
        "L_policy_delta": L_delta.detach(),
        "L_policy_grip": L_grip.detach(),
        "L_policy_first_grip": L_first_grip.detach(),
        "policy_pose_l1": _weighted_mean((pose_pred - pose_tgt).abs().mean(dim=(1, 2)), weight).detach(),
        "policy_first_pose_l1": _weighted_mean((pose_pred[:, 0] - pose_tgt[:, 0]).abs().mean(dim=1), weight).detach(),
        "policy_grip_acc": _weighted_mean(grip_acc, weight).detach(),
        "policy_first_grip_acc": _weighted_mean(first_grip_acc, weight).detach(),
        "policy_grip_transition_acc": transition_acc.detach(),
        "policy_grip_transition_rate": grip_transition.float().mean().detach(),
        "policy_grip_prob_mean": grip_prob.mean().detach(),
    }
    metrics.update(raw_loss_metrics)
    if pose_raw_pred is not None:
        pose_raw_tgt = batch["action_tgt"].float()[..., :6]
        raw_l1 = (pose_raw_pred.float() - pose_raw_tgt).abs()
        pose_raw_l1 = _weighted_mean(raw_l1.mean(dim=(1, 2)), weight).detach()
        first_pose_raw_l1 = _weighted_mean(raw_l1[:, 0].mean(dim=1), weight).detach()
        pose_raw_xyz_l1 = _weighted_mean(raw_l1[..., :3].mean(dim=(1, 2)), weight).detach()
        first_pose_raw_xyz_l1 = _weighted_mean(raw_l1[:, 0, :3].mean(dim=1), weight).detach()
        first_pose_raw_y_l1 = _weighted_mean(raw_l1[:, 0, 1], weight).detach()
        metrics["policy_pose_raw_l1"] = pose_raw_l1
        metrics["policy_first_pose_raw_l1"] = first_pose_raw_l1
        metrics["policy_pose_raw_xyz_l1"] = pose_raw_xyz_l1
        metrics["policy_first_pose_raw_xyz_l1"] = first_pose_raw_xyz_l1
        metrics["policy_first_pose_raw_y_l1"] = first_pose_raw_y_l1
        metrics["L_serve_exact"] = (
            first_pose_raw_l1
            + 0.25 * pose_raw_l1
            + 0.05 * (1.0 - metrics["policy_first_grip_acc"])
        ).detach()
        metrics["L_serve_spatial"] = (
            first_pose_raw_xyz_l1
            + 0.25 * pose_raw_xyz_l1
            + 0.05 * (1.0 - metrics["policy_first_grip_acc"])
        ).detach()
    return metrics


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: dict,
    policy_forward_mode: str = "action_policy",
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    agg: dict[str, float] = {}
    n = 0
    for batch in loader:
        b = _batch_to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = _policy_forward(model, b, mode=policy_forward_mode)
            losses = _policy_losses(out, b, loss_cfg, model=model)
        for key, value in losses.items():
            agg[key] = agg.get(key, 0.0) + float(value.detach().float())
        n += 1
        if max_batches is not None and n >= max_batches:
            break
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
    ds = LiberoExpertCacheDataset(
        manifests,
        plan_state_dim=int(cfg["data"].get("plan_state_dim", 8)),
        include_action_history=int(base_cfg.get("model", {}).get("policy_action_history_len", 0) or 0) > 0,
    )
    val_frac = float(cfg["data"].get("val_frac", 0.1))
    split_info: dict[str, Any]
    train_ds, val_ds, split_info = _split_dataset(
        ds,
        val_frac=val_frac,
        seed=int(cfg["data"].get("seed", 0)),
        split_mode=str(cfg["data"].get("val_split", "random")),
    )
    if bool(cfg["data"].get("preload", False)):
        if rank == 0:
            print(json.dumps({"preload": True, "windows": len(ds)}), flush=True)
        mem_ds = InMemoryDataset(ds)
        train_ds = Subset(mem_ds, list(getattr(train_ds, "indices", [])))
        val_ds = Subset(mem_ds, list(getattr(val_ds, "indices", [])))
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
        allowed_prefixes = ("action_policy.", "geom.")
        bad_missing = [key for key in missing if not key.startswith(allowed_prefixes)]
        bad_skipped = [key for key in skipped if not key.startswith(allowed_prefixes)]
        if bad_missing or unexpected or bad_skipped:
            raise RuntimeError({"bad_missing": bad_missing, "bad_skipped": bad_skipped[:20], "unexpected": unexpected[:20]})
    model = model.to(device)
    _broadcast_model_state(model, world)
    _load_action_stats(model, Path(cfg["data"]["action_stats"]), device)
    policy_forward_mode = str(cfg["train"].get("policy_forward_mode", "action_policy"))
    if policy_forward_mode not in {"action_policy", "act_policy"}:
        raise ValueError(f"unknown train.policy_forward_mode={policy_forward_mode!r}")
    frozen, trainable = _freeze(model, list(cfg["train"].get("trainable_prefixes", ["action_policy."])))
    if rank == 0:
        print(json.dumps({
            "loaded": str(init_ckpt),
            "missing_action_policy_keys": len([key for key in missing if key.startswith("action_policy.")]),
            "skipped_action_policy_keys": len([key for key in skipped if key.startswith("action_policy.")]),
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "split": split_info,
            "frozen_M": frozen / 1e6,
            "trainable_M": trainable / 1e6,
            "world": world,
            "backend": dist.get_backend() if distributed else "none",
            "policy_forward_mode": policy_forward_mode,
        }, sort_keys=True), flush=True)

    opt = torch.optim.AdamW(
        _build_optimizer_param_groups(model, cfg["train"], rank),
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

    select_metric = str(cfg["train"].get("val_select_metric", "L_total"))
    select_mode = str(cfg["train"].get("val_select_mode", "min")).lower()
    if select_mode in {"minimize", "lower"}:
        select_mode = "min"
    if select_mode in {"maximize", "higher"}:
        select_mode = "max"
    if select_mode not in {"min", "max"}:
        raise ValueError(f"train.val_select_mode must be min/max, got {select_mode!r}")
    best_val = float("inf") if select_mode == "min" else -float("inf")
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
                out = _policy_forward(model, b, mode=policy_forward_mode)
                losses = _policy_losses(out, b, cfg["loss"], model=model)
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
                eval_max_batches = cfg["train"].get("eval_max_batches")
                metrics = _evaluate(
                    model,
                    val_loader,
                    device,
                    cfg["loss"],
                    policy_forward_mode=policy_forward_mode,
                    max_batches=int(eval_max_batches) if eval_max_batches is not None else None,
                )
                if rank == 0:
                    out_root.mkdir(parents=True, exist_ok=True)
                    (out_root / "metrics_latest.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
                    metric_name = select_metric if select_metric in metrics else "L_total"
                    val_total = float(metrics[metric_name])
                    print(json.dumps({"step": step, "val": metrics}, sort_keys=True), flush=True)
                    ckpt = {
                        "model": model.state_dict(),
                        "step": step,
                        "cfg": cfg,
                        "base_cfg": base_cfg,
                        "metrics": metrics,
                        "select_metric": metric_name,
                        "select_mode": select_mode,
                        "select_value": val_total,
                    }
                    if bool(cfg["train"].get("save_optimizer", True)):
                        ckpt["opt"] = opt.state_dict()
                        ckpt["sched"] = sched.state_dict()
                    torch.save(ckpt, ckpt_dir / "latest.pt")
                    if keep_eval_ckpts:
                        torch.save(ckpt, eval_ckpt_dir / f"eval_step_{step:06d}.pt")
                    is_best = val_total <= best_val if select_mode == "min" else val_total >= best_val
                    if is_best:
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
