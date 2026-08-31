"""Train the direct WM3D action policy head on LIBERO expert token cache."""
from __future__ import annotations

import argparse
import hashlib
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
from wm3d_v3.training.train import load_train_config
from wm3d_v3.training.v7_freeze_guard import (
    ModuleFingerprint,
    assert_module_frozen,
    assert_no_grad,
    assert_optimizer_excludes,
)


ACTION_POLICY_PREFIX = "action_policy."
STRICT_STAGE2_MODE = "v7_strict_policy_only"


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


def _validate_strict_policy_only(train_cfg: dict[str, Any]) -> bool:
    strict = bool(train_cfg.get(STRICT_STAGE2_MODE, False))
    if not strict:
        return False
    prefixes = list(train_cfg.get("trainable_prefixes", []))
    if prefixes != [ACTION_POLICY_PREFIX]:
        raise RuntimeError(
            f"{STRICT_STAGE2_MODE}=true requires trainable_prefixes exactly "
            f"[{ACTION_POLICY_PREFIX!r}], got {prefixes!r}"
        )
    if not train_cfg.get("action_policy_init_ckpt"):
        raise RuntimeError(
            f"{STRICT_STAGE2_MODE}=true requires train.action_policy_init_ckpt; "
            "refusing to initialize the Stage2 policy from scratch"
        )
    return True


def _set_policy_only_train_mode(model: torch.nn.Module) -> None:
    target = _unwrap_model(model)
    policy = getattr(target, "action_policy", None)
    if policy is None:
        raise RuntimeError("strict Stage2 requires model.action_policy")
    target.eval()
    policy.train()


def _frozen_top_level_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    target = _unwrap_model(model)
    return [
        (name, module)
        for name, module in target.named_children()
        if name != "action_policy"
    ]


def _assert_strict_freeze_contract(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    check_grad: bool = False,
) -> None:
    target = _unwrap_model(model)
    policy = getattr(target, "action_policy", None)
    if policy is None:
        raise RuntimeError("strict Stage2 requires model.action_policy")
    policy_ids = {id(parameter) for parameter in policy.parameters()}
    trainable_ids = {id(parameter) for parameter in target.parameters() if parameter.requires_grad}
    if trainable_ids != policy_ids:
        unexpected = [
            name
            for name, parameter in target.named_parameters()
            if parameter.requires_grad and not name.startswith(ACTION_POLICY_PREFIX)
        ]
        missing = [
            name
            for name, parameter in target.named_parameters()
            if name.startswith(ACTION_POLICY_PREFIX) and not parameter.requires_grad
        ]
        raise RuntimeError(
            "strict Stage2 trainable parameter set is not exactly action_policy.*: "
            + json.dumps({"unexpected": unexpected[:16], "missing": missing[:16]}, sort_keys=True)
        )
    for _name, module in _frozen_top_level_modules(target):
        assert_module_frozen(module)
        if optimizer is not None:
            assert_optimizer_excludes(module, optimizer)
        if check_grad:
            assert_no_grad(module)
    if check_grad:
        root_offenders = [
            name
            for name, parameter in target.named_parameters()
            if not name.startswith(ACTION_POLICY_PREFIX)
            and parameter.grad is not None
            and bool(torch.any(parameter.grad != 0))
        ]
        if root_offenders:
            raise RuntimeError(
                f"strict Stage2 frozen parameters received gradients: {root_offenders[:16]}"
            )
    if optimizer is not None:
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_ids != policy_ids:
            raise RuntimeError("strict Stage2 optimizer parameters are not exactly action_policy.*")


def _normalize_state_dict_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        key = str(raw_key)
        while key.startswith("module."):
            key = key[len("module.") :]
        if key in normalized:
            raise RuntimeError(f"checkpoint contains duplicate normalized key {key!r}")
        normalized[key] = value
    return normalized


def _checkpoint_state(payload: Any, path: Path) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint {path} is not a mapping")
    state = payload.get("model", payload.get("state_dict"))
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint {path} has no model/state_dict mapping")
    if not all(torch.is_tensor(value) for value in state.values()):
        raise RuntimeError(f"checkpoint {path} state dict contains non-tensor values")
    return _normalize_state_dict_keys(state)


def _load_stage1_init(
    model: torch.nn.Module,
    payload: dict[str, Any],
    *,
    path: Path,
    strict_non_policy: bool,
) -> dict[str, Any]:
    source = _checkpoint_state(payload, path)
    target = model.state_dict()
    if strict_non_policy:
        source_keys = {key for key in source if not key.startswith(ACTION_POLICY_PREFIX)}
        target_keys = {key for key in target if not key.startswith(ACTION_POLICY_PREFIX)}
        missing = sorted(target_keys.difference(source_keys))
        unexpected = sorted(source_keys.difference(target_keys))
        shape_mismatch = sorted(
            key
            for key in source_keys.intersection(target_keys)
            if tuple(source[key].shape) != tuple(target[key].shape)
        )
        if missing or unexpected or shape_mismatch:
            raise RuntimeError(
                "V7 Stage1 non-policy checkpoint contract mismatch: "
                + json.dumps(
                    {
                        "missing": missing[:32],
                        "unexpected": unexpected[:32],
                        "shape_mismatch": shape_mismatch[:32],
                    },
                    sort_keys=True,
                )
            )
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    skipped = sorted(set(source).difference(compatible))
    return {
        "path": str(path),
        "loaded": len(compatible),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "skipped": skipped,
    }


def _native_policy_contract_from_model_cfg(model_cfg: dict[str, Any]) -> dict[str, object]:
    state_cfg = model_cfg.get("state") or {}
    head_type = str(model_cfg.get("policy_head_type", "native"))
    if head_type != "native":
        raise RuntimeError(
            "non-native action-policy warm starts require an explicit action_policy_contract"
        )
    return {
        "version": "wm3d_action_policy_v1",
        "head_type": head_type,
        "horizon": int(model_cfg.get("policy_horizon") or state_cfg.get("k", 8)),
        "context_schema": {
            "lowdim_dim": int(model_cfg.get("policy_lowdim_dim", 0)),
            "action_history_len": int(model_cfg.get("policy_action_history_len", 0)),
            "action_history_dim": int(model_cfg.get("policy_action_history_dim", 7)),
            "use_context_rgb": bool(model_cfg.get("policy_use_context_rgb", False)),
            "use_task": bool(model_cfg.get("policy_use_task", True)),
            "task_dim": int(
                model_cfg.get("policy_task_dim")
                or state_cfg.get("cond_dim", 2048)
            ),
            "patch_pool": str(model_cfg.get("policy_patch_pool", "mean")),
            "max_spatial_tokens": int(model_cfg.get("policy_max_spatial_tokens", 64)),
        },
        "joint_behavior": {
            "policy_context_source": str(model_cfg.get("policy_context_source", "input")),
            "policy_core_action_cond": str(model_cfg.get("policy_core_action_cond", "same")),
            "policy_action_add_trunk": bool(model_cfg.get("policy_action_add_trunk", True)),
        },
    }


def _checkpoint_action_policy_contract(payload: dict[str, Any], path: Path) -> dict[str, object]:
    explicit = payload.get("action_policy_contract")
    if isinstance(explicit, dict):
        return explicit
    base_cfg = payload.get("base_cfg")
    if not isinstance(base_cfg, dict) or not isinstance(base_cfg.get("model"), dict):
        raise RuntimeError(
            f"action-policy checkpoint {path} has neither action_policy_contract "
            "nor base_cfg.model for a native contract audit"
        )
    return _native_policy_contract_from_model_cfg(base_cfg["model"])


def _load_action_policy_init_strict(
    model: torch.nn.Module,
    payload: dict[str, Any],
    *,
    path: Path,
) -> dict[str, Any]:
    target_model = _unwrap_model(model)
    if getattr(target_model, "action_policy", None) is None:
        raise RuntimeError("action-policy warm start requires model.action_policy")
    expected_contract = target_model.action_policy_checkpoint_contract()
    source_contract = _checkpoint_action_policy_contract(payload, path)
    if source_contract != expected_contract:
        raise RuntimeError(
            "action-policy checkpoint contract mismatch: "
            + json.dumps(
                {"expected": expected_contract, "actual": source_contract},
                sort_keys=True,
            )
        )
    source = _checkpoint_state(payload, path)
    target = target_model.state_dict()
    source_keys = {key for key in source if key.startswith(ACTION_POLICY_PREFIX)}
    target_keys = {key for key in target if key.startswith(ACTION_POLICY_PREFIX)}
    missing = sorted(target_keys.difference(source_keys))
    unexpected = sorted(source_keys.difference(target_keys))
    shape_mismatch = sorted(
        key
        for key in source_keys.intersection(target_keys)
        if tuple(source[key].shape) != tuple(target[key].shape)
    )
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "action-policy checkpoint keys/shapes are not exact: "
            + json.dumps(
                {
                    "missing": missing[:32],
                    "unexpected": unexpected[:32],
                    "shape_mismatch": shape_mismatch[:32],
                },
                sort_keys=True,
            )
        )
    load_result = target_model.load_state_dict(
        {key: source[key] for key in sorted(source_keys)},
        strict=False,
    )
    missing_policy = [
        key for key in load_result.missing_keys if key.startswith(ACTION_POLICY_PREFIX)
    ]
    unexpected_policy = [
        key for key in load_result.unexpected_keys if key.startswith(ACTION_POLICY_PREFIX)
    ]
    if missing_policy or unexpected_policy:
        raise RuntimeError(
            "action-policy strict load failed: "
            + json.dumps(
                {"missing": missing_policy, "unexpected": unexpected_policy},
                sort_keys=True,
            )
        )
    return {
        "path": str(path),
        "loaded": len(source_keys),
        "contract": expected_contract,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hashes(
    cfg_path: Path,
    cfg: dict[str, Any],
    *,
    init_ckpt: Path,
    action_policy_init_ckpt: Path | None,
) -> dict[str, str]:
    paths: list[tuple[str, Path]] = [
        ("stage2_cfg", cfg_path),
        ("stage1_init_ckpt", init_ckpt),
        ("action_stats", Path(cfg["data"]["action_stats"])),
    ]
    if action_policy_init_ckpt is not None:
        paths.append(("action_policy_init_ckpt", action_policy_init_ckpt))
    manifest_cfg = cfg["data"]["manifest"]
    manifests = manifest_cfg if isinstance(manifest_cfg, list) else [manifest_cfg]
    paths.extend((f"manifest_{idx}", Path(path)) for idx, path in enumerate(manifests))
    hashes: dict[str, str] = {}
    for name, path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"cannot hash missing Stage2 artifact {name}: {path}")
        hashes[name] = _sha256_file(path)
    return hashes


def _stage2_checkpoint_metadata(
    model: torch.nn.Module,
    *,
    artifact_hashes: dict[str, str],
    stage1_report: dict[str, Any],
    action_policy_init_report: dict[str, Any],
    strict_policy_only: bool,
    trainable_prefixes: list[str],
    frozen_fingerprint: ModuleFingerprint | None,
    last_verified_fingerprint: str | None,
    verified_at_step: int,
) -> dict[str, Any]:
    target = _unwrap_model(model)
    action_policy_contract = target.action_policy_checkpoint_contract()
    contract_sha256 = hashlib.sha256(
        json.dumps(
            action_policy_contract,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "action_policy_contract": action_policy_contract,
        "action_policy_contract_sha256": contract_sha256,
        "artifact_sha256": dict(artifact_hashes),
        "stage1_init_report": dict(stage1_report),
        "action_policy_init_report": dict(action_policy_init_report),
        "freeze_contract": {
            "strict_policy_only": bool(strict_policy_only),
            "trainable_prefixes": list(trainable_prefixes),
            "frozen_fingerprint_before": (
                frozen_fingerprint.sha256 if frozen_fingerprint else None
            ),
            "frozen_fingerprint_last_verified": last_verified_fingerprint,
            "fingerprint_excluded_prefixes": (
                list(frozen_fingerprint.excluded_prefixes)
                if frozen_fingerprint
                else []
            ),
            "verified_at_step": int(verified_at_step),
        },
    }


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
    if "s_wrist" in batch:
        out["s_wrist"] = batch["s_wrist"].to(device, non_blocking=True)
    if "view_mask" in batch:
        out["view_mask"] = batch["view_mask"].to(device, non_blocking=True).bool()
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
        return target.act_policy(
            batch["s"],
            batch["c"],
            wrist_s=batch.get("s_wrist"),
            view_mask=batch.get("view_mask"),
            **policy_kwargs,
        )
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
    restore_policy_only_train: bool = False,
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
    if restore_policy_only_train:
        _set_policy_only_train_mode(model)
    else:
        model.train()
    return {key: val / max(1, n) for key, val in sorted(agg.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--print_every", type=int, default=25)
    args = ap.parse_args()

    rank, world, local_rank, distributed = _setup_distributed()
    cfg = load_train_config(args.cfg)
    base_cfg = load_train_config(Path(cfg["base_cfg"]))
    strict_policy_only = _validate_strict_policy_only(cfg["train"])
    require_multiview = bool(cfg["data"].get("require_multiview", False))
    if strict_policy_only and not require_multiview:
        raise RuntimeError(
            f"{STRICT_STAGE2_MODE}=true requires data.require_multiview=true"
        )
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
        require_multiview=require_multiview,
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
    action_policy_init_ckpt = (
        Path(cfg["train"]["action_policy_init_ckpt"])
        if cfg["train"].get("action_policy_init_ckpt")
        else None
    )
    stage1_report: dict[str, Any] = {}
    action_policy_init_report: dict[str, Any] = {}
    artifact_hashes: dict[str, str] = {}
    if rank == 0:
        stage1_payload = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        if strict_policy_only:
            stage1_report = _load_stage1_init(
                model,
                stage1_payload,
                path=init_ckpt,
                strict_non_policy=True,
            )
        else:
            source_state = _checkpoint_state(stage1_payload, init_ckpt)
            missing, unexpected, skipped = _load_init_state(model, source_state)
            allowed_prefixes = (ACTION_POLICY_PREFIX, "geom.")
            bad_missing = [key for key in missing if not key.startswith(allowed_prefixes)]
            bad_skipped = [key for key in skipped if not key.startswith(allowed_prefixes)]
            if bad_missing or unexpected or bad_skipped:
                raise RuntimeError(
                    {
                        "bad_missing": bad_missing,
                        "bad_skipped": bad_skipped[:20],
                        "unexpected": unexpected[:20],
                    }
                )
            stage1_report = {
                "path": str(init_ckpt),
                "loaded": len(source_state) - len(skipped),
                "missing": missing,
                "unexpected": unexpected,
                "skipped": skipped,
            }
        del stage1_payload
        if action_policy_init_ckpt is not None:
            policy_payload = torch.load(
                action_policy_init_ckpt,
                map_location="cpu",
                weights_only=False,
            )
            action_policy_init_report = _load_action_policy_init_strict(
                model,
                policy_payload,
                path=action_policy_init_ckpt,
            )
            del policy_payload
        if strict_policy_only:
            artifact_hashes = _artifact_hashes(
                args.cfg,
                cfg,
                init_ckpt=init_ckpt,
                action_policy_init_ckpt=action_policy_init_ckpt,
            )
    model = model.to(device)
    _broadcast_model_state(model, world)
    _load_action_stats(model, Path(cfg["data"]["action_stats"]), device)
    policy_forward_mode = str(cfg["train"].get("policy_forward_mode", "action_policy"))
    if policy_forward_mode not in {"action_policy", "act_policy"}:
        raise ValueError(f"unknown train.policy_forward_mode={policy_forward_mode!r}")
    if strict_policy_only and policy_forward_mode != "act_policy":
        raise RuntimeError(
            f"{STRICT_STAGE2_MODE}=true requires policy_forward_mode='act_policy' "
            "so training and closed-loop serving share the same fused-view path"
        )
    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", cfg["train"].get("gradient_accumulation_steps", 1)))
    if grad_accum_steps < 1:
        raise ValueError(f"train.grad_accum_steps must be >= 1, got {grad_accum_steps}")
    frozen, trainable = _freeze(model, list(cfg["train"].get("trainable_prefixes", ["action_policy."])))
    frozen_fingerprint = (
        ModuleFingerprint.capture(
            _unwrap_model(model),
            exclude_prefixes=(ACTION_POLICY_PREFIX,),
        )
        if strict_policy_only
        else None
    )
    last_verified_fingerprint = frozen_fingerprint.sha256 if frozen_fingerprint else None
    if strict_policy_only:
        _assert_strict_freeze_contract(model)
    if rank == 0:
        per_gpu_batch_size = int(cfg["train"]["batch_size"])
        print(json.dumps({
            "loaded": str(init_ckpt),
            "stage1_init": stage1_report,
            "action_policy_init": action_policy_init_report,
            "strict_policy_only": strict_policy_only,
            "frozen_fingerprint": last_verified_fingerprint,
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "split": split_info,
            "frozen_M": frozen / 1e6,
            "trainable_M": trainable / 1e6,
            "world": world,
            "backend": dist.get_backend() if distributed else "none",
            "policy_forward_mode": policy_forward_mode,
            "per_gpu_batch_size": per_gpu_batch_size,
            "grad_accum_steps": grad_accum_steps,
            "effective_batch_size": per_gpu_batch_size * world * grad_accum_steps,
        }, sort_keys=True), flush=True)

    opt = torch.optim.AdamW(
        _build_optimizer_param_groups(model, cfg["train"], rank),
        betas=(0.9, 0.95),
    )
    if strict_policy_only:
        _assert_strict_freeze_contract(model, opt)
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
    micro_step = 0
    fingerprint_every = int(cfg["train"].get("freeze_fingerprint_every_steps", 0) or 0)
    if fingerprint_every < 0:
        raise ValueError("train.freeze_fingerprint_every_steps must be >= 0")
    if strict_policy_only:
        _set_policy_only_train_mode(model)
    else:
        model.train()
    opt.zero_grad(set_to_none=True)
    while step < max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            b = _batch_to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = _policy_forward(model, b, mode=policy_forward_mode)
                losses = _policy_losses(out, b, cfg["loss"], model=model)
            (losses["L_total"] / grad_accum_steps).backward()
            micro_step += 1
            if micro_step % grad_accum_steps != 0:
                continue
            _all_reduce_gradients(model, world)
            if strict_policy_only:
                _assert_strict_freeze_contract(model, opt, check_grad=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if rank == 0 and args.print_every and step % args.print_every == 0:
                print(
                    f"[libero-action-policy] step {step} "
                    f"L_total={float(losses['L_total'].detach().float()):.4f} "
                    f"pose_l1={float(losses['policy_pose_l1']):.4f} "
                    f"first_l1={float(losses['policy_first_pose_l1']):.4f} "
                    f"grip_acc={float(losses['policy_grip_acc']):.3f} "
                    f"lr={sched.get_last_lr()[0]:.2e} "
                    f"accum={grad_accum_steps}",
                    flush=True,
                )
            step += 1
            if (
                strict_policy_only
                and fingerprint_every > 0
                and step % fingerprint_every == 0
            ):
                assert frozen_fingerprint is not None
                frozen_fingerprint.assert_unchanged(_unwrap_model(model))
                last_verified_fingerprint = frozen_fingerprint.sha256
            if step % int(cfg["train"].get("eval_every", 250)) == 0 or step >= max_steps:
                if strict_policy_only and step >= max_steps:
                    assert frozen_fingerprint is not None
                    frozen_fingerprint.assert_unchanged(_unwrap_model(model))
                    last_verified_fingerprint = frozen_fingerprint.sha256
                eval_max_batches = cfg["train"].get("eval_max_batches")
                metrics = _evaluate(
                    model,
                    val_loader,
                    device,
                    cfg["loss"],
                    policy_forward_mode=policy_forward_mode,
                    max_batches=int(eval_max_batches) if eval_max_batches is not None else None,
                    restore_policy_only_train=strict_policy_only,
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
                        **_stage2_checkpoint_metadata(
                            model,
                            artifact_hashes=artifact_hashes,
                            stage1_report=stage1_report,
                            action_policy_init_report=action_policy_init_report,
                            strict_policy_only=strict_policy_only,
                            trainable_prefixes=list(
                                cfg["train"].get("trainable_prefixes", [ACTION_POLICY_PREFIX])
                            ),
                            frozen_fingerprint=frozen_fingerprint,
                            last_verified_fingerprint=last_verified_fingerprint,
                            verified_at_step=(
                                step
                                if strict_policy_only
                                and (
                                    step >= max_steps
                                    or (fingerprint_every > 0 and step % fingerprint_every == 0)
                                )
                                else 0
                            ),
                        ),
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
