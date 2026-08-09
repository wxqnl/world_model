"""v3 joint training. DDP-aware, bf16, gradient checkpointing optional."""
from __future__ import annotations
import argparse
import datetime
import gc
import hashlib
import json
import math
import os
import random
import socket
import sys
import threading
import time
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Sampler,
    Subset,
    WeightedRandomSampler,
)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.training.v7_branch_loss import (
    TrueBranchLossConfig,
    true_branch_reconstruction_matching_loss,
)
from wm3d_v3.training.v7_future_value_loss import (
    FutureValueLossConfig,
    true_branch_future_value_loss,
)
from wm3d_v3.training.v7_native_action_loss import (
    NativeActionLossConfig,
    native_action_loss,
)
from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.mixed_source_dataset import (
    MixedSourceWindowDataset,
    SourceHomogeneousDistributedBatchSampler,
    partition_v7_compact_dataset,
)
from wm3d_v3.data.splits import episode_split, load_clip_split_file, random_window_indices
from wm3d_v3.data.v7_compact_dataset import (
    V7CompactDatasetConfig,
    V7CompactWindowDataset,
    V7SameRootBranchDataset,
    V7SameRootBranchDatasetConfig,
)
from wm3d_v3.data.window_dataset import OXEWindowDataset, WindowConfig
from wm3d_v3.losses import (
    LossWeights,
    _normalize_depth,
    compute_losses,
    compute_native_action_losses,
)
from wm3d_v3.models.action_stream import ActionConfig
from wm3d_v3.models.dual_stream import DualConfig
from wm3d_v3.models.hunyuan_latent_adapter import (
    HunyuanLatentAdapter,
    HunyuanLatentAdapterConfig,
)
from wm3d_v3.training.lr_schedule import (
    build_lr_scheduler,
    resolve_optimizer_settings,
)
from wm3d_v3.models.joint_model import JointConfig, JointWorldModel
from wm3d_v3.models.model_factory import build_joint_world_model
from wm3d_v3.models.state_stream import StateConfig
from wm3d_v3.stage1.action_contract import action_contract_key


def _deep_merge_config(base: dict, overlay: dict) -> dict:
    result = dict(base)
    for key, value in overlay.items():
        if (
            not str(key).endswith("_exact")
            and key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge_config(result[key], value)
        else:
            result[key] = value
    return result


_DIRECT_POLICY_OXE_OVERRIDE_KEYS = frozenset(
    {
        "T",
        "k",
        "stride",
        "load_rgb",
        "load_geom",
        "load_state_tgt",
        "load_geom_extra",
        "require_geom_extra",
        "use_window_tokens",
        "cache_root",
        "window_geom_subdir",
        "window_geom_cache_root",
        "trust_window_geom_cache",
        "max_windows_per_episode",
        "causal_dual_view_required",
        "causal_dual_view_representation",
        "load_policy_state",
        "require_policy_state",
        "strict_policy_state_prescan",
        "policy_lowdim_dim",
        "policy_object_state_dim",
        "policy_plan_state_dim",
        "policy_action_history_len",
        "policy_action_history_dim",
        "window_geom_shard_index",
        "window_geom_shard_root",
    }
)


def apply_direct_policy_oxe_overrides(source_cfg: dict, data_cfg: dict) -> dict:
    """Apply loader-only K/action-policy overrides to every audited OXE source.

    Identity, canonical-action adapters, statistics, manifests, and audit gates are
    deliberately not overrideable here. This keeps the five-source V7 contract
    intact while allowing an action-only stage to use a longer policy horizon.
    """

    raw = data_cfg.get("direct_policy_oxe_overrides") or {}
    if not isinstance(raw, dict):
        raise ValueError("data.direct_policy_oxe_overrides must be a mapping")
    forbidden = sorted(set(raw) - _DIRECT_POLICY_OXE_OVERRIDE_KEYS)
    if forbidden:
        raise ValueError(
            "data.direct_policy_oxe_overrides contains non-loader keys: "
            + ", ".join(forbidden)
        )
    return _deep_merge_config(source_cfg, raw)


def load_train_config(path: Path, _seen: set[Path] | None = None) -> dict:
    """Load a YAML config with one explicit V7 `_base_` inheritance chain."""

    path = path.resolve()
    seen = set(_seen or ())
    if path in seen:
        raise ValueError(f"cyclic config inheritance: {path}")
    seen.add(path)
    payload = yaml.safe_load(path.read_text()) or {}
    base_ref = payload.pop("_base_", None)
    if base_ref is None:
        return payload
    base_path = Path(base_ref)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    return _deep_merge_config(load_train_config(base_path, seen), payload)


def seed_process_rng(seed: int) -> None:
    """Seed Python, NumPy and Torch RNGs for one process."""

    seed = int(seed)
    if seed < 0:
        raise ValueError(f"training seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resume_compat_config(cfg: dict) -> dict:
    """Drop only bounded-run controls that may change across an exact resume."""

    compatible = json.loads(json.dumps(_cache_safe_value(cfg)))
    train_cfg = compatible.get("train") or {}
    for key in (
        "fresh_init_required",
        "forbid_resume",
        "allow_same_run_exact_resume",
        "reset_optimizer",
        "resume_checkpoint",
        "resolved_config_sha256",
        "resolved_resume_compat_sha256",
        "resolved_run_lineage",
    ):
        train_cfg.pop(key, None)
    out_cfg = compatible.get("out") or {}
    out_cfg.pop("require_empty_checkpoint_dir", None)
    return compatible


def config_sha256(payload: dict) -> str:
    text = json.dumps(
        _cache_safe_value(payload), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def capture_rng_contract(base_seed: int, rank: int) -> dict:
    """Checkpoint RNG evidence for the step-addressed stochastic contract."""

    result = {
        "schema": "wm3d_v7_step_addressed_rng_v1",
        "base_seed": int(base_seed),
        "rank": int(rank),
        "rank_stride": 100_003,
        "step_offset": 10_000_019,
        "python_state": random.getstate(),
        "numpy_state": np.random.get_state(),
        "torch_cpu_state": torch.get_rng_state().cpu(),
    }


def module_state_sha256(module: torch.nn.Module | None) -> str | None:
    """Stable digest for small newly initialized stage-specific modules."""

    if module is None:
        return None
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
    if torch.cuda.is_available():
        result["torch_cuda_state"] = torch.cuda.get_rng_state().cpu()
    return result


def build_exact_resume_startup_event(
    *,
    checkpoint_path: Path,
    checkpoint_payload: dict,
    resolved_config_sha256: str,
    resolved_resume_compat_sha256: str,
    resolved_run_lineage: str,
    model_load_result,
    model_strict: bool,
    optimizer: torch.optim.Optimizer,
    scheduler,
    restored_global_step: int,
    sampler_state: dict,
    runtime_sampler_epoch: int,
    runtime_micro_batches_in_epoch: int,
    runtime_source_cycle_position: int,
    runtime_sampler_num_replicas: int,
    runtime_sampler_rank_scope: str,
    next_batch_source: str,
    rng_contract: dict,
    base_seed: int,
) -> dict:
    """Build factual evidence only after every exact-resume restore succeeded."""

    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise RuntimeError(f"resume evidence checkpoint is missing: {checkpoint_path}")
    checkpoint_step = int(checkpoint_payload.get("step", -1))
    if checkpoint_step != int(restored_global_step):
        raise RuntimeError(
            "resume evidence global step mismatch: "
            f"checkpoint={checkpoint_step} runtime={restored_global_step}"
        )
    if checkpoint_payload.get("resume_compat_sha256") != resolved_resume_compat_sha256:
        raise RuntimeError("resume evidence config compatibility digest mismatch")
    if checkpoint_payload.get("run_lineage") != resolved_run_lineage:
        raise RuntimeError("resume evidence run lineage mismatch")

    model_load = {
        "strict": bool(model_strict),
        "missing_keys": list(getattr(model_load_result, "missing_keys", []) or []),
        "unexpected_keys": list(
            getattr(model_load_result, "unexpected_keys", []) or []
        ),
        "skipped_keys": list(getattr(model_load_result, "skipped_keys", []) or []),
        "expanded_keys": list(getattr(model_load_result, "expanded_keys", []) or []),
    }
    if not model_load["strict"] or any(
        model_load[key]
        for key in ("missing_keys", "unexpected_keys", "skipped_keys", "expanded_keys")
    ):
        raise RuntimeError("resume startup evidence requires a clean strict model load")

    optimizer_state = optimizer.state_dict()
    checkpoint_optimizer = checkpoint_payload.get("opt") or {}
    optimizer_metadata = {
        "loaded": True,
        "state_entries": len(optimizer_state.get("state") or {}),
        "param_groups": len(optimizer_state.get("param_groups") or []),
        "checkpoint_state_entries": len(checkpoint_optimizer.get("state") or {}),
        "checkpoint_param_groups": len(
            checkpoint_optimizer.get("param_groups") or []
        ),
    }
    optimizer_metadata["metadata_matches_checkpoint"] = (
        optimizer_metadata["state_entries"]
        == optimizer_metadata["checkpoint_state_entries"]
        and optimizer_metadata["param_groups"]
        == optimizer_metadata["checkpoint_param_groups"]
    )
    if (
        optimizer_metadata["state_entries"] <= 0
        or optimizer_metadata["param_groups"] <= 0
        or not optimizer_metadata["metadata_matches_checkpoint"]
    ):
        raise RuntimeError("resume optimizer metadata does not match the checkpoint")

    scheduler_state = scheduler.state_dict()
    checkpoint_scheduler = checkpoint_payload.get("sched") or {}
    scheduler_metadata = {
        "loaded": True,
        "last_epoch": int(scheduler_state.get("last_epoch", -1)),
        "step_count": int(scheduler_state.get("_step_count", -1)),
        "checkpoint_last_epoch": int(checkpoint_scheduler.get("last_epoch", -2)),
        "checkpoint_step_count": int(checkpoint_scheduler.get("_step_count", -2)),
    }
    scheduler_metadata["metadata_matches_checkpoint"] = (
        scheduler_metadata["last_epoch"]
        == scheduler_metadata["checkpoint_last_epoch"]
        and scheduler_metadata["step_count"]
        == scheduler_metadata["checkpoint_step_count"]
    )
    if not scheduler_metadata["metadata_matches_checkpoint"]:
        raise RuntimeError("resume scheduler metadata does not match the checkpoint")

    sampler_state_copy = _cache_safe_value(dict(sampler_state))
    expected_sampler = {
        "epoch": int(runtime_sampler_epoch),
        "micro_batches_consumed_in_epoch": int(runtime_micro_batches_in_epoch),
        "source_cycle_position": int(runtime_source_cycle_position),
        "sampler_num_replicas": int(runtime_sampler_num_replicas),
        "sampler_rank_scope": str(runtime_sampler_rank_scope),
    }
    for key, expected in expected_sampler.items():
        if sampler_state_copy.get(key) != expected:
            raise RuntimeError(
                f"resume sampler evidence mismatch for {key}: "
                f"checkpoint={sampler_state_copy.get(key)!r} runtime={expected!r}"
            )

    rng_schema = str(rng_contract.get("schema", ""))
    rng_base_seed = int(rng_contract.get("base_seed", -1))
    if (
        rng_schema != "wm3d_v7_step_addressed_rng_v1"
        or rng_base_seed != int(base_seed)
    ):
        raise RuntimeError("resume RNG contract evidence is incompatible")

    return {
        "schema": "wm3d_v7_action_dynamics_resume_telemetry_v1",
        "record_type": "startup_event",
        "event": "exact_resume_restored",
        "wall_time_unix_s": float(time.time()),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_basename": checkpoint_path.name,
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "checkpoint_size_bytes": int(checkpoint_path.stat().st_size),
        "checkpoint_step": checkpoint_step,
        "resolved_config_sha256": str(resolved_config_sha256),
        "resume_compat_sha256": str(resolved_resume_compat_sha256),
        "run_lineage": str(resolved_run_lineage),
        "model_load": model_load,
        "optimizer": optimizer_metadata,
        "scheduler": scheduler_metadata,
        "global_step": int(restored_global_step),
        "sampler_state": sampler_state_copy,
        "sampler_restore": {
            **expected_sampler,
            "verified": True,
            "fast_forward_applied": int(runtime_micro_batches_in_epoch) > 0,
            "fast_forward_start_batch": int(runtime_micro_batches_in_epoch),
            "fast_forward_without_dataset_io": True,
            "skipped_dataset_fetches": int(runtime_micro_batches_in_epoch),
            "next_source_cycle_position": int(runtime_source_cycle_position),
            "next_batch_source": str(next_batch_source),
        },
        "rng_contract": {
            "schema": rng_schema,
            "base_seed": rng_base_seed,
            "verified": True,
            "rng_mode": "step_addressed_reconstruction",
            "raw_rng_restored": False,
            "step_addressed_rng_reconstructed": True,
            "torch_rng_state_restored": False,
            "semantics": "step_addressed_contract_verified_not_state_restore",
        },
    }


def append_exact_resume_startup_event_once(path: Path, event: dict) -> None:
    """Append one non-step resume record; duplicate evidence is a hard error."""

    if event.get("schema") != "wm3d_v7_action_dynamics_resume_telemetry_v1":
        raise ValueError("unexpected resume startup event schema")
    if event.get("event") != "exact_resume_restored" or "step" in event:
        raise ValueError("resume startup evidence must be a non-step record")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if (
                    row.get("schema") == event["schema"]
                    and row.get("event") == event["event"]
                ):
                    raise RuntimeError(
                        "canary telemetry already contains exact-resume startup evidence"
                    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_rank_local_cache_shard_config(
    mixed_sampler_cfg: dict,
    *,
    contract_profile: str,
) -> dict:
    """Validate the canary-only unequal local-cache sampler contract."""

    raw = mixed_sampler_cfg.get("rank_local_cache_shards") or {}
    if not isinstance(raw, dict):
        raise ValueError("mixed_batch_sampler.rank_local_cache_shards must be a mapping")
    shard_cfg = dict(raw)
    if not bool(shard_cfg.get("enabled", False)):
        return {"enabled": False}
    if not str(contract_profile).startswith("canary"):
        raise ValueError("rank-local cache shards are restricted to canary profiles")
    required = {
        "scope": "canary_system_validation_only",
        "allow_unequal_source_lengths": True,
        "require_each_source_at_least_global_batch": True,
        "record_startup_telemetry": True,
    }
    for key, expected in required.items():
        if shard_cfg.get(key) != expected:
            raise ValueError(
                f"rank_local_cache_shards.{key} must equal {expected!r}"
            )
    if int(mixed_sampler_cfg.get("num_batches_per_epoch", -1)) != 50_000:
        raise ValueError("rank-local cache shards require num_batches_per_epoch=50000")
    if int(mixed_sampler_cfg.get("val_num_batches_per_epoch", -1)) != 100:
        raise ValueError("rank-local cache shards require val_num_batches_per_epoch=100")
    if mixed_sampler_cfg.get("synchronized_across_ranks") is not True:
        raise ValueError("rank-local cache shards require synchronized_across_ranks=true")
    if mixed_sampler_cfg.get("accumulation_group_same_source") is not True:
        raise ValueError("rank-local cache shards require accumulation_group_same_source=true")
    if mixed_sampler_cfg.get("shuffle_cycle") is not True:
        raise ValueError("rank-local cache shards require shuffle_cycle=true")
    cycle_counts = mixed_sampler_cfg.get("source_cycle_counts_exact") or {}
    cycle_steps = int(mixed_sampler_cfg.get("cycle_optimizer_steps", -1))
    if not isinstance(cycle_counts, dict) or sum(int(v) for v in cycle_counts.values()) != cycle_steps:
        raise ValueError("rank-local cache shard source cycle must match cycle_optimizer_steps")
    node_roles = shard_cfg.get("node_roles")
    if not isinstance(node_roles, dict):
        raise ValueError("rank_local_cache_shards.node_roles must be a mapping")
    expected_roles = {
        "0": {"host": "node43", "oxe_cache": "full_primary"},
        "1": {
            "host": "node44",
            "oxe_cache": "strict_subset_partial",
        },
    }
    if node_roles != expected_roles:
        raise ValueError(
            "rank_local_cache_shards.node_roles must explicitly declare "
            "node43=full_primary and node44=strict_subset_partial"
        )
    return {**shard_cfg, "enabled": True}


def _dataset_runtime_window_identity(dataset) -> str:
    """Hash the already-built sample topology without opening payload archives."""

    digest = hashlib.sha256()
    target = dataset
    subset_depth = 0
    while isinstance(target, Subset):
        indices = np.asarray(target.indices, dtype=np.int64).reshape(-1)
        digest.update(f"subset:{subset_depth}:{indices.size}\n".encode("utf-8"))
        digest.update(indices.astype("<i8", copy=False).tobytes())
        target = target.dataset
        subset_depth += 1

    digest.update(
        f"class:{target.__class__.__module__}.{target.__class__.__qualname__}\n".encode(
            "utf-8"
        )
    )
    digest.update(f"length:{len(target)}\n".encode("utf-8"))
    cfg = getattr(target, "cfg", None)
    for field in (
        "split",
        "T",
        "k",
        "stride",
        "canonical_action_cache_manifest_sha256",
        "trusted_manifest_sha256",
    ):
        if cfg is not None and hasattr(cfg, field):
            digest.update(
                f"cfg:{field}:{getattr(cfg, field)!s}\n".encode("utf-8")
            )

    records = getattr(target, "records", None)
    if records is not None:
        digest.update(f"records:{len(records)}\n".encode("utf-8"))
        stable_fields = (
            "clip_hash",
            "root_id",
            "clip_id",
            "dataset",
            "source",
            "split",
            "model_frames",
            "n_frames",
            "fps",
            "action_kind",
            "geometry_segments",
        )
        for record in records:
            if isinstance(record, dict):
                identity = {
                    field: record[field]
                    for field in stable_fields
                    if field in record
                }
            else:
                identity = {
                    field: getattr(record, field)
                    for field in stable_fields
                    if hasattr(record, field)
                }
            digest.update(
                json.dumps(
                    _cache_safe_value(identity),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(b"\n")

    index = getattr(target, "index", None)
    if index is not None:
        index_array = np.asarray(index, dtype=np.int64)
        digest.update(
            f"index:{index_array.shape}:{index_array.size}\n".encode("utf-8")
        )
        digest.update(index_array.astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def mixed_source_runtime_identity_digests(
    dataset: MixedSourceWindowDataset,
) -> dict[str, str]:
    return {
        source_name: _dataset_runtime_window_identity(source_dataset)
        for source_name, source_dataset in zip(
            dataset.source_names, dataset.datasets
        )
    }


def validate_global_mixed_source_contract_matrix(
    source_names: tuple[str, ...] | list[str],
    train_lengths_by_rank: list[list[int]],
    val_lengths_by_rank: list[list[int]],
    train_identity_by_rank: list[list[str]],
    val_identity_by_rank: list[list[str]],
) -> dict:
    """Prove a global-rank sampler sees one identical dataset on every rank."""

    names = tuple(str(name) for name in source_names)
    if not names:
        raise RuntimeError("global mixed sampler requires non-empty source names")
    world = len(train_lengths_by_rank)
    collections = {
        "train lengths": train_lengths_by_rank,
        "val lengths": val_lengths_by_rank,
        "train identities": train_identity_by_rank,
        "val identities": val_identity_by_rank,
    }
    width = len(names)
    for label, rows in collections.items():
        if len(rows) != world or not rows:
            raise RuntimeError(f"global mixed sampler {label} rank count mismatch")
        if any(len(row) != width for row in rows):
            raise RuntimeError(f"global mixed sampler {label} width mismatch")
        if any(row != rows[0] for row in rows[1:]):
            raise RuntimeError(
                f"global mixed sampler {label} differ across ranks; "
                "formal training requires identical per-source window pools"
            )
    return {
        "enabled": True,
        "mode": "global-identical",
        "source_names": list(names),
        "world_size": int(world),
        "train_source_lengths": dict(zip(names, train_lengths_by_rank[0])),
        "val_source_lengths": dict(zip(names, val_lengths_by_rank[0])),
        "train_source_identity_sha256": dict(
            zip(names, train_identity_by_rank[0])
        ),
        "val_source_identity_sha256": dict(zip(names, val_identity_by_rank[0])),
    }


def _distributed_source_count_and_name_audit(
    source_names: tuple[str, ...],
    *,
    world: int,
    device: torch.device,
    label: str,
) -> None:
    source_count = torch.tensor([len(source_names)], device=device, dtype=torch.int32)
    minimum = source_count.clone()
    maximum = source_count.clone()
    if world > 1:
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if not torch.equal(minimum, maximum):
        raise RuntimeError(f"{label} source counts differ across ranks")
    name_digest = hashlib.sha256("\0".join(source_names).encode("utf-8")).digest()
    local = torch.tensor(list(name_digest), device=device, dtype=torch.uint8)
    gathered = [torch.empty_like(local) for _ in range(world)]
    if world > 1:
        dist.all_gather(gathered, local)
    else:
        gathered[0].copy_(local)
    if any(not torch.equal(value, gathered[0]) for value in gathered[1:]):
        raise RuntimeError(f"{label} source names/order differ across ranks")


def _capacity_tensor(
    values: list[int], *, world: int, device: torch.device, label: str
) -> torch.Tensor:
    normalized = [int(value) for value in values]
    if any(value < 0 for value in normalized):
        raise RuntimeError(f"{label} contains a negative source length")
    int32_max = int(torch.iinfo(torch.int32).max)
    if normalized and max(normalized) * max(1, int(world)) > int32_max:
        raise RuntimeError(f"{label} exceeds the int32 distributed audit range")
    return torch.tensor(normalized, device=device, dtype=torch.int32)


def audit_distributed_global_mixed_source_contract(
    train_dataset: MixedSourceWindowDataset,
    val_dataset: MixedSourceWindowDataset,
    *,
    world: int,
    rank: int,
    device: torch.device,
) -> dict:
    """Fail before loader construction if a global sampler pool is not identical."""

    if tuple(train_dataset.source_names) != tuple(val_dataset.source_names):
        raise RuntimeError("global mixed sampler local train/val source names differ")
    names = tuple(train_dataset.source_names)
    _distributed_source_count_and_name_audit(
        names, world=world, device=device, label="global mixed sampler"
    )
    train_values = [
        train_dataset.source_spans[name][1] - train_dataset.source_spans[name][0]
        for name in names
    ]
    val_values = [
        val_dataset.source_spans[name][1] - val_dataset.source_spans[name][0]
        for name in names
    ]
    local_train = _capacity_tensor(
        train_values, world=world, device=device, label="global train capacities"
    )
    local_val = _capacity_tensor(
        val_values, world=world, device=device, label="global val capacities"
    )
    train_min = local_train.clone()
    train_max = local_train.clone()
    train_sum = local_train.clone()
    val_min = local_val.clone()
    val_max = local_val.clone()
    val_sum = local_val.clone()
    if world > 1:
        for tensor, operation in (
            (train_min, dist.ReduceOp.MIN),
            (train_max, dist.ReduceOp.MAX),
            (train_sum, dist.ReduceOp.SUM),
            (val_min, dist.ReduceOp.MIN),
            (val_max, dist.ReduceOp.MAX),
            (val_sum, dist.ReduceOp.SUM),
        ):
            dist.all_reduce(tensor, op=operation)
    gathered_train = [torch.empty_like(local_train) for _ in range(world)]
    gathered_val = [torch.empty_like(local_val) for _ in range(world)]
    if world > 1:
        dist.all_gather(gathered_train, local_train)
        dist.all_gather(gathered_val, local_val)
    else:
        gathered_train[0].copy_(local_train)
        gathered_val[0].copy_(local_val)

    train_digests = mixed_source_runtime_identity_digests(train_dataset)
    val_digests = mixed_source_runtime_identity_digests(val_dataset)
    local_digest_bytes = bytes.fromhex(
        "".join(train_digests[name] for name in names)
        + "".join(val_digests[name] for name in names)
    )
    local_digest_tensor = torch.tensor(
        list(local_digest_bytes), device=device, dtype=torch.uint8
    )
    gathered_digests = [torch.empty_like(local_digest_tensor) for _ in range(world)]
    if world > 1:
        dist.all_gather(gathered_digests, local_digest_tensor)
    else:
        gathered_digests[0].copy_(local_digest_tensor)

    digest_width = 32
    rank_train_identities: list[list[str]] = []
    rank_val_identities: list[list[str]] = []
    for tensor in gathered_digests:
        raw = bytes(tensor.cpu().tolist())
        chunks = [
            raw[offset : offset + digest_width].hex()
            for offset in range(0, len(raw), digest_width)
        ]
        rank_train_identities.append(chunks[: len(names)])
        rank_val_identities.append(chunks[len(names) :])
    audit = validate_global_mixed_source_contract_matrix(
        names,
        [tensor.cpu().tolist() for tensor in gathered_train],
        [tensor.cpu().tolist() for tensor in gathered_val],
        rank_train_identities,
        rank_val_identities,
    )
    audit.update(
        {
            "rank": int(rank),
            "all_reduce_source_lengths": {
                "train_min": dict(zip(names, train_min.cpu().tolist())),
                "train_max": dict(zip(names, train_max.cpu().tolist())),
                "train_sum": dict(zip(names, train_sum.cpu().tolist())),
                "val_min": dict(zip(names, val_min.cpu().tolist())),
                "val_max": dict(zip(names, val_max.cpu().tolist())),
                "val_sum": dict(zip(names, val_sum.cpu().tolist())),
            },
        }
    )
    return audit


def validate_rank_local_source_length_matrix(
    source_names: tuple[str, ...] | list[str],
    train_lengths_by_rank: list[list[int]],
    val_lengths_by_rank: list[list[int]],
    *,
    global_batch: int,
    local_world_size: int,
) -> dict:
    """Validate unequal per-rank capacity while preserving a shared schedule."""

    names = tuple(str(name) for name in source_names)
    if not names:
        raise ValueError("rank-local cache audit requires non-empty source names")
    if len(train_lengths_by_rank) != len(val_lengths_by_rank) or not train_lengths_by_rank:
        raise ValueError("rank-local train/val rank vectors must be non-empty and aligned")
    world = len(train_lengths_by_rank)
    if local_world_size <= 0 or world % int(local_world_size) != 0:
        raise ValueError("rank-local cache audit requires complete equal-size nodes")
    if world // int(local_world_size) != 2:
        raise ValueError("node43/node44 rank-local canary requires exactly two nodes")
    width = len(names)
    matrices = {
        "train": [[int(value) for value in row] for row in train_lengths_by_rank],
        "val": [[int(value) for value in row] for row in val_lengths_by_rank],
    }
    for split, rows in matrices.items():
        if any(len(row) != width for row in rows):
            raise ValueError(f"rank-local {split} source-length vector width mismatch")
        minimum = min(value for row in rows for value in row)
        if minimum < int(global_batch):
            raise ValueError(
                f"rank-local {split} source has {minimum} windows, fewer than "
                f"global batch {global_batch}"
            )
        for node_index in range(2):
            start = node_index * int(local_world_size)
            stop = start + int(local_world_size)
            node_rows = rows[start:stop]
            if any(row != node_rows[0] for row in node_rows[1:]):
                raise ValueError(
                    f"rank-local {split} source lengths differ inside node {node_index}"
                )
    oxe_indices = [index for index, name in enumerate(names) if name.startswith("oxe_")]
    if not oxe_indices:
        raise ValueError("rank-local cache shard contract requires OXE sources")
    node43_train = matrices["train"][0]
    node44_train = matrices["train"][int(local_world_size)]
    node43_val = matrices["val"][0]
    node44_val = matrices["val"][int(local_world_size)]
    non_strict_sources = {
        split: [
            names[index]
            for index in oxe_indices
            if node44[index] >= node43[index]
        ]
        for split, node43, node44 in (
            ("train", node43_train, node44_train),
            ("val", node43_val, node44_val),
        )
    }
    non_strict_sources = {
        split: sources
        for split, sources in non_strict_sources.items()
        if sources
    }
    if non_strict_sources:
        raise ValueError(
            "node44 strict-subset runtime train/val length must be smaller for "
            f"every OXE source; violations={non_strict_sources}"
        )
    node43_oxe_train = sum(node43_train[index] for index in oxe_indices)
    node44_oxe_train = sum(node44_train[index] for index in oxe_indices)

    def summary(rows: list[list[int]]) -> dict:
        columns = list(zip(*rows))
        return {
            "min": {name: min(columns[index]) for index, name in enumerate(names)},
            "max": {name: max(columns[index]) for index, name in enumerate(names)},
            "sum": {name: sum(columns[index]) for index, name in enumerate(names)},
            "by_rank": [
                {name: row[index] for index, name in enumerate(names)}
                for row in rows
            ],
        }

    return {
        "enabled": True,
        "source_names": list(names),
        "world_size": world,
        "local_world_size": int(local_world_size),
        "global_batch": int(global_batch),
        "node_oxe_runtime_lengths": {
            "node43": {
                "role": "full_primary",
                "train": int(node43_oxe_train),
                "val": int(sum(node43_val[index] for index in oxe_indices)),
            },
            "node44": {
                "role": "strict_subset_partial",
                "train": int(node44_oxe_train),
                "val": int(sum(node44_val[index] for index in oxe_indices)),
            },
            "membership_contract": "node44_strict_subset_of_node43",
            "membership_evidence": "formal_manifest_last_wins_preflight",
            "runtime_check": "per_source_train_val_lengths_strictly_smaller",
        },
        "train_source_lengths": summary(matrices["train"]),
        "val_source_lengths": summary(matrices["val"]),
        "unequal_source_lengths_observed": any(
            len(set(column)) > 1 for column in zip(*matrices["train"])
        ),
    }


def audit_distributed_rank_local_source_lengths(
    train_dataset: MixedSourceWindowDataset,
    val_dataset: MixedSourceWindowDataset,
    mixed_sampler_cfg: dict,
    *,
    batch_size: int,
    world: int,
    rank: int,
    device: torch.device,
    local_world_size: int,
    contract_profile: str,
) -> dict:
    """All-rank capacity audit for rank-local source caches."""

    shard_cfg = validate_rank_local_cache_shard_config(
        mixed_sampler_cfg, contract_profile=contract_profile
    )
    if not shard_cfg["enabled"]:
        return {"enabled": False}
    if tuple(train_dataset.source_names) != tuple(val_dataset.source_names):
        raise RuntimeError("rank-local train/val source names differ")
    names = tuple(train_dataset.source_names)
    _distributed_source_count_and_name_audit(
        names, world=world, device=device, label="rank-local mixed sampler"
    )
    local_train = _capacity_tensor(
        [
            train_dataset.source_spans[name][1]
            - train_dataset.source_spans[name][0]
            for name in names
        ],
        world=world,
        device=device,
        label="rank-local train capacities",
    )
    local_val = _capacity_tensor(
        [
            val_dataset.source_spans[name][1] - val_dataset.source_spans[name][0]
            for name in names
        ],
        world=world,
        device=device,
        label="rank-local val capacities",
    )
    train_min = local_train.clone()
    train_max = local_train.clone()
    train_sum = local_train.clone()
    val_min = local_val.clone()
    val_max = local_val.clone()
    val_sum = local_val.clone()
    if world > 1:
        dist.all_reduce(train_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(train_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(train_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(val_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(val_sum, op=dist.ReduceOp.SUM)
    gathered_train = [torch.empty_like(local_train) for _ in range(world)]
    gathered_val = [torch.empty_like(local_val) for _ in range(world)]
    if world > 1:
        dist.all_gather(gathered_train, local_train)
        dist.all_gather(gathered_val, local_val)
    else:
        gathered_train[0].copy_(local_train)
        gathered_val[0].copy_(local_val)
    audit = validate_rank_local_source_length_matrix(
        names,
        [tensor.cpu().tolist() for tensor in gathered_train],
        [tensor.cpu().tolist() for tensor in gathered_val],
        global_batch=int(batch_size) * int(world),
        local_world_size=int(local_world_size),
    )
    audit.update(
        {
            "rank": int(rank),
            "num_batches_per_epoch": int(mixed_sampler_cfg["num_batches_per_epoch"]),
            "val_num_batches_per_epoch": int(
                mixed_sampler_cfg["val_num_batches_per_epoch"]
            ),
            "node_roles": shard_cfg["node_roles"],
            "all_reduce_source_lengths": {
                "train_min": dict(zip(names, train_min.cpu().tolist())),
                "train_max": dict(zip(names, train_max.cpu().tolist())),
                "train_sum": dict(zip(names, train_sum.cpu().tolist())),
                "val_min": dict(zip(names, val_min.cpu().tolist())),
                "val_max": dict(zip(names, val_max.cpu().tolist())),
                "val_sum": dict(zip(names, val_sum.cpu().tolist())),
            },
        }
    )
    return audit


def finalize_rank_local_loader_length_audit(
    audit: dict,
    *,
    train_loader_length: int,
    val_loader_length: int,
    world: int,
    device: torch.device,
) -> dict:
    """Prove every rank enters train/val collectives the same number of times."""

    if not audit.get("enabled", False):
        return audit
    local = torch.tensor(
        [int(train_loader_length), int(val_loader_length)],
        device=device,
        dtype=torch.int32,
    )
    minimum = local.clone()
    maximum = local.clone()
    if world > 1:
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    expected = torch.tensor(
        [audit["num_batches_per_epoch"], audit["val_num_batches_per_epoch"]],
        device=device,
        dtype=torch.int32,
    )
    if not torch.equal(minimum, maximum) or not torch.equal(minimum, expected):
        raise RuntimeError(
            "rank-local loader lengths are not globally fixed: "
            f"min={minimum.cpu().tolist()} max={maximum.cpu().tolist()} "
            f"expected={expected.cpu().tolist()}"
        )
    audit = dict(audit)
    audit["loader_lengths_all_ranks"] = {
        "train": int(minimum[0].item()),
        "val": int(minimum[1].item()),
    }
    return audit


def save_step_checkpoint_once(
    checkpoint: dict,
    checkpoint_dir: Path,
    step: int,
) -> Path:
    """Write one physical checkpoint and atomically repoint latest.pt.

    The previous path serialized the same 15 GB payload twice at every save.
    A relative symlink preserves the existing resume interface while removing
    the duplicate I/O and DDP barrier stall.
    """

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    step_path = checkpoint_dir / f"step_{int(step):08d}.pt"
    temporary_step = checkpoint_dir / f".{step_path.name}.tmp.{os.getpid()}"
    torch.save(checkpoint, temporary_step)
    os.replace(temporary_step, step_path)

    latest_path = checkpoint_dir / "latest.pt"
    temporary_latest = checkpoint_dir / f".latest.pt.tmp.{os.getpid()}"
    try:
        temporary_latest.unlink(missing_ok=True)
        os.symlink(step_path.name, temporary_latest)
        os.replace(temporary_latest, latest_path)
    finally:
        temporary_latest.unlink(missing_ok=True)
    return step_path


def setup_ddp():
    if "RANK" in os.environ:
        backend = os.environ.get("WM3D_DDP_BACKEND", "nccl")
        env_rank = int(os.environ.get("RANK", "0"))
        local = int(os.environ.get("LOCAL_RANK", env_rank))
        timeout_minutes = int(os.environ.get("WM3D_DDP_TIMEOUT_MINUTES", "30") or 30)
        if timeout_minutes <= 0:
            raise ValueError("WM3D_DDP_TIMEOUT_MINUTES must be positive")
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(minutes=timeout_minutes),
        )
        rank = dist.get_rank()
        world = dist.get_world_size()
        return rank, world, local
    return 0, 1, 0


def eager_initialize_distributed_transport(
    *, rank: int, world: int, device: torch.device
) -> dict:
    """Initialize and verify the default communicator before asymmetric I/O.

    ProcessGroupNCCL initializes its communicator lazily on the first CUDA
    collective.  Formal V7 builds large, host-local dataset indices before its
    first source audit, so different ranks can reach lazy initialization after
    substantially different CPU/I/O histories.  Establishing the communicator
    immediately after ``setup_ddp`` keeps that initialization symmetric, then
    the later source audit reuses the already-proven communicator.
    """

    if world <= 1:
        return {
            "enabled": False,
            "backend": "single_process",
            "world_size": int(world),
        }
    if not dist.is_initialized():
        raise RuntimeError("distributed transport probe requires an initialized process group")
    backend = str(dist.get_backend()).lower()
    collective_device = device if backend == "nccl" else torch.device("cpu")
    probe = torch.tensor([int(rank) + 1], dtype=torch.int64, device=collective_device)
    dist.all_reduce(probe, op=dist.ReduceOp.SUM)
    expected = int(world) * (int(world) + 1) // 2
    observed = int(probe.item())
    if observed != expected:
        raise RuntimeError(
            "distributed transport probe sum mismatch: "
            f"observed={observed} expected={expected}"
        )
    dist.barrier()
    return {
        "enabled": True,
        "backend": backend,
        "world_size": int(world),
        "rank_sum": observed,
        "expected_rank_sum": expected,
        "device_type": collective_device.type,
    }


def _sampler_scope(cfg: dict, world: int, rank: int, local: int) -> tuple[int, int, str]:
    if bool((cfg.get("data") or {}).get("node_sharded_window_cache", False)):
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE") or torch.cuda.device_count() or 1)
        return local_world, int(local), "local-node"
    return int(world), int(rank), "global"


def _legacy_build_model(cfg: dict) -> JointWorldModel:
    sc = StateConfig(**cfg["model"]["state"])
    ac = ActionConfig(**cfg["model"]["action"])
    dc = DualConfig(state=sc, action=ac,
                    xattn_layers_state=tuple(cfg["model"]["xattn_layers_state"]),
                    xattn_n_heads=cfg["model"]["xattn_n_heads"])
    jc = JointConfig(
        dual=dc,
        enable_multiview_fuser=cfg["model"].get("enable_multiview_fuser", False),
        multiview_heads=cfg["model"].get("multiview_heads", 16),
        multiview_dropout=cfg["model"].get("multiview_dropout", 0.0),
        multiview_use_camera_pose=cfg["model"].get("multiview_use_camera_pose", True),
        multiview_pose_dim=cfg["model"].get("multiview_pose_dim", 16),
        enable_token_codec=cfg["model"].get("enable_token_codec", False),
        token_codec_latent_dim=cfg["model"].get("token_codec_latent_dim", 384),
        token_codec_checkpoint=cfg["model"].get("token_codec_checkpoint"),
        action_proj_hidden=cfg["model"]["action_proj_hidden"],
        action_proj_layers=cfg["model"]["action_proj_layers"],
        geom_hidden=cfg["model"]["geom_hidden"],
        geom_upsample_mode=cfg["model"].get("geom_upsample_mode", "transpose"),
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
        enable_future_value=cfg["model"].get("enable_future_value", False),
        future_value_hidden=cfg["model"].get("future_value_hidden", 256),
        future_value_layers=cfg["model"].get("future_value_layers", 2),
        future_value_heads=cfg["model"].get("future_value_heads", 4),
        future_value_task_dim=cfg["model"].get("future_value_task_dim"),
        future_value_max_horizon=cfg["model"].get("future_value_max_horizon", 32),
        enable_action_proposer=cfg["model"].get("enable_action_proposer", False),
        proposer_hidden=cfg["model"].get("proposer_hidden", 512),
        proposer_layers=cfg["model"].get("proposer_layers", 3),
        proposer_candidates=cfg["model"].get("proposer_candidates", 4),
        proposer_horizon=cfg["model"].get("proposer_horizon"),
        proposer_task_dim=cfg["model"].get("proposer_task_dim"),
        proposer_dropout=cfg["model"].get("proposer_dropout", 0.0),
        proposer_use_task=cfg["model"].get("proposer_use_task", True),
        enable_action_policy=cfg["model"].get("enable_action_policy", False),
        policy_hidden=cfg["model"].get("policy_hidden", 768),
        policy_layers=cfg["model"].get("policy_layers", 6),
        policy_heads=cfg["model"].get("policy_heads", 8),
        policy_chunk_layers=cfg["model"].get("policy_chunk_layers", 2),
        policy_horizon=cfg["model"].get("policy_horizon"),
        policy_task_dim=cfg["model"].get("policy_task_dim"),
        policy_max_context=cfg["model"].get("policy_max_context"),
        policy_dropout=cfg["model"].get("policy_dropout", 0.1),
        policy_use_task=cfg["model"].get("policy_use_task", True),
        policy_patch_pool=cfg["model"].get("policy_patch_pool", "mean"),
        policy_max_spatial_tokens=cfg["model"].get("policy_max_spatial_tokens", 64),
        policy_context_source=cfg["model"].get("policy_context_source", "input"),
        policy_core_action_cond=cfg["model"].get("policy_core_action_cond", "same"),
        policy_use_context_rgb=cfg["model"].get("policy_use_context_rgb", False),
        policy_rgb_spatial_tokens=cfg["model"].get("policy_rgb_spatial_tokens", 64),
        policy_lowdim_dim=cfg["model"].get("policy_lowdim_dim", 0),
        policy_object_state_dim=cfg["model"].get("policy_object_state_dim", 0),
        policy_plan_state_dim=cfg["model"].get("policy_plan_state_dim", 0),
        policy_action_history_len=cfg["model"].get("policy_action_history_len", 0),
        policy_action_history_dim=cfg["model"].get("policy_action_history_dim", 7),
        policy_action_history_as_token=cfg["model"].get("policy_action_history_as_token", True),
        policy_grip_history_adapter=cfg["model"].get("policy_grip_history_adapter", False),
        policy_grip_history_hidden=cfg["model"].get("policy_grip_history_hidden", 128),
        policy_grip_history_zero_init=cfg["model"].get("policy_grip_history_zero_init", True),
        policy_enable_grip_delta_head=cfg["model"].get("policy_enable_grip_delta_head", False),
        policy_grip_delta_hidden=cfg["model"].get("policy_grip_delta_hidden", 256),
        policy_grip_delta_zero_init=cfg["model"].get("policy_grip_delta_zero_init", True),
        policy_grip_delta_use_composed_action_cond=cfg["model"].get("policy_grip_delta_use_composed_action_cond", False),
        policy_grip_delta_soft_compose_action_cond=cfg["model"].get("policy_grip_delta_soft_compose_action_cond", False),
        policy_grip_delta_straight_through_action_cond=cfg["model"].get(
            "policy_grip_delta_straight_through_action_cond", False
        ),
        policy_grip_owner=cfg["model"].get("policy_grip_owner", "auto"),
        policy_use_progress=cfg["model"].get("policy_use_progress", False),
        policy_progress_dim=cfg["model"].get("policy_progress_dim", 1),
        policy_progress_mode=cfg["model"].get("policy_progress_mode", "token"),
        policy_enable_local_residual=cfg["model"].get("policy_enable_local_residual", False),
        policy_local_hidden=cfg["model"].get("policy_local_hidden", 256),
        policy_local_layers=cfg["model"].get("policy_local_layers", 2),
        policy_local_residual_scale=cfg["model"].get("policy_local_residual_scale", 1.0),
        policy_local_use_lowdim=cfg["model"].get("policy_local_use_lowdim", True),
        policy_local_use_plan_state=cfg["model"].get("policy_local_use_plan_state", True),
        policy_local_use_progress=cfg["model"].get("policy_local_use_progress", True),
        policy_local_use_action_history=cfg["model"].get("policy_local_use_action_history", True),
        policy_enable_waypoint_head=cfg["model"].get("policy_enable_waypoint_head", False),
        policy_waypoint_hidden=cfg["model"].get("policy_waypoint_hidden", 256),
        policy_waypoint_layers=cfg["model"].get("policy_waypoint_layers", 2),
        policy_waypoint_num_stages=cfg["model"].get("policy_waypoint_num_stages", 4),
        policy_waypoint_stage_dim=cfg["model"].get("policy_waypoint_stage_dim", 4),
        policy_waypoint_active_stages=tuple(cfg["model"].get("policy_waypoint_active_stages", ())),
        policy_waypoint_residual_scale=cfg["model"].get("policy_waypoint_residual_scale", 1.0),
        policy_waypoint_mode=cfg["model"].get("policy_waypoint_mode", "residual"),
        policy_waypoint_use_summary=cfg["model"].get("policy_waypoint_use_summary", True),
        policy_waypoint_use_lowdim=cfg["model"].get("policy_waypoint_use_lowdim", True),
        policy_waypoint_use_plan_state=cfg["model"].get("policy_waypoint_use_plan_state", True),
        policy_waypoint_use_progress=cfg["model"].get("policy_waypoint_use_progress", True),
        policy_waypoint_use_action_history=cfg["model"].get("policy_waypoint_use_action_history", True),
        policy_enable_prior=cfg["model"].get("policy_enable_prior", False),
        policy_prior_chunk_layers=cfg["model"].get("policy_prior_chunk_layers", 1),
        policy_action_add_trunk=cfg["model"].get("policy_action_add_trunk", True),
        policy_zero_init_output=cfg["model"].get("policy_zero_init_output", False),
        policy_enable_flow_head=cfg["model"].get("policy_enable_flow_head", False),
        policy_flow_use_as_policy=cfg["model"].get("policy_flow_use_as_policy", False),
        policy_flow_layers=cfg["model"].get("policy_flow_layers", 2),
        policy_flow_hidden=cfg["model"].get("policy_flow_hidden", cfg["model"].get("policy_hidden", 768)),
        policy_flow_action_dim=cfg["model"].get("policy_flow_action_dim", 7),
        policy_flow_default_steps=cfg["model"].get("policy_flow_default_steps", 8),
        policy_flow_noise_scale=cfg["model"].get("policy_flow_noise_scale", 1.0),
        policy_flow_zero_init_output=cfg["model"].get("policy_flow_zero_init_output", False),
        policy_head_type=cfg["model"].get("policy_head_type", "native"),
        policy_oft_max_horizon=cfg["model"].get("policy_oft_max_horizon", 16),
        policy_oft_query_layers=cfg["model"].get("policy_oft_query_layers", 2),
        policy_oft_mlp_hidden=cfg["model"].get("policy_oft_mlp_hidden", 0),
        policy_oft_adapter_name=cfg["model"].get("policy_oft_adapter_name", "canonical_7d"),
        policy_oft_action_dim=cfg["model"].get("policy_oft_action_dim", 7),
        policy_oft_grip_indices=tuple(cfg["model"].get("policy_oft_grip_indices", (6,))),
        policy_oft_adapters=tuple(cfg["model"].get("policy_oft_adapters", ())),
        enable_bridging=cfg["model"].get("enable_bridging", True),
        enable_aux_idm=cfg["model"].get("enable_aux_idm", False),
        aux_idm_hidden=cfg["model"].get("aux_idm_hidden", 1024),
        aux_idm_layers=cfg["model"].get("aux_idm_layers", 3),
        enable_world_prior=cfg["model"].get("enable_world_prior", False),
        world_prior_hidden=cfg["model"].get("world_prior_hidden", cfg["model"].get("state", {}).get("hidden", 768)),
        world_prior_layers=cfg["model"].get("world_prior_layers", 4),
        world_prior_heads=cfg["model"].get("world_prior_heads", cfg["model"].get("state", {}).get("n_heads", 8)),
        world_prior_mlp_mult=cfg["model"].get("world_prior_mlp_mult", 4),
        world_prior_dropout=cfg["model"].get("world_prior_dropout", 0.0),
        world_prior_task_dim=cfg["model"].get("world_prior_task_dim"),
        world_prior_action_dim=cfg["model"].get("world_prior_action_dim", 7),
        world_prior_use_context=cfg["model"].get("world_prior_use_context", True),
        world_prior_use_action=cfg["model"].get("world_prior_use_action", True),
        world_prior_predict_initial=cfg["model"].get("world_prior_predict_initial", True),
    )
    return JointWorldModel(jc)


def build_model(cfg: dict) -> JointWorldModel:
    return build_joint_world_model(cfg["model"])


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


def _window_config(data_cfg: dict, model_cfg: dict | None = None) -> WindowConfig:
    model_cfg = model_cfg or {}
    policy_state_default = any(
        int(model_cfg.get(k, 0) or 0) > 0
        for k in ("policy_lowdim_dim", "policy_object_state_dim", "policy_plan_state_dim", "policy_action_history_len")
    ) or bool(model_cfg.get("policy_use_progress", False))
    shard_indices = data_cfg.get("window_geom_shard_indices")
    shard_roots = data_cfg.get("window_geom_shard_roots")
    canonical_action_enabled = bool(data_cfg.get("canonical_action_enabled", False))
    canonical_action_sources: tuple[str, ...] = ()
    canonical_action_stats_by_source: dict[str, Path] | None = None
    canonical_action_cache_manifest: Path | None = None
    canonical_action_cache_manifest_sha256: str | None = None
    action_contract_evidence_sha256: str | None = None
    if canonical_action_enabled:
        # These are deliberately source-local, content-addressed inputs.  Do
        # not alias the legacy pooled action_stats/action_contract paths: doing
        # so can silently mix coordinate frames or replay the 98 MiB raw DROID
        # index in every worker/rank.
        required_keys = (
            "canonical_action_sources",
            "canonical_action_stats_by_source",
            "canonical_action_cache_manifest",
            "canonical_action_cache_manifest_sha256",
            "action_contract_evidence_sha256",
        )
        missing = [key for key in required_keys if not data_cfg.get(key)]
        if missing:
            raise ValueError(
                "canonical OXE action config is missing explicit required keys: "
                + ", ".join(missing)
            )
        if data_cfg.get("action_stats") is not None:
            raise ValueError(
                "canonical OXE action config forbids legacy pooled action_stats"
            )
        raw_sources = data_cfg["canonical_action_sources"]
        if not isinstance(raw_sources, (list, tuple)):
            raise ValueError("canonical_action_sources must be an explicit sequence")
        canonical_action_sources = tuple(str(value) for value in raw_sources)
        raw_stats = data_cfg["canonical_action_stats_by_source"]
        if not isinstance(raw_stats, dict):
            raise ValueError("canonical_action_stats_by_source must be a mapping")
        canonical_action_stats_by_source = {
            str(source): Path(path) for source, path in raw_stats.items()
        }
        canonical_action_cache_manifest = Path(
            data_cfg["canonical_action_cache_manifest"]
        )
        canonical_action_cache_manifest_sha256 = str(
            data_cfg["canonical_action_cache_manifest_sha256"]
        )
        action_contract_evidence_sha256 = str(
            data_cfg["action_contract_evidence_sha256"]
        )
    return WindowConfig(T=data_cfg["T"], k=data_cfg["k"],
                        stride=data_cfg["stride"],
                        cache_root=Path(data_cfg["cache_root"]),
                        tokens_subdir=data_cfg.get("tokens_subdir", "vggt_pooled"),
                        action_stats=Path(data_cfg["action_stats"])
                        if data_cfg.get("action_stats") else None,
                        manifest_path=Path(data_cfg["manifest"])
                        if data_cfg.get("manifest") else None,
                        action_contract_path=Path(data_cfg["action_contract_path"])
                        if data_cfg.get("action_contract_path") else None,
                        action_contract_evidence_sha256=action_contract_evidence_sha256,
                        require_action_contract=bool(
                            data_cfg.get("require_action_contract", False)
                        ),
                        default_action_frame_offset=int(
                            data_cfg.get("default_action_frame_offset", 0)
                        ),
                        droid_cache_index_path=Path(data_cfg["droid_cache_index"])
                        if data_cfg.get("droid_cache_index") else None,
                        require_task_emb=bool(data_cfg.get("require_task_emb", False)),
                        load_task_text=bool(data_cfg.get("load_task_text", False)),
                        load_rgb=bool(data_cfg.get("load_rgb", True)),
                        load_geom=bool(data_cfg.get("load_geom", True)),
                        load_state_tgt=bool(data_cfg.get("load_state_tgt", True)),
                        load_geom_extra=bool(data_cfg.get("load_geom_extra", False)),
                        require_geom_extra=bool(data_cfg.get("require_geom_extra", False)),
                        window_geom_subdir=data_cfg.get("window_geom_subdir", "vggt_window_geom_p64"),
                        window_geom_cache_root=Path(data_cfg["window_geom_cache_root"])
                        if data_cfg.get("window_geom_cache_root") else None,
                        window_geom_shard_index=Path(data_cfg["window_geom_shard_index"])
                        if data_cfg.get("window_geom_shard_index") else None,
                        window_geom_shard_root=Path(data_cfg["window_geom_shard_root"])
                        if data_cfg.get("window_geom_shard_root") else None,
                        window_geom_shard_indices=tuple(Path(p) for p in shard_indices)
                        if shard_indices else None,
                        window_geom_shard_roots=tuple(Path(p) if p is not None else None for p in shard_roots)
                        if shard_roots else None,
                        use_window_tokens=bool(data_cfg.get("use_window_tokens", False)),
                        causal_dual_view_required=bool(
                            data_cfg.get("causal_dual_view_required", False)
                        ),
                        causal_dual_view_representation=data_cfg.get(
                            "causal_dual_view_representation"
                        ),
                        max_windows_per_episode=int(data_cfg.get("max_windows_per_episode", 0) or 0),
                        trust_window_geom_cache=bool(data_cfg.get("trust_window_geom_cache", False)),
                        trusted_manifest_fast_init=bool(
                            data_cfg.get("trusted_manifest_fast_init", False)
                        ),
                        trusted_manifest_sha256=(
                            str(data_cfg["trusted_manifest_sha256"])
                            if data_cfg.get("trusted_manifest_sha256")
                            else None
                        ),
                        allow_pseudo_progress_targets=bool(data_cfg.get("allow_pseudo_progress_targets", False)),
                        require_progress=bool(data_cfg.get("require_progress", bool(model_cfg.get("policy_use_progress", False)))),
                        load_policy_state=bool(data_cfg.get("load_policy_state", policy_state_default)),
                        require_policy_state=bool(data_cfg.get("require_policy_state", False)),
                        strict_policy_state_prescan=bool(data_cfg.get("strict_policy_state_prescan", False)),
                        policy_lowdim_dim=int(data_cfg.get("policy_lowdim_dim", model_cfg.get("policy_lowdim_dim", 0)) or 0),
                        policy_object_state_dim=int(data_cfg.get("policy_object_state_dim", model_cfg.get("policy_object_state_dim", 0)) or 0),
                        policy_plan_state_dim=int(data_cfg.get("policy_plan_state_dim", model_cfg.get("policy_plan_state_dim", 0)) or 0),
                        policy_action_history_len=int(data_cfg.get("policy_action_history_len", model_cfg.get("policy_action_history_len", 0)) or 0),
                        policy_action_history_dim=int(data_cfg.get("policy_action_history_dim", model_cfg.get("policy_action_history_dim", 7)) or 7),
                        canonical_action_enabled=canonical_action_enabled,
                        canonical_action_sources=canonical_action_sources,
                        canonical_action_stats_by_source=canonical_action_stats_by_source,
                        canonical_action_cache_manifest=canonical_action_cache_manifest,
                        canonical_action_cache_manifest_sha256=canonical_action_cache_manifest_sha256)


def _sample_record(dataset, sample_idx: int):
    if isinstance(dataset, Subset):
        return _sample_record(dataset.dataset, int(dataset.indices[sample_idx]))
    if hasattr(dataset, "index") and hasattr(dataset, "records"):
        record_idx, _start = dataset.index[int(sample_idx)]
        return dataset.records[record_idx]
    if hasattr(dataset, "records"):
        return dataset.records[int(sample_idx)]
    return None


def _record_value(record, key: str, default=None):
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _base_dataset_sample(dataset, sample_idx: int):
    if isinstance(dataset, Subset):
        return _base_dataset_sample(dataset.dataset, int(dataset.indices[sample_idx]))
    return dataset, int(sample_idx)


def _record_grip_series(base_ds, record_idx: int, rec, grip_cache: dict[int, np.ndarray]) -> np.ndarray:
    cached = grip_cache.get(int(record_idx))
    if cached is not None:
        return cached
    if isinstance(rec, dict) and rec.get("path"):
        path = Path(rec["path"])
        with np.load(path, allow_pickle=False) as archive:
            actions = np.asarray(archive["actions"], dtype=np.float32)
    else:
        cid = str(rec.clip_id).replace("/", "__")
        path = Path(base_ds.cfg.cache_root) / "actions" / f"{cid}.npy"
        actions = np.load(path, mmap_mode="r")
    try:
        if actions.ndim < 2 or actions.shape[1] <= 6:
            raise ValueError(f"action cache {path} has shape {tuple(actions.shape)}, expected [..., >=7]")
        grip = np.asarray(actions[:, 6] > 0.5, dtype=np.bool_).copy()
    finally:
        mmap_obj = getattr(actions, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()
        del actions
    grip_cache[int(record_idx)] = grip
    return grip


def _sample_grip_window_stats(dataset, sample_idx: int, grip_cache: dict[int, np.ndarray]):
    base_ds, base_idx = _base_dataset_sample(dataset, sample_idx)
    if not (hasattr(base_ds, "index") and hasattr(base_ds, "records") and hasattr(base_ds, "cfg")):
        return None
    record_idx, start = base_ds.index[int(base_idx)]
    rec = base_ds.records[int(record_idx)]
    grip_series = _record_grip_series(base_ds, int(record_idx), rec, grip_cache)
    T = int(base_ds.cfg.T)
    k = int(base_ds.cfg.k)
    # V7 compact actions use action[t] for frame t -> t+1, so the first
    # factual future transition starts at the final context frame (T-1).
    action_offset = -1 if isinstance(rec, dict) and rec.get("schema") == "wm3d_v7_compact_geom_v3" else 0
    start_i = int(start) + T + action_offset
    grip = grip_series[start_i : start_i + k]
    if grip.size <= 0:
        return None
    prev = np.concatenate([grip[:1], grip[:-1]])
    transition = grip != prev
    transition_up = transition & grip
    transition_down = transition & (~grip)
    prev_boundary_grip = bool(grip_series[max(0, start_i - 1)])
    first_grip = bool(grip[0])
    boundary = prev_boundary_grip != first_grip
    boundary_up = bool(boundary and first_grip)
    boundary_down = bool(boundary and not first_grip)
    transition_indices = np.flatnonzero(transition)
    first_transition_up = bool(transition_up[transition_indices[0]]) if transition_indices.size else False
    if boundary_up:
        primary_partition = "boundary_up"
    elif boundary_down:
        primary_partition = "boundary_down"
    elif transition_indices.size:
        primary_partition = "transition_up" if first_transition_up else "transition_down"
    else:
        primary_partition = "positive_noevent" if first_grip else "negative_noevent"
    return {
        "has_positive": bool(grip.any()),
        "first_positive": bool(grip[0]),
        "positive_frac": float(grip.mean()),
        "has_transition": bool(transition.any()),
        "transition_frac": float(transition.mean()),
        "has_transition_up": bool(transition_up.any()),
        "has_transition_down": bool(transition_down.any()),
        "transition_up_frac": float(transition_up.mean()),
        "transition_down_frac": float(transition_down.mean()),
        "has_boundary_transition": boundary,
        "has_boundary_up": boundary_up,
        "has_boundary_down": boundary_down,
        "primary_partition": primary_partition,
    }


def _sampler_cache_path(template: str | None, dataset_len: int) -> Path | None:
    if not template:
        return None
    host = socket.gethostname().split(".")[0]
    path = str(template).format(hostname=host, host=host, dataset_len=int(dataset_len))
    return Path(path)


def _cache_safe_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _cache_safe_value(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_cache_safe_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


_SAMPLER_FINGERPRINT_SCHEMA = 2


def _sampler_metadata_uses_grip_source(sampler_cfg: dict) -> bool:
    return bool(sampler_cfg.get("action_grip_weighting", False)) or bool(
        sampler_cfg.get("grip_event_balance", sampler_cfg.get("event_balance", False))
    )


def _action_source_provenance_digest(base_ds, records, sampler_cfg: dict) -> str:
    """Hash action-source identity without decoding every action array.

    Grip metadata depends on the raw ``actions/*.npy`` contents.  Device/inode,
    size, mtime and ctime make in-place edits and replacements invalidate the
    cache while keeping cache hits to one cheap stat per episode rather than a
    full action-array scan.
    """
    if not _sampler_metadata_uses_grip_source(sampler_cfg):
        return "unused"
    cfg_obj = getattr(base_ds, "cfg", None)
    cache_root = getattr(cfg_obj, "cache_root", None)
    if cache_root is None:
        return "unavailable:no-cache-root"
    actions_root = Path(cache_root) / "actions"
    digest = hashlib.sha256()
    seen: set[str] = set()
    for rec in records:
        clip_id = str(getattr(rec, "clip_id", ""))
        source_path = actions_root / f"{clip_id.replace('/', '__')}.npy"
        source_key = str(source_path)
        if source_key in seen:
            continue
        seen.add(source_key)
        digest.update(source_key.encode("utf-8"))
        digest.update(b"\0")
        try:
            stat = source_path.stat()
        except FileNotFoundError:
            digest.update(b"missing\0")
            continue
        provenance = (
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
        )
        digest.update(repr(provenance).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _sample_weight_cache_fingerprint(dataset, sampler_cfg: dict, expected_len: int) -> str:
    base_ds, _ = _base_dataset_sample(dataset, 0) if expected_len > 0 else (dataset, 0)
    cfg_obj = getattr(base_ds, "cfg", None)
    ds_cfg = {}
    if cfg_obj is not None:
        ds_cfg = {
            "cache_root": str(getattr(cfg_obj, "cache_root", "")),
            "T": int(getattr(cfg_obj, "T", 0) or 0),
            "k": int(getattr(cfg_obj, "k", 0) or 0),
            "stride": int(getattr(cfg_obj, "stride", 0) or 0),
            "action_horizon": int(getattr(cfg_obj, "action_horizon", 0) or 0),
            "policy_action_history_len": int(getattr(cfg_obj, "policy_action_history_len", 0) or 0),
        }
    records = getattr(base_ds, "records", None) or []
    record_digest = hashlib.sha256()
    for rec in records:
        record_digest.update(
            str(_record_value(rec, "clip_id", _record_value(rec, "clip_hash", ""))).encode("utf-8")
        )
        record_digest.update(b"\0")
        record_digest.update(
            str(_record_value(rec, "dataset", _record_value(rec, "v7_source", ""))).encode("utf-8")
        )
        record_digest.update(b"\0")
        repeat_weight = float(_record_value(rec, "repeat_weight", 1.0) or 1.0)
        record_digest.update(format(repeat_weight, ".17g").encode("ascii"))
        record_digest.update(b"\0")
    index_digest = hashlib.sha256()
    if isinstance(dataset, Subset):
        subset_indices = np.asarray(dataset.indices, dtype=np.int64)
        index_digest.update(subset_indices.tobytes())
    window_index = getattr(base_ds, "index", None)
    if window_index is not None:
        try:
            index_array = np.asarray(window_index, dtype=np.int64)
            index_digest.update(str(tuple(index_array.shape)).encode("ascii"))
            index_digest.update(index_array.tobytes())
        except (TypeError, ValueError):
            for item in window_index:
                index_digest.update(repr(item).encode("utf-8"))
                index_digest.update(b"\0")
    normalized_sampler_cfg = {
        str(k): _cache_safe_value(v)
        for k, v in sorted(sampler_cfg.items(), key=lambda kv: str(kv[0]))
        if k not in {
            "cache_path",
            "cache_wait_seconds",
            "cache_wait_timeout_seconds",
            "cache_lock_stale_seconds",
            "cache_fail_fast",
        }
    }
    payload = {
        "fingerprint_schema": _SAMPLER_FINGERPRINT_SCHEMA,
        "dataset_len": int(expected_len),
        "dataset_type": type(base_ds).__name__,
        "window_cfg": ds_cfg,
        "record_count": len(records),
        "record_digest": record_digest.hexdigest(),
        "action_source_provenance_digest": _action_source_provenance_digest(
            base_ds,
            records,
            sampler_cfg,
        ),
        "index_digest": index_digest.hexdigest(),
        "sampler_cfg": normalized_sampler_cfg,
    }
    text = yaml.safe_dump(payload, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_SAMPLER_METADATA_SCHEMA = 3
_SAMPLER_METADATA_MEMORY: dict[str, tuple[torch.Tensor, dict[str, torch.Tensor]]] = {}


def _buckets_from_primary_labels(labels: np.ndarray | torch.Tensor) -> dict[str, torch.Tensor]:
    label_tensor = torch.as_tensor(labels, dtype=torch.long).reshape(-1).cpu()
    buckets = {name: torch.empty(0, dtype=torch.long) for name in _GRIP_EVENT_BUCKET_NAMES}
    buckets["all"] = torch.arange(label_tensor.numel(), dtype=torch.long)
    for label_id, name in enumerate(_GRIP_PRIMARY_BUCKET_NAMES):
        buckets[name] = torch.nonzero(label_tensor == label_id, as_tuple=False).reshape(-1)
    buckets["boundary"] = torch.cat([buckets["boundary_up"], buckets["boundary_down"]])
    buckets["transition"] = torch.cat([buckets["transition_up"], buckets["transition_down"]])
    buckets["positive"] = torch.cat(
        [buckets["positive_noevent"], buckets["transition_up"], buckets["boundary_up"]]
    )
    buckets["negative"] = torch.cat(
        [buckets["negative_noevent"], buckets["transition_down"], buckets["boundary_down"]]
    )
    return buckets


def _primary_labels_from_buckets(buckets: dict[str, torch.Tensor], expected_len: int) -> torch.Tensor:
    labels = torch.full((int(expected_len),), -1, dtype=torch.int16)
    for label_id, name in enumerate(_GRIP_PRIMARY_BUCKET_NAMES):
        indices = buckets.get(name, torch.empty(0, dtype=torch.long)).long().cpu()
        if indices.numel() == 0:
            continue
        if bool((labels.index_select(0, indices) >= 0).any()):
            raise ValueError(f"sampler primary partition overlap detected at {name}")
        labels.index_fill_(0, indices, int(label_id))
    return labels


def _bucket_membership_from_buckets(
    buckets: dict[str, torch.Tensor],
    expected_len: int,
) -> torch.Tensor:
    membership = torch.zeros(int(expected_len), dtype=torch.int32)
    for bit, name in enumerate(_GRIP_EVENT_BUCKET_NAMES):
        indices = buckets.get(name, torch.empty(0, dtype=torch.long)).long().cpu()
        if indices.numel() > 0:
            membership[indices] |= 1 << bit
    return membership


def _buckets_from_membership(membership: np.ndarray | torch.Tensor) -> dict[str, torch.Tensor]:
    membership_tensor = torch.as_tensor(membership, dtype=torch.int32).reshape(-1).cpu()
    return {
        name: torch.nonzero((membership_tensor & (1 << bit)) != 0, as_tuple=False).reshape(-1)
        for bit, name in enumerate(_GRIP_EVENT_BUCKET_NAMES)
    }


def _load_sampler_metadata_cache(
    path: Path,
    expected_len: int,
    expected_fingerprint: str | None = None,
    *,
    fail_fast: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "schema",
                "weights",
                "primary_partition",
                "bucket_membership",
                "dataset_len",
                "fingerprint",
            }
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"legacy/incomplete metadata cache missing fields: {missing}")
            schema = int(np.asarray(data["schema"]).reshape(-1)[0])
            weights = np.asarray(data["weights"], dtype=np.float64)
            primary = np.asarray(data["primary_partition"], dtype=np.int16)
            membership = np.asarray(data["bucket_membership"], dtype=np.int32)
            cached_len = int(np.asarray(data["dataset_len"]).reshape(-1)[0])
            cached_fingerprint = str(np.asarray(data["fingerprint"]).reshape(-1)[0])
        if schema != _SAMPLER_METADATA_SCHEMA:
            raise ValueError(f"metadata schema mismatch cached={schema} expected={_SAMPLER_METADATA_SCHEMA}")
        if (
            weights.shape != (int(expected_len),)
            or primary.shape != (int(expected_len),)
            or membership.shape != (int(expected_len),)
            or cached_len != int(expected_len)
        ):
            raise ValueError(
                "metadata length mismatch "
                f"weights={tuple(weights.shape)} primary={tuple(primary.shape)} "
                f"membership={tuple(membership.shape)} "
                f"cached={cached_len} expected={expected_len}"
            )
        if expected_fingerprint is not None and cached_fingerprint != expected_fingerprint:
            print(f"[sampler] ignoring metadata cache {path}: fingerprint mismatch", flush=True)
            return None
        if np.any((primary < -1) | (primary >= len(_GRIP_PRIMARY_BUCKET_NAMES))):
            raise ValueError("metadata cache contains invalid primary partition labels")
        out = torch.as_tensor(weights.copy(), dtype=torch.double)
        if not torch.isfinite(out).all() or float(out.sum()) <= 0.0:
            raise ValueError("metadata cache contains non-finite or zero weights")
        buckets = _buckets_from_membership(membership.copy())
        cached_primary = torch.as_tensor(primary.copy(), dtype=torch.int16)
        membership_primary = _primary_labels_from_buckets(buckets, expected_len)
        if not torch.equal(membership_primary, cached_primary):
            mismatch = torch.nonzero(
                membership_primary != cached_primary,
                as_tuple=False,
            ).reshape(-1)
            preview = mismatch[:8].tolist()
            raise ValueError(
                "metadata cache primary_partition disagrees with bucket_membership "
                f"at {int(mismatch.numel())} samples; first_indices={preview}"
            )
        if expected_fingerprint is not None:
            _SAMPLER_METADATA_MEMORY[expected_fingerprint] = (out, buckets)
        return out, buckets
    except Exception as exc:
        print(f"[sampler] failed to load metadata cache {path}: {type(exc).__name__}: {exc}", flush=True)
        if fail_fast:
            raise
        return None


def _load_sample_weight_cache(
    path: Path,
    expected_len: int,
    expected_fingerprint: str | None = None,
    *,
    fail_fast: bool = True,
) -> torch.Tensor | None:
    metadata = _load_sampler_metadata_cache(
        path,
        expected_len,
        expected_fingerprint,
        fail_fast=fail_fast,
    )
    return metadata[0] if metadata is not None else None


def _compute_dataset_sample_weights_uncached(dataset, sampler_cfg: dict) -> torch.Tensor | None:
    n = len(dataset)
    if n <= 0:
        return None
    dataset_weights = {str(k): float(v) for k, v in (sampler_cfg.get("dataset_weights") or {}).items()}
    balance_by_dataset = bool(sampler_cfg.get("balance_by_dataset", True))
    records = [_sample_record(dataset, i) for i in range(n)]
    counts: dict[str, int] = {}
    for rec in records:
        name = (
            str(_record_value(rec, "dataset", _record_value(rec, "v7_source", "unknown")))
            if rec is not None
            else "unknown"
        )
        counts[name] = counts.get(name, 0) + 1
    weights = []
    for rec in records:
        name = (
            str(_record_value(rec, "dataset", _record_value(rec, "v7_source", "unknown")))
            if rec is not None
            else "unknown"
        )
        repeat_weight = float(_record_value(rec, "repeat_weight", 1.0) or 1.0) if rec is not None else 1.0
        w = repeat_weight * dataset_weights.get(name, 1.0)
        if balance_by_dataset:
            w /= max(1, counts.get(name, 1))
        weights.append(max(0.0, w))
    mixed_outcome_boost = float(
        sampler_cfg.get("same_root_mixed_outcome_boost", 0.0) or 0.0
    )
    negative_fraction_boost = float(
        sampler_cfg.get("same_root_negative_branch_fraction_boost", 0.0) or 0.0
    )
    outcome_weight_cap = float(
        sampler_cfg.get("same_root_outcome_weight_cap", 0.0) or 0.0
    )
    if mixed_outcome_boost or negative_fraction_boost:
        for index, rec in enumerate(records):
            branches = max(1, int(_record_value(rec, "branches", 1) or 1))
            negative = max(
                0,
                int(_record_value(rec, "terminal_negative_branches", 0) or 0),
            )
            multiplier = (
                1.0
                + mixed_outcome_boost
                * float(bool(_record_value(rec, "mixed_terminal_outcomes", False)))
                + negative_fraction_boost * min(1.0, negative / branches)
            )
            if outcome_weight_cap > 0:
                multiplier = min(multiplier, outcome_weight_cap)
            weights[index] = max(0.0, weights[index] * multiplier)
    grip_stats = None
    need_grip_stats = bool(sampler_cfg.get("action_grip_weighting", False)) or bool(
        sampler_cfg.get("grip_event_balance", sampler_cfg.get("event_balance", False))
    )
    if need_grip_stats:
        grip_cache: dict[int, np.ndarray] = {}
        grip_stats = [_sample_grip_window_stats(dataset, i, grip_cache) for i in range(n)]
    if bool(sampler_cfg.get("action_grip_weighting", False)):
        pos_boost = float(sampler_cfg.get("grip_positive_boost", 0.0) or 0.0)
        first_pos_boost = float(sampler_cfg.get("grip_first_positive_boost", 0.0) or 0.0)
        transition_boost = float(sampler_cfg.get("grip_transition_boost", 0.0) or 0.0)
        boundary_boost = float(sampler_cfg.get("grip_boundary_transition_boost", 0.0) or 0.0)
        pos_frac_boost = float(sampler_cfg.get("grip_positive_fraction_boost", 0.0) or 0.0)
        transition_frac_boost = float(sampler_cfg.get("grip_transition_fraction_boost", 0.0) or 0.0)
        transition_up_boost = float(sampler_cfg.get("grip_transition_up_boost", 0.0) or 0.0)
        transition_down_boost = float(sampler_cfg.get("grip_transition_down_boost", 0.0) or 0.0)
        transition_up_frac_boost = float(sampler_cfg.get("grip_transition_up_fraction_boost", 0.0) or 0.0)
        transition_down_frac_boost = float(sampler_cfg.get("grip_transition_down_fraction_boost", 0.0) or 0.0)
        boundary_up_boost = float(sampler_cfg.get("grip_boundary_up_boost", 0.0) or 0.0)
        boundary_down_boost = float(sampler_cfg.get("grip_boundary_down_boost", 0.0) or 0.0)
        cap = float(sampler_cfg.get("grip_weight_cap", 0.0) or 0.0)
        for i, stats in enumerate(grip_stats or []):
            if stats is None:
                continue
            mult = (
                1.0
                + pos_boost * float(stats["has_positive"])
                + first_pos_boost * float(stats["first_positive"])
                + transition_boost * float(stats["has_transition"])
                + boundary_boost * float(stats["has_boundary_transition"])
                + transition_up_boost * float(stats.get("has_transition_up", False))
                + transition_down_boost * float(stats.get("has_transition_down", False))
                + boundary_up_boost * float(stats.get("has_boundary_up", False))
                + boundary_down_boost * float(stats.get("has_boundary_down", False))
                + pos_frac_boost * float(stats["positive_frac"])
                + transition_frac_boost * float(stats["transition_frac"])
                + transition_up_frac_boost * float(stats.get("transition_up_frac", 0.0))
                + transition_down_frac_boost * float(stats.get("transition_down_frac", 0.0))
            )
            if cap > 0:
                mult = min(mult, cap)
            weights[i] = max(0.0, weights[i] * mult)
    if bool(sampler_cfg.get("grip_event_balance", sampler_cfg.get("event_balance", False))):
        # CE class balancing only helps on events that appear in the batch. This pass
        # raises rare gripper event windows to a target sampling mass so up/down
        # transitions are consistently present without hand-tuning BCE margins.
        base = np.asarray(weights, dtype=np.float64)
        total = float(base.sum())
        if total > 0.0 and grip_stats is not None:
            mult = np.ones(n, dtype=np.float64)
            strict_event_balance = _grip_partition_contract_enabled(sampler_cfg)
            default_target = float(sampler_cfg.get("grip_event_balance_target_fraction", 0.12) or 0.0)
            cap = float(sampler_cfg.get("grip_event_balance_cap", 0.0) or 0.0)
            groups = (
                (
                    "transition_up",
                    np.asarray(
                        [
                            bool(
                                s
                                and (
                                    s.get("primary_partition") == "transition_up"
                                    if strict_event_balance
                                    else s.get("has_transition_up", False)
                                )
                            )
                            for s in grip_stats
                        ],
                        dtype=np.bool_,
                    ),
                    float(sampler_cfg.get("grip_transition_up_target_fraction", default_target) or 0.0),
                ),
                (
                    "transition_down",
                    np.asarray(
                        [
                            bool(
                                s
                                and (
                                    s.get("primary_partition") == "transition_down"
                                    if strict_event_balance
                                    else s.get("has_transition_down", False)
                                )
                            )
                            for s in grip_stats
                        ],
                        dtype=np.bool_,
                    ),
                    float(sampler_cfg.get("grip_transition_down_target_fraction", default_target) or 0.0),
                ),
                (
                    "boundary_up",
                    np.asarray(
                        [
                            bool(
                                s
                                and (
                                    s.get("primary_partition") == "boundary_up"
                                    if strict_event_balance
                                    else s.get("has_boundary_up", False)
                                )
                            )
                            for s in grip_stats
                        ],
                        dtype=np.bool_,
                    ),
                    float(sampler_cfg.get("grip_boundary_up_target_fraction", default_target) or 0.0),
                ),
                (
                    "boundary_down",
                    np.asarray(
                        [
                            bool(
                                s
                                and (
                                    s.get("primary_partition") == "boundary_down"
                                    if strict_event_balance
                                    else s.get("has_boundary_down", False)
                                )
                            )
                            for s in grip_stats
                        ],
                        dtype=np.bool_,
                    ),
                    float(sampler_cfg.get("grip_boundary_down_target_fraction", default_target) or 0.0),
                ),
            )
            event_stats = []
            for name, mask, target in groups:
                target = max(0.0, min(0.95, target))
                event_weight = float(base[mask].sum())
                rest_weight = max(0.0, total - event_weight)
                if event_weight <= 0.0 or rest_weight <= 0.0 or target <= 0.0:
                    event_stats.append((name, int(mask.sum()), 1.0))
                    continue
                factor = target * rest_weight / max(1e-12, (1.0 - target) * event_weight)
                factor = max(1.0, factor)
                if cap > 0.0:
                    factor = min(factor, cap)
                mult[mask] = np.maximum(mult[mask], factor)
                event_stats.append((name, int(mask.sum()), factor))
            weights = (base * mult).tolist()
            print(
                "[sampler] grip_event_balance "
                + " ".join(f"{name}:count={count}:factor={factor:.4g}" for name, count, factor in event_stats),
                flush=True,
            )
    out = torch.as_tensor(weights, dtype=torch.double)
    if not torch.isfinite(out).all() or float(out.sum()) <= 0.0:
        raise ValueError("weighted_sampler produced no positive finite sample weights")
    return out


def _local_sampler_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ.get("LOCAL_RANK", "0") or 0)
    if dist.is_available() and dist.is_initialized():
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE") or torch.cuda.device_count() or 1)
        return int(dist.get_rank()) % max(1, local_world)
    return 0


def _reclaim_stale_cache_lock(lock_dir: Path, stale_seconds: float) -> bool:
    if stale_seconds <= 0.0 or not lock_dir.exists():
        return False
    try:
        age = time.time() - lock_dir.stat().st_mtime
        if age < stale_seconds:
            return False
        lock_dir.rmdir()
        print(f"[sampler] reclaimed stale metadata lock {lock_dir} age_seconds={age:.1f}", flush=True)
        return True
    except (FileNotFoundError, OSError):
        return False


class _CacheLockHeartbeat:
    def __init__(self, lock_dir: Path, stale_seconds: float) -> None:
        self.lock_dir = lock_dir
        self.interval = max(0.05, min(30.0, stale_seconds / 3.0 if stale_seconds > 0.0 else 30.0))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sampler-cache-heartbeat", daemon=True)

    def __enter__(self):
        self._touch()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval * 2.0))

    def _touch(self) -> None:
        try:
            os.utime(self.lock_dir, None)
        except FileNotFoundError:
            pass

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._touch()


def _compute_sampler_metadata_uncached(
    dataset,
    sampler_cfg: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor] | None:
    weights = _compute_dataset_sample_weights_uncached(dataset, sampler_cfg)
    if weights is None:
        return None
    if _sampler_metadata_uses_grip_source(sampler_cfg):
        buckets = _compute_grip_event_buckets_uncached(dataset, sampler_cfg)
        primary = _compute_primary_partition_labels_uncached(dataset)
    else:
        empty = torch.empty(0, dtype=torch.long)
        buckets = {name: empty.clone() for name in _GRIP_EVENT_BUCKET_NAMES}
        buckets["all"] = torch.arange(len(dataset), dtype=torch.long)
        primary = torch.full((len(dataset),), -1, dtype=torch.int16)
    return weights, buckets, primary


def _write_sampler_metadata_cache(
    path: Path,
    weights: torch.Tensor,
    buckets: dict[str, torch.Tensor],
    dataset_len: int,
    fingerprint: str,
    primary_partition: torch.Tensor | None = None,
) -> None:
    primary = (
        _primary_labels_from_buckets(buckets, dataset_len)
        if primary_partition is None
        else primary_partition.to(dtype=torch.int16, device="cpu").reshape(-1)
    )
    if primary.shape != (int(dataset_len),):
        raise ValueError(f"primary partition shape mismatch: {tuple(primary.shape)} expected={(dataset_len,)}")
    bucket_primary = _primary_labels_from_buckets(buckets, dataset_len)
    if not torch.equal(bucket_primary, primary):
        mismatch = torch.nonzero(bucket_primary != primary, as_tuple=False).reshape(-1)
        raise ValueError(
            "sampler metadata primary partition disagrees with bucket membership "
            f"at {int(mismatch.numel())} samples; first_indices={mismatch[:8].tolist()}"
        )
    membership = _bucket_membership_from_buckets(buckets, dataset_len)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    np.savez_compressed(
        tmp,
        schema=np.array([_SAMPLER_METADATA_SCHEMA], dtype=np.int64),
        weights=weights.detach().cpu().numpy(),
        primary_partition=primary.numpy(),
        bucket_membership=membership.numpy(),
        dataset_len=np.array([dataset_len], dtype=np.int64),
        fingerprint=np.array([fingerprint], dtype="<U128"),
    )
    written = tmp if tmp.exists() else Path(str(tmp) + ".npz")
    os.replace(written, path)


def _positive_finite_cache_seconds(value, name: str) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"sampler {name} must be finite and positive; got {value!r}") from exc
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError(f"sampler {name} must be finite and positive; got {seconds!r}")
    return seconds


def build_dataset_sample_weights(dataset, sampler_cfg: dict | bool | None) -> torch.Tensor | None:
    if not sampler_cfg:
        return None
    if sampler_cfg is True:
        sampler_cfg = {"enabled": True}
    if not isinstance(sampler_cfg, dict) or not bool(sampler_cfg.get("enabled", False)):
        return None
    n = len(dataset)
    if n <= 0:
        return None
    cache_path = _sampler_cache_path(sampler_cfg.get("cache_path"), n)
    fingerprint = _sample_weight_cache_fingerprint(dataset, sampler_cfg, n) if cache_path is not None else None
    if cache_path is None:
        return _compute_dataset_sample_weights_uncached(dataset, sampler_cfg)
    assert fingerprint is not None
    wait_seconds = max(
        0.05,
        _positive_finite_cache_seconds(
            sampler_cfg.get("cache_wait_seconds", 2.0),
            "cache_wait_seconds",
        ),
    )
    stale_seconds = _positive_finite_cache_seconds(
        sampler_cfg.get("cache_lock_stale_seconds", 1800.0),
        "cache_lock_stale_seconds",
    )
    default_timeout = max(300.0, stale_seconds * 2.0)
    if not math.isfinite(default_timeout) or default_timeout <= 0.0:
        raise ValueError(
            "derived sampler cache_wait_timeout_seconds must be finite and positive; "
            f"cache_lock_stale_seconds={stale_seconds!r} derived={default_timeout!r}"
        )
    wait_timeout = _positive_finite_cache_seconds(
        sampler_cfg.get("cache_wait_timeout_seconds", default_timeout),
        "cache_wait_timeout_seconds",
    )
    cache_fail_fast = bool(sampler_cfg.get("cache_fail_fast", True))
    cached_metadata = _load_sampler_metadata_cache(
        cache_path,
        n,
        fingerprint,
        fail_fast=cache_fail_fast,
    )
    if cached_metadata is not None:
        print(f"[sampler] loaded sample weights cache {cache_path}", flush=True)
        return cached_metadata[0]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = Path(str(cache_path) + ".lock")
    wait_started = time.monotonic()
    if _local_sampler_rank() != 0:
        while True:
            cached_metadata = _load_sampler_metadata_cache(
                cache_path,
                n,
                fingerprint,
                fail_fast=cache_fail_fast,
            )
            if cached_metadata is not None:
                print(f"[sampler] local rank {_local_sampler_rank()} loaded metadata cache {cache_path}", flush=True)
                return cached_metadata[0]
            _reclaim_stale_cache_lock(lock_dir, stale_seconds)
            if time.monotonic() - wait_started >= wait_timeout:
                raise TimeoutError(f"timed out waiting for local rank 0 metadata producer: {cache_path}")
            time.sleep(wait_seconds)
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            cached_metadata = _load_sampler_metadata_cache(
                cache_path,
                n,
                fingerprint,
                fail_fast=cache_fail_fast,
            )
            if cached_metadata is not None:
                print(f"[sampler] loaded sample weights cache {cache_path}", flush=True)
                return cached_metadata[0]
            if _reclaim_stale_cache_lock(lock_dir, stale_seconds):
                continue
            if time.monotonic() - wait_started >= wait_timeout:
                raise TimeoutError(f"timed out acquiring metadata cache lock: {lock_dir}")
            time.sleep(wait_seconds)
    try:
        with _CacheLockHeartbeat(lock_dir, stale_seconds):
            cached_metadata = _load_sampler_metadata_cache(
                cache_path,
                n,
                fingerprint,
                fail_fast=cache_fail_fast,
            )
            if cached_metadata is not None:
                return cached_metadata[0]
            metadata = _compute_sampler_metadata_uncached(dataset, sampler_cfg)
            if metadata is None:
                return None
            weights, buckets, primary = metadata
            _write_sampler_metadata_cache(
                cache_path,
                weights,
                buckets,
                n,
                fingerprint,
                primary_partition=primary,
            )
            _SAMPLER_METADATA_MEMORY[fingerprint] = (weights, buckets)
            print(f"[sampler] wrote sampler metadata cache {cache_path}", flush=True)
            return weights
    finally:
        try:
            lock_dir.rmdir()
        except OSError:
            pass


_GRIP_EVENT_BUCKET_NAMES = (
    "all",
    "positive",
    "negative",
    "positive_noevent",
    "negative_noevent",
    "transition",
    "transition_up",
    "transition_down",
    "boundary",
    "boundary_up",
    "boundary_down",
)

_GRIP_PRIMARY_BUCKET_NAMES = (
    "boundary_up",
    "boundary_down",
    "transition_up",
    "transition_down",
    "positive_noevent",
    "negative_noevent",
)

_GRIP_EVENT_FALLBACKS: dict[str, tuple[str, ...]] = {
    "boundary_up": ("boundary_up", "transition_up", "positive", "transition", "all"),
    "boundary_down": ("boundary_down", "transition_down", "negative", "transition", "all"),
    "boundary": ("boundary", "transition", "all"),
    "transition_up": ("transition_up", "positive", "transition", "all"),
    "transition_down": ("transition_down", "negative", "transition", "all"),
    "transition": ("transition", "all"),
    "positive_noevent": ("positive_noevent", "positive", "all"),
    "negative_noevent": ("negative_noevent", "negative", "all"),
    "positive": ("positive", "all"),
    "negative": ("negative", "all"),
    "all": ("all",),
}


def _grip_event_default_cycle() -> tuple[str, ...]:
    return (
        "boundary_up",
        "boundary_down",
        "transition_up",
        "transition_down",
        "transition_down",
        "transition_up",
        "positive_noevent",
        "negative_noevent",
        "boundary_down",
        "boundary_up",
        "transition_down",
        "transition_up",
        "negative_noevent",
        "positive_noevent",
        "all",
        "all",
    )


def _normalize_grip_event_cycle(sampler_cfg: dict, key: str = "event_balance_cycle") -> tuple[str, ...]:
    raw = sampler_cfg.get(key)
    if raw is None and key == "event_balance_cycle":
        raw = sampler_cfg.get("grip_event_balance_cycle")
    if raw is None:
        return _grip_event_default_cycle()
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        values = [str(part).strip() for part in raw if str(part).strip()]
    if not values:
        return _grip_event_default_cycle()
    unknown = [name for name in values if name not in _GRIP_EVENT_FALLBACKS]
    if unknown:
        raise ValueError(f"unknown grip event balance bucket(s): {unknown}")
    return tuple(values)


def _compute_grip_event_buckets_uncached(
    dataset,
    sampler_cfg: dict | None = None,
) -> dict[str, torch.Tensor]:
    n = len(dataset)
    buckets: dict[str, list[int]] = {name: [] for name in _GRIP_EVENT_BUCKET_NAMES}
    if n <= 0:
        return {name: torch.empty(0, dtype=torch.long) for name in _GRIP_EVENT_BUCKET_NAMES}
    strict_primary = _grip_partition_contract_enabled(sampler_cfg)
    grip_cache: dict[int, np.ndarray] = {}
    for i in range(n):
        idx = int(i)
        buckets["all"].append(idx)
        stats = _sample_grip_window_stats(dataset, idx, grip_cache)
        if stats is None:
            continue
        if strict_primary:
            primary = str(stats.get("primary_partition", ""))
            if primary not in _GRIP_PRIMARY_BUCKET_NAMES:
                raise ValueError(f"unknown gripper primary partition {primary!r} for sample {idx}")
            buckets[primary].append(idx)
            continue
        has_positive = bool(stats.get("has_positive", False))
        has_transition = bool(stats.get("has_transition", False))
        has_boundary = bool(stats.get("has_boundary_transition", False))
        buckets["positive" if has_positive else "negative"].append(idx)
        if not has_transition and not has_boundary:
            buckets["positive_noevent" if has_positive else "negative_noevent"].append(idx)
        if has_transition:
            buckets["transition"].append(idx)
        if bool(stats.get("has_transition_up", False)):
            buckets["transition_up"].append(idx)
        if bool(stats.get("has_transition_down", False)):
            buckets["transition_down"].append(idx)
        if has_boundary:
            buckets["boundary"].append(idx)
        if bool(stats.get("has_boundary_up", False)):
            buckets["boundary_up"].append(idx)
        if bool(stats.get("has_boundary_down", False)):
            buckets["boundary_down"].append(idx)
    tensors = {name: torch.as_tensor(values, dtype=torch.long) for name, values in buckets.items()}
    if not strict_primary:
        return tensors
    labels = _primary_labels_from_buckets(tensors, n)
    return _buckets_from_primary_labels(labels)


def _compute_primary_partition_labels_uncached(dataset) -> torch.Tensor:
    labels = torch.full((len(dataset),), -1, dtype=torch.int16)
    grip_cache: dict[int, np.ndarray] = {}
    for sample_index in range(len(dataset)):
        stats = _sample_grip_window_stats(dataset, sample_index, grip_cache)
        if stats is None:
            continue
        primary = str(stats.get("primary_partition", ""))
        if primary not in _GRIP_PRIMARY_BUCKET_NAMES:
            raise ValueError(f"unknown gripper primary partition {primary!r} for sample {sample_index}")
        labels[sample_index] = _GRIP_PRIMARY_BUCKET_NAMES.index(primary)
    return labels


def build_grip_event_buckets(dataset, sampler_cfg: dict | None = None) -> dict[str, torch.Tensor]:
    n = len(dataset)
    if isinstance(sampler_cfg, dict) and n > 0:
        cache_path = _sampler_cache_path(sampler_cfg.get("cache_path"), n)
        if cache_path is not None:
            fingerprint = _sample_weight_cache_fingerprint(dataset, sampler_cfg, n)
            memory = _SAMPLER_METADATA_MEMORY.get(fingerprint)
            if memory is not None:
                return {name: values.clone() for name, values in memory[1].items()}
            metadata = _load_sampler_metadata_cache(
                cache_path,
                n,
                fingerprint,
                fail_fast=bool(sampler_cfg.get("cache_fail_fast", True)),
            )
            if metadata is not None:
                return {name: values.clone() for name, values in metadata[1].items()}
    return _compute_grip_event_buckets_uncached(dataset, sampler_cfg)


class GripEventBalancedDistributedSampler(Sampler[int]):
    """Distributed sampler with scheduled gripper-event coverage.

    Plain weighted sampling can still miss rare up/down/boundary events for many
    optimizer steps when each rank has batch size 1. Fixed event-balanced
    sampling solves exposure but can bias the grip prior. This sampler uses a
    strong event cycle early, then anneals into weighted sampling and an optional
    calibration cycle so transition learning and threshold calibration both stay
    visible to training.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        buckets: dict[str, torch.Tensor],
        *,
        cycle: tuple[str, ...],
        calibration_cycle: tuple[str, ...] | None,
        num_replicas: int,
        rank: int,
        replacement: bool = True,
        num_samples: int | None = None,
        seed: int = 0,
        batch_size_per_rank: int = 1,
        start_prob: float = 1.0,
        final_prob: float = 1.0,
        warmup_steps: int = 0,
        anneal_steps: int = 0,
        calibration_start_step: int | None = None,
        rank_rotate: bool = False,
        audit: bool = True,
        strict_primary: bool = False,
    ) -> None:
        self.weights = weights.double().cpu()
        self.buckets = {name: tensor.long().cpu() for name, tensor in buckets.items()}
        self.cycle = tuple(cycle)
        self.calibration_cycle = tuple(calibration_cycle) if calibration_cycle else tuple(cycle)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = bool(replacement)
        self.batch_size_per_rank = max(1, int(batch_size_per_rank))
        self.start_prob = max(0.0, min(1.0, float(start_prob)))
        self.final_prob = max(0.0, min(1.0, float(final_prob)))
        self.warmup_steps = max(0, int(warmup_steps))
        self.anneal_steps = max(0, int(anneal_steps))
        self.calibration_start_step = (
            self.warmup_steps if calibration_start_step is None else max(0, int(calibration_start_step))
        )
        self.rank_rotate = bool(rank_rotate)
        self.strict_primary = bool(strict_primary)
        positive_count = int((self.weights > 0).sum().item())
        if positive_count <= 0:
            raise ValueError("GripEventBalancedDistributedSampler needs positive sample weights")
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        if not self.cycle:
            raise ValueError("grip event balance cycle must not be empty")
        if not self.calibration_cycle:
            raise ValueError("grip event balance calibration cycle must not be empty")
        if self.strict_primary:
            invalid_cycle = sorted(
                {
                    name
                    for name in (*self.cycle, *self.calibration_cycle)
                    if name not in _GRIP_PRIMARY_BUCKET_NAMES
                }
            )
            if invalid_cycle:
                raise ValueError(
                    "strict primary event sampler cycle/calibration must contain only canonical primary "
                    f"partitions; invalid entries: {invalid_cycle}"
                )
        if self.replacement:
            self.num_samples = int(num_samples) if num_samples is not None else int(math.ceil(len(weights) / self.num_replicas))
        else:
            max_per_rank = positive_count // self.num_replicas
            if max_per_rank <= 0:
                raise ValueError(
                    "GripEventBalancedDistributedSampler with replacement=False needs at least "
                    "num_replicas positive sample weights"
                )
            requested = int(num_samples) if num_samples is not None else max_per_rank
            self.num_samples = min(requested, max_per_rank)
        self.total_size = self.num_samples * self.num_replicas
        self.seed = int(seed)
        self.epoch = 0
        self.bucket_weights: dict[str, torch.Tensor] = {}
        for name, indices in self.buckets.items():
            if indices.numel() <= 0:
                continue
            w = self.weights.index_select(0, indices).clone()
            valid = torch.isfinite(w) & (w > 0)
            if not bool(valid.all()):
                w = torch.where(valid, w, torch.zeros_like(w))
            if float(w.sum().item()) <= 0.0:
                w = torch.ones(indices.numel(), dtype=torch.double)
            self.bucket_weights[name] = w.double().cpu()
        primary_tensors = [
            self.buckets.get(name, torch.empty(0, dtype=torch.long))
            for name in _GRIP_PRIMARY_BUCKET_NAMES
        ]
        primary_assignments = torch.cat(primary_tensors) if primary_tensors else torch.empty(0, dtype=torch.long)
        primary_unique = torch.unique(primary_assignments)
        self.unique_ratio = (
            float(primary_unique.numel()) / float(primary_assignments.numel())
            if primary_assignments.numel() > 0
            else 1.0
        )
        total_weight = float(self.weights.clamp_min(0).sum().item())
        self.partition_mass = {
            name: (
                float(self.weights.index_select(0, self.buckets.get(name, torch.empty(0, dtype=torch.long))).sum().item())
                / max(total_weight, 1e-12)
            )
            for name in _GRIP_PRIMARY_BUCKET_NAMES
        }
        primary_complete = (
            primary_assignments.numel() == len(self.weights)
            and primary_unique.numel() == len(self.weights)
            and torch.equal(torch.sort(primary_unique).values, torch.arange(len(self.weights), dtype=torch.long))
        )
        if self.strict_primary and not primary_complete:
            raise ValueError(
                "strict primary event sampler requires every dataset sample to have exactly one primary partition; "
                f"assignments={primary_assignments.numel()} unique={primary_unique.numel()} dataset={len(self.weights)}"
            )
        if audit and self.rank == 0:
            host = socket.gethostname().split(".")[0]
            empty = torch.empty(0, dtype=torch.long)
            counts = " ".join(f"{name}={int(self.buckets.get(name, empty).numel())}" for name in _GRIP_EVENT_BUCKET_NAMES)
            cycle_s = ",".join(self.cycle)
            calib_s = ",".join(self.calibration_cycle)
            mass_s = ",".join(f"{name}:{self.partition_mass[name]:.5f}" for name in _GRIP_PRIMARY_BUCKET_NAMES)
            print(
                "[sampler] grip_event_balanced "
                f"host={host} replicas={self.num_replicas} batch_size_per_rank={self.batch_size_per_rank} "
                f"start_prob={self.start_prob:.3f} final_prob={self.final_prob:.3f} "
                f"warmup_steps={self.warmup_steps} anneal_steps={self.anneal_steps} "
                f"calibration_start_step={self.calibration_start_step} rank_rotate={int(self.rank_rotate)} "
                f"strict_primary={int(self.strict_primary)} cycle={cycle_s} calibration_cycle={calib_s} "
                f"unique_ratio={self.unique_ratio:.6f} "
                f"partition_mass={mass_s} {counts}",
                flush=True,
            )

    def _resolve_bucket(self, name: str) -> str:
        empty = torch.empty(0, dtype=torch.long)
        if self.strict_primary:
            if name not in _GRIP_PRIMARY_BUCKET_NAMES:
                raise ValueError(f"strict primary sampler received non-primary bucket {name!r}")
            if self.buckets.get(name, empty).numel() <= 0:
                raise ValueError(f"strict primary sampler requested empty partition {name!r}")
            return name
        for candidate in _GRIP_EVENT_FALLBACKS.get(name, (name, "all")):
            if self.buckets.get(candidate, empty).numel() > 0:
                return candidate
        raise ValueError("grip event sampler has no non-empty fallback bucket")

    def _optimizer_step_for_sample(self, local_i: int) -> int:
        steps_per_epoch = int(math.ceil(self.num_samples / self.batch_size_per_rank))
        return int(self.epoch * steps_per_epoch + int(local_i) // self.batch_size_per_rank)

    def _event_probability(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.start_prob
        if self.anneal_steps <= 0:
            return self.final_prob
        t = min(1.0, max(0.0, float(step - self.warmup_steps) / float(self.anneal_steps)))
        return self.start_prob + (self.final_prob - self.start_prob) * t

    def _cycle_for_step(self, step: int) -> tuple[str, ...]:
        if step >= self.calibration_start_step:
            return self.calibration_cycle
        return self.cycle

    def __iter__(self):
        if self.strict_primary:
            return iter(self._strict_global_samples())
        weighted_g = torch.Generator()
        weighted_g.manual_seed(self.seed + 104729 + self.epoch)
        weighted_global = torch.multinomial(self.weights, self.total_size, self.replacement, generator=weighted_g).tolist()
        weighted_local = weighted_global[self.rank:self.total_size:self.num_replicas]

        mix_g = torch.Generator()
        mix_g.manual_seed(self.seed + 130363 + self.epoch)
        use_event: list[bool] = []
        positions: dict[str, list[int]] = {}
        for local_i in range(self.num_samples):
            step = self._optimizer_step_for_sample(local_i)
            prob = self._event_probability(step)
            take_event = prob >= 1.0 or (prob > 0.0 and float(torch.rand((), generator=mix_g).item()) < prob)
            use_event.append(take_event)
            if not take_event:
                continue
            cycle = self._cycle_for_step(step)
            batch_pos = local_i % self.batch_size_per_rank
            rank_for_cycle = (self.rank + step) % self.num_replicas if self.rank_rotate else self.rank
            global_slot = step * self.num_replicas * self.batch_size_per_rank + batch_pos * self.num_replicas + rank_for_cycle
            requested = cycle[global_slot % len(cycle)]
            bucket = self._resolve_bucket(requested)
            positions.setdefault(bucket, []).append(local_i)

        sampled = [int(x) for x in weighted_local]
        event_g = torch.Generator()
        # Each DDP rank owns a different event draw stream.  Reusing the same
        # seed made ranks assigned to the same bucket draw identical samples,
        # reducing the effective global batch exactly on rare event steps.
        event_g.manual_seed(self.seed + self.epoch * 1009 + self.rank * 1000003)
        for bucket, pos_list in positions.items():
            indices = self.buckets[bucket]
            weights = self.bucket_weights[bucket]
            draws = torch.multinomial(weights, len(pos_list), self.replacement, generator=event_g).tolist()
            for out_pos, draw_idx in zip(pos_list, draws):
                sampled[out_pos] = int(indices[int(draw_idx)].item())
        return iter(sampled)

    def _strict_global_samples(self) -> list[int]:
        if not self.replacement:
            return self._strict_global_samples_without_replacement()
        sampled_global = [-1] * self.total_size
        mix_g = torch.Generator()
        mix_g.manual_seed(self.seed + 130363 + self.epoch)
        draw_g = torch.Generator()
        draw_g.manual_seed(self.seed + self.epoch * 1009 + 15485863)
        epoch_selected = torch.zeros(len(self.weights), dtype=torch.bool)
        for batch_start in range(0, self.num_samples, self.batch_size_per_rank):
            batch_stop = min(self.num_samples, batch_start + self.batch_size_per_rank)
            batch_size = (batch_stop - batch_start) * self.num_replicas
            selected = epoch_selected.clone() if not self.replacement else torch.zeros_like(epoch_selected)
            positive_remaining = int(((self.weights > 0) & ~selected).sum().item())
            if positive_remaining < batch_size:
                raise ValueError(
                    "strict primary sampler cannot form a unique global batch: "
                    f"required={batch_size} positive_available={positive_remaining} "
                    f"replacement={int(self.replacement)}"
                )

            event_positions: dict[str, list[int]] = {}
            weighted_positions: list[int] = []
            for local_i in range(batch_start, batch_stop):
                step = self._optimizer_step_for_sample(local_i)
                probability = self._event_probability(step)
                cycle = self._cycle_for_step(step)
                batch_pos = local_i % self.batch_size_per_rank
                for sampler_rank in range(self.num_replicas):
                    global_index = sampler_rank + local_i * self.num_replicas
                    take_event = probability >= 1.0 or (
                        probability > 0.0
                        and float(torch.rand((), generator=mix_g).item()) < probability
                    )
                    if not take_event:
                        weighted_positions.append(global_index)
                        continue
                    rank_for_cycle = (
                        (sampler_rank + step) % self.num_replicas
                        if self.rank_rotate
                        else sampler_rank
                    )
                    global_slot = (
                        step * self.num_replicas * self.batch_size_per_rank
                        + batch_pos * self.num_replicas
                        + rank_for_cycle
                    )
                    bucket = self._resolve_bucket(cycle[global_slot % len(cycle)])
                    event_positions.setdefault(bucket, []).append(global_index)

            for bucket, output_positions in event_positions.items():
                indices = self.buckets[bucket]
                bucket_weights = self.weights.index_select(0, indices).clone()
                eligible = (bucket_weights > 0) & ~selected.index_select(0, indices)
                available = int(eligible.sum().item())
                if len(output_positions) > available:
                    raise ValueError(
                        "strict primary sampler cannot satisfy unique event slots: "
                        f"bucket={bucket} requested={len(output_positions)} available={available}"
                    )
                bucket_weights = torch.where(eligible, bucket_weights, torch.zeros_like(bucket_weights))
                draws = torch.multinomial(
                    bucket_weights,
                    len(output_positions),
                    replacement=False,
                    generator=draw_g,
                )
                chosen = indices.index_select(0, draws)
                for output_position, sample_index in zip(output_positions, chosen.tolist()):
                    sampled_global[output_position] = int(sample_index)
                selected.index_fill_(0, chosen, True)

            if weighted_positions:
                remaining_weights = torch.where(
                    (self.weights > 0) & ~selected,
                    self.weights,
                    torch.zeros_like(self.weights),
                )
                available = int((remaining_weights > 0).sum().item())
                if len(weighted_positions) > available:
                    raise ValueError(
                        "strict primary sampler cannot satisfy unique weighted slots: "
                        f"requested={len(weighted_positions)} available={available}"
                    )
                chosen = torch.multinomial(
                    remaining_weights,
                    len(weighted_positions),
                    replacement=False,
                    generator=draw_g,
                )
                for output_position, sample_index in zip(weighted_positions, chosen.tolist()):
                    sampled_global[output_position] = int(sample_index)
                selected.index_fill_(0, chosen, True)

            global_batch = [
                sampled_global[sampler_rank + local_i * self.num_replicas]
                for local_i in range(batch_start, batch_stop)
                for sampler_rank in range(self.num_replicas)
            ]
            if any(sample_index < 0 for sample_index in global_batch):
                raise RuntimeError(
                    "strict primary sampler left global batch slots unassigned: "
                    f"epoch={self.epoch} local_batch={batch_start // self.batch_size_per_rank} "
                    f"samples={global_batch}"
                )
            if len(global_batch) != len(set(global_batch)):
                raise RuntimeError(
                    "strict primary sampler produced duplicate dataset samples in one global batch: "
                    f"epoch={self.epoch} local_batch={batch_start // self.batch_size_per_rank} "
                    f"samples={global_batch}"
                )
            if not self.replacement:
                epoch_selected = selected
        return [
            sampled_global[self.rank + local_i * self.num_replicas]
            for local_i in range(self.num_samples)
        ]

    def _strict_global_samples_without_replacement(self) -> list[int]:
        """Draw the bounded probe epoch with one multinomial call per partition."""
        sampled_global = [-1] * self.total_size
        mix_g = torch.Generator()
        mix_g.manual_seed(self.seed + 130363 + self.epoch)
        draw_g = torch.Generator()
        draw_g.manual_seed(self.seed + self.epoch * 1009 + 15485863)
        event_positions: dict[str, list[int]] = {}
        weighted_positions: list[int] = []

        for local_i in range(self.num_samples):
            step = self._optimizer_step_for_sample(local_i)
            probability = self._event_probability(step)
            cycle = self._cycle_for_step(step)
            batch_pos = local_i % self.batch_size_per_rank
            for sampler_rank in range(self.num_replicas):
                global_index = sampler_rank + local_i * self.num_replicas
                take_event = probability >= 1.0 or (
                    probability > 0.0
                    and float(torch.rand((), generator=mix_g).item()) < probability
                )
                if not take_event:
                    weighted_positions.append(global_index)
                    continue
                rank_for_cycle = (
                    (sampler_rank + step) % self.num_replicas
                    if self.rank_rotate
                    else sampler_rank
                )
                global_slot = (
                    step * self.num_replicas * self.batch_size_per_rank
                    + batch_pos * self.num_replicas
                    + rank_for_cycle
                )
                bucket = self._resolve_bucket(cycle[global_slot % len(cycle)])
                event_positions.setdefault(bucket, []).append(global_index)

        selected = torch.zeros(len(self.weights), dtype=torch.bool)
        for bucket, output_positions in event_positions.items():
            indices = self.buckets[bucket]
            bucket_weights = self.weights.index_select(0, indices)
            eligible = bucket_weights > 0
            available = int(eligible.sum().item())
            if len(output_positions) > available:
                raise ValueError(
                    "strict primary sampler cannot satisfy unique event slots: "
                    f"bucket={bucket} requested={len(output_positions)} available={available}"
                )
            draws = torch.multinomial(
                torch.where(eligible, bucket_weights, torch.zeros_like(bucket_weights)),
                len(output_positions),
                replacement=False,
                generator=draw_g,
            )
            chosen = indices.index_select(0, draws)
            for output_position, sample_index in zip(output_positions, chosen.tolist()):
                sampled_global[output_position] = int(sample_index)
            selected.index_fill_(0, chosen, True)

        if weighted_positions:
            remaining_weights = torch.where(
                (self.weights > 0) & ~selected,
                self.weights,
                torch.zeros_like(self.weights),
            )
            available = int((remaining_weights > 0).sum().item())
            if len(weighted_positions) > available:
                raise ValueError(
                    "strict primary sampler cannot satisfy unique weighted slots: "
                    f"requested={len(weighted_positions)} available={available}"
                )
            chosen = torch.multinomial(
                remaining_weights,
                len(weighted_positions),
                replacement=False,
                generator=draw_g,
            )
            for output_position, sample_index in zip(weighted_positions, chosen.tolist()):
                sampled_global[output_position] = int(sample_index)

        if any(sample_index < 0 for sample_index in sampled_global):
            raise RuntimeError("strict primary sampler left global epoch slots unassigned")
        if len(sampled_global) != len(set(sampled_global)):
            raise RuntimeError("strict primary sampler produced duplicate samples in a no-replacement epoch")
        return [
            sampled_global[self.rank + local_i * self.num_replicas]
            for local_i in range(self.num_samples)
        ]

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def build_grip_event_balanced_sampler(
    dataset,
    sampler_cfg: dict | bool | None,
    weights: torch.Tensor,
    *,
    num_replicas: int,
    rank: int,
    replacement: bool = True,
    num_samples: int | None = None,
    seed: int = 0,
    batch_size_per_rank: int = 1,
) -> GripEventBalancedDistributedSampler | None:
    if not isinstance(sampler_cfg, dict):
        return None
    enabled = bool(sampler_cfg.get("event_balance", False) or sampler_cfg.get("grip_event_balance", False))
    if not enabled:
        return None
    strict_primary = _grip_partition_contract_enabled(sampler_cfg)
    buckets = build_grip_event_buckets(dataset, sampler_cfg)
    cycle = _normalize_grip_event_cycle(sampler_cfg, "event_balance_cycle")
    calibration_cycle = None
    if sampler_cfg.get("event_balance_calibration_cycle") is not None:
        calibration_cycle = _normalize_grip_event_cycle(sampler_cfg, "event_balance_calibration_cycle")
    return GripEventBalancedDistributedSampler(
        weights,
        buckets,
        cycle=cycle,
        calibration_cycle=calibration_cycle,
        num_replicas=num_replicas,
        rank=rank,
        replacement=replacement,
        num_samples=num_samples,
        seed=seed,
        batch_size_per_rank=batch_size_per_rank,
        start_prob=float(sampler_cfg.get("event_balance_start_prob", 1.0)),
        final_prob=float(sampler_cfg.get("event_balance_final_prob", 1.0)),
        warmup_steps=int(sampler_cfg.get("event_balance_warmup_steps", 0) or 0),
        anneal_steps=int(sampler_cfg.get("event_balance_anneal_steps", 0) or 0),
        calibration_start_step=(
            int(sampler_cfg["event_balance_calibration_start_step"])
            if sampler_cfg.get("event_balance_calibration_start_step") is not None
            else None
        ),
        rank_rotate=bool(sampler_cfg.get("event_balance_rank_rotate", False)),
        audit=bool(sampler_cfg.get("event_balance_audit", True)),
        strict_primary=strict_primary,
    )


class WeightedDistributedSampler(Sampler[int]):
    def __init__(
        self,
        weights: torch.Tensor,
        *,
        num_replicas: int,
        rank: int,
        replacement: bool = True,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        self.weights = weights.double().cpu()
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = bool(replacement)
        positive_count = int((self.weights > 0).sum().item())
        if positive_count <= 0:
            raise ValueError("WeightedDistributedSampler needs at least one positive sample weight")
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        if self.replacement:
            self.num_samples = int(num_samples) if num_samples is not None else int(math.ceil(len(weights) / self.num_replicas))
        else:
            max_per_rank = positive_count // self.num_replicas
            if max_per_rank <= 0:
                raise ValueError(
                    "WeightedDistributedSampler with replacement=False needs at least "
                    "num_replicas positive sample weights"
                )
            requested = int(num_samples) if num_samples is not None else max_per_rank
            self.num_samples = min(requested, max_per_rank)
        self.total_size = self.num_samples * self.num_replicas
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        sampled = torch.multinomial(self.weights, self.total_size, self.replacement, generator=g).tolist()
        return iter(sampled[self.rank:self.total_size:self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def _equalized_num_samples_per_rank(local_lengths: list[int], sampler_world: int) -> int:
    if sampler_world <= 0:
        raise ValueError("sampler_world must be positive")
    if not local_lengths:
        raise ValueError("local_lengths must not be empty")
    min_len = min(int(x) for x in local_lengths)
    if min_len <= 0:
        raise ValueError(f"cannot equalize empty local datasets: {local_lengths}")
    return max(1, min_len // int(sampler_world))


def _validate_action_audit_gate(oxe_cfg: dict, source_name: str) -> None:
    """Validate the immutable per-source action audit before factual use.

    The data builder does not itself decide whether a source is factual.  It
    does, however, verify a configured audit gate eagerly so a stale or failed
    report cannot be hidden behind a train-time allowlist.
    """

    gate = oxe_cfg.get("action_audit_gate")
    if gate is None:
        return
    if isinstance(gate, (str, Path)):
        gate = {
            "path": str(gate),
            "sha256": oxe_cfg.get("action_audit_gate_sha256"),
        }
    elif not isinstance(gate, dict):
        raise ValueError(f"{source_name} action_audit_gate must be a path or mapping")
    report_path = gate.get("path") or gate.get("report") or gate.get("report_path")
    if not report_path:
        raise ValueError(f"{source_name} action_audit_gate requires a report path")
    report_bytes = Path(report_path).read_bytes()
    expected_sha256 = str(gate.get("sha256") or gate.get("report_sha256") or "")
    observed_sha256 = hashlib.sha256(report_bytes).hexdigest()
    if expected_sha256 and observed_sha256 != expected_sha256:
        raise ValueError(
            f"{source_name} action audit digest mismatch: "
            f"observed={observed_sha256} expected={expected_sha256}"
        )
    report = json.loads(report_bytes)
    passed = report.get("passed") is True or report.get("status") == "passed"
    if not passed or report.get("failures"):
        raise ValueError(
            f"{source_name} action audit did not pass: "
            f"status={report.get('status')!r} failures={report.get('failures')!r}"
        )
    expected_source = gate.get("source")
    if expected_source is not None and str(report.get("source")) != str(expected_source):
        raise ValueError(
            f"{source_name} action audit source mismatch: "
            f"observed={report.get('source')!r} expected={expected_source!r}"
        )


def _validate_oxe_action_adapter(oxe_cfg: dict, source_name: str) -> None:
    """Require the one canonical continuous signed-grip OXE adapter."""

    adapter = oxe_cfg.get("action_adapter")
    if adapter is None:
        if bool(oxe_cfg.get("canonical_action_enabled", False)):
            raise ValueError(f"{source_name} canonical action source lacks action_adapter")
        return
    if not isinstance(adapter, dict):
        raise ValueError(f"{source_name} action_adapter must be a mapping")
    if adapter.get("version") != "wm3d_v7_base_delta_axisangle_gripclose_v1":
        raise ValueError(f"{source_name} action_adapter version is not canonical V7")
    if adapter.get("gripper_semantics") != "signed_close_positive_continuous":
        raise ValueError(
            f"{source_name} action_adapter gripper_semantics must be "
            "signed_close_positive_continuous"
        )
    if adapter.get("translation_unit") != "meter":
        raise ValueError(f"{source_name} action_adapter translation_unit must be meter")
    if adapter.get("source_frame") != "base":
        raise ValueError(f"{source_name} action_adapter source_frame must be base")


def _mixed_oxe_source_config(raw_oxe_cfg: dict, data_cfg: dict) -> dict:
    """Apply shared loader defaults without overriding a source contract."""

    oxe_cfg = dict(raw_oxe_cfg)
    if "load_task_text" not in oxe_cfg:
        shared_oxe = data_cfg.get("oxe") or {}
        if isinstance(shared_oxe, dict) and "load_task_text" in shared_oxe:
            oxe_cfg["load_task_text"] = bool(shared_oxe["load_task_text"])
        else:
            oxe_cfg["load_task_text"] = bool(data_cfg.get("load_task_text", False))
    return oxe_cfg


def _build_mixed_oxe_source(
    source_name: str,
    raw_oxe_cfg: dict,
    data_cfg: dict,
    model_cfg: dict | None,
) -> tuple[OXEWindowDataset, OXEWindowDataset]:
    """Build one independently-normalized, episode-split OXE source."""

    oxe_cfg = _mixed_oxe_source_config(raw_oxe_cfg, data_cfg)
    if bool(data_cfg.get("disable_rgb_io", False)):
        # S1 keeps the frozen S0 renderer out of the hot path.  This global
        # switch lets a stage inherit the fully audited per-source S0 loader
        # contracts without copying and editing both long OXE source blocks.
        oxe_cfg["load_rgb"] = False
    for key in ("T", "k"):
        oxe_cfg.setdefault(key, data_cfg[key])
    oxe_cfg.setdefault("stride", 4)
    if bool(oxe_cfg.get("canonical_action_enabled", False)):
        if oxe_cfg.get("action_stats") is not None:
            raise ValueError(
                f"OXE source {source_name!r} canonical action mode forbids "
                "legacy pooled action_stats"
            )
    else:
        oxe_cfg.setdefault("action_stats", data_cfg.get("action_stats"))
    _validate_oxe_action_adapter(oxe_cfg, source_name)
    _validate_action_audit_gate(oxe_cfg, source_name)

    records = read_manifest(oxe_cfg["manifest"])
    include_datasets = {str(name) for name in oxe_cfg.get("include_datasets", ())}
    if not include_datasets:
        raise ValueError(
            f"mixed OXE source {source_name!r} requires explicit include_datasets"
        )
    records = [record for record in records if str(record.dataset) in include_datasets]
    if not records:
        raise RuntimeError(
            f"OXE source {source_name!r} cohort is empty: {sorted(include_datasets)}"
        )
    allowed_action_kinds = {str(kind) for kind in oxe_cfg.get("allowed_action_kinds", ())}
    observed_action_kinds = {str(record.action_kind) for record in records}
    if allowed_action_kinds and not observed_action_kinds.issubset(allowed_action_kinds):
        raise ValueError(
            f"OXE source {source_name!r} contains unapproved action kinds: "
            f"{sorted(observed_action_kinds - allowed_action_kinds)}"
        )
    allowed_fps = {float(value) for value in oxe_cfg.get("allowed_fps", ())}
    observed_fps = {float(record.fps) for record in records}
    if allowed_fps and not observed_fps.issubset(allowed_fps):
        raise ValueError(
            f"OXE source {source_name!r} contains unapproved frame rates: "
            f"{sorted(observed_fps - allowed_fps)}"
        )

    evidence_path = oxe_cfg.get("action_contract_evidence_path")
    if evidence_path:
        evidence_bytes = Path(evidence_path).read_bytes()
        expected_sha256 = str(oxe_cfg.get("action_contract_evidence_sha256", ""))
        observed_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"OXE source {source_name!r} temporal evidence digest mismatch: "
                f"observed={observed_sha256} expected={expected_sha256}"
            )
        evidence = json.loads(evidence_bytes)
        evidence_groups = evidence.get("groups") or {}
        configured_offset = int(oxe_cfg.get("default_action_frame_offset", 0))
        for contract_key in sorted({action_contract_key(record) for record in records}):
            claim = evidence_groups.get(contract_key)
            if (
                not isinstance(claim, dict)
                or claim.get("status") != "passed"
                or int(claim.get("offset", 99)) != configured_offset
            ):
                raise ValueError(
                    f"OXE source {source_name!r} lacks matching passed temporal "
                    f"evidence for {contract_key}: expected_offset={configured_offset}"
                )

    split_cfg = _data_split_cfg(oxe_cfg)
    if split_cfg.get("mode", "episode") != "episode":
        raise ValueError(f"mixed OXE source {source_name!r} requires an episode-level split")
    split = episode_split(
        records,
        val_frac=float(_split_value(oxe_cfg, split_cfg, "val_frac", 0.03)),
        seed=int(_split_value(oxe_cfg, split_cfg, "seed", 0)),
    )
    train_records = [record for record in records if record.clip_id in split.train_clip_ids]
    val_records = [record for record in records if record.clip_id in split.val_clip_ids]
    window_cfg = _window_config(oxe_cfg, model_cfg)
    train_dataset = OXEWindowDataset(train_records, window_cfg)
    val_dataset = OXEWindowDataset(val_records, window_cfg)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(
            f"mixed OXE source {source_name!r} has an empty cached train/val split"
        )
    return train_dataset, val_dataset


def build_datasets(cfg: dict, overfit_ids=None):
    data_cfg = cfg["data"]
    if data_cfg.get("dataset_type") == "worldarena_native_s0":
        if overfit_ids:
            raise ValueError(
                "overfit_ids is not supported by the WorldArena native S0 loader"
            )
        from wm3d_v3.data.worldarena_bimanual_dataset import (
            WorldArenaNativeS0WindowDataset,
        )

        action_stats = Path(
            data_cfg.get("worldarena_action_stats", data_cfg["action_stats"])
        )
        train_dataset = WorldArenaNativeS0WindowDataset(
            index=Path(data_cfg["index"]),
            action_stats=action_stats,
            split="train",
            start_offsets=tuple(
                int(value)
                for value in data_cfg.get("train_start_offsets", range(24))
            ),
        )
        val_dataset = WorldArenaNativeS0WindowDataset(
            index=Path(data_cfg.get("val_index", data_cfg["index"])),
            action_stats=action_stats,
            split="val",
            start_offsets=tuple(
                int(value)
                for value in data_cfg.get(
                    "val_start_offsets", (0, 4, 8, 12, 16, 20, 23)
                )
            ),
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                "[rank0] WorldArena native S0 dataset "
                f"train={len(train_dataset)} val={len(val_dataset)} "
                "train_episodes=0..35 val_episodes=36..39 test_excluded=true "
                "T=16 K=8 native_action=left7||right7_normalized direct14=true",
                flush=True,
            )
        return train_dataset, val_dataset
    if data_cfg.get("dataset_type") == "worldarena_wan":
        if overfit_ids:
            raise ValueError("overfit_ids is not supported by the WorldArena Wan loader")
        from wm3d_v3.data.worldarena_bimanual_dataset import (
            WorldArenaWanWindowDataset,
        )

        common = {
            "index": Path(data_cfg["index"]),
            "action_stats": Path(
                data_cfg.get("worldarena_action_stats", data_cfg["action_stats"])
            ),
        }
        train_dataset = WorldArenaWanWindowDataset(
            split="train",
            start_offsets=tuple(
                int(value) for value in data_cfg.get("train_start_offsets", range(16))
            ),
            **common,
        )
        val_dataset = WorldArenaWanWindowDataset(
            index=Path(data_cfg.get("val_index", data_cfg["index"])),
            action_stats=Path(
                data_cfg.get("worldarena_action_stats", data_cfg["action_stats"])
            ),
            split="val",
            start_offsets=tuple(
                int(value) for value in data_cfg.get("val_start_offsets", (0, 4, 8, 12, 15))
            ),
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                "[rank0] WorldArena Wan dataset "
                f"train={len(train_dataset)} val={len(val_dataset)} "
                "train_episodes=0..35 val_episodes=36..39 test_excluded=true "
                "T=16 K=8 bimanual_fusion=true "
                "native_action=dominant7 renderer_action=left7||right7_normalized",
                flush=True,
            )
        return train_dataset, val_dataset
    if data_cfg.get("dataset_type") == "robonet_wan":
        if overfit_ids:
            raise ValueError("overfit_ids is not supported by the RoboNet Wan loader")
        from wm3d_v3.training.train_robonet_s1_adapt import (
            RoboNetWanWindowDataset,
        )

        common = {
            "action_stats": Path(data_cfg["action_stats"]),
            "seed": int(data_cfg.get("seed", 0)),
        }
        train_primary = RoboNetWanWindowDataset(
            Path(data_cfg["index"]),
            split="train",
            window_offset_policy=str(
                data_cfg.get("train_window_offset_policy", "cycle_0_2")
            ),
            **common,
        )
        train_parts: list[Dataset] = [train_primary]
        extra_index = data_cfg.get("extra_train_index")
        extra_repeats = int(data_cfg.get("extra_train_repeats", 0) or 0)
        if extra_index and extra_repeats > 0:
            extra_train = RoboNetWanWindowDataset(
                Path(extra_index),
                split="train",
                window_offset_policy=str(
                    data_cfg.get("extra_train_window_offset_policy", "fixed_zero")
                ),
                expected_window_policy=data_cfg.get("extra_train_window_policy"),
                **common,
            )
            train_parts.extend([extra_train] * extra_repeats)
        train_dataset: Dataset = (
            train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
        )
        val_dataset = RoboNetWanWindowDataset(
            Path(data_cfg.get("val_index", data_cfg["index"])),
            split="val",
            window_offset_policy=str(
                data_cfg.get("val_window_offset_policy", "fixed_zero")
            ),
            expected_window_policy=data_cfg.get("val_window_policy"),
            **common,
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                "[rank0] RoboNet Wan dataset "
                f"train={len(train_dataset)} primary={len(train_primary)} "
                f"extra_repeats={extra_repeats} val={len(val_dataset)} "
                "official_test_excluded=true T=16 K=8 offsets=train:0..2,val:0",
                flush=True,
            )
        return train_dataset, val_dataset
    if data_cfg.get("dataset_type") == "v7_mixed":
        if overfit_ids:
            raise ValueError("overfit_ids is not supported by the mixed V7 loader")
        compact_common = {
            "index_path": Path(data_cfg["compact_index"]),
            "T": int(data_cfg["T"]),
            "k": int(data_cfg["k"]),
            "stride": int(data_cfg["stride"]),
            "seed": int(data_cfg.get("seed", 0)),
            "require_task_emb": bool(data_cfg.get("require_task_emb", True)),
            "action_stats": (
                Path(data_cfg["action_stats"])
                if data_cfg.get("action_stats")
                else None
            ),
            "require_action_stats": bool(data_cfg.get("require_action_stats", True)),
            "rgb_sidecar_indices": tuple(
                Path(path) for path in data_cfg.get("rgb_sidecar_indices", ())
            ),
            "require_rgb_sidecar": bool(data_cfg.get("require_rgb_sidecar", False)),
            "action_only": bool(data_cfg.get("direct_policy_action_only", False)),
            "causal_dual_view_required": bool(
                data_cfg.get("compact_causal_dual_view_required", False)
            ),
            "causal_dual_view_representation": data_cfg.get(
                "compact_causal_dual_view_representation"
            ),
            "policy_action_history_len": int(
                data_cfg.get(
                    "policy_action_history_len",
                    (cfg.get("model") or {}).get("policy_action_history_len", 0),
                )
                or 0
            ),
            "policy_action_history_dim": int(
                data_cfg.get(
                    "policy_action_history_dim",
                    (cfg.get("model") or {}).get("policy_action_history_dim", 7),
                )
                or 7
            ),
        }
        robocasa_train = V7CompactWindowDataset(
            V7CompactDatasetConfig(
                **compact_common,
                split="train",
                view_dropout=float(data_cfg.get("view_dropout", 0.0)),
            )
        )
        robocasa_val = V7CompactWindowDataset(
            V7CompactDatasetConfig(
                **compact_common,
                split="val",
                view_dropout=0.0,
            )
        )
        robocasa_partitions = tuple(
            str(name)
            for name in data_cfg.get(
                "robocasa_partitions", ("atomic", "composite")
            )
        )
        train_partitions = partition_v7_compact_dataset(
            robocasa_train, robocasa_partitions
        )
        val_partitions = partition_v7_compact_dataset(
            robocasa_val, robocasa_partitions
        )

        raw_oxe_sources = data_cfg.get("oxe_sources")
        if raw_oxe_sources is not None:
            if not isinstance(raw_oxe_sources, (dict, list, tuple)) or not raw_oxe_sources:
                raise ValueError("data.oxe_sources must be a non-empty mapping or list")
            configured_oxe_sources: list[tuple[str, dict]] = []
            if isinstance(raw_oxe_sources, dict):
                source_items = list(raw_oxe_sources.items())
            else:
                source_items = [(str(index), value) for index, value in enumerate(raw_oxe_sources)]
            for source_key, source_cfg in source_items:
                if not isinstance(source_cfg, dict):
                    raise ValueError(f"data.oxe_sources.{source_key} must be a mapping")
                source_cfg = dict(source_cfg)
                if "source_name" not in source_cfg and not isinstance(raw_oxe_sources, dict):
                    raise ValueError(f"data.oxe_sources[{source_key}] requires source_name")
                source_name = str(source_cfg.pop("source_name", f"oxe_{source_key}"))
                configured_oxe_sources.append((source_name, source_cfg))
        else:
            # Backward-compatible single OXE stream.  The strict source-policy
            # gate below still prevents this legacy aggregate from silently
            # becoming factual unless it has an explicit audit admission.
            legacy_oxe_cfg = dict(data_cfg.get("oxe") or {})
            if not legacy_oxe_cfg:
                raise ValueError(
                    "data.oxe or data.oxe_sources is required for dataset_type=v7_mixed"
                )
            configured_oxe_sources = [
                (str(data_cfg.get("oxe_source_name", "oxe_bridge")), legacy_oxe_cfg)
            ]

        train_sources: list[tuple[str, object]] = []
        val_sources: list[tuple[str, object]] = []
        oxe_source_names: list[str] = []
        for source_name, source_cfg in configured_oxe_sources:
            if source_name in oxe_source_names:
                raise ValueError(f"duplicate mixed OXE source_name {source_name!r}")
            source_cfg = apply_direct_policy_oxe_overrides(source_cfg, data_cfg)
            oxe_train, oxe_val = _build_mixed_oxe_source(
                source_name, source_cfg, data_cfg, cfg.get("model")
            )
            oxe_source_names.append(source_name)
            train_sources.append((source_name, oxe_train))
            val_sources.append((source_name, oxe_val))
        for partition_name in robocasa_partitions:
            source_name = f"robocasa_{partition_name}"
            train_sources.append((source_name, train_partitions[partition_name]))
            val_sources.append((source_name, val_partitions[partition_name]))
        # S1 can reuse the already-cached large factual stream while adding
        # source-homogeneous true same-root counterfactual batches.  Keeping
        # the schemas in separate batches is important: branch samples carry
        # [K,H,7] actions / [K,H,P,D] futures while factual samples do not.
        branch_index = data_cfg.get("branch_index")
        if branch_index:
            branch_common = {
                "index_path": Path(branch_index),
                "T": int(data_cfg["T"]),
                "k": int(data_cfg["k"]),
                "require_task_emb": bool(data_cfg.get("require_task_emb", True)),
                "action_stats": (
                    Path(data_cfg["action_stats"])
                    if data_cfg.get("action_stats")
                    else None
                ),
                "require_action_stats": bool(
                    data_cfg.get("require_action_stats", True)
                ),
            }
            branch_source_name = str(
                data_cfg.get("branch_source_name", "robocasa_same_root_cf")
            )
            train_sources.append(
                (
                    branch_source_name,
                    V7SameRootBranchDataset(
                        V7SameRootBranchDatasetConfig(
                            **branch_common, split="train"
                        )
                    ),
                )
            )
            val_sources.append(
                (
                    branch_source_name,
                    V7SameRootBranchDataset(
                        V7SameRootBranchDatasetConfig(
                            **branch_common, split="val"
                        )
                    ),
                )
            )
        train_dataset = MixedSourceWindowDataset(
            train_sources, mono_sources=tuple(oxe_source_names)
        )
        val_dataset = MixedSourceWindowDataset(
            val_sources, mono_sources=tuple(oxe_source_names)
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                "[rank0] mixed dataset train="
                + " ".join(
                    f"{name}:{stop - start}"
                    for name, (start, stop) in train_dataset.source_spans.items()
                )
                + " val="
                + " ".join(
                    f"{name}:{stop - start}"
                    for name, (start, stop) in val_dataset.source_spans.items()
                ),
                flush=True,
            )
        return train_dataset, val_dataset
    if data_cfg.get("dataset_type") == "v7_same_root_branch":
        if overfit_ids:
            raise ValueError("overfit_ids is not supported by the same-root loader")
        common = {
            "index_path": Path(data_cfg["branch_index"]),
            "T": int(data_cfg["T"]),
            "k": int(data_cfg["k"]),
            "require_task_emb": bool(data_cfg.get("require_task_emb", True)),
            "action_stats": Path(data_cfg["action_stats"]) if data_cfg.get("action_stats") else None,
            "require_action_stats": bool(data_cfg.get("require_action_stats", True)),
        }
        return (
            V7SameRootBranchDataset(
                V7SameRootBranchDatasetConfig(**common, split="train")
            ),
            V7SameRootBranchDataset(
                V7SameRootBranchDatasetConfig(**common, split="val")
            ),
        )
    if data_cfg.get("dataset_type") == "v7_compact":
        if overfit_ids:
            raise ValueError("overfit_ids is not supported by the strict v7 compact loader")
        common = {
            "index_path": Path(data_cfg["compact_index"]),
            "T": int(data_cfg["T"]),
            "k": int(data_cfg["k"]),
            "stride": int(data_cfg["stride"]),
            "seed": int(data_cfg.get("seed", 0)),
            "require_task_emb": bool(data_cfg.get("require_task_emb", True)),
            "action_stats": Path(data_cfg["action_stats"]) if data_cfg.get("action_stats") else None,
            "require_action_stats": bool(data_cfg.get("require_action_stats", True)),
            "rgb_sidecar_indices": tuple(
                Path(path) for path in data_cfg.get("rgb_sidecar_indices", ())
            ),
            "require_rgb_sidecar": bool(data_cfg.get("require_rgb_sidecar", False)),
            "action_only": bool(data_cfg.get("direct_policy_action_only", False)),
            "causal_dual_view_required": bool(
                data_cfg.get("compact_causal_dual_view_required", False)
            ),
            "causal_dual_view_representation": data_cfg.get(
                "compact_causal_dual_view_representation"
            ),
            "policy_action_history_len": int(
                data_cfg.get(
                    "policy_action_history_len",
                    (cfg.get("model") or {}).get("policy_action_history_len", 0),
                )
                or 0
            ),
            "policy_action_history_dim": int(
                data_cfg.get(
                    "policy_action_history_dim",
                    (cfg.get("model") or {}).get("policy_action_history_dim", 7),
                )
                or 7
            ),
        }
        train_dataset = V7CompactWindowDataset(
            V7CompactDatasetConfig(
                **common,
                split="train",
                view_dropout=float(data_cfg.get("view_dropout", 0.0)),
            )
        )
        val_dataset = V7CompactWindowDataset(
            V7CompactDatasetConfig(
                **common,
                split="val",
                view_dropout=0.0,
            )
        )
        return train_dataset, val_dataset

    records = read_manifest(cfg["data"]["manifest"])
    if overfit_ids:
        records = [r for r in records if r.clip_id in overfit_ids]
        if not records:
            raise RuntimeError(f"no records matched overfit ids: {overfit_ids}")
    split_cfg = _data_split_cfg(data_cfg)
    has_episode_split_keys = (
        data_cfg.get("split_file")
        or any(k in split_cfg for k in ("file", "path", "train_clip_ids", "val_clip_ids", "heldout_dataset"))
    )
    mode = split_cfg.get("mode", "episode" if has_episode_split_keys else "random_window")
    val_frac = float(_split_value(data_cfg, split_cfg, "val_frac", 0.0))
    seed = int(_split_value(data_cfg, split_cfg, "seed", 0))
    wcfg = _window_config(data_cfg, cfg.get("model"))

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
        if bool(_split_value(data_cfg, split_cfg, "val_from_train", False)):
            train_records = list(records)
        else:
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


def normalize_action_grip_contract(value: str | None) -> str:
    contract = str(value or "close01").strip().lower().replace("-", "_")
    aliases = {
        "close01": "close01",
        "legacy_close01": "close01",
        "signed": "signed_close",
        "signed_close": "signed_close",
        "close_signed": "signed_close",
        "signed_close_continuous_with_close01_supervision": "signed_close",
    }
    if contract not in aliases:
        raise ValueError(
            "train.action_grip_contract must be close01 or signed_close; "
            f"got {value!r}"
        )
    return aliases[contract]


def grip_to_close01(grip: torch.Tensor, contract: str) -> torch.Tensor:
    """Map the configured gripper contract to a soft BCE close target."""

    contract = normalize_action_grip_contract(contract)
    values = grip.float()
    if not torch.isfinite(values).all():
        raise ValueError("gripper target contains non-finite values")
    tolerance = 1e-4
    if contract == "signed_close":
        if bool(((values < -1.0 - tolerance) | (values > 1.0 + tolerance)).any()):
            raise ValueError("signed-close gripper target must be in [-1,1]")
        return ((values.clamp(-1.0, 1.0) + 1.0) * 0.5).to(dtype=grip.dtype)
    if bool(((values < -tolerance) | (values > 1.0 + tolerance)).any()):
        raise ValueError("close01 gripper target must be in [0,1]")
    return values.clamp(0.0, 1.0).to(dtype=grip.dtype)


def make_training_action_condition(
    action_tgt: torch.Tensor,
    action_tgt_norm: torch.Tensor,
    grip_contract: str,
) -> torch.Tensor:
    """Build factual action tokens without destroying signed gripper values."""

    contract = normalize_action_grip_contract(grip_contract)
    if contract == "close01":
        return make_action_condition(action_tgt, action_tgt_norm)
    if action_tgt.shape[-1] < 7:
        raise ValueError("signed-close action target must contain seven dimensions")
    if action_tgt_norm.shape != action_tgt[..., :6].shape:
        raise ValueError(
            f"normalized action shape {tuple(action_tgt_norm.shape)} does not match "
            f"pose target {tuple(action_tgt[..., :6].shape)}"
        )
    signed_grip = action_tgt[..., 6:7]
    # Validate through the close01 map, but preserve the raw signed value for
    # factual dynamics (DROID may be continuous rather than binary).
    grip_to_close01(signed_grip, contract)
    return torch.cat(
        [
            action_tgt_norm.to(device=action_tgt.device, dtype=action_tgt.dtype),
            signed_grip,
        ],
        dim=-1,
    )


def targets_with_close01_grip(tgt: dict, grip_contract: str) -> dict:
    """Return a shallow target copy whose BCE-facing grip fields are close01."""

    if "action_tgt" not in tgt:
        return tgt
    result = dict(tgt)
    action = tgt["action_tgt"].clone()
    close01 = tgt.get("action_grip_close01")
    if close01 is None:
        close01 = grip_to_close01(action[..., 6], grip_contract)
    close01 = close01.to(device=action.device, dtype=action.dtype)
    if close01.shape != action[..., 6].shape:
        raise ValueError(
            f"action_grip_close01 shape {tuple(close01.shape)} must match "
            f"{tuple(action[..., 6].shape)}"
        )
    action[..., 6] = close01
    result["action_tgt"] = action
    result["action_grip_close01"] = close01
    if "action_prev_grip_signed" in tgt:
        # Canonical OXE emits action_prev_grip in close01 for legacy policy
        # users and the signed physical value separately.  Use the signed
        # field here so it is converted exactly once.
        result["action_prev_grip"] = grip_to_close01(
            tgt["action_prev_grip_signed"], grip_contract
        )
    elif "action_prev_grip" in tgt:
        result["action_prev_grip"] = grip_to_close01(
            tgt["action_prev_grip"], grip_contract
        )
    return result


def batch_to_device(
    batch: dict,
    device: torch.device,
    k: int,
    direct_policy_only: bool = False,
    *,
    action_grip_contract: str = "close01",
    source_name: str | None = None,
    require_factual_action_contract: bool = False,
) -> tuple:
    s = batch["s_in"].to(device, non_blocking=True)
    c = batch["c"].to(device, non_blocking=True)
    context_rgb = None
    tgt = {}
    if direct_policy_only:
        if "rgb_in" in batch:
            context_rgb = batch["rgb_in"][:, -1].to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
            tgt["rgb_ref_p"] = context_rgb
    else:
        if s.shape[-1] == 2048:
            tgt["s_init_tgt"] = s[:, -1]
        else:
            tgt["s_init_tgt_codec"] = s[:, -1]
        if "s_tgt" in batch:
            tgt["s_tgt"] = batch["s_tgt"].to(device, non_blocking=True)
        if "s_tgt_codec" in batch:
            tgt["s_tgt_codec"] = batch["s_tgt_codec"].to(device, non_blocking=True)
        if "depth_tgt" in batch:
            tgt["depth_tgt"] = batch["depth_tgt"].to(device, non_blocking=True)
        if "depth_in" in batch:
            tgt["depth_ref"] = batch["depth_in"][:, -1].to(device, non_blocking=True)
        if "rgb_in" in batch and "rgb_tgt" in batch:
            context_rgb = batch["rgb_in"][:, -1].to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
            # rgb_tgt: [B, k, 256, 256, 3]; permute -> [B, k, 3, 256, 256]
            rgb_tgt_p = batch["rgb_tgt"].to(device, non_blocking=True).permute(0, 1, 4, 2, 3)
            tgt["rgb_tgt_p"] = rgb_tgt_p
            tgt["rgb_ref_p"] = context_rgb
    action_tgt = batch["action_tgt"].to(device, non_blocking=True)
    action_tgt_norm = batch["action_tgt_norm"].to(device, non_blocking=True)
    canonical_version = "wm3d_v7_base_delta_axisangle_gripclose_v1"
    action_stats_keys: list[str] | None = None
    if require_factual_action_contract:
        if source_name is None:
            raise ValueError("factual action contract validation requires source_name")
        action_valid = batch.get("action_valid")
        contract_versions = batch.get("action_contract_version")
        stats_keys = batch.get("action_stats_key")
        robocasa_source = source_name.startswith("robocasa_")
        if robocasa_source:
            # V7CompactWindowDataset validates action_valid_mask and immutable
            # adapter/audit identity while loading.  Adapt its explicit legacy
            # metadata into the common batch contract here.
            contract_keys = batch.get("action_contract_key")
            if contract_keys is None:
                raise RuntimeError(f"{source_name} lacks compact action_contract_key")
            if isinstance(contract_keys, str):
                contract_keys = [contract_keys] * int(action_tgt.shape[0])
            expected_key = f"robocasa365|5|{canonical_version}"
            if any(str(value) != expected_key for value in contract_keys):
                raise RuntimeError(
                    f"{source_name} compact action contract mismatch: {contract_keys}"
                )
            action_valid = torch.ones(
                int(action_tgt.shape[0]), dtype=torch.bool
            )
            contract_versions = [canonical_version] * int(action_tgt.shape[0])
            stats_keys = ["robocasa365"] * int(action_tgt.shape[0])
        if action_valid is None or not bool(torch.as_tensor(action_valid).bool().all()):
            raise RuntimeError(f"factual source {source_name!r} has invalid action samples")
        if contract_versions is None:
            raise RuntimeError(f"factual source {source_name!r} lacks action_contract_version")
        if isinstance(contract_versions, str):
            contract_versions = [contract_versions] * int(action_tgt.shape[0])
        if any(str(value) != canonical_version for value in contract_versions):
            raise RuntimeError(
                f"factual source {source_name!r} action contract version mismatch: "
                f"{contract_versions}"
            )
        if stats_keys is None:
            raise RuntimeError(f"factual source {source_name!r} lacks action_stats_key")
        if isinstance(stats_keys, str):
            stats_keys = [stats_keys] * int(action_tgt.shape[0])
        action_stats_keys = [str(value) for value in stats_keys]
        if len(action_stats_keys) != int(action_tgt.shape[0]):
            raise RuntimeError("action_stats_key batch length mismatch")
        if len(set(action_stats_keys)) != 1:
            raise RuntimeError(
                f"source-homogeneous batch mixed action stats: {action_stats_keys}"
            )
        for label, values in (
            ("action_tgt", action_tgt),
            ("action_tgt_norm", action_tgt_norm),
        ):
            if not bool(torch.isfinite(values).all()):
                raise RuntimeError(
                    f"factual source {source_name!r} has non-finite {label}"
                )
    tgt["action_tgt"] = action_tgt
    tgt["action_tgt_norm"] = action_tgt_norm
    if "action_grip_close01" in batch:
        action_grip_close01 = batch["action_grip_close01"].to(
            device, non_blocking=True
        )
    else:
        if require_factual_action_contract and not str(source_name).startswith("robocasa_"):
            raise RuntimeError(
                f"factual OXE source {source_name!r} lacks action_grip_close01"
            )
        action_grip_close01 = grip_to_close01(
            action_tgt[..., 6], action_grip_contract
        )
    tgt["action_grip_close01"] = action_grip_close01
    if require_factual_action_contract and not bool(
        torch.isfinite(action_grip_close01).all()
    ):
        raise RuntimeError(
            f"factual source {source_name!r} has non-finite action_grip_close01"
        )
    if action_stats_keys is not None:
        tgt["action_stats_keys"] = action_stats_keys
        tgt["action_contract_version"] = canonical_version
    if "action_prev_grip" in batch:
        tgt["action_prev_grip"] = batch["action_prev_grip"].to(device, non_blocking=True)
    for key in (
        "point_tgt",
        "point_conf_tgt",
        "pose_geom_tgt",
        "pose_geom_conf_tgt",
        "depth_conf_tgt",
        "lowdim_state",
        "object_state",
        "plan_state",
        "action_history",
        "s_wrist",
        "view_mask",
        "anchor_camera_pose",
        "wrist_camera_pose",
        "branch_actions",
        "branch_s_tgt",
        "branch_s_tgt_codec",
        "branch_valid",
        "branch_rewards",
        "branch_dones",
        "branch_success",
        "action_pose_mean",
        "action_pose_std",
        "action_prev_grip_signed",
        "left_action",
        "right_action",
        "renderer_action_cond",
        "renderer_zero_action_cond",
        "native_action_cond",
    ):
        if key in batch:
            tgt[key] = batch[key].to(device, non_blocking=True)
    if "progress_tgt" in batch:
        tgt["progress_tgt"] = batch["progress_tgt"].to(device, non_blocking=True)
    if "terminal_success_tgt" in batch:
        tgt["terminal_success_tgt"] = batch["terminal_success_tgt"].to(device, non_blocking=True)
    if "plausibility_tgt" in batch:
        tgt["plausibility_tgt"] = batch["plausibility_tgt"].to(device, non_blocking=True)
    action_cond = make_training_action_condition(
        action_tgt, action_tgt_norm, action_grip_contract
    )
    if "native_action_cond" in batch:
        native_action_cond = batch["native_action_cond"].to(
            device, non_blocking=True
        )
        expected_prefix = (int(action_tgt.shape[0]), int(k))
        if (
            native_action_cond.ndim != 3
            or tuple(native_action_cond.shape[:2]) != expected_prefix
            or int(native_action_cond.shape[-1]) != 14
        ):
            raise RuntimeError(
                "native_action_cond must be exact [B,K,14], got "
                f"{tuple(native_action_cond.shape)} for B={expected_prefix[0]} K={k}"
            )
        if not bool(torch.isfinite(native_action_cond).all()):
            raise RuntimeError("native_action_cond contains non-finite values")
        action_cond = native_action_cond
    return s, c, action_cond, context_rgb, tgt


def multiview_kwargs_from_targets(tgt: dict) -> dict:
    mapping = {
        "s_wrist": "wrist_s",
        "view_mask": "view_mask",
        "anchor_camera_pose": "anchor_camera_pose",
        "wrist_camera_pose": "wrist_camera_pose",
    }
    return {destination: tgt[source] for source, destination in mapping.items() if source in tgt}


def decode_codec_targets(model: torch.nn.Module, tgt: dict) -> None:
    target_model = model.module if isinstance(model, DDP) else model
    if "s_init_tgt_codec" in tgt:
        tgt["s_init_tgt"] = target_model.decode_input_tokens(tgt["s_init_tgt_codec"]).detach()
    if "s_tgt_codec" in tgt:
        tgt["s_tgt"] = target_model.decode_input_tokens(tgt["s_tgt_codec"]).detach()
    if "branch_s_tgt_codec" in tgt:
        tgt["branch_s_tgt"] = target_model.decode_input_tokens(tgt["branch_s_tgt_codec"]).detach()


def compute_true_branch_future_value_losses(
    out: dict[str, torch.Tensor],
    tgt: dict[str, torch.Tensor],
    train_cfg: dict,
) -> dict[str, torch.Tensor]:
    """Compute Stage 1 value losses only from audited true branch outcomes."""

    if float(train_cfg.get("future_value_weight", 0.0) or 0.0) <= 0.0:
        return {}
    has_actions = "branch_actions" in tgt
    has_success = "branch_success" in tgt
    if not has_actions and not has_success:
        # Large-scale factual action batches are valid in the joint S1 run;
        # value/ranking supervision is applied only on audited branch batches.
        return {}
    if has_actions != has_success:
        raise RuntimeError(
            "true branch batch must provide branch_actions and branch_success together"
        )
    required_out = {"candidate_progress_logit", "candidate_success_logit"}
    missing_out = sorted(required_out - out.keys())
    if missing_out:
        raise RuntimeError(
            "future_value_weight requires enable_future_value and candidate actions; "
            f"missing outputs: {missing_out}"
        )
    if "branch_success" not in tgt:
        raise RuntimeError(
            "future value training requires true simulator branch_success; pseudo outcomes are forbidden"
        )
    return true_branch_future_value_loss(
        out["candidate_progress_logit"],
        out["candidate_success_logit"],
        tgt["branch_success"],
        branch_valid=tgt.get("branch_valid"),
        cfg=FutureValueLossConfig(
            trajectory_weight=float(train_cfg.get("future_value_trajectory_weight", 1.0)),
            terminal_weight=float(train_cfg.get("future_value_terminal_weight", 1.0)),
            ranking_weight=float(train_cfg.get("future_value_ranking_weight", 0.0)),
            positive_weight=float(train_cfg.get("future_value_positive_weight", 1.0)),
        ),
    )


def compute_future_value_task_swap_metrics(
    future_value_head: torch.nn.Module,
    candidate_futures: torch.Tensor,
    task_emb: torch.Tensor,
    base_out: dict[str, torch.Tensor],
    branch_success: torch.Tensor,
    branch_valid: torch.Tensor | None,
    task_bank: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Measure whether Stage 1 value predictions depend on the correct task."""

    if task_emb.shape[0] > 1:
        wrong_task = task_emb.roll(1, dims=0)
    elif task_bank:
        query = F.normalize(task_emb.float(), dim=-1)
        bank = torch.cat(task_bank, dim=0).to(task_emb.device)
        distance = 1.0 - query @ F.normalize(bank.float(), dim=-1).T
        wrong_task = bank[distance.argmax(dim=1)].to(task_emb.dtype)
    else:
        wrong_task = None
    current = task_emb.detach()
    for row in current:
        row = row[None]
        if not task_bank or all(
            not torch.equal(row, previous.to(row.device)) for previous in task_bank
        ):
            task_bank.append(row)
            if len(task_bank) > 32:
                task_bank.pop(0)
    if wrong_task is None or torch.equal(wrong_task, task_emb):
        return {}
    swapped = future_value_head(candidate_futures.detach(), wrong_task)
    target = branch_success.bool().any(dim=-1).float()
    valid = (
        torch.ones_like(target, dtype=torch.bool)
        if branch_valid is None
        else branch_valid.bool()
    )
    base_bce = F.binary_cross_entropy_with_logits(
        base_out["candidate_success_logit"].float(), target, reduction="none"
    )[valid].mean()
    swapped_bce = F.binary_cross_entropy_with_logits(
        swapped["candidate_success_logit"].float(), target, reduction="none"
    )[valid].mean()
    base_rank_terms = []
    swapped_rank_terms = []
    for base_row, swapped_row, target_row, valid_row in zip(
        base_out["candidate_success_logit"].float(),
        swapped["candidate_success_logit"].float(),
        target,
        valid,
    ):
        positive = (target_row > 0.5) & valid_row
        negative = (target_row <= 0.5) & valid_row
        if bool(positive.any()) and bool(negative.any()):
            base_margin = base_row[positive][:, None] - base_row[negative][None, :]
            swapped_margin = (
                swapped_row[positive][:, None] - swapped_row[negative][None, :]
            )
            base_rank_terms.append(F.softplus(-base_margin).mean())
            swapped_rank_terms.append(F.softplus(-swapped_margin).mean())
    if base_rank_terms:
        base_rank = torch.stack(base_rank_terms).mean()
        swapped_rank = torch.stack(swapped_rank_terms).mean()
        rank_delta = swapped_rank - base_rank
    else:
        rank_delta = base_bce.new_zeros(())
    sensitivity = (
        base_out["candidate_success_logit"].float()
        - swapped["candidate_success_logit"].float()
    ).abs()[valid].mean()
    return {
        "future_value_task_swap_logit_l1": sensitivity.detach(),
        "future_value_task_swap_bce_delta": (swapped_bce - base_bce).detach(),
        "future_value_task_swap_ranking_delta": rank_delta.detach(),
    }


def validate_future_value_stage_preflight(cfg: dict, args: argparse.Namespace) -> bool:
    """Fail closed before a Stage 1 run can mutate or learn from bad labels."""

    train_cfg = cfg.get("train") or {}
    if float(train_cfg.get("future_value_weight", 0.0) or 0.0) <= 0.0:
        return False
    errors: list[str] = []
    data_cfg = cfg.get("data") or {}
    model_cfg = cfg.get("model") or {}
    dataset_type = data_cfg.get("dataset_type")
    if dataset_type not in {"v7_same_root_branch", "v7_mixed"}:
        errors.append(
            "future value requires dataset_type=v7_same_root_branch or v7_mixed"
        )
    if not bool(model_cfg.get("enable_future_value")):
        errors.append("future value requires model.enable_future_value=true")
    future_value_only = bool(train_cfg.get("future_value_only"))
    if dataset_type == "v7_mixed" and future_value_only:
        errors.append("v7_mixed S1 requires joint action/value training")
    trainable_prefixes = list(train_cfg.get("trainable_prefixes") or [])
    if future_value_only:
        if trainable_prefixes != ["future_value_head."]:
            errors.append(
                "future-value-only training requires trainable_prefixes exactly "
                "future_value_head."
            )
    else:
        allowed_joint_prefixes = {
            "dual.",
            "multiview_fuser.",
            "dual.state.action_cond_proj.",
            "dual.state.action_cond_pos",
            "dual.state.cond_proj.",
            "dual.state.decoder.",
            "dual.state.out_proj.",
            "dual.action.action_cond_proj.",
            "dual.action.action_cond_pos",
            "dual.action.cond_proj.",
            "dual.action.decoder.",
            "dual.action.z_head.",
            "dual.xattn_blocks.",
            "dual.action_up.",
            "dual.action_down.",
            "action_proj.",
            "action_policy.",
            "future_value_head.",
        }
        observed = set(trainable_prefixes)
        if (
            not any(prefix.startswith("dual.") for prefix in observed)
            or "future_value_head." not in observed
        ):
            errors.append(
                "joint action/value Stage 1 requires trainable dual. and "
                "future_value_head."
            )
        unexpected = sorted(observed - allowed_joint_prefixes)
        if unexpected:
            errors.append(
                "joint action/value Stage 1 has forbidden trainable prefixes: "
                f"{unexpected}"
            )
        if float(train_cfg.get("true_branch_weight", 0.0) or 0.0) <= 0.0:
            errors.append(
                "joint action/value Stage 1 requires a positive true_branch_weight"
            )
        policy_flow_weight = float(
            train_cfg.get("policy_flow_weight", 0.0) or 0.0
        )
        if policy_flow_weight > 0.0:
            if not bool(model_cfg.get("enable_action_policy")):
                errors.append(
                    "policy_flow_weight requires model.enable_action_policy=true"
                )
            if not bool(model_cfg.get("policy_enable_flow_head")):
                errors.append(
                    "policy_flow_weight requires model.policy_enable_flow_head=true"
                )
            if not bool(model_cfg.get("policy_flow_use_as_policy")):
                errors.append(
                    "formal S1 flow proposals require "
                    "model.policy_flow_use_as_policy=true"
                )
            if "action_policy." not in observed:
                errors.append(
                    "policy_flow_weight requires trainable action_policy. tensors"
                )
            policy_context_source = str(
                model_cfg.get("policy_context_source", "input")
            ).strip().lower()
            input_sources = {"", "input", "s", "tokens", "cached"}
            action_free_future_sources = {
                "core",
                "core_pred",
                "pred",
                "pred_tokens",
                "serving",
            }
            if policy_context_source in action_free_future_sources:
                policy_core_action_cond = str(
                    model_cfg.get("policy_core_action_cond", "same")
                ).strip().lower()
                if policy_core_action_cond not in {
                    "none",
                    "no_action",
                    "off",
                    "disabled",
                }:
                    errors.append(
                        "future-conditioned S1 flow proposals must use an "
                        "action-free core prediction; got "
                        f"policy_core_action_cond={policy_core_action_cond!r}"
                    )
                native_future_weight = float(
                    train_cfg.get("native_future_no_teacher_weight", 0.0) or 0.0
                )
                if not math.isfinite(native_future_weight) or native_future_weight <= 0.0:
                    errors.append(
                        "future-conditioned S1 flow proposals require a positive "
                        "native_future_no_teacher_weight so core_pred is explicitly "
                        "anchored to the demonstrated future"
                    )
            elif policy_context_source not in input_sources:
                errors.append(
                    "joint S1 action-flow proposals must use current inputs or "
                    "an action-free core future without teacher-action leakage; got "
                    f"policy_context_source={policy_context_source!r}"
                )
            policy_horizon = int(
                model_cfg.get("policy_horizon") or data_cfg.get("k") or 0
            )
            dynamics_horizon = int(data_cfg.get("k") or 0)
            if policy_horizon != dynamics_horizon or policy_horizon <= 0:
                errors.append(
                    "joint S1 action-flow proposal horizon must equal the native "
                    f"dynamics horizon; policy={policy_horizon} dynamics={dynamics_horizon}"
                )
    condition_dropout = train_cfg.get("condition_dropout") or {}
    dropout_enabled = bool(condition_dropout.get("enabled"))
    forbidden_dropout = {
        name: float(condition_dropout.get(name, 0.0) or 0.0)
        for name in (
            "text_only_p",
            "text_context_p",
            "text_action_p",
            "context_p",
        )
    }
    nonzero_forbidden = {
        name: value
        for name, value in forbidden_dropout.items()
        if dropout_enabled and value != 0.0
    }
    action_p = (
        float(condition_dropout.get("action_p", 0.0) or 0.0)
        if dropout_enabled
        else 0.0
    )
    native_no_teacher_weight = float(
        train_cfg.get("native_action_no_teacher_weight", 0.0) or 0.0
    )
    if not math.isfinite(native_no_teacher_weight) or native_no_teacher_weight < 0.0:
        errors.append(
            "native_action_no_teacher_weight must be finite and non-negative; "
            f"got {native_no_teacher_weight}"
        )
    native_no_teacher_enabled = native_no_teacher_weight > 0.0

    # Text/context dropout corrupts task-grounded value labels.  Joint S1 has
    # two mutually exclusive native-action modes:
    #   1. legacy bounded action-only dropout on the main world pass; or
    #   2. a decoupled no-teacher action-only pass while the main world pass
    #      remains action-conditioned (action_p == 0).
    # The corrected V7 path uses mode 2 so policy supervision cannot turn the
    # action-conditioned world prediction into an action-ambiguous average.
    if future_value_only:
        if nonzero_forbidden or action_p != 0.0 or native_no_teacher_enabled:
            errors.append(
                "future-value-only Stage 1 forbids condition dropout and the "
                "native no-teacher action path"
            )
    else:
        if nonzero_forbidden:
            errors.append(
                "task-grounded S1 permits no text/context dropout; only "
                f"action-only dropout is allowed, got {nonzero_forbidden}"
            )
        if native_no_teacher_enabled:
            if action_p != 0.0:
                errors.append(
                    "decoupled native no-teacher action training requires "
                    "condition_dropout.action_p == 0; do not combine it with "
                    f"legacy action-only dropout (action_p={action_p})"
                )
        elif dropout_enabled and not (0.0 < action_p <= 0.5):
            errors.append(
                "task-grounded S1 condition dropout requires either joint "
                "action-only dropout with 0 < action_p <= 0.5 or a positive "
                "native_action_no_teacher_weight with action_p == 0; "
                f"action_p={action_p}"
            )
    if args.resume is None or not args.reset_optim:
        errors.append("Stage 1 requires --resume STAGE0_CKPT --reset_optim")
    elif not Path(args.resume).is_file():
        errors.append(f"Stage 1 resume checkpoint does not exist: {args.resume}")

    branch_index = data_cfg.get("branch_index")
    if not branch_index:
        errors.append("future value requires data.branch_index")
    else:
        index_path = Path(branch_index)
        report_path = index_path.with_suffix(".report.json")
        if not report_path.is_file():
            errors.append(f"missing immutable branch report: {report_path}")
        else:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            required = {
                "same_root_current_runtime_exact": True,
                "historical_runtime_reconstruction_exact": False,
                "pseudo_outcomes": False,
                "task_embedding_real": True,
                "outcome_reduction": "20hz_blocks_any_success_any_done_max_reward_v1",
            }
            for key, expected in required.items():
                if report.get(key) != expected:
                    errors.append(f"branch report {key}={report.get(key)!r}, expected {expected!r}")
            formal_precondition = (
                (cfg.get("contract") or {}).get("formal_branch_precondition")
                or {}
            )
            if str((cfg.get("contract") or {}).get("profile", "")) == "formal":
                outcome = report.get("outcome_audit") or {}
                for split, minimum_key in (
                    ("train", "minimum_train_roots"),
                    ("val", "minimum_val_roots"),
                    ("test", "minimum_test_roots"),
                ):
                    observed_roots = int((outcome.get(split) or {}).get("roots", 0))
                    required_roots = int(formal_precondition.get(minimum_key, 0))
                    if observed_roots < required_roots:
                        errors.append(
                            f"formal branch {split} roots={observed_roots}, "
                            f"need {required_roots}"
                        )
                observed_mixed = int(
                    (outcome.get("train") or {}).get("mixed_roots", 0)
                )
                required_mixed = int(
                    formal_precondition.get("minimum_train_mixed_roots", 0)
                )
                if observed_mixed < required_mixed:
                    errors.append(
                        f"formal branch train mixed_roots={observed_mixed}, "
                        f"need {required_mixed}"
                    )
                if bool(
                    formal_precondition.get("require_root_episode_disjoint", False)
                ) and report.get("root_episode_disjoint") is not True:
                    errors.append("formal branch report lacks root_episode_disjoint=true")
                if bool(
                    formal_precondition.get("require_split_sha256", False)
                ) and not report.get("split_identity_sha256"):
                    errors.append("formal branch report lacks split_identity_sha256")
            if not index_path.is_file():
                errors.append(f"missing branch index: {index_path}")
            else:
                actual_index_sha = sha256_path(index_path)
                if report.get("output_index_sha256") != actual_index_sha:
                    errors.append(
                        "branch report/index SHA mismatch: "
                        f"{report.get('output_index_sha256')} != {actual_index_sha}"
                    )
            audit = report.get("outcome_audit") or {}
            min_tasks = int(train_cfg.get("future_value_min_tasks", 1))
            min_tasks_per_split = int(
                train_cfg.get("future_value_min_tasks_per_split", 1)
            )
            if int(report.get("tasks", 0)) < min_tasks:
                errors.append(
                    f"same-root report has {report.get('tasks', 0)} tasks; requires {min_tasks}"
                )
            for split in ("train", "val"):
                split_audit = audit.get(split) or {}
                if int(split_audit.get("positive_branches", 0)) <= 0:
                    errors.append(f"{split} has no positive true terminal branch")
                if int(split_audit.get("negative_branches", 0)) <= 0:
                    errors.append(f"{split} has no negative true terminal branch")
                if int(split_audit.get("mixed_roots", 0)) <= 0:
                    errors.append(f"{split} has no same-root mixed true outcome")
                if int(split_audit.get("tasks", 0)) < min_tasks_per_split:
                    errors.append(
                        f"{split} has {split_audit.get('tasks', 0)} tasks; "
                        f"requires {min_tasks_per_split}"
                    )
            # The immutable index/report hashes are checked on every rank, but
            # decoding every branch payload is a host-filesystem audit.  Doing
            # that 14/16 times before DDP initialization only multiplies I/O;
            # rank 0 performs the exact payload audit and torchrun aborts the
            # whole job if it fails.
            preflight_rank = int(os.environ.get("RANK", "0") or 0)
            if index_path.is_file() and preflight_rank == 0:
                actual_audit = audit_true_branch_outcomes(index_path)
                for split in ("train", "val"):
                    expected_split = audit.get(split) or {}
                    if actual_audit[split] != {
                        key: int(expected_split.get(key, -1))
                        for key in (
                            "roots",
                            "positive_branches",
                            "negative_branches",
                            "mixed_roots",
                            "tasks",
                        )
                    }:
                        errors.append(
                            f"{split} branch payload outcome audit mismatch: "
                            f"actual={actual_audit[split]} report={expected_split}"
                        )
    if errors:
        raise RuntimeError("future value Stage 1 preflight failed:\n- " + "\n- ".join(errors))
    return True


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_true_branch_outcomes(index_path: Path) -> dict[str, dict[str, int]]:
    """Recompute the Stage 1 signal gate from immutable payload contents."""

    audit = {
        split: {
            "roots": 0,
            "positive_branches": 0,
            "negative_branches": 0,
            "mixed_roots": 0,
            "task_names": set(),
        }
        for split in ("train", "val", "test")
    }
    with index_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise RuntimeError(f"same-root branch index is empty: {index_path}")
    for row in rows:
        split = str(row.get("split"))
        if split not in audit:
            raise RuntimeError(f"invalid same-root split: {split}")
        if (
            row.get("same_root_current_runtime_exact") is not True
            or row.get("historical_runtime_reconstruction_exact") is not False
            or row.get("pseudo_outcomes") is not False
        ):
            raise RuntimeError(f"unqualified same-root row: {row.get('root_id')}")
        with np.load(row["path"], allow_pickle=False) as archive:
            success = np.asarray(archive["branch_success"], dtype=np.bool_)
            valid = np.asarray(archive["branch_valid"], dtype=np.bool_)
        if success.ndim != 2 or valid.shape != (success.shape[0],) or not valid.all():
            raise RuntimeError(f"invalid same-root outcome payload: {row.get('root_id')}")
        terminal = success.any(axis=-1)
        audit[split]["roots"] += 1
        audit[split]["positive_branches"] += int(terminal.sum())
        audit[split]["negative_branches"] += int((~terminal).sum())
        audit[split]["mixed_roots"] += int(terminal.any() and not terminal.all())
        audit[split]["task_names"].add(str(row.get("task") or row.get("task_text") or ""))
    return {
        split: {
            **{key: value for key, value in values.items() if key != "task_names"},
            "tasks": len(values["task_names"]),
        }
        for split, values in audit.items()
    }


def validate_future_value_resume_load(
    load_result,
    *,
    allowed_missing_prefixes: tuple[str, ...] = ("future_value_head.",),
) -> None:
    """Allow only explicitly declared new S1 heads during a Stage 0 transition."""

    missing = list(getattr(load_result, "missing_keys", []) or [])
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
    skipped = list(getattr(load_result, "skipped_keys", []) or [])
    expanded = list(getattr(load_result, "expanded_keys", []) or [])
    if not allowed_missing_prefixes:
        raise ValueError("allowed_missing_prefixes must not be empty")
    bad_missing = [
        key for key in missing if not key.startswith(allowed_missing_prefixes)
    ]
    if bad_missing or unexpected or skipped or expanded:
        raise RuntimeError(
            "Stage 1 checkpoint is not a strict Stage 0-compatible world: "
            f"bad_missing={bad_missing[:20]} unexpected={unexpected[:20]} "
            f"skipped={skipped[:20]} expanded={expanded[:20]}"
        )


def validate_action_policy_resume_load(load_result) -> None:
    """S1 action policy may initialize only newly introduced policy tensors."""

    missing = list(getattr(load_result, "missing_keys", []) or [])
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
    skipped = list(getattr(load_result, "skipped_keys", []) or [])
    expanded = list(getattr(load_result, "expanded_keys", []) or [])
    bad_missing = [key for key in missing if not key.startswith("action_policy.")]
    if bad_missing or unexpected or skipped or expanded or not missing:
        raise RuntimeError(
            "Stage 1 action-policy checkpoint is not a strict Stage 0-compatible world: "
            f"bad_missing={bad_missing[:20]} unexpected={unexpected[:20]} "
            f"skipped={skipped[:20]} expanded={expanded[:20]} "
            f"policy_missing={len(missing)}"
        )


def validate_action_pretraining_preflight(cfg: dict) -> bool:
    """Reject ambiguous direct/flow policy training before CUDA is initialized.

    The legacy S1 configuration passed flow-matching inputs while supervising
    ``policy_*`` as the direct head. With ``policy_flow_use_as_policy=true``
    those keys are the one-step flow reconstruction, so the deterministic
    regression output heads receive no direct-policy loss. New action
    pretraining must name the base head explicitly and keep flow sampling an
    opt-in proposal path at serving time.
    """

    train_cfg = cfg.get("train") or {}
    model_cfg = cfg.get("model") or {}
    data_cfg = cfg.get("data") or {}
    direct_weight = float(train_cfg.get("direct_policy_weight", 0.0) or 0.0)
    flow_weight = float(train_cfg.get("policy_flow_weight", 0.0) or 0.0)
    joint_native = bool(train_cfg.get("joint_native_action_pretraining", False))
    if direct_weight <= 0.0 and flow_weight <= 0.0:
        return False

    errors: list[str] = []
    if not bool(model_cfg.get("enable_action_policy")):
        errors.append("action pretraining requires model.enable_action_policy=true")
    trainable_prefixes = tuple(train_cfg.get("trainable_prefixes") or ())
    if not joint_native and not any(
        str(prefix).startswith("action_policy.") for prefix in trainable_prefixes
    ):
        errors.append("action pretraining requires trainable action_policy. tensors")
    if joint_native and trainable_prefixes:
        errors.append(
            "joint native action pretraining must train the full WM3D model; "
            "leave trainable_prefixes empty"
        )

    direct_head = str(train_cfg.get("direct_policy_head", "policy")).strip().lower()
    valid_direct_heads = {
        "policy", "direct", "full", "base", "base_policy", "prior",
        "prior_policy", "oxe_prior",
    }
    if direct_weight > 0.0 and direct_head not in valid_direct_heads:
        errors.append(f"unsupported train.direct_policy_head={direct_head!r}")
    if flow_weight > 0.0 and not bool(model_cfg.get("policy_enable_flow_head")):
        errors.append("policy_flow_weight requires model.policy_enable_flow_head=true")

    flow_overrides_policy = bool(model_cfg.get("policy_flow_use_as_policy"))
    if (
        direct_weight > 0.0
        and flow_weight > 0.0
        and flow_overrides_policy
        and direct_head in {"policy", "direct", "full"}
    ):
        errors.append(
            "direct+flow training would supervise the flow reconstruction as the "
            "direct head; set train.direct_policy_head=base"
        )

    if joint_native:
        context_source = str(
            model_cfg.get("policy_context_source", "input")
        ).strip().lower()
        action_free_core_sources = {
            "core",
            "core_pred",
            "pred",
            "pred_tokens",
            "serving",
        }
        if bool(train_cfg.get("direct_policy_only", False)):
            errors.append(
                "joint native action pretraining requires direct_policy_only=false"
            )
        if direct_weight <= 0.0:
            errors.append(
                "joint native action pretraining requires positive direct_policy_weight"
            )
        if direct_head not in {"base", "base_policy"}:
            errors.append(
                "joint native action pretraining requires direct_policy_head=base"
            )
        if context_source not in action_free_core_sources:
            errors.append(
                "joint native action pretraining policy must read differentiable "
                "WM3D core predictions"
            )
        native_future_weight = float(
            train_cfg.get("native_future_no_teacher_weight", 0.0) or 0.0
        )
        if not math.isfinite(native_future_weight) or native_future_weight <= 0.0:
            errors.append(
                "joint native action pretraining with core_pred requires a positive "
                "native_future_no_teacher_weight so the action-free future remains "
                "anchored to demonstrated native-3D targets"
            )
        core_action_mode = str(
            model_cfg.get("policy_core_action_cond", "same")
        ).strip().lower()
        if core_action_mode not in {"none", "no_action", "off", "disabled"}:
            errors.append(
                "joint native action pretraining policy core must be action-free"
            )
        if bool(model_cfg.get("policy_flow_use_as_policy", False)):
            errors.append(
                "direct policy must remain the serving owner during joint pretraining"
            )
        model_history = int(model_cfg.get("policy_action_history_len", 0) or 0)
        data_history = int(
            data_cfg.get("policy_action_history_len", model_history) or 0
        )
        if model_history < 1:
            errors.append(
                "joint native action pretraining requires causal canonical action history"
            )
        if data_history != model_history:
            errors.append("data/model policy_action_history_len must match")
        if not bool(train_cfg.get("direct_policy_grip_partition_contract", False)):
            errors.append(
                "joint native action pretraining requires the strict gripper "
                "partition contract"
            )
        grip_owner = str(
            train_cfg.get("direct_policy_grip_owner", "auto")
        ).strip().lower()
        model_grip_owner = str(
            model_cfg.get("policy_grip_owner", "auto")
        ).strip().lower()
        if grip_owner not in {"absolute", "delta_composed"}:
            errors.append(
                "joint native action pretraining requires one explicit gripper owner"
            )
        if model_grip_owner != grip_owner:
            errors.append(
                "model.policy_grip_owner must equal train.direct_policy_grip_owner"
            )
        if not bool(
            train_cfg.get("direct_policy_require_action_prev_grip", False)
        ):
            errors.append(
                "joint native action pretraining requires action_prev_grip evidence"
            )
        if grip_owner == "absolute":
            if float(
                train_cfg.get("direct_policy_grip_natural_bce_weight", 0.0) or 0.0
            ) <= 0.0:
                errors.append(
                    "absolute gripper ownership requires a positive natural-distribution "
                    "gripper BCE weight so the deployment threshold remains calibrated"
                )
        elif grip_owner == "delta_composed":
            if not bool(model_cfg.get("policy_enable_grip_delta_head", False)):
                errors.append(
                    "delta_composed gripper ownership requires "
                    "model.policy_enable_grip_delta_head=true"
                )
            if not bool(
                model_cfg.get("policy_grip_delta_use_composed_action_cond", False)
            ):
                errors.append(
                    "delta_composed gripper ownership requires causal composed execution"
                )
            if not bool(
                model_cfg.get("policy_grip_delta_soft_compose_action_cond", False)
            ):
                errors.append(
                    "delta_composed gripper ownership requires differentiable soft composition"
                )
            if not bool(
                model_cfg.get("policy_grip_delta_straight_through_action_cond", False)
            ):
                errors.append(
                    "delta_composed gripper ownership requires straight-through hard serving"
                )
            if float(
                train_cfg.get("direct_policy_grip_delta_ce_weight", 0.0) or 0.0
            ) <= 0.0:
                errors.append(
                    "delta_composed gripper ownership requires positive event CE"
                )
            if float(
                train_cfg.get("direct_policy_grip_delta_natural_ce_weight", 0.0) or 0.0
            ) <= 0.0:
                errors.append(
                    "delta_composed gripper ownership requires positive natural event CE "
                    "to calibrate hold/up/down priors"
                )
            absolute_owner_weights = {
                key: float(train_cfg.get(key, 0.0) or 0.0)
                for key in (
                    "direct_policy_grip_weight",
                    "direct_policy_first_grip_weight",
                    "direct_policy_grip_natural_bce_weight",
                    "direct_policy_first_grip_natural_bce_weight",
                )
            }
            nonzero_absolute = [
                key for key, value in absolute_owner_weights.items() if value != 0.0
            ]
            if nonzero_absolute:
                errors.append(
                    "delta_composed gripper ownership forbids absolute-head serving losses: "
                    + ", ".join(nonzero_absolute)
                )
        factual = train_cfg.get("factual_action_conditioning") or {}
        if not bool(factual.get("enabled", False)) or int(
            factual.get("start_step", -1)
        ) != 0:
            errors.append(
                "joint native action pretraining must preserve factual action "
                "conditioning from step 0"
            )

    if bool(train_cfg.get("enforce_separate_direct_flow_heads", False)):
        if direct_weight <= 0.0 or flow_weight <= 0.0:
            errors.append(
                "enforce_separate_direct_flow_heads requires positive direct_policy_weight "
                "and policy_flow_weight"
            )
        if direct_head not in {"base", "base_policy"}:
            errors.append(
                "separated direct/flow training requires train.direct_policy_head=base"
            )
        if flow_overrides_policy:
            errors.append(
                "separated direct/flow serving requires model.policy_flow_use_as_policy=false"
            )
        if not bool(train_cfg.get("direct_policy_only")):
            errors.append(
                "the repaired 1B action warm-up requires train.direct_policy_only=true"
            )
        if trainable_prefixes != ("action_policy.",):
            errors.append(
                "the repaired 1B action warm-up requires trainable_prefixes exactly action_policy."
            )
        context_source = str(
            train_cfg.get("direct_policy_context_source", "input")
        ).strip().lower()
        input_sources = {"", "input", "s", "tokens", "cached"}
        core_sources = {
            "core", "core_detach", "core_pred", "core_pred_detach", "pred",
            "pred_detach", "pred_tokens", "pred_tokens_detach", "serving",
            "serving_detach",
        }
        if context_source not in input_sources | core_sources:
            errors.append(
                "repaired action pretraining requires an observed-input or frozen-core context; "
                f"got direct_policy_context_source={context_source!r}"
            )
        if context_source in core_sources:
            core_action_mode = str(
                train_cfg.get("direct_policy_core_action_cond", "none")
            ).strip().lower()
            if core_action_mode not in {"none", "no_action", "off", "disabled"}:
                errors.append(
                    "frozen-core action pretraining must be action-free; "
                    f"got direct_policy_core_action_cond={core_action_mode!r}"
                )

    if errors:
        raise RuntimeError("; ".join(errors))
    return True


def validate_stage_transition_preflight(cfg: dict, args: argparse.Namespace) -> str | None:
    """Require the immutable predecessor declared by a formal V7 stage."""

    transition = (cfg.get("train") or {}).get("stage_transition")
    if not transition:
        return None
    if not isinstance(transition, dict):
        raise RuntimeError("train.stage_transition must be a mapping")
    expected_raw = transition.get("required_resume_path")
    if not expected_raw:
        raise RuntimeError("stage_transition requires required_resume_path")
    expected = Path(expected_raw).resolve()
    actual = Path(args.resume).resolve() if args.resume is not None else None
    errors: list[str] = []
    if actual is None:
        errors.append(f"formal stage requires --resume {expected}")
    elif actual != expected:
        errors.append(f"formal stage resume mismatch: got {actual}, expected {expected}")
    elif not actual.is_file():
        errors.append(f"formal stage predecessor checkpoint does not exist: {actual}")
    if bool(transition.get("require_reset_optim", True)) and not args.reset_optim:
        errors.append("formal stage transition requires --reset_optim")
    load_mode = str(transition.get("load_mode") or "exact")
    if load_mode not in {
        "v6_native",
        "exact",
        "stage0_to_value",
        "stage0_to_policy",
        "stage0_to_value_policy",
    }:
        errors.append(f"unsupported stage_transition.load_mode={load_mode!r}")
    if errors:
        raise RuntimeError("; ".join(errors))
    return load_mode


def validate_empty_checkpoint_dir_preflight(
    cfg: dict,
    *,
    resume_checkpoint: Path | None = None,
) -> None:
    """Keep a fresh stage lineage from silently reusing an old output run."""

    if resume_checkpoint is not None:
        return
    out_cfg = cfg.get("out") or {}
    if not bool(out_cfg.get("require_empty_checkpoint_dir", False)):
        return
    checkpoint_dir = Path(out_cfg["root"]) / str(out_cfg.get("ckpt_dir", "ckpt"))
    existing = sorted(checkpoint_dir.glob("step_*.pt")) if checkpoint_dir.exists() else []
    if existing:
        raise RuntimeError(
            "new-stage output checkpoint directory is not empty: "
            f"{existing[-1]}"
        )


def validate_exact_stage_resume_load(load_result) -> None:
    """A same-architecture stage transition may not skip or invent tensors."""

    fields = {
        name: list(getattr(load_result, name, []) or [])
        for name in ("missing_keys", "unexpected_keys", "skipped_keys", "expanded_keys")
    }
    if any(fields.values()):
        raise RuntimeError(f"formal exact stage transition checkpoint mismatch: {fields}")


def validate_stage0_native_warm_start_load(load_result) -> None:
    """Allow only the intentional V6 -> V7 additions during Stage 0 warm start."""

    missing = list(getattr(load_result, "missing_keys", []) or [])
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
    skipped = list(getattr(load_result, "skipped_keys", []) or [])
    expanded = list(getattr(load_result, "expanded_keys", []) or [])
    allowed_missing = ("multiview_fuser.", "token_codec.")
    allowed_skipped = ("context_pixel.",)
    bad_missing = [key for key in missing if not key.startswith(allowed_missing)]
    bad_skipped = [key for key in skipped if not key.startswith(allowed_skipped)]
    if bad_missing or bad_skipped or unexpected or expanded:
        raise RuntimeError(
            "Stage 0 checkpoint is not a strict V6-native-compatible warm start: "
            f"bad_missing={bad_missing[:20]} bad_skipped={bad_skipped[:20]} "
            f"unexpected={unexpected[:20]} expanded={expanded[:20]}"
        )


def action_policy_kwargs_from_targets(tgt: dict) -> dict:
    kwargs = {}
    for src, dst in (
        ("lowdim_state", "lowdim_state"),
        ("object_state", "object_state"),
        ("plan_state", "plan_state"),
        ("action_history", "action_history"),
    ):
        if src in tgt:
            kwargs[dst] = tgt[src]
    if "progress_state" in tgt:
        kwargs["progress_state"] = tgt["progress_state"]
    elif "progress_tgt" in tgt:
        progress = tgt["progress_tgt"]
        if progress.ndim > 1:
            progress = progress[:, :1]
        kwargs["progress_state"] = progress
    return kwargs


def _policy_flow_clean_action(
    action_tgt: torch.Tensor,
    action_tgt_norm: torch.Tensor,
    train_cfg: dict,
) -> torch.Tensor:
    horizon = min(int(action_tgt.shape[1]), int(action_tgt_norm.shape[1]))
    pose = action_tgt_norm[:, :horizon].float()
    flow_action_dim = int(train_cfg.get("policy_flow_action_dim", 7) or 7)
    if flow_action_dim == 6:
        return pose
    if flow_action_dim != 7:
        raise ValueError(f"policy_flow_action_dim must be 6 or 7, got {flow_action_dim}")
    grip_close01 = action_tgt[:, :horizon, 6].float()
    if not torch.isfinite(grip_close01).all():
        raise ValueError("policy flow grip target contains non-finite values")
    if bool((grip_close01 < 0.0).any()) or bool((grip_close01 > 1.0).any()):
        raise ValueError(
            "policy flow requires an explicit close01 grip target; signed or "
            "continuous raw grip values must be canonicalized first"
        )
    grip_binary = (grip_close01 > 0.5).float()
    grip_logit_scale = float(train_cfg.get("policy_flow_grip_logit_scale", 2.5) or 2.5)
    grip_logit = torch.where(
        grip_binary > 0.5,
        grip_binary.new_full(grip_binary.shape, grip_logit_scale),
        grip_binary.new_full(grip_binary.shape, -grip_logit_scale),
    )
    return torch.cat([pose, grip_logit[..., None]], dim=-1)


def make_policy_flow_training_kwargs(tgt: dict, train_cfg: dict) -> dict:
    if float(train_cfg.get("policy_flow_weight", 0.0) or 0.0) <= 0.0:
        return {}
    x1 = _policy_flow_clean_action(tgt["action_tgt"], tgt["action_tgt_norm"], train_cfg)
    noise_scale = float(train_cfg.get("policy_flow_noise_scale", 1.0) or 1.0)
    x0 = torch.randn_like(x1) * noise_scale
    t_eps = float(train_cfg.get("policy_flow_t_eps", 1e-4) or 1e-4)
    t = torch.rand(x1.shape[0], 1, 1, device=x1.device, dtype=x1.dtype)
    t = t.clamp(min=t_eps, max=1.0 - t_eps)
    xt = (1.0 - t) * x0 + t * x1
    return {"flow_action": xt, "flow_t": t}


def _forward_joint_model(
    model: torch.nn.Module,
    s: torch.Tensor,
    c: torch.Tensor,
    *,
    action_cond: torch.Tensor | None,
    context_rgb: torch.Tensor | None,
    prior_clean_tokens: torch.Tensor | None = None,
    pixel: bool = False,
    bridging: bool = False,
    policy_kwargs: dict | None = None,
    multiview_kwargs: dict | None = None,
    candidate_actions: torch.Tensor | None = None,
    candidate_include_geometry: bool = False,
    native_action_no_teacher: bool = False,
) -> dict:
    kwargs = {
        "action_cond": action_cond,
        "context_rgb": context_rgb,
        "prior_clean_tokens": prior_clean_tokens,
        "pixel": pixel,
        "bridging": bridging,
        "candidate_actions": candidate_actions,
        "candidate_include_geometry": candidate_include_geometry,
    }
    if native_action_no_teacher:
        kwargs["native_action_no_teacher"] = True
    if policy_kwargs:
        kwargs.update(policy_kwargs)
    if multiview_kwargs:
        kwargs.update(multiview_kwargs)
    return model(s, c, **kwargs)


def compute_native_no_teacher_action_loss(
    out: dict[str, torch.Tensor],
    tgt: dict,
    native_action_weights: LossWeights,
    *,
    train_cfg: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Supervise the internal no-teacher pass emitted by the main forward."""

    if str(native_action_weights.action_pose_space).strip().lower() != "normalized":
        raise ValueError(
            "multi-source native no-teacher supervision must use normalized pose space"
        )
    required = {
        "native_action_no_teacher_pose_norm",
        "native_action_no_teacher_gripper_logit",
    }
    missing = sorted(required - out.keys())
    if missing:
        raise RuntimeError(
            "main forward did not emit the native no-teacher path: "
            + ", ".join(missing)
        )
    pose_norm = out["native_action_no_teacher_pose_norm"].float()
    action_tgt_norm = tgt["action_tgt_norm"].to(
        device=pose_norm.device, dtype=pose_norm.dtype
    )
    action_tgt_physical = tgt["action_tgt"][..., :6].to(
        device=pose_norm.device, dtype=pose_norm.dtype
    )
    grip_close01 = tgt.get("action_grip_close01")
    if grip_close01 is None:
        raise ValueError("native no-teacher action loss requires action_grip_close01")
    grip_close01 = grip_close01.to(
        device=pose_norm.device, dtype=pose_norm.dtype
    )
    horizon = min(
        int(pose_norm.shape[1]),
        int(action_tgt_norm.shape[1]),
        int(grip_close01.shape[1]),
    )
    if horizon <= 0:
        raise ValueError("native no-teacher action loss has an empty horizon")
    pose_mean = tgt.get("action_pose_mean")
    pose_std = tgt.get("action_pose_std")
    if pose_mean is None or pose_std is None:
        raise RuntimeError(
            "native multi-source action training requires per-sample "
            "action_pose_mean/action_pose_std"
        )
    train_cfg = train_cfg or {}
    result = native_action_loss(
        pose_norm[:, :horizon],
        out["native_action_no_teacher_gripper_logit"][:, :horizon].float(),
        action_tgt_norm[:, :horizon],
        action_tgt_physical[:, :horizon],
        grip_close01[:, :horizon],
        pose_mean,
        pose_std,
        previous_grip_close01=tgt.get("action_prev_grip"),
        cfg=NativeActionLossConfig(
            huber_delta=float(native_action_weights.huber_delta),
            grip_weight=float(native_action_weights.grip),
            grip_positive_weight=float(
                native_action_weights.action_grip_positive_weight
            ),
            first_step_weight=float(
                train_cfg.get("native_action_first_step_weight", 0.5)
            ),
            trajectory_weight=float(
                train_cfg.get("native_action_trajectory_weight", 0.2)
            ),
            translation_direction_weight=float(
                train_cfg.get("native_action_translation_direction_weight", 0.1)
            ),
            rotation_direction_weight=float(
                train_cfg.get("native_action_rotation_direction_weight", 0.15)
            ),
            translation_magnitude_weight=float(
                train_cfg.get("native_action_translation_magnitude_weight", 0.2)
            ),
            rotation_magnitude_weight=float(
                train_cfg.get("native_action_rotation_magnitude_weight", 0.25)
            ),
            grip_event_weight=float(
                train_cfg.get("native_action_grip_event_weight", 1.0)
            ),
            first_grip_weight=float(
                train_cfg.get("native_action_first_grip_weight", 0.5)
            ),
            translation_active_threshold_m=float(
                train_cfg.get("native_action_translation_active_threshold_m", 1e-4)
            ),
            rotation_active_threshold_rad=float(
                train_cfg.get("native_action_rotation_active_threshold_rad", 1e-3)
            ),
        ),
    )
    return {
        "L_action": result.pop("loss"),
        **result,
    }


def compute_native_no_teacher_future_loss(
    out: dict[str, torch.Tensor],
    tgt: dict,
    train_cfg: dict,
) -> dict[str, torch.Tensor]:
    """Anchor the action-free policy context to the demonstrated future.

    This makes ``core_pred`` an actual task-conditioned future representation,
    rather than allowing it to become an unconstrained policy-only latent.
    """

    key = "native_action_no_teacher_pred_tokens"
    if key not in out:
        raise RuntimeError("no-teacher future anchor requires the no-action dual output")
    if "s_tgt" not in tgt:
        raise RuntimeError("no-teacher future anchor requires decoded s_tgt tokens")
    predicted = out[key].float()
    target = tgt["s_tgt"].to(device=predicted.device, dtype=predicted.dtype)
    horizon = min(int(predicted.shape[1]), int(target.shape[1]))
    if horizon <= 0 or predicted.shape[2:] != target.shape[2:]:
        raise ValueError(
            "no-teacher future/target shape mismatch: "
            f"predicted={tuple(predicted.shape)} target={tuple(target.shape)}"
        )
    predicted = predicted[:, :horizon]
    target = target[:, :horizon]
    mse = F.mse_loss(predicted, target)
    cosine = 1.0 - F.cosine_similarity(
        predicted,
        target,
        dim=-1,
        eps=1e-6,
    ).mean()
    cosine_weight = float(
        train_cfg.get("native_future_no_teacher_cosine_weight", 0.1) or 0.0
    )
    if not math.isfinite(cosine_weight) or cosine_weight < 0.0:
        raise ValueError(
            "native_future_no_teacher_cosine_weight must be finite and non-negative"
        )
    return {
        "loss": mse + cosine_weight * cosine,
        "mse": mse,
        "cosine": cosine,
    }


def resolve_action_training_weights(
    base_weights: LossWeights,
    train_cfg: dict,
    *,
    strict: bool,
) -> tuple[LossWeights, LossWeights, LossWeights]:
    """Return factual, representation and no-teacher weight contracts."""

    key = "factual_main_action_loss_weight"
    if strict and key not in train_cfg and float(base_weights.action) != 0.0:
        raise ValueError(
            f"train.{key} or loss.action must be explicitly set to 0 for "
            "action-conditioned S0"
        )
    factual_action_weight = float(train_cfg.get(key, 0.0) or 0.0)
    if not math.isfinite(factual_action_weight) or factual_action_weight != 0.0:
        raise ValueError(
            "teacher-leaky factual main action loss is forbidden in S0; "
            f"train.{key} must equal 0, got {factual_action_weight}"
        )
    factual = replace(base_weights, action=0.0)
    representation = replace(factual, idm_reg=0.0)
    native = replace(
        base_weights,
        action_pose_space=str(
            train_cfg.get("native_action_no_teacher_pose_space", "normalized")
        ),
        action_grip_positive_weight=float(
            train_cfg.get(
                "native_action_no_teacher_grip_positive_weight",
                base_weights.action_grip_positive_weight,
            )
        ),
    )
    if str(native.action_pose_space).strip().lower() != "normalized":
        raise ValueError(
            "native_action_no_teacher_pose_space must be normalized for per-source stats"
        )
    return factual, representation, native


def _direct_policy_only_forward(
    target_model: torch.nn.Module,
    s: torch.Tensor,
    c: torch.Tensor,
    *,
    action_cond: torch.Tensor | None,
    context_rgb: torch.Tensor | None,
    policy_kwargs: dict,
    train_cfg: dict,
    multiview_kwargs: dict | None = None,
) -> dict:
    if target_model.action_policy is None:
        raise RuntimeError("direct_policy_only requires enable_action_policy")
    kwargs = dict(policy_kwargs)
    kwargs["context_rgb"] = context_rgb
    if multiview_kwargs:
        # ``multiview_kwargs_from_targets`` intentionally uses ``wrist_s``
        # because it is normally forwarded through ``JointWorldModel.forward``.
        # The policy-only fast path calls ``fuse_views`` directly, whose public
        # parameter is named ``wrist_tokens``. Keep the shared batch contract
        # intact and adapt only at this direct-call boundary.
        fuse_kwargs = dict(multiview_kwargs)
        if "wrist_s" in fuse_kwargs:
            if "wrist_tokens" in fuse_kwargs:
                raise RuntimeError(
                    "direct policy multiview inputs supplied both wrist_s and "
                    "wrist_tokens"
                )
            fuse_kwargs["wrist_tokens"] = fuse_kwargs.pop("wrist_s")
        s = target_model.fuse_views(s, **fuse_kwargs)
    source = str(train_cfg.get("direct_policy_context_source", "input")).strip().lower()
    if source in {"", "input", "s", "tokens", "cached"}:
        policy_tokens = s
    elif source in {
        "core",
        "core_detach",
        "core_pred",
        "core_pred_detach",
        "pred",
        "pred_detach",
        "pred_tokens",
        "pred_tokens_detach",
        "serving",
        "serving_detach",
    }:
        dual = getattr(target_model, "dual", None)
        if dual is None:
            raise RuntimeError("direct_policy_context_source=core_pred requires target_model.dual")
        action_mode = str(train_cfg.get("direct_policy_core_action_cond", "none")).strip().lower()
        core_action_cond = action_cond if action_mode in {"teacher", "gt", "action", "action_cond"} else None
        if core_action_cond is not None:
            core_horizon = None
            try:
                core_horizon = int(target_model.cfg.dual.state.k)
            except Exception:
                state_cfg = getattr(getattr(dual, "state", None), "cfg", None)
                if state_cfg is not None and hasattr(state_cfg, "k"):
                    core_horizon = int(getattr(state_cfg, "k"))
            if core_horizon is not None:
                if int(core_action_cond.shape[1]) > core_horizon:
                    core_action_cond = core_action_cond[:, :core_horizon]
                elif int(core_action_cond.shape[1]) < core_horizon:
                    pad = core_action_cond[:, -1:].expand(
                        -1,
                        core_horizon - int(core_action_cond.shape[1]),
                        -1,
                    )
                    core_action_cond = torch.cat([core_action_cond, pad], dim=1)
        core_eval = bool(train_cfg.get("direct_policy_core_eval", True))
        was_training = bool(dual.training)
        if core_eval:
            dual.eval()
        try:
            with torch.no_grad():
                dual_out = dual(s, c, action_cond=core_action_cond)
                policy_tokens = dual_out["pred_tokens"].detach()
        finally:
            if core_eval and was_training:
                dual.train()
    else:
        raise RuntimeError(f"unsupported direct_policy_context_source={source!r}")
    return target_model.action_policy(policy_tokens, task_emb=c, **kwargs)


def _main_teacher_action_weight_for_batch(
    *,
    direct_policy_only: bool,
    representation_only_batch: bool,
    factual_weights: LossWeights,
    representation_weights: LossWeights,
) -> float:
    """Report the WM teacher-action loss weight without conflating policy S1.

    Direct-policy training has its own flow/regression objectives and does not
    execute ``compute_losses`` for the teacher-conditioned WM branch.
    """

    if direct_policy_only:
        return 0.0
    selected = (
        representation_weights if representation_only_batch else factual_weights
    )
    return float(selected.action)


def _mask_to_shape(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return mask.reshape((mask.shape[0],) + (1,) * (x.ndim - 1))


def apply_condition_dropout(
    s: torch.Tensor,
    c: torch.Tensor,
    action_cond: torch.Tensor | None,
    context_rgb: torch.Tensor | None,
    train_cfg: dict,
    *,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, dict[str, torch.Tensor]]:
    cfg = train_cfg.get("condition_dropout") or {}
    if not isinstance(cfg, dict):
        raise ValueError("train.condition_dropout must be a mapping when provided")
    enabled = bool(cfg.get("enabled", False))
    action_p = float(cfg.get("action_p", 0.0))
    context_p = float(cfg.get("context_p", cfg.get("rgb_context_p", 0.0)))
    text_only_p = float(cfg.get("text_only_p", 0.0))
    text_context_p = float(cfg.get("text_context_p", 0.0))
    text_action_p = float(cfg.get("text_action_p", 0.0))
    if not training or not enabled or (
        action_p <= 0.0 and context_p <= 0.0
        and text_only_p <= 0.0 and text_context_p <= 0.0 and text_action_p <= 0.0
    ):
        zero = s.new_zeros(())
        return s, c, action_cond, context_rgb, {
            "drop_action_frac": zero,
            "drop_context_frac": zero,
            "drop_text_only_frac": zero,
            "drop_text_context_frac": zero,
            "drop_text_action_frac": zero,
        }

    bsz = s.shape[0]
    explicit_total = max(0.0, text_only_p) + max(0.0, text_context_p) + max(0.0, text_action_p)
    if explicit_total > 1.0 + 1e-6:
        raise ValueError(
            "condition_dropout text_only_p + text_context_p + text_action_p must be <= 1.0"
        )
    u = torch.rand(bsz, device=s.device)
    text_only_mask = u < max(0.0, text_only_p)
    text_context_mask = (u >= max(0.0, text_only_p)) & (
        u < max(0.0, text_only_p) + max(0.0, text_context_p)
    )
    text_action_mask = (u >= max(0.0, text_only_p) + max(0.0, text_context_p)) & (
        u < explicit_total
    )
    residual_mask = u >= explicit_total

    action_mask = text_only_mask | text_context_mask
    context_mask = text_only_mask | text_action_mask
    if residual_mask.any():
        residual_action = torch.rand(bsz, device=s.device) < max(0.0, min(1.0, action_p))
        residual_context = torch.rand(bsz, device=s.device) < max(0.0, min(1.0, context_p))
        action_mask = action_mask | (residual_mask & residual_action)
        context_mask = context_mask | (residual_mask & residual_context)

    s_out = torch.where(_mask_to_shape(context_mask, s), torch.zeros_like(s), s)
    rgb_out = None
    if context_rgb is not None:
        rgb_out = torch.where(_mask_to_shape(context_mask, context_rgb), torch.zeros_like(context_rgb), context_rgb)
    action_out = action_cond
    if action_cond is not None:
        action_out = torch.where(_mask_to_shape(action_mask, action_cond), torch.zeros_like(action_cond), action_cond)
    return s_out, c, action_out, rgb_out, {
        "drop_action_frac": action_mask.float().mean(),
        "drop_context_frac": context_mask.float().mean(),
        "drop_text_only_frac": text_only_mask.float().mean(),
        "drop_text_context_frac": text_context_mask.float().mean(),
        "drop_text_action_frac": text_action_mask.float().mean(),
    }


def prior_clean_tokens_from_targets(tgt: dict) -> torch.Tensor | None:
    if "s_init_tgt" not in tgt or "s_tgt" not in tgt:
        return None
    return torch.cat([tgt["s_init_tgt"][:, None], tgt["s_tgt"]], dim=1)


def _expand_action_policy_horizon_tensor(key: str, value: torch.Tensor, target_shape: torch.Size) -> torch.Tensor | None:
    if key not in {
        "action_policy.horizon_embed",
        "action_policy.prior_horizon_embed",
        "action_policy.local_residual_head.7.weight",
        "action_policy.local_residual_head.7.bias",
    }:
        return None
    if value.ndim == 3 and len(target_shape) == 3:
        if value.shape[0] != target_shape[0] or value.shape[2] != target_shape[2]:
            return None
        expanded = F.interpolate(
            value.permute(0, 2, 1).float(),
            size=int(target_shape[1]),
            mode="linear",
            align_corners=True,
        ).permute(0, 2, 1)
        return expanded.to(dtype=value.dtype, device=value.device)
    action_dim = 7
    if value.ndim == 2 and len(target_shape) == 2:
        if value.shape[1] != target_shape[1]:
            return None
        old_h = int(value.shape[0]) // action_dim
        new_h = int(target_shape[0]) // action_dim
        if old_h <= 0 or new_h <= 0 or old_h * action_dim != value.shape[0] or new_h * action_dim != target_shape[0]:
            return None
        hidden = int(value.shape[1])
        x = value.reshape(old_h, action_dim, hidden).permute(1, 2, 0).reshape(1, action_dim * hidden, old_h)
        expanded = F.interpolate(x.float(), size=new_h, mode="linear", align_corners=True)
        expanded = expanded.reshape(action_dim, hidden, new_h).permute(2, 0, 1).reshape(new_h * action_dim, hidden)
        return expanded.to(dtype=value.dtype, device=value.device)
    if value.ndim == 1 and len(target_shape) == 1:
        old_h = int(value.shape[0]) // action_dim
        new_h = int(target_shape[0]) // action_dim
        if old_h <= 0 or new_h <= 0 or old_h * action_dim != value.shape[0] or new_h * action_dim != target_shape[0]:
            return None
        x = value.reshape(old_h, action_dim).permute(1, 0).unsqueeze(0)
        expanded = F.interpolate(x.float(), size=new_h, mode="linear", align_corners=True)
        expanded = expanded.squeeze(0).permute(1, 0).reshape(new_h * action_dim)
        return expanded.to(dtype=value.dtype, device=value.device)
    return None


def load_compatible_state_dict(model: torch.nn.Module, state: dict, strict: bool):
    if strict:
        return model.load_state_dict(state, strict=True)
    current = model.state_dict()
    compatible = {}
    skipped = []
    expanded = []
    for key, value in state.items():
        if key in current and current[key].shape == value.shape:
            compatible[key] = value
        elif key in current:
            expanded_value = _expand_action_policy_horizon_tensor(key, value, current[key].shape)
            if expanded_value is not None and expanded_value.shape == current[key].shape:
                compatible[key] = expanded_value
                expanded.append(key)
            else:
                skipped.append(key)
        else:
            skipped.append(key)
    result = model.load_state_dict(compatible, strict=False)
    return SimpleNamespace(
        missing_keys=result.missing_keys,
        unexpected_keys=result.unexpected_keys,
        skipped_keys=skipped,
        expanded_keys=expanded,
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


def apply_trainable_filter(model: torch.nn.Module, train_cfg: dict, rank: int) -> None:
    """Optionally freeze parameters by name prefix before DDP/optimizer setup."""
    trainable_prefixes = tuple(train_cfg.get("trainable_prefixes") or ())
    freeze_prefixes = tuple(train_cfg.get("freeze_prefixes") or ())
    if not trainable_prefixes and not freeze_prefixes:
        return

    frozen = 0
    trainable = 0
    trainable_tensors = 0
    for name, param in model.named_parameters():
        enabled = True
        if trainable_prefixes:
            enabled = name.startswith(trainable_prefixes)
        if freeze_prefixes and name.startswith(freeze_prefixes):
            enabled = False
        param.requires_grad = enabled
        if enabled:
            trainable += param.numel()
            trainable_tensors += 1
        else:
            frozen += param.numel()
    if trainable_tensors == 0:
        raise RuntimeError(
            "trainable filter left no trainable parameters; "
            f"trainable_prefixes={trainable_prefixes} freeze_prefixes={freeze_prefixes}"
        )
    if rank == 0:
        print(
            f"[rank0] trainable filter: trainable={trainable/1e6:.2f}M "
            f"frozen={frozen/1e6:.2f}M trainable_prefixes={list(trainable_prefixes)} "
            f"freeze_prefixes={list(freeze_prefixes)}"
        )


def _normalise_param_name(name: str) -> str:
    while name.startswith("module."):
        name = name[len("module."):]
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


def build_optimizer_param_groups(
    model: torch.nn.Module,
    hunyuan_adapter: torch.nn.Module | None,
    train_cfg: dict,
    rank: int,
) -> list[dict]:
    """Build AdamW groups with optional prefix-based LR multipliers.

    Config format:

    train:
      lr_multipliers:
        context_pixel: 0.05
        geom: 0.05
        hunyuan_adapter: 1.0
    """
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])
    raw_multipliers = train_cfg.get("lr_multipliers") or {}
    if not isinstance(raw_multipliers, dict):
        raise ValueError("train.lr_multipliers must be a mapping of prefix -> multiplier")
    lr_multipliers = {str(k): float(v) for k, v in raw_multipliers.items()}

    grouped: dict[float, dict] = {}
    group_counts: dict[float, int] = {}

    def add_param(name: str, param: torch.nn.Parameter) -> None:
        if not param.requires_grad:
            return
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

    for name, param in model.named_parameters():
        add_param(name, param)
    if hunyuan_adapter is not None:
        for name, param in hunyuan_adapter.named_parameters():
            add_param(f"hunyuan_adapter.{name}", param)

    groups = [grouped[mult] for mult in sorted(grouped)]
    if not groups:
        raise RuntimeError("optimizer has no trainable parameters after filters")
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
    return groups


def _all_reduce_gradients(model: torch.nn.Module, world: int) -> None:
    backend = dist.get_backend()
    bucket_mb = float(os.environ.get("WM3D_GRAD_BUCKET_MB", "256"))
    bucket_limit = max(1, int(bucket_mb * 1024 * 1024))

    # Manual sync is used by direct_policy_only because that path bypasses the DDP
    # wrapper and calls action_policy directly. Every rank must execute the same
    # collectives in the same order, even when a conditional branch leaves a
    # parameter unused on one rank. Sync zero gradients for unused trainable
    # parameters so ranks cannot diverge or hang in NCCL.
    def source_grad(param: torch.nn.Parameter) -> tuple[torch.Tensor, bool]:
        if param.grad is not None:
            return param.grad.detach(), True
        return torch.zeros_like(param, memory_format=torch.preserve_format), False

    def assign_synced(param: torch.nn.Parameter, synced: torch.Tensor, had_grad: bool) -> None:
        synced = synced.to(device=param.device, dtype=param.dtype)
        if had_grad and param.grad is not None:
            param.grad.copy_(synced)
        else:
            param.grad = synced.clone(memory_format=torch.preserve_format)

    def reduced_grad_presence(bucket: list[tuple[torch.nn.Parameter, torch.Tensor, bool]], device: torch.device) -> torch.Tensor:
        flags = torch.tensor([1.0 if had_grad else 0.0 for _param, _grad, had_grad in bucket], device=device)
        dist.all_reduce(flags, op=dist.ReduceOp.SUM)
        return flags

    def flush_cuda_bucket(bucket: list[tuple[torch.nn.Parameter, torch.Tensor, bool]]) -> None:
        if not bucket:
            return
        tensors = [grad for _param, grad, _had_grad in bucket]
        flat = torch._utils._flatten_dense_tensors(tensors)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world)
        global_grad_count = reduced_grad_presence(bucket, flat.device)
        for idx, ((param, _grad, had_grad), synced) in enumerate(
            zip(bucket, torch._utils._unflatten_dense_tensors(flat, tensors))
        ):
            if float(global_grad_count[idx].item()) <= 0.0:
                param.grad = None
            else:
                assign_synced(param, synced, had_grad)

    def flush_gloo_bucket(bucket: list[tuple[torch.nn.Parameter, torch.Tensor, bool]]) -> None:
        if not bucket:
            return
        tensors = [cpu for _param, cpu, _had_grad in bucket]
        flat = torch._utils._flatten_dense_tensors(tensors)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world)
        global_grad_count = reduced_grad_presence(bucket, flat.device)
        for idx, ((param, _cpu, had_grad), synced) in enumerate(
            zip(bucket, torch._utils._unflatten_dense_tensors(flat, tensors))
        ):
            if float(global_grad_count[idx].item()) <= 0.0:
                param.grad = None
            else:
                assign_synced(param, synced, had_grad)

    cuda_buckets: dict[tuple[torch.device, torch.dtype], list[tuple[torch.nn.Parameter, torch.Tensor, bool]]] = {}
    cuda_bytes: dict[tuple[torch.device, torch.dtype], int] = {}
    gloo_bucket: list[tuple[torch.nn.Parameter, torch.Tensor, bool]] = []
    gloo_bytes = 0

    for param in model.parameters():
        if not param.requires_grad:
            continue
        grad, had_grad = source_grad(param)
        if backend == "gloo" and grad.is_cuda:
            cpu_grad = grad.float().cpu()
            nbytes = cpu_grad.numel() * cpu_grad.element_size()
            if gloo_bucket and gloo_bytes + nbytes > bucket_limit:
                flush_gloo_bucket(gloo_bucket)
                gloo_bucket = []
                gloo_bytes = 0
            gloo_bucket.append((param, cpu_grad, had_grad))
            gloo_bytes += nbytes
            continue
        key = (grad.device, grad.dtype)
        bucket = cuda_buckets.setdefault(key, [])
        cur_bytes = cuda_bytes.get(key, 0)
        nbytes = grad.numel() * grad.element_size()
        if bucket and cur_bytes + nbytes > bucket_limit:
            flush_cuda_bucket(bucket)
            cuda_buckets[key] = []
            cuda_bytes[key] = 0
            bucket = cuda_buckets[key]
            cur_bytes = 0
        bucket.append((param, grad, had_grad))
        cuda_bytes[key] = cur_bytes + nbytes

    for bucket in cuda_buckets.values():
        flush_cuda_bucket(bucket)
    flush_gloo_bucket(gloo_bucket)


def _distributed_finite_count(value: torch.Tensor | bool, device: torch.device, world: int) -> int:
    if isinstance(value, torch.Tensor):
        finite = bool(value.detach().item())
    else:
        finite = bool(value)
    if world <= 1:
        return int(finite)
    # NCCL on this cluster has failed on tiny int32 reductions before; float32 is robust.
    reduce_device = torch.device("cpu") if dist.get_backend() == "gloo" else device
    flag = torch.tensor([1.0 if finite else 0.0], device=reduce_device, dtype=torch.float32)
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    return int(flag.item())


def apply_source_gradient_policy(
    model: torch.nn.Module,
    *,
    representation_only: bool,
    allowed_prefixes: tuple[str, ...] = ("pixel.", "context_pixel."),
) -> int:
    """Apply the post-backward representation gate and return kept gradients.

    Factual sources are an intentional no-op: their native transition/core
    gradients must survive through the optimizer step.
    """

    target_model = model.module if isinstance(model, DDP) else model
    present = sum(
        1 for _name, parameter in target_model.named_parameters()
        if parameter.grad is not None
    )
    if not representation_only:
        return present
    kept = 0
    for name, parameter in target_model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith(allowed_prefixes):
            kept += 1
        else:
            parameter.grad = None
    if kept == 0:
        raise RuntimeError(
            "representation-only batch produced no permitted gradients; "
            f"allowed_prefixes={allowed_prefixes}"
        )
    return kept


def factual_action_gradient_counts(model: torch.nn.Module) -> dict[str, int]:
    """Count finite nonzero gradients for the three P0 parameter owners."""

    target_model = model.module if isinstance(model, DDP) else model
    groups = {
        "state_action_condition_projection": (
            "dual.state.action_cond_proj.",
            "dual.state.action_cond_pos",
        ),
        "native_state_dynamics": (
            "dual.state.layers.",
            "dual.state.decoder.",
            "dual.state.out_proj.",
        ),
        "no_teacher_action_head": ("action_proj.",),
    }
    counts = {name: 0 for name in groups}
    for name, parameter in target_model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        if not (torch.isfinite(grad).all() and bool((grad != 0).any())):
            continue
        for group_name, prefixes in groups.items():
            if name.startswith(prefixes):
                counts[group_name] += 1
                break
    return counts


def register_nonzero_gradient_evidence(
    tensor: torch.Tensor,
) -> dict[str, bool]:
    """Capture finite, nonzero intermediate gradients during backward.

    Reading ``tensor.grad`` after a DDP gradient-accumulation group is not
    reliable for non-leaf views.  A tensor hook observes the gradient at the
    exact autograd edge and therefore works for both accumulated micro-batches
    and ordinary single-batch smoke tests.
    """

    evidence = {"finite_nonzero": False}

    def _capture(grad: torch.Tensor) -> torch.Tensor:
        if bool(torch.isfinite(grad).all()) and bool((grad != 0).any()):
            evidence["finite_nonzero"] = True
        return grad

    tensor.register_hook(_capture)
    return evidence


def branch_objective_gradient_counts(
    model: torch.nn.Module,
    out: dict[str, Any],
) -> dict[str, int]:
    """Runtime evidence that imagine/select objectives reached their owners."""

    target_model = model.module if isinstance(model, DDP) else model
    candidate_future = out.get("candidate_pred_tokens")
    candidate_future_grad = 0
    if isinstance(candidate_future, torch.Tensor) and candidate_future.grad is not None:
        grad = candidate_future.grad.detach()
        candidate_future_grad = int(
            torch.isfinite(grad).all() and bool((grad != 0).any())
        )
    future_value_head_grad_params = 0
    for name, parameter in target_model.named_parameters():
        if not name.startswith("future_value_head.") or parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        if torch.isfinite(grad).all() and bool((grad != 0).any()):
            future_value_head_grad_params += 1
    return {
        "candidate_future_grad_tensors": candidate_future_grad,
        "future_value_head_grad_params": future_value_head_grad_params,
    }


def count_factual_action_core_gradients(model: torch.nn.Module) -> int:
    """Backward-compatible scalar for the action-conditioning projection."""

    return factual_action_gradient_counts(model)[
        "state_action_condition_projection"
    ]


def _score_evaluator(out: dict, train_cfg: dict) -> torch.Tensor:
    progress_w = float(train_cfg.get("evaluator_score_progress_weight", 1.0))
    terminal_w = float(train_cfg.get("evaluator_score_terminal_weight", 1.0))
    plaus_w = float(train_cfg.get("evaluator_score_plausibility_weight", 0.0))
    terms: list[torch.Tensor] = []
    weights: list[float] = []
    if progress_w and "progress" in out:
        terms.append(torch.sigmoid(out["progress"].float()).mean(dim=1) * progress_w)
        weights.append(abs(progress_w))
    if terminal_w and "terminal_success_logit" in out:
        terms.append(torch.sigmoid(out["terminal_success_logit"].float()) * terminal_w)
        weights.append(abs(terminal_w))
    if plaus_w and "plausibility_logit" in out:
        terms.append(torch.sigmoid(out["plausibility_logit"].float()) * plaus_w)
        weights.append(abs(plaus_w))
    if not terms:
        return out["pred_tokens"].new_zeros(out["pred_tokens"].shape[0], dtype=torch.float32)
    return torch.stack(terms, dim=0).sum(dim=0) / max(1e-6, sum(weights))


def _counterfactual_action_cond(
    action_cond: torch.Tensor,
    variant: str,
    step: int,
    scaled_factor: float = 2.0,
    *,
    grip_contract: str = "close01",
) -> torch.Tensor:
    cur = action_cond.clone()
    if variant == "zero":
        cur.zero_()
    elif variant == "shuffled":
        if cur.shape[0] > 1:
            cur = torch.roll(cur, shifts=1 + (step % max(1, cur.shape[0] - 1)), dims=0)
        # A batch-one temporal roll is not a counterfactual: it changes the
        # alignment and can still expose the same action trajectory.  The
        # caller must mark this negative invalid instead.
    elif variant == "sign_flip":
        cur[..., :6].neg_()
    elif variant == "scaled":
        cur[..., :6].mul_(scaled_factor)
    elif variant == "grip_toggle":
        if normalize_action_grip_contract(grip_contract) == "signed_close":
            cur[..., 6:7] = -cur[..., 6:7]
        else:
            cur[..., 6:7] = 1.0 - cur[..., 6:7]
    else:
        raise ValueError(f"unknown evaluator counterfactual variant: {variant}")
    return cur


def deranged_action_negative(
    action_cond: torch.Tensor,
    *,
    step: int,
    min_distance: float = 0.0,
    stats_keys: list[str] | tuple[str, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the best cyclic non-self local-batch action derangement.

    Returns ``(negative, permutation, valid_mask, rms_distance)``.  The
    derangement is always constructed before any memory-saving sub-sampling.
    A one-sample local batch is explicitly invalid and is never converted into
    a temporal roll.
    """

    if action_cond.ndim < 2:
        raise ValueError("action_cond must have a batch dimension")
    batch_size = int(action_cond.shape[0])
    identity = torch.arange(batch_size, device=action_cond.device)
    if batch_size < 2:
        distance = action_cond.new_zeros((batch_size,), dtype=torch.float32)
        valid = torch.zeros(batch_size, device=action_cond.device, dtype=torch.bool)
        return action_cond.clone(), identity, valid, distance
    if not math.isfinite(float(min_distance)) or float(min_distance) < 0.0:
        raise ValueError(f"counterfactual min_distance must be finite and non-negative: {min_distance}")
    if stats_keys is not None and len(stats_keys) != batch_size:
        raise ValueError("stats_keys length must equal the local batch size")

    flat = action_cond.float().flatten(1)
    best = None
    # Rotate the shift preference by step while evaluating every non-self
    # cyclic derangement.  This is deterministic and avoids a factorial search.
    shifts = [1 + ((step + offset) % (batch_size - 1)) for offset in range(batch_size - 1)]
    for shift in shifts:
        permutation = torch.roll(identity, shifts=shift, dims=0)
        negative = action_cond.index_select(0, permutation)
        distance = (flat - negative.float().flatten(1)).pow(2).mean(dim=1).sqrt()
        valid = distance >= float(min_distance)
        if stats_keys is not None:
            key_match = torch.tensor(
                [stats_keys[i] == stats_keys[int(permutation[i])] for i in range(batch_size)],
                device=action_cond.device,
                dtype=torch.bool,
            )
            valid = valid & key_match
        score = (int(valid.sum().item()), float(distance[valid].mean().item()) if valid.any() else -1.0)
        if best is None or score > best[0]:
            best = (score, negative, permutation, valid, distance)
    assert best is not None
    _score, negative, permutation, valid, distance = best
    if bool((permutation == identity).any()):
        raise RuntimeError("counterfactual derangement produced a self-match")
    return negative, permutation, valid, distance


def build_action_negative(
    action_cond: torch.Tensor,
    variant: str,
    *,
    step: int,
    min_distance: float,
    grip_contract: str,
    scaled_factor: float = 2.0,
    stats_keys: list[str] | tuple[str, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if variant == "shuffled":
        negative, _permutation, valid, distance = deranged_action_negative(
            action_cond,
            step=step,
            min_distance=min_distance,
            stats_keys=stats_keys,
        )
        return negative, valid, distance
    negative = _counterfactual_action_cond(
        action_cond,
        variant,
        step,
        scaled_factor=scaled_factor,
        grip_contract=grip_contract,
    )
    distance = (
        action_cond.float().flatten(1) - negative.float().flatten(1)
    ).pow(2).mean(dim=1).sqrt()
    return negative, distance >= float(min_distance), distance


def compute_evaluator_pairwise_loss(
    model: torch.nn.Module,
    s: torch.Tensor,
    c: torch.Tensor,
    real_out: dict,
    action_cond: torch.Tensor,
    train_cfg: dict,
    *,
    step: int,
    policy_kwargs: dict | None = None,
) -> dict[str, torch.Tensor]:
    weight = float(train_cfg.get("evaluator_pairwise_weight", 0.0))
    if weight <= 0:
        zero = real_out["pred_tokens"].new_zeros(())
        return {
            "L_evaluator_pairwise": zero,
            "evaluator_pairwise_acc": zero,
            "evaluator_pairwise_gap": zero,
        }
    variants = train_cfg.get("evaluator_pairwise_variants") or []
    if isinstance(variants, str):
        variants = [v.strip() for v in variants.split(",") if v.strip()]
    if not variants:
        zero = real_out["pred_tokens"].new_zeros(())
        return {
            "L_evaluator_pairwise": zero,
            "evaluator_pairwise_acc": zero,
            "evaluator_pairwise_gap": zero,
        }
    margin = float(train_cfg.get("evaluator_pairwise_margin", 0.1))
    scaled_factor = float(train_cfg.get("evaluator_pairwise_scaled_factor", 2.0))
    real_score = _score_evaluator(real_out, train_cfg)
    losses = []
    accs = []
    gaps = []
    rollout_policy_kwargs = dict(policy_kwargs or {})
    rollout_policy_kwargs.update(
        skip_action_proposer=True,
        skip_action_policy=True,
        skip_native_prediction_heads=True,
        detach_progress_input=bool(
            train_cfg.get("candidate_planner_detach_world_for_judge", False)
        ),
    )
    for variant in variants:
        var_cond = _counterfactual_action_cond(action_cond, variant, step, scaled_factor=scaled_factor)
        var_out = _forward_joint_model(
            model,
            s,
            c,
            action_cond=var_cond,
            context_rgb=None,
            pixel=False,
            bridging=False,
            policy_kwargs=rollout_policy_kwargs,
        )
        var_score = _score_evaluator(var_out, train_cfg)
        gap = real_score - var_score
        losses.append(torch.relu(margin - gap).mean())
        accs.append((gap > 0).float().mean())
        gaps.append(gap.mean())
    return {
        "L_evaluator_pairwise": torch.stack(losses).mean(),
        "evaluator_pairwise_acc": torch.stack(accs).mean(),
        "evaluator_pairwise_gap": torch.stack(gaps).mean(),
    }


def _context_pixel_action_zero_losses(ref: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = ref.new_zeros(())
    return {
        "L_context_pixel_action_rank": zero,
        "L_context_pixel_action_separation": zero,
        "context_pixel_action_acc": zero,
        "context_pixel_action_gap": zero,
        "context_pixel_action_rgb_gap": zero,
        "L_core_action_rank": zero,
        "L_core_action_separation": zero,
        "core_action_acc": zero,
        "core_action_gap": zero,
        "core_action_sep": zero,
        "core_action_negative_distance": zero,
        "core_action_negative_valid": zero,
        "L_native_core_action_cf_rank": zero,
        "L_native_core_action_cf_separation": zero,
        "native_core_action_cf_acc": zero,
        "native_core_action_cf_gap": zero,
        "native_core_action_cf_sep": zero,
        "native_core_action_cf_negative_distance": zero,
        "native_core_action_cf_negative_valid": zero,
        "native_core_action_cf_valid_fraction": zero,
        "native_core_action_cf_nonself_fraction": zero,
        "native_core_action_cf_min_action_distance": zero,
        "native_core_action_cf_predicted_separation": zero,
        "native_core_action_cf_wrong_action_input_grad": zero,
        "native_core_action_cf_wrong_pred_grad": zero,
    }


def action_dynamics_telemetry_metrics(
    losses: dict[str, torch.Tensor], ref: torch.Tensor
) -> dict[str, float]:
    """Serialize the fail-closed action-dynamics canary metric contract."""

    keys = {
        "L_action_raw": "L_action_raw",
        "L_action_weighted": "L_action_weighted",
        "L_native_action_no_teacher": "native_no_teacher_L_action",
        "L_native_future_no_teacher": "native_future_no_teacher_loss",
        "native_future_no_teacher_mse": "native_future_no_teacher_mse",
        "native_future_no_teacher_cosine": "native_future_no_teacher_cosine",
        "L_native_core_action_cf_rank": "L_native_core_action_cf_rank",
        "L_native_core_action_cf_separation": "L_native_core_action_cf_separation",
        "native_core_action_cf_acc": "native_core_action_cf_acc",
        "native_core_action_cf_gap": "native_core_action_cf_gap",
        "native_core_action_cf_negative_distance": "native_core_action_cf_negative_distance",
        "native_core_action_cf_negative_valid": "native_core_action_cf_negative_valid",
        "native_core_action_cf_valid_fraction": "native_core_action_cf_valid_fraction",
        "native_core_action_cf_nonself_fraction": "native_core_action_cf_nonself_fraction",
        "native_core_action_cf_min_action_distance": "native_core_action_cf_min_action_distance",
        "native_core_action_cf_predicted_separation": "native_core_action_cf_predicted_separation",
        "native_core_action_cf_wrong_action_input_grad": "native_core_action_cf_wrong_action_input_grad",
        "native_core_action_cf_wrong_pred_grad": "native_core_action_cf_wrong_pred_grad",
    }
    return {
        output_key: float(
            losses.get(loss_key, ref.new_zeros(())).detach().float()
        )
        for output_key, loss_key in keys.items()
    }


def canary_timing_fields(optimizer_step_seconds: float | None) -> dict[str, float]:
    """Return finite GPU-step and wall-clock evidence for canary throughput."""

    if (
        optimizer_step_seconds is None
        or not math.isfinite(float(optimizer_step_seconds))
        or float(optimizer_step_seconds) <= 0.0
    ):
        raise RuntimeError(
            "canary telemetry requires a finite positive optimizer-step duration; "
            "set train.measure_step_time=true"
        )
    wall_time = float(time.time())
    if not math.isfinite(wall_time):
        raise RuntimeError("canary wall clock is non-finite")
    return {
        "optimizer_step_seconds": float(optimizer_step_seconds),
        "wall_time_unix_s": wall_time,
    }


def _parse_counterfactual_variants(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v).strip() for v in value if str(v).strip()]


def _batch_index_select(value, indices: torch.Tensor, batch_size: int):
    """Select batch-aligned tensors inside nested kwargs."""

    if torch.is_tensor(value):
        if value.ndim > 0 and int(value.shape[0]) == batch_size:
            return value.index_select(0, indices.to(device=value.device))
        return value
    if isinstance(value, dict):
        return {
            key: _batch_index_select(item, indices, batch_size)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_batch_index_select(item, indices, batch_size) for item in value)
    if isinstance(value, list):
        return [_batch_index_select(item, indices, batch_size) for item in value]
    return value


def scheduled_aux_weight(
    train_cfg: dict,
    weight_key: str,
    step: int,
    *,
    schedule_prefix: str | None = None,
) -> float:
    """Resolve a sparse, linearly-ramped auxiliary objective weight.

    A weight ``foo_weight`` is scheduled by the optional keys
    ``foo_start_step``, ``foo_ramp_steps`` and ``foo_every``.  Defaults retain
    the historical behavior: active from step zero, no ramp, every step.
    """

    weight = float(train_cfg.get(weight_key, 0.0) or 0.0)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(f"{weight_key} must be finite and non-negative; got {weight}")
    prefix = schedule_prefix
    if prefix is None:
        prefix = weight_key[:-7] if weight_key.endswith("_weight") else weight_key
    start = int(train_cfg.get(f"{prefix}_start_step", 0) or 0)
    ramp = int(train_cfg.get(f"{prefix}_ramp_steps", 0) or 0)
    every = int(train_cfg.get(f"{prefix}_every", 1) or 1)
    if start < 0:
        raise ValueError(f"{prefix}_start_step must be non-negative; got {start}")
    if ramp < 0:
        raise ValueError(f"{prefix}_ramp_steps must be non-negative; got {ramp}")
    if every <= 0:
        raise ValueError(f"{prefix}_every must be positive; got {every}")
    force = bool(train_cfg.get("diagnostic_force_action_aux", False)) or bool(
        train_cfg.get(f"diagnostic_force_{prefix}", False)
    )
    if weight == 0.0:
        return 0.0
    if force:
        return weight
    if step < start or step % every != 0:
        return 0.0
    if ramp == 0:
        return weight
    progress = min(1.0, max(0.0, float(step - start) / float(ramp)))
    return weight * progress


def scheduled_native_core_cf_weight(train_cfg: dict, kind: str, step: int) -> float:
    if kind not in {"rank", "separation"}:
        raise ValueError(f"unsupported native core CF weight kind: {kind}")
    canonical_key = f"native_core_action_cf_{kind}_weight"
    legacy_key = f"core_action_{kind}_weight"
    if canonical_key in train_cfg:
        if not bool(train_cfg.get("native_core_action_cf_enabled", True)):
            return 0.0
        return scheduled_aux_weight(
            train_cfg,
            canonical_key,
            step,
            schedule_prefix="native_core_action_cf",
        )
    return scheduled_aux_weight(
        train_cfg,
        legacy_key,
        step,
        schedule_prefix="core_action_rank",
    )


def native_core_cf_value(train_cfg: dict, suffix: str, default):
    canonical_key = f"native_core_action_cf_{suffix}"
    legacy_key = f"core_action_{suffix}"
    return train_cfg.get(canonical_key, train_cfg.get(legacy_key, default))


def _config_name_set(train_cfg: dict, key: str, *, required: bool = False) -> set[str]:
    raw = train_cfg.get(key)
    if raw is None:
        if required:
            raise ValueError(f"train.{key} is required by the explicit action source policy")
        return set()
    if isinstance(raw, str):
        raw = [value.strip() for value in raw.split(",") if value.strip()]
    if not isinstance(raw, (list, tuple, set)):
        raise ValueError(f"train.{key} must be a list or comma-separated string")
    names = {str(value).strip() for value in raw if str(value).strip()}
    if required and not names:
        raise ValueError(f"train.{key} must not be empty")
    return names


def resolve_action_source_policy(
    train_cfg: dict,
    source_names: list[str] | tuple[str, ...],
    *,
    strict: bool,
) -> dict[str, str]:
    """Resolve every source to factual-action or representation-only.

    OXE admission requires two independent declarations: the source must be in
    ``factual_action_sources`` *and* in ``audited_action_sources``.  This is a
    deliberate fail-closed boundary; a source name prefix can never grant
    action access by itself.
    """

    source_set = {str(name) for name in source_names}
    factual_conditioning = train_cfg.get("factual_action_conditioning") or {}
    if not isinstance(factual_conditioning, dict):
        raise ValueError("train.factual_action_conditioning must be a mapping")
    nested_factual = factual_conditioning.get("required_sources")
    if nested_factual is not None:
        if not bool(factual_conditioning.get("enabled", False)):
            raise ValueError("factual_action_conditioning.required_sources requires enabled=true")
        if int(factual_conditioning.get("start_step", 0) or 0) != 0:
            raise ValueError("factual action conditioning must start at optimizer step 0")
        if bool(factual_conditioning.get("detach_action_condition", False)):
            raise ValueError("factual action conditioning must not detach action_cond")
        if bool(factual_conditioning.get("require_nonzero_action", False)):
            raise ValueError(
                "require_nonzero_action is invalid: an all-zero hold action can be "
                "a factual canonical action"
            )
        validity_fields = {
            "require_valid_action_contract": True,
            "require_finite_action": True,
            "allow_zero_hold_action": True,
        }
        if any(key in factual_conditioning for key in validity_fields):
            for key, expected in validity_fields.items():
                if factual_conditioning.get(key) is not expected:
                    raise ValueError(
                        f"factual_action_conditioning.{key} must be {expected}"
                    )
    explicit = (
        "factual_action_sources" in train_cfg
        or "representation_only_sources" in train_cfg
        or nested_factual is not None
    )
    if strict and not explicit:
        raise ValueError(
            "mixed-source training requires explicit train.factual_action_sources "
            "and train.representation_only_sources; legacy oxe_representation_only "
            "is forbidden"
        )
    if not explicit:
        return {name: "factual_action" for name in source_set}
    if "factual_action_sources" in train_cfg:
        factual = _config_name_set(train_cfg, "factual_action_sources", required=True)
    else:
        nested_cfg = dict(train_cfg)
        nested_cfg["factual_action_sources"] = nested_factual
        factual = _config_name_set(nested_cfg, "factual_action_sources", required=True)
    if "representation_only_sources" in train_cfg:
        representation = _config_name_set(
            train_cfg, "representation_only_sources", required=False
        )
    else:
        representation = source_set - factual
    overlap = factual & representation
    if overlap:
        raise ValueError(f"action source policy overlap: {sorted(overlap)}")
    unknown = (factual | representation) - source_set
    missing = source_set - (factual | representation)
    if unknown:
        raise ValueError(f"action source policy references unknown sources: {sorted(unknown)}")
    if missing:
        raise ValueError(f"action source policy leaves sources unassigned: {sorted(missing)}")
    audited = _config_name_set(train_cfg, "audited_action_sources")
    unaudited_oxe = {
        name for name in factual if name.startswith("oxe_") and name not in audited
    }
    if unaudited_oxe:
        raise ValueError(
            "unaudited OXE sources cannot be action-enabled: "
            f"{sorted(unaudited_oxe)}"
        )
    action_aux = _config_name_set(train_cfg, "action_aux_sources")
    invalid_aux = action_aux - factual
    if invalid_aux:
        raise ValueError(
            "action_aux_sources must be factual-action sources: "
            f"{sorted(invalid_aux)}"
        )
    return {
        name: ("factual_action" if name in factual else "representation_only")
        for name in source_set
    }


def action_aux_source_allowed(
    train_cfg: dict,
    source_name: str | None,
    *,
    representation_only: bool,
) -> bool:
    """Keep action auxiliaries off representation-only or unaudited sources."""

    if representation_only:
        return False
    factual = train_cfg.get("factual_action_sources")
    factual_conditioning = train_cfg.get("factual_action_conditioning") or {}
    if factual is not None or factual_conditioning.get("required_sources") is not None:
        if factual is not None:
            factual_names = _config_name_set(train_cfg, "factual_action_sources")
        else:
            nested_cfg = dict(train_cfg)
            nested_cfg["factual_action_sources"] = factual_conditioning.get("required_sources")
            factual_names = _config_name_set(nested_cfg, "factual_action_sources")
        if source_name not in factual_names:
            return False
    allowed = train_cfg.get("action_aux_sources")
    if allowed is None:
        return True
    if isinstance(allowed, str):
        allowed = [value.strip() for value in allowed.split(",") if value.strip()]
    allowed_names = {str(value).strip() for value in allowed if str(value).strip()}
    return source_name is not None and source_name in allowed_names


def validate_factual_action_condition(
    action_condition: torch.Tensor | None,
    *,
    source_name: str,
    require_finite: bool,
) -> None:
    """Validate presence/finiteness while explicitly admitting zero hold actions."""

    if action_condition is None:
        raise RuntimeError(
            f"factual source {source_name!r} has no canonical action condition"
        )
    if require_finite and not bool(torch.isfinite(action_condition).all()):
        raise RuntimeError(
            f"factual source {source_name!r} has a non-finite action condition"
        )


def action_condition_telemetry(
    action_condition: torch.Tensor | None,
) -> dict[str, float | bool]:
    if action_condition is None:
        return {
            "action_condition_abs_mean": 0.0,
            "action_condition_nonzero_fraction": 0.0,
            "action_condition_std": 0.0,
            "action_condition_finite": False,
        }
    values = action_condition.detach().float()
    finite = bool(torch.isfinite(values).all())
    if not finite:
        return {
            "action_condition_abs_mean": float("nan"),
            "action_condition_nonzero_fraction": float("nan"),
            "action_condition_std": float("nan"),
            "action_condition_finite": False,
        }
    return {
        "action_condition_abs_mean": float(values.abs().mean()),
        "action_condition_nonzero_fraction": float((values != 0).float().mean()),
        "action_condition_std": float(values.std(unbiased=False)),
        "action_condition_finite": True,
    }


def compute_context_pixel_action_rank_loss(
    model: torch.nn.Module,
    s: torch.Tensor,
    c: torch.Tensor,
    real_out: dict,
    action_cond: torch.Tensor | None,
    context_rgb: torch.Tensor | None,
    tgt: dict,
    train_cfg: dict,
    *,
    step: int,
    prior_clean_tokens: torch.Tensor | None = None,
    policy_kwargs: dict | None = None,
    stats_keys: list[str] | tuple[str, ...] | None = None,
    collect_gradient_evidence: bool = False,
) -> dict[str, torch.Tensor]:
    rgb_rank_weight = scheduled_aux_weight(
        train_cfg,
        "context_pixel_action_rank_weight",
        step,
        schedule_prefix="context_pixel_action_rank",
    )
    rgb_sep_weight = scheduled_aux_weight(
        train_cfg,
        "context_pixel_action_separation_weight",
        step,
        schedule_prefix="context_pixel_action_rank",
    )
    core_rank_weight = scheduled_native_core_cf_weight(train_cfg, "rank", step)
    core_sep_weight = scheduled_native_core_cf_weight(train_cfg, "separation", step)
    ref = real_out["pred_tokens"]
    rgb_active = max(rgb_rank_weight, rgb_sep_weight) > 0.0
    core_active = max(core_rank_weight, core_sep_weight) > 0.0
    if not rgb_active and not core_active:
        return _context_pixel_action_zero_losses(ref)
    if action_cond is None:
        return _context_pixel_action_zero_losses(ref)
    if core_active and "s_tgt" not in tgt:
        raise ValueError("core action rank requires decoded tgt['s_tgt']")
    if rgb_active and (
        context_rgb is None or "rgb" not in real_out or "rgb_tgt_p" not in tgt
    ):
        raise ValueError("RGB action rank requires context RGB, real RGB and rgb_tgt_p")
    variants = _parse_counterfactual_variants(
        native_core_cf_value(
            train_cfg,
            "variants",
            train_cfg.get("context_pixel_action_rank_variants", ["shuffled"]),
        )
    )
    if not variants:
        return _context_pixel_action_zero_losses(ref)

    batch_size = int(s.shape[0])
    rank_batch_size = int(
        native_core_cf_value(
            train_cfg,
            "batch_size",
            train_cfg.get("context_pixel_action_rank_batch_size", 0),
        )
        or 0
    )
    if rank_batch_size < 0:
        raise ValueError(
            "context_pixel_action_rank_batch_size must be non-negative; "
            f"got {rank_batch_size}"
        )
    rgb_margin = float(train_cfg.get("context_pixel_action_rank_margin", 0.003))
    rgb_sep_margin = float(train_cfg.get("context_pixel_action_separation_margin", 0.006))
    core_margin = float(native_core_cf_value(train_cfg, "margin", 0.001))
    core_sep_margin = float(
        native_core_cf_value(train_cfg, "separation_margin", 0.01)
    )
    min_action_distance = float(
        native_core_cf_value(train_cfg, "negative_min_distance", 0.05)
    )
    scaled_factor = float(train_cfg.get("context_pixel_action_scaled_factor", 2.0))
    grip_contract = normalize_action_grip_contract(
        train_cfg.get("action_grip_contract", "close01")
    )
    wrong_policy_kwargs = dict(policy_kwargs or {})
    wrong_policy_kwargs.update(
        skip_action_proposer=True,
        skip_action_policy=True,
        # The native heads must execute on the wrong-action graph.  The core
        # rank itself uses pred_tokens, but this explicit contract prevents a
        # future fast path from bypassing the native transition machinery.
        skip_native_prediction_heads=False,
    )
    rgb_rank_losses = []
    rgb_sep_losses = []
    rgb_accs = []
    rgb_gaps = []
    rgb_seps = []
    core_rank_losses = []
    core_sep_losses = []
    core_accs = []
    core_gaps = []
    core_seps = []
    negative_distances = []
    negative_min_distances = []
    negative_valid_fractions = []
    negative_nonself_fractions = []
    # These buffers are mutated by hooks during the one ordinary total-loss
    # backward.  The wrong-action branch is used nowhere else, so they provide
    # honest CF-only path evidence without a second 1B-model backward or DDP
    # reducer interference.
    wrong_action_input_grad_evidence = ref.new_zeros(())
    wrong_pred_grad_evidence = ref.new_zeros(())

    def record_nonzero_gradient(buffer: torch.Tensor):
        def hook(gradient: torch.Tensor):
            evidence = (
                torch.isfinite(gradient).all() & (gradient != 0).any()
            ).to(dtype=buffer.dtype)
            buffer.copy_(torch.maximum(buffer, evidence))
            return gradient

        return hook

    for variant in variants:
        # Build the negative on the full local source-homogeneous batch first.
        wrong_cond_full, valid_full, distance_full = build_action_negative(
            action_cond,
            variant,
            step=step,
            min_distance=min_action_distance,
            grip_contract=grip_contract,
            scaled_factor=scaled_factor,
            stats_keys=stats_keys,
        )
        negative_valid_fractions.append(valid_full.float().mean())
        if variant == "shuffled":
            negative_nonself_fractions.append(
                ref.new_tensor(1.0 if batch_size > 1 else 0.0)
            )
        else:
            negative_nonself_fractions.append((distance_full > 0).float().mean())
        if valid_full.any():
            negative_distances.append(distance_full[valid_full].mean())
            negative_min_distances.append(distance_full[valid_full].min())
        valid_indices = torch.nonzero(valid_full, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            continue
        if 0 < rank_batch_size < int(valid_indices.numel()):
            every = max(
                1,
                int(
                    native_core_cf_value(
                        train_cfg,
                        "every",
                        train_cfg.get("context_pixel_action_rank_every", 1),
                    )
                    or 1
                ),
            )
            start = (step // every) % int(valid_indices.numel())
            valid_indices = torch.roll(valid_indices, shifts=-start)[:rank_batch_size]

        selected_s = _batch_index_select(s, valid_indices, batch_size)
        selected_c = _batch_index_select(c, valid_indices, batch_size)
        selected_wrong_cond = _batch_index_select(
            wrong_cond_full, valid_indices, batch_size
        )
        if core_active and collect_gradient_evidence and torch.is_grad_enabled():
            selected_wrong_cond = selected_wrong_cond.detach().requires_grad_(True)
            selected_wrong_cond.register_hook(
                record_nonzero_gradient(wrong_action_input_grad_evidence)
            )
        selected_context_rgb = _batch_index_select(
            context_rgb, valid_indices, batch_size
        )
        selected_prior = _batch_index_select(
            prior_clean_tokens, valid_indices, batch_size
        )
        selected_policy_kwargs = _batch_index_select(
            wrong_policy_kwargs, valid_indices, batch_size
        )
        wrong_out = _forward_joint_model(
            model,
            selected_s,
            selected_c,
            action_cond=selected_wrong_cond,
            context_rgb=selected_context_rgb if rgb_active else None,
            prior_clean_tokens=selected_prior,
            pixel=rgb_active,
            bridging=False,
            policy_kwargs=selected_policy_kwargs,
        )
        if core_active:
            real_pred = _batch_index_select(ref, valid_indices, batch_size).float()
            target_pred = _batch_index_select(
                tgt["s_tgt"], valid_indices, batch_size
            ).to(device=real_pred.device, dtype=real_pred.dtype).detach()
            wrong_pred = wrong_out["pred_tokens"].float()
            if collect_gradient_evidence and wrong_pred.requires_grad:
                wrong_pred.register_hook(
                    record_nonzero_gradient(wrong_pred_grad_evidence)
                )
            if wrong_pred.shape != real_pred.shape or target_pred.shape != real_pred.shape:
                raise ValueError(
                    "core action rank tensor mismatch: "
                    f"real={tuple(real_pred.shape)} wrong={tuple(wrong_pred.shape)} "
                    f"target={tuple(target_pred.shape)}"
                )
            real_core_err = (real_pred - target_pred).pow(2).flatten(1).mean(dim=1)
            wrong_core_err = (wrong_pred - target_pred).pow(2).flatten(1).mean(dim=1)
            core_gap = wrong_core_err - real_core_err
            core_sep = (
                wrong_pred - real_pred.detach()
            ).pow(2).flatten(1).mean(dim=1).sqrt()
            core_rank_losses.append(torch.relu(core_margin - core_gap).mean())
            core_sep_losses.append(torch.relu(core_sep_margin - core_sep).mean())
            core_accs.append((core_gap > 0).float().mean())
            core_gaps.append(core_gap.mean())
            core_seps.append(core_sep.mean())

        if rgb_active:
            selected_real_rgb = _batch_index_select(
                real_out["rgb"], valid_indices, batch_size
            ).float()
            selected_rgb_tgt = _batch_index_select(
                tgt["rgb_tgt_p"], valid_indices, batch_size
            ).to(device=selected_real_rgb.device, dtype=selected_real_rgb.dtype)
            selected_rgb_ref = _batch_index_select(
                tgt.get("rgb_ref_p"), valid_indices, batch_size
            )
            if selected_rgb_ref is None:
                selected_rgb_ref = selected_context_rgb
            selected_rgb_ref = selected_rgb_ref.to(
                device=selected_real_rgb.device, dtype=selected_real_rgb.dtype
            )
            threshold = float(train_cfg.get("context_pixel_action_motion_threshold", 0.03))
            motion_gain = float(train_cfg.get("context_pixel_action_motion_gain", 4.0))
            motion = (
                selected_rgb_tgt.float() - selected_rgb_ref[:, None].float()
            ).abs().mean(dim=2, keepdim=True)
            motion_weight = 1.0 + motion_gain * (motion > threshold).to(
                dtype=selected_real_rgb.dtype
            )

            def weighted_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                error = (a.float() - b.float()).abs()
                weight = motion_weight.to(device=error.device, dtype=error.dtype).expand_as(error)
                return (error * weight).flatten(1).sum(dim=1) / weight.flatten(1).sum(dim=1).clamp_min(1e-6)

            wrong_rgb = wrong_out["rgb"].float()
            real_rgb_err = weighted_l1(selected_real_rgb, selected_rgb_tgt)
            wrong_rgb_err = weighted_l1(wrong_rgb, selected_rgb_tgt)
            rgb_gap = wrong_rgb_err - real_rgb_err
            rgb_sep = weighted_l1(wrong_rgb, selected_real_rgb.detach())
            rgb_rank_losses.append(torch.relu(rgb_margin - rgb_gap).mean())
            rgb_sep_losses.append(torch.relu(rgb_sep_margin - rgb_sep).mean())
            rgb_accs.append((rgb_gap > 0).float().mean())
            rgb_gaps.append(rgb_gap.mean())
            rgb_seps.append(rgb_sep.mean())

    result = _context_pixel_action_zero_losses(ref)

    def mean_or_zero(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values).mean() if values else ref.new_zeros(())

    result.update(
        {
            "L_context_pixel_action_rank": mean_or_zero(rgb_rank_losses),
            "L_context_pixel_action_separation": mean_or_zero(rgb_sep_losses),
            "context_pixel_action_acc": mean_or_zero(rgb_accs),
            "context_pixel_action_gap": mean_or_zero(rgb_gaps),
            "context_pixel_action_rgb_gap": mean_or_zero(rgb_seps),
            "L_core_action_rank": mean_or_zero(core_rank_losses),
            "L_core_action_separation": mean_or_zero(core_sep_losses),
            "core_action_acc": mean_or_zero(core_accs),
            "core_action_gap": mean_or_zero(core_gaps),
            "core_action_sep": mean_or_zero(core_seps),
            "core_action_negative_distance": mean_or_zero(negative_distances),
            "core_action_negative_valid": mean_or_zero(negative_valid_fractions),
        }
    )
    result.update(
        {
            "L_native_core_action_cf_rank": result["L_core_action_rank"],
            "L_native_core_action_cf_separation": result["L_core_action_separation"],
            "native_core_action_cf_acc": result["core_action_acc"],
            "native_core_action_cf_gap": result["core_action_gap"],
            "native_core_action_cf_sep": result["core_action_sep"],
            "native_core_action_cf_negative_distance": result[
                "core_action_negative_distance"
            ],
            "native_core_action_cf_negative_valid": result[
                "core_action_negative_valid"
            ],
            "native_core_action_cf_valid_fraction": result[
                "core_action_negative_valid"
            ],
            "native_core_action_cf_nonself_fraction": mean_or_zero(
                negative_nonself_fractions
            ),
            "native_core_action_cf_min_action_distance": mean_or_zero(
                negative_min_distances
            ),
            "native_core_action_cf_predicted_separation": result[
                "core_action_sep"
            ],
            "native_core_action_cf_wrong_action_input_grad": (
                wrong_action_input_grad_evidence
            ),
            "native_core_action_cf_wrong_pred_grad": wrong_pred_grad_evidence,
        }
    )
    return result


def compute_evaluator_candidate_pairwise_loss(
    model: torch.nn.Module,
    s: torch.Tensor,
    c: torch.Tensor,
    real_out: dict,
    action_tgt: torch.Tensor,
    action_tgt_norm: torch.Tensor,
    train_cfg: dict,
    policy_kwargs: dict | None = None,
    s_tgt: torch.Tensor | None = None,
    depth_tgt: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Rank proposer candidates by action error or observed-future error.

    Real-vs-corrupted pairwise loss proves the evaluator notices obviously bad
    actions. TTC needs a sharper target: among the K candidates that the proposer
    actually emits, the learned score should prefer either the candidate closest
    to the demonstrated action chunk or the rollout closest to the observed future.
    """
    pairwise_weight = float(train_cfg.get("evaluator_candidate_pairwise_weight", 0.0))
    ce_weight = float(train_cfg.get("evaluator_candidate_ce_weight", 0.0))
    if max(pairwise_weight, ce_weight) <= 0 or "proposer_action_cond" not in real_out:
        zero = real_out["pred_tokens"].new_zeros(())
        return {
            "L_evaluator_candidate_pairwise": zero,
            "L_evaluator_candidate_ce": zero,
            "evaluator_candidate_pairwise_acc": zero,
            "evaluator_candidate_pairwise_gap": zero,
            "evaluator_candidate_selected_pose_l1": zero,
            "evaluator_candidate_anchor_pose_l1": zero,
            "evaluator_candidate_oracle_pose_l1": zero,
            "evaluator_candidate_oracle_match": zero,
            "evaluator_candidate_target_selected_error": zero,
            "evaluator_candidate_target_anchor_error": zero,
            "evaluator_candidate_target_oracle_error": zero,
        }

    # Candidate supervision must improve the rollout/evaluator path, not let the
    # proposer move its samples to make the ranking objective artificially easy.
    candidate_cond = real_out["proposer_action_cond"].detach()
    if candidate_cond.ndim != 4 or candidate_cond.shape[-1] != 7:
        raise ValueError(f"proposer_action_cond must be [B,K,k,7], got {tuple(candidate_cond.shape)}")
    bsz, n_candidates, _horizon, _dim = candidate_cond.shape
    if n_candidates < 2:
        zero = real_out["pred_tokens"].new_zeros(())
        return {
            "L_evaluator_candidate_pairwise": zero,
            "L_evaluator_candidate_ce": zero,
            "evaluator_candidate_pairwise_acc": zero,
            "evaluator_candidate_pairwise_gap": zero,
            "evaluator_candidate_selected_pose_l1": zero,
            "evaluator_candidate_anchor_pose_l1": zero,
            "evaluator_candidate_oracle_pose_l1": zero,
            "evaluator_candidate_oracle_match": zero,
            "evaluator_candidate_target_selected_error": zero,
            "evaluator_candidate_target_anchor_error": zero,
            "evaluator_candidate_target_oracle_error": zero,
        }

    pose_l1 = (
        candidate_cond[..., :6].float()
        - action_tgt_norm.float()[:, None]
    ).abs().mean(dim=(2, 3))
    grip_tgt = (action_tgt[..., 6] > 0.5).float()
    grip_prob = candidate_cond[..., 6].float().clamp(1e-5, 1.0 - 1e-5)
    with torch.autocast(device_type="cuda", enabled=False):
        grip_bce = F.binary_cross_entropy(
            grip_prob.float(),
            grip_tgt[:, None].expand_as(grip_prob).float(),
            reduction="none",
        ).mean(dim=2)
    grip_weight = float(train_cfg.get("evaluator_candidate_grip_weight", 0.1))
    action_oracle_err = (pose_l1 + grip_weight * grip_bce).detach()
    ce_target = str(train_cfg.get("evaluator_candidate_ce_target", "hard")).lower()
    if ce_target not in {"hard", "soft_oracle_error", "future_error"}:
        raise ValueError(
            "train.evaluator_candidate_ce_target must be hard, "
            f"soft_oracle_error, or future_error, got {ce_target!r}"
        )
    if ce_target == "future_error" and (s_tgt is None or depth_tgt is None):
        raise ValueError("future_error candidate supervision requires s_tgt and depth_tgt")

    scores = []
    future_errors = []
    rollout_policy_kwargs = dict(policy_kwargs or {})
    rollout_policy_kwargs.update(
        skip_action_proposer=True,
        skip_action_policy=True,
        skip_native_prediction_heads=True,
        detach_progress_input=bool(
            train_cfg.get("candidate_planner_detach_world_for_judge", False)
        ),
    )
    for ci in range(n_candidates):
        cand_out = _forward_joint_model(
            model,
            s,
            c,
            action_cond=candidate_cond[:, ci].to(device=s.device, dtype=s.dtype),
            context_rgb=None,
            pixel=False,
            bridging=False,
            policy_kwargs=rollout_policy_kwargs,
        )
        scores.append(_score_evaluator(cand_out, train_cfg))
        if ce_target == "future_error":
            future_errors.append(
                _candidate_world_future_error(
                    cand_out,
                    s_tgt,
                    depth_tgt,
                    token_weight=float(
                        train_cfg.get("evaluator_candidate_future_token_weight", 1.0)
                    ),
                    depth_weight=float(
                        train_cfg.get("evaluator_candidate_future_depth_weight", 0.3)
                    ),
                )
            )
    score_t = torch.stack(scores, dim=1)
    oracle_err = (
        torch.stack(future_errors, dim=1).detach()
        if ce_target == "future_error"
        else action_oracle_err
    )
    oracle_idx = oracle_err.argmin(dim=1)
    # Supervise every quality-ordered pair. Best-vs-rest alone leaves the
    # evaluator underdetermined for the candidate sets seen at serving time.
    quality_gap = oracle_err[:, None, :] - oracle_err[:, :, None]
    min_quality_gap = float(train_cfg.get("evaluator_candidate_min_quality_gap", 1e-4))
    ordered = quality_gap > min_quality_gap
    score_gap = score_t[:, :, None] - score_t[:, None, :]
    margin = float(train_cfg.get("evaluator_candidate_pairwise_margin", 0.05))
    if bool(ordered.any().detach().cpu()):
        ordered_score_gap = score_gap[ordered]
        pairwise_loss = torch.relu(margin - ordered_score_gap).mean()
        pairwise_acc = (ordered_score_gap > 0).float().mean()
        pairwise_gap = ordered_score_gap.mean()
    else:
        pairwise_loss = score_t.new_zeros(())
        pairwise_acc = score_t.new_zeros(())
        pairwise_gap = score_t.new_zeros(())
    ce_temp = max(float(train_cfg.get("evaluator_candidate_ce_temperature", 0.1)), 1e-4)
    if ce_target in {"hard", "future_error"}:
        ce_loss = F.cross_entropy(score_t.float() / ce_temp, oracle_idx)
    elif ce_target == "soft_oracle_error":
        target_temp = max(
            float(train_cfg.get("evaluator_candidate_target_temperature", 0.1)),
            1e-4,
        )
        target_prob = torch.softmax(-oracle_err.float() / target_temp, dim=1)
        pred_log_prob = torch.log_softmax(score_t.float() / ce_temp, dim=1)
        ce_loss = -(target_prob * pred_log_prob).sum(dim=1).mean()
    selected_idx = score_t.argmax(dim=1)
    selected_pose_l1 = pose_l1.gather(1, selected_idx[:, None]).squeeze(1)
    selected_target_error = oracle_err.gather(1, selected_idx[:, None]).squeeze(1)

    return {
        "L_evaluator_candidate_pairwise": pairwise_loss,
        "L_evaluator_candidate_ce": ce_loss,
        "evaluator_candidate_pairwise_acc": pairwise_acc,
        "evaluator_candidate_pairwise_gap": pairwise_gap,
        "evaluator_candidate_selected_pose_l1": selected_pose_l1.mean(),
        "evaluator_candidate_anchor_pose_l1": pose_l1[:, 0].mean(),
        "evaluator_candidate_oracle_pose_l1": pose_l1.min(dim=1).values.mean(),
        "evaluator_candidate_oracle_match": (selected_idx == oracle_idx).float().mean(),
        "evaluator_candidate_target_selected_error": selected_target_error.mean(),
        "evaluator_candidate_target_anchor_error": oracle_err[:, 0].mean(),
        "evaluator_candidate_target_oracle_error": oracle_err.min(dim=1).values.mean(),
    }


def _candidate_world_future_zero(ref: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = ref.new_zeros(())
    return {
        "L_candidate_world_future_rank": zero,
        "candidate_world_future_gap": zero,
        "candidate_world_future_acc": zero,
        "candidate_world_future_top1": zero,
        "candidate_world_future_active_fraction": zero,
        "candidate_world_future_valid_fraction": zero,
        "candidate_world_future_action_delta": zero,
        "candidate_world_future_factual_error": zero,
        "candidate_world_future_counterfactual_error": zero,
    }


def _candidate_world_future_error(
    out: dict[str, torch.Tensor],
    s_tgt: torch.Tensor,
    depth_tgt: torch.Tensor,
    *,
    token_weight: float,
    depth_weight: float,
) -> torch.Tensor:
    pred = out["pred_tokens"].float()
    token_error = (pred - s_tgt.float()).pow(2).flatten(1).mean(dim=1)
    if depth_weight <= 0.0:
        return token_weight * token_error
    depth = out["depth"].float()
    if depth.shape[-2:] != depth_tgt.shape[-2:]:
        batch, horizon = depth.shape[:2]
        depth = F.interpolate(
            depth.reshape(batch * horizon, 1, *depth.shape[-2:]),
            size=depth_tgt.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, horizon, *depth_tgt.shape[-2:])
    depth_error = (
        _normalize_depth(depth) - _normalize_depth(depth_tgt.float())
    ).abs().flatten(1).mean(dim=1)
    return token_weight * token_error + depth_weight * depth_error


def compute_candidate_world_future_rank_loss(
    model: torch.nn.Module,
    s: torch.Tensor,
    c: torch.Tensor,
    real_out: dict,
    factual_action_cond: torch.Tensor,
    s_tgt: torch.Tensor,
    depth_tgt: torch.Tensor,
    train_cfg: dict,
    policy_kwargs: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Make the demonstrated action explain the observed future best.

    The proposer is frozen and supplies hard counterfactual actions.  The loss
    updates only the action-conditioned world path selected by the caller's
    trainable-prefix contract.  It never trains a scalar candidate identity
    classifier, so success is measured on future prediction itself.
    """

    weight = float(train_cfg.get("candidate_world_future_rank_weight", 0.0) or 0.0)
    if weight <= 0.0 or "proposer_action_cond" not in real_out:
        return _candidate_world_future_zero(real_out["pred_tokens"])

    candidates = real_out["proposer_action_cond"].detach()
    if candidates.ndim != 4 or candidates.shape[-1] != 7:
        raise ValueError(
            f"proposer_action_cond must be [B,K,T,7], got {tuple(candidates.shape)}"
        )
    token_weight = float(train_cfg.get("candidate_world_future_token_weight", 1.0))
    depth_weight = float(train_cfg.get("candidate_world_future_depth_weight", 0.3))
    factual_error = _candidate_world_future_error(
        real_out,
        s_tgt,
        depth_tgt,
        token_weight=token_weight,
        depth_weight=depth_weight,
    )

    rollout_policy_kwargs = dict(policy_kwargs or {})
    rollout_policy_kwargs.update(skip_action_proposer=True, skip_action_policy=True)
    checkpoint_rollouts = bool(
        train_cfg.get("candidate_world_future_activation_checkpoint", False)
    ) and torch.is_grad_enabled()
    counterfactual_errors = []
    for candidate_index in range(candidates.shape[1]):
        candidate_action = candidates[:, candidate_index].to(
            device=s.device, dtype=s.dtype
        )

        def candidate_error(action: torch.Tensor) -> torch.Tensor:
            candidate_out = _forward_joint_model(
                model,
                s,
                c,
                action_cond=action,
                context_rgb=None,
                pixel=False,
                bridging=False,
                policy_kwargs=rollout_policy_kwargs,
            )
            return _candidate_world_future_error(
                candidate_out,
                s_tgt,
                depth_tgt,
                token_weight=token_weight,
                depth_weight=depth_weight,
            )

        if checkpoint_rollouts:
            candidate_value = activation_checkpoint(
                candidate_error,
                candidate_action,
                use_reentrant=False,
            )
        else:
            candidate_value = candidate_error(candidate_action)
        counterfactual_errors.append(candidate_value)
    counterfactual_error = torch.stack(counterfactual_errors, dim=1)
    action_delta = (
        candidates[..., :6].float()
        - factual_action_cond[:, None, :, :6].float()
    ).abs().mean(dim=(2, 3))
    min_delta = float(train_cfg.get("candidate_world_future_min_action_delta", 0.05))
    valid = action_delta >= min_delta
    gap = counterfactual_error - factual_error[:, None]
    margin = float(train_cfg.get("candidate_world_future_rank_margin", 5.0e-4))
    violations = torch.relu(margin - gap)
    if bool(valid.any().detach().cpu()):
        loss = violations[valid].mean()
        gap_mean = gap[valid].mean()
        accuracy = (gap[valid] > 0.0).float().mean()
        active = (violations[valid] > 0.0).float().mean()
        delta_mean = action_delta[valid].mean()
        counterfactual_mean = counterfactual_error[valid].mean()
    else:
        zero = factual_error.new_zeros(())
        loss = gap_mean = accuracy = active = delta_mean = counterfactual_mean = zero
    all_errors = torch.cat([factual_error[:, None], counterfactual_error], dim=1)
    return {
        "L_candidate_world_future_rank": loss,
        "candidate_world_future_gap": gap_mean,
        "candidate_world_future_acc": accuracy,
        "candidate_world_future_top1": (all_errors.argmin(dim=1) == 0).float().mean(),
        "candidate_world_future_active_fraction": active,
        "candidate_world_future_valid_fraction": valid.float().mean(),
        "candidate_world_future_action_delta": delta_mean,
        "candidate_world_future_factual_error": factual_error.mean(),
        "candidate_world_future_counterfactual_error": counterfactual_mean,
    }


def _scheduled_train_scalar(train_cfg: dict, key: str, step: int) -> float:
    value = float(train_cfg.get(key, 0.0) or 0.0)
    if value == 0.0:
        return 0.0
    start = int(train_cfg.get(f"{key}_start_step", 0) or 0)
    if int(step) < start:
        return 0.0
    ramp = int(train_cfg.get(f"{key}_ramp_steps", 0) or 0)
    if ramp > 0:
        progress = min(1.0, max(0.0, (float(step) - float(start)) / float(ramp)))
        return value * progress
    return value


_GRIP_PARTITION_NAMES = (
    "boundary_hold",
    "boundary_up",
    "boundary_down",
    "inclip_hold",
    "inclip_up",
    "inclip_down",
)

_STRICT_GRIP_OVERLAP_WEIGHT_KEYS = (
    "direct_policy_grip_transition_weight",
    "direct_policy_grip_boundary_weight",
    "direct_policy_grip_transition_bce_weight",
    "direct_policy_grip_transition_up_bce_weight",
    "direct_policy_grip_transition_down_bce_weight",
    "direct_policy_grip_transition_margin_weight",
    "direct_policy_grip_transition_delta_bce_weight",
    "direct_policy_grip_transition_delta_up_bce_weight",
    "direct_policy_grip_transition_delta_down_bce_weight",
    "direct_policy_grip_nontransition_delta_weight",
    "direct_policy_grip_event_logit_margin_weight",
    "direct_policy_grip_boundary_logit_margin_weight",
    "direct_policy_grip_rate_mse_weight",
    "direct_policy_grip_boundary_bce_weight",
    "direct_policy_grip_boundary_up_bce_weight",
    "direct_policy_grip_boundary_down_bce_weight",
    "direct_policy_grip_boundary_rate_mse_weight",
)
_LEGACY_GRIP_CONTRACT_WARNING_EMITTED = False


def _grip_partition_contract_enabled(config: dict | None) -> bool:
    return bool((config or {}).get("direct_policy_grip_partition_contract", False))


def _validate_grip_partition_contract_config(train_cfg: dict, *, warn_legacy: bool = True) -> bool:
    strict = _grip_partition_contract_enabled(train_cfg)
    grip_owner = str(train_cfg.get("direct_policy_grip_owner", "auto")).strip().lower()
    if grip_owner not in ("auto", "absolute", "delta_composed"):
        raise ValueError(
            "direct_policy_grip_owner must be auto/absolute/delta_composed, "
            f"got {grip_owner!r}"
        )
    if strict and grip_owner == "absolute":
        delta_state_weight = float(train_cfg.get("direct_policy_grip_delta_state_bce_weight", 0.0) or 0.0)
        if delta_state_weight != 0.0:
            raise ValueError(
                "direct_policy_grip_owner=absolute forbids direct_policy_grip_delta_state_bce_weight; "
                "the delta head is event auxiliary only and cannot become a second state owner"
            )
    nonzero_overlap = [
        key
        for key in _STRICT_GRIP_OVERLAP_WEIGHT_KEYS
        if float(train_cfg.get(key, 0.0) or 0.0) != 0.0
    ]
    if strict and nonzero_overlap:
        raise ValueError(
            "direct_policy_grip_partition_contract=true forbids legacy overlapping gripper auxiliary weights; "
            "set these weights to 0: " + ", ".join(nonzero_overlap)
        )
    global _LEGACY_GRIP_CONTRACT_WARNING_EMITTED
    has_delta_contract_loss = any(
        float(train_cfg.get(key, 0.0) or 0.0) != 0.0
        for key in (
            "direct_policy_grip_delta_ce_weight",
            "direct_policy_grip_delta_state_bce_weight",
            "direct_policy_grip_delta_transition_up_margin_weight",
            "direct_policy_grip_delta_transition_down_margin_weight",
            "direct_policy_grip_delta_boundary_up_margin_weight",
            "direct_policy_grip_delta_boundary_down_margin_weight",
        )
    )
    if warn_legacy and not strict and (nonzero_overlap or has_delta_contract_loss) and not _LEGACY_GRIP_CONTRACT_WARNING_EMITTED:
        warnings.warn(
            "Legacy gripper loss semantics are active: transition/boundary event tokens may be consumed by "
            "multiple auxiliary losses and state weights retain multiplicative overlap. Set "
            "direct_policy_grip_partition_contract=true, zero the reported legacy overlap weights, and "
            "regenerate sampler metadata to migrate to the six-partition contract.",
            FutureWarning,
            stacklevel=2,
        )
        _LEGACY_GRIP_CONTRACT_WARNING_EMITTED = True
    return strict


def build_gripper_event_partitions(
    grip_tgt: torch.Tensor,
    action_prev_grip: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Return the six mutually-exclusive gripper event masks over ``[B, K]``."""
    if grip_tgt.ndim != 2 or grip_tgt.shape[1] <= 0:
        raise ValueError(f"grip_tgt must be [B,K] with K > 0, got {tuple(grip_tgt.shape)}")
    target = grip_tgt > 0.5
    if action_prev_grip is None:
        previous_boundary = target[:, 0]
    else:
        previous_boundary = action_prev_grip.reshape(action_prev_grip.shape[0], -1)[:, -1] > 0.5
        if previous_boundary.shape[0] != target.shape[0]:
            raise ValueError("action_prev_grip batch size does not match grip_tgt")
    previous = torch.cat([previous_boundary[:, None], target[:, :-1]], dim=1)
    up = target & (~previous)
    down = (~target) & previous
    hold = ~(up | down)
    boundary = torch.zeros_like(target)
    boundary[:, 0] = True
    inclip = ~boundary
    partitions = {
        "boundary_hold": boundary & hold,
        "boundary_up": boundary & up,
        "boundary_down": boundary & down,
        "inclip_hold": inclip & hold,
        "inclip_up": inclip & up,
        "inclip_down": inclip & down,
    }
    one_hot_count = torch.stack([partitions[name] for name in _GRIP_PARTITION_NAMES], dim=-1).sum(dim=-1)
    if not bool((one_hot_count == 1).all().detach().cpu()):
        raise RuntimeError("invalid gripper partition: every [B,K] element must have exactly one label")
    return partitions


def _global_mean_with_ddp_grad(
    local_sum: torch.Tensor,
    global_sum: torch.Tensor,
    global_count: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    """Expose a global mean while compensating for DDP's gradient averaging."""
    count = global_count.to(device=local_sum.device, dtype=local_sum.dtype)
    if float(count.detach().cpu()) <= 0.0:
        return local_sum * 0.0
    global_value = global_sum.to(device=local_sum.device, dtype=local_sum.dtype) / count
    grad_carrier = local_sum * (float(world_size) / count)
    return grad_carrier + (global_value - grad_carrier.detach())


def _distributed_sum_count_mean(
    local_sum: torch.Tensor,
    local_count: torch.Tensor | float | int,
) -> torch.Tensor:
    sums = local_sum.reshape(1)
    counts = torch.as_tensor(local_count, device=local_sum.device, dtype=local_sum.dtype).reshape(1)
    return _distributed_sum_count_means(sums, counts)[0]


def _distributed_sum_count_means(
    local_sums: torch.Tensor,
    local_counts: torch.Tensor,
) -> torch.Tensor:
    if local_sums.ndim != 1 or local_counts.shape != local_sums.shape:
        raise ValueError("local_sums and local_counts must be same-shaped 1D tensors")
    counts = local_counts.to(device=local_sums.device, dtype=local_sums.dtype)
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return local_sums / counts.clamp_min(1.0)
    world_size = dist.get_world_size()
    reduce_on_cpu = dist.get_backend() == "gloo" and local_sums.is_cuda
    reduce_device = torch.device("cpu") if reduce_on_cpu else local_sums.device
    reduced = torch.cat(
        [local_sums.detach().to(device=reduce_device), counts.detach().to(device=reduce_device)]
    )
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    n = local_sums.numel()
    return torch.stack(
        [
            _global_mean_with_ddp_grad(local_sums[i], reduced[i], reduced[n + i], world_size)
            for i in range(n)
        ]
    )


def _masked_distributed_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(device=values.device, dtype=values.dtype)
    return _distributed_sum_count_mean((values * mask_f).sum(), mask_f.sum())


def _distributed_counts(local_counts: torch.Tensor) -> torch.Tensor:
    counts = local_counts.detach().clone()
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return counts
    reduce_on_cpu = dist.get_backend() == "gloo" and counts.is_cuda
    reduced = counts.cpu() if reduce_on_cpu else counts
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced.to(device=counts.device) if reduce_on_cpu else reduced


def _partition_balanced_distributed_mean(
    values: torch.Tensor,
    partitions: dict[str, torch.Tensor],
    partition_weights: dict[str, float],
) -> torch.Tensor:
    masks = [partitions[name].to(device=values.device, dtype=values.dtype) for name in _GRIP_PARTITION_NAMES]
    local_sums = torch.stack([(values * mask).sum() for mask in masks])
    local_counts = torch.stack([mask.sum() for mask in masks])
    means = _distributed_sum_count_means(local_sums, local_counts)
    global_counts = _distributed_counts(local_counts)
    coefficients = values.new_tensor([partition_weights[name] for name in _GRIP_PARTITION_NAMES])
    active = (global_counts > 0).to(dtype=values.dtype)
    weighted = coefficients * active
    return (means * weighted).sum() / weighted.sum().clamp_min(1e-6)


def compute_policy_flow_matching_loss(
    out: dict,
    action_tgt: torch.Tensor,
    action_tgt_norm: torch.Tensor,
    train_cfg: dict,
) -> dict[str, torch.Tensor]:
    ref = action_tgt_norm
    weight = float(train_cfg.get("policy_flow_weight", 0.0) or 0.0)
    required = ("policy_flow_velocity", "policy_flow_input", "policy_flow_t")
    if weight <= 0.0 or any(key not in out for key in required):
        zero = ref.new_zeros(())
        return {
            "L_policy_flow": zero,
            "policy_flow_mse": zero,
            "policy_flow_pose_mse": zero,
            "policy_flow_grip_mse": zero,
            "policy_flow_recon_l1": zero,
            "policy_flow_recon_pose_l1": zero,
            "policy_flow_recon_grip_l1": zero,
        }

    velocity = out["policy_flow_velocity"].float()
    flow_input = out["policy_flow_input"].float()
    flow_t = out["policy_flow_t"].float()
    clean = _policy_flow_clean_action(action_tgt, action_tgt_norm, train_cfg).to(
        device=velocity.device,
        dtype=velocity.dtype,
    )
    horizon = min(int(velocity.shape[1]), int(flow_input.shape[1]), int(clean.shape[1]))
    velocity = velocity[:, :horizon]
    flow_input = flow_input[:, :horizon]
    clean = clean[:, :horizon]
    flow_t = flow_t[:, :horizon] if flow_t.shape[1] == horizon else flow_t[:, :1].expand(-1, horizon, -1)
    noise = (flow_input - flow_t * clean) / (1.0 - flow_t).clamp_min(1e-4)
    target_velocity = clean - noise
    mse = F.mse_loss(velocity, target_velocity, reduction="none")
    pose_mse = mse[..., :6].mean()
    has_flow_grip = int(mse.shape[-1]) > 6
    grip_mse = mse[..., 6].mean() if has_flow_grip else mse.new_zeros(())
    recon = flow_input + (1.0 - flow_t) * velocity
    recon_l1 = (recon - clean).abs()
    pose_weight = float(train_cfg.get("policy_flow_pose_weight", 1.0) or 1.0)
    grip_weight = float(train_cfg.get("policy_flow_grip_weight", 1.0) or 1.0)
    pose_dims = min(6, int(mse.shape[-1]))
    weighted_dims = pose_weight * float(pose_dims)
    weighted_sum = pose_weight * float(pose_dims) * pose_mse
    if has_flow_grip:
        weighted_dims += grip_weight
        weighted_sum = weighted_sum + grip_weight * grip_mse
    L_flow = weighted_sum / max(weighted_dims, 1e-8)
    first_weight = float(train_cfg.get("policy_flow_first_weight", 0.0) or 0.0)
    if first_weight > 0.0:
        L_flow = L_flow + first_weight * mse[:, 0].mean()
    return {
        "L_policy_flow": L_flow,
        "policy_flow_mse": mse.mean(),
        "policy_flow_pose_mse": pose_mse,
        "policy_flow_grip_mse": grip_mse,
        "policy_flow_recon_l1": recon_l1.mean(),
        "policy_flow_recon_pose_l1": recon_l1[..., :6].mean(),
        "policy_flow_recon_grip_l1": recon_l1[..., 6].mean() if has_flow_grip else recon_l1.new_zeros(()),
    }


def compute_direct_policy_loss(
    out: dict,
    action_tgt: torch.Tensor,
    action_tgt_norm: torch.Tensor,
    train_cfg: dict,
    action_prev_grip: torch.Tensor | None = None,
    step: int = 0,
) -> dict[str, torch.Tensor]:
    weight = float(train_cfg.get("direct_policy_weight", 0.0))
    report_metrics = bool(train_cfg.get("direct_policy_report_metrics", False))
    strict_partition_contract = _validate_grip_partition_contract_config(train_cfg)
    grip_owner = str(train_cfg.get("direct_policy_grip_owner", "auto")).strip().lower()
    head = str(train_cfg.get("direct_policy_head", "policy")).strip().lower()
    if head in ("policy", "direct", "full"):
        pose_key, grip_key, cond_key = "policy_pose_norm", "policy_gripper_logit", "policy_action_cond"
    elif head in ("prior", "prior_policy", "oxe_prior"):
        pose_key, grip_key, cond_key = "prior_policy_pose_norm", "prior_policy_gripper_logit", "prior_policy_action_cond"
    elif head in ("base", "base_policy"):
        pose_key, grip_key, cond_key = "base_policy_pose_norm", "base_policy_gripper_logit", None
    else:
        raise ValueError(f"unknown direct_policy_head={head!r}; expected policy, prior, or base")
    if (weight <= 0 and not report_metrics) or pose_key not in out or grip_key not in out or (cond_key is not None and cond_key not in out):
        zero = action_tgt_norm.new_zeros(())
        return {
            "L_direct_policy": zero,
            "direct_policy_pose_l1": zero,
            "direct_policy_first_pose_l1": zero,
            "direct_policy_worst_step_pose_loss": zero,
            "direct_policy_cumulative_pose_loss": zero,
            "direct_policy_endpoint_pose_loss": zero,
            "direct_policy_direction_loss": zero,
            "direct_policy_grip_acc": zero,
            "direct_policy_grip_transition_acc": zero,
            "direct_policy_grip_transition_rate": zero,
            "direct_policy_grip_pos_rate": zero,
            "direct_policy_grip_prob_mean": zero,
            "direct_policy_grip_pred_pos_rate": zero,
            "direct_policy_grip_pos_acc": zero,
            "direct_policy_grip_neg_acc": zero,
            "direct_policy_grip_precision": zero,
            "direct_policy_grip_recall": zero,
            "direct_policy_grip_bce": zero,
            "direct_policy_grip_natural_bce": zero,
            "direct_policy_first_grip_natural_bce": zero,
            "direct_policy_grip_transition_bce": zero,
            "direct_policy_grip_transition_margin": zero,
            "direct_policy_grip_transition_delta_bce": zero,
            "direct_policy_grip_transition_delta_up_bce": zero,
            "direct_policy_grip_transition_delta_down_bce": zero,
            "direct_policy_grip_nontransition_delta_abs": zero,
            "direct_policy_grip_event_logit_margin": zero,
            "direct_policy_grip_boundary_logit_margin": zero,
            "direct_policy_grip_rate_mse": zero,
            "direct_policy_grip_boundary_bce": zero,
            "direct_policy_grip_boundary_rate_mse": zero,
            "direct_policy_grip_boundary_pos_rate": zero,
            "direct_policy_grip_boundary_prob_mean": zero,
            "direct_policy_grip_transition_up_acc": zero,
            "direct_policy_grip_transition_down_acc": zero,
            "direct_policy_grip_boundary_up_acc": zero,
            "direct_policy_grip_boundary_down_acc": zero,
            "direct_policy_grip_boundary_acc": zero,
            "direct_policy_grip_boundary_rate": zero,
            "direct_policy_grip_delta_ce": zero,
            "direct_policy_grip_delta_natural_ce": zero,
            "direct_policy_grip_delta_acc": zero,
            "direct_policy_grip_delta_hold_acc": zero,
            "direct_policy_grip_delta_up_acc": zero,
            "direct_policy_grip_delta_down_acc": zero,
            "direct_policy_grip_delta_boundary_up_acc": zero,
            "direct_policy_grip_delta_boundary_down_acc": zero,
            "direct_policy_grip_delta_state_bce": zero,
            "direct_policy_grip_delta_state_acc": zero,
            "direct_policy_grip_delta_state_pos_acc": zero,
            "direct_policy_grip_delta_state_neg_acc": zero,
            "direct_policy_grip_delta_state_transition_up_acc": zero,
            "direct_policy_grip_delta_state_transition_down_acc": zero,
            "direct_policy_grip_delta_transition_up_margin": zero,
            "direct_policy_grip_delta_transition_down_margin": zero,
            "direct_policy_grip_delta_boundary_up_margin": zero,
            "direct_policy_grip_delta_boundary_down_margin": zero,
            "direct_policy_grip_composed_acc": zero,
            "direct_policy_grip_composed_pos_acc": zero,
            "direct_policy_grip_composed_neg_acc": zero,
            "direct_policy_grip_composed_transition_up_acc": zero,
            "direct_policy_grip_composed_transition_down_acc": zero,
            "direct_policy_grip_composed_boundary_up_acc": zero,
            "direct_policy_grip_composed_boundary_down_acc": zero,
        }
    if float(train_cfg.get("direct_policy_grip_delta_ce_weight", 0.0) or 0.0) > 0.0 and "policy_grip_delta_logits" not in out:
        raise RuntimeError(
            "direct_policy_grip_delta_ce_weight is positive, but the action policy did not emit "
            "policy_grip_delta_logits. Enable model.policy_enable_grip_delta_head=true instead of "
            "silently training the absolute gripper head only."
        )
    pose_pred = out[pose_key].float()
    grip_logit = out[grip_key].float()
    horizon = min(pose_pred.shape[1], action_tgt_norm.shape[1], grip_logit.shape[1], action_tgt.shape[1])
    pose_pred = pose_pred[:, :horizon]
    grip_logit = grip_logit[:, :horizon]
    pose_tgt = action_tgt_norm[:, :horizon].float()
    grip_tgt = (action_tgt[:, :horizon, 6] > 0.5).float()
    if bool(train_cfg.get("direct_policy_require_action_prev_grip", False)) and action_prev_grip is None:
        raise RuntimeError(
            "direct_policy_require_action_prev_grip=true, but action_prev_grip is missing. "
            "Strict gripper auxiliary losses require the previous gripper state so "
            "t=0 boundary events are supervised instead of silently being treated as holds."
        )
    huber_delta = float(train_cfg.get("direct_policy_huber_delta", train_cfg.get("huber_delta", 1.0)))

    pose_element_err = F.smooth_l1_loss(
        pose_pred,
        pose_tgt,
        beta=huber_delta,
        reduction="none",
    )
    pose_step_err = pose_element_err.mean(dim=2)
    pose_err = pose_step_err.mean(dim=1)
    first_pose_err = F.smooth_l1_loss(pose_pred[:, 0], pose_tgt[:, 0], beta=huber_delta, reduction="none").mean(dim=1)
    worst_step_pose_err = pose_step_err.amax(dim=1)
    cumulative_pose_err = F.smooth_l1_loss(
        pose_pred.cumsum(dim=1),
        pose_tgt.cumsum(dim=1),
        beta=huber_delta,
        reduction="none",
    ).mean(dim=(1, 2))
    endpoint_pose_err = F.smooth_l1_loss(
        pose_pred.sum(dim=1),
        pose_tgt.sum(dim=1),
        beta=huber_delta,
        reduction="none",
    ).mean(dim=1)
    direction_loss = 1.0 - F.cosine_similarity(
        pose_pred.reshape(pose_pred.shape[0], -1),
        pose_tgt.reshape(pose_tgt.shape[0], -1),
        dim=1,
        eps=1.0e-6,
    )
    if horizon > 1:
        delta_err = F.smooth_l1_loss(
            pose_pred[:, 1:] - pose_pred[:, :-1],
            pose_tgt[:, 1:] - pose_tgt[:, :-1],
            beta=huber_delta,
            reduction="none",
        ).mean(dim=(1, 2))
    else:
        delta_err = pose_err.new_zeros(pose_err.shape)

    grip_bce_raw = F.binary_cross_entropy_with_logits(grip_logit, grip_tgt, reduction="none")
    # The strict event-partition objective intentionally changes the effective
    # gripper prior so rare open/close transitions are not drowned by holds.
    # It must therefore be paired with an un-reweighted BCE on the natural
    # sample distribution.  This keeps the serving threshold (0.5 by default)
    # calibrated while the balanced term supplies transition recall.
    natural_grip_bce = _distributed_sum_count_mean(
        grip_bce_raw.sum(), grip_bce_raw.new_tensor(grip_bce_raw.numel())
    )
    natural_first_grip_bce = _distributed_sum_count_mean(
        grip_bce_raw[:, 0].sum(), grip_bce_raw.new_tensor(grip_bce_raw.shape[0])
    )
    grip_prob = torch.sigmoid(grip_logit)
    grip_bce = grip_bce_raw
    focal_gamma = float(train_cfg.get("direct_policy_grip_focal_gamma", 0.0) or 0.0)
    if focal_gamma > 0:
        p_t = torch.where(grip_tgt > 0.5, grip_prob, 1.0 - grip_prob).clamp(1e-6, 1.0)
        grip_bce = grip_bce * (1.0 - p_t).pow(focal_gamma)
    boundary_transition = grip_tgt.new_zeros(grip_tgt[:, 0].shape)
    delta_state_weight_now = _scheduled_train_scalar(train_cfg, "direct_policy_grip_delta_state_bce_weight", int(step))
    delta_boundary_specific_weight_now = max(
        _scheduled_train_scalar(train_cfg, "direct_policy_grip_delta_boundary_up_margin_weight", int(step)),
        _scheduled_train_scalar(train_cfg, "direct_policy_grip_delta_boundary_down_margin_weight", int(step)),
        float(train_cfg.get("direct_policy_grip_delta_boundary_up_weight", 0.0) or 0.0),
        float(train_cfg.get("direct_policy_grip_delta_boundary_down_weight", 0.0) or 0.0),
        float(train_cfg.get("direct_policy_grip_delta_state_boundary_up_weight", 0.0) or 0.0),
        float(train_cfg.get("direct_policy_grip_delta_state_boundary_down_weight", 0.0) or 0.0),
    )
    if action_prev_grip is not None:
        prev_grip = action_prev_grip.float().reshape(action_prev_grip.shape[0], -1)[:, -1]
        prev_tgt = torch.cat([(prev_grip > 0.5).float()[:, None], grip_tgt[:, :-1]], dim=1)
    else:
        if float(delta_state_weight_now) > 0.0 or float(delta_boundary_specific_weight_now) > 0.0:
            raise RuntimeError(
                "Delta gripper state/boundary supervision is enabled, but action_prev_grip is missing. "
                "The previous gripper state is required so t=0 boundary events are supervised instead of silently becoming hold."
            )
        prev_tgt = torch.cat([grip_tgt[:, :1], grip_tgt[:, :-1]], dim=1)
    event_partitions = build_gripper_event_partitions(grip_tgt, action_prev_grip)
    boundary_hold_mask = event_partitions["boundary_hold"]
    boundary_up_full_mask = event_partitions["boundary_up"]
    boundary_down_full_mask = event_partitions["boundary_down"]
    inclip_hold_mask = event_partitions["inclip_hold"]
    inclip_up_mask = event_partitions["inclip_up"]
    inclip_down_mask = event_partitions["inclip_down"]
    transition_up_mask = boundary_up_full_mask | inclip_up_mask
    transition_down_mask = boundary_down_full_mask | inclip_down_mask
    transition_mask = transition_up_mask | transition_down_mask
    grip_transition = transition_mask.float()
    boundary_transition = (boundary_up_full_mask | boundary_down_full_mask)[:, 0].float()
    event_target = torch.zeros_like(grip_tgt, dtype=torch.long)
    event_target = torch.where(transition_up_mask, torch.ones_like(event_target), event_target)
    event_target = torch.where(transition_down_mask, torch.full_like(event_target, 2), event_target)
    transition_weight = _scheduled_train_scalar(train_cfg, "direct_policy_grip_transition_weight", int(step))
    class_weight = torch.ones_like(grip_bce)
    pos_weight_static = float(train_cfg.get("direct_policy_grip_pos_weight", 0.0) or 0.0)
    neg_weight_static = float(train_cfg.get("direct_policy_grip_neg_weight", 0.0) or 0.0)
    if pos_weight_static > 0 or neg_weight_static > 0:
        pos_w = grip_bce.new_tensor(pos_weight_static if pos_weight_static > 0 else 1.0)
        neg_w = grip_bce.new_tensor(neg_weight_static if neg_weight_static > 0 else 1.0)
        class_weight = class_weight * torch.where(grip_tgt > 0.5, pos_w, neg_w)
    if bool(train_cfg.get("direct_policy_grip_class_balance", False)):
        if strict_partition_contract:
            pos = _distributed_sum_count_mean(grip_tgt.sum(), grip_tgt.numel())
        else:
            pos = grip_tgt.mean()
        pos = pos.clamp(1e-4, 1.0 - 1e-4)
        neg = (1.0 - pos).clamp(1e-4, 1.0 - 1e-4)
        max_class_weight = float(train_cfg.get("direct_policy_grip_class_balance_cap", 8.0))
        pos_w = (0.5 / pos).clamp(max=max_class_weight)
        neg_w = (0.5 / neg).clamp(max=max_class_weight)
        class_weight = class_weight * torch.where(grip_tgt > 0.5, pos_w, neg_w)
    step_weight = class_weight
    if transition_weight > 0:
        step_weight = step_weight * (1.0 + transition_weight * grip_transition)
    grip_err = (grip_bce * step_weight).sum(dim=1) / step_weight.sum(dim=1).clamp_min(1e-6)
    first_grip_bce = F.binary_cross_entropy_with_logits(grip_logit[:, 0], grip_tgt[:, 0], reduction="none")
    if focal_gamma > 0:
        first_p_t = torch.where(grip_tgt[:, 0] > 0.5, grip_prob[:, 0], 1.0 - grip_prob[:, 0]).clamp(1e-6, 1.0)
        first_grip_bce = first_grip_bce * (1.0 - first_p_t).pow(focal_gamma)
    boundary_weight = _scheduled_train_scalar(train_cfg, "direct_policy_grip_boundary_weight", int(step))
    first_grip_err = first_grip_bce * class_weight[:, 0] * (1.0 + boundary_weight * boundary_transition)
    inclip_transition_mask = inclip_up_mask | inclip_down_mask
    transition_grip_bce = grip_prob.new_zeros(())
    transition_grip_up_bce = grip_prob.new_zeros(())
    transition_grip_down_bce = grip_prob.new_zeros(())
    boundary_mask = boundary_transition > 0.5
    boundary_up_mask = boundary_up_full_mask[:, 0]
    boundary_down_mask = boundary_down_full_mask[:, 0]
    transition_grip_margin = grip_prob.new_zeros(())
    margin_num = grip_logit.new_zeros(())
    margin_den = grip_logit.new_zeros(())
    margin_value = float(train_cfg.get("direct_policy_grip_transition_margin", 1.0) or 1.0)
    up_margin_value = float(train_cfg.get("direct_policy_grip_transition_up_margin", margin_value) or margin_value)
    down_margin_value = float(train_cfg.get("direct_policy_grip_transition_down_margin", margin_value) or margin_value)
    boundary_margin_value = float(train_cfg.get("direct_policy_grip_boundary_margin", margin_value) or margin_value)
    boundary_up_margin_value = float(train_cfg.get("direct_policy_grip_boundary_up_margin", boundary_margin_value) or boundary_margin_value)
    boundary_down_margin_value = float(train_cfg.get("direct_policy_grip_boundary_down_margin", boundary_margin_value) or boundary_margin_value)
    boundary_margin_weight = float(train_cfg.get("direct_policy_grip_transition_margin_boundary_weight", 1.0) or 1.0)
    include_boundary_margin = bool(train_cfg.get("direct_policy_grip_transition_margin_include_boundary", True))
    logit_delta = None
    if horizon > 1:
        logit_delta = grip_logit[:, 1:] - grip_logit[:, :-1]
        in_clip_transition_up = inclip_up_mask[:, 1:].float()
        in_clip_transition_down = inclip_down_mask[:, 1:].float()
        margin_num = margin_num + (F.relu(up_margin_value - logit_delta) * in_clip_transition_up).sum()
        margin_num = margin_num + (F.relu(down_margin_value + logit_delta) * in_clip_transition_down).sum()
        margin_den = margin_den + in_clip_transition_up.sum() + in_clip_transition_down.sum()
    if not strict_partition_contract and include_boundary_margin and action_prev_grip is not None:
        margin_num = margin_num + boundary_margin_weight * (
            F.relu(boundary_up_margin_value - grip_logit[:, 0]) * boundary_up_mask.float()
        ).sum()
        margin_num = margin_num + boundary_margin_weight * (
            F.relu(boundary_down_margin_value + grip_logit[:, 0]) * boundary_down_mask.float()
        ).sum()
        margin_den = margin_den + boundary_margin_weight * (
            boundary_up_mask.float().sum() + boundary_down_mask.float().sum()
        )
    transition_grip_margin = (
        _distributed_sum_count_mean(margin_num, margin_den)
        if strict_partition_contract
        else margin_num / margin_den.clamp_min(1.0)
    )
    boundary_grip_logit_margin = grip_prob.new_zeros(())
    boundary_logit_margin_num = grip_logit.new_zeros(())
    boundary_logit_margin_den = grip_logit.new_zeros(())
    if action_prev_grip is not None:
        boundary_logit_margin_num = boundary_logit_margin_num + (
            F.relu(boundary_up_margin_value - grip_logit[:, 0]) * boundary_up_mask.float()
        ).sum()
        boundary_logit_margin_num = boundary_logit_margin_num + (
            F.relu(boundary_down_margin_value + grip_logit[:, 0]) * boundary_down_mask.float()
        ).sum()
        boundary_logit_margin_den = boundary_logit_margin_den + boundary_up_mask.float().sum() + boundary_down_mask.float().sum()
    boundary_grip_logit_margin = (
        _distributed_sum_count_mean(boundary_logit_margin_num, boundary_logit_margin_den)
        if strict_partition_contract
        else boundary_logit_margin_num / boundary_logit_margin_den.clamp_min(1.0)
    )
    transition_event_logit_margin = grip_prob.new_zeros(())
    event_margin_num = grip_logit.new_zeros(())
    event_margin_den = grip_logit.new_zeros(())
    event_up_margin = float(train_cfg.get("direct_policy_grip_transition_up_logit_margin", 0.0) or 0.0)
    event_down_margin = float(train_cfg.get("direct_policy_grip_transition_down_logit_margin", 0.0) or 0.0)
    if event_up_margin > 0:
        up_event_mask = (inclip_up_mask if strict_partition_contract else transition_up_mask).float()
        event_margin_num = event_margin_num + (F.relu(event_up_margin - grip_logit) * up_event_mask).sum()
        event_margin_den = event_margin_den + up_event_mask.sum()
    if event_down_margin > 0:
        down_event_mask = (inclip_down_mask if strict_partition_contract else transition_down_mask).float()
        event_margin_num = event_margin_num + (F.relu(event_down_margin + grip_logit) * down_event_mask).sum()
        event_margin_den = event_margin_den + down_event_mask.sum()
    transition_event_logit_margin = (
        _distributed_sum_count_mean(event_margin_num, event_margin_den)
        if strict_partition_contract
        else event_margin_num / event_margin_den.clamp_min(1.0)
    )
    transition_delta_bce = grip_prob.new_zeros(())
    transition_delta_up_bce = grip_prob.new_zeros(())
    transition_delta_down_bce = grip_prob.new_zeros(())
    nontransition_delta_abs = grip_prob.new_zeros(())
    if logit_delta is not None:
        delta_scale = max(float(train_cfg.get("direct_policy_grip_transition_delta_logit_scale", 1.0) or 1.0), 1e-6)
        direction_delta_logits = logit_delta / delta_scale
        delta_up_mask = transition_up_mask[:, 1:]
        delta_down_mask = transition_down_mask[:, 1:]
        nontransition_mask = ~(transition_mask[:, 1:])
        up_raw = F.binary_cross_entropy_with_logits(
            direction_delta_logits,
            torch.ones_like(direction_delta_logits),
            reduction="none",
        )
        down_raw = F.binary_cross_entropy_with_logits(
            -direction_delta_logits,
            torch.ones_like(direction_delta_logits),
            reduction="none",
        )
        if strict_partition_contract:
            direction_masks = (delta_up_mask, delta_down_mask, nontransition_mask)
            direction_values = (up_raw, down_raw, direction_delta_logits.abs())
            direction_local_sums = torch.stack(
                [(value * mask.to(dtype=value.dtype)).sum() for value, mask in zip(direction_values, direction_masks)]
            )
            direction_local_counts = torch.stack(
                [mask.to(dtype=direction_delta_logits.dtype).sum() for mask in direction_masks]
            )
            transition_delta_up_bce, transition_delta_down_bce, nontransition_delta_abs = (
                _distributed_sum_count_means(direction_local_sums, direction_local_counts).unbind()
            )
            event_sum = direction_local_sums[0] + direction_local_sums[1]
            event_count = direction_local_counts[0] + direction_local_counts[1]
            transition_delta_bce = _distributed_sum_count_mean(event_sum, event_count)
        else:
            if bool(delta_up_mask.any().detach().cpu()):
                transition_delta_up_bce = up_raw[delta_up_mask].mean()
            if bool(delta_down_mask.any().detach().cpu()):
                transition_delta_down_bce = down_raw[delta_down_mask].mean()
            delta_count = delta_up_mask.float().sum() + delta_down_mask.float().sum()
            transition_delta_bce = (
                transition_delta_up_bce * delta_up_mask.float().sum()
                + transition_delta_down_bce * delta_down_mask.float().sum()
            ) / delta_count.clamp_min(1.0)
            if bool(nontransition_mask.any().detach().cpu()):
                nontransition_delta_abs = direction_delta_logits[nontransition_mask].abs().mean()
    if strict_partition_contract:
        rate_means = _distributed_sum_count_means(
            torch.stack([grip_prob.sum(), grip_tgt.sum()]),
            grip_prob.new_tensor([grip_prob.numel(), grip_tgt.numel()]),
        )
        grip_rate_mse = (rate_means[0] - rate_means[1]).pow(2)
    else:
        grip_rate_mse = (grip_prob.mean() - grip_tgt.mean()).pow(2)
    boundary_grip_bce = grip_prob.new_zeros(())
    boundary_grip_up_bce = grip_prob.new_zeros(())
    boundary_grip_down_bce = grip_prob.new_zeros(())
    boundary_grip_prob_mean = grip_prob.new_zeros(())
    boundary_grip_pos_rate = grip_prob.new_zeros(())
    boundary_grip_rate_mse = grip_prob.new_zeros(())
    boundary_event_full_mask = boundary_up_full_mask | boundary_down_full_mask
    bce_value = grip_bce * class_weight
    if strict_partition_contract:
        bce_masks = (
            inclip_transition_mask,
            inclip_up_mask,
            inclip_down_mask,
            boundary_event_full_mask,
            boundary_up_full_mask,
            boundary_down_full_mask,
        )
        bce_local_sums = torch.stack(
            [(bce_value * mask.to(dtype=bce_value.dtype)).sum() for mask in bce_masks]
        )
        bce_local_counts = torch.stack([mask.to(dtype=bce_value.dtype).sum() for mask in bce_masks])
        (
            transition_grip_bce,
            transition_grip_up_bce,
            transition_grip_down_bce,
            boundary_grip_bce,
            boundary_grip_up_bce,
            boundary_grip_down_bce,
        ) = _distributed_sum_count_means(bce_local_sums, bce_local_counts).unbind()
        boundary_rate_sums = torch.stack(
            [
                (grip_prob * boundary_event_full_mask).sum(),
                (grip_tgt * boundary_event_full_mask).sum(),
            ]
        )
        boundary_rate_count = boundary_event_full_mask.to(dtype=grip_prob.dtype).sum()
        boundary_grip_prob_mean, boundary_grip_pos_rate = _distributed_sum_count_means(
            boundary_rate_sums,
            boundary_rate_count.expand(2),
        ).unbind()
        boundary_grip_rate_mse = (boundary_grip_prob_mean - boundary_grip_pos_rate).pow(2)
    else:
        if bool(transition_mask.any().detach().cpu()):
            transition_grip_bce = bce_value[transition_mask].mean()
        if bool(transition_up_mask.any().detach().cpu()):
            transition_grip_up_bce = bce_value[transition_up_mask].mean()
        if bool(transition_down_mask.any().detach().cpu()):
            transition_grip_down_bce = bce_value[transition_down_mask].mean()
        boundary_value = first_grip_bce * class_weight[:, 0]
        if bool(boundary_mask.any().detach().cpu()):
            boundary_grip_bce = boundary_value[boundary_mask].mean()
            boundary_grip_prob_mean = grip_prob[:, 0][boundary_mask].mean()
            boundary_grip_pos_rate = grip_tgt[:, 0][boundary_mask].mean()
            boundary_grip_rate_mse = (boundary_grip_prob_mean - boundary_grip_pos_rate).pow(2)
        if bool(boundary_up_mask.any().detach().cpu()):
            boundary_grip_up_bce = boundary_value[boundary_up_mask].mean()
        if bool(boundary_down_mask.any().detach().cpu()):
            boundary_grip_down_bce = boundary_value[boundary_down_mask].mean()
    tgt_pos = grip_tgt > 0.5
    tgt_neg = ~tgt_pos
    delta_ce = grip_prob.new_zeros(())
    delta_natural_ce = grip_prob.new_zeros(())
    delta_acc = grip_prob.new_zeros(())
    delta_hold_acc = grip_prob.new_zeros(())
    delta_up_acc = grip_prob.new_zeros(())
    delta_down_acc = grip_prob.new_zeros(())
    delta_boundary_up_acc = grip_prob.new_zeros(())
    delta_boundary_down_acc = grip_prob.new_zeros(())
    delta_state_bce = grip_prob.new_zeros(())
    delta_state_acc = grip_prob.new_zeros(())
    delta_state_pos_acc = grip_prob.new_zeros(())
    delta_state_neg_acc = grip_prob.new_zeros(())
    delta_state_transition_up_acc = grip_prob.new_zeros(())
    delta_state_transition_down_acc = grip_prob.new_zeros(())
    delta_transition_up_margin = grip_prob.new_zeros(())
    delta_transition_down_margin = grip_prob.new_zeros(())
    delta_boundary_up_margin = grip_prob.new_zeros(())
    delta_boundary_down_margin = grip_prob.new_zeros(())
    composed_acc = grip_prob.new_zeros(())
    composed_pos_acc = grip_prob.new_zeros(())
    composed_neg_acc = grip_prob.new_zeros(())
    composed_transition_up_acc = grip_prob.new_zeros(())
    composed_transition_down_acc = grip_prob.new_zeros(())
    composed_boundary_up_acc = grip_prob.new_zeros(())
    composed_boundary_down_acc = grip_prob.new_zeros(())
    delta_logits = out.get("policy_grip_delta_logits")
    if delta_logits is not None:
        delta_logits = delta_logits.float()[:, :horizon]
        if delta_logits.shape[-1] != 3:
            raise RuntimeError(f"policy_grip_delta_logits must have 3 classes hold/up/down, got {tuple(delta_logits.shape)}")
        delta_ce_raw = F.cross_entropy(
            delta_logits.reshape(-1, 3),
            event_target.reshape(-1),
            reduction="none",
        ).reshape_as(grip_tgt)
        delta_natural_ce = _distributed_sum_count_mean(
            delta_ce_raw.sum(), delta_ce_raw.new_tensor(delta_ce_raw.numel())
        )
        hold_w = float(train_cfg.get("direct_policy_grip_delta_hold_weight", 0.2) or 0.0)
        up_w = float(train_cfg.get("direct_policy_grip_delta_up_weight", 1.0) or 0.0)
        down_w = float(train_cfg.get("direct_policy_grip_delta_down_weight", 1.0) or 0.0)
        boundary_w = float(train_cfg.get("direct_policy_grip_delta_boundary_weight", 1.0) or 0.0)
        boundary_up_w = float(train_cfg.get("direct_policy_grip_delta_boundary_up_weight", 0.0) or 0.0)
        boundary_down_w = float(train_cfg.get("direct_policy_grip_delta_boundary_down_weight", 0.0) or 0.0)
        if strict_partition_contract:
            ce_partition_weights = {
                "boundary_hold": float(train_cfg.get("direct_policy_grip_delta_boundary_hold_weight", hold_w)),
                "boundary_up": float(
                    train_cfg.get("direct_policy_grip_delta_boundary_up_partition_weight", up_w + boundary_w + boundary_up_w)
                ),
                "boundary_down": float(
                    train_cfg.get("direct_policy_grip_delta_boundary_down_partition_weight", down_w + boundary_w + boundary_down_w)
                ),
                "inclip_hold": float(train_cfg.get("direct_policy_grip_delta_inclip_hold_weight", hold_w)),
                "inclip_up": float(train_cfg.get("direct_policy_grip_delta_inclip_up_weight", up_w)),
                "inclip_down": float(train_cfg.get("direct_policy_grip_delta_inclip_down_weight", down_w)),
            }
            delta_ce = _partition_balanced_distributed_mean(
                delta_ce_raw,
                event_partitions,
                ce_partition_weights,
            )
        elif bool(train_cfg.get("direct_policy_grip_delta_class_balance", False)):
            terms: list[torch.Tensor] = []
            weights_terms: list[float] = []
            hold_mask_for_loss = event_target == 0
            up_mask_for_loss = event_target == 1
            down_mask_for_loss = event_target == 2
            if hold_w > 0 and bool(hold_mask_for_loss.any().detach().cpu()):
                terms.append(delta_ce_raw[hold_mask_for_loss].mean() * hold_w)
                weights_terms.append(hold_w)
            if up_w > 0 and bool(up_mask_for_loss.any().detach().cpu()):
                terms.append(delta_ce_raw[up_mask_for_loss].mean() * up_w)
                weights_terms.append(up_w)
            if down_w > 0 and bool(down_mask_for_loss.any().detach().cpu()):
                terms.append(delta_ce_raw[down_mask_for_loss].mean() * down_w)
                weights_terms.append(down_w)
            if boundary_w > 0 and bool(boundary_mask.any().detach().cpu()):
                terms.append(delta_ce_raw[:, 0][boundary_mask].mean() * boundary_w)
                weights_terms.append(boundary_w)
            if boundary_up_w > 0 and bool(boundary_up_mask.any().detach().cpu()):
                terms.append(delta_ce_raw[:, 0][boundary_up_mask].mean() * boundary_up_w)
                weights_terms.append(boundary_up_w)
            if boundary_down_w > 0 and bool(boundary_down_mask.any().detach().cpu()):
                terms.append(delta_ce_raw[:, 0][boundary_down_mask].mean() * boundary_down_w)
                weights_terms.append(boundary_down_w)
            if terms:
                delta_ce = torch.stack(terms).sum() / max(1e-6, float(sum(weights_terms)))
        else:
            delta_weight = torch.full_like(grip_tgt, hold_w)
            delta_weight = torch.where(event_target == 1, torch.full_like(delta_weight, up_w), delta_weight)
            delta_weight = torch.where(event_target == 2, torch.full_like(delta_weight, down_w), delta_weight)
            if boundary_w > 0:
                delta_weight[:, 0] = delta_weight[:, 0] * (1.0 + boundary_w * boundary_transition)
            if boundary_up_w > 0:
                delta_weight[:, 0] = delta_weight[:, 0] * (1.0 + boundary_up_w * boundary_up_mask.float())
            if boundary_down_w > 0:
                delta_weight[:, 0] = delta_weight[:, 0] * (1.0 + boundary_down_w * boundary_down_mask.float())
            delta_ce = (delta_ce_raw * delta_weight).sum() / delta_weight.sum().clamp_min(1e-6)
        hold_logit = delta_logits[..., 0]
        up_logit = delta_logits[..., 1]
        down_logit = delta_logits[..., 2]
        up_gap = up_logit - torch.maximum(hold_logit, down_logit)
        down_gap = down_logit - torch.maximum(hold_logit, up_logit)
        base_delta_margin = float(train_cfg.get("direct_policy_grip_delta_logit_margin", 0.5) or 0.5)
        transition_up_delta_margin_value = float(
            train_cfg.get("direct_policy_grip_delta_transition_up_margin", base_delta_margin) or base_delta_margin
        )
        transition_down_delta_margin_value = float(
            train_cfg.get("direct_policy_grip_delta_transition_down_margin", base_delta_margin) or base_delta_margin
        )
        boundary_up_delta_margin_value = float(
            train_cfg.get("direct_policy_grip_delta_boundary_up_margin", transition_up_delta_margin_value)
            or transition_up_delta_margin_value
        )
        boundary_down_delta_margin_value = float(
            train_cfg.get("direct_policy_grip_delta_boundary_down_margin", transition_down_delta_margin_value)
            or transition_down_delta_margin_value
        )
        margin_raw = (
            F.relu(transition_up_delta_margin_value - up_gap),
            F.relu(transition_down_delta_margin_value - down_gap),
            F.relu(boundary_up_delta_margin_value - up_gap),
            F.relu(boundary_down_delta_margin_value - down_gap),
        )
        if strict_partition_contract:
            margin_masks = (inclip_up_mask, inclip_down_mask, boundary_up_full_mask, boundary_down_full_mask)
            margin_local_sums = torch.stack(
                [(value * mask.to(dtype=value.dtype)).sum() for value, mask in zip(margin_raw, margin_masks)]
            )
            margin_local_counts = torch.stack(
                [mask.to(dtype=up_gap.dtype).sum() for mask in margin_masks]
            )
            (
                delta_transition_up_margin,
                delta_transition_down_margin,
                delta_boundary_up_margin,
                delta_boundary_down_margin,
            ) = _distributed_sum_count_means(margin_local_sums, margin_local_counts).unbind()
        else:
            if bool(transition_up_mask.any().detach().cpu()):
                delta_transition_up_margin = margin_raw[0][transition_up_mask].mean()
            if bool(transition_down_mask.any().detach().cpu()):
                delta_transition_down_margin = margin_raw[1][transition_down_mask].mean()
            if bool(boundary_up_mask.any().detach().cpu()):
                delta_boundary_up_margin = margin_raw[2][:, 0][boundary_up_mask].mean()
            if bool(boundary_down_mask.any().detach().cpu()):
                delta_boundary_down_margin = margin_raw[3][:, 0][boundary_down_mask].mean()
        pred_event = delta_logits.argmax(dim=-1)
        event_match = pred_event == event_target
        delta_acc = event_match.float().mean()
        hold_mask = event_target == 0
        if bool(hold_mask.any().detach().cpu()):
            delta_hold_acc = event_match[hold_mask].float().mean()
        if bool(transition_up_mask.any().detach().cpu()):
            delta_up_acc = event_match[transition_up_mask].float().mean()
        if bool(transition_down_mask.any().detach().cpu()):
            delta_down_acc = event_match[transition_down_mask].float().mean()
        if bool(boundary_up_mask.any().detach().cpu()):
            delta_boundary_up_acc = event_match[:, 0][boundary_up_mask].float().mean()
        if bool(boundary_down_mask.any().detach().cpu()):
            delta_boundary_down_acc = event_match[:, 0][boundary_down_mask].float().mean()
        delta_prob = torch.softmax(delta_logits.float(), dim=-1).to(dtype=grip_tgt.dtype)
        soft_current = prev_tgt[:, 0].clamp(0.0, 1.0)
        soft_steps: list[torch.Tensor] = []
        for ti in range(horizon):
            hold_prob = delta_prob[:, ti, 0]
            up_prob = delta_prob[:, ti, 1]
            soft_current = (up_prob + hold_prob * soft_current).clamp(1e-4, 1.0 - 1e-4)
            soft_steps.append(soft_current)
        soft_composed = torch.stack(soft_steps, dim=1)
        state_bce_raw = -(
            grip_tgt * soft_composed.log()
            + (1.0 - grip_tgt) * (1.0 - soft_composed).log()
        )
        state_pos_w = float(train_cfg.get("direct_policy_grip_delta_state_pos_weight", 1.0) or 0.0)
        state_neg_w = float(train_cfg.get("direct_policy_grip_delta_state_neg_weight", 1.0) or 0.0)
        state_transition_w = float(train_cfg.get("direct_policy_grip_delta_state_transition_weight", 0.0) or 0.0)
        state_transition_up_w = float(train_cfg.get("direct_policy_grip_delta_state_transition_up_weight", 0.0) or 0.0)
        state_transition_down_w = float(train_cfg.get("direct_policy_grip_delta_state_transition_down_weight", 0.0) or 0.0)
        state_boundary_w = float(train_cfg.get("direct_policy_grip_delta_state_boundary_weight", 0.0) or 0.0)
        state_boundary_up_w = float(train_cfg.get("direct_policy_grip_delta_state_boundary_up_weight", 0.0) or 0.0)
        state_boundary_down_w = float(train_cfg.get("direct_policy_grip_delta_state_boundary_down_weight", 0.0) or 0.0)
        state_weight = torch.where(
            grip_tgt > 0.5,
            torch.full_like(grip_tgt, state_pos_w),
            torch.full_like(grip_tgt, state_neg_w),
        )
        if strict_partition_contract:
            state_base_weight = state_weight
            state_partition_extra = {
                "boundary_hold": 0.0,
                "boundary_up": state_transition_w + state_transition_up_w + state_boundary_w + state_boundary_up_w,
                "boundary_down": state_transition_w + state_transition_down_w + state_boundary_w + state_boundary_down_w,
                "inclip_hold": 0.0,
                "inclip_up": state_transition_w + state_transition_up_w,
                "inclip_down": state_transition_w + state_transition_down_w,
            }
            state_weight = torch.zeros_like(grip_tgt)
            for partition_name in _GRIP_PARTITION_NAMES:
                partition_value = state_base_weight + state_partition_extra[partition_name]
                explicit_key = f"direct_policy_grip_delta_state_{partition_name}_weight"
                if explicit_key in train_cfg:
                    partition_value = torch.full_like(partition_value, float(train_cfg[explicit_key]))
                state_weight = torch.where(event_partitions[partition_name], partition_value, state_weight)
            delta_state_bce = _partition_balanced_distributed_mean(
                state_bce_raw * state_weight,
                event_partitions,
                {name: 1.0 for name in _GRIP_PARTITION_NAMES},
            )
        else:
            if state_transition_w > 0:
                state_weight = state_weight * (1.0 + state_transition_w * grip_transition.float())
            if state_transition_up_w > 0:
                state_weight = state_weight * (1.0 + state_transition_up_w * transition_up_mask.float())
            if state_transition_down_w > 0:
                state_weight = state_weight * (1.0 + state_transition_down_w * transition_down_mask.float())
            if state_boundary_w > 0:
                state_weight[:, 0] = state_weight[:, 0] * (1.0 + state_boundary_w * boundary_transition.float())
            if state_boundary_up_w > 0:
                state_weight[:, 0] = state_weight[:, 0] * (1.0 + state_boundary_up_w * boundary_up_mask.float())
            if state_boundary_down_w > 0:
                state_weight[:, 0] = state_weight[:, 0] * (1.0 + state_boundary_down_w * boundary_down_mask.float())
            delta_state_bce = (state_bce_raw * state_weight).sum() / state_weight.sum().clamp_min(1e-6)
        soft_state_pred = soft_composed > 0.5
        soft_state_match = soft_state_pred == (grip_tgt > 0.5)
        delta_state_acc = soft_state_match.float().mean()
        if bool(tgt_pos.any().detach().cpu()):
            delta_state_pos_acc = soft_state_match[tgt_pos].float().mean()
        if bool(tgt_neg.any().detach().cpu()):
            delta_state_neg_acc = soft_state_match[tgt_neg].float().mean()
        if bool(transition_up_mask.any().detach().cpu()):
            delta_state_transition_up_acc = soft_state_match[transition_up_mask].float().mean()
        if bool(transition_down_mask.any().detach().cpu()):
            delta_state_transition_down_acc = soft_state_match[transition_down_mask].float().mean()
        current = prev_tgt[:, 0] > 0.5
        composed_steps: list[torch.Tensor] = []
        for ti in range(horizon):
            current = torch.where(
                pred_event[:, ti] == 1,
                torch.ones_like(current),
                torch.where(pred_event[:, ti] == 2, torch.zeros_like(current), current),
            )
            composed_steps.append(current)
        composed_pred = torch.stack(composed_steps, dim=1)
        composed_match = composed_pred == (grip_tgt > 0.5)
        composed_acc = composed_match.float().mean()
        if bool(tgt_pos.any().detach().cpu()):
            composed_pos_acc = composed_match[tgt_pos].float().mean()
        if bool(tgt_neg.any().detach().cpu()):
            composed_neg_acc = composed_match[tgt_neg].float().mean()
        if bool(transition_up_mask.any().detach().cpu()):
            composed_transition_up_acc = composed_match[transition_up_mask].float().mean()
        if bool(transition_down_mask.any().detach().cpu()):
            composed_transition_down_acc = composed_match[transition_down_mask].float().mean()
        if bool(boundary_up_mask.any().detach().cpu()):
            composed_boundary_up_acc = composed_match[:, 0][boundary_up_mask].float().mean()
        if bool(boundary_down_mask.any().detach().cpu()):
            composed_boundary_down_acc = composed_match[:, 0][boundary_down_mask].float().mean()

    if strict_partition_contract:
        pose_local_sums = torch.stack(
            [
                pose_err.sum(),
                first_pose_err.sum(),
                delta_err.sum(),
                worst_step_pose_err.sum(),
                cumulative_pose_err.sum(),
                endpoint_pose_err.sum(),
                direction_loss.sum(),
            ]
        )
        pose_local_counts = pose_local_sums.new_tensor(
            [
                pose_err.numel(),
                first_pose_err.numel(),
                delta_err.numel(),
                worst_step_pose_err.numel(),
                cumulative_pose_err.numel(),
                endpoint_pose_err.numel(),
                direction_loss.numel(),
            ]
        )
        (
            pose_loss,
            first_pose_loss,
            pose_delta_loss,
            worst_step_pose_loss,
            cumulative_pose_loss,
            endpoint_pose_loss,
            direction_pose_loss,
        ) = _distributed_sum_count_means(pose_local_sums, pose_local_counts).unbind()
        if grip_owner == "absolute":
            partition_weights = {name: 1.0 for name in _GRIP_PARTITION_NAMES}
            grip_loss = _partition_balanced_distributed_mean(
                grip_bce * class_weight,
                event_partitions,
                partition_weights,
            )
            first_partitions = {name: mask[:, 0] for name, mask in event_partitions.items()}
            first_grip_loss = _partition_balanced_distributed_mean(
                first_grip_bce * class_weight[:, 0],
                first_partitions,
                partition_weights,
            )
        else:
            non_event_mask = boundary_hold_mask | inclip_hold_mask
            absolute_weight = class_weight * non_event_mask.to(dtype=class_weight.dtype)
            grip_loss = _distributed_sum_count_mean(
                (grip_bce * absolute_weight).sum(),
                absolute_weight.sum(),
            )
            boundary_hold = boundary_hold_mask[:, 0].to(dtype=class_weight.dtype)
            boundary_hold_weight = class_weight[:, 0] * boundary_hold
            first_grip_loss = _distributed_sum_count_mean(
                (first_grip_bce * boundary_hold_weight).sum(),
                boundary_hold_weight.sum(),
            )
    else:
        pose_loss = pose_err.mean()
        first_pose_loss = first_pose_err.mean()
        pose_delta_loss = delta_err.mean()
        worst_step_pose_loss = worst_step_pose_err.mean()
        cumulative_pose_loss = cumulative_pose_err.mean()
        endpoint_pose_loss = endpoint_pose_err.mean()
        direction_pose_loss = direction_loss.mean()
        grip_loss = grip_err.mean()
        first_grip_loss = first_grip_err.mean()
    L_direct = (
        float(train_cfg.get("direct_policy_pose_weight", 1.0)) * pose_loss
        + float(train_cfg.get("direct_policy_first_pose_weight", 2.0)) * first_pose_loss
        + float(train_cfg.get("direct_policy_delta_weight", 0.2)) * pose_delta_loss
        + float(train_cfg.get("direct_policy_worst_step_pose_weight", 0.0)) * worst_step_pose_loss
        + float(train_cfg.get("direct_policy_cumulative_pose_weight", 0.0)) * cumulative_pose_loss
        + float(train_cfg.get("direct_policy_endpoint_pose_weight", 0.0)) * endpoint_pose_loss
        + float(train_cfg.get("direct_policy_direction_weight", 0.0)) * direction_pose_loss
        + float(train_cfg.get("direct_policy_grip_weight", 0.3)) * grip_loss
        + float(train_cfg.get("direct_policy_first_grip_weight", 0.5)) * first_grip_loss
        + float(train_cfg.get("direct_policy_grip_natural_bce_weight", 0.0) or 0.0)
        * natural_grip_bce
        + float(train_cfg.get("direct_policy_first_grip_natural_bce_weight", 0.0) or 0.0)
        * natural_first_grip_bce
        + _scheduled_train_scalar(train_cfg, "direct_policy_grip_transition_bce_weight", int(step)) * transition_grip_bce
        + _scheduled_train_scalar(train_cfg, "direct_policy_grip_transition_up_bce_weight", int(step)) * transition_grip_up_bce
        + _scheduled_train_scalar(train_cfg, "direct_policy_grip_transition_down_bce_weight", int(step)) * transition_grip_down_bce
        + _scheduled_train_scalar(train_cfg, "direct_policy_grip_transition_margin_weight", int(step)) * transition_grip_margin
        + _scheduled_train_scalar(train_cfg, "direct_policy_grip_transition_delta_bce_weight", int(step)) * transition_delta_bce
        + _scheduled_train_scalar(train_cfg, "direct_policy_grip_transition_delta_up_bce_weight", int(step)) * transition_delta_up_bce
        + _scheduled_train_scalar(train_cfg, "direct_policy_grip_transition_delta_down_bce_weight", int(step)) * transition_delta_down_bce
        + _scheduled_train_scalar(train_cfg, "direct_policy_grip_nontransition_delta_weight", int(step)) * nontransition_delta_abs
        + _scheduled_train_scalar(train_cfg, "direct_policy_grip_event_logit_margin_weight", int(step)) * transition_event_logit_margin
        + _scheduled_train_scalar(train_cfg, "direct_policy_grip_boundary_logit_margin_weight", int(step)) * boundary_grip_logit_margin
        + _scheduled_train_scalar(train_cfg, "direct_policy_grip_rate_mse_weight", int(step)) * grip_rate_mse
        + _scheduled_train_scalar(train_cfg, "direct_policy_grip_boundary_bce_weight", int(step)) * boundary_grip_bce
        + _scheduled_train_scalar(train_cfg, "direct_policy_grip_boundary_up_bce_weight", int(step)) * boundary_grip_up_bce
        + _scheduled_train_scalar(train_cfg, "direct_policy_grip_boundary_down_bce_weight", int(step)) * boundary_grip_down_bce
        + _scheduled_train_scalar(train_cfg, "direct_policy_grip_boundary_rate_mse_weight", int(step)) * boundary_grip_rate_mse
        + _scheduled_train_scalar(train_cfg, "direct_policy_grip_delta_ce_weight", int(step)) * delta_ce
        + _scheduled_train_scalar(train_cfg, "direct_policy_grip_delta_natural_ce_weight", int(step)) * delta_natural_ce
        + _scheduled_train_scalar(train_cfg, "direct_policy_grip_delta_state_bce_weight", int(step)) * delta_state_bce
        + _scheduled_train_scalar(train_cfg, "direct_policy_grip_delta_transition_up_margin_weight", int(step)) * delta_transition_up_margin
        + _scheduled_train_scalar(train_cfg, "direct_policy_grip_delta_transition_down_margin_weight", int(step)) * delta_transition_down_margin
        + _scheduled_train_scalar(train_cfg, "direct_policy_grip_delta_boundary_up_margin_weight", int(step)) * delta_boundary_up_margin
        + _scheduled_train_scalar(train_cfg, "direct_policy_grip_delta_boundary_down_margin_weight", int(step)) * delta_boundary_down_margin
    )
    grip_match = ((grip_prob > 0.5) == (grip_tgt > 0.5)).float()
    transition_acc = grip_match[transition_mask].mean() if bool(transition_mask.any().detach().cpu()) else grip_prob.new_zeros(())
    pred_pos = grip_prob > 0.5
    tgt_pos = grip_tgt > 0.5
    tgt_neg = ~tgt_pos
    tp = (pred_pos & tgt_pos).float().sum()
    pred_pos_count = pred_pos.float().sum()
    tgt_pos_count = tgt_pos.float().sum()
    pos_acc = grip_match[tgt_pos].mean() if bool(tgt_pos.any().detach().cpu()) else grip_prob.new_zeros(())
    neg_acc = grip_match[tgt_neg].mean() if bool(tgt_neg.any().detach().cpu()) else grip_prob.new_zeros(())
    transition_up_acc = (
        grip_match[transition_up_mask].mean()
        if bool(transition_up_mask.any().detach().cpu())
        else grip_prob.new_zeros(())
    )
    transition_down_acc = (
        grip_match[transition_down_mask].mean()
        if bool(transition_down_mask.any().detach().cpu())
        else grip_prob.new_zeros(())
    )
    boundary_acc = (
        grip_match[:, 0][boundary_mask].mean()
        if bool(boundary_mask.any().detach().cpu())
        else grip_prob.new_zeros(())
    )
    boundary_up_acc = (
        grip_match[:, 0][boundary_up_mask].mean()
        if bool(boundary_up_mask.any().detach().cpu())
        else grip_prob.new_zeros(())
    )
    boundary_down_acc = (
        grip_match[:, 0][boundary_down_mask].mean()
        if bool(boundary_down_mask.any().detach().cpu())
        else grip_prob.new_zeros(())
    )
    return {
        "L_direct_policy": L_direct,
        "direct_policy_pose_l1": (pose_pred - pose_tgt).abs().mean(),
        "direct_policy_first_pose_l1": (pose_pred[:, 0] - pose_tgt[:, 0]).abs().mean(),
        "direct_policy_worst_step_pose_loss": worst_step_pose_loss,
        "direct_policy_cumulative_pose_loss": cumulative_pose_loss,
        "direct_policy_endpoint_pose_loss": endpoint_pose_loss,
        "direct_policy_direction_loss": direction_pose_loss,
        "direct_policy_grip_acc": grip_match.mean(),
        "direct_policy_grip_transition_acc": transition_acc,
        "direct_policy_grip_transition_up_acc": transition_up_acc,
        "direct_policy_grip_transition_down_acc": transition_down_acc,
        "direct_policy_grip_transition_up_count": transition_up_mask.float().sum(),
        "direct_policy_grip_transition_down_count": transition_down_mask.float().sum(),
        "direct_policy_grip_transition_rate": grip_transition.mean(),
        "direct_policy_grip_pos_rate": grip_tgt.mean(),
        "direct_policy_grip_prob_mean": grip_prob.mean(),
        "direct_policy_grip_pred_pos_rate": pred_pos.float().mean(),
        "direct_policy_grip_pos_acc": pos_acc,
        "direct_policy_grip_neg_acc": neg_acc,
        "direct_policy_grip_precision": tp / pred_pos_count.clamp_min(1.0),
        "direct_policy_grip_recall": tp / tgt_pos_count.clamp_min(1.0),
        "direct_policy_grip_bce": grip_bce_raw.mean(),
        "direct_policy_grip_natural_bce": natural_grip_bce,
        "direct_policy_first_grip_natural_bce": natural_first_grip_bce,
        "direct_policy_grip_transition_bce": transition_grip_bce,
        "direct_policy_grip_transition_up_bce": transition_grip_up_bce,
        "direct_policy_grip_transition_down_bce": transition_grip_down_bce,
        "direct_policy_grip_transition_margin": transition_grip_margin,
        "direct_policy_grip_transition_delta_bce": transition_delta_bce,
        "direct_policy_grip_transition_delta_up_bce": transition_delta_up_bce,
        "direct_policy_grip_transition_delta_down_bce": transition_delta_down_bce,
        "direct_policy_grip_nontransition_delta_abs": nontransition_delta_abs,
        "direct_policy_grip_event_logit_margin": transition_event_logit_margin,
        "direct_policy_grip_boundary_logit_margin": boundary_grip_logit_margin,
        "direct_policy_grip_rate_mse": grip_rate_mse,
        "direct_policy_grip_boundary_bce": boundary_grip_bce,
        "direct_policy_grip_boundary_up_bce": boundary_grip_up_bce,
        "direct_policy_grip_boundary_down_bce": boundary_grip_down_bce,
        "direct_policy_grip_boundary_rate_mse": boundary_grip_rate_mse,
        "direct_policy_grip_boundary_pos_rate": boundary_grip_pos_rate,
        "direct_policy_grip_boundary_prob_mean": boundary_grip_prob_mean,
        "direct_policy_grip_boundary_acc": boundary_acc,
        "direct_policy_grip_boundary_up_acc": boundary_up_acc,
        "direct_policy_grip_boundary_down_acc": boundary_down_acc,
        "direct_policy_grip_boundary_up_count": boundary_up_mask.float().sum(),
        "direct_policy_grip_boundary_down_count": boundary_down_mask.float().sum(),
        "direct_policy_grip_inclip_transition_up_count": transition_up_mask[:, 1:].float().sum() if horizon > 1 else grip_prob.new_zeros(()),
        "direct_policy_grip_inclip_transition_down_count": transition_down_mask[:, 1:].float().sum() if horizon > 1 else grip_prob.new_zeros(()),
        "direct_policy_grip_partition_boundary_hold_count": boundary_hold_mask.float().sum(),
        "direct_policy_grip_partition_boundary_up_count": boundary_up_full_mask.float().sum(),
        "direct_policy_grip_partition_boundary_down_count": boundary_down_full_mask.float().sum(),
        "direct_policy_grip_partition_inclip_hold_count": inclip_hold_mask.float().sum(),
        "direct_policy_grip_partition_inclip_up_count": inclip_up_mask.float().sum(),
        "direct_policy_grip_partition_inclip_down_count": inclip_down_mask.float().sum(),
        "direct_policy_grip_count": grip_tgt.new_tensor(float(grip_tgt.numel())),
        "direct_policy_grip_boundary_rate": boundary_transition.mean(),
        "direct_policy_grip_delta_ce": delta_ce,
        "direct_policy_grip_delta_natural_ce": delta_natural_ce,
        "direct_policy_grip_delta_acc": delta_acc,
        "direct_policy_grip_delta_hold_acc": delta_hold_acc,
        "direct_policy_grip_delta_up_acc": delta_up_acc,
        "direct_policy_grip_delta_down_acc": delta_down_acc,
        "direct_policy_grip_delta_boundary_up_acc": delta_boundary_up_acc,
        "direct_policy_grip_delta_boundary_down_acc": delta_boundary_down_acc,
        "direct_policy_grip_delta_state_bce": delta_state_bce,
        "direct_policy_grip_delta_state_acc": delta_state_acc,
        "direct_policy_grip_delta_state_pos_acc": delta_state_pos_acc,
        "direct_policy_grip_delta_state_neg_acc": delta_state_neg_acc,
        "direct_policy_grip_delta_state_transition_up_acc": delta_state_transition_up_acc,
        "direct_policy_grip_delta_state_transition_down_acc": delta_state_transition_down_acc,
        "direct_policy_grip_delta_transition_up_margin": delta_transition_up_margin,
        "direct_policy_grip_delta_transition_down_margin": delta_transition_down_margin,
        "direct_policy_grip_delta_boundary_up_margin": delta_boundary_up_margin,
        "direct_policy_grip_delta_boundary_down_margin": delta_boundary_down_margin,
        "direct_policy_grip_composed_acc": composed_acc,
        "direct_policy_grip_composed_pos_acc": composed_pos_acc,
        "direct_policy_grip_composed_neg_acc": composed_neg_acc,
        "direct_policy_grip_composed_transition_up_acc": composed_transition_up_acc,
        "direct_policy_grip_composed_transition_down_acc": composed_transition_down_acc,
        "direct_policy_grip_composed_boundary_up_acc": composed_boundary_up_acc,
        "direct_policy_grip_composed_boundary_down_acc": composed_boundary_down_acc,
    }


def build_hunyuan_latent_adapter(cfg: dict, device: torch.device) -> HunyuanLatentAdapter:
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    adapter_cfg = HunyuanLatentAdapterConfig(
        token_dim=model_cfg["state"]["D"],
        token_grid=int(model_cfg["state"]["P"] ** 0.5),
        hidden=int(train_cfg.get("hunyuan_adapter_hidden", 192)),
        latent_channels=int(train_cfg.get("hunyuan_latent_channels", 16)),
        latent_time=int(train_cfg.get("hunyuan_latent_time", 3)),
        latent_hw=int(train_cfg.get("hunyuan_latent_hw", 32)),
        action_dim=int(train_cfg.get("hunyuan_action_dim", 7)),
        task_dim=int(train_cfg.get("hunyuan_task_dim", model_cfg["state"].get("cond_dim", 2048))),
        n_blocks=int(train_cfg.get("hunyuan_adapter_blocks", 4)),
        use_motion=bool(train_cfg.get("hunyuan_use_motion", True)),
        use_rough_rgb=bool(train_cfg.get("hunyuan_use_rough_rgb", True)),
        use_context=bool(train_cfg.get("hunyuan_use_context", True)),
        use_action=bool(train_cfg.get("hunyuan_use_action", True)),
        use_task=bool(train_cfg.get("hunyuan_use_task", True)),
        use_point=bool(train_cfg.get("hunyuan_use_point", False)),
        use_pose=bool(train_cfg.get("hunyuan_use_pose", False)),
        point_dim=int(train_cfg.get("hunyuan_point_dim", 3)),
        pose_dim=int(train_cfg.get("hunyuan_pose_dim", 9)),
    )
    adapter = HunyuanLatentAdapter(adapter_cfg).to(device)
    if bool(train_cfg.get("hunyuan_zero_init_output", False)):
        adapter.zero_init_output()
    return adapter


def load_hunyuan_vae(train_cfg: dict, device: torch.device):
    model_base = Path(train_cfg.get("hunyuan_model_base", "/data/Minko/models/hunyuan_video"))
    repo = Path(train_cfg.get("hunyuan_repo", "/data/Minko/external/HunyuanVideo"))
    os.environ.setdefault("MODEL_BASE", str(model_base))
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from hyvideo.vae import load_vae  # type: ignore

    vae, _, _, _ = load_vae(
        "884-16c-hy",
        train_cfg.get("hunyuan_vae_precision", "fp16"),
        device=device,
    )
    vae.requires_grad_(False)
    vae.eval()
    return vae


def target_video_from_batch(context_rgb: torch.Tensor, rgb_tgt_p: torch.Tensor) -> torch.Tensor:
    video = torch.cat([context_rgb[:, None], rgb_tgt_p], dim=1)
    return video.permute(0, 2, 1, 3, 4).contiguous()


@torch.no_grad()
def encode_hunyuan_latents(vae, video_bcthw: torch.Tensor) -> torch.Tensor:
    x = video_bcthw.mul(2.0).sub(1.0)
    posterior = vae.encode(x.to(dtype=vae.dtype)).latent_dist
    latents = posterior.mode()
    return latents * float(vae.config.scaling_factor)


def _hunyuan_zero_losses(zero: torch.Tensor, prefix: str = "") -> dict[str, torch.Tensor]:
    head = f"{prefix}_" if prefix else ""
    return {
        f"L_{head}hunyuan_latent": zero,
        f"{head}hunyuan_latent_mse": zero,
        f"{head}hunyuan_latent_l1": zero,
        f"{head}hunyuan_latent_temporal_mse": zero,
        f"{head}hunyuan_latent_motion_l1": zero,
    }


def _compute_hunyuan_controls_loss(
    adapter: HunyuanLatentAdapter,
    vae,
    *,
    tokens: torch.Tensor,
    depth: torch.Tensor,
    target_latents: torch.Tensor,
    context_rgb: torch.Tensor,
    action_cond: torch.Tensor,
    task_emb: torch.Tensor,
    train_cfg: dict,
    rough_rgb: torch.Tensor | None = None,
    motion_hint: torch.Tensor | None = None,
    point: torch.Tensor | None = None,
    pose: torch.Tensor | None = None,
    prefix: str = "",
) -> dict[str, torch.Tensor]:
    if bool(train_cfg.get("hunyuan_detach_world", False)):
        tokens = tokens.detach()
        depth = depth.detach()
        rough_rgb = rough_rgb.detach() if rough_rgb is not None else None
        point = point.detach() if point is not None else None
        pose = pose.detach() if pose is not None else None

    delta_latents = adapter(
        tokens,
        depth,
        context_rgb=context_rgb,
        motion_hint=motion_hint,
        rough_rgb=rough_rgb,
        action_cond=action_cond,
        task_emb=task_emb,
        point=point,
        pose=pose,
        target_latents=target_latents,
    )
    pred_latents = delta_latents
    if bool(train_cfg.get("hunyuan_residual_from_rough", False)) and rough_rgb is not None:
        rough_video = target_video_from_batch(context_rgb, rough_rgb.float())
        rough_latents = encode_hunyuan_latents(vae, rough_video)
        pred_latents = rough_latents.to(dtype=delta_latents.dtype) + float(
            train_cfg.get("hunyuan_residual_scale", 1.0)
        ) * delta_latents

    zero = tokens.new_zeros(())
    latent_mse = F.mse_loss(pred_latents.float(), target_latents.float())
    latent_l1 = F.l1_loss(pred_latents.float(), target_latents.float())
    if pred_latents.shape[2] > 1 and target_latents.shape[2] > 1:
        pred_dt = pred_latents.float()[:, :, 1:] - pred_latents.float()[:, :, :-1]
        target_dt = target_latents.float()[:, :, 1:] - target_latents.float()[:, :, :-1]
        temporal_mse = F.mse_loss(pred_dt, target_dt)
        motion_l1 = F.l1_loss(pred_dt.abs(), target_dt.abs())
    else:
        temporal_mse = zero
        motion_l1 = zero
    loss = (
        float(train_cfg.get("hunyuan_latent_mse_weight", 1.0)) * latent_mse
        + float(train_cfg.get("hunyuan_latent_l1_weight", 0.05)) * latent_l1
        + float(train_cfg.get("hunyuan_latent_temporal_weight", 0.0)) * temporal_mse
        + float(train_cfg.get("hunyuan_latent_motion_weight", 0.0)) * motion_l1
    )
    head = f"{prefix}_" if prefix else ""
    return {
        f"L_{head}hunyuan_latent": loss,
        f"{head}hunyuan_latent_mse": latent_mse.detach(),
        f"{head}hunyuan_latent_l1": latent_l1.detach(),
        f"{head}hunyuan_latent_temporal_mse": temporal_mse.detach(),
        f"{head}hunyuan_latent_motion_l1": motion_l1.detach(),
    }


def compute_hunyuan_latent_loss(
    adapter: HunyuanLatentAdapter,
    vae,
    out: dict,
    tgt: dict,
    context_rgb: torch.Tensor | None,
    action_cond: torch.Tensor,
    task_emb: torch.Tensor,
    train_cfg: dict,
) -> dict[str, torch.Tensor]:
    zero = out["pred_tokens"].new_zeros(())
    losses = _hunyuan_zero_losses(zero)
    losses.update(_hunyuan_zero_losses(zero, prefix="prior"))
    if context_rgb is None or "rgb_tgt_p" not in tgt:
        return losses

    target_video = target_video_from_batch(context_rgb, tgt["rgb_tgt_p"])
    target_latents = encode_hunyuan_latents(vae, target_video)

    rough_rgb = out.get("rgb") if bool(train_cfg.get("hunyuan_use_rough_rgb", True)) else None
    losses.update(_compute_hunyuan_controls_loss(
        adapter,
        vae,
        tokens=out["pred_tokens"],
        depth=out["depth"],
        target_latents=target_latents,
        context_rgb=context_rgb,
        action_cond=action_cond,
        task_emb=task_emb,
        train_cfg=train_cfg,
        rough_rgb=rough_rgb,
        motion_hint=out.get("motion_hint"),
        point=out.get("point"),
        pose=out.get("pose_geom"),
    ))

    prior_enabled = bool(train_cfg.get("enable_prior_hunyuan_latent_loss", False))
    prior_weight = float(train_cfg.get("prior_hunyuan_latent_weight", 0.0))
    if prior_enabled and prior_weight > 0 and "prior_hunyuan_tokens" in out and "prior_hunyuan_depth" in out:
        prior_rough = out.get("prior_rgb") if bool(train_cfg.get("prior_hunyuan_use_rough_rgb", train_cfg.get("hunyuan_use_rough_rgb", True))) else None
        losses.update(_compute_hunyuan_controls_loss(
            adapter,
            vae,
            tokens=out["prior_hunyuan_tokens"],
            depth=out["prior_hunyuan_depth"],
            target_latents=target_latents,
            context_rgb=context_rgb,
            action_cond=action_cond,
            task_emb=task_emb,
            train_cfg=train_cfg,
            rough_rgb=prior_rough,
            motion_hint=None,
            point=out.get("prior_hunyuan_point"),
            pose=out.get("prior_hunyuan_pose"),
            prefix="prior",
        ))
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--disable_pixel_until", type=int, default=0,
                    help="train first N epochs without L_rgb (stage 1)")
    ap.add_argument(
        "--no_pixel",
        action="store_true",
        help="never activate RGB/video renderer or RGB/LPIPS losses; train latent/action/evaluator heads only",
    )
    ap.add_argument("--reset_optim", action="store_true",
                    help="on --resume, load model weights only; recreate fresh optimizer/scheduler/step")
    ap.add_argument("--print_every", type=int, default=0,
                    help="print stdout step log every N steps (0=off)")
    ap.add_argument("--out_root", type=Path, default=None,
                    help="override out.root without mutating the frozen YAML")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="override train.max_steps for a bounded systems proof")
    ap.add_argument(
        "--stop_after_step",
        type=int,
        default=None,
        help=(
            "stop this invocation at an exact optimizer step without mutating "
            "the configured max_steps or LR schedule (for canary save/resume)"
        ),
    )
    ap.add_argument("--strict_resume", action="store_true",
                    help="strict state_dict load on resume (default: allow mismatches)")
    args = ap.parse_args()
    cfg = load_train_config(args.cfg)
    if args.out_root is not None:
        cfg.setdefault("out", {})["root"] = str(args.out_root)
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max_steps must be positive")
        cfg.setdefault("train", {})["max_steps"] = int(args.max_steps)
    if args.stop_after_step is not None and args.stop_after_step <= 0:
        raise ValueError("--stop_after_step must be positive")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {args.resume}")
    startup_train_cfg = cfg.get("train", {}) or {}
    if args.resume is not None and bool(startup_train_cfg.get("forbid_resume", False)):
        if not bool(startup_train_cfg.get("allow_same_run_exact_resume", True)):
            raise ValueError(
                "forbid_resume=true permits only an explicitly enabled same-run exact resume"
            )
    validate_empty_checkpoint_dir_preflight(
        cfg,
        resume_checkpoint=args.resume,
    )
    stage_transition_mode = validate_stage_transition_preflight(cfg, args)
    future_value_stage = validate_future_value_stage_preflight(cfg, args)
    action_pretraining_stage = validate_action_pretraining_preflight(cfg)
    rank, world, local = setup_ddp()
    device = torch.device(f"cuda:{local}")
    eager_transport_audit = eager_initialize_distributed_transport(
        rank=rank,
        world=world,
        device=device,
    )
    train_cfg = cfg.get("train", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    base_seed = int(train_cfg.get("seed", data_cfg.get("seed", 0)) or 0)
    seed_process_rng(base_seed)
    # Model construction uses the same seed on every rank.  DDP then
    # broadcasts rank-0 state before rank-aware stochastic training begins.
    train_cfg["resolved_seed"] = base_seed
    resolved_resume_compat_sha256 = config_sha256(_resume_compat_config(cfg))
    lineage_payload = (
        str(train_cfg.get("run_lineage") or "")
        or f"{(cfg.get('out') or {}).get('root','')}|"
        f"{(cfg.get('contract') or {}).get('schema','')}|"
        f"{resolved_resume_compat_sha256}"
    )
    resolved_run_lineage = hashlib.sha256(lineage_payload.encode("utf-8")).hexdigest()
    train_cfg["resolved_resume_compat_sha256"] = resolved_resume_compat_sha256
    train_cfg["resolved_run_lineage"] = resolved_run_lineage
    resolved_config_sha256 = config_sha256(cfg)
    train_cfg["resolved_config_sha256"] = resolved_config_sha256
    if rank == 0:
        print(
            "[rank0] distributed_transport_preflight="
            + json.dumps(eager_transport_audit, sort_keys=True),
            flush=True,
        )
        if action_pretraining_stage:
            print(
                "[rank0] action_pretraining_preflight=passed "
                f"direct_head={startup_train_cfg.get('direct_policy_head', 'policy')} "
                f"flow_use_as_policy={bool((cfg.get('model') or {}).get('policy_flow_use_as_policy'))}",
                flush=True,
            )
        print(
            f"[rank0] seed model_init={base_seed} runtime_stride=100003 "
            f"config_sha256={resolved_config_sha256} "
            f"resume_compat_sha256={resolved_resume_compat_sha256} "
            f"run_lineage={resolved_run_lineage}",
            flush=True,
        )
    if (
        bool(data_cfg.get("node_sharded_window_cache", False))
        and world > int(os.environ.get("LOCAL_WORLD_SIZE", "1") or 1)
        and not bool(train_cfg.get("equalize_node_steps", False))
    ):
        raise RuntimeError(
            "node_sharded_window_cache with multi-node DDP requires train.equalize_node_steps=true; "
            "otherwise nodes can enter train/val collectives at different times."
        )

    overfit_ids = cfg.get("overfit_clip_ids") if args.overfit else None
    tr_ds, val_ds = build_datasets(cfg, overfit_ids=overfit_ids)
    train_cfg = cfg["train"]
    mixed_source_training = isinstance(tr_ds, MixedSourceWindowDataset)
    if mixed_source_training:
        if not isinstance(val_ds, MixedSourceWindowDataset):
            raise RuntimeError("mixed train dataset requires a mixed validation dataset")
        if tuple(tr_ds.source_names) != tuple(val_ds.source_names):
            raise RuntimeError(
                "train/val source ordering differs: "
                f"train={tr_ds.source_names} val={val_ds.source_names}"
            )
        action_source_names = tuple(tr_ds.source_names)
    else:
        action_source_names = ("single",)
    if "audited_action_sources" not in train_cfg:
        audited_from_data: list[str] = []
        raw_oxe_sources = data_cfg.get("oxe_sources") or ()
        if isinstance(raw_oxe_sources, dict):
            source_items = raw_oxe_sources.items()
        else:
            source_items = ((str(index), value) for index, value in enumerate(raw_oxe_sources))
        for source_key, source_cfg in source_items:
            if isinstance(source_cfg, dict) and source_cfg.get("action_audit_gate"):
                audited_from_data.append(
                    str(source_cfg.get("source_name", f"oxe_{source_key}"))
                )
        train_cfg["audited_action_sources"] = audited_from_data
    action_source_policy = resolve_action_source_policy(
        train_cfg,
        action_source_names,
        strict=mixed_source_training,
    )
    configured_grip_contract = train_cfg.get("action_grip_contract")
    if configured_grip_contract is None:
        configured_grip_contract = (
            ((cfg.get("contract") or {}).get("canonical_action") or {}).get(
                "gripper_semantics"
            )
        )
        if configured_grip_contract == "signed_close_positive":
            configured_grip_contract = "signed_close"
    action_grip_contract = normalize_action_grip_contract(configured_grip_contract)
    train_cfg["action_grip_contract"] = action_grip_contract
    if mixed_source_training and any(
        policy == "factual_action" for policy in action_source_policy.values()
    ) and action_grip_contract != "signed_close":
        raise ValueError(
            "explicit mixed-source factual action training requires "
            "train.action_grip_contract=signed_close"
        )
    if rank == 0:
        print(
            f"[rank0] action_source_policy={action_source_policy} "
            f"action_grip_contract={action_grip_contract}",
            flush=True,
        )
    bs = train_cfg["batch_size_per_gpu"]; nw = train_cfg["num_workers"]
    gradient_accumulation_steps = int(
        train_cfg.get(
            "gradient_accumulation_steps",
            train_cfg.get("grad_accum_steps", 1),
        )
    )
    if gradient_accumulation_steps <= 0:
        raise ValueError("train.gradient_accumulation_steps must be positive")
    if gradient_accumulation_steps > 1 and not bool(
        train_cfg.get("abort_on_nonfinite", False)
    ):
        raise ValueError(
            "gradient accumulation requires abort_on_nonfinite=true so a partial "
            "accumulation group can never be silently committed"
        )
    strict_partition_contract = _validate_grip_partition_contract_config(train_cfg)
    weighted_sampler_cfg = train_cfg.get("weighted_sampler") or cfg.get("weighted_sampler")
    if isinstance(weighted_sampler_cfg, dict):
        weighted_sampler_cfg = dict(weighted_sampler_cfg)
        weighted_sampler_cfg["direct_policy_grip_partition_contract"] = strict_partition_contract
    sample_weights = build_dataset_sample_weights(tr_ds, weighted_sampler_cfg)
    if rank == 0 and sample_weights is not None:
        print(
            f"[rank0] weighted_sampler active n={int(sample_weights.numel())} "
            f"nonzero={int((sample_weights > 0).sum().item())} "
            f"min={float(sample_weights.min().item()):.6g} "
            f"max={float(sample_weights.max().item()):.6g} "
            f"mean={float(sample_weights.mean().item()):.6g}",
            flush=True,
        )
    mixed_sampler_cfg = train_cfg.get("mixed_batch_sampler") or {}
    rank_local_cache_audit = {"enabled": False}
    global_mixed_source_audit = {"enabled": False}
    sampler_replicas = int(world)
    sampler_rank = int(rank)
    sampler_scope = "global"
    if isinstance(tr_ds, MixedSourceWindowDataset):
        if sample_weights is not None:
            raise ValueError(
                "mixed source training uses mixed_batch_sampler, not weighted_sampler"
            )
        if not bool(mixed_sampler_cfg.get("enabled", False)):
            raise ValueError(
                "dataset_type=v7_mixed requires train.mixed_batch_sampler.enabled=true"
            )
        if not bool(
            mixed_sampler_cfg.get(
                "shuffle_source_cycle",
                mixed_sampler_cfg.get("shuffle_cycle", True),
            )
        ):
            raise ValueError(
                "mixed action training requires shuffle_source_cycle=true to "
                "avoid exact-cycle/every-N auxiliary aliasing"
            )
        source_cycle_counts = {
            str(name): int(count)
            for name, count in (
                mixed_sampler_cfg.get(
                    "source_cycle_counts_exact",
                    mixed_sampler_cfg.get("source_cycle_counts") or {},
                )
            ).items()
        }
        train_num_batches = mixed_sampler_cfg.get("num_batches_per_epoch")
        val_num_batches = mixed_sampler_cfg.get("val_num_batches_per_epoch")
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1") or 1)
        rank_local_cache_audit = audit_distributed_rank_local_source_lengths(
            tr_ds,
            val_ds,
            mixed_sampler_cfg,
            batch_size=bs,
            world=world,
            rank=rank,
            device=device,
            local_world_size=local_world_size,
            contract_profile=str((cfg.get("contract") or {}).get("profile", "")),
        )
        if rank_local_cache_audit.get("enabled", False):
            if not bool(data_cfg.get("node_sharded_window_cache", False)):
                raise ValueError(
                    "rank-local mixed sampler requires data.node_sharded_window_cache=true"
                )
            if not bool(train_cfg.get("equalize_node_steps", False)):
                raise ValueError(
                    "rank-local mixed sampler requires train.equalize_node_steps=true"
                )
            sampler_replicas = local_world_size
            sampler_rank = local
        else:
            global_mixed_source_audit = (
                audit_distributed_global_mixed_source_contract(
                    tr_ds,
                    val_ds,
                    world=world,
                    rank=rank,
                    device=device,
                )
            )
            sampler_replicas = world
            sampler_rank = rank
        tr_s = SourceHomogeneousDistributedBatchSampler(
            tr_ds,
            source_cycle_counts,
            batch_size=bs,
            num_replicas=sampler_replicas,
            rank=sampler_rank,
            num_batches=(
                int(train_num_batches) if train_num_batches is not None else None
            ),
            seed=int(mixed_sampler_cfg.get("seed", data_cfg.get("seed", 0))),
            batches_per_source_group=gradient_accumulation_steps,
        )
        val_s = SourceHomogeneousDistributedBatchSampler(
            val_ds,
            source_cycle_counts,
            batch_size=bs,
            num_replicas=sampler_replicas,
            rank=sampler_rank,
            num_batches=(
                int(val_num_batches) if val_num_batches is not None else None
            ),
            seed=int(mixed_sampler_cfg.get("seed", data_cfg.get("seed", 0)))
            + 100_000,
            allow_small_source_replacement=bool(
                train_cfg.get("val_allow_small_source_replacement", False)
            ),
        )
        expected_oxe_fraction = mixed_sampler_cfg.get("expected_oxe_fraction")
        observed_oxe_fraction = sum(
            fraction
            for name, fraction in tr_s.source_fractions.items()
            if name.startswith("oxe_")
        )
        if expected_oxe_fraction is not None and not math.isclose(
            observed_oxe_fraction,
            float(expected_oxe_fraction),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "mixed source cycle violates expected OXE fraction: "
                f"observed={observed_oxe_fraction} "
                f"expected={float(expected_oxe_fraction)}"
            )
        loader_kwargs = {
            "num_workers": nw,
            "pin_memory": True,
            "persistent_workers": bool(nw > 0),
        }
        if nw > 0:
            loader_kwargs["prefetch_factor"] = int(
                train_cfg.get("prefetch_factor", 3)
            )
        tr_loader = DataLoader(tr_ds, batch_sampler=tr_s, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_sampler=val_s, **loader_kwargs)
        rank_local_cache_audit = finalize_rank_local_loader_length_audit(
            rank_local_cache_audit,
            train_loader_length=len(tr_loader),
            val_loader_length=len(val_loader),
            world=world,
            device=device,
        )
        if rank == 0:
            print(
                f"[rank0] mixed_batch_sampler fractions={tr_s.source_fractions} "
                f"oxe_fraction={observed_oxe_fraction:.6f} "
                f"micro_batches_per_epoch={len(tr_s)} micro_global_batch={bs * world} "
                f"gradient_accumulation_steps={gradient_accumulation_steps} "
                f"effective_global_batch={bs * world * gradient_accumulation_steps}",
                flush=True,
            )
            if rank_local_cache_audit.get("enabled", False):
                print(
                    "[rank0] rank_local_cache_shards="
                    + json.dumps(rank_local_cache_audit, sort_keys=True),
                    flush=True,
                )
            elif global_mixed_source_audit.get("enabled", False):
                print(
                    "[rank0] global_mixed_source_identity="
                    + json.dumps(global_mixed_source_audit, sort_keys=True),
                    flush=True,
                )
    elif world > 1:
        sampler_world, sampler_rank, sampler_scope = _sampler_scope(cfg, world, rank, local)
        if rank == 0 and sampler_scope != "global":
            print(
                f"[rank0] sampler_scope={sampler_scope} "
                f"sampler_world={sampler_world} global_world={world}",
                flush=True,
            )
        equalize_node_steps = bool(train_cfg.get("equalize_node_steps", False)) and sampler_scope == "local-node"
        equalized_num_samples = None
        if equalize_node_steps:
            # NCCL can reject int64 MIN on some stacks; use fp32 for this small length sync.
            local_len = torch.tensor([float(len(tr_ds))], device=device, dtype=torch.float32)
            min_len = local_len.clone()
            dist.all_reduce(min_len, op=dist.ReduceOp.MIN)
            equalized_num_samples = _equalized_num_samples_per_rank(
                [int(min_len.item())],
                sampler_world,
            )
            if rank == 0:
                print(
                    f"[rank0] equalize_node_steps=True local_len={int(local_len.item())} "
                    f"min_global_len={int(min_len.item())} "
                    f"num_samples_per_local_rank={equalized_num_samples}",
                    flush=True,
                )
        if sample_weights is not None or equalized_num_samples is not None:
            sampler_num_samples = None
            if isinstance(weighted_sampler_cfg, dict) and weighted_sampler_cfg.get("num_samples_per_rank") is not None:
                sampler_num_samples = int(weighted_sampler_cfg["num_samples_per_rank"])
            if equalized_num_samples is not None and sampler_num_samples is None:
                sampler_num_samples = int(equalized_num_samples)
            if sample_weights is None:
                sample_weights = torch.ones(len(tr_ds), dtype=torch.double)
            sampler_replacement = (
                bool((weighted_sampler_cfg or {}).get("replacement", False))
                if isinstance(weighted_sampler_cfg, dict)
                else False
            )
            sampler_seed = (
                int((weighted_sampler_cfg or {}).get("seed", cfg["data"].get("seed", 0)))
                if isinstance(weighted_sampler_cfg, dict)
                else int(cfg["data"].get("seed", 0))
            )
            tr_s = build_grip_event_balanced_sampler(
                tr_ds,
                weighted_sampler_cfg,
                sample_weights,
                num_replicas=sampler_world,
                rank=sampler_rank,
                replacement=sampler_replacement,
                num_samples=sampler_num_samples,
                seed=sampler_seed,
                batch_size_per_rank=bs,
            )
            if tr_s is None:
                tr_s = WeightedDistributedSampler(
                    sample_weights,
                    num_replicas=sampler_world,
                    rank=sampler_rank,
                    replacement=sampler_replacement,
                    num_samples=sampler_num_samples,
                    seed=sampler_seed,
                )
        else:
            tr_s = DistributedSampler(tr_ds, num_replicas=sampler_world, rank=sampler_rank, shuffle=True, drop_last=True)
        tr_loader = DataLoader(tr_ds, batch_size=bs, sampler=tr_s, num_workers=nw,
                                pin_memory=True, drop_last=True)
        if equalize_node_steps:
            local_val_len = torch.tensor([len(val_ds)], device=device, dtype=torch.long)
            min_val_len = local_val_len.clone()
            dist.all_reduce(min_val_len, op=dist.ReduceOp.MIN)
            val_num_samples = _equalized_num_samples_per_rank([int(min_val_len.item())], sampler_world)
            if rank == 0:
                print(
                    f"[rank0] equalized val local_len={int(local_val_len.item())} "
                    f"min_global_len={int(min_val_len.item())} "
                    f"num_samples_per_local_rank={val_num_samples}",
                    flush=True,
                )
            val_s = WeightedDistributedSampler(
                torch.ones(len(val_ds), dtype=torch.double),
                num_replicas=sampler_world,
                rank=sampler_rank,
                replacement=False,
                num_samples=val_num_samples,
                seed=int(cfg["data"].get("seed", 0)) + 100000,
            )
        else:
            val_s = DistributedSampler(val_ds, num_replicas=sampler_world, rank=sampler_rank, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=bs, sampler=val_s, num_workers=nw, pin_memory=True)
    else:
        if sample_weights is not None:
            num_samples = int(weighted_sampler_cfg.get("num_samples", len(sample_weights))) if isinstance(weighted_sampler_cfg, dict) else len(sample_weights)
            sampler_replacement = bool(weighted_sampler_cfg.get("replacement", True)) if isinstance(weighted_sampler_cfg, dict) else True
            sampler_seed = int(weighted_sampler_cfg.get("seed", cfg["data"].get("seed", 0))) if isinstance(weighted_sampler_cfg, dict) else int(cfg["data"].get("seed", 0))
            tr_s = build_grip_event_balanced_sampler(
                tr_ds,
                weighted_sampler_cfg,
                sample_weights,
                num_replicas=1,
                rank=0,
                replacement=sampler_replacement,
                num_samples=num_samples,
                seed=sampler_seed,
                batch_size_per_rank=bs,
            )
            if tr_s is None:
                tr_s = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=sampler_replacement)
            tr_loader = DataLoader(tr_ds, batch_size=bs, sampler=tr_s, num_workers=nw, pin_memory=True, drop_last=True)
        else:
            tr_s = None
            tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    weights = LossWeights(**cfg["loss"])
    factual_weights, representation_weights, native_action_weights = (
        resolve_action_training_weights(
            weights,
            train_cfg,
            strict=mixed_source_training,
        )
    )
    direct_policy_only = bool(train_cfg.get("direct_policy_only", False))
    future_value_only = bool(train_cfg.get("future_value_only", False))
    hunyuan_training_enabled = (
        bool(train_cfg.get("enable_hunyuan_latent_loss", False))
        and float(train_cfg.get("hunyuan_latent_weight", 0.0)) > 0
        and not args.no_pixel
        and not direct_policy_only
    )
    pixel_training_enabled = (
        bool(cfg["model"].get("enable_pixel", True))
        and (
            bool(train_cfg.get("enable_pixel_loss", True))
            or (hunyuan_training_enabled and bool(train_cfg.get("hunyuan_use_rough_rgb", True)))
        )
        and not args.no_pixel
        and not direct_policy_only
    )
    if int(args.disable_pixel_until) > 0 and world > 1 and not bool(train_cfg.get("manual_grad_sync", direct_policy_only)) and not bool(train_cfg.get("find_unused_parameters", False)):
        raise RuntimeError(
            "--disable_pixel_until with DDP find_unused_parameters=false can leave pixel/context-pixel "
            "parameters unused during early epochs. Set find_unused_parameters=true, use --no_pixel, "
            "manual_grad_sync=true, or do not disable pixel epochs."
        )
    model = build_model(cfg).to(device)
    future_value_init_sha256 = module_state_sha256(
        getattr(model, "future_value_head", None)
    )
    stage_transition_audit: dict[str, object] = {
        "schema": "wm3d_v7_stage_transition_audit_v1",
        "model_init_seed": base_seed,
        "future_value_init_sha256": future_value_init_sha256,
    }
    if not pixel_training_enabled:
        for module_name in ("pixel", "context_pixel"):
            module = getattr(model, module_name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = False
    apply_trainable_filter(model, cfg["train"], rank)
    if future_value_stage and future_value_only:
        bad_trainable = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith("future_value_head.")
        ]
        value_trainable = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith("future_value_head.")
        ]
        if bad_trainable or not value_trainable:
            raise RuntimeError(
                "future value optimizer isolation failed: "
                f"bad_trainable={bad_trainable[:20]} value_trainable={len(value_trainable)}"
            )
    hunyuan_adapter = None
    hunyuan_vae = None
    if hunyuan_training_enabled:
        hunyuan_adapter = build_hunyuan_latent_adapter(cfg, device)
        hunyuan_vae = load_hunyuan_vae(train_cfg, device)
        if rank == 0:
            h_params = sum(p.numel() for p in hunyuan_adapter.parameters() if p.requires_grad)
            print(f"[rank0] HunyuanLatentAdapter: {h_params/1e6:.2f}M", flush=True)

    if rank == 0:
        n_p = model.num_trainable_params()
        print(f"[rank0] JointWorldModel: {n_p/1e6:.1f}M; train_windows={len(tr_ds)} val_windows={len(val_ds)}")
    manual_grad_sync = bool(train_cfg.get("manual_grad_sync", direct_policy_only))
    use_ddp_wrapper = world > 1 and not manual_grad_sync
    if use_ddp_wrapper:
        model = DDP(
            model,
            device_ids=[local],
            find_unused_parameters=train_cfg.get("find_unused_parameters", False),
            # A counterfactual objective can execute a second forward before
            # the shared backward. DDP's per-forward buffer broadcast mutates
            # action_proj.{mean,std} in-place and invalidates the factual action
            # graph, even though those buffers are immutable and loaded
            # identically on every rank. Keep the historical default unless a
            # multi-forward configuration explicitly disables it.
            broadcast_buffers=bool(train_cfg.get("ddp_broadcast_buffers", True)),
        )
        if hunyuan_adapter is not None:
            hunyuan_adapter = DDP(
                hunyuan_adapter,
                device_ids=[local],
                find_unused_parameters=train_cfg.get("find_unused_parameters", False),
            )

    runtime_seed = base_seed + rank * 100_003
    seed_process_rng(runtime_seed)
    if rank == 0:
        print(f"[rank0] runtime RNG seed={runtime_seed}", flush=True)

    lpips_fn = None
    if pixel_training_enabled and weights.rgb_lpips > 0:
        import lpips

        lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False
    if rank == 0:
        print(
            f"[rank0] pixel_training_enabled={pixel_training_enabled} "
            f"hunyuan_training_enabled={hunyuan_training_enabled} "
            f"lpips_active={lpips_fn is not None} disable_pixel_until={args.disable_pixel_until} "
            f"direct_policy_only={direct_policy_only} manual_grad_sync={manual_grad_sync}",
            flush=True,
        )

    optimizer_settings = resolve_optimizer_settings(cfg)
    train_cfg_for_opt = dict(train_cfg)
    train_cfg_for_opt["lr"] = optimizer_settings.peak_lr
    train_cfg_for_opt["weight_decay"] = optimizer_settings.weight_decay
    optimizer_param_groups = build_optimizer_param_groups(model, hunyuan_adapter, train_cfg_for_opt, rank)
    trainable_params = [
        param
        for group in optimizer_param_groups
        for param in group["params"]
    ]
    opt = torch.optim.AdamW(
        optimizer_param_groups,
        betas=optimizer_settings.betas,
    )
    grad_clip_value = float(optimizer_settings.grad_clip if optimizer_settings.grad_clip is not None else train_cfg["grad_clip"])
    max_steps = int(train_cfg.get("max_steps", 0) or 0)
    stop_after_step = int(args.stop_after_step or 0)
    if stop_after_step > 0 and max_steps > 0 and stop_after_step > max_steps:
        raise ValueError(
            f"--stop_after_step={stop_after_step} exceeds configured max_steps={max_steps}"
        )
    invocation_stop_step = stop_after_step or max_steps
    if len(tr_loader) % gradient_accumulation_steps != 0:
        raise ValueError(
            f"micro-batches per epoch ({len(tr_loader)}) must be divisible by "
            f"gradient_accumulation_steps ({gradient_accumulation_steps})"
        )
    optimizer_steps_per_epoch = len(tr_loader) // gradient_accumulation_steps
    total_steps = max(1, optimizer_steps_per_epoch * train_cfg["epochs"])
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    sched = build_lr_scheduler(opt, cfg, total_steps)
    if rank == 0:
        lr_schedule_type = str((cfg.get("lr_schedule") or train_cfg.get("lr_schedule") or {}).get("type", train_cfg.get("lr_schedule_type", "cosine"))).lower()
        print(
            f"[rank0] optimizer={optimizer_settings.type} betas={optimizer_settings.betas} "
            f"peak_lr={optimizer_settings.peak_lr:.3g} min_lr={optimizer_settings.min_lr:.3g} "
            f"weight_decay={optimizer_settings.weight_decay:.3g} grad_clip={grad_clip_value:.3g} "
            f"lr_schedule={lr_schedule_type} total_optimizer_steps={total_steps} "
            f"optimizer_steps_per_epoch={optimizer_steps_per_epoch} "
            f"gradient_accumulation_steps={gradient_accumulation_steps} "
            f"invocation_stop_step={invocation_stop_step}",
            flush=True,
        )
    out_root = Path(cfg["out"]["root"])
    ckpt_dir = out_root / cfg["out"]["ckpt_dir"]
    contract_profile = str((cfg.get("contract") or {}).get("profile", ""))
    telemetry_enabled = bool(train_cfg.get("canary_telemetry_enabled", False)) or contract_profile.startswith("canary")
    telemetry_path = out_root / str(
        train_cfg.get("canary_telemetry_jsonl", "canary_telemetry.jsonl")
    )
    if rank == 0:
        (out_root / cfg["out"]["tb_dir"]).mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if telemetry_enabled:
            telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            if args.resume is None and telemetry_path.exists():
                raise RuntimeError(
                    f"fresh canary telemetry already exists: {telemetry_path}"
                )
        tb = SummaryWriter(out_root / cfg["out"]["tb_dir"])

    start_epoch = 0
    resume_micro_batches_in_epoch = 0
    restored_source_cycle_position = 0
    step = 0
    best_val = float("inf")
    if args.resume is not None and args.resume.exists():
        resume_load_on_cpu = bool(train_cfg.get("resume_load_on_cpu", False))
        resume_mmap = bool(train_cfg.get("resume_mmap", resume_load_on_cpu))
        resume_load_kwargs = {
            "map_location": "cpu" if resume_load_on_cpu else device,
            "weights_only": False,
        }
        if resume_mmap:
            if not resume_load_on_cpu:
                raise ValueError("train.resume_mmap requires train.resume_load_on_cpu=true")
            resume_load_kwargs["mmap"] = True
        if rank == 0:
            print(
                f"[rank0] loading resume checkpoint from {args.resume} "
                f"map_location={'cpu' if resume_load_on_cpu else device} "
                f"mmap={int(resume_mmap)}",
                flush=True,
            )
        sd = torch.load(args.resume, **resume_load_kwargs)
        exact_same_run_resume = mixed_source_training and bool(
            train_cfg.get("require_exact_same_run_resume", True)
        )
        if exact_same_run_resume:
            if args.reset_optim:
                raise RuntimeError(
                    "exact same-run resume forbids --reset_optim; optimizer, "
                    "scheduler and sampler state must continue together"
                )
            missing_training_state = [
                key for key in ("opt", "sched", "sampler_state") if key not in sd
            ]
            if missing_training_state:
                raise RuntimeError(
                    "exact same-run resume checkpoint lacks training state: "
                    + ", ".join(missing_training_state)
                )
            checkpoint_lineage = sd.get("run_lineage")
            checkpoint_compat = sd.get("resume_compat_sha256")
            if checkpoint_lineage != resolved_run_lineage:
                raise RuntimeError(
                    "resume run lineage mismatch: "
                    f"checkpoint={checkpoint_lineage} current={resolved_run_lineage}"
                )
            if checkpoint_compat != resolved_resume_compat_sha256:
                raise RuntimeError(
                    "resume-compatible config digest mismatch: "
                    f"checkpoint={checkpoint_compat} current={resolved_resume_compat_sha256}"
                )
            checkpoint_root = str(((sd.get("cfg") or {}).get("out") or {}).get("root", ""))
            if checkpoint_root != str(cfg["out"]["root"]):
                raise RuntimeError(
                    "same-run resume output root mismatch: "
                    f"checkpoint={checkpoint_root!r} current={cfg['out']['root']!r}"
                )
            rng_contract = sd.get("rng_contract_rank0") or {}
            if (
                rng_contract.get("schema") != "wm3d_v7_step_addressed_rng_v1"
                or int(rng_contract.get("base_seed", -1)) != base_seed
            ):
                raise RuntimeError("resume RNG contract is missing or incompatible")
        target = model.module if isinstance(model, DDP) else model
        load_res = load_compatible_state_dict(
            target,
            sd["model"],
            strict=(args.strict_resume or exact_same_run_resume),
        )
        if exact_same_run_resume:
            validate_exact_stage_resume_load(load_res)
        if future_value_stage:
            allowed_missing_prefixes = (
                ("future_value_head.", "action_policy.")
                if stage_transition_mode == "stage0_to_value_policy"
                else ("future_value_head.",)
            )
            validate_future_value_resume_load(
                load_res,
                allowed_missing_prefixes=allowed_missing_prefixes,
            )
            stage_transition_audit.update(
                {
                    "resume_path": str(Path(args.resume).resolve()),
                    "shared_tensor_load": "exact",
                    "allowed_missing_prefixes": list(allowed_missing_prefixes),
                    "missing_keys": list(
                        getattr(load_res, "missing_keys", []) or []
                    ),
                    "unexpected_keys": list(
                        getattr(load_res, "unexpected_keys", []) or []
                    ),
                    "skipped_keys": list(
                        getattr(load_res, "skipped_keys", []) or []
                    ),
                    "expanded_keys": list(
                        getattr(load_res, "expanded_keys", []) or []
                    ),
                    "optimizer_scheduler_reset": bool(args.reset_optim),
                }
            )
            if rank == 0:
                print(
                    "[rank0] stage_transition_audit="
                    + json.dumps(stage_transition_audit, sort_keys=True),
                    flush=True,
                )
        if stage_transition_mode == "stage0_to_policy":
            validate_action_policy_resume_load(load_res)
            if rank == 0:
                print(
                    "[rank0] stage0_to_policy_load="
                    + json.dumps(
                        {
                            "resume_path": str(Path(args.resume).resolve()),
                            "allowed_missing_prefixes": ["action_policy."],
                            "missing_keys": list(
                                getattr(load_res, "missing_keys", []) or []
                            ),
                            "unexpected_keys": list(
                                getattr(load_res, "unexpected_keys", []) or []
                            ),
                            "skipped_keys": list(
                                getattr(load_res, "skipped_keys", []) or []
                            ),
                            "expanded_keys": list(
                                getattr(load_res, "expanded_keys", []) or []
                            ),
                            "optimizer_scheduler_reset": bool(args.reset_optim),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if stage_transition_mode == "exact":
            validate_exact_stage_resume_load(load_res)
        if stage_transition_mode == "v6_native" or bool(
            train_cfg.get("strict_v6_native_warm_start", False)
        ):
            validate_stage0_native_warm_start_load(load_res)
        if hunyuan_adapter is not None and "hunyuan_adapter" in sd:
            adapter_target = hunyuan_adapter.module if isinstance(hunyuan_adapter, DDP) else hunyuan_adapter
            adapter_load_res = load_compatible_state_dict(
                adapter_target,
                sd["hunyuan_adapter"],
                strict=args.strict_resume,
            )
        else:
            adapter_load_res = None
        if args.reset_optim:
            if rank == 0:
                miss = len(getattr(load_res, "missing_keys", []) or [])
                un = len(getattr(load_res, "unexpected_keys", []) or [])
                skip = len(getattr(load_res, "skipped_keys", []) or [])
                expanded = len(getattr(load_res, "expanded_keys", []) or [])
                print(f"[rank0] resumed weights only from {args.resume} (optim RESET) — "
                      f"missing={miss} unexpected={un} skipped={skip} expanded={expanded}")
                if adapter_load_res is not None:
                    amiss = len(getattr(adapter_load_res, "missing_keys", []) or [])
                    aun = len(getattr(adapter_load_res, "unexpected_keys", []) or [])
                    askip = len(getattr(adapter_load_res, "skipped_keys", []) or [])
                    print(f"[rank0] resumed hunyuan adapter from checkpoint — "
                          f"missing={amiss} unexpected={aun} skipped={askip}")
        else:
            opt.load_state_dict(sd["opt"]); sched.load_state_dict(sd["sched"])
            sampler_state = sd.get("sampler_state")
            if sampler_state is not None:
                if sampler_state.get("schema") != "wm3d_v7_exact_source_cycle_v1":
                    raise RuntimeError(
                        "resume sampler schema is missing or incompatible: "
                        f"{sampler_state.get('schema')!r}"
                    )
                if int(sampler_state.get("gradient_accumulation_steps", -1)) != gradient_accumulation_steps:
                    raise RuntimeError(
                        "resume gradient accumulation differs from checkpoint sampler state"
                    )
                if isinstance(tr_s, SourceHomogeneousDistributedBatchSampler):
                    expected_cycle_steps = sum(tr_s.source_cycle_counts.values())
                    if int(sampler_state.get("source_cycle_optimizer_steps", -1)) != expected_cycle_steps:
                        raise RuntimeError("resume source cycle length differs from checkpoint")
                    expected_seed = int(
                        mixed_sampler_cfg.get("seed", data_cfg.get("seed", 0))
                    )
                    if int(sampler_state.get("sampler_seed", -1)) != expected_seed:
                        raise RuntimeError("resume source sampler seed differs from checkpoint")
                    if rank_local_cache_audit.get("enabled", False):
                        if int(
                            sampler_state.get("sampler_num_replicas", -1)
                        ) != int(sampler_replicas):
                            raise RuntimeError(
                                "resume source sampler replica scope differs from checkpoint"
                            )
                        if sampler_state.get("sampler_rank_scope") != "local-node":
                            raise RuntimeError(
                                "resume source sampler rank scope differs from checkpoint"
                            )
                start_epoch = int(sampler_state["epoch"])
                resume_micro_batches_in_epoch = int(
                    sampler_state["micro_batches_consumed_in_epoch"]
                )
                if resume_micro_batches_in_epoch < 0:
                    raise RuntimeError("checkpoint sampler micro-batch position is negative")
                if resume_micro_batches_in_epoch >= len(tr_loader):
                    start_epoch += resume_micro_batches_in_epoch // len(tr_loader)
                    resume_micro_batches_in_epoch %= len(tr_loader)
                if isinstance(tr_s, SourceHomogeneousDistributedBatchSampler):
                    restored_position = (
                        (resume_micro_batches_in_epoch // gradient_accumulation_steps)
                        % sum(tr_s.source_cycle_counts.values())
                    )
                    if restored_position != int(
                        sampler_state.get("source_cycle_position", -1)
                    ):
                        raise RuntimeError(
                            "resume source-cycle position does not match checkpoint"
                        )
                    restored_source_cycle_position = int(restored_position)
            else:
                if mixed_source_training and bool(
                    train_cfg.get("require_exact_sampler_resume", True)
                ):
                    raise RuntimeError(
                        "mixed-source resume checkpoint lacks sampler_state; "
                        "exact source-cycle restoration is impossible"
                    )
                start_epoch = sd["epoch"] + 1
            step = sd["step"]
            best_val = sd.get("best_val", best_val)
            if rank == 0:
                print(
                    f"[rank0] resumed from {args.resume} at epoch {start_epoch} "
                    f"micro_batch={resume_micro_batches_in_epoch} step={step}",
                    flush=True,
                )
            if exact_same_run_resume and telemetry_enabled:
                if not isinstance(
                    tr_s, SourceHomogeneousDistributedBatchSampler
                ):
                    raise RuntimeError(
                        "exact mixed-source resume requires the fast-forward sampler"
                    )
                tr_s.set_epoch(int(start_epoch))
                tr_s.set_start_batch(int(resume_micro_batches_in_epoch))
                preview_batch = next(iter(tr_s), None)
                if preview_batch is None:
                    raise RuntimeError(
                        "exact resume fast-forward produced no next training batch"
                    )
                preview_source_ids = {
                    tr_ds._locate(index)[0] for index in preview_batch
                }
                if len(preview_source_ids) != 1:
                    raise RuntimeError(
                        "exact resume preview batch is not source-homogeneous"
                    )
                next_batch_source = tr_ds.source_names[
                    next(iter(preview_source_ids))
                ]
                if rank == 0:
                    startup_event = build_exact_resume_startup_event(
                        checkpoint_path=args.resume,
                        checkpoint_payload=sd,
                        resolved_config_sha256=resolved_config_sha256,
                        resolved_resume_compat_sha256=resolved_resume_compat_sha256,
                        resolved_run_lineage=resolved_run_lineage,
                        model_load_result=load_res,
                        model_strict=bool(args.strict_resume or exact_same_run_resume),
                        optimizer=opt,
                        scheduler=sched,
                        restored_global_step=int(step),
                        sampler_state=sampler_state,
                        runtime_sampler_epoch=int(start_epoch),
                        runtime_micro_batches_in_epoch=int(
                            resume_micro_batches_in_epoch
                        ),
                        runtime_source_cycle_position=int(
                            restored_source_cycle_position
                        ),
                        runtime_sampler_num_replicas=int(sampler_replicas),
                        runtime_sampler_rank_scope=(
                            "local-node"
                            if rank_local_cache_audit.get("enabled", False)
                            else "global"
                        ),
                        next_batch_source=next_batch_source,
                        rng_contract=rng_contract,
                        base_seed=base_seed,
                    )
                    append_exact_resume_startup_event_once(
                        telemetry_path, startup_event
                    )
                    print(
                        "[rank0] exact resume startup evidence appended "
                        f"schema={startup_event['schema']} "
                        f"checkpoint_step={startup_event['checkpoint_step']} "
                        f"checkpoint_sha256={startup_event['checkpoint_sha256']}",
                        flush=True,
                    )
                if world > 1:
                    dist.barrier()
        del sd
        if resume_load_on_cpu:
            gc.collect()
            torch.cuda.empty_cache()
    load_action_stats_if_available(model, cfg, rank, device)

    k = cfg["data"]["k"]
    measure_step_time = bool(train_cfg.get("measure_step_time", False))
    step_duration_window: list[float] = []
    mixed_source_batch_counts = (
        [0 for _name in tr_ds.source_names]
        if isinstance(tr_ds, MixedSourceWindowDataset)
        else None
    )
    action_aux_trigger_counts = {
        name: {
            "optimizer_groups": 0,
            "no_teacher": 0,
            "no_teacher_future": 0,
            "core_cf": 0,
            "rgb_cf": 0,
            "state_action_condition_projection_grad_params": 0,
            "native_state_dynamics_grad_params": 0,
            "no_teacher_action_head_grad_params": 0,
            "no_teacher_future_pred_grad_tensors": 0,
            "candidate_future_grad_tensors": 0,
            "future_value_head_grad_params": 0,
        }
        for name in action_source_names
    }
    counterfactual_path_gradient_counts = {
        name: {
            "wrong_action_input_grad": 0,
            "native_pred_tokens_grad": 0,
        }
        for name in action_source_names
    }
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        if invocation_stop_step > 0 and step >= invocation_stop_step:
            break
        epoch_resume_skip = (
            resume_micro_batches_in_epoch if epoch == start_epoch else 0
        )
        if tr_s is not None and hasattr(tr_s, "set_epoch"):
            tr_s.set_epoch(epoch)
        if isinstance(tr_s, SourceHomogeneousDistributedBatchSampler):
            tr_s.set_start_batch(epoch_resume_skip)
        elif epoch_resume_skip:
            raise RuntimeError(
                "resume micro-batch fast-forward requires "
                "SourceHomogeneousDistributedBatchSampler"
            )
        if hasattr(tr_ds, "set_epoch"):
            tr_ds.set_epoch(epoch)
        do_pixel = pixel_training_enabled and epoch >= args.disable_pixel_until
        model.train()
        accumulation_index = 0
        accumulation_source_id: int | None = None
        micro_batches_consumed_in_epoch = epoch_resume_skip
        opt.zero_grad(set_to_none=True)
        for micro_batch_index, batch in enumerate(tr_loader):
            absolute_micro_batch_index = epoch_resume_skip + micro_batch_index
            micro_batches_consumed_in_epoch = absolute_micro_batch_index + 1
            if invocation_stop_step > 0 and step >= invocation_stop_step:
                break
            current_source_id: int | None = None
            source_name: str | None = None
            representation_only_batch = False
            if mixed_source_batch_counts is not None:
                source_ids = torch.as_tensor(batch["source_id"]).reshape(-1)
                unique_source_ids = torch.unique(source_ids)
                if unique_source_ids.numel() != 1:
                    raise RuntimeError(
                        "mixed source batch is not homogeneous: "
                        f"source_ids={source_ids.tolist()}"
                    )
                source_id = int(unique_source_ids.item())
                current_source_id = source_id
                source_name = tr_ds.source_names[source_id]
                representation_only_batch = (
                    action_source_policy[source_name] == "representation_only"
                )
                if accumulation_index == 0:
                    accumulation_source_id = source_id
                elif source_id != accumulation_source_id:
                    raise RuntimeError(
                        "mixed source changed inside one gradient-accumulation "
                        f"group: first={accumulation_source_id} current={source_id}"
                    )
                mixed_source_batch_counts[source_id] += 1
            else:
                source_name = "single"
                representation_only_batch = (
                    action_source_policy[source_name] == "representation_only"
                )
            action_aux_batch = action_aux_source_allowed(
                train_cfg,
                source_name,
                representation_only=representation_only_batch,
            )
            native_action_no_teacher_weight = (
                scheduled_aux_weight(
                    train_cfg,
                    "native_action_no_teacher_weight",
                    step,
                )
                if action_aux_batch
                else 0.0
            )
            native_future_no_teacher_weight = (
                scheduled_aux_weight(
                    train_cfg,
                    "native_future_no_teacher_weight",
                    step,
                )
                if action_aux_batch
                else 0.0
            )
            context_pixel_action_rank_weight = (
                scheduled_aux_weight(
                    train_cfg,
                    "context_pixel_action_rank_weight",
                    step,
                    schedule_prefix="context_pixel_action_rank",
                )
                if action_aux_batch
                else 0.0
            )
            context_pixel_action_separation_weight = (
                scheduled_aux_weight(
                    train_cfg,
                    "context_pixel_action_separation_weight",
                    step,
                    schedule_prefix="context_pixel_action_rank",
                )
                if action_aux_batch
                else 0.0
            )
            core_action_rank_weight = (
                scheduled_native_core_cf_weight(train_cfg, "rank", step)
                if action_aux_batch
                else 0.0
            )
            core_action_separation_weight = (
                scheduled_native_core_cf_weight(train_cfg, "separation", step)
                if action_aux_batch
                else 0.0
            )
            if accumulation_index == 0:
                action_aux_trigger_counts[source_name]["optimizer_groups"] += 1
                if native_action_no_teacher_weight > 0.0:
                    action_aux_trigger_counts[source_name]["no_teacher"] += 1
                if native_future_no_teacher_weight > 0.0:
                    action_aux_trigger_counts[source_name]["no_teacher_future"] += 1
                if max(core_action_rank_weight, core_action_separation_weight) > 0.0:
                    action_aux_trigger_counts[source_name]["core_cf"] += 1
                if max(
                    context_pixel_action_rank_weight,
                    context_pixel_action_separation_weight,
                ) > 0.0:
                    action_aux_trigger_counts[source_name]["rgb_cf"] += 1
            sync_gradients = (
                accumulation_index + 1 == gradient_accumulation_steps
            )
            if isinstance(model, DDP):
                model.require_backward_grad_sync = sync_gradients
            if isinstance(hunyuan_adapter, DDP):
                hunyuan_adapter.require_backward_grad_sync = sync_gradients
            if measure_step_time and accumulation_index == 0:
                torch.cuda.synchronize(device)
                step_started = time.perf_counter()
            # Rank-aware, step-addressable RNG makes stochastic model paths
            # exactly reproducible after a mid-epoch sampler resume.
            stochastic_seed = (
                base_seed
                + rank * 100_003
                + step * gradient_accumulation_steps
                + accumulation_index
                + 10_000_019
            )
            seed_process_rng(stochastic_seed)
            s, c, action_cond, context_rgb, tgt = batch_to_device(
                batch,
                device,
                k,
                direct_policy_only=direct_policy_only,
                action_grip_contract=action_grip_contract,
                source_name=source_name,
                require_factual_action_contract=(
                    mixed_source_training and not representation_only_batch
                ),
            )
            if representation_only_batch and action_cond is not None:
                action_cond = torch.zeros_like(action_cond)
            main_teacher_action_weight = _main_teacher_action_weight_for_batch(
                direct_policy_only=direct_policy_only,
                representation_only_batch=representation_only_batch,
                factual_weights=factual_weights,
                representation_weights=representation_weights,
            )
            decode_codec_targets(model, tgt)
            loss_tgt = targets_with_close01_grip(tgt, action_grip_contract)
            multiview_kwargs = multiview_kwargs_from_targets(tgt)
            no_teacher_future_gradient_evidence: dict[str, bool] | None = None
            candidate_future_gradient_evidence: dict[str, bool] | None = None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if direct_policy_only:
                    target_model = model.module if isinstance(model, DDP) else model
                    policy_kwargs = action_policy_kwargs_from_targets(tgt)
                    policy_kwargs.update(
                        make_policy_flow_training_kwargs(loss_tgt, train_cfg)
                    )
                    out = _direct_policy_only_forward(
                        target_model,
                        s,
                        c,
                        action_cond=action_cond,
                        context_rgb=context_rgb,
                        policy_kwargs=policy_kwargs,
                        train_cfg=train_cfg,
                        multiview_kwargs=multiview_kwargs,
                    )
                    losses = {"L_total": out["policy_pose_norm"].new_zeros(())}
                    zero = out["policy_pose_norm"].new_zeros(())
                    rank_losses = {
                        "L_evaluator_pairwise": zero,
                        "evaluator_pairwise_acc": zero,
                        "evaluator_pairwise_gap": zero,
                        "evaluator_real_progress": zero,
                        "evaluator_variant_progress": zero,
                    }
                    candidate_losses = {
                        "L_evaluator_candidate_pairwise": zero,
                        "L_evaluator_candidate_ce": zero,
                        "evaluator_candidate_pairwise_acc": zero,
                        "evaluator_candidate_pairwise_gap": zero,
                        "evaluator_candidate_selected_pose_l1": zero,
                        "evaluator_candidate_anchor_pose_l1": zero,
                        "evaluator_candidate_oracle_pose_l1": zero,
                        "evaluator_candidate_oracle_match": zero,
                    }
                else:
                    s_cond, c_cond, action_cond_model, context_rgb_cond, dropout_losses = apply_condition_dropout(
                        s, c, action_cond, context_rgb, train_cfg, training=True
                    )
                    factual_contract = train_cfg.get("factual_action_conditioning") or {}
                    if not representation_only_batch:
                        validate_factual_action_condition(
                            action_cond_model,
                            source_name=str(source_name),
                            require_finite=bool(
                                factual_contract.get("require_finite_action", True)
                            ),
                        )
                    prior_clean_tokens = prior_clean_tokens_from_targets(tgt)
                    policy_kwargs = action_policy_kwargs_from_targets(tgt)
                    policy_kwargs.update(
                        make_policy_flow_training_kwargs(loss_tgt, train_cfg)
                    )
                    out = _forward_joint_model(
                        model,
                        s_cond,
                        c_cond,
                        action_cond=action_cond_model,
                        context_rgb=context_rgb_cond,
                        prior_clean_tokens=prior_clean_tokens,
                        pixel=do_pixel,
                        bridging=False,
                        policy_kwargs=policy_kwargs,
                        multiview_kwargs=multiview_kwargs,
                        candidate_actions=tgt.get("branch_actions"),
                        candidate_include_geometry=bool(train_cfg.get("true_branch_include_geometry", False)),
                        native_action_no_teacher=max(
                            native_action_no_teacher_weight,
                            native_future_no_teacher_weight,
                        )
                        > 0.0,
                    )
                    if future_value_only:
                        losses = {"L_total": out["candidate_success_logit"].new_zeros(())}
                    else:
                        batch_weights = (
                            representation_weights
                            if representation_only_batch
                            else factual_weights
                        )
                        losses = compute_losses(
                            out,
                            loss_tgt,
                            batch_weights,
                            lpips_fn if do_pixel else None,
                        )
                        losses["L_action_raw"] = losses["L_action"]
                        losses["L_action_weighted"] = (
                            float(batch_weights.action) * losses["L_action"]
                        )
                        losses["teacher_action_diagnostic"] = losses["L_action"]
                        losses["L_action"] = losses["L_action"].new_zeros(())
                        if native_action_no_teacher_weight > 0.0:
                            native_action_losses = (
                                compute_native_no_teacher_action_loss(
                                    out,
                                    loss_tgt,
                                    native_action_weights,
                                    train_cfg=train_cfg,
                                )
                            )
                            losses["L_total"] = losses["L_total"] + (
                                native_action_no_teacher_weight
                                * native_action_losses["L_action"]
                            )
                            # Physical metrics above are reconstructed with
                            # this batch's explicit source statistics.
                            losses.update(
                                {
                                    f"native_no_teacher_{name}": value.detach()
                                    for name, value in native_action_losses.items()
                                }
                            )
                            losses["native_no_teacher_physical_metric_valid"] = (
                                losses["L_total"].new_ones(())
                            )
                        if native_future_no_teacher_weight > 0.0:
                            no_teacher_future = out[
                                "native_action_no_teacher_pred_tokens"
                            ]
                            if no_teacher_future.requires_grad:
                                if telemetry_enabled:
                                    no_teacher_future.retain_grad()
                                no_teacher_future_gradient_evidence = (
                                    register_nonzero_gradient_evidence(
                                        no_teacher_future
                                    )
                                )
                            native_future_losses = (
                                compute_native_no_teacher_future_loss(
                                    out,
                                    loss_tgt,
                                    train_cfg,
                                )
                            )
                            losses["L_total"] = losses["L_total"] + (
                                native_future_no_teacher_weight
                                * native_future_losses["loss"]
                            )
                            losses.update(
                                {
                                    f"native_future_no_teacher_{name}": value.detach()
                                    for name, value in native_future_losses.items()
                                }
                            )
                        branch_losses = {}
                        if "branch_actions" in tgt:
                            if "branch_s_tgt" not in tgt:
                                raise RuntimeError("true branch actions require branch_s_tgt; pseudo targets are forbidden")
                            if (
                                (
                                    telemetry_enabled
                                    or bool(
                                        train_cfg.get(
                                            "true_branch_require_nonzero_gradient",
                                            False,
                                        )
                                    )
                                )
                                and out["candidate_pred_tokens"].requires_grad
                            ):
                                if telemetry_enabled:
                                    out["candidate_pred_tokens"].retain_grad()
                                candidate_future_gradient_evidence = (
                                    register_nonzero_gradient_evidence(
                                        out["candidate_pred_tokens"]
                                    )
                                )
                            branch_losses = true_branch_reconstruction_matching_loss(
                                out["candidate_pred_tokens"],
                                tgt["branch_s_tgt"],
                                branch_valid=tgt.get("branch_valid"),
                                cfg=TrueBranchLossConfig(
                                    temperature=float(train_cfg.get("true_branch_temperature", 0.1)),
                                    reconstruction_weight=float(
                                        train_cfg.get("true_branch_reconstruction_weight", 1.0)
                                    ),
                                    matching_weight=float(train_cfg.get("true_branch_matching_weight", 1.0)),
                                    effect_temperature=float(
                                        train_cfg.get("true_branch_effect_temperature", 0.07)
                                    ),
                                    effect_reconstruction_weight=float(
                                        train_cfg.get("true_branch_effect_reconstruction_weight", 0.0)
                                    ),
                                    effect_matching_weight=float(
                                        train_cfg.get("true_branch_effect_matching_weight", 0.0)
                                    ),
                                    effect_norm_weight=float(
                                        train_cfg.get("true_branch_effect_norm_weight", 0.0)
                                    ),
                                    effect_min_rms=float(
                                        train_cfg.get("true_branch_effect_min_rms", 1e-3)
                                    ),
                                ),
                            )
                            losses["L_total"] = losses["L_total"] + float(
                                train_cfg.get("true_branch_weight", 1.0)
                            ) * branch_losses["loss"]
                            losses.update(
                                {f"true_{name}": value.detach() for name, value in branch_losses.items() if name != "pairwise_distance"}
                            )
                    future_value_losses = compute_true_branch_future_value_losses(
                        out, tgt, train_cfg
                    )
                    if future_value_losses:
                        losses["L_total"] = losses["L_total"] + float(
                            train_cfg.get("future_value_weight", 1.0)
                        ) * future_value_losses["loss"]
                        losses.update(
                            {
                                f"future_value_{name}": value.detach()
                                for name, value in future_value_losses.items()
                            }
                        )
                    losses.update({k: v.detach() for k, v in dropout_losses.items()})
                    rank_losses = compute_evaluator_pairwise_loss(
                        model,
                        s_cond,
                        c_cond,
                        out,
                        action_cond_model,
                        cfg["train"],
                        step=step,
                        policy_kwargs=policy_kwargs,
                    )
                    candidate_losses = compute_evaluator_candidate_pairwise_loss(
                        model,
                        s,
                        c,
                        out,
                        loss_tgt["action_tgt"],
                        loss_tgt["action_tgt_norm"],
                        cfg["train"],
                        policy_kwargs=policy_kwargs,
                    )
                if direct_policy_only:
                    context_pixel_action_losses = _context_pixel_action_zero_losses(losses["L_total"])
                elif not action_aux_batch:
                    context_pixel_action_losses = _context_pixel_action_zero_losses(losses["L_total"])
                else:
                    context_pixel_action_losses = compute_context_pixel_action_rank_loss(
                        model,
                        s_cond,
                        c_cond,
                        out,
                        action_cond_model,
                        context_rgb_cond,
                        tgt,
                        cfg["train"],
                        step=step,
                        prior_clean_tokens=prior_clean_tokens,
                        policy_kwargs=policy_kwargs,
                        stats_keys=tgt.get(
                            "action_stats_keys",
                            [source_name] * int(action_cond_model.shape[0]),
                        ),
                        collect_gradient_evidence=telemetry_enabled,
                    )
                direct_losses = compute_direct_policy_loss(
                    out,
                    loss_tgt["action_tgt"],
                    loss_tgt["action_tgt_norm"],
                    cfg["train"],
                    action_prev_grip=loss_tgt.get("action_prev_grip"),
                    step=step,
                )
                flow_losses = compute_policy_flow_matching_loss(
                    out,
                    loss_tgt["action_tgt"],
                    loss_tgt["action_tgt_norm"],
                    cfg["train"],
                )
                hunyuan_losses = {}
                if hunyuan_training_enabled and hunyuan_adapter is not None and hunyuan_vae is not None:
                    hunyuan_losses = compute_hunyuan_latent_loss(
                        hunyuan_adapter,
                        hunyuan_vae,
                        out,
                        tgt,
                        context_rgb_cond if 'context_rgb_cond' in locals() else context_rgb,
                        action_cond_model if 'action_cond_model' in locals() else action_cond,
                        c_cond if 'c_cond' in locals() else c,
                        train_cfg,
                    )
                if cfg["train"].get("evaluator_pairwise_weight", 0.0):
                    losses["L_total"] = (
                        losses["L_total"]
                        + float(cfg["train"]["evaluator_pairwise_weight"]) * rank_losses["L_evaluator_pairwise"]
                    )
                if cfg["train"].get("evaluator_candidate_pairwise_weight", 0.0):
                    losses["L_total"] = (
                        losses["L_total"]
                        + float(cfg["train"]["evaluator_candidate_pairwise_weight"])
                        * candidate_losses["L_evaluator_candidate_pairwise"]
                    )
                if cfg["train"].get("evaluator_candidate_ce_weight", 0.0):
                    losses["L_total"] = (
                        losses["L_total"]
                        + float(cfg["train"]["evaluator_candidate_ce_weight"])
                        * candidate_losses["L_evaluator_candidate_ce"]
                    )
                if context_pixel_action_rank_weight > 0.0:
                    losses["L_total"] = (
                        losses["L_total"]
                        + context_pixel_action_rank_weight
                        * context_pixel_action_losses["L_context_pixel_action_rank"]
                    )
                if context_pixel_action_separation_weight > 0.0:
                    losses["L_total"] = (
                        losses["L_total"]
                        + context_pixel_action_separation_weight
                        * context_pixel_action_losses["L_context_pixel_action_separation"]
                    )
                if core_action_rank_weight > 0.0:
                    losses["L_total"] = (
                        losses["L_total"]
                        + core_action_rank_weight
                        * context_pixel_action_losses["L_native_core_action_cf_rank"]
                    )
                if core_action_separation_weight > 0.0:
                    losses["L_total"] = (
                        losses["L_total"]
                        + core_action_separation_weight
                        * context_pixel_action_losses["L_native_core_action_cf_separation"]
                    )
                if cfg["train"].get("direct_policy_weight", 0.0):
                    losses["L_total"] = (
                        losses["L_total"]
                        + float(cfg["train"]["direct_policy_weight"]) * direct_losses["L_direct_policy"]
                    )
                if cfg["train"].get("policy_flow_weight", 0.0):
                    losses["L_total"] = (
                        losses["L_total"]
                        + float(cfg["train"]["policy_flow_weight"]) * flow_losses["L_policy_flow"]
                    )
                if train_cfg.get("hunyuan_latent_weight", 0.0) and hunyuan_losses:
                    losses["L_total"] = (
                        losses["L_total"]
                        + float(train_cfg["hunyuan_latent_weight"]) * hunyuan_losses["L_hunyuan_latent"]
                    )
                if train_cfg.get("prior_hunyuan_latent_weight", 0.0) and hunyuan_losses:
                    losses["L_total"] = (
                        losses["L_total"]
                        + float(train_cfg["prior_hunyuan_latent_weight"])
                        * hunyuan_losses["L_prior_hunyuan_latent"]
                    )
                losses.update({k: v.detach() for k, v in rank_losses.items()})
                losses.update({k: v.detach() for k, v in candidate_losses.items()})
                losses.update({k: v.detach() for k, v in context_pixel_action_losses.items()})
                losses.update({k: v.detach() for k, v in direct_losses.items()})
                losses.update({k: v.detach() for k, v in flow_losses.items()})
                losses.update({k: v.detach() for k, v in hunyuan_losses.items()})
            loss = losses["L_total"]
            finite_count = _distributed_finite_count(torch.isfinite(loss), device, world)
            if finite_count != world:
                if rank == 0:
                    print(
                        f"[rank0] WARN skip non-finite loss at step {step} "
                        f"finite_ranks={finite_count}/{world}",
                        flush=True,
                    )
                if bool(train_cfg.get("abort_on_nonfinite", False)):
                    raise RuntimeError(f"non-finite loss at step {step}: finite_ranks={finite_count}/{world}")
                opt.zero_grad(set_to_none=True); sched.step(); step += 1
                continue
            (loss / float(gradient_accumulation_steps)).backward()
            if isinstance(model, DDP):
                model.require_backward_grad_sync = True
            if isinstance(hunyuan_adapter, DDP):
                hunyuan_adapter.require_backward_grad_sync = True
            accumulation_index += 1
            if not sync_gradients:
                continue
            accumulation_index = 0
            accumulation_source_id = None
            if world > 1 and manual_grad_sync:
                _all_reduce_gradients(model, world)
                if hunyuan_adapter is not None:
                    _all_reduce_gradients(hunyuan_adapter, world)
            factual_gradient_groups = {
                "state_action_condition_projection": 0,
                "native_state_dynamics": 0,
                "no_teacher_action_head": 0,
            }
            branch_gradient_groups = {
                "candidate_future_grad_tensors": 0,
                "future_value_head_grad_params": 0,
            }
            no_teacher_future_grad_tensors = 0
            if native_future_no_teacher_weight > 0.0:
                if (
                    no_teacher_future_gradient_evidence is not None
                    and no_teacher_future_gradient_evidence["finite_nonzero"]
                ):
                    no_teacher_future_grad_tensors = 1
                    action_aux_trigger_counts[source_name][
                        "no_teacher_future_pred_grad_tensors"
                    ] += 1
            if representation_only_batch:
                allowed_prefixes = tuple(
                    str(value)
                    for value in train_cfg.get(
                        "representation_trainable_prefixes",
                        train_cfg.get(
                            "oxe_representation_trainable_prefixes",
                            ("pixel.", "context_pixel."),
                        ),
                    )
                )
                apply_source_gradient_policy(
                    model,
                    representation_only=True,
                    allowed_prefixes=allowed_prefixes,
                )
            else:
                apply_source_gradient_policy(
                    model,
                    representation_only=False,
                )
                factual_gradient_groups = factual_action_gradient_counts(model)
                for group_name, group_count in factual_gradient_groups.items():
                    action_aux_trigger_counts[source_name][
                        f"{group_name}_grad_params"
                    ] += group_count
                if "branch_actions" in tgt:
                    branch_gradient_groups = branch_objective_gradient_counts(
                        model,
                        out,
                    )
                    if (
                        candidate_future_gradient_evidence is not None
                        and candidate_future_gradient_evidence["finite_nonzero"]
                    ):
                        branch_gradient_groups[
                            "candidate_future_grad_tensors"
                        ] = 1
                    for group_name, group_count in branch_gradient_groups.items():
                        action_aux_trigger_counts[source_name][group_name] += group_count
                    if bool(
                        train_cfg.get(
                            "true_branch_require_nonzero_gradient",
                            False,
                        )
                    ) and (
                        branch_gradient_groups["candidate_future_grad_tensors"] == 0
                        or branch_gradient_groups["future_value_head_grad_params"] == 0
                    ):
                        raise RuntimeError(
                            "true same-root batch produced no nonzero imagine/select "
                            f"gradient evidence: {branch_gradient_groups}"
                        )
                factual_contract = train_cfg.get("factual_action_conditioning") or {}
                if bool(factual_contract.get("require_nonzero_gradient", False)) and (
                    factual_gradient_groups["state_action_condition_projection"] == 0
                    or factual_gradient_groups["native_state_dynamics"] == 0
                ):
                    raise RuntimeError(
                        f"factual source {source_name!r} produced no nonzero "
                        "action-conditioned native-core gradients"
                    )
            if telemetry_enabled:
                counterfactual_evidence = torch.stack(
                    [
                        losses.get(
                            "native_core_action_cf_wrong_action_input_grad",
                            loss.new_zeros(()),
                        ).detach().float(),
                        losses.get(
                            "native_core_action_cf_wrong_pred_grad",
                            loss.new_zeros(()),
                        ).detach().float(),
                    ]
                )
                if world > 1:
                    # Every rank follows the same source schedule.  MAX makes
                    # the evidence represent the complete 16-rank global
                    # batch instead of failing because rank 0 alone happened
                    # to receive an invalid/too-close local negative.
                    dist.all_reduce(
                        counterfactual_evidence, op=dist.ReduceOp.MAX
                    )
                if float(counterfactual_evidence[0]) > 0.0:
                    counterfactual_path_gradient_counts[source_name][
                        "wrong_action_input_grad"
                    ] += 1
                if float(counterfactual_evidence[1]) > 0.0:
                    counterfactual_path_gradient_counts[source_name][
                        "native_pred_tokens_grad"
                    ] += 1
            gn = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_value)
            grad_finite_count = _distributed_finite_count(torch.isfinite(gn), device, world)
            if grad_finite_count != world:
                if rank == 0:
                    print(
                        f"[rank0] WARN skip non-finite grad_norm at step {step} "
                        f"finite_ranks={grad_finite_count}/{world}",
                        flush=True,
                    )
                if bool(train_cfg.get("abort_on_nonfinite", False)):
                    raise RuntimeError(f"non-finite grad_norm at step {step}: finite_ranks={grad_finite_count}/{world}")
                opt.zero_grad(set_to_none=True); sched.step(); step += 1
                continue
            opt.step(); sched.step()
            opt.zero_grad(set_to_none=True)
            optimizer_step_seconds: float | None = None
            if measure_step_time:
                torch.cuda.synchronize(device)
                local_duration = torch.tensor(
                    [time.perf_counter() - step_started],
                    device=device,
                    dtype=torch.float64,
                )
                if world > 1:
                    dist.all_reduce(local_duration, op=dist.ReduceOp.MAX)
                optimizer_step_seconds = float(local_duration.item())
                step_duration_window.append(optimizer_step_seconds)
                if len(step_duration_window) > 1000:
                    del step_duration_window[:-1000]
            if rank == 0 and telemetry_enabled:
                source_cycle_optimizer_steps = (
                    sum(tr_s.source_cycle_counts.values())
                    if isinstance(tr_s, SourceHomogeneousDistributedBatchSampler)
                    else None
                )
                source_cycle_position = (
                    (micro_batches_consumed_in_epoch // gradient_accumulation_steps)
                    % source_cycle_optimizer_steps
                    if source_cycle_optimizer_steps
                    else None
                )
                source_cycle_position_before_optimizer_step = (
                    (
                        (
                            micro_batches_consumed_in_epoch
                            - gradient_accumulation_steps
                        )
                        // gradient_accumulation_steps
                    )
                    % source_cycle_optimizer_steps
                    if source_cycle_optimizer_steps
                    else None
                )
                condition_stats = action_condition_telemetry(
                    action_cond_model
                    if not direct_policy_only
                    else None
                )
                telemetry_record = {
                    "schema": "wm3d_v7_action_dynamics_canary_telemetry_v1",
                    **canary_timing_fields(optimizer_step_seconds),
                    "step": step + 1,
                    "epoch": epoch,
                    "micro_batches_consumed_in_epoch": micro_batches_consumed_in_epoch,
                    "source": source_name,
                    "source_policy": action_source_policy[source_name],
                    "rank_local_cache_shards": rank_local_cache_audit,
                    "global_mixed_source_identity": global_mixed_source_audit,
                    "main_teacher_action_weight": main_teacher_action_weight,
                    "factual_action_condition_enabled": bool(
                        not representation_only_batch
                        and not direct_policy_only
                        and action_cond_model is not None
                    ),
                    "effective_action_dynamics_weights": {
                        "native_action_no_teacher": native_action_no_teacher_weight,
                        "native_future_no_teacher": native_future_no_teacher_weight,
                        "native_core_action_cf_rank": core_action_rank_weight,
                        "native_core_action_cf_separation": core_action_separation_weight,
                    },
                    "source_cycle_position": source_cycle_position,
                    "source_cycle_position_before_optimizer_step": (
                        source_cycle_position_before_optimizer_step
                    ),
                    **condition_stats,
                    "aux_active": {
                        "native_action_no_teacher": native_action_no_teacher_weight > 0.0,
                        "native_core_action_cf": max(
                            core_action_rank_weight,
                            core_action_separation_weight,
                        ) > 0.0,
                        "context_pixel_action_cf": max(
                            context_pixel_action_rank_weight,
                            context_pixel_action_separation_weight,
                        ) > 0.0,
                    },
                    "factual_action_gradient_counts": factual_gradient_groups,
                    "no_teacher_future_pred_grad_tensors": (
                        no_teacher_future_grad_tensors
                    ),
                    "counterfactual_path_gradient_evidence": {
                        "wrong_action_input_grad": {
                            name: counts["wrong_action_input_grad"]
                            for name, counts in counterfactual_path_gradient_counts.items()
                        },
                        "native_pred_tokens_grad": {
                            name: counts["native_pred_tokens_grad"]
                            for name, counts in counterfactual_path_gradient_counts.items()
                        },
                    },
                    "metrics": {
                        "loss": float(loss.detach().float()),
                        "L_rgb": float(losses.get("L_rgb_l1", loss.new_zeros(())).detach().float()),
                        "L_depth": float(losses.get("L_depth", loss.new_zeros(())).detach().float()),
                        "L_point": float(losses.get("L_point", loss.new_zeros(())).detach().float()),
                        **action_dynamics_telemetry_metrics(losses, loss),
                    },
                }
                with telemetry_path.open("a", encoding="utf-8") as telemetry_file:
                    telemetry_file.write(
                        json.dumps(telemetry_record, sort_keys=True) + "\n"
                    )
            if rank == 0 and args.print_every and step % args.print_every == 0:
                rgb_l1 = float(losses.get("L_rgb_l1", torch.tensor(0.)).detach().float())
                lpv = float(losses.get("L_rgb_lpips", torch.tensor(0.)).detach().float())
                prop = float(losses.get("L_proposer_pose", torch.tensor(0.)).detach().float())
                prog = float(losses.get("L_progress", torch.tensor(0.)).detach().float())
                pair = float(losses.get("L_evaluator_pairwise", torch.tensor(0.)).detach().float())
                pair_acc = float(losses.get("evaluator_pairwise_acc", torch.tensor(0.)).detach().float())
                cand = float(losses.get("L_evaluator_candidate_pairwise", torch.tensor(0.)).detach().float())
                cand_ce = float(losses.get("L_evaluator_candidate_ce", torch.tensor(0.)).detach().float())
                cand_acc = float(losses.get("evaluator_candidate_pairwise_acc", torch.tensor(0.)).detach().float())
                cand_sel = float(losses.get("evaluator_candidate_selected_pose_l1", torch.tensor(0.)).detach().float())
                cand_anchor = float(losses.get("evaluator_candidate_anchor_pose_l1", torch.tensor(0.)).detach().float())
                ctx_rank = float(losses.get("L_context_pixel_action_rank", torch.tensor(0.)).detach().float())
                ctx_sep = float(losses.get("L_context_pixel_action_separation", torch.tensor(0.)).detach().float())
                ctx_acc = float(losses.get("context_pixel_action_acc", torch.tensor(0.)).detach().float())
                ctx_gap = float(losses.get("context_pixel_action_gap", torch.tensor(0.)).detach().float())
                ctx_rgb_gap = float(losses.get("context_pixel_action_rgb_gap", torch.tensor(0.)).detach().float())
                core_cf_rank = float(losses.get("L_native_core_action_cf_rank", torch.tensor(0.)).detach().float())
                core_cf_sep_loss = float(losses.get("L_native_core_action_cf_separation", torch.tensor(0.)).detach().float())
                core_cf_acc = float(losses.get("native_core_action_cf_acc", torch.tensor(0.)).detach().float())
                core_cf_gap = float(losses.get("native_core_action_cf_gap", torch.tensor(0.)).detach().float())
                core_cf_sep = float(losses.get("native_core_action_cf_sep", torch.tensor(0.)).detach().float())
                core_cf_neg_dist = float(losses.get("native_core_action_cf_negative_distance", torch.tensor(0.)).detach().float())
                core_cf_neg_valid = float(losses.get("native_core_action_cf_negative_valid", torch.tensor(0.)).detach().float())
                direct = float(losses.get("L_direct_policy", torch.tensor(0.)).detach().float())
                direct_pose = float(losses.get("direct_policy_pose_l1", torch.tensor(0.)).detach().float())
                direct_first = float(losses.get("direct_policy_first_pose_l1", torch.tensor(0.)).detach().float())
                direct_grip = float(losses.get("direct_policy_grip_acc", torch.tensor(0.)).detach().float())
                direct_grip_tacc = float(losses.get("direct_policy_grip_transition_acc", torch.tensor(0.)).detach().float())
                direct_grip_trate = float(losses.get("direct_policy_grip_transition_rate", torch.tensor(0.)).detach().float())
                direct_grip_pos = float(losses.get("direct_policy_grip_pos_rate", torch.tensor(0.)).detach().float())
                direct_grip_prob = float(losses.get("direct_policy_grip_prob_mean", torch.tensor(0.)).detach().float())
                direct_grip_pred_pos = float(losses.get("direct_policy_grip_pred_pos_rate", torch.tensor(0.)).detach().float())
                direct_grip_pos_acc = float(losses.get("direct_policy_grip_pos_acc", torch.tensor(0.)).detach().float())
                direct_grip_neg_acc = float(losses.get("direct_policy_grip_neg_acc", torch.tensor(0.)).detach().float())
                direct_grip_precision = float(losses.get("direct_policy_grip_precision", torch.tensor(0.)).detach().float())
                direct_grip_recall = float(losses.get("direct_policy_grip_recall", torch.tensor(0.)).detach().float())
                direct_grip_tbce = float(losses.get("direct_policy_grip_transition_bce", torch.tensor(0.)).detach().float())
                direct_grip_tup_bce = float(losses.get("direct_policy_grip_transition_up_bce", torch.tensor(0.)).detach().float())
                direct_grip_tdown_bce = float(losses.get("direct_policy_grip_transition_down_bce", torch.tensor(0.)).detach().float())
                direct_grip_tmargin = float(losses.get("direct_policy_grip_transition_margin", torch.tensor(0.)).detach().float())
                direct_grip_elmargin = float(losses.get("direct_policy_grip_event_logit_margin", torch.tensor(0.)).detach().float())
                direct_grip_blmargin = float(losses.get("direct_policy_grip_boundary_logit_margin", torch.tensor(0.)).detach().float())
                direct_grip_rate_mse = float(losses.get("direct_policy_grip_rate_mse", torch.tensor(0.)).detach().float())
                direct_grip_bbce = float(losses.get("direct_policy_grip_boundary_bce", torch.tensor(0.)).detach().float())
                direct_grip_bup_bce = float(losses.get("direct_policy_grip_boundary_up_bce", torch.tensor(0.)).detach().float())
                direct_grip_bdown_bce = float(losses.get("direct_policy_grip_boundary_down_bce", torch.tensor(0.)).detach().float())
                direct_grip_brate_mse = float(losses.get("direct_policy_grip_boundary_rate_mse", torch.tensor(0.)).detach().float())
                direct_grip_bpos = float(losses.get("direct_policy_grip_boundary_pos_rate", torch.tensor(0.)).detach().float())
                direct_grip_bprob = float(losses.get("direct_policy_grip_boundary_prob_mean", torch.tensor(0.)).detach().float())
                direct_grip_tup = float(losses.get("direct_policy_grip_transition_up_acc", torch.tensor(0.)).detach().float())
                direct_grip_tdown = float(losses.get("direct_policy_grip_transition_down_acc", torch.tensor(0.)).detach().float())
                direct_grip_bacc = float(losses.get("direct_policy_grip_boundary_acc", torch.tensor(0.)).detach().float())
                direct_grip_bup = float(losses.get("direct_policy_grip_boundary_up_acc", torch.tensor(0.)).detach().float())
                direct_grip_bdown = float(losses.get("direct_policy_grip_boundary_down_acc", torch.tensor(0.)).detach().float())
                direct_grip_brate = float(losses.get("direct_policy_grip_boundary_rate", torch.tensor(0.)).detach().float())
                direct_grip_dce = float(losses.get("direct_policy_grip_delta_ce", torch.tensor(0.)).detach().float())
                direct_grip_dacc = float(losses.get("direct_policy_grip_delta_acc", torch.tensor(0.)).detach().float())
                direct_grip_dhold = float(losses.get("direct_policy_grip_delta_hold_acc", torch.tensor(0.)).detach().float())
                direct_grip_dup = float(losses.get("direct_policy_grip_delta_up_acc", torch.tensor(0.)).detach().float())
                direct_grip_ddown = float(losses.get("direct_policy_grip_delta_down_acc", torch.tensor(0.)).detach().float())
                direct_grip_dbup = float(losses.get("direct_policy_grip_delta_boundary_up_acc", torch.tensor(0.)).detach().float())
                direct_grip_dbdown = float(losses.get("direct_policy_grip_delta_boundary_down_acc", torch.tensor(0.)).detach().float())
                direct_grip_ds_bce = float(losses.get("direct_policy_grip_delta_state_bce", torch.tensor(0.)).detach().float())
                direct_grip_ds_acc = float(losses.get("direct_policy_grip_delta_state_acc", torch.tensor(0.)).detach().float())
                direct_grip_ds_pos = float(losses.get("direct_policy_grip_delta_state_pos_acc", torch.tensor(0.)).detach().float())
                direct_grip_ds_neg = float(losses.get("direct_policy_grip_delta_state_neg_acc", torch.tensor(0.)).detach().float())
                direct_grip_ds_tup = float(losses.get("direct_policy_grip_delta_state_transition_up_acc", torch.tensor(0.)).detach().float())
                direct_grip_ds_tdown = float(losses.get("direct_policy_grip_delta_state_transition_down_acc", torch.tensor(0.)).detach().float())
                direct_grip_comp = float(losses.get("direct_policy_grip_composed_acc", torch.tensor(0.)).detach().float())
                direct_grip_comp_pos = float(losses.get("direct_policy_grip_composed_pos_acc", torch.tensor(0.)).detach().float())
                direct_grip_comp_neg = float(losses.get("direct_policy_grip_composed_neg_acc", torch.tensor(0.)).detach().float())
                direct_grip_comp_tup = float(losses.get("direct_policy_grip_composed_transition_up_acc", torch.tensor(0.)).detach().float())
                direct_grip_comp_tdown = float(losses.get("direct_policy_grip_composed_transition_down_acc", torch.tensor(0.)).detach().float())
                policy_flow = float(losses.get("L_policy_flow", torch.tensor(0.)).detach().float())
                policy_flow_pose = float(losses.get("policy_flow_pose_mse", torch.tensor(0.)).detach().float())
                policy_flow_grip = float(losses.get("policy_flow_grip_mse", torch.tensor(0.)).detach().float())
                policy_flow_recon = float(losses.get("policy_flow_recon_l1", torch.tensor(0.)).detach().float())
                policy_flow_recon_pose = float(losses.get("policy_flow_recon_pose_l1", torch.tensor(0.)).detach().float())
                policy_flow_recon_grip = float(losses.get("policy_flow_recon_grip_l1", torch.tensor(0.)).detach().float())
                hunyuan = float(losses.get("L_hunyuan_latent", torch.tensor(0.)).detach().float())
                prior_hunyuan = float(losses.get("L_prior_hunyuan_latent", torch.tensor(0.)).detach().float())
                prior = float(losses.get("L_world_prior", torch.tensor(0.)).detach().float())
                prior_flow = float(losses.get("L_world_prior_flow", torch.tensor(0.)).detach().float())
                prior_depth = float(losses.get("L_world_prior_depth", torch.tensor(0.)).detach().float())
                drop_a = float(losses.get("drop_action_frac", torch.tensor(0.)).detach().float())
                drop_c = float(losses.get("drop_context_frac", torch.tensor(0.)).detach().float())
                drop_text = float(losses.get("drop_text_only_frac", torch.tensor(0.)).detach().float())
                h_mse = float(losses.get("hunyuan_latent_mse", torch.tensor(0.)).detach().float())
                h_tmp = float(losses.get("hunyuan_latent_temporal_mse", torch.tensor(0.)).detach().float())
                depth = float(losses.get("L_depth", torch.tensor(0.)).detach().float())
                depth_ch = float(losses.get("L_depth_change", torch.tensor(0.)).detach().float())
                depth_mo = float(losses.get("L_depth_motion_l1", torch.tensor(0.)).detach().float())
                action_loss = float(losses.get("L_action", torch.tensor(0.)).detach().float())
                action_loss_raw = float(
                    losses.get("L_action_raw", torch.tensor(0.)).detach().float()
                )
                action_loss_weighted = float(
                    losses.get("L_action_weighted", torch.tensor(0.)).detach().float()
                )
                action_pose = float(losses.get("L_pose_action", torch.tensor(0.)).detach().float())
                action_pose_phys = float(losses.get("L_pose_action_physical", torch.tensor(0.)).detach().float())
                action_pose_norm = float(losses.get("L_pose_action_normalized", torch.tensor(0.)).detach().float())
                action_grip = float(losses.get("L_grip", torch.tensor(0.)).detach().float())
                native_no_teacher_action = float(
                    losses.get("native_no_teacher_L_action", torch.tensor(0.)).detach().float()
                )
                native_no_teacher_pose_norm = float(
                    losses.get(
                        "native_no_teacher_pose_huber",
                        torch.tensor(0.),
                    ).detach().float()
                )
                native_no_teacher_grip = float(
                    losses.get("native_no_teacher_grip_bce", torch.tensor(0.)).detach().float()
                )
                native_future_anchor = float(
                    losses.get("native_future_no_teacher_loss", torch.tensor(0.)).detach().float()
                )
                native_future_mse = float(
                    losses.get("native_future_no_teacher_mse", torch.tensor(0.)).detach().float()
                )
                native_future_cosine = float(
                    losses.get("native_future_no_teacher_cosine", torch.tensor(0.)).detach().float()
                )
                native_trans_gain = float(losses.get("native_no_teacher_translation_gain_vs_zero", torch.tensor(0.)).detach().float())
                native_rot_gain = float(losses.get("native_no_teacher_rotation_gain_vs_zero", torch.tensor(0.)).detach().float())
                native_trans_cos = float(losses.get("native_no_teacher_translation_cosine", torch.tensor(0.)).detach().float())
                native_rot_cos = float(losses.get("native_no_teacher_rotation_cosine", torch.tensor(0.)).detach().float())
                native_grip_bal = float(losses.get("native_no_teacher_grip_balanced_accuracy", torch.tensor(0.)).detach().float())
                native_grip_event = float(losses.get("native_no_teacher_grip_event_recall", torch.tensor(0.)).detach().float())
                true_recon = float(losses.get("true_branch_reconstruction", torch.tensor(0.)).detach().float())
                true_match = float(losses.get("true_branch_matching", torch.tensor(0.)).detach().float())
                true_top1 = float(losses.get("true_branch_matching_top1", torch.tensor(0.)).detach().float())
                true_effect_recon = float(losses.get("true_effect_reconstruction", torch.tensor(0.)).detach().float())
                true_effect_match = float(losses.get("true_effect_matching", torch.tensor(0.)).detach().float())
                true_effect_top1 = float(losses.get("true_effect_matching_top1", torch.tensor(0.)).detach().float())
                true_effect_cos = float(losses.get("true_effect_cosine", torch.tensor(0.)).detach().float())
                value_terminal_bce = float(losses.get("future_value_terminal_bce", torch.tensor(0.)).detach().float())
                value_terminal_acc = float(losses.get("future_value_terminal_acc", torch.tensor(0.)).detach().float())
                value_trajectory_acc = float(losses.get("future_value_trajectory_acc", torch.tensor(0.)).detach().float())
                value_ranking = float(losses.get("future_value_ranking_loss", torch.tensor(0.)).detach().float())
                value_ranking_acc = float(losses.get("future_value_ranking_acc", torch.tensor(0.)).detach().float())
                print(f"[rank0] step {step} (ep {epoch}) src={source_name or 'single'} "
                      f"L_total={float(loss.detach().float()):.4f} "
                      f"rgb_L1={rgb_l1:.4f} lpips={lpv:.4f} "
                      f"depth={depth:.4f} depth_ch={depth_ch:.4f} depth_mo={depth_mo:.4f} "
                      f"main_teacher_action_weight={main_teacher_action_weight:.4f} "
                      f"L_action_raw={action_loss_raw:.4f} "
                      f"L_action_weighted={action_loss_weighted:.4f} "
                      f"factual_action_condition_enabled={int(not representation_only_batch and not direct_policy_only and action_cond_model is not None)} "
                      f"action_objective={action_loss:.4f} action_pose={action_pose:.4f} "
                      f"action_pose_phys={action_pose_phys:.6f} action_pose_norm={action_pose_norm:.4f} "
                      f"action_grip={action_grip:.4f} "
                      f"native_action={native_no_teacher_action:.4f} "
                      f"native_pose_norm={native_no_teacher_pose_norm:.4f} "
                      f"native_grip={native_no_teacher_grip:.4f} "
                      f"native_future={native_future_anchor:.4f} "
                      f"native_future_mse={native_future_mse:.4f} "
                      f"native_future_cos={native_future_cosine:.4f} "
                      f"native_trans_gain={native_trans_gain:.3f} native_rot_gain={native_rot_gain:.3f} "
                      f"native_trans_cos={native_trans_cos:.3f} native_rot_cos={native_rot_cos:.3f} "
                      f"native_grip_bal={native_grip_bal:.3f} native_grip_event={native_grip_event:.3f} "
                      f"native_w={native_action_no_teacher_weight:.4f} "
                      f"native_future_w={native_future_no_teacher_weight:.4f} "
                      f"native_future_grad={no_teacher_future_grad_tensors} "
                      f"true_recon={true_recon:.4f} true_match={true_match:.4f} true_top1={true_top1:.3f} "
                      f"effect_recon={true_effect_recon:.4f} effect_match={true_effect_match:.4f} "
                      f"effect_top1={true_effect_top1:.3f} effect_cos={true_effect_cos:.3f} "
                      f"value_tbce={value_terminal_bce:.4f} value_tacc={value_terminal_acc:.3f} "
                      f"value_traj_acc={value_trajectory_acc:.3f} "
                      f"value_rank={value_ranking:.4f} value_rank_acc={value_ranking_acc:.3f} "
                      f"hunyuan={hunyuan:.4f} prior_hunyuan={prior_hunyuan:.4f} h_mse={h_mse:.4f} h_tmp={h_tmp:.4f} "
                      f"prior={prior:.4f} prior_flow={prior_flow:.4f} prior_depth={prior_depth:.4f} "
                      f"drop_a={drop_a:.2f} drop_ctx={drop_c:.2f} drop_text={drop_text:.2f} "
                      f"prop={prop:.4f} progress={prog:.4f} "
                      f"pair={pair:.4f} pair_acc={pair_acc:.3f} "
                      f"cand={cand:.4f} cand_ce={cand_ce:.4f} cand_acc={cand_acc:.3f} "
                      f"cand_sel={cand_sel:.3f} cand_anchor={cand_anchor:.3f} "
                      f"ctx_rank={ctx_rank:.4f} ctx_sep={ctx_sep:.4f} ctx_acc={ctx_acc:.3f} "
                      f"ctx_rank_w={context_pixel_action_rank_weight:.4f} "
                      f"ctx_sep_w={context_pixel_action_separation_weight:.4f} "
                      f"ctx_gap={ctx_gap:.4f} ctx_rgb_gap={ctx_rgb_gap:.4f} "
                      f"core_cf_rank={core_cf_rank:.4f} core_cf_sep_loss={core_cf_sep_loss:.4f} "
                      f"core_cf_acc={core_cf_acc:.3f} core_cf_gap={core_cf_gap:.6f} "
                      f"core_cf_sep={core_cf_sep:.6f} core_cf_neg_dist={core_cf_neg_dist:.4f} "
                      f"core_cf_neg_valid={core_cf_neg_valid:.3f} "
                      f"core_cf_rank_w={core_action_rank_weight:.4f} "
                      f"core_cf_sep_w={core_action_separation_weight:.4f} "
                      f"factual_grad_action_proj={factual_gradient_groups['state_action_condition_projection']} "
                      f"factual_grad_state_dynamics={factual_gradient_groups['native_state_dynamics']} "
                      f"factual_grad_no_teacher_head={factual_gradient_groups['no_teacher_action_head']} "
                      f"branch_grad_future={branch_gradient_groups['candidate_future_grad_tensors']} "
                      f"branch_grad_value_head={branch_gradient_groups['future_value_head_grad_params']} "
                      f"direct={direct:.4f} direct_pose={direct_pose:.3f} direct_first={direct_first:.3f} "
                      f"direct_grip={direct_grip:.3f} direct_grip_tacc={direct_grip_tacc:.3f} direct_grip_trate={direct_grip_trate:.3f} "
                      f"direct_grip_pos={direct_grip_pos:.3f} direct_grip_prob={direct_grip_prob:.3f} "
                      f"direct_grip_pred_pos={direct_grip_pred_pos:.3f} direct_grip_pos_acc={direct_grip_pos_acc:.3f} "
                      f"direct_grip_neg_acc={direct_grip_neg_acc:.3f} direct_grip_prec={direct_grip_precision:.3f} "
                      f"direct_grip_rec={direct_grip_recall:.3f} direct_grip_tbce={direct_grip_tbce:.4f} "
                      f"direct_grip_tup_bce={direct_grip_tup_bce:.4f} direct_grip_tdown_bce={direct_grip_tdown_bce:.4f} "
                      f"direct_grip_tmargin={direct_grip_tmargin:.4f} direct_grip_elmargin={direct_grip_elmargin:.4f} "
                      f"direct_grip_blmargin={direct_grip_blmargin:.4f} "
                      f"direct_grip_rate_mse={direct_grip_rate_mse:.4f} "
                      f"direct_grip_tup={direct_grip_tup:.3f} direct_grip_tdown={direct_grip_tdown:.3f} "
                      f"direct_grip_bbce={direct_grip_bbce:.4f} direct_grip_brate_mse={direct_grip_brate_mse:.4f} "
                      f"direct_grip_bup_bce={direct_grip_bup_bce:.4f} direct_grip_bdown_bce={direct_grip_bdown_bce:.4f} "
                      f"direct_grip_bpos={direct_grip_bpos:.3f} direct_grip_bprob={direct_grip_bprob:.3f} "
                      f"direct_grip_bacc={direct_grip_bacc:.3f} direct_grip_bup={direct_grip_bup:.3f} direct_grip_bdown={direct_grip_bdown:.3f} direct_grip_brate={direct_grip_brate:.3f} "
                      f"direct_grip_dce={direct_grip_dce:.4f} direct_grip_dacc={direct_grip_dacc:.3f} "
                      f"direct_grip_dhold={direct_grip_dhold:.3f} direct_grip_dup={direct_grip_dup:.3f} direct_grip_ddown={direct_grip_ddown:.3f} "
                      f"direct_grip_dbup={direct_grip_dbup:.3f} direct_grip_dbdown={direct_grip_dbdown:.3f} "
                      f"direct_grip_ds_bce={direct_grip_ds_bce:.4f} direct_grip_ds_acc={direct_grip_ds_acc:.3f} "
                      f"direct_grip_ds_pos={direct_grip_ds_pos:.3f} direct_grip_ds_neg={direct_grip_ds_neg:.3f} "
                      f"direct_grip_ds_tup={direct_grip_ds_tup:.3f} direct_grip_ds_tdown={direct_grip_ds_tdown:.3f} "
                      f"direct_grip_comp={direct_grip_comp:.3f} direct_grip_comp_pos={direct_grip_comp_pos:.3f} direct_grip_comp_neg={direct_grip_comp_neg:.3f} "
                      f"direct_grip_comp_tup={direct_grip_comp_tup:.3f} direct_grip_comp_tdown={direct_grip_comp_tdown:.3f} "
                      f"policy_flow={policy_flow:.4f} policy_flow_pose={policy_flow_pose:.4f} policy_flow_grip={policy_flow_grip:.4f} "
                      f"policy_flow_recon={policy_flow_recon:.4f} policy_flow_recon_pose={policy_flow_recon_pose:.4f} policy_flow_recon_grip={policy_flow_recon_grip:.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
                if measure_step_time and step_duration_window:
                    timing = np.asarray(step_duration_window, dtype=np.float64)
                    print(
                        f"[rank0] timing step={step} world={world} n={len(timing)} "
                        f"mean_s={timing.mean():.4f} p50_s={np.quantile(timing, 0.50):.4f} "
                        f"p90_s={np.quantile(timing, 0.90):.4f}",
                        flush=True,
                    )
                if mixed_source_batch_counts is not None:
                    mixed_total = max(1, sum(mixed_source_batch_counts))
                    source_audit = " ".join(
                        f"{name}={mixed_source_batch_counts[source_id] / mixed_total:.4f}"
                        for source_id, name in enumerate(tr_ds.source_names)
                    )
                    oxe_seen = sum(
                        mixed_source_batch_counts[source_id]
                        for source_id, name in enumerate(tr_ds.source_names)
                        if name.startswith("oxe_")
                    )
                    print(
                        f"[rank0] mixed_source_audit step={step} "
                        f"oxe_fraction={oxe_seen / mixed_total:.4f} {source_audit}",
                        flush=True,
                    )
                    action_audit = " ".join(
                        f"{name}="
                        f"groups:{counts['optimizer_groups']},"
                        f"nt:{counts['no_teacher']},"
                        f"nt_future:{counts['no_teacher_future']},"
                        f"core_cf:{counts['core_cf']},"
                        f"rgb_cf:{counts['rgb_cf']},"
                        f"state_cond_grad:{counts['state_action_condition_projection_grad_params']},"
                        f"state_dyn_grad:{counts['native_state_dynamics_grad_params']},"
                        f"nt_head_grad:{counts['no_teacher_action_head_grad_params']}"
                        f",nt_future_grad:{counts['no_teacher_future_pred_grad_tensors']}"
                        f",future_grad:{counts['candidate_future_grad_tensors']}"
                        f",value_head_grad:{counts['future_value_head_grad_params']}"
                        for name, counts in action_aux_trigger_counts.items()
                    )
                    print(
                        f"[rank0] action_aux_source_audit step={step} {action_audit}",
                        flush=True,
                    )
            if rank == 0 and step % cfg["train"]["log_every"] == 0:
                for k_, v in losses.items():
                    tb.add_scalar(f"train/{k_}", float(v.detach().float()), step)
                tb.add_scalar("lr", sched.get_last_lr()[0], step)
                tb.add_scalar("stage_pixel", float(do_pixel), step)
                if mixed_source_batch_counts is not None:
                    mixed_total = max(1, sum(mixed_source_batch_counts))
                    for source_id, name in enumerate(tr_ds.source_names):
                        tb.add_scalar(
                            f"data_fraction/{name}",
                            mixed_source_batch_counts[source_id] / mixed_total,
                            step,
                        )
                for name, counts in action_aux_trigger_counts.items():
                    for metric_name, value in counts.items():
                        tb.add_scalar(
                            f"action_source/{name}/{metric_name}",
                            float(value),
                            step,
                        )
            ckpt_every_steps = int(cfg["train"].get("ckpt_every_steps", 0) or 0)
            checkpoint_milestones = {
                int(value)
                for value in cfg["train"].get("checkpoint_milestone_steps", ())
                if int(value) > 0
            }
            checkpoint_due = (
                (ckpt_every_steps > 0 and (step + 1) % ckpt_every_steps == 0)
                or (step + 1) in checkpoint_milestones
            )
            if rank == 0 and checkpoint_due:
                source_cycle_optimizer_steps = (
                    sum(tr_s.source_cycle_counts.values())
                    if isinstance(tr_s, SourceHomogeneousDistributedBatchSampler)
                    else None
                )
                source_cycle_position = (
                    (micro_batches_consumed_in_epoch // gradient_accumulation_steps)
                    % source_cycle_optimizer_steps
                    if source_cycle_optimizer_steps
                    else None
                )
                step_ckpt = {
                    "model": (model.module if isinstance(model, DDP) else model).state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "epoch": epoch,
                    "step": step + 1,
                    "val_total": None,
                    "best_val": best_val,
                    "cfg": cfg,
                    "resolved_config_sha256": resolved_config_sha256,
                    "resume_compat_sha256": resolved_resume_compat_sha256,
                    "run_lineage": resolved_run_lineage,
                    "stage_transition_audit": stage_transition_audit,
                    "rng_contract_rank0": capture_rng_contract(base_seed, rank),
                    "sampler_state": {
                        "schema": "wm3d_v7_exact_source_cycle_v1",
                        "epoch": epoch,
                        "micro_batches_consumed_in_epoch": micro_batches_consumed_in_epoch,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "source_cycle_optimizer_steps": source_cycle_optimizer_steps,
                        "source_cycle_position": source_cycle_position,
                        "sampler_seed": int(
                            mixed_sampler_cfg.get("seed", data_cfg.get("seed", 0))
                        ),
                        "sampler_num_replicas": int(sampler_replicas),
                        "sampler_rank_scope": (
                            "local-node"
                            if rank_local_cache_audit.get("enabled", False)
                            else "global"
                        ),
                    },
                }
                if hunyuan_adapter is not None:
                    adapter_target = hunyuan_adapter.module if isinstance(hunyuan_adapter, DDP) else hunyuan_adapter
                    step_ckpt["hunyuan_adapter"] = adapter_target.state_dict()
                    step_ckpt["hunyuan_adapter_cfg"] = adapter_target.cfg.__dict__
                save_step_checkpoint_once(step_ckpt, ckpt_dir, step + 1)
            if world > 1 and checkpoint_due:
                dist.barrier()
            step += 1
            if invocation_stop_step > 0 and step >= invocation_stop_step:
                if rank == 0:
                    print(
                        f"[rank0] reached invocation_stop_step={invocation_stop_step}; "
                        "running validation/checkpoint",
                        flush=True,
                    )
                break
        validate_every_epochs = max(1, int(train_cfg.get("validate_every_epochs", 1)))
        reached_invocation_stop = (
            invocation_stop_step > 0 and step >= invocation_stop_step
        )
        if not reached_invocation_stop and (epoch + 1) % validate_every_epochs != 0:
            continue
        # Val
        model.eval()
        agg = {}
        agg_count = {}
        nb = 0
        future_value_task_bank: list[torch.Tensor] = []
        max_val_batches = int(train_cfg.get("max_val_batches", 0))
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                if max_val_batches > 0 and bi >= max_val_batches:
                    break
                val_source_name = "single"
                if isinstance(val_ds, MixedSourceWindowDataset):
                    val_source_ids = torch.as_tensor(batch["source_id"]).reshape(-1)
                    unique_val_source_ids = torch.unique(val_source_ids)
                    if unique_val_source_ids.numel() != 1:
                        raise RuntimeError("validation mixed-source batch is not homogeneous")
                    val_source_name = val_ds.source_names[int(unique_val_source_ids.item())]
                val_representation_only = (
                    action_source_policy[val_source_name] == "representation_only"
                )
                val_native_action_weight = (
                    scheduled_aux_weight(
                        train_cfg,
                        "native_action_no_teacher_weight",
                        step,
                    )
                    if action_aux_source_allowed(
                        train_cfg,
                        val_source_name,
                        representation_only=val_representation_only,
                    )
                    else 0.0
                )
                val_native_future_weight = (
                    scheduled_aux_weight(
                        train_cfg,
                        "native_future_no_teacher_weight",
                        step,
                    )
                    if action_aux_source_allowed(
                        train_cfg,
                        val_source_name,
                        representation_only=val_representation_only,
                    )
                    else 0.0
                )
                s, c, action_cond, context_rgb, tgt = batch_to_device(
                    batch,
                    device,
                    k,
                    direct_policy_only=direct_policy_only,
                    action_grip_contract=action_grip_contract,
                    source_name=val_source_name,
                    require_factual_action_contract=(
                        mixed_source_training and not val_representation_only
                    ),
                )
                if val_representation_only and action_cond is not None:
                    action_cond = torch.zeros_like(action_cond)
                decode_codec_targets(model, tgt)
                loss_tgt = targets_with_close01_grip(tgt, action_grip_contract)
                multiview_kwargs = multiview_kwargs_from_targets(tgt)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if direct_policy_only:
                        target_model = model.module if isinstance(model, DDP) else model
                        policy_kwargs = action_policy_kwargs_from_targets(tgt)
                        policy_kwargs.update(
                            make_policy_flow_training_kwargs(loss_tgt, train_cfg)
                        )
                        out = _direct_policy_only_forward(
                            target_model,
                            s,
                            c,
                            action_cond=action_cond,
                            context_rgb=context_rgb,
                            policy_kwargs=policy_kwargs,
                            train_cfg=train_cfg,
                            multiview_kwargs=multiview_kwargs,
                        )
                        losses = {"L_total": out["policy_pose_norm"].new_zeros(())}
                    else:
                        prior_clean_tokens = prior_clean_tokens_from_targets(tgt)
                        policy_kwargs = action_policy_kwargs_from_targets(tgt)
                        policy_kwargs.update(
                            make_policy_flow_training_kwargs(loss_tgt, train_cfg)
                        )
                        out = _forward_joint_model(
                            model,
                            s,
                            c,
                            action_cond=action_cond,
                            context_rgb=context_rgb,
                            prior_clean_tokens=prior_clean_tokens,
                            pixel=do_pixel,
                            bridging=False,
                            policy_kwargs=policy_kwargs,
                            multiview_kwargs=multiview_kwargs,
                            candidate_actions=tgt.get("branch_actions"),
                            candidate_include_geometry=bool(train_cfg.get("true_branch_include_geometry", False)),
                            native_action_no_teacher=max(
                                val_native_action_weight,
                                val_native_future_weight,
                            )
                            > 0.0,
                        )
                        if future_value_only:
                            losses = {"L_total": out["candidate_success_logit"].new_zeros(())}
                        else:
                            val_weights = (
                                representation_weights
                                if val_representation_only
                                else factual_weights
                            )
                            losses = compute_losses(
                                out,
                                loss_tgt,
                                val_weights,
                                lpips_fn if do_pixel else None,
                            )
                            losses["L_action_raw"] = losses["L_action"]
                            losses["L_action_weighted"] = (
                                float(val_weights.action) * losses["L_action"]
                            )
                            losses["teacher_action_diagnostic"] = losses["L_action"]
                            losses["L_action"] = losses["L_action"].new_zeros(())
                            if val_native_action_weight > 0.0:
                                native_action_losses = compute_native_no_teacher_action_loss(
                                    out,
                                    loss_tgt,
                                    native_action_weights,
                                    train_cfg=train_cfg,
                                )
                                losses["L_total"] = losses["L_total"] + (
                                    val_native_action_weight
                                    * native_action_losses["L_action"]
                                )
                                for name, value in native_action_losses.items():
                                    losses[f"native_no_teacher_{name}"] = value.detach()
                                    safe_source = str(val_source_name).replace("/", "_")
                                    losses[
                                        f"native_source_{safe_source}_{name}"
                                    ] = value.detach()
                                losses["native_no_teacher_physical_metric_valid"] = (
                                    losses["L_total"].new_ones(())
                                )
                            if val_native_future_weight > 0.0:
                                native_future_losses = (
                                    compute_native_no_teacher_future_loss(
                                        out,
                                        loss_tgt,
                                        train_cfg,
                                    )
                                )
                                losses["L_total"] = losses["L_total"] + (
                                    val_native_future_weight
                                    * native_future_losses["loss"]
                                )
                                losses.update(
                                    {
                                        f"native_future_no_teacher_{name}": value.detach()
                                        for name, value in native_future_losses.items()
                                    }
                                )
                            if "branch_actions" in tgt:
                                if "branch_s_tgt" not in tgt:
                                    raise RuntimeError("true branch actions require branch_s_tgt")
                                branch_losses = true_branch_reconstruction_matching_loss(
                                    out["candidate_pred_tokens"],
                                    tgt["branch_s_tgt"],
                                    branch_valid=tgt.get("branch_valid"),
                                    cfg=TrueBranchLossConfig(
                                        temperature=float(train_cfg.get("true_branch_temperature", 0.1)),
                                        reconstruction_weight=float(
                                            train_cfg.get("true_branch_reconstruction_weight", 1.0)
                                        ),
                                        matching_weight=float(train_cfg.get("true_branch_matching_weight", 1.0)),
                                        effect_temperature=float(
                                            train_cfg.get("true_branch_effect_temperature", 0.07)
                                        ),
                                        effect_reconstruction_weight=float(
                                            train_cfg.get("true_branch_effect_reconstruction_weight", 0.0)
                                        ),
                                        effect_matching_weight=float(
                                            train_cfg.get("true_branch_effect_matching_weight", 0.0)
                                        ),
                                        effect_norm_weight=float(
                                            train_cfg.get("true_branch_effect_norm_weight", 0.0)
                                        ),
                                        effect_min_rms=float(
                                            train_cfg.get("true_branch_effect_min_rms", 1e-3)
                                        ),
                                    ),
                                )
                                losses["L_total"] = losses["L_total"] + float(
                                    train_cfg.get("true_branch_weight", 1.0)
                                ) * branch_losses["loss"]
                                losses.update(
                                    {f"true_{name}": value.detach() for name, value in branch_losses.items() if name != "pairwise_distance"}
                                )
                        future_value_losses = compute_true_branch_future_value_losses(
                            out, tgt, train_cfg
                        )
                        if future_value_losses:
                            losses["L_total"] = losses["L_total"] + float(
                                train_cfg.get("future_value_weight", 1.0)
                            ) * future_value_losses["loss"]
                            losses.update(
                                {
                                    f"future_value_{name}": value.detach()
                                    for name, value in future_value_losses.items()
                                }
                            )
                            target_model = model.module if isinstance(model, DDP) else model
                            swap_metrics = compute_future_value_task_swap_metrics(
                                target_model.future_value_head,
                                out["candidate_pred_tokens"],
                                c,
                                out,
                                tgt["branch_success"],
                                tgt.get("branch_valid"),
                                future_value_task_bank,
                            )
                            losses.update(swap_metrics)
                    direct_losses = compute_direct_policy_loss(
                        out,
                        loss_tgt["action_tgt"],
                        loss_tgt["action_tgt_norm"],
                        cfg["train"],
                        action_prev_grip=loss_tgt.get("action_prev_grip"),
                        step=step,
                    )
                    flow_losses = compute_policy_flow_matching_loss(
                        out,
                        loss_tgt["action_tgt"],
                        loss_tgt["action_tgt_norm"],
                        cfg["train"],
                    )
                    hunyuan_losses = {}
                    if hunyuan_training_enabled and hunyuan_adapter is not None and hunyuan_vae is not None:
                        hunyuan_losses = compute_hunyuan_latent_loss(
                            hunyuan_adapter,
                            hunyuan_vae,
                            out,
                            tgt,
                            context_rgb,
                            action_cond,
                            c,
                            train_cfg,
                        )
                    if cfg["train"].get("direct_policy_weight", 0.0):
                        losses["L_total"] = (
                            losses["L_total"]
                            + float(cfg["train"]["direct_policy_weight"]) * direct_losses["L_direct_policy"]
                        )
                    if cfg["train"].get("policy_flow_weight", 0.0):
                        losses["L_total"] = (
                            losses["L_total"]
                            + float(cfg["train"]["policy_flow_weight"]) * flow_losses["L_policy_flow"]
                        )
                    if train_cfg.get("hunyuan_latent_weight", 0.0) and hunyuan_losses:
                        losses["L_total"] = (
                            losses["L_total"]
                            + float(train_cfg["hunyuan_latent_weight"]) * hunyuan_losses["L_hunyuan_latent"]
                        )
                    if train_cfg.get("prior_hunyuan_latent_weight", 0.0) and hunyuan_losses:
                        losses["L_total"] = (
                            losses["L_total"]
                            + float(train_cfg["prior_hunyuan_latent_weight"])
                            * hunyuan_losses["L_prior_hunyuan_latent"]
                        )
                    losses.update({k: v.detach() for k, v in direct_losses.items()})
                    losses.update({k: v.detach() for k, v in flow_losses.items()})
                    losses.update({k: v.detach() for k, v in hunyuan_losses.items()})
                for kk, v in losses.items():
                    agg[kk] = agg.get(kk, 0.0) + float(v.detach().float())
                    agg_count[kk] = agg_count.get(kk, 0.0) + 1.0
                nb += 1
        if world > 1:
            local_keys = sorted(agg.keys())
            gathered_keys = [None for _ in range(world)]
            dist.all_gather_object(gathered_keys, local_keys)
            keys = sorted({key for key_list in gathered_keys for key in (key_list or [])})
            reduce_device = "cpu" if dist.get_backend() == "gloo" else device
            v = torch.tensor(
                [agg.get(kk, 0.0) for kk in keys]
                + [agg_count.get(kk, 0.0) for kk in keys]
                + [float(nb)],
                device=reduce_device,
                dtype=torch.float64,
            )
            dist.all_reduce(v)
            for i, kk in enumerate(keys):
                agg[kk] = float(v[i].item())
                agg_count[kk] = float(v[len(keys) + i].item())
            nb = int(v[-1].item())
        if rank == 0:
            def validation_mean(metric: str) -> float:
                return agg.get(metric, 0.0) / max(1.0, agg_count.get(metric, 0.0))

            for kk, vv in agg.items():
                tb.add_scalar(f"val/{kk}", validation_mean(kk), step)
                tb.add_scalar(
                    f"val_count/{kk}", agg_count.get(kk, 0.0), step
                )
            val_total = validation_mean("L_total")
            final_cycle_steps = (
                sum(tr_s.source_cycle_counts.values())
                if isinstance(tr_s, SourceHomogeneousDistributedBatchSampler)
                else None
            )
            ckpt = {"model": (model.module if isinstance(model, DDP) else model).state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "step": step, "val_total": val_total,
                    "best_val": best_val, "cfg": cfg,
                    "resolved_config_sha256": resolved_config_sha256,
                    "resume_compat_sha256": resolved_resume_compat_sha256,
                    "run_lineage": resolved_run_lineage,
                    "stage_transition_audit": stage_transition_audit,
                    "rng_contract_rank0": capture_rng_contract(base_seed, rank),
                    "sampler_state": {
                        "schema": "wm3d_v7_exact_source_cycle_v1",
                        "epoch": epoch,
                        "micro_batches_consumed_in_epoch": micro_batches_consumed_in_epoch,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "source_cycle_optimizer_steps": final_cycle_steps,
                        "source_cycle_position": (
                            (micro_batches_consumed_in_epoch // gradient_accumulation_steps)
                            % final_cycle_steps
                            if final_cycle_steps else None
                        ),
                        "sampler_seed": int(
                            mixed_sampler_cfg.get("seed", data_cfg.get("seed", 0))
                        ),
                        "sampler_num_replicas": int(sampler_replicas),
                        "sampler_rank_scope": (
                            "local-node"
                            if rank_local_cache_audit.get("enabled", False)
                            else "global"
                        ),
                    }}
            if hunyuan_adapter is not None:
                adapter_target = hunyuan_adapter.module if isinstance(hunyuan_adapter, DDP) else hunyuan_adapter
                ckpt["hunyuan_adapter"] = adapter_target.state_dict()
                ckpt["hunyuan_adapter_cfg"] = adapter_target.cfg.__dict__
            if (epoch + 1) % cfg["train"]["ckpt_every_epochs"] == 0:
                torch.save(ckpt, ckpt_dir / f"epoch_{epoch:03d}.pt")
            if bool(train_cfg.get("save_best_checkpoint", True)) and val_total < best_val:
                best_val = val_total
                ckpt["best_val"] = best_val
                torch.save(ckpt, ckpt_dir / "best.pt")
            # A bounded proof or deadline-driven formal run must always be
            # resumable, even when max_steps is not aligned to the periodic
            # checkpoint interval and best-checkpoint saving is disabled.
            if reached_invocation_stop and (
                ckpt_every_steps <= 0 or step % ckpt_every_steps != 0
            ):
                save_step_checkpoint_once(ckpt, ckpt_dir, step)
            val_direct = validation_mean("L_direct_policy")
            val_direct_pose = validation_mean("direct_policy_pose_l1")
            val_direct_first_pose = validation_mean("direct_policy_first_pose_l1")
            val_direct_grip = validation_mean("direct_policy_grip_acc")
            val_direct_grip_transition = validation_mean("direct_policy_grip_transition_acc")
            val_direct_grip_pos = validation_mean("direct_policy_grip_pos_rate")
            val_direct_grip_prob = validation_mean("direct_policy_grip_prob_mean")
            val_hunyuan = validation_mean("L_hunyuan_latent")
            val_prior_hunyuan = validation_mean("L_prior_hunyuan_latent")
            val_hunyuan_mse = validation_mean("hunyuan_latent_mse")
            val_true_recon = validation_mean("true_branch_reconstruction")
            val_true_match = validation_mean("true_branch_matching")
            val_true_top1 = validation_mean("true_branch_matching_top1")
            val_effect_recon = validation_mean("true_effect_reconstruction")
            val_effect_match = validation_mean("true_effect_matching")
            val_effect_top1 = validation_mean("true_effect_matching_top1")
            val_effect_cos = validation_mean("true_effect_cosine")
            val_effect_norm = validation_mean("true_effect_norm_ratio")
            val_effect_gain_zero = validation_mean("true_effect_gain_vs_zero")
            val_value_terminal_bce = validation_mean("future_value_terminal_bce")
            val_value_terminal_acc = validation_mean("future_value_terminal_acc")
            val_value_trajectory_acc = validation_mean("future_value_trajectory_acc")
            val_value_ranking = validation_mean("future_value_ranking_loss")
            val_value_ranking_acc = validation_mean("future_value_ranking_acc")
            val_value_task_swap_l1 = validation_mean("future_value_task_swap_logit_l1")
            val_native_trans_gain = validation_mean(
                "native_no_teacher_translation_gain_vs_zero"
            )
            val_native_rot_gain = validation_mean(
                "native_no_teacher_rotation_gain_vs_zero"
            )
            val_native_trans_cos = validation_mean(
                "native_no_teacher_translation_cosine"
            )
            val_native_rot_cos = validation_mean(
                "native_no_teacher_rotation_cosine"
            )
            val_native_grip_bal = validation_mean(
                "native_no_teacher_grip_balanced_accuracy"
            )
            val_native_grip_event = validation_mean(
                "native_no_teacher_grip_event_recall"
            )
            print(
                f"[rank0] epoch {epoch}: val_total {val_total:.4f} "
                f"val_true_recon={val_true_recon:.4f} "
                f"val_true_match={val_true_match:.4f} "
                f"val_true_top1={val_true_top1:.3f} "
                f"val_effect_recon={val_effect_recon:.4f} "
                f"val_effect_match={val_effect_match:.4f} "
                f"val_effect_top1={val_effect_top1:.3f} "
                f"val_effect_cos={val_effect_cos:.3f} "
                f"val_effect_norm={val_effect_norm:.3f} "
                f"val_effect_gain_zero={val_effect_gain_zero:.3f} "
                f"val_value_tbce={val_value_terminal_bce:.4f} "
                f"val_value_tacc={val_value_terminal_acc:.3f} "
                f"val_value_traj_acc={val_value_trajectory_acc:.3f} "
                f"val_value_rank={val_value_ranking:.4f} "
                f"val_value_rank_acc={val_value_ranking_acc:.3f} "
                f"val_value_task_swap_l1={val_value_task_swap_l1:.4f} "
                f"val_native_trans_gain={val_native_trans_gain:.3f} "
                f"val_native_rot_gain={val_native_rot_gain:.3f} "
                f"val_native_trans_cos={val_native_trans_cos:.3f} "
                f"val_native_rot_cos={val_native_rot_cos:.3f} "
                f"val_native_grip_bal={val_native_grip_bal:.3f} "
                f"val_native_grip_event={val_native_grip_event:.3f} "
                f"val_direct={val_direct:.4f} "
                f"val_direct_pose_l1={val_direct_pose:.4f} "
                f"val_direct_first_pose_l1={val_direct_first_pose:.4f} "
                f"val_direct_grip_acc={val_direct_grip:.3f} "
                f"val_direct_grip_transition_acc={val_direct_grip_transition:.3f} "
                f"val_direct_grip_pos={val_direct_grip_pos:.3f} "
                f"val_direct_grip_prob={val_direct_grip_prob:.3f} "
                f"val_hunyuan={val_hunyuan:.4f} "
                f"val_prior_hunyuan={val_prior_hunyuan:.4f} "
                f"val_hunyuan_mse={val_hunyuan_mse:.4f} "
                f"(best {best_val:.4f}, pixel={do_pixel})"
            )
        # Keep all ranks at the epoch boundary while rank 0 serializes a large
        # checkpoint.  Otherwise the next DDP forward can wait on rank 0 for
        # minutes and look like a collective hang.
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
