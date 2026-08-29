"""Unified WM3D Stage0 trainer for every model/data/runtime profile."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, default_collate

from wm3d.data.manifest_contract import load_data_profile
from wm3d.data.step_sampler import StepAddressedBatchSampler
from wm3d.data.unified_cache_dataset import UnifiedCacheDataset
from wm3d.data.streaming_raw import (
    STREAMING_DATA_CLOSURE_SCHEMA,
    StreamingRawDataset,
    load_streaming_metadata_seal,
)
from wm3d.data.direct_raw import (
    DIRECT_RAW_DATA_CLOSURE_SCHEMA,
    DirectRawDataset,
)
from wm3d.data.grouped_normalization import GroupedRobotNormalizer
from wm3d.data.formal_cache_adapter import (
    FORMAL_CACHE_CLOSURE_SCHEMA,
    build_formal_cache_dataset,
)
from wm3d.models.model_factory import build_world_model
from wm3d.models.native_world_model import NativeWorldModel
from wm3d.models.direct_vggt_builder import build_direct_vggt_teacher
from wm3d.training.async_input import AsyncCudaInputPipeline
from wm3d.training.distributed_checkpoint import (
    DistributedCheckpointManager,
    ResumeExpectations,
    canonical_sha256,
)
from wm3d.training.distributed_runtime import (
    autocast_context,
    destroy_distributed,
    initialize_adamw_state,
    initialize_distributed,
    no_sync_context,
    reduce_metrics,
    strategy_from_mapping,
    wrap_model,
)
from wm3d.training.native_objective import (
    build_rgb_perceptual_model,
    compute_native_objective,
    objective_config_from_mapping,
)
from wm3d.training.gradient_ownership import (
    GradientOwnershipError,
    audit_gradient_ownership,
    required_gradient_owner_names,
    validate_gradient_ownership_receipt,
)
from wm3d.training.runtime_contract import load_materialized_runtime
from wm3d.training.resource_preflight import (
    ResourcePreflightError,
    current_rank_identity,
    run_resource_preflight,
    validate_current_rank_identities,
    validate_resource_receipt,
)
from wm3d.training.launch_qualification import (
    LaunchQualificationError,
    build_launch_qualification,
    publish_launch_qualification,
    resource_contract_sha256,
    validate_launch_qualification,
    verify_clean_runtime_checkout,
)


RUN_CONTRACT_SCHEMA = "wm3d_v8_run_contract_v3"
RESOURCE_PREFLIGHT_PREFIX = "resource_preflight_"


class PretrainError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Explicit committed step_XXXXXXXX directory; 'latest' is forbidden.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--stop-after-step",
        type=int,
        default=None,
        help=(
            "Stop cleanly after this optimizer step without changing the sealed "
            "total_steps contract; used by exact-resume validation."
        ),
    )
    return parser.parse_args()


def _environment_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    if value not in {"0", "1"}:
        raise PretrainError(
            f"environment flag {name} must be exactly '0' or '1'"
        )
    return value == "1"


def _configure_reproducibility(
    seed: int, *, cudnn_benchmark: bool = False
) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o640)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise PretrainError("run contract cannot be a symlink")
    if path.exists():
        _require_stable_run_contract(path, value)
        return
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_stable_run_contract(
    path: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Read one immutable run-contract snapshot and require exact equality."""

    path = Path(path)
    if path.is_symlink():
        raise PretrainError("run contract cannot be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PretrainError("stable run contract is missing") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise PretrainError("run contract must be a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise PretrainError("run contract changed while it was read")
    finally:
        os.close(descriptor)
    try:
        current = os.lstat(path)
    except OSError as exc:
        raise PretrainError("run contract changed after it was read") from exc
    if (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    ) != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns):
        raise PretrainError("run contract changed after it was read")
    try:
        value = json.loads(b"".join(chunks))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PretrainError("run contract is not valid JSON") from exc
    if value != expected:
        raise PretrainError("stable run contract is missing or differs")
    return dict(value)


def _resource_preflight(
    config: Mapping[str, Any], config_sha: str, context: Any
) -> str | None:
    resources = config["runtime_profile"].get("resources")
    if resources is None:
        return None
    output_root = Path(config["run"]["output_root"])
    closure = config["data_closure"]
    cache_root = Path(
        closure.get(
            "cache_root", closure.get("lru_root", closure.get("metadata_root", ""))
        )
    )
    if not cache_root.is_absolute():
        raise PretrainError("resource preflight data root is not absolute")
    if closure.get("schema") == STREAMING_DATA_CLOSURE_SCHEMA:
        cache_root.mkdir(parents=True, exist_ok=True)
    status: list[Any] = [None]
    try:
        receipt = run_resource_preflight(
            resources=resources,
            context=context,
            runtime_config_sha256=config_sha,
            cache_root=cache_root,
            output_root=output_root,
        )
        if context.is_rank0:
            receipt_path = output_root / (
                f"{RESOURCE_PREFLIGHT_PREFIX}{int(receipt['created_unix_ns'])}.json"
            )
            try:
                _atomic_json_no_clobber(receipt_path, receipt)
                receipt_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                if receipt.get("passed") is True and receipt.get("errors") == []:
                    status[0] = {
                        "ok": True,
                        "sha256": receipt_sha256,
                        "path": str(receipt_path),
                    }
                else:
                    status[0] = {
                        "ok": False,
                        "type": ResourcePreflightError.__name__,
                        "error": "; ".join(str(value) for value in receipt["errors"]),
                        "sha256": receipt_sha256,
                        "path": str(receipt_path),
                    }
            except Exception as exc:
                status[0] = {
                    "ok": False,
                    "type": type(exc).__name__,
                    "error": str(exc),
                }
    except ResourcePreflightError as exc:
        if context.is_rank0:
            status[0] = {
                "ok": False,
                "type": type(exc).__name__,
                "error": str(exc),
            }
    dist.broadcast_object_list(status, src=0)
    if not status[0]["ok"]:
        raise PretrainError(f"resource preflight failed: {status[0]}")
    return str(status[0]["sha256"])


def _require_recent_resource_preflight(
    config: Mapping[str, Any], config_sha: str, context: Any
) -> dict[str, Any] | None:
    resources = config["runtime_profile"].get("resources")
    if resources is None:
        return None
    selection: list[Any] = [None]
    if context.is_rank0:
        try:
            output_root = Path(config["run"]["output_root"])
            candidates = sorted(output_root.glob(f"{RESOURCE_PREFLIGHT_PREFIX}*.json"))
            if not candidates:
                raise PretrainError(
                    "resource-qualified runtime requires a prior --preflight-only receipt"
                )
            expected_world = int(config["runtime_profile"]["expected_world_size"])
            errors: list[str] = []
            valid: list[tuple[int, Path, dict[str, Any]]] = []
            for path in candidates:
                try:
                    if path.is_symlink() or not path.is_file():
                        raise PretrainError("not a regular file")
                    receipt = json.loads(path.read_text(encoding="utf-8"))
                    created_ns = validate_resource_receipt(
                        receipt,
                        resources=resources,
                        runtime_config_sha256=config_sha,
                        world_size=expected_world,
                    )
                    valid.append((created_ns, path, receipt))
                except Exception as exc:
                    errors.append(f"{path.name}: {exc}")
            if not valid:
                raise PretrainError(
                    "no matching fresh resource preflight receipt: " + "; ".join(errors)
                )
            _, path, receipt = max(valid, key=lambda item: item[0])
            selection[0] = {
                "ok": True,
                "receipt": receipt,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "path": str(path.resolve(strict=True)),
                "created_unix_ns": int(receipt["created_unix_ns"]),
            }
        except Exception as exc:
            selection[0] = {
                "ok": False,
                "type": type(exc).__name__,
                "error": str(exc),
            }
    dist.broadcast_object_list(selection, src=0)
    if not selection[0]["ok"]:
        raise PretrainError(f"resource preflight receipt failed: {selection[0]}")
    identity_error: Exception | None = None
    identity: dict[str, Any] | None = None
    try:
        identity = current_rank_identity(int(os.environ["LOCAL_RANK"]))
    except Exception as exc:
        identity_error = exc
    local = {
        "identity": identity,
        "error": None if identity_error is None else str(identity_error),
    }
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    failures = [item["error"] for item in gathered if item.get("error") is not None]
    if failures:
        raise PretrainError(f"current rank identity probe failed: {failures}")
    validate_current_rank_identities(
        selection[0]["receipt"], [item["identity"] for item in gathered]
    )
    return {
        "path": str(selection[0]["path"]),
        "sha256": str(selection[0]["sha256"]),
        "created_unix_ns": int(selection[0]["created_unix_ns"]),
    }


def _checkpoint_steps(train: Mapping[str, Any]) -> set[int]:
    total = int(train["total_steps"])
    interval = int(train["checkpoint_interval"])
    result = {int(value) for value in train["checkpoint_steps"]}
    result.update(range(interval, total + 1, interval))
    result.add(total)
    if any(step <= 0 or step > total for step in result):
        raise PretrainError("checkpoint step lies outside training range")
    return result


def _learning_rate(step: int, runtime: Mapping[str, Any]) -> float:
    train = runtime["train"]
    optimizer = runtime["optimizer"]
    schedule = runtime["schedule"]
    total = int(train["total_steps"])
    warmup = int(schedule["warmup_steps"])
    peak = float(optimizer["peak_lr"])
    minimum = float(optimizer["min_lr"])
    start = float(optimizer.get("start_lr", 0.0))
    if step < warmup:
        progress = float(step + 1) / float(max(1, warmup))
        return start + (peak - start) * progress
    remaining = total - warmup
    stable = int(round(remaining * float(schedule["stable_fraction"])))
    decay_start = warmup + stable
    if step < decay_start:
        return peak
    progress = min(1.0, (step - decay_start) / max(1, total - decay_start))
    return minimum + (peak - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


_MODEL_INPUTS = {
    "world_tokens",
    "view_mask",
    "world_times_s",
    "task_embedding",
    "history_fine_action_values",
    "history_fine_action_mask",
    "history_fine_action_dt",
    "history_fine_sample_mask",
    "history_coarse_action_values",
    "history_coarse_action_mask",
    "future_factual_fine_action_values",
    "future_factual_fine_action_mask",
    "future_factual_fine_action_dt",
    "future_factual_fine_sample_mask",
    "future_factual_coarse_action_values",
    "future_factual_coarse_action_mask",
    "action_group_ids",
    "action_group_mask",
    "action_semantic_ids",
    "current_state_values",
    "current_state_mask",
    "state_semantic_ids",
    "embodiment_ids",
    "policy_query_dt",
    "policy_query_mask",
    "action_normalization_offset",
    "action_normalization_scale",
    "aux_values",
    "aux_mask",
    "aux_type_ids",
}

_APPEARANCE_MODEL_INPUTS = (
    "context_rgb",
    "context_rgb_mask",
    "appearance_context_tokens",
    "appearance_context_mask",
    "target_appearance_tokens",
    "target_appearance_mask",
)


def _relative_world_times_for_model(
    world_times_s: torch.Tensor, *, context_length: int
) -> torch.Tensor:
    """Recenter timestamps before FSDP mixed precision casts model inputs.

    Absolute episode times can be hundreds of seconds. Casting those values to
    BF16 at the root FSDP boundary can collapse distinct observed timestamps.
    The model consumes time relative to the last context frame, so performing
    that invariant transformation in the source dtype preserves the real
    intervals without changing model semantics.
    """

    if world_times_s.ndim != 2:
        raise PretrainError("world_times_s must be a batched matrix")
    if context_length <= 0 or context_length > int(world_times_s.shape[1]):
        raise PretrainError("world context length is incompatible with timestamps")
    if not bool(torch.isfinite(world_times_s).all()):
        raise PretrainError("world_times_s contains non-finite values")
    if not bool(torch.diff(world_times_s, dim=1).gt(0).all()):
        raise PretrainError("world_times_s must be strictly increasing per sample")
    relative = world_times_s - world_times_s[:, context_length - 1 : context_length]
    # Recenter in the source precision first, then present FP32 to the model.
    # FSDP casts root inputs itself, while DDP relies on autocast, which does
    # not convert FP64 inputs for BF16 linear layers.
    return relative.to(dtype=torch.float32)


def _appearance_teacher_ratio(step: int, runtime: Mapping[str, Any]) -> float:
    train = runtime["train"]
    fields = (
        "appearance_teacher_start_ratio",
        "appearance_teacher_end_ratio",
        "appearance_teacher_decay_steps",
    )
    if not all(name in train for name in fields):
        return 0.0
    start = float(train[fields[0]])
    end = float(train[fields[1]])
    progress = min(1.0, max(0.0, float(step) / float(train[fields[2]])))
    return start + (end - start) * progress


def _training_appearance_teacher_ratio(
    step: int, runtime: Mapping[str, Any]
) -> float:
    """Keep the stable teacher schedule while training the inference endpoint.

    A full teacher decay over a short canary damaged RGB quality, while the
    original 10k schedule gave the teacher-free renderer almost no early
    gradient. A deterministic, globally shared teacher-free step trains the
    exact inference path without adding a second decoder forward or changing
    the scheduled batches.
    """

    ratio = _appearance_teacher_ratio(step, runtime)
    every = runtime["train"].get("appearance_teacher0_every_steps")
    if every is not None and (int(step) + 1) % int(every) == 0:
        return 0.0
    return ratio


def _forward(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    appearance_teacher_ratio: float = 0.0,
    compute_zero_action_control: bool = False,
) -> Mapping[str, torch.Tensor]:
    indices = batch["rgb_frame_indices"]
    if indices.ndim != 2 or not bool((indices == indices[:1]).all()):
        raise PretrainError("RGB supervision indices drifted within a batch")
    kwargs = {name: batch[name] for name in _MODEL_INPUTS}
    kwargs["rgb_frame_indices"] = indices[0].tolist()
    kwargs["world_times_s"] = _relative_world_times_for_model(
        kwargs["world_times_s"],
        context_length=int(kwargs["world_tokens"].shape[1]),
    )
    if "target_rgb_mask" in batch:
        rgb_mask = batch["target_rgb_mask"]
        if rgb_mask.ndim != 6 or tuple(rgb_mask.shape[-3:]) != (1, 1, 1):
            raise PretrainError("target_rgb_mask must be [B,F,V,1,1,1]")
        kwargs["rgb_view_mask"] = rgb_mask[..., 0, 0, 0]
    for name in _APPEARANCE_MODEL_INPUTS:
        if name in batch:
            kwargs[name] = batch[name]
    if "appearance_context_tokens" in batch:
        kwargs["appearance_teacher_ratio"] = appearance_teacher_ratio
    if compute_zero_action_control:
        kwargs["compute_zero_action_control"] = True
    return model(**kwargs)


def _action_counterfactual_enabled(objective: Any) -> bool:
    return bool(
        objective.action_counterfactual_token_advantage > 0.0
        or objective.action_counterfactual_rgb_advantage > 0.0
    )


def _zero_future_factual_action(
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    zero_batch = dict(batch)
    for name in (
        "future_factual_fine_action_values",
        "future_factual_coarse_action_values",
    ):
        if name not in batch:
            raise PretrainError(f"action counterfactual requires {name}")
        zero_batch[name] = torch.zeros_like(batch[name])
    return zero_batch


def _forward_with_action_counterfactual(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    appearance_teacher_ratio: float,
    objective: Any,
) -> Mapping[str, torch.Tensor]:
    output = dict(
        _forward(
            model,
            batch,
            appearance_teacher_ratio=appearance_teacher_ratio,
            compute_zero_action_control=(
                objective.action_counterfactual_token_advantage > 0.0
            ),
        )
    )
    if not _action_counterfactual_enabled(objective):
        return output
    if objective.action_counterfactual_token_advantage > 0.0:
        if "zero_action_pred_tokens" not in output:
            raise PretrainError("model did not return the zero-action token control")
    if objective.action_counterfactual_rgb_advantage > 0.0:
        with torch.no_grad():
            zero_output = _forward(
                model,
                _zero_future_factual_action(batch),
                appearance_teacher_ratio=appearance_teacher_ratio,
            )
        output["zero_action_rgb"] = zero_output["rgb"].detach()
    return output


def _build_mixed_dataset(
    runtime: Mapping[str, Any],
    *,
    split: str,
    profile: Any | None = None,
    device: torch.device | None = None,
    rank: int = 0,
) -> tuple[Any, Any]:
    closure = runtime["data_closure"]
    if closure.get("schema") == FORMAL_CACHE_CLOSURE_SCHEMA:
        return build_formal_cache_dataset(runtime, split=split, profile=profile)
    if profile is None:
        profile = load_data_profile(
            Path(closure["data_profile_path"]), verify_source_manifests=True
        )
    normalization_model_sha = runtime["bindings"]["model_profile_sha256"]
    if closure.get("schema") in {
        STREAMING_DATA_CLOSURE_SCHEMA,
        DIRECT_RAW_DATA_CLOSURE_SCHEMA,
    }:
        metadata_seal = load_streaming_metadata_seal(
            Path(str(closure["metadata_seal_path"])),
            expected_sha256=str(closure["metadata_seal_sha256"]),
        )
        normalization_model_sha = str(metadata_seal["model_profile_sha256"])
    normalizer = GroupedRobotNormalizer.load(
        Path(closure["grouped_normalization_path"]),
        expected_sha256=closure["grouped_normalization_sha256"],
        expected_data_profile_sha256=closure["data_profile_sha256"],
        expected_model_profile_sha256=normalization_model_sha,
        expected_window_index_sha256=closure["cache_index_sha256"],
        data_profile=profile,
    )
    if closure.get("schema") == DIRECT_RAW_DATA_CLOSURE_SCHEMA:
        dataset = DirectRawDataset(
            closure=closure,
            data_profile=profile,
            model_profile=runtime["model_profile"],
            split=split,
            grouped_normalizer=normalizer,
            rank=rank,
        )
        return dataset, profile
    if closure.get("schema") == STREAMING_DATA_CLOSURE_SCHEMA:
        if device is None:
            raise PretrainError("streaming_raw dataset requires the current rank device")
        dataset = StreamingRawDataset(
            closure=closure,
            data_profile=profile,
            model_profile=runtime["model_profile"],
            split=split,
            grouped_normalizer=normalizer,
            device=device,
            rank=rank,
        )
        return dataset, profile
    dataset = UnifiedCacheDataset(
        cache_root=Path(closure["cache_root"]),
        index_path=Path(closure["cache_index_path"]),
        index_sha256=closure["cache_index_sha256"],
        data_profile=profile,
        model_profile=runtime["model_profile"],
        split=split,
        verify_shard_sha_on_open=True,
        grouped_normalizer=normalizer,
    )
    return dataset, profile


def _last_true_extent(mask: torch.Tensor, axis: int) -> int:
    positions = torch.nonzero(mask, as_tuple=False)
    if positions.numel() == 0:
        return 1
    return int(positions[:, axis].max().item()) + 1


def _collate_and_trim(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack one batch, then remove storage-only action padding.

    Cache shards use fixed capacities so they can be memory-mapped and batched.
    Compute uses only the largest real S/C in this batch, so a profile capable
    of 50Hz/100Hz sources does not impose those lengths on slower sources.
    """

    batch = default_collate(samples)
    if not isinstance(batch, dict):
        raise PretrainError("cache collate must produce a mapping")
    sample_masks = (
        batch["history_fine_sample_mask"],
        batch["future_factual_fine_sample_mask"],
    )
    substeps = max(_last_true_extent(mask, axis=3) for mask in sample_masks)
    for name in (
        "history_fine_action_values",
        "history_fine_action_mask",
        "history_fine_action_dt",
        "history_fine_sample_mask",
        "future_factual_fine_action_values",
        "future_factual_fine_action_mask",
        "future_factual_fine_action_dt",
        "future_factual_fine_sample_mask",
    ):
        # Values/masks are [B,F,G,S,A], dt/sample masks are [B,F,G,S].
        batch[name] = batch[name][..., :substeps, :] if batch[name].ndim == 5 else batch[name][..., :substeps]

    queries = _last_true_extent(batch["policy_query_mask"], axis=2)
    for name in (
        "policy_query_dt",
        "policy_query_mask",
        "target_fine_action",
        "target_fine_action_mask",
    ):
        # Query tensors are group-major [B,G,C,(A)].
        batch[name] = batch[name][..., :queries, :] if batch[name].ndim == 4 else batch[name][..., :queries]
    return batch


class _StreamingLookaheadBatchSampler:
    """Prime bounded raw CPU preparation before DataLoader fetches a batch."""

    def __init__(self, sampler: Any, dataset: Any, *, lookahead_batches: int = 2):
        if lookahead_batches <= 0:
            raise ValueError("lookahead_batches must be positive")
        self.sampler = sampler
        self.dataset = dataset
        self.lookahead_batches = int(lookahead_batches)

    def __len__(self) -> int:
        return len(self.sampler)

    def __iter__(self):
        iterator = iter(self.sampler)
        pending: deque[list[int]] = deque()
        for _ in range(self.lookahead_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            pending.append(batch)
            self.dataset.prefetch_indices(batch)
        while pending:
            current = pending.popleft()
            # Resume here only after DataLoader has consumed the current batch.
            # Direct/streaming datasets retain one future per window, so
            # replenishing before the yield sees the old batch still resident
            # and periodically rejects the next batch at bounded capacity.
            yield current
            try:
                future = next(iterator)
            except StopIteration:
                future = None
            if future is not None:
                pending.append(future)
                self.dataset.prefetch_indices(future)


def _make_loader(
    dataset: Any,
    profile: Any,
    runtime_profile: Mapping[str, Any],
    *,
    rank: int,
    world_size: int,
    start_step: int,
    num_steps: int,
    seed: int,
    gradient_accumulation: int,
    micro_batch_size: int | None = None,
) -> DataLoader:
    train = runtime_profile["train"]
    sampler = StepAddressedBatchSampler(
        dataset.source_spans,
        dataset.source_names,
        {
            name: profile.source_weights[name]
            for name in dataset.source_names
        },
        world_size=world_size,
        rank=rank,
        micro_batch_size=(
            int(train["micro_batch_size"])
            if micro_batch_size is None
            else int(micro_batch_size)
        ),
        gradient_accumulation=gradient_accumulation,
        start_optimizer_step=start_step,
        num_optimizer_steps=num_steps,
        seed=seed,
        source_episode_spans=getattr(dataset, "source_episode_spans", None),
    )
    if callable(getattr(dataset, "prefetch_indices", None)):
        sampler = _StreamingLookaheadBatchSampler(
            sampler, dataset, lookahead_batches=2
        )
    workers = (
        0
        if bool(getattr(dataset, "requires_main_process", False))
        else int(train["num_workers"])
    )
    kwargs: dict[str, Any] = {
        "batch_sampler": sampler,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": bool(train["persistent_workers"]) and workers > 0,
        "collate_fn": _collate_and_trim,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(train["prefetch_factor"])
    return DataLoader(dataset, **kwargs)


def _require_sampling_capacity(
    dataset: Any,
    profile: Any,
    runtime_profile: Mapping[str, Any],
    *,
    rank: int,
    world_size: int,
    gradient_accumulation: int,
    seed: int,
    micro_batch_size: int | None = None,
) -> None:
    """Reject undersized source splits before model construction or training."""

    _make_loader(
        dataset,
        profile,
        runtime_profile,
        rank=rank,
        world_size=world_size,
        start_step=0,
        num_steps=1,
        seed=seed,
        gradient_accumulation=gradient_accumulation,
        micro_batch_size=micro_batch_size,
    )


def _validation_micro_batch_size(runtime_profile: Mapping[str, Any]) -> int:
    train = runtime_profile["train"]
    return int(train.get("validation_micro_batch_size", train["micro_batch_size"]))


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    dataset: Any,
    profile: Any,
    runtime_profile: Mapping[str, Any],
    objective: Any,
    perceptual_model: torch.nn.Module | None,
    context: Any,
    input_adapter: Any | None = None,
    training_step: int = 0,
) -> dict[str, Any]:
    count = int(runtime_profile["train"]["validation_steps"])
    loader = _make_loader(
        dataset,
        profile,
        runtime_profile,
        rank=context.rank,
        world_size=context.world_size,
        start_step=0,
        num_steps=count,
        seed=int(runtime_profile["train"]["validation_seed"]),
        gradient_accumulation=1,
        micro_batch_size=_validation_micro_batch_size(runtime_profile),
    )
    settings = [("teacher0", 0.0)]
    if bool(runtime_profile["train"].get("appearance_validation_three_way", False)):
        settings.extend(
            (
                (
                    "scheduled",
                    _appearance_teacher_ratio(training_step, runtime_profile),
                ),
                ("teacher1", 1.0),
            )
        )
    totals: dict[str, dict[str, torch.Tensor]] = {
        label: {} for label, _ratio in settings
    }
    model.eval()
    for cpu_batch in loader:
        if input_adapter is not None:
            cpu_batch = input_adapter.materialize(cpu_batch)
        batch = _batch_to_device(cpu_batch, context.device)
        for label, ratio in settings:
            with autocast_context(strategy_from_mapping(runtime_profile["distributed"])):
                losses = compute_native_objective(
                    output=_forward_with_action_counterfactual(
                        model,
                        batch,
                        appearance_teacher_ratio=ratio,
                        objective=objective,
                    ),
                    batch=batch,
                    config=objective,
                    perceptual_model=perceptual_model,
                    rgb_perceptual_chunk_size=int(
                        runtime_profile["train"].get("rgb_perceptual_chunk_size", 4)
                    ),
                )
            for name, value in losses.items():
                totals[label][name] = (
                    totals[label].get(name, torch.zeros_like(value)) + value
                )
    model.train()
    reduced = {
        label: reduce_metrics(
            {name: value / count for name, value in label_totals.items()}
        )
        for label, label_totals in totals.items()
    }
    result: dict[str, Any] = dict(reduced["teacher0"])
    if len(settings) > 1:
        result["appearance_three_way"] = {
            label: {"teacher_ratio": ratio, "metrics": reduced[label]}
            for label, ratio in settings
        }
    return result


def _run_contract(
    config: Mapping[str, Any],
    parameter_counts: Mapping[str, int],
    native_model: NativeWorldModel,
    warmstart_model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = {
        "schema": RUN_CONTRACT_SCHEMA,
        "name": config["run"]["name"],
        "lineage": config["run"]["lineage"],
        "data_closure_sha256": config["bindings"]["data_closure_sha256"],
        "model_contract_sha256": config["bindings"]["model_contract_sha256"],
        "code_commit": config["run"]["code_commit"],
        "environment_lock_sha256": config["run"]["environment_lock_sha256"],
        "topology_contract_sha256": _topology_contract_sha256(config),
        "resource_contract_sha256": resource_contract_sha256(
            config["runtime_profile"].get("resources")
        ),
        "parameter_counts": dict(parameter_counts),
        "required_gradient_owners": list(required_gradient_owner_names(native_model)),
    }
    if warmstart_model is not None:
        contract["warmstart_model"] = dict(warmstart_model)
    return contract


def _resume_expectations(
    config: Mapping[str, Any], config_sha: str, *, step: int, world_size: int
) -> ResumeExpectations:
    runtime = config["runtime_profile"]
    distributed = runtime["distributed"]
    return ResumeExpectations(
        step=int(step),
        run_lineage=config["run"]["lineage"],
        runtime_config_sha256=config_sha,
        data_closure_sha256=config["bindings"]["data_closure_sha256"],
        model_contract_sha256=config["bindings"]["model_contract_sha256"],
        world_size=int(world_size),
        shard_degree=int(distributed["shard_degree"]),
        distributed_strategy=str(distributed["strategy"]),
        global_batch_size=int(runtime["train"]["global_batch_size"]),
        topology_contract_sha256=_topology_contract_sha256(config),
        allow_topology_reshard=bool(runtime["checkpoint"]["allow_topology_reshard"]),
    )


def _rank_identities(context: Any) -> list[dict[str, Any]]:
    local: dict[str, Any] | None = None
    error: str | None = None
    try:
        current = current_rank_identity(int(context.local_rank))
        local = {"rank": int(context.rank), **current}
    except Exception as exc:
        error = str(exc)
    gathered: list[Any] = [None] * int(context.world_size)
    dist.all_gather_object(gathered, {"identity": local, "error": error})
    failures = [value["error"] for value in gathered if value.get("error")]
    if failures:
        raise PretrainError(f"current rank identity probe failed: {failures}")
    identities = [value["identity"] for value in gathered]
    if [value.get("rank") for value in identities] != list(range(context.world_size)):
        raise PretrainError("current rank identity closure is invalid")
    return identities


def _publish_and_validate_launch(
    *,
    config: Mapping[str, Any],
    config_sha: str,
    context: Any,
    strategy: Any,
    run_contract: Mapping[str, Any],
    resource_preflight: Mapping[str, Any] | None,
    source_checkpoint: Mapping[str, Any] | None,
    launch_kind: str,
    resource_runtime_config_sha256: str | None = None,
) -> tuple[str, str]:
    identities = _rank_identities(context)
    resources = config["runtime_profile"].get("resources")
    output_root = Path(config["run"]["output_root"])
    publication: list[Any] = [None]
    if context.is_rank0:
        try:
            value = build_launch_qualification(
                launch_kind=launch_kind,
                runtime_config_sha256=config_sha,
                run_contract=run_contract,
                resources=resources,
                resource_preflight=resource_preflight,
                rank_identities=identities,
                world_size=context.world_size,
                local_world_size=context.local_world_size,
                distributed_strategy=strategy.strategy,
                shard_degree=strategy.shard_degree,
                source_checkpoint=source_checkpoint,
            )
            path, digest = publish_launch_qualification(output_root, value)
            publication[0] = {
                "ok": True,
                "path": str(path.resolve(strict=True)),
                "sha256": digest,
                "qualification": value,
            }
        except Exception as exc:
            publication[0] = {
                "ok": False,
                "type": type(exc).__name__,
                "error": str(exc),
            }
    dist.broadcast_object_list(publication, src=0)
    if not publication[0]["ok"]:
        raise PretrainError(f"launch qualification publication failed: {publication[0]}")
    try:
        # Training nodes have independent local filesystems.  Rank 0 keeps the
        # durable receipt, while every rank validates the exact value carried
        # by the existing distributed broadcast instead of reopening rank 0's
        # local pathname.
        value = publication[0]["qualification"]
        validate_launch_qualification(
            value,
            launch_kind=launch_kind,
            runtime_config_sha256=config_sha,
            run_contract=run_contract,
            resources=resources,
            rank_identities=identities,
            world_size=context.world_size,
            local_world_size=context.local_world_size,
            distributed_strategy=strategy.strategy,
            shard_degree=strategy.shard_degree,
            source_checkpoint=source_checkpoint,
            resource_runtime_config_sha256=resource_runtime_config_sha256,
        )
    except LaunchQualificationError as exc:
        raise PretrainError("launch qualification validation failed") from exc
    return str(publication[0]["path"]), str(publication[0]["sha256"])


def _topology_contract_sha256(config: Mapping[str, Any]) -> str:
    """Hash training semantics that must survive a topology reshard.

    Launch-only values (world size, local micro-batch, accumulation, workers,
    checkpoint cadence) are deliberately excluded.  The effective global
    batch, sampling seed, objective, optimizer/schedule, numerical dtypes,
    code, environment, model, and data remain immutable.
    """

    runtime = config["runtime_profile"]
    train = runtime["train"]
    distributed = runtime["distributed"]
    return canonical_sha256(
        {
            "run_lineage": config["run"]["lineage"],
            "code_commit": config["run"]["code_commit"],
            "environment_lock_sha256": config["run"]["environment_lock_sha256"],
            "model_contract_sha256": config["bindings"]["model_contract_sha256"],
            "data_closure_sha256": config["bindings"]["data_closure_sha256"],
            "objective_profile_sha256": config["bindings"]["objective_profile_sha256"],
            "resource_contract": runtime.get("resources"),
            "optimizer": runtime["optimizer"],
            "schedule": runtime["schedule"],
            "train": {
                "global_batch_size": int(train["global_batch_size"]),
                "total_steps": int(train["total_steps"]),
                "seed": int(train["seed"]),
                "gradient_clip": float(train["gradient_clip"]),
            },
            "distributed_numerics": {
                "strategy": distributed["strategy"],
                "shard_degree": int(distributed["shard_degree"]),
                "param_dtype": distributed["param_dtype"],
                "reduce_dtype": distributed["reduce_dtype"],
                "output_dtype": distributed["output_dtype"],
            },
        }
    )


def _restore_gradient_ownership(
    metadata: Mapping[str, Any], native_model: NativeWorldModel
) -> Mapping[str, Any]:
    saved = metadata.get("gradient_ownership")
    if not isinstance(saved, Mapping):
        raise PretrainError(
            "resume checkpoint lacks a passed Stage0 gradient-ownership audit"
        )
    try:
        validate_gradient_ownership_receipt(saved, native_model)
    except GradientOwnershipError as exc:
        raise PretrainError(
            "resume checkpoint gradient-ownership audit is invalid"
        ) from exc
    return saved


def main() -> None:
    args = parse_args()
    config, config_sha = load_materialized_runtime(args.runtime)
    runtime = config["runtime_profile"]
    strategy = strategy_from_mapping(runtime["distributed"])
    context = initialize_distributed(strategy)
    async_pipeline: AsyncCudaInputPipeline | None = None
    try:
        if int(runtime["expected_world_size"]) != context.world_size:
            raise PretrainError(
                f"WORLD_SIZE={context.world_size} != {runtime['expected_world_size']}"
            )
        preflight_result_sha256 = None
        resource_preflight = None
        if not args.preflight_only:
            resource_preflight = _require_recent_resource_preflight(
                config, config_sha, context
            )
        repo = Path(__file__).resolve().parents[2]
        try:
            verify_clean_runtime_checkout(
                repo,
                str(config["run"]["code_commit"]),
                allow_commit_mismatch=_environment_flag(
                    "WM3D_EXECUTION_HOTFIX"
                ),
            )
        except LaunchQualificationError as exc:
            raise PretrainError("runtime code provenance failed") from exc
        train_dataset, profile = _build_mixed_dataset(
            config, split="train", device=context.device, rank=context.rank
        )
        _require_sampling_capacity(
            train_dataset,
            profile,
            runtime,
            rank=context.rank,
            world_size=context.world_size,
            gradient_accumulation=int(runtime["train"]["gradient_accumulation"]),
            seed=int(runtime["train"]["seed"]),
        )
        validation_dataset: UnifiedCacheDataset | None = None
        if int(runtime["train"]["validate_every"]) > 0:
            validation_dataset, _ = _build_mixed_dataset(
                config,
                split="val",
                profile=profile,
                device=context.device,
                rank=context.rank,
            )
            _require_sampling_capacity(
                validation_dataset,
                profile,
                runtime,
                rank=context.rank,
                world_size=context.world_size,
                gradient_accumulation=1,
                seed=int(runtime["train"]["validation_seed"]),
                micro_batch_size=_validation_micro_batch_size(runtime),
            )
        model_cfg = config["model_profile"]["model"]
        cache_representation = profile.cache_representation
        if int(cache_representation["spatial_tokens"]) < int(model_cfg["P"]):
            raise PretrainError(
                "model P exceeds the shared episode-cache representation"
            )
        for name in ("token_dim", "num_views"):
            if int(cache_representation[name]) != int(model_cfg[name]):
                raise PretrainError(
                    f"model/cache {name} mismatch: "
                    f"{model_cfg[name]} != {cache_representation[name]}"
                )
        if int(cache_representation["rgb_size"]) < int(model_cfg["rgb_size"]):
            raise PretrainError("model RGB size exceeds cached RGB representation")
        if args.preflight_only:
            # Qualify resources after the potentially long full-corpus data
            # validation so the freshness window starts when preflight is
            # actually ready to hand off to training.
            preflight_result_sha256 = _resource_preflight(
                config, config_sha, context
            )
            if context.is_rank0:
                print(
                    json.dumps(
                        {
                            "passed": True,
                            "runtime_sha256": config_sha,
                            "world_size": context.world_size,
                            "train_windows": len(train_dataset),
                            "validation_windows": (
                                None
                                if validation_dataset is None
                                else len(validation_dataset)
                            ),
                            "sources": profile.source_order,
                            "resource_preflight_sha256": preflight_result_sha256,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            return

        input_adapter = None
        if config["data_closure"].get("schema") == DIRECT_RAW_DATA_CLOSURE_SCHEMA:
            input_adapter = build_direct_vggt_teacher(
                config, device=context.device
            )
            input_adapter.eval()

        seed = int(runtime["train"]["seed"])
        _configure_reproducibility(
            seed,
            cudnn_benchmark=bool(
                runtime["train"].get("cudnn_benchmark", False)
            )
            or _environment_flag("WM3D_CUDNN_BENCHMARK"),
        )
        # FSDP2 constructs on meta and initializes global DTensor shards after
        # wrapping.  No rank ever owns a full 5B fp32 CPU/GPU replica.
        construction_device = (
            torch.device("meta")
            if strategy.initialization == "meta_sharded"
            else context.device
        )
        with torch.device(construction_device):
            effective_model_profile = dict(config["model_profile"])
            effective_model = dict(effective_model_profile["model"])
            effective_model["activation_checkpointing"] = bool(
                runtime["train"].get(
                    "activation_checkpointing",
                    effective_model["activation_checkpointing"],
                )
            )
            effective_model["rgb_decode_chunk_size"] = int(
                runtime["train"].get(
                    "rgb_decode_chunk_size",
                    effective_model["rgb_decode_chunk_size"],
                )
            )
            effective_model_profile["model"] = effective_model
            model = build_world_model(effective_model_profile)
        if not isinstance(model, NativeWorldModel):
            raise PretrainError("unified pretrain currently requires native_world_model")
        native_model = model
        parameter_counts = native_model.parameter_counts()
        wrapped = wrap_model(
            model,
            context,
            strategy,
            initialization_seed=(
                seed if strategy.initialization == "meta_sharded" else None
            ),
        )
        model = wrapped.model
        if args.resume is not None and "model_warmstart_checkpoint" in runtime["train"]:
            raise PretrainError("exact resume and model warmstart are mutually exclusive")
        warmstart_model: Mapping[str, Any] | None = None
        warmstart_path = runtime["train"].get("model_warmstart_checkpoint")
        if warmstart_path is not None:
            source_path = Path(str(warmstart_path)).resolve(strict=True)
            source_manager = DistributedCheckpointManager(source_path.parent)
            prefixes = tuple(
                str(value)
                for value in runtime["train"][
                    "model_warmstart_new_parameter_prefixes"
                ]
            )
            warmstart_model = source_manager.load_model_warmstart(
                path=source_path,
                model=model,
                new_parameter_prefixes=prefixes,
            )
        optimizer_cfg = runtime["optimizer"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_cfg["peak_lr"]),
            betas=tuple(float(value) for value in optimizer_cfg["betas"]),
            eps=float(optimizer_cfg["eps"]),
            weight_decay=float(optimizer_cfg["weight_decay"]),
            foreach=False,
        )
        initialize_adamw_state(optimizer)
        objective = objective_config_from_mapping(config["objective_profile"]["objective"])
        perceptual_model = build_rgb_perceptual_model(
            objective, device=context.device
        )

        output_root = Path(config["run"]["output_root"])
        status: list[Any] = [None]
        if context.is_rank0:
            try:
                output_root.mkdir(parents=True, exist_ok=True)
                if output_root.is_symlink():
                    raise PretrainError("output root cannot be a symlink")
                contract = _run_contract(
                    config,
                    parameter_counts,
                    native_model,
                    warmstart_model=warmstart_model,
                )
                _atomic_json_no_clobber(output_root / "run_contract.json", contract)
                status[0] = {"ok": True, "contract": contract}
            except Exception as exc:
                status[0] = {"ok": False, "type": type(exc).__name__, "error": str(exc)}
        dist.broadcast_object_list(status, src=0)
        if not status[0]["ok"]:
            raise PretrainError(f"run contract failed: {status[0]}")

        manager = DistributedCheckpointManager(output_root / "checkpoints")
        start_step = 0
        gradient_ownership: Mapping[str, Any] | None = None
        source_checkpoint: Mapping[str, Any] | None = None
        expectations: ResumeExpectations | None = None
        if args.resume is not None:
            if re.fullmatch(r"step_[0-9]{8}", args.resume.name) is None:
                raise PretrainError("resume must be an explicit step_XXXXXXXX directory")
            expected_step = int(args.resume.name.split("_")[1])
            expectations = _resume_expectations(
                config,
                config_sha,
                step=expected_step,
                world_size=context.world_size,
            )
            inspection: list[Any] = [None]
            if context.is_rank0:
                try:
                    inspection[0] = {
                        "ok": True,
                        "source": manager.inspect_committed(
                            path=args.resume, expected=expectations
                        ),
                    }
                except Exception as exc:
                    inspection[0] = {
                        "ok": False,
                        "type": type(exc).__name__,
                        "error": str(exc),
                    }
            dist.broadcast_object_list(inspection, src=0)
            if not inspection[0]["ok"]:
                raise PretrainError(
                    f"resume checkpoint inspection failed: {inspection[0]}"
                )
            source_checkpoint = inspection[0]["source"]
        launch_kind = (
            "fresh"
            if source_checkpoint is None
            else (
                "exact_resume"
                if source_checkpoint["resume_mode"] == "exact"
                else "topology_reshard"
            )
        )
        launch_qualification_path, launch_qualification_sha256 = (
            _publish_and_validate_launch(
                config=config,
                config_sha=config_sha,
                context=context,
                strategy=strategy,
                run_contract=status[0]["contract"],
                resource_preflight=resource_preflight,
                source_checkpoint=source_checkpoint,
                launch_kind=launch_kind,
            )
        )
        if args.resume is not None:
            assert expectations is not None
            metadata, progress = manager.load(
                path=args.resume,
                model=model,
                optimizer=optimizer,
                expected=expectations,
            )
            if int(progress.get("next_optimizer_step", -1)) != expected_step:
                raise PretrainError("checkpoint sampler progress is not exact")
            start_step = int(metadata["step"])
            gradient_ownership = _restore_gradient_ownership(metadata, native_model)
        total_steps = int(runtime["train"]["total_steps"])
        if start_step >= total_steps:
            raise PretrainError("checkpoint already reached total_steps")
        stop_after_step = (
            total_steps if args.stop_after_step is None else int(args.stop_after_step)
        )
        if not start_step < stop_after_step <= total_steps:
            raise PretrainError(
                "--stop-after-step must be greater than the resume step and no "
                "larger than sealed total_steps"
            )
        loader = _make_loader(
            train_dataset,
            profile,
            runtime,
            rank=context.rank,
            world_size=context.world_size,
            start_step=start_step,
            num_steps=stop_after_step - start_step,
            seed=seed,
            gradient_accumulation=int(runtime["train"]["gradient_accumulation"]),
        )
        iterator = iter(loader)
        if input_adapter is not None and _environment_flag(
            "WM3D_DIRECT_ASYNC_PIPELINE"
        ):
            async_pipeline = AsyncCudaInputPipeline(
                iterator=iterator,
                adapter=input_adapter,
                transfer=_batch_to_device,
                device=context.device,
            )
            async_pipeline.submit()
        validate_every = int(runtime["train"]["validate_every"])
        checkpoint_steps = _checkpoint_steps(runtime["train"])
        if stop_after_step not in checkpoint_steps:
            raise PretrainError(
                "--stop-after-step must be a sealed checkpoint step so the "
                "stopped run is exactly resumable"
            )
        log_path = output_root / "train_metrics.jsonl"
        accumulation = int(runtime["train"]["gradient_accumulation"])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        last_log = time.monotonic()
        for step in range(start_step, stop_after_step):
            lr = _learning_rate(step, runtime)
            for group in optimizer.param_groups:
                group["lr"] = lr
            accumulated: dict[str, torch.Tensor] = {}
            source_id: int | None = None
            completed_candidate = step + 1
            serialize_after_step = (
                completed_candidate in checkpoint_steps
                or completed_candidate == stop_after_step
                or (
                    validate_every > 0
                    and completed_candidate % validate_every == 0
                )
            )
            for micro_step in range(accumulation):
                final_micro = micro_step == accumulation - 1
                if async_pipeline is None:
                    cpu_batch = next(iterator)
                    if input_adapter is not None:
                        cpu_batch = input_adapter.materialize(cpu_batch)
                    batch = _batch_to_device(cpu_batch, context.device)
                else:
                    batch = async_pipeline.consume()
                unique = torch.unique(batch["source_id"])
                if unique.numel() != 1:
                    raise PretrainError("one micro-batch mixed multiple sources")
                current_source = int(unique.item())
                if source_id is None:
                    source_id = current_source
                elif source_id != current_source:
                    raise PretrainError("one optimizer step mixed multiple sources")
                if async_pipeline is not None and not (
                    final_micro and serialize_after_step
                ):
                    async_pipeline.submit()
                with no_sync_context(model, enabled=not final_micro):
                    with autocast_context(strategy):
                        losses = compute_native_objective(
                            output=_forward_with_action_counterfactual(
                                model,
                                batch,
                                appearance_teacher_ratio=_training_appearance_teacher_ratio(
                                    step, runtime
                                ),
                                objective=objective,
                            ),
                            batch=batch,
                            config=objective,
                            perceptual_model=perceptual_model,
                            rgb_perceptual_chunk_size=int(
                                runtime["train"].get("rgb_perceptual_chunk_size", 4)
                            ),
                        )
                    (losses["total"] / accumulation).backward()
                for name, value in losses.items():
                    accumulated[name] = (
                        accumulated.get(name, torch.zeros_like(value))
                        + value.detach() / accumulation
                    )
            completed = step + 1
            # Audit before clipping/zeroing.  Fresh runs must prove all owners
            # on their first real optimizer step; resumed runs inherit the
            # immutable proof stored in the committed checkpoint metadata.
            if gradient_ownership is None:
                gradient_ownership = audit_gradient_ownership(native_model)
                if context.is_rank0:
                    _atomic_json_no_clobber(
                        output_root / "gradient_ownership.json",
                        gradient_ownership,
                    )
                    print(
                        json.dumps(
                            {"step": completed, "gradient_ownership": gradient_ownership},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            clip_value = float(runtime["train"]["gradient_clip"])
            if hasattr(model, "clip_grad_norm_"):
                grad_norm = model.clip_grad_norm_(clip_value)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            grad_norm_value = grad_norm.full_tensor() if hasattr(grad_norm, "full_tensor") else grad_norm
            if not bool(torch.isfinite(grad_norm_value).all()):
                raise FloatingPointError(f"non-finite gradient norm at step {step}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if completed % int(runtime["train"]["log_every"]) == 0 or completed == stop_after_step:
                metrics = reduce_metrics(accumulated)
                grad_tensor = grad_norm_value.detach().float().reshape(-1).mean()
                dist.all_reduce(grad_tensor, op=dist.ReduceOp.SUM)
                grad_tensor.div_(context.world_size)
                if context.is_rank0:
                    record = {
                        "step": completed,
                        "lr": lr,
                        "source_id": source_id,
                        "grad_norm": float(grad_tensor.cpu()),
                        "seconds_per_log_interval": time.monotonic() - last_log,
                        **metrics,
                    }
                    streaming_metrics = getattr(
                        train_dataset, "streaming_metrics", None
                    )
                    if streaming_metrics is not None:
                        record["streaming_raw"] = dict(streaming_metrics)
                    direct_raw_metrics = getattr(
                        train_dataset, "direct_raw_metrics", None
                    )
                    if direct_raw_metrics is not None:
                        record["direct_raw"] = dict(direct_raw_metrics)
                    if input_adapter is not None:
                        record["direct_vggt"] = dict(input_adapter.metrics)
                    if async_pipeline is not None:
                        record["direct_pipeline"] = dict(async_pipeline.metrics)
                    if completed == 1:
                        record["gradient_ownership"] = gradient_ownership
                    last_log = time.monotonic()
                    print(json.dumps(record, sort_keys=True), flush=True)
                    _append_jsonl(log_path, record)
            if validate_every > 0 and completed % validate_every == 0:
                assert validation_dataset is not None
                validation = _validate(
                    model,
                    validation_dataset,
                    profile,
                    runtime,
                    objective,
                    perceptual_model,
                    context,
                    input_adapter=input_adapter,
                    training_step=completed,
                )
                if context.is_rank0:
                    record = {"step": completed, "validation": validation}
                    print(json.dumps(record, sort_keys=True), flush=True)
                    _append_jsonl(log_path, record)
            if completed in checkpoint_steps:
                manager.save(
                    step=completed,
                    model=model,
                    optimizer=optimizer,
                    metadata={
                        "run_name": config["run"]["name"],
                        "run_lineage": config["run"]["lineage"],
                        "runtime_config_sha256": config_sha,
                        "data_closure_sha256": config["bindings"]["data_closure_sha256"],
                        "model_contract_sha256": config["bindings"]["model_contract_sha256"],
                        "shard_degree": int(runtime["distributed"]["shard_degree"]),
                        "distributed_strategy": strategy.strategy,
                        "global_batch_size": int(runtime["train"]["global_batch_size"]),
                        "topology_contract_sha256": _topology_contract_sha256(config),
                        "launch_qualification_path": launch_qualification_path,
                        "launch_qualification_sha256": launch_qualification_sha256,
                        "sampler_progress": {"next_optimizer_step": completed},
                        "initial_seed": seed,
                        "gradient_ownership": gradient_ownership,
                    },
                    rank_state={"next_optimizer_step": completed},
                )
            if (
                async_pipeline is not None
                and completed < stop_after_step
                and not async_pipeline.pending
            ):
                async_pipeline.submit()
        dist.barrier()
    finally:
        if async_pipeline is not None:
            async_pipeline.close()
        destroy_distributed()


if __name__ == "__main__":
    main()
