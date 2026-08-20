#!/usr/bin/env python3
"""Run one long-lived GPU cache worker over a deterministic task partition."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from wm3d.data.cache_tasks import AtomicTaskClaim, cache_task_from_mapping
from wm3d.data.cache_writer import UnifiedFrameCache, write_cache_task
from wm3d.data.episode_io import (
    decode_episode_views,
    open_episode_accessor,
    select_episode_cache_rows,
    VerifiedAssetStore,
)
from wm3d.data.episode_robot import build_episode_robot_cache
from wm3d.data.manifest_contract import (
    canonical_sha256,
    canonical_timestamp_sha256,
    iter_jsonl,
    load_data_profile,
    sha256_file,
)
from wm3d.data.source_adapters import (
    adapt_action_series,
    adapt_state_series,
    load_adapter_contract,
)
from wm3d.data.task_embedding_store import TaskEmbeddingStore
from wm3d.encoders.native_vggt import NativeVGGTConfig, NativeVGGTEncoder


class CacheWorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedTask:
    task: Any
    frames: UnifiedFrameCache | None
    robot_tensors: Mapping[str, torch.Tensor]
    source_evidence: Mapping[str, Any]
    images: torch.Tensor
    view_mask: torch.Tensor
    prepare_seconds: float
    encode_seconds: float


def _strict_encoder(path: Path) -> NativeVGGTConfig:
    value = yaml.safe_load(path.resolve(strict=True).read_text(encoding="utf-8"))
    fields = set(NativeVGGTConfig.__dataclass_fields__)
    optional = {"appearance_token_grid", "appearance_feature_layer"}
    required = fields - optional
    if not isinstance(value, dict) or not required.issubset(value) or set(value) - fields:
        raise CacheWorkerError(
            f"VGGT contract fields mismatch: missing={sorted(required-set(value or {}))} "
            f"unknown={sorted(set(value or {})-fields)}"
        )
    config = NativeVGGTConfig(**value)
    config.validate()
    return config


def _view_batch(
    *,
    decoded: Mapping[str, Any],
    slots: tuple[str, ...],
    input_size: int,
    color_order_by_view: Mapping[str, str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    color_order_by_view = dict(color_order_by_view or {})
    unknown = sorted(set(color_order_by_view) - set(slots))
    if unknown:
        raise CacheWorkerError(
            f"adapter color_order names are absent from canonical slots: {unknown}"
        )
    first = next(iter(decoded.values())).frames
    count, _height, _width, channels = first.shape
    if channels != 3:
        raise CacheWorkerError("decoded view must be RGB")
    if input_size <= 0:
        raise CacheWorkerError("encoder input size must be positive")
    images = torch.zeros(
        count, len(slots), 3, input_size, input_size, dtype=torch.float32
    )
    mask = torch.zeros(count, len(slots), dtype=torch.bool)
    for slot, name in enumerate(slots):
        item = decoded.get(name)
        if item is None:
            continue
        frames = item.frames
        if frames.shape[0] != count or frames.ndim != 4 or frames.shape[-1] != 3:
            raise CacheWorkerError("real view frame count/RGB layout differs")
        color_order = color_order_by_view.get(name, "rgb")
        if color_order == "bgr":
            frames = frames[..., ::-1].copy()
        elif color_order != "rgb":
            raise CacheWorkerError(
                f"unsupported decoded color_order {color_order!r} for view {name!r}"
            )
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255)
        height, width = tensor.shape[-2:]
        scale = float(input_size) / float(max(height, width))
        resized_h = max(14, int(round(height * scale / 14.0)) * 14)
        resized_w = max(14, int(round(width * scale / 14.0)) * 14)
        resized_h = min(input_size, resized_h)
        resized_w = min(input_size, resized_w)
        tensor = F.interpolate(
            tensor,
            size=(resized_h, resized_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp_(0.0, 1.0)
        pad_h = input_size - resized_h
        pad_w = input_size - resized_w
        tensor = F.pad(
            tensor,
            (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
            ),
            mode="constant",
            value=1.0,
        )
        images[:, slot] = tensor
        mask[:, slot] = True
    if not bool(mask.any(dim=-1).all()):
        raise CacheWorkerError("one selected frame has no real camera")
    return images, mask


def _encode(
    *,
    encoder: NativeVGGTEncoder,
    images: torch.Tensor,
    view_mask: torch.Tensor,
    device: torch.device,
    batch_frames: int,
) -> dict[str, torch.Tensor]:
    if batch_frames <= 0:
        raise CacheWorkerError("encoder batch_frames must be positive")
    output: dict[str, list[torch.Tensor]] = {}
    effective_batch_frames = batch_frames
    start = 0
    while start < len(images):
        stop = min(len(images), start + effective_batch_frames)
        chunk_images = images[start:stop].to(device, non_blocking=True).unsqueeze(0)
        chunk_view_mask = view_mask[start:stop].to(
            device, non_blocking=True
        ).unsqueeze(0)
        try:
            encoded = encoder(chunk_images, chunk_view_mask)
        except torch.OutOfMemoryError:
            del chunk_images, chunk_view_mask
            if effective_batch_frames <= 1:
                raise
            attempted_batch_frames = effective_batch_frames
            effective_batch_frames = max(1, effective_batch_frames // 2)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "attempted_batch_frames": attempted_batch_frames,
                        "remaining_frames": len(images) - start,
                        "retry_batch_frames": effective_batch_frames,
                        "streaming_raw_encoder": "oom_backoff",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        del chunk_images, chunk_view_mask
        for name, value in encoded.items():
            if name == "world_tokens":
                # Fused targets are reconstructed from per-view tokens and
                # geometry confidence at training time, so this duplicate is
                # deliberately not persisted.
                continue
            output.setdefault(name, []).append(value[0].detach().cpu())
        del encoded
        start = stop
    return {name: torch.cat(parts, dim=0) for name, parts in output.items()}


def _prepare_task(
    *,
    task: Any,
    source: Any,
    adapter: Any,
    profile: Any,
    task_store: TaskEmbeddingStore,
    asset_verifier: VerifiedAssetStore,
    encoder_input_size: int,
    task_bank_index_sha256: str,
    decode_workers: int,
    decode_visuals: bool = True,
) -> PreparedTask:
    started = time.perf_counter()
    accessor = open_episode_accessor(
        task=task,
        source_root=source.raw_root,
        adapter=adapter,
        asset_verifier=asset_verifier,
    )
    observation_clock = np.asarray(
        accessor.array(adapter.observation_time_key), dtype=np.float64
    ).reshape(-1)
    if (
        observation_clock.shape != (task.observation_samples,)
        or canonical_timestamp_sha256(observation_clock)
        != task.observation_clock["timestamp_sha256"]
    ):
        raise CacheWorkerError("raw observation clock differs from planned task")
    selection = profile.cache_representation["state_frame_selection"]
    selected_rows = select_episode_cache_rows(
        observation_clock,
        minimum_separation_s=float(selection["minimum_separation_seconds"]),
    )
    if decode_visuals:
        slots = tuple(str(item) for item in profile.cache_representation["view_slots"])
        decoded, video_evidence = decode_episode_views(
            task=task,
            source_root=source.raw_root,
            canonical_view_slots=slots,
            selected_observation_rows=selected_rows,
            asset_verifier=asset_verifier,
            decode_workers=decode_workers,
        )
        images, view_mask = _view_batch(
            decoded=decoded,
            slots=slots,
            input_size=encoder_input_size,
            color_order_by_view={
                view.name: view.color_order for view in adapter.views
            },
        )
    else:
        video_evidence = {}
        images = torch.empty(0, dtype=torch.float32)
        view_mask = torch.empty(0, dtype=torch.bool)
    embodiment = profile.embodiments[task.embodiment]
    robot = build_episode_robot_cache(
        embodiment=embodiment,
        action_series=adapt_action_series(
            accessor=accessor, contract=adapter, embodiment=embodiment
        ),
        state_series=adapt_state_series(
            accessor=accessor, contract=adapter, embodiment=embodiment
        ),
        task_embedding=task_store.get(task.task_text),
        observation_times_s=observation_clock,
        max_groups=max(len(item.groups) for item in profile.embodiments.values()),
        max_action_dim=max(
            group.action_dim
            for item in profile.embodiments.values()
            for group in item.groups
        ),
        max_state_dim=max(
            max((group.state_dim for group in item.groups), default=0)
            for item in profile.embodiments.values()
        ),
    )
    return PreparedTask(
        task=task,
        frames=None,
        robot_tensors=robot.as_tensors(),
        source_evidence={
            "observation_clock_sha256": canonical_timestamp_sha256(observation_clock),
            "selected_source_rows": selected_rows.tolist(),
            "videos": video_evidence,
            "task_bank_index_sha256": task_bank_index_sha256,
        },
        images=images,
        view_mask=view_mask,
        prepare_seconds=time.perf_counter() - started,
        encode_seconds=0.0,
    )


def _encode_task(
    prepared: PreparedTask,
    *,
    encoder: NativeVGGTEncoder,
    device: torch.device,
    batch_frames: int,
) -> PreparedTask:
    started = time.perf_counter()
    encoded = _encode(
        encoder=encoder,
        images=prepared.images,
        view_mask=prepared.view_mask,
        device=device,
        batch_frames=batch_frames,
    )
    confidence = encoded["geometry_confidence"].float()
    depth = encoded["depth"]
    point = encoded["point"]
    real_patch = encoded["view_mask"][..., None]
    geometry_valid = confidence > 0
    depth_mask = real_patch & geometry_valid & torch.isfinite(depth) & (depth > 0)
    point_mask = real_patch & geometry_valid & torch.isfinite(point).all(dim=-1)
    world_mask = (real_patch & geometry_valid).any(dim=1)
    camera_mask = encoded["view_mask"] & torch.isfinite(
        encoded["camera_pose"]
    ).all(dim=-1)
    frames = UnifiedFrameCache(
        source_observation_rows=torch.from_numpy(
            np.asarray(prepared.source_evidence["selected_source_rows"], dtype=np.int64)
        ),
        frame_times_s=prepared.robot_tensors["observation_times_s"][
            torch.as_tensor(
                prepared.source_evidence["selected_source_rows"], dtype=torch.long
            )
        ].to(torch.float64),
        view_tokens=encoded["view_tokens"],
        rgb=encoded["rgb"],
        view_mask=encoded["view_mask"],
        world_token_mask=world_mask,
        depth=depth,
        depth_mask=depth_mask,
        point=point,
        point_mask=point_mask,
        camera_pose=encoded["camera_pose"],
        camera_pose_mask=camera_mask,
        geometry_confidence=encoded["geometry_confidence"],
        appearance_tokens=encoded.get("appearance_tokens"),
    )
    encoded_prepared = PreparedTask(
        task=prepared.task,
        frames=frames,
        robot_tensors=prepared.robot_tensors,
        source_evidence=prepared.source_evidence,
        images=torch.empty(0),
        view_mask=torch.empty(0, dtype=torch.bool),
        prepare_seconds=prepared.prepare_seconds,
        encode_seconds=time.perf_counter() - started,
    )
    return encoded_prepared


def _write_task(
    prepared: PreparedTask, *, cache_root: Path
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    if prepared.frames is None:
        raise CacheWorkerError("prepared task has no encoded frames")
    result = write_cache_task(
        task=prepared.task,
        cache_root=cache_root,
        frames=prepared.frames,
        robot_tensors=prepared.robot_tensors,
        source_evidence=prepared.source_evidence,
    )
    return result, time.perf_counter() - started


def _finish_write(
    future: Future[tuple[dict[str, Any], float]], *,
    prepared: PreparedTask,
    worker_index: int,
    ordinal: int,
    assigned: int,
    counters: dict[str, int],
    worker_started: float,
) -> None:
    result, write_seconds = future.result()
    counters["published"] += int(result["status"] == "published")
    counters["already_complete"] += int(result["status"] == "already_complete")
    counters["finished"] += 1
    frames = int(result.get("frames", 0))
    print(
        json.dumps(
            {
                "worker": worker_index,
                "ordinal": ordinal,
                "assigned": assigned,
                **counters,
                "task_id": prepared.task.task_id,
                "frames": frames,
                "prepare_seconds": round(prepared.prepare_seconds, 3),
                "encode_seconds": round(prepared.encode_seconds, 3),
                "write_seconds": round(write_seconds, 3),
                "elapsed_seconds": round(time.perf_counter() - worker_started, 3),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--data-profile", type=Path, required=True)
    parser.add_argument("--encoder-contract", type=Path, required=True)
    parser.add_argument("--task-bank-root", type=Path, required=True)
    parser.add_argument("--task-bank-index-sha256", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-frames", type=int, default=8)
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=0,
        help="每个 GPU worker 的视频解码线程数；0 表示读取 data profile。",
    )
    parser.add_argument(
        "--writer-threads",
        type=int,
        default=0,
        help="每个 GPU worker 的并行写盘线程数；0 表示读取 data profile。",
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.worker_count or args.worker_count <= 0:
        raise CacheWorkerError("worker index/count are invalid")
    if args.batch_frames <= 0:
        raise CacheWorkerError("batch-frames must be positive")
    profile = load_data_profile(args.data_profile, verify_source_manifests=True)
    decode_workers = int(
        args.decode_workers or profile.cache["decode_workers_per_gpu"]
    )
    writer_threads = int(
        args.writer_threads or profile.cache["writer_threads_per_worker"]
    )
    if decode_workers <= 0 or writer_threads <= 0:
        raise CacheWorkerError("decode/writer worker counts must be positive")
    sources = {item.name: item for item in profile.sources}
    adapters = {
        item.name: load_adapter_contract(
            item.adapter_config_path,
            expected_sha256=item.adapter_contract_sha256,
        )
        for item in profile.sources
    }
    compatible_adapter_kinds = {
        "lerobot": "lerobot_parquet_video",
        "agibot": "agibot_parquet_video",
        "npz": "npz",
    }
    for source in profile.sources:
        expected_raw_format = compatible_adapter_kinds.get(source.adapter)
        if expected_raw_format is None or adapters[source.name].raw_format != expected_raw_format:
            raise CacheWorkerError(
                f"source {source.name!r} adapter kind {source.adapter!r} does not "
                f"match contract raw_format {adapters[source.name].raw_format!r}"
            )
    tasks = [
        cache_task_from_mapping(dict(row))
        for _line, row in iter_jsonl(args.task_manifest)
    ]
    if not tasks:
        raise CacheWorkerError("task manifest is empty")
    selected = [
        task
        for ordinal, task in enumerate(sorted(tasks, key=lambda item: item.task_id))
        if ordinal % args.worker_count == args.worker_index
    ]
    encoder_config = _strict_encoder(args.encoder_contract)
    representation = profile.cache_representation
    encoder_digest = sha256_file(args.encoder_contract.resolve(strict=True))
    representation_digest = canonical_sha256(representation)
    for task in tasks:
        source = sources.get(task.source)
        if source is None:
            raise CacheWorkerError(f"task source {task.source!r} is absent from profile")
        if task.adapter_contract_sha256 != source.adapter_contract_sha256:
            raise CacheWorkerError("task adapter contract differs from data profile")
        if task.source_manifest_sha256 != source.manifest_sha256:
            raise CacheWorkerError("task source manifest differs from data profile")
        if task.encoder_contract_sha256 != encoder_digest:
            raise CacheWorkerError("task encoder contract differs from worker encoder")
        if task.task_bank_index_sha256 != args.task_bank_index_sha256:
            raise CacheWorkerError("task plan differs from worker task bank index")
        if task.representation_contract_sha256 != representation_digest:
            raise CacheWorkerError("task representation contract differs from data profile")
    exact = {
        "token_grid": int(representation["token_grid"]),
        "target_rgb_size": int(representation["rgb_size"]),
        "token_dim": int(representation["token_dim"]),
        "max_views": int(representation["num_views"]),
    }
    for name, expected in exact.items():
        if int(getattr(encoder_config, name)) != expected:
            raise CacheWorkerError(
                f"encoder/data representation {name} mismatch: "
                f"{getattr(encoder_config, name)} != {expected}"
            )
    expected_appearance_grid = int(representation.get("appearance_token_grid", 0))
    if int(encoder_config.appearance_token_grid) != expected_appearance_grid:
        raise CacheWorkerError(
            "encoder/data representation appearance_token_grid mismatch: "
            f"{encoder_config.appearance_token_grid} != {expected_appearance_grid}"
        )
    expected_appearance_layer = int(
        representation.get("appearance_feature_layer", -1)
    )
    if int(encoder_config.appearance_feature_layer) != expected_appearance_layer:
        raise CacheWorkerError(
            "encoder/data representation appearance_feature_layer mismatch: "
            f"{encoder_config.appearance_feature_layer} != "
            f"{expected_appearance_layer}"
        )
    task_encoder_digests = {task.task_encoder_contract_sha256 for task in tasks}
    if len(task_encoder_digests) != 1:
        raise CacheWorkerError("task plan mixes task encoder contracts")
    worker_started = time.perf_counter()
    pending: list[tuple[int, Any, Any]] = []
    counters = {
        "published": 0,
        "already_complete": 0,
        "failed": 0,
        "finished": 0,
    }
    for ordinal, task in enumerate(selected, 1):
        source = sources.get(task.source)
        if source is None or source.embodiment != task.embodiment:
            raise CacheWorkerError("task source/embodiment is absent from profile")
        if AtomicTaskClaim(args.cache_root, task).completed():
            counters["already_complete"] += 1
            counters["finished"] += 1
        else:
            pending.append((ordinal, task, source))

    if not pending:
        elapsed = time.perf_counter() - worker_started
        print(
            json.dumps(
                {
                    "worker": args.worker_index,
                    "assigned": len(selected),
                    **counters,
                    "decode_workers": decode_workers,
                    "writer_threads": writer_threads,
                    "batch_frames": args.batch_frames,
                    "elapsed_seconds": round(elapsed, 3),
                    "tasks_per_second": 0.0,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    device = torch.device(args.device)
    encoder = NativeVGGTEncoder(encoder_config, device=str(device)).eval()
    task_store = TaskEmbeddingStore(
        root=args.task_bank_root,
        index_sha256=args.task_bank_index_sha256,
        expected_data_profile_sha256=profile.profile_sha256,
        expected_source_manifest_sha256_by_name={
            source.name: source.manifest_sha256 for source in profile.sources
        },
        expected_encoder_contract_sha256=next(iter(task_encoder_digests)),
    )
    asset_verifier = VerifiedAssetStore()

    def prepare(row: tuple[int, Any, Any]) -> tuple[int, PreparedTask]:
        ordinal, task, source = row
        return ordinal, _prepare_task(
            task=task,
            source=source,
            adapter=adapters[task.source],
            profile=profile,
            task_store=task_store,
            asset_verifier=asset_verifier,
            encoder_input_size=encoder_config.input_rgb_size,
            task_bank_index_sha256=args.task_bank_index_sha256,
            decode_workers=decode_workers,
        )

    queued_writes: deque[
        tuple[int, PreparedTask, Future[tuple[dict[str, Any], float]]]
    ] = deque()
    with (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="wm3d-prepare") as preparer,
        ThreadPoolExecutor(
            max_workers=writer_threads, thread_name_prefix="wm3d-writer"
        ) as writer,
    ):
        next_prepared: Future[tuple[int, PreparedTask]] | None = (
            preparer.submit(prepare, pending[0]) if pending else None
        )
        for position in range(len(pending)):
            ordinal, task, _source = pending[position]
            current_prepared = next_prepared
            next_prepared = (
                preparer.submit(prepare, pending[position + 1])
                if position + 1 < len(pending)
                else None
            )
            try:
                if current_prepared is None:
                    raise AssertionError("cache prepare pipeline lost its future")
                prepared_ordinal, prepared = current_prepared.result()
                if prepared_ordinal != ordinal or prepared.task.task_id != task.task_id:
                    raise AssertionError("cache prepare pipeline changed task order")
                encoded = _encode_task(
                    prepared,
                    encoder=encoder,
                    device=device,
                    batch_frames=args.batch_frames,
                )
                queued_writes.append(
                    (
                        ordinal,
                        encoded,
                        writer.submit(_write_task, encoded, cache_root=args.cache_root),
                    )
                )
                if len(queued_writes) > writer_threads:
                    done_ordinal, done_prepared, done_future = queued_writes.popleft()
                    _finish_write(
                        done_future,
                        prepared=done_prepared,
                        worker_index=args.worker_index,
                        ordinal=done_ordinal,
                        assigned=len(selected),
                        counters=counters,
                        worker_started=worker_started,
                    )
            except Exception as exc:
                counters["failed"] += 1
                print(
                    json.dumps(
                        {
                            "worker": args.worker_index,
                            "task_id": task.task_id,
                            "type": type(exc).__name__,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                )
                if args.fail_fast:
                    raise
        while queued_writes:
            ordinal, prepared, future = queued_writes.popleft()
            try:
                _finish_write(
                    future,
                    prepared=prepared,
                    worker_index=args.worker_index,
                    ordinal=ordinal,
                    assigned=len(selected),
                    counters=counters,
                    worker_started=worker_started,
                )
            except Exception as exc:
                counters["failed"] += 1
                print(
                    json.dumps(
                        {
                            "worker": args.worker_index,
                            "task_id": prepared.task.task_id,
                            "type": type(exc).__name__,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                )
                if args.fail_fast:
                    raise
    elapsed = time.perf_counter() - worker_started
    print(
        json.dumps(
            {
                "worker": args.worker_index,
                "assigned": len(selected),
                **counters,
                "decode_workers": decode_workers,
                "writer_threads": writer_threads,
                "batch_frames": args.batch_frames,
                "elapsed_seconds": round(elapsed, 3),
                "tasks_per_second": round(len(pending) / elapsed, 4) if elapsed else 0.0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if counters["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
