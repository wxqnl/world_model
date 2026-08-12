#!/usr/bin/env python3
"""Run one long-lived GPU cache worker over a deterministic task partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from wm3d_v3.data.cache_tasks import AtomicTaskClaim, cache_task_from_mapping
from wm3d_v3.data.cache_writer import UnifiedFrameCache, write_cache_task
from wm3d_v3.data.episode_io import (
    decode_episode_views,
    open_episode_accessor,
    select_episode_cache_rows,
    VerifiedAssetStore,
)
from wm3d_v3.data.episode_robot import build_episode_robot_cache
from wm3d_v3.data.manifest_contract import (
    canonical_sha256,
    canonical_timestamp_sha256,
    iter_jsonl,
    load_data_profile,
    sha256_file,
)
from wm3d_v3.data.source_adapters import (
    adapt_action_series,
    adapt_state_series,
    load_adapter_contract,
)
from wm3d_v3.data.task_embedding_store import TaskEmbeddingStore
from wm3d_v3.encoders.native_vggt import NativeVGGTConfig, NativeVGGTEncoder


class CacheWorkerError(RuntimeError):
    pass


def _strict_encoder(path: Path) -> NativeVGGTConfig:
    value = yaml.safe_load(path.resolve(strict=True).read_text(encoding="utf-8"))
    required = set(NativeVGGTConfig.__dataclass_fields__)
    if not isinstance(value, dict) or set(value) != required:
        raise CacheWorkerError(
            f"VGGT contract fields mismatch: missing={sorted(required-set(value or {}))} "
            f"unknown={sorted(set(value or {})-required)}"
        )
    config = NativeVGGTConfig(**value)
    config.validate()
    return config


def _view_batch(
    *,
    decoded: Mapping[str, Any],
    slots: tuple[str, ...],
    input_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    output: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(images), batch_frames):
        stop = min(len(images), start + batch_frames)
        encoded = encoder(
            images[start:stop].to(device, non_blocking=True).unsqueeze(0),
            view_mask[start:stop].to(device, non_blocking=True).unsqueeze(0),
        )
        for name, value in encoded.items():
            if name == "world_tokens":
                # Fused targets are reconstructed from per-view tokens and
                # geometry confidence at training time, so this duplicate is
                # deliberately not persisted.
                continue
            output.setdefault(name, []).append(value[0].detach().cpu())
    return {name: torch.cat(parts, dim=0) for name, parts in output.items()}


def _process_task(
    *,
    task: Any,
    source: Any,
    adapter: Any,
    profile: Any,
    encoder: NativeVGGTEncoder,
    task_store: TaskEmbeddingStore,
    device: torch.device,
    batch_frames: int,
    cache_root: Path,
    asset_verifier: VerifiedAssetStore,
    encoder_input_size: int,
    task_bank_index_sha256: str,
) -> dict[str, Any]:
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
    slots = tuple(str(item) for item in profile.cache_representation["view_slots"])
    decoded, video_evidence = decode_episode_views(
        task=task,
        source_root=source.raw_root,
        canonical_view_slots=slots,
        selected_observation_rows=selected_rows,
        asset_verifier=asset_verifier,
    )
    images, view_mask = _view_batch(
        decoded=decoded, slots=slots, input_size=encoder_input_size
    )
    encoded = _encode(
        encoder=encoder,
        images=images,
        view_mask=view_mask,
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
    camera_mask = encoded["view_mask"] & torch.isfinite(encoded["camera_pose"]).all(dim=-1)
    frames = UnifiedFrameCache(
        source_observation_rows=torch.from_numpy(selected_rows),
        frame_times_s=torch.from_numpy(observation_clock[selected_rows].copy()),
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
    )
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
    return write_cache_task(
        task=task,
        cache_root=cache_root,
        frames=frames,
        robot_tensors=robot.as_tensors(),
        source_evidence={
            "observation_clock_sha256": canonical_timestamp_sha256(observation_clock),
            "selected_source_rows": selected_rows.tolist(),
            "videos": video_evidence,
            "task_bank_index_sha256": task_bank_index_sha256,
        },
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
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.worker_count or args.worker_count <= 0:
        raise CacheWorkerError("worker index/count are invalid")
    if args.batch_frames <= 0:
        raise CacheWorkerError("batch-frames must be positive")
    profile = load_data_profile(args.data_profile, verify_source_manifests=True)
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
    device = torch.device(args.device)
    encoder = NativeVGGTEncoder(encoder_config, device=str(device)).eval()
    task_encoder_digests = {task.task_encoder_contract_sha256 for task in tasks}
    if len(task_encoder_digests) != 1:
        raise CacheWorkerError("task plan mixes task encoder contracts")
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
    published = complete = failed = 0
    for ordinal, task in enumerate(selected, 1):
        try:
            source = sources.get(task.source)
            if source is None or source.embodiment != task.embodiment:
                raise CacheWorkerError("task source/embodiment is absent from profile")
            # Avoid decoding or loading raw robot arrays for receipts already
            # proven complete.  write_cache_task repeats this check atomically.
            if AtomicTaskClaim(args.cache_root, task).completed():
                complete += 1
                continue
            result = _process_task(
                task=task,
                source=source,
                adapter=adapters[task.source],
                profile=profile,
                encoder=encoder,
                task_store=task_store,
                device=device,
                batch_frames=args.batch_frames,
                cache_root=args.cache_root,
                asset_verifier=asset_verifier,
                encoder_input_size=encoder_config.input_rgb_size,
                task_bank_index_sha256=args.task_bank_index_sha256,
            )
            published += int(result["status"] == "published")
            complete += int(result["status"] == "already_complete")
        except Exception as exc:
            failed += 1
            record = {
                "worker": args.worker_index,
                "task_id": task.task_id,
                "type": type(exc).__name__,
                "error": str(exc),
            }
            print(json.dumps(record, ensure_ascii=False, sort_keys=True), file=sys.stderr)
            if args.fail_fast:
                raise
        print(
            json.dumps(
                {
                    "worker": args.worker_index,
                    "ordinal": ordinal,
                    "assigned": len(selected),
                    "published": published,
                    "already_complete": complete,
                    "failed": failed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
