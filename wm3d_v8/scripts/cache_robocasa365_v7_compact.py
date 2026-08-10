#!/usr/bin/env python3
"""Build the WM3D-v7 compact, paired-view RoboCasa365 cache.

The cache intentionally stores no RGB.  It keeps fixed PCA latents (int8),
8x8 anchor-view geometry targets, audited 5 Hz actions, and one task embedding
per clip.  Video files are decoded sequentially once per selected video group.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wm3d_v3.data.v7_action_contract import (
    ActionAdapter,
    canonicalize_dense_action,
    resample_canonical_actions,
)
from wm3d_v3.data.v7_contracts import (
    CONTEXT_FRAMES,
    FUTURE_FRAMES,
    MODEL_HZ,
    WINDOW_STRIDE,
    V7ClipRecord,
    read_v7_manifest,
)
from wm3d_v3.data.v8_causal_dual_view import (
    causal_dual_view_metadata,
    encode_causal_dual_view,
    encode_observed_context,
    validate_causal_dual_view_archive,
)
from wm3d_v3.encoders.qwen_vl_encoder import QwenVLEmbed
from wm3d_v3.encoders.vggt_encoder import VGGTEncoder
from wm3d_v3.models.token_codec import PCATokenCodec, TokenCodecConfig
from wm3d_v3.stage1.action_window_geometry import VGGT_MODEL_REVISION


SCHEMA_VERSION = "wm3d_v7_compact_geom_v3"
CAMERA_ROLES = {
    "anchor": "observation.images.robot0_agentview_left",
    "wrist": "observation.images.robot0_eye_in_hand",
}


@dataclass
class WorkItem:
    record: V7ClipRecord
    metadata: dict[str, Any]
    model_frame_indices: np.ndarray
    actions: np.ndarray
    raw_actions: np.ndarray
    native_frame_indices: np.ndarray
    model_timestamps: np.ndarray
    action_valid_mask: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    anchor_codes: np.ndarray | None = None
    anchor_scale: np.ndarray | None = None
    future_codes: np.ndarray | None = None
    future_scale: np.ndarray | None = None
    window_starts: np.ndarray | None = None
    wrist_codes: np.ndarray | None = None
    wrist_scale: np.ndarray | None = None
    depth_patch: np.ndarray | None = None
    depth_conf_patch: np.ndarray | None = None
    point_patch: np.ndarray | None = None
    point_conf_patch: np.ndarray | None = None
    pose_enc: np.ndarray | None = None
    geometry_segment_id: np.ndarray | None = None


def _load_episode_metadata(root: Path) -> tuple[dict[int, dict[str, Any]], dict[tuple[int, int], int]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    if not rows:
        raise RuntimeError(f"no LeRobot episode metadata under {root}")
    by_episode = {int(row["episode_index"]): row for row in rows}
    file_base: dict[tuple[int, int], int] = {}
    for row in rows:
        key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        start = int(row["dataset_from_index"])
        file_base[key] = min(start, file_base.get(key, start))
    return by_episode, file_base


def _episode_index(record: V7ClipRecord) -> int:
    try:
        return int(record.native_episode_id.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"cannot parse episode index: {record.native_episode_id}") from exc


def _filter_records_by_rgb_sidecar(
    records: list[V7ClipRecord],
    rgb_sidecar_index: Path,
) -> tuple[list[V7ClipRecord], str]:
    """Keep only split-aligned clips with immutable RGB supervision."""

    rgb_sidecar_index = Path(rgb_sidecar_index)
    available: dict[str, str] = {}
    for line_number, line in enumerate(
        rgb_sidecar_index.read_text().splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{rgb_sidecar_index}:{line_number}: invalid JSON: {exc}"
            ) from exc
        if row.get("schema") != "wm3d_v7_rgb_sidecar_v1":
            raise ValueError(
                f"{rgb_sidecar_index}:{line_number}: invalid RGB sidecar schema"
            )
        clip_hash = str(row.get("clip_hash") or "")
        split = str(row.get("split") or "")
        if not clip_hash or split not in {"train", "val", "test"}:
            raise ValueError(
                f"{rgb_sidecar_index}:{line_number}: invalid RGB identity"
            )
        previous = available.get(clip_hash)
        if previous is not None and previous != split:
            raise ValueError(
                f"{rgb_sidecar_index}:{line_number}: conflicting split for "
                f"{clip_hash}"
            )
        available[clip_hash] = split
    if not available:
        raise ValueError(f"empty RGB sidecar index: {rgb_sidecar_index}")
    filtered = [
        record for record in records
        if available.get(record.clip_hash) == record.split
    ]
    if not filtered:
        raise ValueError("manifest has no split-aligned RGB sidecar clips")
    return filtered, _sha256_file(rgb_sidecar_index)


def _select_records(records: list[V7ClipRecord], max_clips: int) -> list[V7ClipRecord]:
    records = [record for record in records if record.action_valid]
    if max_clips <= 0 or max_clips >= len(records):
        return records
    # A small proof cache still needs independent train/val/test coverage.
    buckets = {split: [record for record in records if record.split == split] for split in ("train", "val", "test")}
    selected: list[V7ClipRecord] = []
    while len(selected) < max_clips and any(buckets.values()):
        for split in ("train", "val", "test"):
            if buckets[split] and len(selected) < max_clips:
                selected.append(buckets[split].pop(0))
    return selected


def _video_path(root: Path, metadata: dict[str, Any], camera: str) -> Path:
    chunk = int(metadata[f"videos/{camera}/chunk_index"])
    file_index = int(metadata[f"videos/{camera}/file_index"])
    return root / "videos" / camera / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"


def _video_group_key(
    root: Path,
    record: V7ClipRecord,
    metadata_by_episode: dict[int, dict[str, Any]],
) -> tuple[str, str]:
    metadata = metadata_by_episode[_episode_index(record)]
    return tuple(
        _video_path(root, metadata, CAMERA_ROLES[role]).relative_to(root).as_posix()
        for role in ("anchor", "wrist")
    )


def _selection_sha256(records: list[V7ClipRecord]) -> str:
    clip_hashes = sorted(record.clip_hash for record in records)
    if len(clip_hashes) != len(set(clip_hashes)):
        raise RuntimeError("selected manifest contains duplicate clip_hash values")
    return hashlib.sha256(("\n".join(clip_hashes) + "\n").encode("utf-8")).hexdigest()


def _partition_by_video_group(
    records: list[V7ClipRecord],
    root: Path,
    metadata_by_episode: dict[int, dict[str, Any]],
    num_shards: int,
) -> tuple[list[list[V7ClipRecord]], list[int], list[int]]:
    """Deterministic LPT partitioning without splitting a paired video file."""
    groups: dict[tuple[str, str], list[V7ClipRecord]] = {}
    for record in records:
        groups.setdefault(_video_group_key(root, record, metadata_by_episode), []).append(record)
    weighted_groups = [
        (
            sum((record.native_end_frame - record.native_start_frame) // 4 for record in group),
            key,
            group,
        )
        for key, group in groups.items()
    ]
    weighted_groups.sort(key=lambda value: (-value[0], value[1]))
    shard_records: list[list[V7ClipRecord]] = [[] for _ in range(num_shards)]
    shard_costs = [0 for _ in range(num_shards)]
    shard_groups = [0 for _ in range(num_shards)]
    owner_by_group: dict[tuple[str, str], int] = {}
    for cost, key, group in weighted_groups:
        shard_id = min(range(num_shards), key=lambda index: (shard_costs[index], index))
        shard_records[shard_id].extend(group)
        shard_costs[shard_id] += int(cost)
        shard_groups[shard_id] += 1
        owner_by_group[key] = shard_id
    assigned = [record.clip_hash for shard in shard_records for record in shard]
    selected = [record.clip_hash for record in records]
    if sorted(assigned) != sorted(selected) or len(assigned) != len(set(assigned)):
        raise RuntimeError("whole-video shard plan does not exactly cover selected clips")
    for record in records:
        key = _video_group_key(root, record, metadata_by_episode)
        if key not in owner_by_group:
            raise RuntimeError(f"video group was not assigned: {key}")
    return shard_records, shard_costs, shard_groups


def _shard_index_path(base: Path, shard_id: int, num_shards: int) -> Path:
    if num_shards == 1:
        return base
    return base.with_name(
        f"{base.stem}.shard-{shard_id:05d}-of-{num_shards:05d}{base.suffix}"
    )


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)

def _publish_text_no_clobber(path: Path, text: str) -> str:
    encoded = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"existing report is non-identical: {path}")
        return _sha256_file(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise FileExistsError(
                    f"existing report is non-identical: {path}"
                )
        return _sha256_file(path)
    finally:
        temporary.unlink(missing_ok=True)




def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _load_adapter(audit_path: Path, *, allow_legacy_proof_audit: bool) -> ActionAdapter:
    payload = json.loads(audit_path.read_text())
    factual = payload.get("factual_action_audit") or {}
    if not bool(factual.get("passed")):
        if not (
            allow_legacy_proof_audit
            and bool((payload.get("audit") or {}).get("passed"))
        ):
            raise RuntimeError(
                "formal cache requires factual_action_audit.passed=true; "
                "counterfactual replay is audited independently"
            )
    return ActionAdapter(**payload["adapter"])


def _load_codec(path: Path, device: torch.device) -> PCATokenCodec:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    codec = PCATokenCodec(
        TokenCodecConfig(token_dim=int(payload["token_dim"]), latent_dim=int(payload["latent_dim"]))
    )
    codec.set_basis(payload["mean"], payload["components"])
    return codec.to(device).eval()


def _prepare_items(
    records: list[V7ClipRecord],
    root: Path,
    metadata_by_episode: dict[int, dict[str, Any]],
    file_base: dict[tuple[int, int], int],
    adapter: ActionAdapter,
) -> list[WorkItem]:
    table_cache: dict[Path, dict[str, np.ndarray]] = {}
    items: list[WorkItem] = []
    for record in records:
        episode = _episode_index(record)
        metadata = metadata_by_episode[episode]
        parquet_path = Path(record.raw_path)
        if parquet_path not in table_cache:
            table = pq.read_table(
                parquet_path,
                columns=["action", "next.reward", "next.done", "episode_index", "frame_index"],
            )
            table_cache[parquet_path] = {
                "action": np.asarray(table["action"].to_pylist(), dtype=np.float32),
                "reward": np.asarray(table["next.reward"].to_numpy(), dtype=np.float32),
                "done": np.asarray(table["next.done"].to_numpy(), dtype=np.bool_),
                "episode": np.asarray(table["episode_index"].to_numpy(), dtype=np.int64),
                "frame": np.asarray(table["frame_index"].to_numpy(), dtype=np.int64),
            }
        arrays = table_cache[parquet_path]
        key = (int(metadata["data/chunk_index"]), int(metadata["data/file_index"]))
        local_episode_start = int(metadata["dataset_from_index"]) - file_base[key]
        local_start = local_episode_start + int(record.native_start_frame)
        local_end = local_episode_start + int(record.native_end_frame)
        raw = arrays["action"][local_start:local_end]
        episode_values = arrays["episode"][local_start:local_end]
        frame_values = arrays["frame"][local_start:local_end]
        expected_frames = np.arange(record.native_start_frame, record.native_end_frame, dtype=np.int64)
        if len(raw) != len(expected_frames) or not np.all(episode_values == episode) or not np.array_equal(frame_values, expected_frames):
            raise RuntimeError(f"parquet/manifest interval mismatch for {record.clip_hash}")
        if not np.all(raw[:, 4] < 0):
            raise RuntimeError(f"non-arm action leaked into action-valid clip {record.clip_hash}")
        dense = np.concatenate((raw[:, 5:11], raw[:, 11:12]), axis=1)
        canonical = canonicalize_dense_action(dense, adapter)
        model_actions = resample_canonical_actions(canonical, source_hz=20.0, target_hz=5.0)
        count = len(model_actions)
        if count < 24:
            raise RuntimeError(
                f"manifest admitted short clip {record.clip_hash}: "
                f"{len(raw)} native frames -> {count} model frames; require >=24"
            )
        model_frame_indices = record.native_start_frame + np.arange(count, dtype=np.int64) * 4
        terminal_indices = local_start + np.arange(count, dtype=np.int64) * 4 + 3
        items.append(
            WorkItem(
                record=record,
                metadata=metadata,
                model_frame_indices=model_frame_indices,
                actions=model_actions,
                raw_actions=raw.copy(),
                native_frame_indices=expected_frames.copy(),
                model_timestamps=model_frame_indices.astype(np.float64) / 20.0,
                action_valid_mask=np.ones(count, dtype=np.bool_),
                rewards=arrays["reward"][terminal_indices],
                dones=arrays["done"][terminal_indices],
            )
        )
    return items


def _resize_frames(frames: np.ndarray, size: int = 224) -> torch.Tensor:
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
    if tensor.shape[-2:] != (size, size):
        tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return tensor


@torch.inference_mode()
def _encode_clip(
    frames: list[np.ndarray],
    *,
    encoder: VGGTEncoder,
    codec: PCATokenCodec,
    batch_frames: int,
    keep_geometry: bool,
) -> dict[str, np.ndarray]:
    images = _resize_frames(np.stack(frames))
    pooled_chunks: list[torch.Tensor] = []
    depth_chunks: list[torch.Tensor] = []
    conf_chunks: list[torch.Tensor] = []
    point_chunks: list[torch.Tensor] = []
    point_conf_chunks: list[torch.Tensor] = []
    pose_chunks: list[torch.Tensor] = []
    geometry_segment_chunks: list[np.ndarray] = []
    encoder.return_depth = bool(keep_geometry)
    encoder.return_depth_conf = bool(keep_geometry)
    encoder.return_geom_extra = bool(keep_geometry)
    for segment_id, start in enumerate(range(0, len(images), batch_frames)):
        stop = min(len(images), start + batch_frames)
        out = encoder(images[start:stop].unsqueeze(0))
        pooled_chunks.append(out["pooled"][0])
        if keep_geometry:
            depth_chunks.append(out["depth"][0])
            conf_chunks.append(out["depth_conf"][0])
            point_chunks.append(out["world_points"][0])
            point_conf_chunks.append(out["world_points_conf"][0])
            pose_chunks.append(out["pose_enc"][0])
            geometry_segment_chunks.append(
                np.full(stop - start, segment_id, dtype=np.int16)
            )
    pooled = torch.cat(pooled_chunks, dim=0).to(codec.mean.device)
    latent = codec.encode(pooled)
    codes, scale = codec.quantize(latent)
    result = {
        "codes": codes.cpu().numpy(),
        "scale": scale.cpu().numpy(),
    }
    if keep_geometry:
        depth = torch.cat(depth_chunks, dim=0).float()
        conf = torch.cat(conf_chunks, dim=0).float()
        depth_patch = F.adaptive_avg_pool2d(depth[:, None], (8, 8)).squeeze(1)
        conf_patch = F.adaptive_avg_pool2d(conf[:, None], (8, 8)).squeeze(1)
        points = torch.cat(point_chunks, dim=0).float().permute(0, 3, 1, 2)
        point_conf = torch.cat(point_conf_chunks, dim=0).float()
        point_patch = F.adaptive_avg_pool2d(points, (8, 8)).permute(0, 2, 3, 1)
        point_conf_patch = F.adaptive_avg_pool2d(point_conf[:, None], (8, 8)).squeeze(1)
        result["depth_patch"] = depth_patch.to(torch.float16).cpu().numpy()
        result["depth_conf_patch"] = conf_patch.to(torch.float16).cpu().numpy()
        result["point_patch"] = point_patch.to(torch.float16).cpu().numpy()
        result["point_conf_patch"] = point_conf_patch.to(torch.float16).cpu().numpy()
        result["pose_enc"] = torch.cat(pose_chunks, dim=0).to(torch.float16).cpu().numpy()
        result["geometry_segment_id"] = np.concatenate(geometry_segment_chunks)
    return result


@torch.inference_mode()
def _encode_causal_clip(
    frames: list[np.ndarray],
    *,
    encoder: VGGTEncoder,
    codec: PCATokenCodec,
    T: int,
    k: int,
    stride: int,
    keep_geometry: bool,
) -> dict[str, np.ndarray]:
    """Encode exact windows from one decoded clip using independent forwards."""

    if T <= 0 or k <= 0 or stride <= 0:
        raise ValueError("T, k, and stride must be positive")
    required = T + k
    starts = np.arange(0, len(frames) - required + 1, stride, dtype=np.int64)
    if len(starts) == 0:
        raise ValueError(
            f"causal clip has {len(frames)} frames; requires at least {required}"
        )
    images = _resize_frames(np.stack(frames))
    windows: list[dict[str, np.ndarray]] = []
    for start in starts.tolist():
        window = images[start : start + required]
        if keep_geometry:
            encoded = encode_causal_dual_view(
                window,
                encoder=encoder,
                codec=codec,
                T=T,
                k=k,
            )
        else:
            encoded = encode_observed_context(
                window[:T],
                encoder=encoder,
                codec=codec,
                T=T,
            )
        windows.append(encoded)

    result = {
        **causal_dual_view_metadata(T=T, k=k),
        "window_starts": starts,
        "context_codes": np.stack(
            [window["context_codes"] for window in windows], axis=0
        ),
        "context_scale": np.stack(
            [window["context_scale"] for window in windows], axis=0
        ),
    }
    if keep_geometry:
        for key in (
            "future_codes",
            "future_scale",
            "future_depth_patch",
            "future_depth_conf_patch",
            "future_point_patch",
            "future_point_conf_patch",
            "future_pose_enc",
        ):
            result[key] = np.stack([window[key] for window in windows], axis=0)
        validate_causal_dual_view_archive(
            result,
            T=T,
            k=k,
            paired_views=False,
        )
    return result



def _encode_video_group(
    video_path: Path,
    group: list[WorkItem],
    *,
    camera: str,
    encoder: VGGTEncoder,
    codec: PCATokenCodec,
    batch_frames: int,
    keep_geometry: bool,
    causal_dual_view: bool,
) -> None:
    wanted: dict[int, list[int]] = {}
    for item_index, item in enumerate(group):
        base = int(round(float(item.metadata[f"videos/{camera}/from_timestamp"]) * 20.0))
        for frame_index in item.model_frame_indices:
            wanted.setdefault(base + int(frame_index), []).append(item_index)
    if not wanted:
        return
    buffers: dict[int, list[np.ndarray]] = {index: [] for index in range(len(group))}
    completed: set[int] = set()
    maximum = max(wanted)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame_number, frame in enumerate(container.decode(stream)):
            owners = wanted.get(frame_number)
            if owners:
                rgb = frame.to_ndarray(format="rgb24")
                for owner in owners:
                    buffers[owner].append(rgb)
                    if len(buffers[owner]) == len(group[owner].model_frame_indices):
                        decoded = buffers.pop(owner)
                        if causal_dual_view:
                            output = _encode_causal_clip(
                                decoded,
                                encoder=encoder,
                                codec=codec,
                                T=CONTEXT_FRAMES,
                                k=FUTURE_FRAMES,
                                stride=WINDOW_STRIDE,
                                keep_geometry=keep_geometry,
                            )
                        else:
                            output = _encode_clip(
                                decoded,
                                encoder=encoder,
                                codec=codec,
                                batch_frames=batch_frames,
                                keep_geometry=keep_geometry,
                            )
                        item = group[owner]
                        if keep_geometry:
                            if causal_dual_view:
                                item.anchor_codes = output["context_codes"]
                                item.anchor_scale = output["context_scale"]
                                item.future_codes = output["future_codes"]
                                item.future_scale = output["future_scale"]
                                item.depth_patch = output["future_depth_patch"]
                                item.depth_conf_patch = output[
                                    "future_depth_conf_patch"
                                ]
                                item.point_patch = output["future_point_patch"]
                                item.point_conf_patch = output[
                                    "future_point_conf_patch"
                                ]
                                item.pose_enc = output["future_pose_enc"]
                                item.window_starts = output["window_starts"]
                            else:
                                item.anchor_codes = output["codes"]
                                item.anchor_scale = output["scale"]
                                item.depth_patch = output["depth_patch"]
                                item.depth_conf_patch = output["depth_conf_patch"]
                                item.point_patch = output["point_patch"]
                                item.point_conf_patch = output["point_conf_patch"]
                                item.pose_enc = output["pose_enc"]
                                item.geometry_segment_id = output[
                                    "geometry_segment_id"
                                ]
                        else:
                            if causal_dual_view:
                                if not np.array_equal(
                                    item.window_starts,
                                    output["window_starts"],
                                ):
                                    raise RuntimeError(
                                        "anchor/wrist causal window identity mismatch"
                                    )
                                item.wrist_codes = output["context_codes"]
                                item.wrist_scale = output["context_scale"]
                            else:
                                item.wrist_codes = output["codes"]
                                item.wrist_scale = output["scale"]
                        completed.add(owner)
            if frame_number >= maximum:
                break
    if len(completed) != len(group):
        missing = sorted(set(range(len(group))) - completed)
        raise RuntimeError(f"decoded only {len(completed)}/{len(group)} clips from {video_path}; missing={missing[:8]}")


def _atomic_savez(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)
def _payload_matches(path: Path, payload: dict[str, Any]) -> bool:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(payload):
                return False
            return all(
                np.array_equal(np.asarray(archive[key]), np.asarray(value))
                for key, value in payload.items()
            )
    except (OSError, ValueError, KeyError):
        return False


def _publish_causal_archive(path: Path, payload: dict[str, Any]) -> str:
    """Publish without clobber; an existing archive must be array-identical."""

    validate_causal_dual_view_archive(
        payload,
        T=CONTEXT_FRAMES,
        k=FUTURE_FRAMES,
        paired_views=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not _payload_matches(path, payload):
            raise FileExistsError(f"existing causal archive is non-identical: {path}")
        return _sha256_file(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not _payload_matches(path, payload):
                raise FileExistsError(
                    f"existing causal archive is non-identical: {path}"
                )
        if not _payload_matches(path, payload):
            raise RuntimeError(f"causal archive replay validation failed: {path}")
        return _sha256_file(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_causal_item(
    item: WorkItem,
    output_root: Path,
    task_embedding: np.ndarray,
    *,
    action_audit_sha256: str,
    manifest_sha256: str,
    selection_sha256: str,
    config_sha256: str,
    codec_sha256: str,
    codec_downstream_report_sha256: str,
    v7_source: str,
) -> dict[str, Any]:
    if v7_source not in {"atomic", "composite", "mg"}:
        raise ValueError(f"unsupported RoboCasa V7 source: {v7_source!r}")
    required = (
        item.anchor_codes,
        item.anchor_scale,
        item.wrist_codes,
        item.wrist_scale,
        item.future_codes,
        item.future_scale,
        item.depth_patch,
        item.depth_conf_patch,
        item.point_patch,
        item.point_conf_patch,
        item.pose_enc,
        item.window_starts,
    )
    if any(value is None for value in required):
        raise RuntimeError(
            f"incomplete causal compact payload for {item.record.clip_hash}"
        )
    payload = {
        **causal_dual_view_metadata(
            T=CONTEXT_FRAMES,
            k=FUTURE_FRAMES,
        ),
        "producer_schema": np.asarray(
            "wm3d_v8_stage0_causal_dual_view_robocasa_producer_v1"
        ),
        "clip_hash": np.asarray(item.record.clip_hash),
        "split": np.asarray(item.record.split),
        "source": np.asarray(item.record.source),
        "v7_source": np.asarray(v7_source),
        "native_episode_id": np.asarray(item.record.native_episode_id),
        "native_start_frame": np.asarray(
            item.record.native_start_frame, dtype=np.int64
        ),
        "native_end_frame": np.asarray(
            item.record.native_end_frame, dtype=np.int64
        ),
        "native_fps": np.asarray(item.record.native_fps, dtype=np.float32),
        "embodiment_id": np.asarray(item.record.embodiment_id),
        "action_adapter_version": np.asarray(
            item.record.action.adapter_version
        ),
        "action_audit_sha256": np.asarray(action_audit_sha256),
        "manifest_sha256": np.asarray(manifest_sha256),
        "selection_sha256": np.asarray(selection_sha256),
        "config_sha256": np.asarray(config_sha256),
        "codec_sha256": np.asarray(codec_sha256),
        "codec_downstream_report_sha256": np.asarray(
            codec_downstream_report_sha256
        ),
        "source_control_hz": np.asarray(
            item.record.action.control_hz, dtype=np.float32
        ),
        "model_control_hz": np.asarray(MODEL_HZ, dtype=np.float32),
        "raw_action_kind": np.asarray(item.record.action.raw_kind),
        "raw_action_key": np.asarray(item.record.action.action_key),
        "raw_actions": item.raw_actions.astype(np.float32),
        "native_frame_indices": item.native_frame_indices.astype(np.int64),
        "model_timestamps": item.model_timestamps.astype(np.float64),
        "action_valid_mask": item.action_valid_mask.astype(np.bool_),
        "task_text": np.asarray(item.record.task_text),
        "task_emb": np.asarray(task_embedding, dtype=np.float16),
        "token_count": np.asarray(
            item.anchor_codes.shape[-2], dtype=np.int64
        ),
        "token_dim": np.asarray(2048, dtype=np.int64),
        "latent_dim": np.asarray(
            item.anchor_codes.shape[-1], dtype=np.int64
        ),
        "window_starts": item.window_starts,
        "context_codes": item.anchor_codes,
        "context_scale": item.anchor_scale,
        "wrist_context_codes": item.wrist_codes,
        "wrist_context_scale": item.wrist_scale,
        "future_codes": item.future_codes,
        "future_scale": item.future_scale,
        "future_depth_patch": item.depth_patch,
        "future_depth_conf_patch": item.depth_conf_patch,
        "future_point_patch": item.point_patch,
        "future_point_conf_patch": item.point_conf_patch,
        "future_pose_enc": item.pose_enc,
        "actions": item.actions.astype(np.float32),
        "rewards": item.rewards.astype(np.float32),
        "dones": item.dones.astype(np.bool_),
    }
    validate_causal_dual_view_archive(
        payload,
        T=CONTEXT_FRAMES,
        k=FUTURE_FRAMES,
        paired_views=True,
    )
    destination = (
        output_root / item.record.split / f"{item.record.clip_hash}.npz"
    )
    artifact_sha256 = _publish_causal_archive(destination, payload)
    return {
        "schema": str(np.asarray(payload["schema"]).item()),
        "representation": str(np.asarray(payload["representation"]).item()),
        "context_future_leakage": False,
        "target_usage": "supervision_only",
        "geometry_coordinate_frame": "first_observed_camera",
        "clip_hash": item.record.clip_hash,
        "split": item.record.split,
        "source": item.record.source,
        "v7_source": v7_source,
        "task_class": item.record.task_class,
        "path": str(destination.resolve()),
        "artifact_sha256": artifact_sha256,
        "model_frames": int(len(item.actions)),
        "T": CONTEXT_FRAMES,
        "k": FUTURE_FRAMES,
        "P": int(item.anchor_codes.shape[-2]),
        "token_D": 2048,
        "latent_dim": int(item.anchor_codes.shape[-1]),
        "windows": int(len(item.window_starts)),
        "window_starts": item.window_starts.astype(np.int64).tolist(),
        "paired_views": True,
        "action_valid": True,
        "action_adapter_version": item.record.action.adapter_version,
        "action_audit_sha256": action_audit_sha256,
        "manifest_sha256": manifest_sha256,
        "selection_sha256": selection_sha256,
        "config_sha256": config_sha256,
        "codec_sha256": codec_sha256,
        "codec_downstream_report_sha256": codec_downstream_report_sha256,
        "factual_action_audit_passed": True,
        "counterfactual_replay_passed": False,
        "pseudo_outcomes": False,
        "geometry_teacher": {
            "name": "VGGT",
            "revision": VGGT_MODEL_REVISION,
            "pseudo_teacher": True,
            "confidence_stored": True,
        },
    }





def _geometry_segments(segment_id: np.ndarray) -> list[list[int]]:
    segments: list[list[int]] = []
    start = 0
    for index in range(1, len(segment_id) + 1):
        if index == len(segment_id) or segment_id[index] != segment_id[start]:
            segments.append([start, index])
            start = index
    return segments


def _write_item(
    item: WorkItem,
    output_root: Path,
    task_embedding: np.ndarray,
    *,
    action_audit_sha256: str,
) -> dict[str, Any]:
    required = (
        item.anchor_codes,
        item.anchor_scale,
        item.wrist_codes,
        item.wrist_scale,
        item.depth_patch,
        item.depth_conf_patch,
        item.point_patch,
        item.point_conf_patch,
        item.pose_enc,
        item.geometry_segment_id,
    )
    if any(value is None for value in required):
        raise RuntimeError(f"incomplete compact cache payload for {item.record.clip_hash}")
    destination = output_root / item.record.split / f"{item.record.clip_hash}.npz"
    _atomic_savez(
        destination,
        schema=np.asarray(SCHEMA_VERSION),
        clip_hash=np.asarray(item.record.clip_hash),
        split=np.asarray(item.record.split),
        source=np.asarray(item.record.source),
        native_episode_id=np.asarray(item.record.native_episode_id),
        native_start_frame=np.asarray(item.record.native_start_frame, dtype=np.int64),
        native_end_frame=np.asarray(item.record.native_end_frame, dtype=np.int64),
        native_fps=np.asarray(item.record.native_fps, dtype=np.float32),
        embodiment_id=np.asarray(item.record.embodiment_id),
        action_adapter_version=np.asarray(item.record.action.adapter_version),
        action_audit_sha256=np.asarray(action_audit_sha256),
        source_control_hz=np.asarray(item.record.action.control_hz, dtype=np.float32),
        model_control_hz=np.asarray(MODEL_HZ, dtype=np.float32),
        raw_action_kind=np.asarray(item.record.action.raw_kind),
        raw_action_key=np.asarray(item.record.action.action_key),
        raw_actions=item.raw_actions.astype(np.float32),
        native_frame_indices=item.native_frame_indices.astype(np.int64),
        model_timestamps=item.model_timestamps.astype(np.float64),
        action_valid_mask=item.action_valid_mask.astype(np.bool_),
        task_text=np.asarray(item.record.task_text),
        task_emb=np.asarray(task_embedding, dtype=np.float16),
        anchor_codes=item.anchor_codes,
        anchor_scale=item.anchor_scale,
        wrist_codes=item.wrist_codes,
        wrist_scale=item.wrist_scale,
        depth_patch=item.depth_patch,
        depth_conf_patch=item.depth_conf_patch,
        point_patch=item.point_patch,
        point_conf_patch=item.point_conf_patch,
        pose_enc=item.pose_enc,
        geometry_segment_id=item.geometry_segment_id,
        actions=item.actions.astype(np.float32),
        rewards=item.rewards.astype(np.float32),
        dones=item.dones.astype(np.bool_),
    )
    segments = _geometry_segments(item.geometry_segment_id)
    required = CONTEXT_FRAMES + FUTURE_FRAMES
    valid_window_starts = [
        start
        for segment_start, segment_stop in segments
        for start in range(
            segment_start,
            max(segment_start, segment_stop - required + 1),
            WINDOW_STRIDE,
        )
    ]
    return {
        "schema": SCHEMA_VERSION,
        "clip_hash": item.record.clip_hash,
        "split": item.record.split,
        "source": item.record.source,
        "task_class": item.record.task_class,
        "path": str(destination.resolve()),
        "model_frames": int(len(item.actions)),
        "windows": len(valid_window_starts),
        "geometry_segments": segments,
        "paired_views": True,
        "action_valid": True,
        "action_adapter_version": item.record.action.adapter_version,
        "action_audit_sha256": action_audit_sha256,
        "factual_action_audit_passed": True,
        "counterfactual_replay_passed": False,
        "pseudo_outcomes": False,
        "geometry_teacher": {
            "name": "VGGT",
            "revision": VGGT_MODEL_REVISION,
            "pseudo_teacher": True,
            "confidence_stored": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--action-audit", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--codec-downstream-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--batch-frames", type=int, default=32)
    parser.add_argument("--rgb-sidecar-index", type=Path)
    parser.add_argument(
        "--causal-dual-view",
        action="store_true",
        help="write the V8 observed-context/target-only schema",
    )
    parser.add_argument(
        "--v7-source",
        choices=("atomic", "composite", "mg"),
        help="RoboCasa partition identity required by causal dual-view caches",
    )
    parser.add_argument(
        "--proof-allow-legacy-action-audit",
        action="store_true",
        help="proof-only compatibility; formal caches require explicit factual_action_audit",
    )
    parser.add_argument(
        "--proof-allow-nonstrict-codec",
        action="store_true",
        help="proof-only compatibility; formal caches require a train-split-only codec",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-task-emb", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise SystemExit("num-shards must be >=1")
    if not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("shard-id must satisfy 0 <= shard-id < num-shards")
    if args.causal_dual_view and args.v7_source is None:
        raise SystemExit(
            "--causal-dual-view requires an explicit --v7-source partition"
        )
    if args.causal_dual_view and args.rgb_sidecar_index is None:
        raise SystemExit(
            "--causal-dual-view requires --rgb-sidecar-index for RGB supervision"
        )

    downstream = json.loads(args.codec_downstream_report.read_text())
    if not bool(downstream.get("formal_cache_allowed")):
        raise SystemExit("token codec downstream gate failed; refusing to build formal cache")
    if not bool(downstream.get("strict_train_split")) and not args.proof_allow_nonstrict_codec:
        raise SystemExit(
            "token codec was not fit on the immutable V7 train split; refusing formal cache"
        )
    manifest_records = list(read_v7_manifest(args.manifest))
    rgb_sidecar_sha256 = None
    if args.causal_dual_view:
        manifest_records, rgb_sidecar_sha256 = _filter_records_by_rgb_sidecar(
            manifest_records, args.rgb_sidecar_index
        )
    all_records = _select_records(manifest_records, int(args.max_clips))
    if not all_records:
        raise SystemExit("manifest selection contains no action-valid clips")
    selection_sha256 = _selection_sha256(all_records)
    metadata, file_base = _load_episode_metadata(args.root)
    shard_records, shard_costs, shard_group_counts = _partition_by_video_group(
        all_records,
        args.root,
        metadata,
        int(args.num_shards),
    )
    records = shard_records[int(args.shard_id)]
    if not records:
        raise SystemExit(
            f"shard {args.shard_id}/{args.num_shards} has no assigned video groups"
        )
    index_path = _shard_index_path(
        args.index, int(args.shard_id), int(args.num_shards)
    )
    if not args.causal_dual_view and int(args.batch_frames) < CONTEXT_FRAMES + FUTURE_FRAMES:
        raise SystemExit(
            f"batch-frames must be >= {CONTEXT_FRAMES + FUTURE_FRAMES} so every "
            "training window stays inside one VGGT world-coordinate gauge"
        )
    action_audit_sha256 = _sha256_file(args.action_audit)
    manifest_sha256 = _sha256_file(args.manifest)
    codec_sha256 = _sha256_file(args.codec)
    codec_downstream_report_sha256 = _sha256_file(args.codec_downstream_report)
    config_identity = {
        "schema": (
            "wm3d_v8_stage0_causal_dual_view_robocasa_producer_v1"
            if args.causal_dual_view
            else SCHEMA_VERSION
        ),
        "root": str(args.root.resolve()),
        "manifest_sha256": manifest_sha256,
        "selection_sha256": selection_sha256,
        "codec_sha256": codec_sha256,
        "codec_downstream_report_sha256": codec_downstream_report_sha256,
        "action_audit_sha256": action_audit_sha256,
        "context_frames": CONTEXT_FRAMES,
        "future_frames": FUTURE_FRAMES,
        "window_stride": WINDOW_STRIDE,
        "vggt_revision": VGGT_MODEL_REVISION,
    }
    if args.causal_dual_view:
        config_identity["v7_source"] = str(args.v7_source)
        config_identity["rgb_sidecar_index_sha256"] = rgb_sidecar_sha256
    config_sha256 = hashlib.sha256(
        json.dumps(config_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    adapter = _load_adapter(
        args.action_audit,
        allow_legacy_proof_audit=bool(args.proof_allow_legacy_action_audit),
    )
    items = _prepare_items(records, args.root, metadata, file_base, adapter)
    if not items:
        raise SystemExit("no action-valid clips with at least 24 model-rate frames")

    device = torch.device(args.device)
    # Resolve and encode language before the expensive geometry pass so a
    # missing local text checkpoint fails fast instead of discarding minutes
    # of in-memory VGGT results.  The two large encoders never coexist.
    task_encoder = None if args.skip_task_emb else QwenVLEmbed(device=str(device))
    task_cache: dict[str, np.ndarray] = {}
    for item in items:
        if item.record.task_text in task_cache:
            continue
        if task_encoder is None:
            task_cache[item.record.task_text] = np.zeros(2048, dtype=np.float16)
        else:
            task_cache[item.record.task_text] = (
                task_encoder.embed(item.record.task_text).numpy().astype(np.float16)
            )
    del task_encoder
    torch.cuda.empty_cache()
    gc.collect()

    codec = _load_codec(args.codec, device)
    encoder = VGGTEncoder(
        device=str(device),
        return_depth=True,
        return_depth_conf=True,
        return_geom_extra=True,
        model_revision=VGGT_MODEL_REVISION,
        local_files_only=True,
    )
    group_keys: dict[tuple[Path, Path], list[WorkItem]] = {}
    for item in items:
        anchor_path = _video_path(args.root, item.metadata, CAMERA_ROLES["anchor"])
        wrist_path = _video_path(args.root, item.metadata, CAMERA_ROLES["wrist"])
        group_keys.setdefault((anchor_path, wrist_path), []).append(item)
    index_rows: list[dict[str, Any]] = []
    partial_index_path = index_path.with_name(index_path.name + ".partial")
    for anchor_path, wrist_path in sorted(
        group_keys, key=lambda paths: (str(paths[0]), str(paths[1]))
    ):
        group = group_keys[(anchor_path, wrist_path)]
        _encode_video_group(
            anchor_path,
            group,
            camera=CAMERA_ROLES["anchor"],
            encoder=encoder,
            codec=codec,
            batch_frames=int(args.batch_frames),
            keep_geometry=True,
            causal_dual_view=bool(args.causal_dual_view),
        )
        _encode_video_group(
            wrist_path,
            group,
            camera=CAMERA_ROLES["wrist"],
            encoder=encoder,
            codec=codec,
            batch_frames=int(args.batch_frames),
            keep_geometry=False,
            causal_dual_view=bool(args.causal_dual_view),
        )
        for item in group:
            if args.causal_dual_view:
                row = _write_causal_item(
                    item,
                    args.output_root,
                    task_cache[item.record.task_text],
                    action_audit_sha256=action_audit_sha256,
                    manifest_sha256=manifest_sha256,
                    selection_sha256=selection_sha256,
                    config_sha256=config_sha256,
                    codec_sha256=codec_sha256,
                    codec_downstream_report_sha256=(
                        codec_downstream_report_sha256
                    ),
                    v7_source=str(args.v7_source),
                )
            else:
                row = _write_item(
                    item,
                    args.output_root,
                    task_cache[item.record.task_text],
                    action_audit_sha256=action_audit_sha256,
                )
            index_rows.append(row)
            item.anchor_codes = None
            item.anchor_scale = None
            item.future_codes = None
            item.future_scale = None
            item.window_starts = None
            item.wrist_codes = None
            item.wrist_scale = None
            item.depth_patch = None
            item.depth_conf_patch = None
            item.point_patch = None
            item.point_conf_patch = None
            item.pose_enc = None
            item.geometry_segment_id = None
        if not args.causal_dual_view:
            _atomic_write_jsonl(partial_index_path, index_rows)

    del encoder
    torch.cuda.empty_cache()
    gc.collect()
    if args.causal_dual_view:
        index_text = "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in sorted(
                index_rows,
                key=lambda value: (value["split"], value["clip_hash"]),
            )
        )
        index_sha256 = _publish_text_no_clobber(index_path, index_text)
    else:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial_index_path, index_path)
        index_sha256 = _sha256_file(index_path)
    report = {
        "schema": (
            "wm3d_v8_stage0_causal_dual_view_robocasa_producer_v1"
            if args.causal_dual_view
            else SCHEMA_VERSION
        ),
        "pass": True,
        "causal_dual_view": bool(args.causal_dual_view),
        "representation": (
            "wm3d_v8_vggt_observed_context_target_split_v1"
            if args.causal_dual_view
            else None
        ),
        "v7_source": str(args.v7_source) if args.causal_dual_view else None,
        "clips": len(index_rows),
        "windows": sum(int(row["windows"]) for row in index_rows),
        "splits": {split: sum(row["split"] == split for row in index_rows) for split in ("train", "val", "test")},
        "paired_views": True,
        "rgb_copied": False,
        "task_embedding_real": not args.skip_task_emb,
        "codec_downstream_gate": str(args.codec_downstream_report.resolve()),
        "manifest_sha256": manifest_sha256,
        "codec_sha256": codec_sha256,
        "codec_downstream_report_sha256": codec_downstream_report_sha256,
        "selection_sha256": selection_sha256,
        "index_sha256": index_sha256,
        "config_sha256": config_sha256,
        "global_selected_clips": len(all_records),
        "rgb_sidecar_index": (
            str(args.rgb_sidecar_index.resolve())
            if args.causal_dual_view else None
        ),
        "rgb_sidecar_index_sha256": rgb_sidecar_sha256,
        "rgb_sidecar_coverage_passed": bool(args.causal_dual_view),
        "action_audit": str(args.action_audit.resolve()),
        "action_audit_sha256": action_audit_sha256,
        "factual_action_audit_passed": True,
        "counterfactual_replay_passed": False,
        "geometry_teacher": {
            "name": "VGGT",
            "revision": VGGT_MODEL_REVISION,
            "pseudo_teacher": True,
        },
        "geometry_segment_frames": None if args.causal_dual_view else int(args.batch_frames),
        "sharding": {
            "algorithm": "whole_video_pair_lpt_v1",
            "shard_id": int(args.shard_id),
            "num_shards": int(args.num_shards),
            "assigned_video_groups": int(shard_group_counts[int(args.shard_id)]),
            "assigned_clips": len(records),
            "estimated_model_frames": int(shard_costs[int(args.shard_id)]),
            "all_shard_estimated_model_frames": [int(value) for value in shard_costs],
        },
    }
    report_path = index_path.with_suffix(".report.json")
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.causal_dual_view:
        _publish_text_no_clobber(report_path, report_text)
    else:
        report_path.write_text(report_text)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
