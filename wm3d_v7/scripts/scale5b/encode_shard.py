#!/usr/bin/env python3
"""Encode one deterministic shard of the native WM3D-V7 5B dataset."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import uuid
from typing import Any, Iterable, Mapping

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.torch import save_file
import torch
import torch.nn.functional as F

from wm3d_v3.data.scale5b_action import (
    ActionNormalization,
    RawActionSeries,
    align_auxiliary_tokens,
    align_grouped_actions,
)
from wm3d_v3.data.scale5b_assets import (
    load_asset_receipt,
    verify_asset_bundle,
)
from wm3d_v3.data.scale5b_codec import JpegPackWriter, quantize_per_vector
from wm3d_v3.data.scale5b_contracts import (
    ContractError,
    DatasetContract,
    atomic_write_json,
    canonical_sha256,
    load_contract,
    resolve_real_directory,
    resolve_regular_file,
    sha256_file,
)
from wm3d_v3.data.scale5b_sources import EpisodeDescriptor, plan_shard
from wm3d_v3.encoders.native5b_vggt import Native5BVGGTEncoder


PART_SCHEMA = "wm3d_v7_native5b_encoded_part_v2"
PART_COMMIT_SCHEMA = "wm3d_v7_native5b_encoded_part_commit_v2"


@dataclass(frozen=True)
class Segment:
    episode: EpisodeDescriptor
    frame_start: int
    frame_stop: int

    @property
    def frames(self) -> int:
        return self.frame_stop - self.frame_start

    @property
    def segment_id(self) -> str:
        return (
            f"{self.episode.episode_id}:"
            f"{self.frame_start:09d}-{self.frame_stop:09d}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--episode-plan", type=Path, required=True)
    parser.add_argument("--action-stats", type=Path, required=True)
    parser.add_argument("--task-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--max-part-frames", type=int, default=512)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--encoder-batch-frames", type=int, default=4)
    parser.add_argument("--encoder-input-size", type=int, default=518)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--vggt-model", default="facebook/VGGT-1B")
    parser.add_argument("--vggt-revision", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_plan(path: Path, shard_id: int, num_shards: int) -> list[EpisodeDescriptor]:
    result: list[EpisodeDescriptor] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            episode = EpisodeDescriptor.from_mapping(json.loads(line))
            if plan_shard(episode.episode_id, num_shards) == shard_id:
                result.append(episode)
    if not result:
        raise ValueError(f"encode shard {shard_id}/{num_shards} has no episodes")
    return sorted(result, key=lambda item: (item.source, item.episode_id))


def _segment_lengths(frames: int, maximum: int, minimum: int) -> list[tuple[int, int]]:
    if frames < minimum:
        return []
    count = max(1, (frames + maximum - 1) // maximum)
    while count > 1 and frames // count < minimum:
        count -= 1
    boundaries = [round(index * frames / count) for index in range(count + 1)]
    result = [(boundaries[index], boundaries[index + 1]) for index in range(count)]
    if any(stop - start < minimum or stop - start > maximum for start, stop in result):
        raise ValueError(f"cannot segment {frames} frames into [{minimum},{maximum}]")
    return result


def _part_plan(
    episodes: Iterable[EpisodeDescriptor],
    *,
    maximum: int,
    minimum: int,
) -> list[list[Segment]]:
    segments: list[Segment] = []
    for episode in episodes:
        frames = int(np.floor(episode.duration_seconds * 5.0 + 1.0e-9))
        segments.extend(
            Segment(episode, start, stop)
            for start, stop in _segment_lengths(frames, maximum, minimum)
        )
    parts: list[list[Segment]] = []
    current: list[Segment] = []
    current_frames = 0
    for segment in segments:
        if current and current_frames + segment.frames > maximum:
            parts.append(current)
            current = []
            current_frames = 0
        current.append(segment)
        current_frames += segment.frames
    if current:
        parts.append(current)
    if not parts:
        raise ValueError("encode shard contains no segment long enough for one window")
    return parts


def _decode_nearest(
    path: Path,
    targets: np.ndarray,
    *,
    expected_fps: float,
) -> torch.Tensor:
    if targets.ndim != 1 or targets.size == 0 or not np.all(np.diff(targets) > 0):
        raise ValueError("video target timestamps must be increasing")
    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        seek_seconds = max(0.0, float(targets[0]) - 1.0 / max(expected_fps, 1.0))
        container.seek(
            int(seek_seconds / time_base),
            stream=stream,
            backward=True,
            any_frame=False,
        )
        fallback_index = 0
        stop = float(targets[-1]) + 2.0 / max(expected_fps, 1.0)
        for frame in container.decode(stream):
            if frame.pts is not None:
                timestamp = float(frame.pts * stream.time_base)
            elif frame.time is not None:
                timestamp = float(frame.time)
            else:
                timestamp = fallback_index / expected_fps
            fallback_index += 1
            if timestamp + 1.0 / expected_fps < targets[0]:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            timestamps.append(timestamp)
            if timestamp >= stop:
                break
    if not frames:
        raise ValueError(f"video decode produced no frames: {path}")
    times = np.asarray(timestamps, dtype=np.float64)
    right = np.searchsorted(times, targets, side="left")
    right = np.clip(right, 0, len(times) - 1)
    left = np.clip(right - 1, 0, len(times) - 1)
    choose_right = np.abs(times[right] - targets) < np.abs(times[left] - targets)
    indices = np.where(choose_right, right, left)
    tolerance = max(0.075, 0.75 / expected_fps)
    error = np.abs(times[indices] - targets)
    if float(error.max()) > tolerance:
        raise ValueError(
            f"video timestamp error {float(error.max()):.6f}s exceeds "
            f"{tolerance:.6f}s: {path}"
        )
    array = np.stack([frames[int(index)] for index in indices])
    return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()


def _read_episode_actions(
    episode: EpisodeDescriptor,
) -> tuple[
    np.ndarray,
    dict[str, RawActionSeries],
    dict[str, RawActionSeries],
]:
    root = resolve_real_directory(
        Path(episode.raw_root),
        f"{episode.episode_id} raw root",
    )
    path = resolve_regular_file(root, episode.data_relative_path)
    columns = sorted(
        {episode.episode_column, episode.timestamp_column}
        | {spec.column for spec in episode.action_columns}
        | {spec.column for spec in episode.auxiliary_columns}
    )
    table = pq.read_table(path, columns=columns).slice(
        episode.data_row_start,
        episode.data_row_stop - episode.data_row_start,
    )
    payload = table.to_pydict()
    episode_column = np.asarray(payload[episode.episode_column])
    if episode_column.size and not np.all(episode_column == episode.episode_index):
        raise ValueError(f"episode row binding drift for {episode.episode_id}")
    timestamps = np.asarray(payload[episode.timestamp_column], dtype=np.float64)
    if timestamps.size != table.num_rows or not np.isfinite(timestamps).all():
        raise ValueError(f"invalid timestamps for {episode.episode_id}")
    timestamps -= timestamps[0]
    if not np.all(np.diff(timestamps) > 0):
        # Some LeRobot exports round timestamps.  Frame index is the canonical
        # fallback, and its FPS is bound in the episode plan.
        timestamps = np.arange(table.num_rows, dtype=np.float64) / episode.source_fps
    groups: dict[str, RawActionSeries] = {}
    for spec in episode.action_columns:
        values = np.asarray(payload[spec.column], dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        values = values[:, spec.indices]
        valid = np.isfinite(values)
        if not bool(valid.all()):
            raise ValueError(
                f"non-finite factual action {spec.group_name} in "
                f"{episode.episode_id}"
            )
        groups[spec.group_name] = RawActionSeries(
            timestamps=timestamps,
            values=values,
            valid=valid,
        )
    auxiliary: dict[str, RawActionSeries] = {}
    for spec in episode.auxiliary_columns:
        values = np.asarray(payload[spec.column], dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        values = values[:, spec.indices]
        valid = np.isfinite(values)
        # Missing force/tactile/LiDAR samples are represented only by the
        # validity plane.  Sanitizing the storage value prevents np.interp
        # from propagating NaNs into adjacent timestamps.
        values = np.where(valid, values, 0.0)
        auxiliary[spec.modality_name] = RawActionSeries(
            timestamps=timestamps,
            values=values,
            valid=valid,
        )
    return timestamps, groups, auxiliary


def _normalizations(
    value: Mapping[str, Any],
    episode: EpisodeDescriptor,
) -> dict[str, ActionNormalization]:
    groups = value["groups"]
    result: dict[str, ActionNormalization] = {}
    for spec in episode.action_columns:
        key = f"{episode.embodiment}::{spec.group_name}"
        item = groups[key]
        result[spec.group_name] = ActionNormalization(
            center=np.asarray(item["center"], dtype=np.float64),
            scale=np.asarray(item["scale"], dtype=np.float64),
            clip=float(item["clip"]),
        )
    return result


def _auxiliary_normalizations(
    value: Mapping[str, Any],
    episode: EpisodeDescriptor,
) -> dict[str, ActionNormalization]:
    groups = value["groups"]
    result: dict[str, ActionNormalization] = {}
    for spec in episode.auxiliary_columns:
        key = f"{episode.embodiment}::aux::{spec.modality_name}"
        item = groups[key]
        result[spec.modality_name] = ActionNormalization(
            center=np.asarray(item["center"], dtype=np.float64),
            scale=np.asarray(item["scale"], dtype=np.float64),
            clip=float(item["clip"]),
        )
    return result


def _resize_views(value: torch.Tensor, size: int) -> torch.Tensor:
    frames, views, channels = value.shape[:3]
    resized = F.interpolate(
        value.reshape(frames * views, channels, value.shape[-2], value.shape[-1]).float(),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized.reshape(frames, views, channels, size, size).div_(255.0)


def _encode_segment(
    segment: Segment,
    *,
    contract: DatasetContract,
    embodiment: Any,
    stats: Mapping[str, Any],
    encoder: Native5BVGGTEncoder,
    encoder_batch_frames: int,
    encoder_input_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    episode = segment.episode
    relative_times = (
        np.arange(segment.frame_start, segment.frame_stop, dtype=np.float64)
        / contract.feature_fps
    )
    view_values: list[torch.Tensor | None] = []
    view_available: list[bool] = []
    root = resolve_real_directory(
        Path(episode.raw_root),
        f"{episode.episode_id} raw root",
    )
    for view in episode.views:
        if view.relative_path is None:
            view_values.append(None)
            view_available.append(False)
            continue
        path = resolve_regular_file(root, view.relative_path)
        targets = view.start_seconds + relative_times
        view_values.append(
            _decode_nearest(path, targets, expected_fps=episode.source_fps)
        )
        view_available.append(True)
    exemplar = next(value for value in view_values if value is not None)
    images = torch.stack(
        [
            value if value is not None else torch.zeros_like(exemplar)
            for value in view_values
        ],
        dim=1,
    )
    images = _resize_views(images, encoder_input_size)
    camera_mask = torch.tensor(view_available, dtype=torch.bool)
    frame_view_mask = camera_mask[None].expand(segment.frames, -1).contiguous()
    encoded_parts: dict[str, list[torch.Tensor]] = {}
    for start in range(0, segment.frames, encoder_batch_frames):
        chunk = images[start : start + encoder_batch_frames].unsqueeze(0)
        chunk_mask = frame_view_mask[
            start : start + encoder_batch_frames
        ].unsqueeze(0)
        output = encoder(
            chunk.to(next(encoder.parameters()).device),
            view_mask=chunk_mask.to(next(encoder.parameters()).device),
        )
        for name, tensor in output.items():
            encoded_parts.setdefault(name, []).append(tensor[0].cpu())
    encoded = {
        name: torch.cat(parts, dim=0).contiguous()
        for name, parts in encoded_parts.items()
    }
    _timestamps, series, auxiliary_series = _read_episode_actions(episode)
    action = align_grouped_actions(
        visual_timestamps=relative_times,
        group_series=series,
        embodiment=embodiment,
        normalizations=_normalizations(stats, episode),
        max_groups=contract.max_action_groups,
        max_action_dim=contract.max_action_dim,
        action_substeps=contract.action_substeps,
        feature_fps=contract.feature_fps,
    )
    auxiliary = align_auxiliary_tokens(
        visual_timestamps=relative_times,
        modality_series=auxiliary_series,
        embodiment=embodiment,
        normalizations=_auxiliary_normalizations(stats, episode),
        max_aux_tokens=contract.max_aux_tokens,
        aux_dim=contract.aux_dim,
        max_aux_type_id=contract.max_aux_type_id,
    )
    return encoded, {**action, **auxiliary}


def _atomic_safetensors(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    save_file({name: value.contiguous() for name, value in tensors.items()}, temporary)
    descriptor = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _file_evidence(root: Path, names: Iterable[str]) -> dict[str, dict[str, Any]]:
    result = {}
    for name in sorted(names):
        path = root / name
        result[name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def _verify_existing_part(
    path: Path,
    *,
    expected_part_name: str,
    expected_part_index: int,
    expected_shard_id: int,
    expected_num_shards: int,
    expected_lineage: Mapping[str, str],
) -> bool:
    expected_payload = {
        "features.safetensors",
        "actions.safetensors",
        "rgb.jpgpack",
        "windows.parquet",
    }
    try:
        if path.is_symlink() or not path.is_dir():
            return False
        commit_path = resolve_regular_file(path, "COMMITTED.json")
        manifest_path = resolve_regular_file(path, "manifest.json")
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files")
        if not isinstance(files, dict) or set(files) != expected_payload:
            return False
        actual_entries = {item.name for item in path.iterdir()}
        if actual_entries != expected_payload | {
            "manifest.json",
            "COMMITTED.json",
        }:
            return False
        if (
            commit.get("schema") != PART_COMMIT_SCHEMA
            or commit.get("part_name") != expected_part_name
            or manifest.get("schema") != PART_SCHEMA
            or manifest.get("part_name") != expected_part_name
            or int(manifest.get("part_index", -1)) != expected_part_index
            or int(manifest.get("worker_shard_id", -1))
            != expected_shard_id
            or int(manifest.get("worker_num_shards", -1))
            != expected_num_shards
            or int(manifest.get("frames", 0)) <= 0
            or int(manifest.get("windows", 0)) <= 0
            or sha256_file(manifest_path) != commit.get("manifest_sha256")
            or canonical_sha256(manifest)
            != commit.get("manifest_content_sha256")
        ):
            return False
        for key, value in expected_lineage.items():
            if manifest.get(key) != value:
                return False
        for name, evidence in files.items():
            file_path = resolve_regular_file(path, name)
            if (
                file_path.stat().st_size != int(evidence["size"])
                or sha256_file(file_path) != evidence["sha256"]
            ):
                return False
    except (
        ContractError,
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ):
        return False
    return True


def _publish_part(
    *,
    part: list[Segment],
    part_index: int,
    args: argparse.Namespace,
    contract: DatasetContract,
    stats: Mapping[str, Any],
    task_ids: Mapping[str, int],
    encoder: Native5BVGGTEncoder,
    lineage: Mapping[str, str],
    embodiments: Mapping[str, Any],
) -> dict[str, Any]:
    part_name = f"part-{args.shard_id:05d}-{part_index:06d}"
    parts_root = args.output_root.resolve() / "payload" / "parts"
    parts_root.mkdir(parents=True, exist_ok=True)
    final = parts_root / part_name
    if final.exists():
        if _verify_existing_part(
            final,
            expected_part_name=part_name,
            expected_part_index=part_index,
            expected_shard_id=args.shard_id,
            expected_num_shards=args.num_shards,
            expected_lineage=lineage,
        ):
            manifest = json.loads(
                (final / "manifest.json").read_text(encoding="utf-8")
            )
            return {
                "part": part_name,
                "frames": int(manifest["frames"]),
                "windows": int(manifest["windows"]),
            }
        raise FileExistsError(f"existing encoded part failed verification: {final}")
    temporary = parts_root / f".{part_name}.incomplete.{uuid.uuid4().hex}"
    temporary.mkdir(mode=0o750)
    jpeg_writer = JpegPackWriter(
        temporary / "rgb.jpgpack",
        quality=args.jpeg_quality,
    )
    feature_buffers: dict[str, list[torch.Tensor]] = {}
    action_buffers: dict[str, list[torch.Tensor]] = {}
    index_rows: list[dict[str, Any]] = []
    frame_cursor = 0
    segment_records: list[dict[str, Any]] = []
    try:
        for segment in part:
            encoded, action = _encode_segment(
                segment,
                contract=contract,
                embodiment=embodiments[segment.episode.embodiment],
                stats=stats,
                encoder=encoder,
                encoder_batch_frames=args.encoder_batch_frames,
                encoder_input_size=args.encoder_input_size,
            )
            view_q, view_scale = quantize_per_vector(encoded["view_tokens"])
            summary_q, summary_scale = quantize_per_vector(encoded["frame_summary"])
            for name, value in {
                "view_tokens_q": view_q,
                "view_tokens_scale": view_scale,
                "view_mask": encoded["view_mask"].to(torch.bool),
                "depth": encoded["depth"].to(torch.float16),
                "point": encoded["point"].to(torch.float16),
                "geometry_confidence": encoded["geometry_confidence"].to(torch.float16),
                "camera_pose": encoded["camera_pose"].to(torch.float32),
                "frame_summary_q": summary_q,
                "frame_summary_scale": summary_scale,
                "aux_tokens": action["aux_tokens"].to(torch.float16),
                "aux_mask": action["aux_mask"].to(torch.bool),
            }.items():
                feature_buffers.setdefault(name, []).append(value)
            for frame in encoded["rgb"]:
                jpeg_writer.append(frame)
            for name in (
                "action_values",
                "action_dim_mask",
                "contact",
                "contact_mask",
            ):
                action_buffers.setdefault(name, []).append(action[name])
            episode_start = frame_cursor
            episode_stop = frame_cursor + segment.frames
            task_id = task_ids[segment.episode.task_text]
            for local_start in range(
                0,
                segment.frames - (contract.T + contract.K) + 1,
                args.window_stride,
            ):
                offset = frame_cursor + local_start
                index_rows.append(
                    {
                        "window_id": hashlib.sha256(
                            f"{segment.segment_id}:{local_start}".encode()
                        ).hexdigest(),
                        "episode_id": segment.segment_id,
                        "source": segment.episode.source,
                        "split": segment.episode.split,
                        "feature_shard": (
                            f"payload/parts/{part_name}/features.safetensors"
                        ),
                        "action_shard": (
                            f"payload/parts/{part_name}/actions.safetensors"
                        ),
                        "rgb_pack": f"payload/parts/{part_name}/rgb.jpgpack",
                        "frame_offset": offset,
                        "action_offset": offset,
                        "frame_count": contract.T + contract.K,
                        "episode_frame_start": episode_start,
                        "episode_frame_stop": episode_stop,
                        "task_id": task_id,
                        "embodiment_id": int(
                            embodiments[segment.episode.embodiment].embodiment_id
                        ),
                        "action_group_ids": action["action_group_ids"].tolist(),
                        "action_group_mask": action["action_group_mask"].tolist(),
                    }
                )
            segment_records.append(
                {
                    "segment_id": segment.segment_id,
                    "source": segment.episode.source,
                    "split": segment.episode.split,
                    "frames": segment.frames,
                    "windows": sum(
                        row["episode_id"] == segment.segment_id for row in index_rows
                    ),
                }
            )
            frame_cursor = episode_stop
        offsets, lengths = jpeg_writer.close()
        feature_buffers["rgb_offsets"] = [offsets]
        feature_buffers["rgb_lengths"] = [lengths]
        feature = {
            name: torch.cat(values, dim=0)
            for name, values in feature_buffers.items()
        }
        actions = {
            name: torch.cat(values, dim=0)
            for name, values in action_buffers.items()
        }
        _atomic_safetensors(temporary / "features.safetensors", feature)
        _atomic_safetensors(temporary / "actions.safetensors", actions)
        table = pa.Table.from_pylist(index_rows)
        pq.write_table(
            table,
            temporary / "windows.parquet",
            compression="zstd",
            row_group_size=8192,
            write_statistics=True,
        )
        for name in ("features.safetensors", "actions.safetensors", "windows.parquet"):
            descriptor = os.open(temporary / name, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        files = _file_evidence(
            temporary,
            (
                "features.safetensors",
                "actions.safetensors",
                "rgb.jpgpack",
                "windows.parquet",
            ),
        )
        manifest = {
            "schema": PART_SCHEMA,
            "part_name": part_name,
            "part_index": part_index,
            "worker_shard_id": args.shard_id,
            "worker_num_shards": args.num_shards,
            "episode_plan_sha256": lineage["episode_plan_sha256"],
            "dataset_contract_sha256": lineage["dataset_contract_sha256"],
            "action_stats_sha256": lineage["action_stats_sha256"],
            "task_index_sha256": lineage["task_index_sha256"],
            "encoder_asset_receipt_sha256": lineage[
                "encoder_asset_receipt_sha256"
            ],
            "vggt_model": args.vggt_model,
            "vggt_revision": args.vggt_revision,
            "token_codec": "symmetric_int8_per_vector_fp16_scale_v1",
            "rgb_codec": f"jpeg_q{args.jpeg_quality}_independent_records_v1",
            "auxiliary_codec": (
                "type_onehot_plus_robust_values_and_validity_fp16_v1"
            ),
            "frames": frame_cursor,
            "windows": len(index_rows),
            "segments": segment_records,
            "files": files,
        }
        atomic_write_json(temporary / "manifest.json", manifest, exclusive=True)
        commit = {
            "schema": PART_COMMIT_SCHEMA,
            "part_name": part_name,
            "manifest_sha256": sha256_file(temporary / "manifest.json"),
            "manifest_content_sha256": canonical_sha256(manifest),
        }
        atomic_write_json(temporary / "COMMITTED.json", commit, exclusive=True)
        directory = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        os.replace(temporary, final)
        directory = os.open(parts_root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return {
            "part": part_name,
            "frames": frame_cursor,
            "windows": len(index_rows),
        }
    except Exception:
        # The incomplete directory is intentional forensic/restart evidence.
        # It is never consumed because it has no final name/commit.
        jpeg_writer.__exit__(None, None, None)
        raise


def main() -> None:
    args = parse_args()
    if (
        not 0 <= args.shard_id < args.num_shards
        or args.max_part_frames < 40
        or args.window_stride <= 0
        or args.encoder_batch_frames <= 0
    ):
        raise ValueError("invalid encode-shard arguments")
    contract_path = resolve_regular_file(
        args.dataset_contract.parent,
        args.dataset_contract.name,
    )
    plan_path = resolve_regular_file(
        args.episode_plan.parent,
        args.episode_plan.name,
    )
    stats_path = resolve_regular_file(
        args.action_stats.parent,
        args.action_stats.name,
    )
    task_path = resolve_regular_file(
        args.task_index.parent,
        args.task_index.name,
    )
    args.output_root = resolve_real_directory(
        args.output_root,
        "dataset output root",
    )
    contract = load_contract(contract_path)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    task_index = json.loads(task_path.read_text(encoding="utf-8"))
    if stats.get("schema") != "wm3d_v7_native5b_action_stats_v2":
        raise ValueError("action-statistics schema mismatch")
    if task_index.get("schema") != "wm3d_v7_native5b_task_bank_v1":
        raise ValueError("task-index schema mismatch")
    asset_report = verify_asset_bundle(args.asset_root, deep=False)
    asset_root = resolve_real_directory(args.asset_root, "encoder asset root")
    asset_receipt = asset_report["receipt"]
    asset_sha = canonical_sha256(asset_receipt)
    vggt_asset = asset_receipt["assets"]["vggt_model"]
    if (
        vggt_asset["repo_id"] != args.vggt_model
        or vggt_asset["revision"] != args.vggt_revision
    ):
        raise ValueError("VGGT model/revision differs from the asset receipt")
    control_asset_path = resolve_regular_file(
        args.output_root,
        "control/encoder_asset_receipt.json",
    )
    control_asset = load_asset_receipt(control_asset_path)
    if canonical_sha256(control_asset) != asset_sha:
        raise ValueError("dataset control and mounted encoder assets differ")
    if task_index.get("encoder_asset_receipt_sha256") != asset_sha:
        raise ValueError("task bank and encoder assets differ")
    source_asset = asset_receipt["assets"]["vggt_source"]
    os.environ["WM3D_VGGT_SOURCE_ROOT"] = str(
        resolve_real_directory(
            asset_root / str(source_asset["path"]),
            "VGGT source snapshot",
        )
    )
    os.environ["WM3D_VGGT_MODEL_SNAPSHOT"] = str(
        resolve_real_directory(
            asset_root / str(vggt_asset["path"]),
            "VGGT model snapshot",
        )
    )
    plan_sha = sha256_file(plan_path)
    if stats.get("episode_plan_sha256") != plan_sha:
        raise ValueError("action-statistics/episode-plan lineage mismatch")
    if task_index.get("episode_plan_sha256") != plan_sha:
        raise ValueError("task-bank/episode-plan lineage mismatch")
    task_ids = {str(row["text"]): int(row["task_id"]) for row in task_index["tasks"]}
    episodes = _load_plan(plan_path, args.shard_id, args.num_shards)
    missing_tasks = {episode.task_text for episode in episodes}.difference(task_ids)
    if missing_tasks:
        raise ValueError(f"task bank misses {len(missing_tasks)} episode tasks")
    parts = _part_plan(
        episodes,
        maximum=args.max_part_frames,
        minimum=contract.T + contract.K,
    )
    encoder = Native5BVGGTEncoder(
        device=args.device,
        model_name=args.vggt_model,
        model_revision=args.vggt_revision,
        local_files_only=True,
        token_grid=12,
        target_rgb_size=384,
        dtype=torch.bfloat16,
    ).eval()
    embodiments = {item.name: item for item in contract.embodiments}
    lineage = {
        "episode_plan_sha256": plan_sha,
        "dataset_contract_sha256": contract.sha256,
        "action_stats_sha256": sha256_file(stats_path),
        "task_index_sha256": sha256_file(task_path),
        "encoder_asset_receipt_sha256": asset_sha,
    }
    reports = [
        _publish_part(
            part=part,
            part_index=part_index,
            args=args,
            contract=contract,
            stats=stats,
            task_ids=task_ids,
            encoder=encoder,
            lineage=lineage,
            embodiments=embodiments,
        )
        for part_index, part in enumerate(parts)
    ]
    summary = {
        "schema": "wm3d_v7_native5b_encode_worker_receipt_v1",
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        **lineage,
        "vggt_model": args.vggt_model,
        "vggt_revision": args.vggt_revision,
        "parts": reports,
    }
    receipt_root = args.output_root.resolve() / "receipts" / "encode_workers"
    receipt_root.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_root / f"worker_{args.shard_id:05d}.json"
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if existing != summary:
            raise FileExistsError(f"encode worker receipt drift: {receipt_path}")
    else:
        atomic_write_json(receipt_path, summary, exclusive=True)
    print(json.dumps({"pass": True, **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
