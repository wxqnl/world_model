"""Transactional, model-size-independent episode cache writer for WM3D."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import uuid
from typing import Any, Mapping

from safetensors.torch import save_file
import torch

from .cache_codec import JpegPackWriter, quantize_per_vector
from .cache_tasks import AtomicTaskClaim, CacheTask
from .episode_robot import validate_episode_robot_tensors
from .manifest_contract import (
    CACHE_EPISODE_INDEX_SCHEMA,
    canonical_timestamp_sha256,
    canonical_sha256,
    sha256_file,
)


CACHE_TASK_PAYLOAD_SCHEMA = "wm3d_v8_episode_cache_payload_v3"


class CacheWriterError(RuntimeError):
    pass


@dataclass(frozen=True)
class UnifiedFrameCache:
    source_observation_rows: torch.Tensor
    frame_times_s: torch.Tensor
    view_tokens: torch.Tensor
    rgb: torch.Tensor
    view_mask: torch.Tensor
    world_token_mask: torch.Tensor
    depth: torch.Tensor
    depth_mask: torch.Tensor
    point: torch.Tensor
    point_mask: torch.Tensor
    camera_pose: torch.Tensor
    camera_pose_mask: torch.Tensor
    geometry_confidence: torch.Tensor
    appearance_tokens: torch.Tensor | None = None


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise CacheWriterError(f"refusing to overwrite non-identical {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _validate_frames(frames: UnifiedFrameCache) -> int:
    count = int(frames.frame_times_s.numel())
    if frames.frame_times_s.ndim != 1 or count < 2:
        raise CacheWriterError("frame clock must contain at least two samples")
    if not bool(torch.isfinite(frames.frame_times_s).all()) or not bool(
        torch.diff(frames.frame_times_s).gt(0).all()
    ):
        raise CacheWriterError("frame times must be finite and strictly increasing")
    if (
        tuple(frames.source_observation_rows.shape) != (count,)
        or frames.source_observation_rows.dtype != torch.int64
        or bool((frames.source_observation_rows < 0).any())
        or not bool(torch.diff(frames.source_observation_rows).gt(0).all())
    ):
        raise CacheWriterError(
            "source_observation_rows must be strictly increasing non-negative int64"
        )
    if frames.view_tokens.ndim != 4 or frames.view_tokens.shape[0] != count:
        raise CacheWriterError("view_tokens must be [N,V,P,D]")
    views, patches = frames.view_tokens.shape[1:3]
    expected = {
        "view_mask": (count, views),
        "world_token_mask": (count, patches),
        "depth": (count, views, patches),
        "depth_mask": (count, views, patches),
        "point": (count, views, patches, 3),
        "point_mask": (count, views, patches),
        "camera_pose": (count, views, 9),
        "camera_pose_mask": (count, views),
        "geometry_confidence": (count, views, patches),
    }
    for name, shape in expected.items():
        value = getattr(frames, name)
        if tuple(value.shape) != shape:
            raise CacheWriterError(f"{name} shape {tuple(value.shape)} != {shape}")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise CacheWriterError(f"{name} contains NaN/Inf")
    if (
        frames.rgb.ndim != 5
        or tuple(frames.rgb.shape[:3]) != (count, views, 3)
        or frames.rgb.dtype != torch.uint8
    ):
        raise CacheWriterError("RGB frames must be uint8 [N,V,3,H,W]")
    if not bool(frames.view_mask.any(dim=1).all()):
        raise CacheWriterError("every frame requires at least one real view")
    if frames.appearance_tokens is not None:
        appearance = frames.appearance_tokens
        if (
            appearance.ndim != 4
            or appearance.shape[0] != count
            or appearance.shape[1] != views
            or appearance.shape[-1] != frames.view_tokens.shape[-1]
        ):
            raise CacheWriterError(
                "appearance_tokens must be [N,V,P_appearance,D]"
            )
        if not appearance.is_floating_point() or not bool(torch.isfinite(appearance).all()):
            raise CacheWriterError("appearance_tokens must be finite floating values")
    if bool((frames.depth_mask & ~frames.view_mask[..., None]).any()) or bool(
        (frames.point_mask & ~frames.view_mask[..., None]).any()
    ):
        raise CacheWriterError("missing camera cannot carry geometry supervision")
    return count


def _json_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        decoded = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise CacheWriterError("source evidence is not finite canonical JSON") from exc
    if not isinstance(decoded, dict):
        raise CacheWriterError("source evidence must be a mapping")
    return decoded


def write_cache_task(
    *,
    task: CacheTask,
    cache_root: Path,
    frames: UnifiedFrameCache,
    robot_tensors: Mapping[str, torch.Tensor],
    source_evidence: Mapping[str, Any],
    jpeg_quality: int = 92,
) -> dict[str, Any]:
    """Publish one episode once; T/K windows are materialized later as indices."""

    frame_count = _validate_frames(frames)
    if frame_count > task.observation_samples or int(frames.source_observation_rows[-1]) >= task.observation_samples:
        raise CacheWriterError(
            f"episode cached frames/rows exceed task observations {task.observation_samples}"
        )
    validate_episode_robot_tensors(robot_tensors)
    observation_clock = robot_tensors["observation_times_s"].detach().cpu().numpy()
    if len(observation_clock) != task.observation_samples:
        raise CacheWriterError(
            "robot observation clock cardinality differs from the cache task"
        )
    task_clock = task.observation_clock
    observed_clock_sha = canonical_timestamp_sha256(observation_clock)
    if (
        int(task_clock.get("sample_count", -1)) != task.observation_samples
        or str(task_clock.get("timestamp_sha256", "")) != observed_clock_sha
    ):
        raise CacheWriterError(
            "robot observation clock is not the SHA-bound source-manifest clock"
        )
    selected_rows = frames.source_observation_rows.detach().cpu().numpy()
    selected_times = frames.frame_times_s.detach().cpu().numpy()
    if not bool(
        torch.equal(
            torch.from_numpy(observation_clock[selected_rows]).to(torch.float64),
            torch.from_numpy(selected_times).to(torch.float64),
        )
    ):
        raise CacheWriterError(
            "cached frame times are not exact selected source observation timestamps"
        )
    evidence_source = _json_evidence(source_evidence)
    root = Path(cache_root).absolute()
    root.mkdir(parents=True, exist_ok=True)
    claim_state = AtomicTaskClaim(root, task)
    if claim_state.completed():
        return {"task_id": task.task_id, "status": "already_complete"}
    with claim_state as claim:
        final = root / "payload" / "tasks" / task.task_id[:2] / task.task_id
        if final.exists():
            raise CacheWriterError(f"uncommitted task payload requires audit: {final}")
        temporary = final.parent / f".{task.task_id}.incomplete.{uuid.uuid4().hex}"
        temporary.mkdir(parents=True)
        jpeg = JpegPackWriter(temporary / "rgb.jpgpack", quality=jpeg_quality)
        offsets, lengths = [], []
        try:
            for frame in frames.rgb:
                one_offsets, one_lengths = jpeg.append(frame)
                offsets.append(one_offsets)
                lengths.append(one_lengths)
            jpeg.close()
        except Exception:
            jpeg.__exit__(None, None, None)
            raise

        view_q, view_scale = quantize_per_vector(frames.view_tokens)
        feature_tensors = {
            "view_tokens_q": view_q,
            "view_tokens_scale": view_scale,
            "source_observation_row": frames.source_observation_rows.cpu().contiguous(),
            "frame_time_s": frames.frame_times_s.cpu().contiguous(),
            "view_mask": frames.view_mask.cpu().contiguous(),
            "world_token_mask": frames.world_token_mask.cpu().contiguous(),
            "depth": frames.depth.cpu().contiguous(),
            "depth_mask": frames.depth_mask.cpu().contiguous(),
            "point": frames.point.cpu().contiguous(),
            "point_mask": frames.point_mask.cpu().contiguous(),
            "camera_pose": frames.camera_pose.cpu().contiguous(),
            "camera_pose_mask": frames.camera_pose_mask.cpu().contiguous(),
            "geometry_confidence": frames.geometry_confidence.cpu().contiguous(),
            "rgb_offsets": torch.stack(offsets),
            "rgb_lengths": torch.stack(lengths),
        }
        if frames.appearance_tokens is not None:
            appearance_q, appearance_scale = quantize_per_vector(
                frames.appearance_tokens
            )
            feature_tensors["appearance_tokens_q"] = appearance_q
            feature_tensors["appearance_tokens_scale"] = appearance_scale
        save_file(
            feature_tensors,
            temporary / "features.safetensors",
        )
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in robot_tensors.items()
            },
            temporary / "robot.safetensors",
        )
        for name in ("rgb.jpgpack", "features.safetensors", "robot.safetensors"):
            _fsync(temporary / name)
        file_evidence = {
            name: {
                "size": (temporary / name).stat().st_size,
                "sha256": sha256_file(temporary / name),
            }
            for name in ("features.safetensors", "rgb.jpgpack", "robot.safetensors")
        }
        manifest = {
            "schema": CACHE_TASK_PAYLOAD_SCHEMA,
            "task": task.as_dict(),
            "frame_count": frame_count,
            "files": file_evidence,
            "source_evidence": evidence_source,
        }
        _publish_bytes(
            temporary / "manifest.json",
            (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        )
        commit = {
            "schema": CACHE_TASK_PAYLOAD_SCHEMA,
            "task_id": task.task_id,
            "manifest_sha256": sha256_file(temporary / "manifest.json"),
            "manifest_content_sha256": canonical_sha256(manifest),
        }
        _publish_bytes(
            temporary / "COMMITTED.json",
            (json.dumps(commit, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        )
        _fsync(temporary)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.rename(temporary, final)
        _fsync(final.parent)

        relative = final.relative_to(root).as_posix()
        row = {
            "schema": CACHE_EPISODE_INDEX_SCHEMA,
            "episode_id": task.episode_id,
            "source": task.source,
            "split": task.split,
            "embodiment": task.embodiment,
            "feature_shard": f"{relative}/features.safetensors",
            "feature_sha256": file_evidence["features.safetensors"]["sha256"],
            "robot_shard": f"{relative}/robot.safetensors",
            "robot_sha256": file_evidence["robot.safetensors"]["sha256"],
            "rgb_pack": f"{relative}/rgb.jpgpack",
            "rgb_pack_sha256": file_evidence["rgb.jpgpack"]["sha256"],
            "frame_count": frame_count,
        }
        fragment = root / "episode_index_fragments" / f"{task.task_id}.jsonl"
        _publish_bytes(
            fragment,
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        )
        outputs = {
            final / name: sha256_file(final / name)
            for name in (
                "features.safetensors",
                "robot.safetensors",
                "rgb.jpgpack",
                "manifest.json",
                "COMMITTED.json",
            )
        }
        outputs[fragment] = sha256_file(fragment)
        claim.publish_receipt(outputs)
    return {
        "task_id": task.task_id,
        "status": "published",
        "frames": frame_count,
        "fragment": str(fragment),
    }
