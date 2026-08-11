#!/usr/bin/env python3
"""Build immutable WM3D-V8 policy proprio sidecars from real source state."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import sys
import tarfile
import tempfile
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from wm3d_v3.data.manifest import OXEClipRecord, read_manifest
from wm3d_v3.data.v7_contracts import V7ClipRecord, read_v7_manifest
from wm3d_v3.data.v8_proprio_contract import (
    V8_EMBODIMENT_VOCAB,
    V8_EMBODIMENT_VOCAB_SHA256,
    V8_PROPRIO_INDEX_SCHEMA,
    V8_PROPRIO_LAYOUT,
    V8_PROPRIO_SCHEMA,
    V8_PROPRIO_STATS_SCHEMA,
    V8_PROPRIO_STD_FLOOR,
    encode_bridge_state,
    encode_droid_state,
    encode_robocasa_state16,
    sha256_file,
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_number}: row is not an object")
            rows.append(row)
    return rows


def _episode_index(record: V7ClipRecord) -> int:
    try:
        return int(record.native_episode_id.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(
            f"cannot parse RoboCasa episode index: {record.native_episode_id}"
        ) from exc


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _publish_bytes_no_clobber(path: Path, payload: bytes) -> str:
    expected = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"output is not a regular file: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"no-clobber conflict for {path}: observed={observed} expected={expected}"
            )
        return observed
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"output is not a regular file: {path}")
            observed = sha256_file(path)
            if observed != expected:
                raise RuntimeError(
                    f"no-clobber conflict for {path}: "
                    f"observed={observed} expected={expected}"
                )
            return observed
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return expected


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = np.zeros(10, dtype=np.float64)
        self.total_sq = np.zeros(10, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 10 or not np.isfinite(array).all():
            raise RuntimeError(f"invalid proprio values for stats: {array.shape}")
        self.count += len(array)
        self.total += array.sum(axis=0)
        self.total_sq += np.square(array).sum(axis=0)

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count <= 0:
            raise RuntimeError("train-only proprio stats have no samples")
        mean = self.total / float(self.count)
        variance = np.maximum(
            self.total_sq / float(self.count) - np.square(mean), 0.0
        )
        std = np.sqrt(variance)
        std = np.where(std < V8_PROPRIO_STD_FLOOR, 1.0, std)
        return mean.astype(np.float32), std.astype(np.float32)


def _payload(
    *,
    identity: str,
    split: str,
    source: str,
    embodiment: str,
    source_state_sha256: str,
    frame_indices: np.ndarray,
    source_frame_indices: np.ndarray,
    raw: np.ndarray,
) -> dict[str, np.ndarray]:
    if V8_EMBODIMENT_VOCAB.get(embodiment) is None:
        raise RuntimeError(f"unknown embodiment {embodiment}")
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    source_frame_indices = np.asarray(source_frame_indices, dtype=np.int64)
    raw = np.asarray(raw, dtype=np.float32)
    if (
        frame_indices.shape != source_frame_indices.shape
        or raw.shape != (len(frame_indices), 10)
        or len(frame_indices) <= 0
        or not np.array_equal(frame_indices, np.arange(len(frame_indices)))
        or (
            len(source_frame_indices) > 1
            and np.any(np.diff(source_frame_indices) <= 0)
        )
        or not np.isfinite(raw).all()
    ):
        raise RuntimeError(f"invalid proprio payload arrays for {identity}")
    return {
        "schema": np.asarray(V8_PROPRIO_SCHEMA),
        "identity": np.asarray(identity),
        "split": np.asarray(split),
        "source": np.asarray(source),
        "embodiment": np.asarray(embodiment),
        "embodiment_id": np.asarray(
            V8_EMBODIMENT_VOCAB[embodiment], dtype=np.int64
        ),
        "source_state_sha256": np.asarray(source_state_sha256),
        "frame_indices": frame_indices,
        "source_frame_indices": source_frame_indices,
        "proprio_raw": raw,
    }


def _row(
    *,
    identity: str,
    split: str,
    source: str,
    embodiment: str,
    source_state_sha256: str,
    frame_count: int,
    path: Path,
    artifact_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": V8_PROPRIO_INDEX_SCHEMA,
        "identity": identity,
        "split": split,
        "source": source,
        "embodiment": embodiment,
        "embodiment_id": V8_EMBODIMENT_VOCAB[embodiment],
        "source_state_sha256": source_state_sha256,
        "frame_count": int(frame_count),
        "path": str(path.resolve()),
        "sha256": artifact_sha256,
    }


def _robocasa_model_source_frames(
    archive: Any,
    *,
    identity: str,
    world_count: int,
) -> np.ndarray:
    """Resolve every 5 Hz model index to its exact sealed 20 Hz source frame."""
    required = {
        "model_timestamps",
        "native_frame_indices",
        "native_fps",
        "source_control_hz",
        "model_control_hz",
    }
    missing = sorted(required - set(archive.files))
    if missing:
        raise RuntimeError(
            f"RoboCasa archive omits sealed model/source timing for {identity}: {missing}"
        )
    timestamps = np.asarray(archive["model_timestamps"], dtype=np.float64)
    native_frames = np.asarray(archive["native_frame_indices"], dtype=np.int64)
    native_hz = float(np.asarray(archive["native_fps"]).item())
    source_hz = float(np.asarray(archive["source_control_hz"]).item())
    model_hz = float(np.asarray(archive["model_control_hz"]).item())
    if (
        timestamps.shape != (world_count,)
        or native_frames.ndim != 1
        or len(native_frames) <= 0
        or not np.isfinite(timestamps).all()
        or not np.isfinite([native_hz, source_hz, model_hz]).all()
        or abs(native_hz - source_hz) > 1e-6
        or abs(model_hz - 5.0) > 1e-6
        or source_hz <= 0.0
    ):
        raise RuntimeError(f"invalid RoboCasa sealed timing for {identity}")
    scaled = timestamps * source_hz
    source_frames = np.rint(scaled).astype(np.int64)
    if (
        not np.allclose(scaled, source_frames, atol=1e-5, rtol=0.0)
        or np.any(np.diff(source_frames) <= 0)
        or not np.isin(source_frames, native_frames).all()
    ):
        raise RuntimeError(f"non-exact RoboCasa 5Hz-to-source mapping for {identity}")
    return source_frames


def _robocasa_records(
    compact_index: Path,
    manifests: Iterable[Path],
) -> Iterable[tuple[str, str, str, np.ndarray, np.ndarray, np.ndarray]]:
    source_records: dict[str, V7ClipRecord] = {}
    for manifest in manifests:
        for record in read_v7_manifest(manifest):
            if record.clip_hash in source_records:
                raise RuntimeError(f"duplicate RoboCasa clip_hash {record.clip_hash}")
            source_records[record.clip_hash] = record
    grouped: dict[Path, list[tuple[str, dict[str, Any], V7ClipRecord]]] = {}
    seen: set[str] = set()
    for row in _jsonl(compact_index):
        identity = str(row.get("clip_hash") or "")
        if not identity or identity in seen:
            raise RuntimeError(f"blank/duplicate compact identity {identity!r}")
        seen.add(identity)
        record = source_records.get(identity)
        if record is None:
            raise RuntimeError(f"compact clip is absent from sealed V7 manifests: {identity}")
        if str(row.get("split")) != record.split:
            raise RuntimeError(f"RoboCasa split mismatch for {identity}")
        parquet_path = Path(record.raw_path)
        grouped.setdefault(parquet_path, []).append((identity, row, record))

    for parquet_path in sorted(grouped, key=str):
        table = pq.read_table(
            parquet_path,
            columns=["observation.state", "episode_index", "frame_index"],
        )
        arrays = {
            "state": np.asarray(
                table["observation.state"].to_pylist(), dtype=np.float32
            ),
            "episode": np.asarray(
                table["episode_index"].to_numpy(), dtype=np.int64
            ),
            "frame": np.asarray(table["frame_index"].to_numpy(), dtype=np.int64),
        }
        source_sha = sha256_file(parquet_path)
        for identity, row, record in sorted(
            grouped[parquet_path], key=lambda item: item[0]
        ):
            archive_path = Path(str(row.get("path") or ""))
            expected_archive_sha = str(
                row.get("artifact_sha256") or row.get("sha256") or ""
            )
            if sha256_file(archive_path) != expected_archive_sha:
                raise RuntimeError(f"compact archive SHA mismatch for {identity}")
            with np.load(archive_path, allow_pickle=False) as archive:
                if str(np.asarray(archive["clip_hash"]).item()) != identity:
                    raise RuntimeError(
                        f"compact archive identity mismatch for {identity}"
                    )
                world_count = len(np.asarray(archive["actions"]))
                model_native = _robocasa_model_source_frames(
                    archive,
                    identity=identity,
                    world_count=world_count,
                )
            episode = _episode_index(record)
            mask = arrays["episode"] == episode
            episode_frames = arrays["frame"][mask]
            episode_states = arrays["state"][mask]
            lookup = {int(frame): index for index, frame in enumerate(episode_frames)}
            try:
                selected = np.stack(
                    [episode_states[lookup[int(frame)]] for frame in model_native]
                )
            except KeyError as exc:
                raise RuntimeError(
                    f"RoboCasa state frame is absent for {identity}: {exc}"
                ) from exc
            raw = np.stack([encode_robocasa_state16(value) for value in selected])
            yield (
                identity,
                record.split,
                source_sha,
                np.arange(world_count, dtype=np.int64),
                model_native,
                raw,
            )


def _canonical_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _jsonl(path):
        identity = str(row.get("clip_id") or "")
        if not identity or identity in rows:
            raise RuntimeError(f"blank/duplicate canonical identity {identity!r}")
        rows[identity] = row
    return rows


def _source_manifest_rows(path: Path) -> dict[str, OXEClipRecord]:
    rows: dict[str, OXEClipRecord] = {}
    for record in read_manifest(path):
        identity = str(record.clip_id or "")
        if not identity or identity in rows:
            raise RuntimeError(
                f"blank/duplicate source manifest identity {identity!r}"
            )
        rows[identity] = record
    return rows


def _droid_records(
    source_manifest: Path,
    canonical_index: Path,
) -> Iterable[tuple[str, str, str, np.ndarray, np.ndarray, np.ndarray]]:
    manifest = _source_manifest_rows(source_manifest)
    canonical = _canonical_rows(canonical_index)
    if set(manifest) != set(canonical):
        missing = sorted(set(manifest) ^ set(canonical))
        raise RuntimeError(f"DROID manifest/index coverage mismatch: {missing[:8]}")
    for identity in sorted(canonical):
        record = manifest[identity]
        row = canonical[identity]
        state_path = Path(str(row.get("state_pose_path") or ""))
        expected_sha = str(row.get("state_pose_sha256") or "")
        if sha256_file(state_path) != expected_sha:
            raise RuntimeError(f"DROID state SHA mismatch for {identity}")
        with np.load(state_path, allow_pickle=False) as state:
            pose = np.asarray(state["pose"], dtype=np.float32)
            grip = np.asarray(state["grip"], dtype=np.float32)
        if pose.shape != (record.n_frames, 6) or grip.shape != (record.n_frames,):
            raise RuntimeError(f"DROID state cardinality mismatch for {identity}")
        raw = np.stack(
            [encode_droid_state(pose[index], grip[index]) for index in range(len(pose))]
        )
        frames = np.arange(record.n_frames, dtype=np.int64)
        yield identity, str(row["split"]), expected_sha, frames, frames.copy(), raw


def _extract_bridge_robot_state(
    episode: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    raw_steps = episode.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise RuntimeError("raw Bridge episode has no steps")
    states: list[np.ndarray] = []
    for index, step in enumerate(raw_steps):
        if not isinstance(step, dict):
            raise RuntimeError(f"Bridge step {index} is not an object")
        observation = step.get("observation")
        if not isinstance(observation, dict) or "state" not in observation:
            raise RuntimeError(f"Bridge step {index} omits observation.state")
        state = np.asarray(observation["state"], dtype=np.float64).reshape(-1)
        if state.size < 7 or not np.isfinite(state).all():
            raise RuntimeError(f"Bridge step {index} has invalid observation.state")
        states.append(state)
    stacked = np.stack(states)
    return stacked[:, :6], stacked[:, 6]


def _bridge_records(
    source_manifest: Path,
    canonical_index: Path,
) -> Iterable[tuple[str, str, str, np.ndarray, np.ndarray, np.ndarray]]:
    manifest = _source_manifest_rows(source_manifest)
    canonical = _canonical_rows(canonical_index)
    if set(manifest) != set(canonical):
        missing = sorted(set(manifest) ^ set(canonical))
        raise RuntimeError(f"Bridge manifest/index coverage mismatch: {missing[:8]}")
    for identity in sorted(canonical):
        record: OXEClipRecord = manifest[identity]
        row = canonical[identity]
        tar_path = Path(record.tar_path)
        with tarfile.open(tar_path, "r") as archive:
            member = archive.extractfile(record.pickle_member)
            if member is None:
                raise RuntimeError(f"missing Bridge member for {identity}")
            payload = member.read()
        source_sha = hashlib.sha256(payload).hexdigest()
        episode = pickle.loads(payload)
        pose, grip = _extract_bridge_robot_state(episode)
        if pose is None or grip is None:
            raise RuntimeError(f"Bridge state is missing for {identity}")
        pose = np.asarray(pose, dtype=np.float32)
        grip = np.asarray(grip, dtype=np.float32)
        if (
            pose.ndim != 2
            or pose.shape[0] != record.n_frames
            or pose.shape[1] < 6
            or grip.shape != (record.n_frames,)
        ):
            raise RuntimeError(f"Bridge state cardinality mismatch for {identity}")
        raw = np.stack(
            [
                encode_bridge_state(
                    np.concatenate((pose[index, :6], grip[index : index + 1]))
                )
                for index in range(record.n_frames)
            ]
        )
        frames = np.arange(record.n_frames, dtype=np.int64)
        yield identity, str(row["split"]), source_sha, frames, frames.copy(), raw


def _limit(records: Iterable[tuple], max_records: int) -> Iterable[tuple]:
    for index, record in enumerate(records):
        if max_records > 0 and index >= max_records:
            break
        yield record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-type", choices=("robocasa", "droid", "bridge"), required=True)
    parser.add_argument("--compact-index", type=Path)
    parser.add_argument("--robocasa-manifest", action="append", type=Path, default=[])
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--canonical-index", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-index", required=True, type=Path)
    parser.add_argument("--output-stats", required=True, type=Path)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = args.source_type
    if source == "robocasa":
        if args.compact_index is None or not args.robocasa_manifest:
            parser.error("robocasa needs --compact-index and --robocasa-manifest")
        records = _robocasa_records(args.compact_index, args.robocasa_manifest)
        embodiment = "panda_robocasa_libero"
    else:
        if args.source_manifest is None or args.canonical_index is None:
            parser.error(f"{source} needs --source-manifest and --canonical-index")
        records = (
            _droid_records(args.source_manifest, args.canonical_index)
            if source == "droid"
            else _bridge_records(args.source_manifest, args.canonical_index)
        )
        embodiment = "franka_droid" if source == "droid" else "widowx_bridge"

    stats = RunningStats()
    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    for identity, split, source_sha, frames, source_frames, raw in _limit(
        records, args.max_records
    ):
        if split not in {"train", "val", "test"}:
            raise RuntimeError(f"invalid split for {identity}: {split}")
        arrays = _payload(
            identity=identity,
            split=split,
            source=source,
            embodiment=embodiment,
            source_state_sha256=source_sha,
            frame_indices=frames,
            source_frame_indices=source_frames,
            raw=raw,
        )
        output = (
            args.output_root
            / split
            / f"{identity.replace('/', '__')}.npz"
        ).resolve()
        payload = _npz_bytes(arrays)
        digest = hashlib.sha256(payload).hexdigest()
        if not args.dry_run:
            digest = _publish_bytes_no_clobber(output, payload)
        rows.append(
            _row(
                identity=identity,
                split=split,
                source=source,
                embodiment=embodiment,
                source_state_sha256=source_sha,
                frame_count=len(raw),
                path=output,
                artifact_sha256=digest,
            )
        )
        split_counts[split] = split_counts.get(split, 0) + 1
        if split == "train":
            stats.update(raw)
    if not rows:
        raise RuntimeError("no proprio records were built")
    rows.sort(key=lambda row: (row["identity"], row["split"]))
    index_payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    index_sha = hashlib.sha256(index_payload).hexdigest()
    mean, std = stats.finish()
    stats_payload = _npz_bytes(
        {
            "schema": np.asarray(V8_PROPRIO_STATS_SCHEMA),
            "split": np.asarray("train"),
            "source": np.asarray(source),
            "index_sha256": np.asarray(index_sha),
            "embodiment_vocab_sha256": np.asarray(
                V8_EMBODIMENT_VOCAB_SHA256
            ),
            "layout": np.asarray(V8_PROPRIO_LAYOUT),
            "mean": mean,
            "std": std,
            "sample_count": np.asarray(stats.count, dtype=np.int64),
        }
    )
    stats_sha = hashlib.sha256(stats_payload).hexdigest()
    if not args.dry_run:
        _publish_bytes_no_clobber(args.output_index.resolve(), index_payload)
        _publish_bytes_no_clobber(args.output_stats.resolve(), stats_payload)
    print(
        json.dumps(
            {
                "source": source,
                "records": len(rows),
                "split_counts": split_counts,
                "train_state_count": stats.count,
                "index_sha256": index_sha,
                "stats_sha256": stats_sha,
                "dry_run": bool(args.dry_run),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
