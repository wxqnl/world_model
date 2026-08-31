from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from io import BytesIO
import hashlib
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any

import numpy as np

from wm3d_v3.stage1.action_window_geometry import (
    ValidatedGeometryIndex,
    load_validated_geometry_index,
)
from wm3d_v3.stage1.robot_mask_cache import (
    ValidatedRobotMaskIndex,
    load_validated_robot_mask_index,
)


WindowIdentity = tuple[str, str, str, int]


class ActionEvidenceWindowSourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ActionEvidenceWindow:
    identity: WindowIdentity
    group_id: str
    legal_starts_sha256: str
    selection_sha256: str
    frame_indices: tuple[int, ...]
    source_rgb: Mapping[str, Any]
    source_action: Mapping[str, Any]
    geometry_output: Mapping[str, Any]
    geometry_path: str
    mask_output: Mapping[str, Any]
    mask_path: str
    mask_informative: bool


@dataclass(frozen=True)
class WindowArrays:
    rgb: np.ndarray
    depth: np.ndarray
    depth_conf: np.ndarray
    robot_masks: np.ndarray
    actions: np.ndarray


@dataclass(frozen=True)
class ValidatedActionEvidenceWindows(Mapping[WindowIdentity, ActionEvidenceWindow]):
    geometry_index: ValidatedGeometryIndex
    robot_mask_index: ValidatedRobotMaskIndex
    entries: Mapping[WindowIdentity, ActionEvidenceWindow]

    def __getitem__(self, key: WindowIdentity) -> ActionEvidenceWindow:
        return self.entries[key]

    def __iter__(self) -> Iterator[WindowIdentity]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def load_arrays(self, identity: WindowIdentity) -> WindowArrays:
        entry = self.entries[identity]
        geometry = _load_bound_npz(
            Path(entry.geometry_path), str(entry.geometry_output["sha256"])
        )
        mask = (
            _load_bound_npz(Path(entry.mask_path), str(entry.mask_output["sha256"]))
            if entry.mask_informative else None
        )
        rgb_all = _load_bound_npy(
            Path(str(entry.source_rgb["path"])), str(entry.source_rgb["sha256"])
        )
        actions = _load_bound_npy(
            Path(str(entry.source_action["path"])), str(entry.source_action["sha256"])
        )
        start = int(identity[3])
        rgb = np.array(rgb_all[start : start + 17], copy=True, order="C")
        if rgb.shape != (17, *rgb_all.shape[1:]):
            raise ActionEvidenceWindowSourceError(f"short RGB window: {identity}")
        if _array_sha256(rgb) != entry.source_rgb["window_sha256"]:
            raise ActionEvidenceWindowSourceError(f"RGB window hash drift: {identity}")
        expected_action_shape = tuple(int(value) for value in entry.source_action["shape"])
        if actions.shape != expected_action_shape or str(actions.dtype) != entry.source_action["dtype"]:
            raise ActionEvidenceWindowSourceError(f"action shape/dtype drift: {identity}")
        depth = np.asarray(geometry.get("depth"))
        depth_conf = np.asarray(geometry.get("depth_conf"))
        robot_masks = (
            np.asarray(mask.get("mask")) if mask is not None
            else np.zeros(rgb.shape[:3],dtype=np.uint8)
        )
        if depth.dtype != np.float16 or depth_conf.dtype != np.float16:
            raise ActionEvidenceWindowSourceError(f"geometry dtype drift: {identity}")
        if depth.shape != depth_conf.shape or depth.ndim != 3 or depth.shape[0] != 17:
            raise ActionEvidenceWindowSourceError(f"geometry shape drift: {identity}")
        if robot_masks.dtype != np.uint8 or robot_masks.shape[0] != 17:
            raise ActionEvidenceWindowSourceError(f"mask shape/dtype drift: {identity}")
        if not np.isin(robot_masks, (0, 1)).all():
            raise ActionEvidenceWindowSourceError(f"mask binary drift: {identity}")
        return WindowArrays(
            rgb=rgb,
            depth=np.array(depth, copy=True),
            depth_conf=np.array(depth_conf, copy=True),
            robot_masks=robot_masks.astype(bool, copy=True),
            actions=np.asarray(actions, dtype=np.float64),
        )


def _read_bound_bytes(path: Path, expected_sha256: str) -> bytes:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as exc:
        raise ActionEvidenceWindowSourceError(f"missing bound artifact: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ActionEvidenceWindowSourceError(f"bound artifact is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 4 * 1024 * 1024)
            if not block:
                break
            chunks.append(block)
            digest.update(block)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns
        )
        after_identity = (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
        )
        if before_identity != after_identity:
            raise ActionEvidenceWindowSourceError(f"bound artifact changed while reading: {path}")
        if digest.hexdigest() != expected_sha256:
            raise ActionEvidenceWindowSourceError(f"bound artifact hash drift: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_bound_npz(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    try:
        with np.load(BytesIO(_read_bound_bytes(path, expected_sha256)), allow_pickle=False) as payload:
            return {key: np.asarray(payload[key]).copy() for key in payload.files}
    except ActionEvidenceWindowSourceError:
        raise
    except Exception as exc:
        raise ActionEvidenceWindowSourceError(f"invalid bound NPZ: {path}") from exc


def _load_bound_npy(path: Path, expected_sha256: str) -> np.ndarray:
    try:
        return np.asarray(
            np.load(BytesIO(_read_bound_bytes(path, expected_sha256)), allow_pickle=False)
        )
    except ActionEvidenceWindowSourceError:
        raise
    except Exception as exc:
        raise ActionEvidenceWindowSourceError(f"invalid bound NPY: {path}") from exc


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(f"{array.dtype}|{array.shape}|".encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def load_bound_npz(path: str | Path, expected_sha256: str) -> dict[str, np.ndarray]:
    return _load_bound_npz(Path(path), expected_sha256)


def load_validated_action_evidence_windows(
    *,
    geometry_index_path: str | Path,
    robot_mask_index_path: str | Path,
    split_artifact: str | Path,
    manifest_path: str | Path,
    cache_root: str | Path,
    droid_cache_index: str | Path,
) -> ValidatedActionEvidenceWindows:
    geometry = load_validated_geometry_index(
        geometry_index_path,
        split_temporal_candidate=split_artifact,
        manifest_path=manifest_path,
        cache_root=cache_root,
        droid_cache_index=droid_cache_index,
    )
    masks = load_validated_robot_mask_index(
        robot_mask_index_path,
        split_path=split_artifact,
        manifest_path=manifest_path,
        cache_root=cache_root,
        droid_cache_index=droid_cache_index,
    )
    if set(geometry) != set(masks) or len(geometry) != 576:
        raise ActionEvidenceWindowSourceError("geometry/mask identity sets differ")
    joined: dict[WindowIdentity, ActionEvidenceWindow] = {}
    for identity in sorted(geometry):
        geometry_entry = geometry[identity]
        mask_entry = masks[identity]
        comparisons = (
            ("group_id", geometry_entry["group_id"], mask_entry["group_id"]),
            ("legal_starts_sha256", geometry_entry["legal_starts_sha256"], mask_entry["legal_starts_sha256"]),
            ("selection_sha256", geometry_entry["selection_sha256"], mask_entry["selection_sha256"]),
            ("frame_indices", geometry_entry["frame_indices"], mask_entry["frame_indices"]),
            ("source_rgb", geometry_entry["source_rgb"], mask_entry["source_rgb"]),
            ("source_action", geometry_entry["source_action"], mask_entry["source_action"]),
        )
        for label, geometry_value, mask_value in comparisons:
            if geometry_value != mask_value:
                raise ActionEvidenceWindowSourceError(
                    f"geometry/mask {label} mismatch: {identity}"
                )
        geometry_path = Path(geometry.index_path).parent / str(geometry_entry["output"]["path"])
        mask_informative="output" in mask_entry
        mask_output=(mask_entry["output"] if mask_informative else mask_entry["non_informative"])
        mask_path=(Path(masks.index_path).parent / str(mask_output["path"])) if mask_informative else None
        joined[identity] = ActionEvidenceWindow(
            identity=identity,
            group_id=str(geometry_entry["group_id"]),
            legal_starts_sha256=str(geometry_entry["legal_starts_sha256"]),
            selection_sha256=str(geometry_entry["selection_sha256"]),
            frame_indices=tuple(int(value) for value in geometry_entry["frame_indices"]),
            source_rgb=geometry_entry["source_rgb"],
            source_action=geometry_entry["source_action"],
            geometry_output=geometry_entry["output"],
            geometry_path=str(geometry_path),
            mask_output=mask_output,
            mask_path="" if mask_path is None else str(mask_path),
            mask_informative=mask_informative,
        )
    return ValidatedActionEvidenceWindows(
        geometry_index=geometry,
        robot_mask_index=masks,
        entries=MappingProxyType(joined),
    )
