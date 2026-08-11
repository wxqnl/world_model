"""Strict current-robot-state ABI for WM3D-V8 Stage0 action policy."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np


V8_PROPRIO_SCHEMA = "wm3d_v8_policy_proprio10_v1"
V8_PROPRIO_INDEX_SCHEMA = "wm3d_v8_policy_proprio10_index_v1"
V8_PROPRIO_STATS_SCHEMA = "wm3d_v8_policy_proprio10_stats_v1"
V8_PROPRIO_DIM = 10
V8_PROPRIO_LAYOUT = (
    "eef_x", "eef_y", "eef_z",
    "R00", "R10", "R20", "R01", "R11", "R21",
    "gripper_close01",
)
V8_PROPRIO_ANCHOR = "first_policy_action_target"
V8_PROPRIO_STD_FLOOR = 1.0e-6
PANDA_NOMINAL_CLOSED_WIDTH_M = 0.0
PANDA_NOMINAL_OPEN_WIDTH_M = 0.08
PANDA_OBSERVATION_MAX_WIDTH_M = float(np.float32(0.12))
BRIDGE_OBSERVATION_MIN_OPEN01 = float(np.float32(-0.05))
BRIDGE_OBSERVATION_MAX_OPEN01 = float(np.float32(1.12))
V8_EMBODIMENT_VOCAB = {
    "franka_droid": 0,
    "widowx_bridge": 1,
    "panda_robocasa_libero": 2,
}
_VOCAB_BYTES = json.dumps(
    V8_EMBODIMENT_VOCAB, sort_keys=True, separators=(",", ":")
).encode("utf-8")
V8_EMBODIMENT_VOCAB_SHA256 = hashlib.sha256(_VOCAB_BYTES).hexdigest()
LOWER_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class V8ProprioContractError(ValueError):
    """Raised when current state cannot satisfy the V8 physical ABI."""


def sha256_file(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def require_pinned_file(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> Path:
    source = Path(path)
    if source.is_symlink():
        raise V8ProprioContractError(f"{label} must not be a symlink: {source}")
    try:
        resolved = source.resolve(strict=True)
    except FileNotFoundError as exc:
        raise V8ProprioContractError(f"{label} is missing: {source}") from exc
    if not resolved.is_file():
        raise V8ProprioContractError(f"{label} is not a regular file: {resolved}")
    expected = str(expected_sha256 or "")
    if LOWER_HEX64.fullmatch(expected) is None:
        raise V8ProprioContractError(f"{label} expected SHA256 is invalid")
    observed = sha256_file(resolved)
    if observed != expected:
        raise V8ProprioContractError(
            f"{label} SHA256 mismatch: observed={observed} expected={expected}"
        )
    return resolved


def _finite_vector(value: np.ndarray, width: int, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise V8ProprioContractError(
            f"{label} must be finite [{width}], got {array.shape}"
        )
    return array


def rotation_matrix_to_6d(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise V8ProprioContractError("rotation matrix must be finite [3,3]")
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=2.0e-4, rtol=0.0
    ):
        raise V8ProprioContractError("rotation matrix is not orthonormal")
    if not np.isclose(
        float(np.linalg.det(rotation)), 1.0, atol=2.0e-4, rtol=0.0
    ):
        raise V8ProprioContractError("rotation matrix determinant is not +1")
    return np.asarray(
        (
            rotation[0, 0], rotation[1, 0], rotation[2, 0],
            rotation[0, 1], rotation[1, 1], rotation[2, 1],
        ),
        dtype=np.float32,
    )


def fixed_xyz_rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = _finite_vector(rpy, 3, label="fixed XYZ Euler")
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=np.float64)
    ry = np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=np.float64)
    rz = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=np.float64)
    return rz @ ry @ rx


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = _finite_vector(quaternion, 4, label="xyzw quaternion")
    norm = float(np.sqrt(x * x + y * y + z * z + w * w))
    if norm < 1.0e-8:
        raise V8ProprioContractError("quaternion norm is zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _require_close01(value: float, *, label: str) -> np.float32:
    close = float(value)
    if not np.isfinite(close) or close < -1.0e-4 or close > 1.0001:
        raise V8ProprioContractError(f"{label} must be close01, got {close}")
    return np.float32(np.clip(close, 0.0, 1.0))


def encode_proprio10(
    position: np.ndarray,
    rotation: np.ndarray,
    gripper_close01: float,
) -> np.ndarray:
    xyz = _finite_vector(position, 3, label="EEF position").astype(np.float32)
    rotation6d = rotation_matrix_to_6d(rotation)
    close = _require_close01(gripper_close01, label="gripper")
    value = np.concatenate((xyz, rotation6d, np.asarray([close], np.float32)))
    if value.shape != (V8_PROPRIO_DIM,) or not np.isfinite(value).all():
        raise V8ProprioContractError("encoded proprio violates [10] ABI")
    return value.astype(np.float32, copy=False)


def encode_rpy_proprio10(
    position: np.ndarray,
    fixed_xyz_rpy: np.ndarray,
    gripper_close01: float,
) -> np.ndarray:
    return encode_proprio10(
        position, fixed_xyz_rpy_to_matrix(fixed_xyz_rpy), gripper_close01
    )


def panda_finger_qpos_to_close01(
    finger_qpos: np.ndarray,
    *,
    closed_width: float = PANDA_NOMINAL_CLOSED_WIDTH_M,
    open_width: float = PANDA_NOMINAL_OPEN_WIDTH_M,
    max_observed_width: float = PANDA_OBSERVATION_MAX_WIDTH_M,
) -> np.float32:
    fingers = _finite_vector(finger_qpos, 2, label="Panda finger qpos")
    bounds = np.asarray(
        [closed_width, open_width, max_observed_width], dtype=np.float64
    )
    if (
        not np.isfinite(bounds).all()
        or closed_width < 0.0
        or not closed_width < open_width <= max_observed_width
    ):
        raise V8ProprioContractError("invalid sealed Panda aperture bounds")
    # RoboCasa/LIBERO store signed opposing finger coordinates. Their physical
    # aperture is the joint separation, not the sum of coordinate magnitudes;
    # the two expressions differ when acquisition noise puts both joints on the
    # same side of zero.
    width = abs(float(fingers[0] - fingers[1]))
    if width < closed_width or width > max_observed_width:
        raise V8ProprioContractError(
            f"Panda finger width {width} is outside sealed observation "
            f"envelope [{closed_width},{max_observed_width}] "
            f"with nominal open width {open_width}"
        )
    open01 = (width - closed_width) / (open_width - closed_width)
    return _require_close01(
        1.0 - float(np.clip(open01, 0.0, 1.0)),
        label="Panda gripper",
    )


def encode_robocasa_state16(state: np.ndarray) -> np.ndarray:
    value = _finite_vector(state, 16, label="RoboCasa observation.state")
    return encode_proprio10(
        value[7:10],
        quaternion_xyzw_to_matrix(value[10:14]),
        panda_finger_qpos_to_close01(value[14:16]),
    )


def encode_libero_proprio10(
    eef_position: np.ndarray,
    eef_quaternion_xyzw: np.ndarray,
    gripper_qpos: np.ndarray,
) -> np.ndarray:
    return encode_proprio10(
        eef_position,
        quaternion_xyzw_to_matrix(eef_quaternion_xyzw),
        panda_finger_qpos_to_close01(gripper_qpos),
    )


def encode_droid_state(
    pose_xyz_rpy: np.ndarray,
    gripper_close01: float,
) -> np.ndarray:
    pose = _finite_vector(pose_xyz_rpy, 6, label="DROID state pose")
    return encode_rpy_proprio10(pose[:3], pose[3:6], gripper_close01)


def encode_bridge_state(state_xyz_rpy_open01: np.ndarray) -> np.ndarray:
    state = _finite_vector(state_xyz_rpy_open01, 7, label="Bridge state")
    open01 = float(state[6])
    if (
        open01 < BRIDGE_OBSERVATION_MIN_OPEN01
        or open01 > BRIDGE_OBSERVATION_MAX_OPEN01
    ):
        raise V8ProprioContractError(f"Bridge open01 is invalid: {open01}")
    return encode_rpy_proprio10(
        state[:3], state[3:6], 1.0 - float(np.clip(open01, 0.0, 1.0))
    )


@dataclass(frozen=True)
class V8ProprioSample:
    raw: np.ndarray
    normalized: np.ndarray
    embodiment_id: int
    stats_key: str
    anchor_frame_index: int


class V8ProprioStore:
    """Content-addressed, exact-frame current-state reader.

    One instance corresponds to one canonical source and one split. It verifies
    index/stats at construction and each payload before first use. There is
    intentionally no interpolation, padding, truncation, or zero fallback.
    """

    def __init__(
        self,
        *,
        index_path: str | Path,
        index_sha256: str,
        stats_path: str | Path,
        stats_sha256: str,
        source: str,
        split: str | None,
        expected_identities: Iterable[str] | None = None,
        exact_coverage: bool = True,
        cache_capacity: int = 64,
    ) -> None:
        if split is not None and split not in {"train", "val", "test"}:
            raise V8ProprioContractError(f"unsupported split {split!r}")
        self.source = str(source)
        self.split = split
        self.index_path = require_pinned_file(
            index_path, index_sha256, label=f"{self.source} proprio index"
        )
        self.index_sha256 = str(index_sha256)
        self.stats_path = require_pinned_file(
            stats_path, stats_sha256, label=f"{self.source} proprio stats"
        )
        self.stats_sha256 = str(stats_sha256)
        self.records: dict[str, dict] = {}
        with self.index_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema") != V8_PROPRIO_INDEX_SCHEMA:
                    raise V8ProprioContractError(
                        f"proprio index line {line_number} has wrong schema"
                    )
                identity = str(row.get("identity") or "")
                if not identity or identity in self.records:
                    raise V8ProprioContractError(
                        f"blank/duplicate proprio identity {identity!r}"
                    )
                if str(row.get("source")) != self.source:
                    raise V8ProprioContractError(
                        f"proprio source mismatch for {identity}"
                    )
                if self.split is not None and str(row.get("split")) != self.split:
                    continue
                embodiment = str(row.get("embodiment") or "")
                embodiment_id = row.get("embodiment_id")
                if V8_EMBODIMENT_VOCAB.get(embodiment) != embodiment_id:
                    raise V8ProprioContractError(
                        f"proprio embodiment mismatch for {identity}"
                    )
                payload_path = Path(str(row.get("path") or ""))
                if payload_path.is_symlink() or not payload_path.is_file():
                    raise V8ProprioContractError(
                        f"proprio payload is missing/not regular: {payload_path}"
                    )
                if LOWER_HEX64.fullmatch(str(row.get("sha256") or "")) is None:
                    raise V8ProprioContractError(
                        f"proprio payload digest is invalid: {identity}"
                    )
                source_sha = str(row.get("source_state_sha256") or "")
                if LOWER_HEX64.fullmatch(source_sha) is None:
                    raise V8ProprioContractError(
                        f"upstream state digest is invalid: {identity}"
                    )
                self.records[identity] = row
        expected = set(str(value) for value in expected_identities or ())
        if expected:
            missing = sorted(expected - self.records.keys())
            extra = sorted(self.records.keys() - expected)
            if missing or (exact_coverage and extra):
                raise V8ProprioContractError(
                    f"proprio coverage mismatch missing={missing[:8]} "
                    f"extra={extra[:8]}"
                )
        with np.load(self.stats_path, allow_pickle=False) as stats:
            if str(np.asarray(stats["schema"]).item()) != V8_PROPRIO_STATS_SCHEMA:
                raise V8ProprioContractError("proprio stats schema mismatch")
            if str(np.asarray(stats["split"]).item()) != "train":
                raise V8ProprioContractError("proprio stats must be train-only")
            if str(np.asarray(stats["source"]).item()) != self.source:
                raise V8ProprioContractError("proprio stats source mismatch")
            if str(np.asarray(stats["index_sha256"]).item()) != self.index_sha256:
                raise V8ProprioContractError("proprio stats/index binding mismatch")
            if (
                str(np.asarray(stats["embodiment_vocab_sha256"]).item())
                != V8_EMBODIMENT_VOCAB_SHA256
            ):
                raise V8ProprioContractError(
                    "proprio embodiment vocabulary mismatch"
                )
            layout = tuple(
                str(value) for value in np.asarray(stats["layout"]).tolist()
            )
            if layout != V8_PROPRIO_LAYOUT:
                raise V8ProprioContractError("proprio stats layout mismatch")
            self.mean = np.asarray(stats["mean"], dtype=np.float32)
            self.std = np.asarray(stats["std"], dtype=np.float32)
            sample_count = int(np.asarray(stats["sample_count"]).item())
        if (
            self.mean.shape != (V8_PROPRIO_DIM,)
            or self.std.shape != (V8_PROPRIO_DIM,)
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.std).all()
            or np.any(self.std < V8_PROPRIO_STD_FLOOR)
            or sample_count <= 0
        ):
            raise V8ProprioContractError("invalid proprio normalization stats")
        self.stats_key = f"{self.source}:{self.stats_sha256}"
        self._cache_capacity = max(1, int(cache_capacity))
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, int]] = (
            OrderedDict()
        )

    def _load(self, identity: str) -> tuple[np.ndarray, np.ndarray, int]:
        cached = self._cache.get(identity)
        if cached is not None:
            self._cache.move_to_end(identity)
            return cached
        try:
            row = self.records[identity]
        except KeyError as exc:
            raise V8ProprioContractError(
                f"proprio identity is not sealed: {identity}"
            ) from exc
        path = Path(row["path"])
        observed_sha = sha256_file(path)
        if observed_sha != row["sha256"]:
            raise V8ProprioContractError(
                f"proprio payload SHA mismatch for {identity}: {observed_sha}"
            )
        with np.load(path, allow_pickle=False) as payload:
            scalar_expected = {
                "schema": V8_PROPRIO_SCHEMA,
                "identity": identity,
                "split": str(row["split"]),
                "source": self.source,
                "embodiment": str(row["embodiment"]),
                "embodiment_id": int(row["embodiment_id"]),
                "source_state_sha256": str(row["source_state_sha256"]),
            }
            for key, expected in scalar_expected.items():
                actual = np.asarray(payload[key]).item()
                if actual != expected:
                    raise V8ProprioContractError(
                        f"proprio payload {key} mismatch for {identity}"
                    )
            frame_indices = np.asarray(payload["frame_indices"], dtype=np.int64)
            source_frame_indices = np.asarray(
                payload["source_frame_indices"], dtype=np.int64
            )
            raw = np.asarray(payload["proprio_raw"], dtype=np.float32)
        if (
            frame_indices.ndim != 1
            or source_frame_indices.shape != frame_indices.shape
            or raw.shape != (len(frame_indices), V8_PROPRIO_DIM)
            or len(frame_indices) != int(row["frame_count"])
            or not np.array_equal(
                frame_indices, np.arange(len(frame_indices), dtype=np.int64)
            )
            or (len(source_frame_indices) > 1 and np.any(np.diff(source_frame_indices) <= 0))
            or not np.isfinite(raw).all()
        ):
            raise V8ProprioContractError(
                f"proprio payload arrays violate ABI for {identity}"
            )
        result = (frame_indices, raw, int(row["embodiment_id"]))
        self._cache[identity] = result
        self._cache.move_to_end(identity)
        while len(self._cache) > self._cache_capacity:
            self._cache.popitem(last=False)
        return result

    def current(self, identity: str, frame_index: int) -> V8ProprioSample:
        frame_indices, raw, embodiment_id = self._load(str(identity))
        location = int(np.searchsorted(frame_indices, int(frame_index)))
        if (
            location >= len(frame_indices)
            or int(frame_indices[location]) != int(frame_index)
        ):
            raise V8ProprioContractError(
                f"no exact proprio frame {frame_index} for {identity}"
            )
        value = np.array(raw[location], dtype=np.float32, copy=True)
        normalized = ((value - self.mean) / self.std).astype(np.float32)
        if not np.isfinite(normalized).all():
            raise V8ProprioContractError(
                f"normalized proprio is non-finite for {identity}"
            )
        return V8ProprioSample(
            raw=value,
            normalized=normalized,
            embodiment_id=embodiment_id,
            stats_key=self.stats_key,
            anchor_frame_index=int(frame_index),
        )
