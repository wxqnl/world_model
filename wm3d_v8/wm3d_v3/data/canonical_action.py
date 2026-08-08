"""Strict WM3D-v7 canonical action contract for OXE robot sources.

The physical action representation is always

``[dx, dy, dz, rx, ry, rz, gripper_close_signed]``

where translation is a base-frame delta in metres, rotation is the SO(3)
rotation vector of a base-frame delta in radians, and the gripper is ``-1``
for fully open and ``+1`` for fully closed (continuous intermediate DROID
commands are preserved).  Pose statistics are source-specific; the signed
gripper is never standardized.

This module intentionally supports only sources whose cached semantics have
been audited.  Callers must not silently route an unsupported source through
the action-conditioned path.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from wm3d_v3.stage1.droid_interval_action import DROID_INTERVAL_ACTION_KIND


CANONICAL_ACTION_SCHEMA = "wm3d_v7_base_delta_axisangle_gripclose_v1"
CANONICAL_ACTION_STATS_SCHEMA = "wm3d_v7_canonical_action_stats_v1"
CANONICAL_ACTION_CACHE_ROW_SCHEMA = "wm3d_v7_canonical_action_cache_row_v1"
CANONICAL_GRIPPER_SEMANTICS = "signed_close_positive_continuous"
# Provenance of the exact canonical payload builder that produced the
# promoted Bridge/DROID payloads.  This deliberately does not claim to be the
# current runtime loader file digest: loader-only validation changes must not
# invalidate immutable payloads whose numerical conversion did not change.
CANONICAL_ACTION_PAYLOAD_BUILDER_PROVENANCE_SHA256 = (
    "4a43463e93cfa03beb444031b28cd0b6c54dbeb4f4602371ab1b1cd8710b9170"
)
BRIDGE_ACTION_KIND = "delta_xyz+rpy+gripper"
LEGACY_DROID_ACTION_KIND = "droid_action_7d"
SUPPORTED_CANONICAL_ACTION_SOURCES = frozenset(("bridge", "droid"))
CANONICAL_SOURCE_NAMES = {
    "bridge": "oxe_bridge_action",
    "droid": "oxe_droid_action",
}
CANONICAL_ACTION_FRAME_OFFSETS = {
    "bridge": -2,
    "droid": -1,
}


class CanonicalActionContractError(ValueError):
    """Raised when cached action data cannot satisfy the v7 contract."""


_HEX_DIGITS = frozenset("0123456789abcdef")


def _require_sha256(value: object, *, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(character not in _HEX_DIGITS for character in digest):
        raise CanonicalActionContractError(f"{label} is not a pinned SHA256")
    return digest


def canonical_action_implementation_sha256() -> str:
    """Promoted payload-builder provenance digest recorded in cache rows."""

    return CANONICAL_ACTION_PAYLOAD_BUILDER_PROVENANCE_SHA256


def canonical_action_runtime_file_sha256() -> str:
    """Digest of this current runtime loader file for deployment identity."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _read_pinned_bytes(path: Path, expected_sha256: str, *, label: str) -> bytes:
    """Read one regular non-symlink file and verify its content digest."""

    path = Path(path)
    if ".." in path.parts:
        raise CanonicalActionContractError(f"{label} contains path traversal: {path}")
    if path.is_symlink() or not path.is_file():
        raise CanonicalActionContractError(
            f"{label} is missing, is not regular, or is a symlink: {path}"
        )
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise CanonicalActionContractError(f"{label} changed while it was read")
    observed = hashlib.sha256(payload).hexdigest()
    if observed != _require_sha256(expected_sha256, label=f"{label} expected digest"):
        raise CanonicalActionContractError(
            f"{label} digest mismatch: observed={observed} expected={expected_sha256}"
        )
    return payload


def canonical_action_source(source: str) -> str:
    """Return the stable lower-case source key used by stats and metadata."""

    return str(source).strip().lower()


def validate_canonical_action_metadata(*, source: str, action_kind: str) -> str:
    """Fail fast when a manifest source/action_kind is not action-safe."""

    key = canonical_action_source(source)
    if key not in SUPPORTED_CANONICAL_ACTION_SOURCES:
        raise CanonicalActionContractError(
            f"source {key!r} has no passed WM3D-v7 canonical action adapter"
        )
    kind = str(action_kind).strip()
    if key == "bridge" and kind != BRIDGE_ACTION_KIND:
        raise CanonicalActionContractError(
            f"Bridge canonicalization requires action_kind {BRIDGE_ACTION_KIND!r}, "
            f"got {action_kind!r}"
        )
    if key == "droid":
        if kind == LEGACY_DROID_ACTION_KIND:
            raise CanonicalActionContractError(
                "legacy droid_action_7d is forbidden by the v7 canonical contract"
            )
        if kind != DROID_INTERVAL_ACTION_KIND:
            raise CanonicalActionContractError(
                f"DROID canonicalization requires action_kind "
                f"{DROID_INTERVAL_ACTION_KIND!r}, got {action_kind!r}"
            )
    return key


def _require_dense7(raw_action: np.ndarray, *, source: str) -> np.ndarray:
    raw = np.asarray(raw_action, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 7 or raw.shape[0] <= 0:
        raise CanonicalActionContractError(
            f"{source} action cache must be finite [N,7], got {raw.shape}"
        )
    if not np.isfinite(raw).all():
        raise CanonicalActionContractError(
            f"{source} action cache contains non-finite values"
        )
    return raw


def _require_close01(values: np.ndarray, *, source: str) -> np.ndarray:
    close01 = np.asarray(values, dtype=np.float32)
    if np.any(close01 < -1e-6) or np.any(close01 > 1.0 + 1e-6):
        raise CanonicalActionContractError(
            f"{source} gripper cache must use close01 in [0,1]"
        )
    return np.clip(close01, 0.0, 1.0)


def euler_xyz_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Convert roll/pitch/yaw to active rotation matrices.

    The convention is fixed/extrinsic XYZ (roll about world X, pitch about
    world Y, yaw about world Z), equivalently intrinsic ZYX, with active matrix
    ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.  Bridge's official
    ``transformation_utils.eulerAnglesToRotationMatrix`` uses this exact
    product and ``action2transform_local`` documents that the rotation axes are
    the same as world.  DROID Cartesian pose sidecars use the same RPY matrix
    convention in their upstream conversion.
    """

    values = np.asarray(rpy, dtype=np.float64)
    if values.shape[-1] != 3 or not np.isfinite(values).all():
        raise CanonicalActionContractError(
            f"Euler XYZ input must be finite [...,3], got {values.shape}"
        )
    roll, pitch, yaw = np.moveaxis(values, -1, 0)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    matrix = np.empty(values.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = cy * cp
    matrix[..., 0, 1] = cy * sp * sr - sy * cr
    matrix[..., 0, 2] = cy * sp * cr + sy * sr
    matrix[..., 1, 0] = sy * cp
    matrix[..., 1, 1] = sy * sp * sr + cy * cr
    matrix[..., 1, 2] = sy * sp * cr - cy * sr
    matrix[..., 2, 0] = -sp
    matrix[..., 2, 1] = cp * sr
    matrix[..., 2, 2] = cp * cr
    return matrix


def rotation_matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    """Return the principal SO(3) logarithm for one or more matrices."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.shape[-2:] != (3, 3) or not np.isfinite(values).all():
        raise CanonicalActionContractError(
            f"rotation matrix must be finite [...,3,3], got {values.shape}"
        )
    flat = values.reshape(-1, 3, 3)
    out = np.empty((flat.shape[0], 3), dtype=np.float64)
    for index, rotation in enumerate(flat):
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=2e-5
        ):
            raise CanonicalActionContractError("input is not a valid SO(3) matrix")
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ],
                dtype=np.float64,
            )
        elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            scale = np.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])) * 2.0
            quaternion = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif rotation[1, 1] > rotation[2, 2]:
            scale = np.sqrt(max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])) * 2.0
            quaternion = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = np.sqrt(max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])) * 2.0
            quaternion = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
        quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
        if quaternion[0] < 0.0:
            quaternion = -quaternion
        vector = quaternion[1:]
        vector_norm = float(np.linalg.norm(vector))
        if vector_norm < 1e-10:
            out[index] = 2.0 * vector
        else:
            angle = 2.0 * np.arctan2(vector_norm, np.clip(quaternion[0], -1.0, 1.0))
            out[index] = vector * (angle / vector_norm)
    return out.reshape(values.shape[:-2] + (3,)).astype(np.float32)


def rotvec_to_rotation_matrix(rotvec: np.ndarray) -> np.ndarray:
    """SO(3) exponential used by contract tests and downstream audits."""

    values = np.asarray(rotvec, dtype=np.float64)
    if values.shape[-1] != 3 or not np.isfinite(values).all():
        raise CanonicalActionContractError(
            f"rotation vector must be finite [...,3], got {values.shape}"
        )
    flat = values.reshape(-1, 3)
    output = np.empty((flat.shape[0], 3, 3), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for index, vector in enumerate(flat):
        angle = float(np.linalg.norm(vector))
        x, y, z = vector
        skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        if angle < 1e-8:
            a = 1.0 - angle * angle / 6.0
            b = 0.5 - angle * angle / 24.0
        else:
            a = np.sin(angle) / angle
            b = (1.0 - np.cos(angle)) / (angle * angle)
        output[index] = identity + a * skew + b * (skew @ skew)
    return output.reshape(values.shape[:-1] + (3, 3))


@dataclass(frozen=True)
class CanonicalActionEpisode:
    source: str
    actions_signed: np.ndarray
    gripper_close01: np.ndarray
    valid_mask: np.ndarray
    contract_version: str = CANONICAL_ACTION_SCHEMA


def canonicalize_episode_actions(
    raw_action: np.ndarray,
    *,
    source: str,
    action_kind: str,
    state_pose: np.ndarray | None = None,
) -> CanonicalActionEpisode:
    """Canonicalize a complete cached episode before any window operation."""

    key = validate_canonical_action_metadata(source=source, action_kind=action_kind)
    raw = _require_dense7(raw_action, source=key)
    close01 = _require_close01(raw[:, 6], source=key)

    if key == "bridge":
        # Bridge rotation_delta is a fixed/world-axis XYZ Euler command in the
        # base frame (Rz @ Ry @ Rx).  Convert the actual SO(3) delta, never
        # treat the three Euler coordinates as a rotation vector.
        rotation = rotation_matrix_to_rotvec(euler_xyz_to_matrix(raw[:, 3:6]))
    else:
        if state_pose is None:
            raise CanonicalActionContractError(
                "DROID canonicalization requires the validated state_pose sidecar"
            )
        pose = np.asarray(state_pose, dtype=np.float32)
        expected_shape = (raw.shape[0] + 1, 6)
        if pose.shape != expected_shape or not np.isfinite(pose).all():
            raise CanonicalActionContractError(
                f"DROID state_pose must be finite {expected_shape}, got {pose.shape}"
            )
        if np.any(raw[:, 3:6] < -np.pi - 1e-5) or np.any(
            raw[:, 3:6] >= np.pi + 1e-5
        ):
            raise CanonicalActionContractError(
                "DROID interval RPY deltas violate the wrapped [-pi,pi) contract"
            )
        # The v6 interval cache stores wrap(command_rpy - state_rpy), not an
        # SO(3) logarithm.  Adding the wrapped difference reconstructs an Euler
        # triple with the same R_cmd (modulo 2*pi).  The left-relative delta
        # R_cmd @ R_state.T is the base-frame rotation required by v7.
        state_rotation = euler_xyz_to_matrix(pose[:-1, 3:6])
        command_rotation = euler_xyz_to_matrix(pose[:-1, 3:6] + raw[:, 3:6])
        base_delta_rotation = command_rotation @ np.swapaxes(state_rotation, -1, -2)
        rotation = rotation_matrix_to_rotvec(base_delta_rotation)

    actions = np.empty_like(raw, dtype=np.float32)
    actions[:, :3] = raw[:, :3]
    actions[:, 3:6] = rotation
    actions[:, 6] = close01 * 2.0 - 1.0
    valid_mask = np.ones(7, dtype=np.bool_)
    return CanonicalActionEpisode(
        source=key,
        actions_signed=actions,
        gripper_close01=close01.astype(np.float32),
        valid_mask=valid_mask,
    )


def _manifest_path(value: object, *, manifest_path: Path, label: str) -> Path:
    raw = Path(str(value or ""))
    if not str(raw) or ".." in raw.parts:
        raise CanonicalActionContractError(f"{label} is empty or contains traversal")
    path = raw if raw.is_absolute() else manifest_path.parent / raw
    path = path.absolute()
    if path.suffix != ".npy":
        raise CanonicalActionContractError(f"{label} must point to a .npy file: {path}")
    # Do not stat tens of thousands of payloads in every DDP rank at startup.
    # The first actual episode read uses _read_pinned_bytes, which rejects a
    # missing/non-regular/symlink path and verifies the exact row digest.
    return path


def _manifest_shape(value: object, *, rank: int, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != rank:
        raise CanonicalActionContractError(f"{label} must be a rank-{rank} JSON shape")
    try:
        shape = tuple(int(dimension) for dimension in value)
    except (TypeError, ValueError) as exc:
        raise CanonicalActionContractError(f"{label} contains a non-integer") from exc
    if any(dimension <= 0 for dimension in shape):
        raise CanonicalActionContractError(f"{label} dimensions must be positive")
    return shape


@dataclass(frozen=True)
class CanonicalActionCacheEntry:
    """One immutable, content-addressed canonical action episode."""

    source: str
    source_name: str
    clip_id: str
    split: str
    action_kind: str
    action_frame_offset: int
    n_frames: int
    action_path: Path
    action_sha256: str
    action_shape: tuple[int, int]
    grip_close01_path: Path
    grip_close01_sha256: str
    grip_close01_shape: tuple[int]
    adapter_version: str
    implementation_sha256: str
    raw_action_sha256: str
    state_pose_sha256: str | None
    action_contract_evidence_sha256: str

    @property
    def action_count(self) -> int:
        return int(self.action_shape[0])

    def load_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Load and verify the exact cached arrays used by training.

        This performs only content-addressed NPY I/O and vectorized contract
        checks.  The expensive Euler/SO(3) episode conversion belongs in the
        offline generator and is never repeated from ``__getitem__``.
        """

        action_payload = _read_pinned_bytes(
            self.action_path,
            self.action_sha256,
            label=f"canonical actions for {self.clip_id}",
        )
        grip_payload = _read_pinned_bytes(
            self.grip_close01_path,
            self.grip_close01_sha256,
            label=f"canonical close01 gripper for {self.clip_id}",
        )
        try:
            actions = np.load(io.BytesIO(action_payload), allow_pickle=False)
            close01 = np.load(io.BytesIO(grip_payload), allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise CanonicalActionContractError(
                f"cannot decode canonical action payload for {self.clip_id}: {exc}"
            ) from exc
        if not isinstance(actions, np.ndarray) or not isinstance(close01, np.ndarray):
            raise CanonicalActionContractError(
                f"canonical payload for {self.clip_id} must contain plain NPY arrays"
            )
        if actions.dtype != np.float32 or tuple(actions.shape) != self.action_shape:
            raise CanonicalActionContractError(
                f"canonical actions for {self.clip_id} disagree with manifest: "
                f"dtype={actions.dtype} shape={actions.shape}"
            )
        if close01.dtype != np.float32 or tuple(close01.shape) != self.grip_close01_shape:
            raise CanonicalActionContractError(
                f"canonical close01 for {self.clip_id} disagrees with manifest: "
                f"dtype={close01.dtype} shape={close01.shape}"
            )
        if close01.shape[0] != actions.shape[0]:
            raise CanonicalActionContractError(
                f"canonical action/gripper length mismatch for {self.clip_id}"
            )
        if not np.isfinite(actions).all() or not np.isfinite(close01).all():
            raise CanonicalActionContractError(
                f"canonical payload for {self.clip_id} contains non-finite values"
            )
        if np.any(close01 < -1e-6) or np.any(close01 > 1.0 + 1e-6):
            raise CanonicalActionContractError(
                f"canonical close01 for {self.clip_id} lies outside [0,1]"
            )
        expected_signed = close01 * np.float32(2.0) - np.float32(1.0)
        if not np.allclose(actions[:, 6], expected_signed, atol=2e-6, rtol=0.0):
            raise CanonicalActionContractError(
                f"canonical signed gripper for {self.clip_id} is not 2*close01-1"
            )
        rotation_norm = np.linalg.norm(actions[:, 3:6], axis=1)
        if np.any(rotation_norm > np.pi + 2e-5):
            raise CanonicalActionContractError(
                f"canonical rotation for {self.clip_id} is outside principal SO(3) Log"
            )
        actions = np.asarray(actions, dtype=np.float32).copy()
        close01 = np.asarray(close01, dtype=np.float32).copy()
        actions.setflags(write=False)
        close01.setflags(write=False)
        return actions, close01


@dataclass(frozen=True)
class CanonicalActionCacheManifest:
    path: Path
    sha256: str
    entries: Mapping[str, CanonicalActionCacheEntry]
    implementation_sha256: str


def load_canonical_action_cache_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_sources: Iterable[str],
) -> CanonicalActionCacheManifest:
    """Load the exact runtime cache index and reject ambiguous provenance."""

    manifest_path = Path(path).absolute()
    digest = _require_sha256(expected_sha256, label="canonical cache manifest digest")
    payload = _read_pinned_bytes(
        manifest_path,
        digest,
        label="canonical action cache manifest",
    )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CanonicalActionContractError(
            "canonical action cache manifest is not UTF-8"
        ) from exc
    normalized_sources = frozenset(
        canonical_action_source(source) for source in expected_sources
    )
    if not normalized_sources:
        raise CanonicalActionContractError(
            "canonical action cache manifest requires expected sources"
        )
    implementation_sha256 = canonical_action_implementation_sha256()
    entries: dict[str, CanonicalActionCacheEntry] = {}
    action_paths: set[Path] = set()
    grip_paths: set[Path] = set()
    observed_sources: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CanonicalActionContractError(
                f"canonical action manifest line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise CanonicalActionContractError(
                f"canonical action manifest line {line_number} is not an object"
            )
        if row.get("schema") != CANONICAL_ACTION_CACHE_ROW_SCHEMA:
            raise CanonicalActionContractError(
                f"canonical action manifest line {line_number} uses an invalid schema"
            )
        source = canonical_action_source(row.get("dataset", ""))
        if source not in normalized_sources:
            raise CanonicalActionContractError(
                f"canonical action manifest contains unexpected source {source!r}"
            )
        source_name = str(row.get("source_name", ""))
        if source_name != CANONICAL_SOURCE_NAMES[source]:
            raise CanonicalActionContractError(
                f"canonical action manifest source_name mismatch for {source}"
            )
        clip_id = str(row.get("clip_id", ""))
        if not clip_id or clip_id in entries:
            raise CanonicalActionContractError(
                f"canonical action manifest has empty/duplicate clip_id {clip_id!r}"
            )
        split = str(row.get("split", ""))
        if split not in {"train", "val"}:
            raise CanonicalActionContractError(
                f"canonical action manifest split for {clip_id} must be train/val"
            )
        action_kind = str(row.get("action_kind", ""))
        validate_canonical_action_metadata(source=source, action_kind=action_kind)
        try:
            action_frame_offset = int(row.get("action_frame_offset"))
            n_frames = int(row.get("n_frames"))
        except (TypeError, ValueError) as exc:
            raise CanonicalActionContractError(
                f"canonical temporal metadata is invalid for {clip_id}"
            ) from exc
        expected_offset = CANONICAL_ACTION_FRAME_OFFSETS[source]
        if action_frame_offset != expected_offset:
            raise CanonicalActionContractError(
                f"canonical action offset for {clip_id} is {action_frame_offset}, "
                f"expected {expected_offset}"
            )
        if n_frames <= 0:
            raise CanonicalActionContractError(
                f"canonical n_frames for {clip_id} must be positive"
            )
        adapter_version = str(row.get("adapter_version", ""))
        if adapter_version != CANONICAL_ACTION_SCHEMA:
            raise CanonicalActionContractError(
                f"canonical action manifest adapter mismatch for {clip_id}"
            )
        row_implementation_sha256 = _require_sha256(
            row.get("implementation_sha256"),
            label=f"canonical implementation digest for {clip_id}",
        )
        if row_implementation_sha256 != implementation_sha256:
            raise CanonicalActionContractError(
                f"canonical artifact for {clip_id} was built by a different adapter "
                f"implementation: file={row_implementation_sha256} "
                f"runtime={implementation_sha256}"
            )
        action_shape_raw = _manifest_shape(
            row.get("action_shape"), rank=2, label=f"action_shape for {clip_id}"
        )
        if action_shape_raw[1] != 7:
            raise CanonicalActionContractError(
                f"canonical action_shape for {clip_id} must be [N,7]"
            )
        grip_shape_raw = _manifest_shape(
            row.get("grip_close01_shape"),
            rank=1,
            label=f"grip_close01_shape for {clip_id}",
        )
        if grip_shape_raw[0] != action_shape_raw[0]:
            raise CanonicalActionContractError(
                f"canonical action/gripper manifest lengths differ for {clip_id}"
            )
        if row.get("action_dtype") != "float32" or row.get("grip_close01_dtype") != "float32":
            raise CanonicalActionContractError(
                f"canonical payload dtypes for {clip_id} must be exact float32"
            )
        action_path = _manifest_path(
            row.get("action_path"),
            manifest_path=manifest_path,
            label=f"action_path for {clip_id}",
        )
        grip_path = _manifest_path(
            row.get("grip_close01_path"),
            manifest_path=manifest_path,
            label=f"grip_close01_path for {clip_id}",
        )
        if action_path in action_paths or grip_path in grip_paths:
            raise CanonicalActionContractError(
                f"canonical manifest aliases a payload path across clips: {clip_id}"
            )
        action_paths.add(action_path)
        grip_paths.add(grip_path)
        state_pose_sha256 = row.get("state_pose_sha256")
        if source == "droid":
            state_pose_sha256 = _require_sha256(
                state_pose_sha256,
                label=f"DROID state_pose digest for {clip_id}",
            )
        elif state_pose_sha256 is not None:
            state_pose_sha256 = _require_sha256(
                state_pose_sha256,
                label=f"state_pose digest for {clip_id}",
            )
        entry = CanonicalActionCacheEntry(
            source=source,
            source_name=source_name,
            clip_id=clip_id,
            split=split,
            action_kind=action_kind,
            action_frame_offset=action_frame_offset,
            n_frames=n_frames,
            action_path=action_path,
            action_sha256=_require_sha256(
                row.get("action_sha256"), label=f"action digest for {clip_id}"
            ),
            action_shape=(int(action_shape_raw[0]), 7),
            grip_close01_path=grip_path,
            grip_close01_sha256=_require_sha256(
                row.get("grip_close01_sha256"),
                label=f"grip_close01 digest for {clip_id}",
            ),
            grip_close01_shape=(int(grip_shape_raw[0]),),
            adapter_version=adapter_version,
            implementation_sha256=row_implementation_sha256,
            raw_action_sha256=_require_sha256(
                row.get("raw_action_sha256"),
                label=f"raw action digest for {clip_id}",
            ),
            state_pose_sha256=state_pose_sha256,
            action_contract_evidence_sha256=_require_sha256(
                row.get("action_contract_evidence_sha256"),
                label=f"action contract evidence digest for {clip_id}",
            ),
        )
        entries[clip_id] = entry
        observed_sources.add(source)
    if not entries:
        raise CanonicalActionContractError("canonical action cache manifest is empty")
    missing_sources = sorted(normalized_sources - observed_sources)
    if missing_sources:
        raise CanonicalActionContractError(
            f"canonical action cache manifest omits sources {missing_sources}"
        )
    return CanonicalActionCacheManifest(
        path=manifest_path,
        sha256=digest,
        entries=entries,
        implementation_sha256=implementation_sha256,
    )


@dataclass(frozen=True)
class CanonicalActionStats:
    source: str
    source_name: str
    stats_key: str
    mean: np.ndarray
    std: np.ndarray
    path: Path
    count: int
    source_manifest_sha256: str
    action_cache_manifest_sha256: str
    action_audit_gate_sha256: str

    def normalize_pose(self, pose: np.ndarray) -> np.ndarray:
        values = np.asarray(pose, dtype=np.float32)
        if values.shape[-1] != 6:
            raise CanonicalActionContractError(
                f"pose to normalize must end in 6 dimensions, got {values.shape}"
            )
        return ((values - self.mean) / self.std).astype(np.float32)


def _npz_scalar(data: Mapping[str, np.ndarray], key: str) -> str:
    if key not in data:
        raise CanonicalActionContractError(
            f"canonical action stats omit required metadata {key!r}"
        )
    value = np.asarray(data[key])
    if value.size != 1:
        raise CanonicalActionContractError(
            f"canonical action stats metadata {key!r} must be scalar"
        )
    return str(value.reshape(()).item())


def load_canonical_action_stats(
    path: str | Path,
    *,
    expected_source: str,
) -> CanonicalActionStats:
    """Load a source-bound stats file; legacy pooled NPZ files are rejected."""

    stats_path = Path(path)
    try:
        with np.load(stats_path, allow_pickle=False) as data:
            schema = (
                _npz_scalar(data, "schema_version")
                if "schema_version" in data
                else CANONICAL_ACTION_STATS_SCHEMA
            )
            contract = _npz_scalar(data, "action_adapter_version")
            source = canonical_action_source(_npz_scalar(data, "dataset"))
            source_name = _npz_scalar(data, "source_name")
            split = _npz_scalar(data, "split")
            gripper_semantics = _npz_scalar(data, "gripper_semantics")
            source_manifest_sha256 = _npz_scalar(data, "source_manifest_sha256")
            action_cache_manifest_sha256 = _npz_scalar(
                data, "action_cache_manifest_sha256"
            )
            action_audit_gate_sha256 = _npz_scalar(
                data, "action_audit_gate_sha256"
            )
            if any(
                key not in data
                for key in ("mean", "std", "count", "grip_close_rate")
            ):
                raise CanonicalActionContractError(
                    "canonical action stats require mean, std, count, and "
                    "grip_close_rate"
                )
            mean = np.asarray(data["mean"], dtype=np.float32).copy()
            std = np.asarray(data["std"], dtype=np.float32).copy()
            count_values = np.asarray(data["count"])
            grip_close_rate_values = np.asarray(data["grip_close_rate"])
    except (OSError, ValueError) as exc:
        if isinstance(exc, CanonicalActionContractError):
            raise
        raise CanonicalActionContractError(
            f"cannot load canonical action stats {stats_path}: {exc}"
        ) from exc
    expected = canonical_action_source(expected_source)
    if schema != CANONICAL_ACTION_STATS_SCHEMA:
        raise CanonicalActionContractError(
            f"stats {stats_path} use schema {schema!r}; legacy/pooled stats are forbidden"
        )
    if contract != CANONICAL_ACTION_SCHEMA:
        raise CanonicalActionContractError(
            f"stats {stats_path} target action contract {contract!r}, expected "
            f"{CANONICAL_ACTION_SCHEMA!r}"
        )
    if source != expected:
        raise CanonicalActionContractError(
            f"stats source mismatch: file={source!r} expected={expected!r}"
        )
    expected_source_name = CANONICAL_SOURCE_NAMES[expected]
    if source_name != expected_source_name:
        raise CanonicalActionContractError(
            f"stats source_name mismatch: file={source_name!r} "
            f"expected={expected_source_name!r}"
        )
    if split != "train":
        raise CanonicalActionContractError(
            f"canonical action stats must be train-only, got split={split!r}"
        )
    if gripper_semantics != CANONICAL_GRIPPER_SEMANTICS:
        raise CanonicalActionContractError(
            "canonical action stats use the wrong gripper semantics"
        )
    for label, digest in (
        ("source_manifest_sha256", source_manifest_sha256),
        ("action_cache_manifest_sha256", action_cache_manifest_sha256),
        ("action_audit_gate_sha256", action_audit_gate_sha256),
    ):
        _require_sha256(digest, label=f"canonical action stats {label}")
    if mean.shape != (6,) or std.shape != (6,):
        raise CanonicalActionContractError(
            f"canonical pose stats must be exact [6], got mean={mean.shape} std={std.shape}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 1e-8):
        raise CanonicalActionContractError(
            "canonical pose stats must be finite with strictly positive std"
        )
    if count_values.size != 1 or int(count_values.reshape(()).item()) <= 0:
        raise CanonicalActionContractError("canonical action stats count must be positive")
    if grip_close_rate_values.size != 1:
        raise CanonicalActionContractError(
            "canonical action stats grip_close_rate must be scalar"
        )
    grip_close_rate = float(grip_close_rate_values.reshape(()).item())
    if not np.isfinite(grip_close_rate) or not (0.0 < grip_close_rate < 1.0):
        raise CanonicalActionContractError(
            "canonical action stats grip_close_rate must lie strictly in (0,1)"
        )
    return CanonicalActionStats(
        source=source,
        source_name=source_name,
        stats_key=source_name,
        mean=mean,
        std=std,
        path=stats_path,
        count=int(count_values.reshape(()).item()),
        source_manifest_sha256=source_manifest_sha256,
        action_cache_manifest_sha256=action_cache_manifest_sha256,
        action_audit_gate_sha256=action_audit_gate_sha256,
    )
