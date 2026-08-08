"""Strict data contracts for WM3D-v7.

The v6 manifest is intentionally left untouched so existing checkpoints and
loaders remain reproducible.  New v7 data must pass through this module before
it is allowed to contribute to action-conditioned or outcome losses.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Iterator, Literal


CANONICAL_ACTION_VERSION = "wm3d_v7_base_delta_axisangle_gripclose_v1"
MODEL_HZ = 5.0
CONTEXT_FRAMES = 16
FUTURE_FRAMES = 8
WINDOW_STRIDE = 2

Split = Literal["train", "val", "test"]
ViewRole = Literal["external_anchor", "wrist", "external_alternate"]


def _stable_sha256(parts: Iterable[object]) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_clip_hash(
    source: str,
    native_episode_id: str,
    native_start_frame: int,
    native_end_frame: int,
    action_adapter_version: str,
) -> str:
    """Return the globally unique identity used for v7 deduplication."""
    return _stable_sha256(
        (
            source,
            native_episode_id,
            int(native_start_frame),
            int(native_end_frame),
            action_adapter_version,
        )
    )


def stable_split(split_group: str) -> Split:
    """Deterministic 80/10/10 split made before any fitting or branch rollout."""
    bucket = int(_stable_sha256((split_group,))[:16], 16) % 10_000
    if bucket < 8_000:
        return "train"
    if bucket < 9_000:
        return "val"
    return "test"


@dataclass(frozen=True)
class V7ViewSpec:
    role: ViewRole
    image_key: str
    camera_id: str
    timestamp_key: str
    intrinsics_key: str | None = None
    extrinsics_key: str | None = None
    calibrated: bool = False


@dataclass(frozen=True)
class V7ActionSpec:
    adapter_version: str
    raw_kind: str
    source_frame: str
    rotation_repr: str
    translation_unit: str
    rotation_unit: str
    control_hz: float
    is_delta: bool
    gripper_semantics: str
    action_key: str
    observation_timestamp_key: str
    action_timestamp_key: str
    future_timestamp_key: str
    action_valid: bool = False
    audit_report: str | None = None
    invalid_reason: str | None = None


@dataclass(frozen=True)
class V7BranchSpec:
    root_id: str
    branch_id: str
    root_seed: int
    action_path: str
    target_tokens_path: str | None
    target_geometry_path: str
    outcome_path: str
    simulator_state_path: str
    true_simulator_rollout: bool


@dataclass(frozen=True)
class V7ClipRecord:
    source: str
    native_episode_id: str
    native_start_frame: int
    native_end_frame: int
    native_fps: float
    raw_path: str
    task_text: str
    task_class: str
    scene_id: str
    robot: str
    embodiment_id: str
    split_group: str
    split: Split
    views: tuple[V7ViewSpec, ...]
    action: V7ActionSpec
    clip_hash: str
    branches: tuple[V7BranchSpec, ...] = field(default_factory=tuple)
    success_key: str | None = None
    subgoal_key: str | None = None
    representation_only: bool = False

    @property
    def action_valid(self) -> bool:
        return bool(self.action.action_valid and not self.representation_only)

    @property
    def has_true_counterfactual(self) -> bool:
        return bool(self.branches) and all(branch.true_simulator_rollout for branch in self.branches)


def validate_record(record: V7ClipRecord, *, require_hash_match: bool = True) -> None:
    if not record.source or not record.native_episode_id:
        raise ValueError("source and native_episode_id are required")
    if record.native_start_frame < 0 or record.native_end_frame <= record.native_start_frame:
        raise ValueError("native frame interval must be positive and non-empty")
    if not math.isfinite(record.native_fps) or record.native_fps <= 0:
        raise ValueError("native_fps must be finite and positive")
    if record.split != stable_split(record.split_group):
        raise ValueError(
            f"split leakage contract failed: record={record.split}, "
            f"expected={stable_split(record.split_group)}"
        )
    if not record.views:
        raise ValueError("at least one camera view is required")
    roles = [view.role for view in record.views]
    if roles.count("external_anchor") != 1:
        raise ValueError("exactly one external_anchor view is required")
    if len(roles) != len(set(roles)):
        raise ValueError(f"duplicate view roles are not allowed: {roles}")
    if record.action.action_valid:
        if record.action.adapter_version != CANONICAL_ACTION_VERSION:
            raise ValueError("action-valid clips must use the canonical v7 adapter version")
        if not record.action.audit_report:
            raise ValueError("action-valid clips require an immutable audit_report")
    if record.branches:
        if len(record.branches) < 2:
            raise ValueError("counterfactual roots require at least two branches")
        roots = {branch.root_id for branch in record.branches}
        branch_ids = {branch.branch_id for branch in record.branches}
        if len(roots) != 1 or len(branch_ids) != len(record.branches):
            raise ValueError("branches must share one root and have unique branch ids")
        for branch in record.branches:
            if not branch.true_simulator_rollout:
                raise ValueError("pseudo counterfactual branches are forbidden")
            if not branch.target_geometry_path or not branch.outcome_path:
                raise ValueError("every branch needs true geometry and outcome targets")
    expected_hash = canonical_clip_hash(
        record.source,
        record.native_episode_id,
        record.native_start_frame,
        record.native_end_frame,
        record.action.adapter_version,
    )
    if require_hash_match and record.clip_hash != expected_hash:
        raise ValueError(f"clip_hash mismatch: {record.clip_hash} != {expected_hash}")


def enumerate_window_keys(
    record: V7ClipRecord,
    *,
    model_hz: float = MODEL_HZ,
    context_frames: int = CONTEXT_FRAMES,
    future_frames: int = FUTURE_FRAMES,
    stride: int = WINDOW_STRIDE,
) -> Iterator[str]:
    """Enumerate distinct context+future windows after canonical 5 Hz resampling."""
    validate_record(record)
    if min(model_hz, context_frames, future_frames, stride) <= 0:
        raise ValueError("window parameters must be positive")
    native_count = record.native_end_frame - record.native_start_frame
    duration_seconds = (native_count - 1) / record.native_fps
    resampled_count = int(math.floor(duration_seconds * model_hz)) + 1
    required = context_frames + future_frames
    for window_start in range(0, max(0, resampled_count - required + 1), stride):
        yield f"{record.clip_hash}:{window_start:08d}"


def _record_from_dict(payload: dict) -> V7ClipRecord:
    views = tuple(V7ViewSpec(**view) for view in payload.get("views", ()))
    action = V7ActionSpec(**payload["action"])
    branches = tuple(V7BranchSpec(**branch) for branch in payload.get("branches", ()))
    values = dict(payload)
    values["views"] = views
    values["action"] = action
    values["branches"] = branches
    return V7ClipRecord(**values)


def read_v7_manifest(path: str | Path, *, validate: bool = True) -> Iterator[V7ClipRecord]:
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = _record_from_dict(json.loads(line))
            if validate:
                validate_record(record)
            if record.clip_hash in seen:
                raise ValueError(f"duplicate clip_hash at line {line_number}: {record.clip_hash}")
            seen.add(record.clip_hash)
            yield record


def write_v7_manifest(path: str | Path, records: Iterable[V7ClipRecord]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_record(record)
            if record.clip_hash in seen:
                raise ValueError(f"duplicate clip_hash: {record.clip_hash}")
            seen.add(record.clip_hash)
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
