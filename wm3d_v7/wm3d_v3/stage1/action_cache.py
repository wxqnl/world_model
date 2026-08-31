from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from wm3d_v3.stage1.action_contract import (
    ActionContractFileError,
    _read_stable_regular_file,
)
from wm3d_v3.data.manifest import OXEClipRecord
from wm3d_v3.stage1.droid_interval_action import (
    DROID_INTERVAL_ACTION_CACHE_SUBDIR,
    DROID_INTERVAL_ACTION_KIND,
    DROID_INTERVAL_ACTION_VALID_COUNT,
    DROID_INTERVAL_STATE_CACHE_SUBDIR,
    DROID_INTERVAL_STATE_COUNT,
    DROID_INTERVAL_TERMINAL_POLICY,
)


DEFAULT_ACTION_CACHE_SUBDIR = "actions"
LEGACY_DROID_ACTION_KIND = "droid_action_7d"
DROID_FINAL_INDEX_SCHEMA = "wm3d_v6_stage1_droid_interval_cache_v2"


class ActionCacheResolutionError(ValueError):
    """Raised when record metadata cannot select a valid action cache."""


@dataclass(frozen=True)
class ActionCacheResolution:
    path: Path
    cache_subdir: str

    def valid_action_count(self, stored_action_count: int) -> int:
        stored_action_count = int(stored_action_count)
        valid_count = stored_action_count
        if valid_count <= 0:


            raise ActionCacheResolutionError(
                f"action cache has no valid actions: path={self.path} "
                f"stored_action_count={stored_action_count}"
            )
        return valid_count

@dataclass(frozen=True)
class ValidatedDroidCacheRecord:

    def __getitem__(self, key: str) -> Any:
        return self.metadata[key]

    def __iter__(self):
        return iter(self.metadata)

    def __len__(self) -> int:
        return len(self.metadata)
    metadata: Mapping[str, Any]
    actions: np.ndarray
    pose: np.ndarray
    grip: np.ndarray


def read_stable_cache_bytes(path: str | Path, label: str) -> bytes:
    try:
        return _read_stable_regular_file(Path(path), label)
    except ActionContractFileError as exc:
        raise ActionCacheResolutionError(str(exc)) from exc



def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file_hash(path: Path, expected: object, label: str) -> bytes:
    payload = read_stable_cache_bytes(path, label)
    if hashlib.sha256(payload).hexdigest() != str(expected):
        raise ActionCacheResolutionError(f"{label} hash mismatch: {path}")
    return payload


def validate_formal_droid_cache_index(
    records: Sequence[OXEClipRecord],
    *,
    cache_root: str | Path,
    index_path: str | Path,
    index_payload_bytes: bytes | None = None,
    expected_index_sha256: str | None = None,
) -> dict[str, ValidatedDroidCacheRecord]:
    index_path = Path(index_path)
    cache_root = Path(cache_root)
    index_bytes = (
        read_stable_cache_bytes(index_path, "finalized DROID cache index")
        if index_payload_bytes is None
        else bytes(index_payload_bytes)
    )
    actual_index_sha256 = hashlib.sha256(index_bytes).hexdigest()
    if expected_index_sha256 is not None and actual_index_sha256 != expected_index_sha256:
        raise ActionCacheResolutionError(
            "finalized DROID cache index differs from promoted snapshot"
        )
    try:
        payload = json.loads(index_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ActionCacheResolutionError(
            f"cannot read finalized DROID cache index {index_path}: {exc}"
        ) from exc
    if payload.get("schema_version") != DROID_FINAL_INDEX_SCHEMA:
        raise ActionCacheResolutionError("unexpected finalized DROID index schema")
    plan_id = str(payload.get("plan_id", ""))
    if len(plan_id) != 64 or any(
        character not in "0123456789abcdef" for character in plan_id
    ):
        raise ActionCacheResolutionError("finalized DROID index plan_id is invalid")
    commit = payload.get("commit")
    output_manifest = payload.get("output_manifest")
    if (
        not isinstance(commit, dict)
        or commit.get("protocol") != "manifest_then_index_marker_v1"
        or not isinstance(output_manifest, dict)
    ):
        raise ActionCacheResolutionError(
            "finalized DROID index commit marker is invalid"
        )
    manifest_path = Path(str(output_manifest.get("path", ""))).absolute()
    if Path(str(commit.get("manifest_path", ""))).absolute() != manifest_path:
        raise ActionCacheResolutionError(
            "finalized DROID commit marker manifest path mismatch"
        )
    _require_file_hash(
        manifest_path,
        output_manifest.get("sha256"),
        "finalized DROID output manifest",
    )
    if commit.get("manifest_sha256") != output_manifest.get("sha256"):
        raise ActionCacheResolutionError(
            "finalized DROID commit marker manifest hash mismatch"
        )
    generation_payload = {
        key: value for key, value in payload.items() if key != "commit"
    }
    generation_id = hashlib.sha256(
        json.dumps(
            generation_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    if commit.get("generation_id") != generation_id:
        raise ActionCacheResolutionError(
            "finalized DROID commit marker generation mismatch"
        )
    index_records = payload.get("records")
    coverage = payload.get("coverage")
    if (
        not isinstance(index_records, dict)
        or not isinstance(coverage, dict)
        or coverage.get("exact") is not True
        or int(coverage.get("planned", -1)) != len(index_records)
        or int(coverage.get("built", -1)) != len(index_records)
    ):
        raise ActionCacheResolutionError(
            "finalized DROID index coverage is not exact"
        )
    contract = payload.get("action_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("action_kind") != DROID_INTERVAL_ACTION_KIND
        or int(contract.get("action_dim", 0)) != 7
        or contract.get("valid_action_count")
        != DROID_INTERVAL_ACTION_VALID_COUNT
        or contract.get("state_count") != DROID_INTERVAL_STATE_COUNT
        or contract.get("terminal_policy") != DROID_INTERVAL_TERMINAL_POLICY
    ):
        raise ActionCacheResolutionError(
            "finalized DROID index action contract is invalid"
        )

    validated: dict[str, ValidatedDroidCacheRecord] = {}
    for record in records:
        if str(record.dataset).strip().lower() != "droid":
            continue
        if record.action_kind != DROID_INTERVAL_ACTION_KIND:
            raise ActionCacheResolutionError(
                f"formal DROID record has invalid action kind: {record.clip_id}"
            )
        entry = index_records.get(record.clip_id)
        if not isinstance(entry, dict):
            raise ActionCacheResolutionError(
                f"finalized DROID index omits {record.clip_id}"
            )
        action_meta = entry.get("actions")
        state_meta = entry.get("state")
        if not isinstance(action_meta, dict) or not isinstance(state_meta, dict):
            raise ActionCacheResolutionError(
                f"finalized DROID index metadata is incomplete: {record.clip_id}"
            )
        safe = _safe_clip_id(record.clip_id)
        action_path = (
            cache_root / DROID_INTERVAL_ACTION_CACHE_SUBDIR / f"{safe}.npy"
        ).absolute()
        state_path = (
            cache_root / DROID_INTERVAL_STATE_CACHE_SUBDIR / f"{safe}.npz"
        ).absolute()
        if Path(str(action_meta.get("path", ""))).absolute() != action_path:
            raise ActionCacheResolutionError(
                f"DROID action path mismatch: {record.clip_id}"
            )
        if Path(str(state_meta.get("path", ""))).absolute() != state_path:
            raise ActionCacheResolutionError(
                f"DROID state path mismatch: {record.clip_id}"
            )
        expected_action_shape = [int(record.n_frames) - 1, 7]
        if (
            action_meta.get("shape") != expected_action_shape
            or action_meta.get("dtype") != "float32"
            or int(action_meta.get("valid_count", -1))
            != int(record.n_frames) - 1
        ):
            raise ActionCacheResolutionError(
                f"DROID action metadata shape/dtype mismatch: {record.clip_id}"
            )
        if (
            state_meta.get("pose_shape") != [int(record.n_frames), 6]
            or state_meta.get("grip_shape") != [int(record.n_frames)]
            or state_meta.get("pose_dtype") != "float32"
            or state_meta.get("grip_dtype") != "float32"
        ):
            raise ActionCacheResolutionError(
                f"DROID state metadata shape/dtype mismatch: {record.clip_id}"
            )
        action_bytes = _require_file_hash(
            action_path,
            action_meta.get("sha256"),
            f"DROID action for {record.clip_id}",
        )
        state_bytes = _require_file_hash(
            state_path,
            state_meta.get("sha256"),
            f"DROID state for {record.clip_id}",
        )
        actions = np.asarray(
            np.load(io.BytesIO(action_bytes), allow_pickle=False),
            dtype=np.float32,
        ).copy()
        if (
            list(actions.shape) != expected_action_shape
            or actions.dtype != np.float32
            or not np.isfinite(actions).all()
        ):
            raise ActionCacheResolutionError(
                f"DROID action sidecar content is invalid: {record.clip_id}"
            )
        with np.load(io.BytesIO(state_bytes), allow_pickle=False) as state:
            pose = np.asarray(state["pose"]).copy()
            grip = np.asarray(state["grip"]).copy()
        if (
            list(pose.shape) != [int(record.n_frames), 6]
            or list(grip.shape) != [int(record.n_frames)]
            or pose.dtype != np.float32
            or grip.dtype != np.float32
            or not np.isfinite(pose).all()
            or not np.isfinite(grip).all()
        ):
            raise ActionCacheResolutionError(
                f"DROID state sidecar content is invalid: {record.clip_id}"
            )
        for array in (actions, pose, grip):
            array.setflags(write=False)
        validated[record.clip_id] = ValidatedDroidCacheRecord(
            metadata=entry,
            actions=actions,
            pose=pose,
            grip=grip,
        )
    return validated


def _safe_clip_id(clip_id: str) -> str:
    return str(clip_id).replace("/", "__")


def resolve_action_cache(
    record: OXEClipRecord,
    *,
    cache_root: str | Path,
    formal_stage1: bool,
) -> ActionCacheResolution:
    """Resolve the only action cache compatible with record metadata."""

    cache_root = Path(cache_root)
    is_droid = str(record.dataset).strip().lower() == "droid"
    action_kind = str(record.action_kind).strip()

    if is_droid:
        if action_kind == DROID_INTERVAL_ACTION_KIND:
            cache_subdir = DROID_INTERVAL_ACTION_CACHE_SUBDIR
        elif formal_stage1:
            if action_kind == LEGACY_DROID_ACTION_KIND:
                raise ActionCacheResolutionError(
                    "formal Stage1 rejects legacy DROID action_kind "
                    f"{LEGACY_DROID_ACTION_KIND!r}; expected "
                    f"{DROID_INTERVAL_ACTION_KIND!r}"
                )
            raise ActionCacheResolutionError(
                "formal Stage1 DROID records require action_kind "
                f"{DROID_INTERVAL_ACTION_KIND!r}, got {action_kind!r}"
            )
        else:
            cache_subdir = DEFAULT_ACTION_CACHE_SUBDIR
    else:
        cache_subdir = DEFAULT_ACTION_CACHE_SUBDIR

    path = cache_root / cache_subdir / f"{_safe_clip_id(record.clip_id)}.npy"
    return ActionCacheResolution(
        path=path,
        cache_subdir=cache_subdir,
    )
