from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from wm3d_v3.data.manifest import OXEClipRecord
from wm3d_v3.stage1.action_contract import (
    ActionContractBoundaryError,
    action_contract_key,
    canonical_dataset_name,
    resolve_action_window,
)
from wm3d_v3.stage1.action_signal_extractor import (
    ActionSignalExtractionError,
    RobotMaskSpec,
    load_record_robot_state,
)
from wm3d_v3.stage1.droid_interval_action import (
    DROID_INTERVAL_ACTION_CACHE_SUBDIR,
    DROID_INTERVAL_ACTION_KIND,
    DROID_INTERVAL_STATE_CACHE_SUBDIR,
)


MASK_REGISTRY_SCHEMA = "wm3d_v6_robot_mask_registry_v1"
TEMPORAL_WINDOW_BINDING_SCHEMA = "wm3d_v6_action_temporal_window_binding_v1"
TEMPORAL_WINDOW_DERIVATION = (
    "sha256(seed|contract_key|group_id|clip_id) mod legal_start_count"
)
EVIDENCE_CONTEXT_LENGTH = 1
EVIDENCE_CANDIDATE_OFFSETS = tuple(range(-2, 3))


class EvidenceExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class TemporalOffsetBinding:
    offset: int
    action_frame_indices: tuple[int, ...]
    previous_gripper_index: int


@dataclass(frozen=True)
class TemporalWindowBinding:
    contract_key: str
    group_id: str
    clip_id: str
    seed: int
    context_length: int
    target_length: int
    usable_frames: int
    n_action_frames: int
    legal_start_count: int
    legal_starts_sha256: str
    selection_sha256: str
    selected_legal_start_index: int
    start: int
    target_frame_indices: tuple[int, ...]
    by_offset: Mapping[int, TemporalOffsetBinding]


def _canonical_sha256(payload: object) -> str:
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_mask_registry(path: Path) -> dict[str, RobotMaskSpec]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != MASK_REGISTRY_SCHEMA:
        raise EvidenceExtractionError(
            f"unexpected robot mask registry schema: "
            f"{payload.get('schema_version')!r}"
        )
    raw_domains = payload.get("domains")
    if not isinstance(raw_domains, dict) or not raw_domains:
        raise EvidenceExtractionError("robot mask registry has no domains")
    masks: dict[str, RobotMaskSpec] = {}
    for raw_name, raw_spec in sorted(raw_domains.items()):
        if not isinstance(raw_spec, dict):
            raise EvidenceExtractionError(
                f"invalid mask specification for {raw_name}"
            )
        domain = canonical_dataset_name(raw_name)
        box = tuple(float(value) for value in raw_spec.get("normalized_box", ()))
        if len(box) != 4:
            raise EvidenceExtractionError(
                f"mask normalized_box must have four values for {domain}"
            )
        spec = RobotMaskSpec(
            normalized_box=box,
            motion_quantile=float(raw_spec.get("motion_quantile", 0.65)),
            min_motion_pixels=int(raw_spec.get("min_motion_pixels", 128)),
        )
        spec.validate()
        if domain in masks:
            raise EvidenceExtractionError(
                f"duplicate canonical mask for {domain}"
            )
        masks[domain] = spec
    return masks


def safe_clip_id(clip_id: str) -> str:
    return str(clip_id).replace("/", "__")


def load_clip_cache(
    record: OXEClipRecord,
    cache_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe = safe_clip_id(record.clip_id)
    domain = canonical_dataset_name(record.dataset)
    if domain == "droid":
        if record.action_kind != DROID_INTERVAL_ACTION_KIND:
            raise EvidenceExtractionError(
                f"DROID contract requires action_kind "
                f"{DROID_INTERVAL_ACTION_KIND!r}, got {record.action_kind!r}"
            )
        action_path = (
            cache_root / DROID_INTERVAL_ACTION_CACHE_SUBDIR / f"{safe}.npy"
        )
    else:
        action_path = cache_root / "actions" / f"{safe}.npy"
    rgb_path = cache_root / "rgb_256" / f"{safe}.npy"
    depth_path = cache_root / "vggt_geom" / f"{safe}.npz"
    missing = [
        str(path)
        for path in (action_path, rgb_path, depth_path)
        if not path.is_file()
    ]
    if missing:
        raise EvidenceExtractionError(f"missing cache files: {missing}")
    actions = np.asarray(np.load(action_path), dtype=np.float64)
    rgb = np.asarray(np.load(rgb_path))
    with np.load(depth_path) as geometry:
        depth_key = "depth" if "depth" in geometry.files else "depth_map"
        if depth_key not in geometry.files:
            raise EvidenceExtractionError(
                f"depth cache has no depth key: {depth_path}"
            )
        depth = np.asarray(geometry[depth_key], dtype=np.float64)
    return actions, rgb, depth


def load_alignment_robot_state(
    record: OXEClipRecord,
    *,
    cache_root: Path,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if canonical_dataset_name(record.dataset) != "droid":
        try:
            return load_record_robot_state(record)
        except ActionSignalExtractionError as exc:
            raise EvidenceExtractionError(str(exc)) from exc
    if record.action_kind != DROID_INTERVAL_ACTION_KIND:
        raise EvidenceExtractionError(
            f"DROID robot state requires action_kind "
            f"{DROID_INTERVAL_ACTION_KIND!r}, got {record.action_kind!r}"
        )
    safe = safe_clip_id(record.clip_id)
    state_path = (
        cache_root / DROID_INTERVAL_STATE_CACHE_SUBDIR / f"{safe}.npz"
    )
    if not state_path.is_file():
        raise EvidenceExtractionError(
            f"missing DROID interval robot-state cache: {state_path}"
        )
    with np.load(state_path) as payload:
        missing = {"pose", "grip"}.difference(payload.files)
        if missing:
            raise EvidenceExtractionError(
                f"DROID interval robot-state cache missing {sorted(missing)}: "
                f"{state_path}"
            )
        pose = np.asarray(payload["pose"], dtype=np.float64)
        grip = np.asarray(payload["grip"], dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] < 6:
        raise EvidenceExtractionError(f"invalid DROID pose shape: {pose.shape}")
    if grip.ndim not in (1, 2):
        raise EvidenceExtractionError(f"invalid DROID grip shape: {grip.shape}")
    grip = grip.reshape(grip.shape[0], -1)[:, 0]
    if pose.shape[0] != grip.shape[0]:
        raise EvidenceExtractionError("DROID pose/grip length mismatch")
    if not np.isfinite(pose).all() or not np.isfinite(grip).all():
        raise EvidenceExtractionError(
            "DROID robot state contains non-finite values"
        )
    return pose[:, :6], grip


def bind_temporal_window(
    record: OXEClipRecord,
    *,
    group_id: str,
    seed: int,
    usable_frames: int,
    n_action_frames: int,
    target_length: int,
    context_length: int = EVIDENCE_CONTEXT_LENGTH,
    candidate_offsets: tuple[int, ...] = EVIDENCE_CANDIDATE_OFFSETS,
) -> TemporalWindowBinding:
    usable_frames = int(usable_frames)
    n_action_frames = int(n_action_frames)
    target_length = int(target_length)
    context_length = int(context_length)
    if target_length < 8:
        raise EvidenceExtractionError("target_length must be at least 8")
    if usable_frames <= 0 or n_action_frames <= 0 or context_length <= 0:
        raise EvidenceExtractionError(
            "temporal binding requires positive frame and context lengths"
        )
    if tuple(int(value) for value in candidate_offsets) != EVIDENCE_CANDIDATE_OFFSETS:
        raise EvidenceExtractionError(
            f"candidate offsets must be {list(EVIDENCE_CANDIDATE_OFFSETS)}"
        )
    max_start = usable_frames - context_length - target_length
    if max_start < 0:
        raise EvidenceExtractionError(
            "clip has no legal temporal window: "
            f"usable={usable_frames} context={context_length} target={target_length}"
        )

    bounded_record = replace(record, n_frames=usable_frames)
    legal_windows: list[tuple[int, dict[int, object]]] = []
    for start in range(max_start + 1):
        resolutions = {}
        try:
            for offset in EVIDENCE_CANDIDATE_OFFSETS:
                resolutions[offset] = resolve_action_window(
                    bounded_record,
                    start=start,
                    T=context_length,
                    k=target_length,
                    offset=offset,
                    n_action_frames=n_action_frames,
                )
        except ActionContractBoundaryError:
            continue
        legal_windows.append((start, resolutions))
    if not legal_windows:
        raise EvidenceExtractionError(
            "clip has no legal temporal window with all candidate offsets "
            f"for {record.clip_id}"
        )

    contract_key = action_contract_key(record)
    selection_sha = hashlib.sha256(
        f"{int(seed)}|{contract_key}|{group_id}|{record.clip_id}".encode("utf-8")
    ).hexdigest()
    selected_index = int(selection_sha, 16) % len(legal_windows)
    start, resolutions = legal_windows[selected_index]
    target_frame_indices = tuple(
        int(value) for value in resolutions[EVIDENCE_CANDIDATE_OFFSETS[0]].target_frame_indices
    )
    for offset, resolution in resolutions.items():
        current_target = tuple(int(value) for value in resolution.target_frame_indices)
        if current_target != target_frame_indices:
            raise EvidenceExtractionError(
                f"target frame indices drift across offsets for {record.clip_id}"
            )
    legal_starts = [window_start for window_start, _ in legal_windows]
    by_offset = {
        int(offset): TemporalOffsetBinding(
            offset=int(offset),
            action_frame_indices=tuple(
                int(value) for value in resolution.action_frame_indices
            ),
            previous_gripper_index=int(resolution.previous_gripper_index),
        )
        for offset, resolution in sorted(resolutions.items())
    }
    return TemporalWindowBinding(
        contract_key=contract_key,
        group_id=str(group_id),
        clip_id=str(record.clip_id),
        seed=int(seed),
        context_length=context_length,
        target_length=target_length,
        usable_frames=usable_frames,
        n_action_frames=n_action_frames,
        legal_start_count=len(legal_windows),
        legal_starts_sha256=_canonical_sha256(legal_starts),
        selection_sha256=selection_sha,
        selected_legal_start_index=selected_index,
        start=int(start),
        target_frame_indices=target_frame_indices,
        by_offset=by_offset,
    )


def temporal_window_binding_payload(
    binding: TemporalWindowBinding,
) -> dict[str, object]:
    return {
        "schema_version": TEMPORAL_WINDOW_BINDING_SCHEMA,
        "derivation": TEMPORAL_WINDOW_DERIVATION,
        "contract_key": binding.contract_key,
        "group_id": binding.group_id,
        "clip_id": binding.clip_id,
        "seed": binding.seed,
        "context_length": binding.context_length,
        "target_length": binding.target_length,
        "candidate_offsets": list(EVIDENCE_CANDIDATE_OFFSETS),
        "usable_frames": binding.usable_frames,
        "n_action_frames": binding.n_action_frames,
        "legal_start_count": binding.legal_start_count,
        "legal_starts_sha256": binding.legal_starts_sha256,
        "selection_sha256": binding.selection_sha256,
        "selected_legal_start_index": binding.selected_legal_start_index,
        "start": binding.start,
        "target_frame_indices": list(binding.target_frame_indices),
        "by_offset": {
            str(offset): {
                "offset": value.offset,
                "action_frame_indices": list(value.action_frame_indices),
                "previous_gripper_index": value.previous_gripper_index,
            }
            for offset, value in sorted(binding.by_offset.items())
        },
    }
