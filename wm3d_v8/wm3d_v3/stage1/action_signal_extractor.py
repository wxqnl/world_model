from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import pickle
from pathlib import Path
import tarfile
from typing import Any
import warnings

import cv2
import numpy as np

from wm3d_v3.data.manifest import OXEClipRecord
from wm3d_v3.stage1.action_alignment import AlignmentSignals
from wm3d_v3.stage1.action_contract import canonical_dataset_name


class ActionSignalExtractionError(ValueError):
    pass


class LegacyRobotMaskWarning(UserWarning):
    """The normalized-box robot-mask API is compatibility-only."""


GEOMETRY_LOG_DEPTH_CHANGE_EPSILON = 1e-3
_FLOW_MAGNITUDE_EPSILON = 1e-3
_MINIMUM_GEOMETRY_SUPPORT = 16
_MINIMUM_BACKGROUND_SCALE_SUPPORT = 16
_BACKGROUND_STATIC_FLOW_QUANTILE = 0.25
_BACKGROUND_STATIC_FLOW_MAX_PIXELS = 0.25
_GEOMETRY_SIGNAL_VARIATION_EPSILON = 1e-6
_AUDIT_TARGET_FRAME_COUNT = 16


@dataclass(frozen=True)
class ActionSignalConfig:
    motion_quantile: float = 0.7
    min_motion_pixels: int = 512

    def validate(self) -> None:
        if not (0.0 <= float(self.motion_quantile) < 1.0):
            raise ActionSignalExtractionError("motion_quantile must be in [0, 1)")
        if int(self.min_motion_pixels) < 16:
            raise ActionSignalExtractionError("min_motion_pixels must be at least 16")


FORMAL_ACTION_SIGNAL_CONFIG = ActionSignalConfig()


@dataclass(frozen=True)
class RobotMaskSpec:
    normalized_box: tuple[float, float, float, float]
    motion_quantile: float = 0.65
    min_motion_pixels: int = 128

    def validate(self) -> None:
        x0, y0, x1, y1 = (float(value) for value in self.normalized_box)
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ActionSignalExtractionError(
                f"invalid normalized robot box {self.normalized_box}"
            )
        if not (0.0 <= float(self.motion_quantile) < 1.0):
            raise ActionSignalExtractionError(
                f"motion_quantile must be in [0, 1), got {self.motion_quantile}"
            )
        if int(self.min_motion_pixels) < 16:
            raise ActionSignalExtractionError(
                "min_motion_pixels must be at least 16"
            )


def _robot_mask(spec: RobotMaskSpec, *, height: int, width: int) -> np.ndarray:
    spec.validate()
    height = int(height)
    width = int(width)
    if height < 8 or width < 8:
        raise ActionSignalExtractionError(
            f"robot mask resolution is too small: {height}x{width}"
        )
    x0, y0, x1, y1 = spec.normalized_box
    left = max(0, min(width - 1, int(np.floor(x0 * width))))
    right = max(left + 1, min(width, int(np.ceil(x1 * width))))
    top = max(0, min(height - 1, int(np.floor(y0 * height))))
    bottom = max(top + 1, min(height, int(np.ceil(y1 * height))))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:bottom, left:right] = 1
    if int(mask.sum()) < int(spec.min_motion_pixels):
        raise ActionSignalExtractionError(
            "registered robot mask contains fewer pixels than min_motion_pixels"
        )
    return mask.astype(bool)


def materialize_registered_robot_masks(
    spec: RobotMaskSpec,
    *,
    frame_count: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Materialize the immutable registry workspace as explicit frame masks."""
    frame_count = int(frame_count)
    if frame_count < 1:
        raise ActionSignalExtractionError("frame_count must be positive")
    mask = _robot_mask(spec, height=height, width=width)
    return np.repeat(mask[None, ...], frame_count, axis=0)


def robot_mask_sha256(
    spec: RobotMaskSpec,
    *,
    height: int,
    width: int,
) -> str:
    mask = _robot_mask(spec, height=height, width=width)
    payload = json.dumps(
        {
            "spec": asdict(spec),
            "height": int(height),
            "width": int(width),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(payload)
    digest.update(mask.astype(np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def _validate_cache_arrays(
    rgb: np.ndarray,
    depth: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rgb_array = np.asarray(rgb)
    depth_array = np.asarray(depth, dtype=np.float64)
    if (
        rgb_array.ndim != 4
        or rgb_array.shape[-1] != 3
        or rgb_array.dtype != np.uint8
    ):
        raise ActionSignalExtractionError(
            f"rgb must be uint8 [T, H, W, 3], got {rgb_array.shape} {rgb_array.dtype}"
        )
    if depth_array.ndim != 3:
        raise ActionSignalExtractionError(
            f"depth must be [T, H, W], got {depth_array.shape}"
        )
    if rgb_array.shape[0] != depth_array.shape[0]:
        raise ActionSignalExtractionError(
            f"rgb/depth frame counts differ: {rgb_array.shape[0]} "
            f"!= {depth_array.shape[0]}"
        )
    if target.ndim != 1 or target.size < 8 or np.any(np.diff(target) != 1):
        raise ActionSignalExtractionError(
            "target_frame_indices must contain at least eight contiguous frames"
        )
    if int(target.min()) < 1 or int(target.max()) >= rgb_array.shape[0]:
        raise ActionSignalExtractionError(
            "target frame indices need an in-bounds predecessor and current frame"
        )
    return rgb_array, depth_array


@dataclass(frozen=True)
class _MotionSupportStats:
    support: np.ndarray
    support_count: int


def _motion_support(
    flow: np.ndarray,
    base_mask: np.ndarray,
    photometric_change: np.ndarray,
    spec: RobotMaskSpec,
) -> _MotionSupportStats:
    magnitude = np.linalg.norm(flow, axis=-1)
    candidates = magnitude[base_mask]
    if candidates.size == 0:
        return _MotionSupportStats(
            support=np.zeros_like(base_mask, dtype=bool),
            support_count=0,
        )
    threshold = float(np.quantile(candidates, float(spec.motion_quantile)))
    support = (
        base_mask
        & photometric_change
        & np.isfinite(magnitude)
        & (magnitude >= max(threshold, _FLOW_MAGNITUDE_EPSILON))
    )
    support_count = int(support.sum())
    return _MotionSupportStats(
        support=support,
        support_count=support_count,
    )


def _resize_mask(mask: np.ndarray, *, height: int, width: int) -> np.ndarray:
    return cv2.resize(
        mask.astype(np.uint8),
        (int(width), int(height)),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def _validate_robot_masks(
    robot_masks: np.ndarray,
    *,
    frame_count: int,
    height: int,
    width: int,
) -> np.ndarray:
    masks = np.asarray(robot_masks)
    expected_shape = (int(frame_count), int(height), int(width))
    if masks.shape != expected_shape:
        raise ActionSignalExtractionError(
            f"robot_masks must be [T, H, W] matching rgb, got {masks.shape}; "
            f"expected {expected_shape}"
        )
    if masks.dtype != np.bool_:
        if not np.issubdtype(masks.dtype, np.number):
            raise ActionSignalExtractionError("robot_masks must be boolean or binary")
        if not np.isfinite(masks).all() or not np.all((masks == 0) | (masks == 1)):
            raise ActionSignalExtractionError("robot_masks must contain only 0/1 values")
    return masks.astype(bool, copy=False)


def _resolve_robot_masks(
    *,
    robot_masks: np.ndarray | None,
    frame_count: int,
    height: int,
    width: int,
) -> np.ndarray:
    if robot_masks is None:
        raise ActionSignalExtractionError(
            "formal signal extraction requires explicit per-frame robot_masks"
        )
    return _validate_robot_masks(
        robot_masks,
        frame_count=frame_count,
        height=height,
        width=width,
    )


def _flow_coordinate_maps(
    flow: np.ndarray,
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flow_array = np.asarray(flow, dtype=np.float32)
    source_height, source_width = flow_array.shape[:2]
    if (source_height, source_width) != (int(height), int(width)):
        flow_array = cv2.resize(
            flow_array,
            (int(width), int(height)),
            interpolation=cv2.INTER_LINEAR,
        )
        flow_array[..., 0] *= float(width) / float(source_width)
        flow_array[..., 1] *= float(height) / float(source_height)
    grid_x, grid_y = np.meshgrid(
        np.arange(int(width), dtype=np.float32),
        np.arange(int(height), dtype=np.float32),
    )
    map_x = grid_x + flow_array[..., 0]
    map_y = grid_y + flow_array[..., 1]
    in_bounds = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0.0)
        & (map_x <= float(width - 1))
        & (map_y >= 0.0)
        & (map_y <= float(height - 1))
    )
    return map_x, map_y, in_bounds


def _warp_mask_to_previous(
    current_mask: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    in_bounds: np.ndarray,
) -> np.ndarray:
    warped = cv2.remap(
        np.asarray(current_mask, dtype=np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    return warped & in_bounds


def _flow_warp_current_depth(
    current_depth: np.ndarray,
    flow: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample current depth on the previous-frame grid using prev->current flow."""
    depth = np.asarray(current_depth, dtype=np.float64)
    height, width = depth.shape
    map_x, map_y, in_bounds = _flow_coordinate_maps(
        flow,
        height=height,
        width=width,
    )
    source_valid = np.isfinite(depth) & (depth > 0.0)
    source_values = np.where(source_valid, depth, 0.0).astype(np.float32)
    source_weights = source_valid.astype(np.float32)
    warped_values = cv2.remap(
        source_values,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warped_weights = cv2.remap(
        source_weights,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    sampling_valid = in_bounds & (warped_weights >= 1.0 - 1e-6)
    warped_depth = np.zeros((height, width), dtype=np.float64)
    np.divide(
        warped_values,
        warped_weights,
        out=warped_depth,
        where=sampling_valid,
    )
    return warped_depth, sampling_valid, map_x, map_y, in_bounds


def _static_background_support(
    candidate: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    """Keep only fixed-rule low-flow background depth correspondences."""
    height, width = candidate.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    magnitude = np.hypot(map_x - grid_x, map_y - grid_y)
    finite_candidate = candidate & np.isfinite(magnitude)
    values = magnitude[finite_candidate]
    if values.size == 0:
        return np.zeros_like(candidate, dtype=bool)
    quantile_cutoff = float(
        np.quantile(values, _BACKGROUND_STATIC_FLOW_QUANTILE)
    )
    cutoff = min(quantile_cutoff, _BACKGROUND_STATIC_FLOW_MAX_PIXELS)
    return finite_candidate & (magnitude <= cutoff)


def extract_cache_alignment_signals(
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    target_frame_indices: tuple[int, ...],
    signal_config: ActionSignalConfig = FORMAL_ACTION_SIGNAL_CONFIG,
    state_pose: np.ndarray | None,
    state_grip: np.ndarray | None,
    robot_masks: np.ndarray | None = None,
) -> AlignmentSignals:
    """Extract signals; robot_masks is mandatory for the formal path.

    Omitting it retains the old normalized-box behavior solely as an explicitly
    warned legacy API. Formal callers materialize the immutable registered
    workspace and pass it explicitly for every frame.
    """
    target = np.asarray(target_frame_indices, dtype=np.int64)
    rgb_array, depth_array = _validate_cache_arrays(rgb, depth, target)
    signal_config.validate()
    frame_masks = _resolve_robot_masks(
        robot_masks=robot_masks,
        frame_count=rgb_array.shape[0],
        height=rgb_array.shape[1],
        width=rgb_array.shape[2],
    )
    minimum_motion_support = int(signal_config.min_motion_pixels)

    flow_vectors: list[np.ndarray] = []
    depth_deltas: list[float] = []
    motion_support_counts: list[int] = []
    valid_depth_counts: list[int] = []
    geometry_support_counts: list[int] = []
    valid_depth_candidate_counts: list[int] = []
    registered_areas: list[int] = []
    registered_depth_areas: list[int] = []
    for frame_index in target:
        previous_mask = frame_masks[int(frame_index) - 1]
        current_mask = frame_masks[int(frame_index)]
        registered_areas.append(int(previous_mask.sum()))
        previous_rgb = rgb_array[int(frame_index) - 1]
        current_rgb = rgb_array[int(frame_index)]
        previous_gray = cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY)
        current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        rgb_map_x, rgb_map_y, rgb_in_bounds = _flow_coordinate_maps(
            flow, height=rgb_array.shape[1], width=rgb_array.shape[2]
        )
        current_mask_at_flow = _warp_mask_to_previous(
            current_mask, rgb_map_x, rgb_map_y, rgb_in_bounds
        )
        robot_correspondence = previous_mask & current_mask_at_flow
        photometric_change = np.any(previous_rgb != current_rgb, axis=-1)
        support_stats = _motion_support(
            flow, robot_correspondence, photometric_change, signal_config
        )
        motion_support_counts.append(support_stats.support_count)
        if support_stats.support_count > 0:
            flow_vector = np.asarray(
                np.median(flow[support_stats.support], axis=0),
                dtype=np.float64,
            )
        else:
            flow_vector = np.zeros(2, dtype=np.float64)
        if not np.isfinite(flow_vector).all():
            flow_vector = np.zeros(2, dtype=np.float64)
        flow_vectors.append(flow_vector)

        previous_depth = depth_array[int(frame_index) - 1]
        current_depth = depth_array[int(frame_index)]
        previous_depth_mask = _resize_mask(
            previous_mask,
            height=depth_array.shape[1],
            width=depth_array.shape[2],
        )
        current_depth_mask = _resize_mask(
            current_mask,
            height=depth_array.shape[1],
            width=depth_array.shape[2],
        )
        registered_depth_areas.append(int(previous_depth_mask.sum()))
        depth_support = _resize_mask(
            support_stats.support,
            height=depth_array.shape[1],
            width=depth_array.shape[2],
        )
        (
            current_depth_at_flow,
            current_depth_valid,
            depth_map_x,
            depth_map_y,
            depth_in_bounds,
        ) = _flow_warp_current_depth(current_depth, flow)
        current_depth_mask_at_flow = _warp_mask_to_previous(
            current_depth_mask,
            depth_map_x,
            depth_map_y,
            depth_in_bounds,
        )
        valid_depth_candidate = (
            depth_support
            & previous_depth_mask
            & current_depth_mask_at_flow
            & depth_in_bounds
        )
        valid_depth_candidate_counts.append(int(valid_depth_candidate.sum()))
        previous_depth_valid = np.isfinite(previous_depth) & (previous_depth > 0.0)
        correspondence_valid = previous_depth_valid & current_depth_valid
        valid_depth = valid_depth_candidate & correspondence_valid
        valid_depth_counts.append(int(valid_depth.sum()))

        background_candidate = (
            correspondence_valid
            & ~previous_depth_mask
            & ~current_depth_mask_at_flow
        )
        background_valid = _static_background_support(
            background_candidate,
            depth_map_x,
            depth_map_y,
        )
        corrected_log_delta = np.zeros_like(previous_depth, dtype=np.float64)
        scale_is_calibrated = (
            int(background_valid.sum()) >= _MINIMUM_BACKGROUND_SCALE_SUPPORT
        )
        if scale_is_calibrated:
            raw_log_delta = np.zeros_like(previous_depth, dtype=np.float64)
            raw_log_delta[correspondence_valid] = (
                np.log(current_depth_at_flow[correspondence_valid])
                - np.log(previous_depth[correspondence_valid])
            )
            background_scale_drift = float(
                np.median(raw_log_delta[background_valid])
            )
            corrected_log_delta[correspondence_valid] = (
                raw_log_delta[correspondence_valid] - background_scale_drift
            )
        geometry_support = (
            valid_depth
            & scale_is_calibrated
            & (
                np.abs(corrected_log_delta)
                > GEOMETRY_LOG_DEPTH_CHANGE_EPSILON
            )
        )
        geometry_support_count = int(geometry_support.sum())
        geometry_support_counts.append(geometry_support_count)
        depth_delta = (
            float(np.median(corrected_log_delta[geometry_support]))
            if geometry_support_count > 0
            else 0.0
        )
        depth_deltas.append(depth_delta if np.isfinite(depth_delta) else 0.0)

    flow_array = np.asarray(flow_vectors, dtype=np.float64)
    depth_array_delta = np.asarray(depth_deltas, dtype=np.float64)
    total_motion_support = int(sum(motion_support_counts))
    total_registered_area = int(sum(registered_areas))
    total_valid_depth = int(sum(valid_depth_counts))
    total_geometry_support = int(sum(geometry_support_counts))
    total_registered_depth_area = int(sum(registered_depth_areas))
    total_valid_depth_candidates = int(sum(valid_depth_candidate_counts))
    flow_informative = bool(
        all(
            count >= minimum_motion_support
            for count in motion_support_counts
        )
        and np.isfinite(flow_array).all()
    )
    if not flow_informative:
        flow_array = np.zeros_like(flow_array)

    geometry_signal_finite = bool(np.isfinite(depth_array_delta).all())
    geometry_signal_nonzero = bool(
        geometry_signal_finite
        and np.any(
            np.abs(depth_array_delta)
            > GEOMETRY_LOG_DEPTH_CHANGE_EPSILON
        )
    )
    geometry_signal_nonflat = bool(
        geometry_signal_finite
        and np.ptp(depth_array_delta) > _GEOMETRY_SIGNAL_VARIATION_EPSILON
    )
    geometry_informative = bool(
        flow_informative
        and all(
            count >= _MINIMUM_GEOMETRY_SUPPORT
            for count in geometry_support_counts
        )
        and geometry_signal_nonzero
        and geometry_signal_nonflat
    )
    if not geometry_informative:
        depth_array_delta = np.zeros_like(depth_array_delta)

    pose_delta = None
    if state_pose is not None:
        pose = np.asarray(state_pose, dtype=np.float64)
        pose_dim = 6 if pose.ndim == 2 and pose.shape[1] >= 6 else 3
        if (
            pose.ndim != 2
            or pose.shape[1] < 3
            or pose.shape[0] <= int(target.max())
            or not np.isfinite(pose[:, :pose_dim]).all()
        ):
            raise ActionSignalExtractionError(
                "state_pose must be finite [T, >=3] and cover target frames"
            )
        pose_delta = pose[target, :pose_dim] - pose[target - 1, :pose_dim]
        if pose_dim == 6:
            pose_delta[:, 3:6] = (
                pose_delta[:, 3:6] + np.pi
            ) % (2.0 * np.pi) - np.pi

    grip = None
    if state_grip is not None:
        grip = np.asarray(state_grip, dtype=np.float64).reshape(-1)
        if grip.shape[0] <= int(target.max()) or not np.isfinite(grip).all():
            raise ActionSignalExtractionError(
                "state_grip must be finite [T] and cover target frames"
            )

    return AlignmentSignals(
        target_frame_indices=tuple(int(value) for value in target),
        state_pose_delta=pose_delta,
        flow_vectors=flow_array,
        depth_delta=depth_array_delta,
        state_grip=grip,
        flow_informative=flow_informative,
        geometry_informative=geometry_informative,
        robot_mask_motion_coverage=float(
            total_motion_support / max(total_registered_area, 1)
        ),
        valid_depth_coverage=float(
            total_valid_depth / max(total_valid_depth_candidates, 1)
        ),
        flow_support_count=total_motion_support,
        flow_support_fraction=float(
            total_motion_support / max(total_registered_area, 1)
        ),
        geometry_support_count=total_geometry_support,
        geometry_support_fraction=float(
            total_geometry_support / max(total_registered_depth_area, 1)
        ),
        flow_fallback_used=False,
        geometry_fallback_used=False,
        flow_support_threshold=int(
            minimum_motion_support * _AUDIT_TARGET_FRAME_COUNT
        ),
        geometry_support_threshold=int(
            _MINIMUM_GEOMETRY_SUPPORT * _AUDIT_TARGET_FRAME_COUNT
        ),
    )


def _stack_observation_series(
    steps: list[dict[str, Any]],
    key: str,
    *,
    minimum_dim: int,
) -> np.ndarray | None:
    values: list[np.ndarray] = []
    for step in steps:
        observation = step.get("observation", {})
        if key not in observation:
            return None
        value = np.asarray(observation[key], dtype=np.float64).reshape(-1)
        if value.size < minimum_dim or not np.isfinite(value).all():
            return None
        values.append(value)
    if not values:
        return None
    return np.stack(values)


def extract_episode_robot_state(
    dataset: str,
    episode: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    dataset = canonical_dataset_name(dataset)
    if dataset == "droid":
        return None, None
    raw_steps = episode.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ActionSignalExtractionError("raw OXE episode has no steps")
    steps: list[dict[str, Any]] = raw_steps

    pose_key_by_dataset = {
        "bridge": "state",
        "fractal20220817_data": "base_pose_tool_reached",
        "taco_play": "robot_obs",
        "jaco_play": "end_effector_cartesian_pos",
        "kuka": "clip_function_input/base_pose_tool_reached",
    }
    pose_key = pose_key_by_dataset[dataset]
    pose = _stack_observation_series(steps, pose_key, minimum_dim=3)

    if dataset == "bridge":
        state = _stack_observation_series(steps, "state", minimum_dim=7)
        grip = None if state is None else state[:, 6]
    elif dataset in {"fractal20220817_data", "kuka"}:
        raw_grip = _stack_observation_series(
            steps,
            "gripper_closed",
            minimum_dim=1,
        )
        grip = None if raw_grip is None else raw_grip[:, 0]
    elif dataset == "taco_play":
        robot_obs = _stack_observation_series(steps, "robot_obs", minimum_dim=15)
        grip = None if robot_obs is None else robot_obs[:, -1]
    elif dataset == "jaco_play":
        joint_pos = _stack_observation_series(steps, "joint_pos", minimum_dim=8)
        grip = None if joint_pos is None else joint_pos[:, -2:].mean(axis=1)
    else:
        raise ActionSignalExtractionError(f"unsupported dataset {dataset}")
    return pose, grip


def load_record_robot_state(
    record: OXEClipRecord,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if canonical_dataset_name(record.dataset) == "droid":
        return None, None
    tar_path = Path(record.tar_path)
    if not tar_path.is_file() or not record.pickle_member:
        raise ActionSignalExtractionError(
            f"missing raw OXE episode reference for {record.clip_id}"
        )
    with tarfile.open(tar_path, "r") as archive:
        member = archive.extractfile(record.pickle_member)
        if member is None:
            raise ActionSignalExtractionError(
                f"missing pickle member {record.pickle_member}"
            )
        episode = pickle.load(member)
    return extract_episode_robot_state(record.dataset, episode)
