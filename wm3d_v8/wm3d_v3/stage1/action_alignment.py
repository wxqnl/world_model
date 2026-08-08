from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from wm3d_v3.stage1.action_contract_evidence import (
    ClipOffsetEvidence,
    ModalityEvidence,
)


class ActionAlignmentError(ValueError):
    pass


@dataclass(frozen=True)
class AlignmentSignals:
    target_frame_indices: tuple[int, ...]
    state_pose_delta: np.ndarray | None
    flow_vectors: np.ndarray
    depth_delta: np.ndarray
    state_grip: np.ndarray | None
    flow_informative: bool = True
    geometry_informative: bool = True
    robot_mask_motion_coverage: float | None = None
    valid_depth_coverage: float | None = None
    flow_support_count: int | None = None
    flow_support_fraction: float | None = None
    geometry_support_count: int | None = None
    geometry_support_fraction: float | None = None
    flow_fallback_used: bool = False
    geometry_fallback_used: bool = False
    flow_support_threshold: int | None = None
    geometry_support_threshold: int | None = None


@dataclass(frozen=True)
class ProjectionCalibrationClip:
    clip_id: str
    actions: np.ndarray
    target_frame_indices: tuple[int, ...]
    action_frame_indices_by_offset: Mapping[int, tuple[int, ...]]
    flow_vectors: np.ndarray
    depth_delta: np.ndarray
    source_informative: bool = True


@dataclass(frozen=True)
class FrozenOffsetProjection:
    offset: int
    flow_weights: tuple[tuple[float, float], ...]
    flow_bias: tuple[float, float]
    depth_weights: tuple[float, float, float]
    depth_bias: float


@dataclass(frozen=True)
class FrozenProjectionSet:
    calibration_clip_ids: tuple[str, ...]
    by_offset: Mapping[int, FrozenOffsetProjection]
    ridge: float


@dataclass(frozen=True)
class GripTransitionMetrics:
    macro_f1: float
    up_recall: float | None
    down_recall: float | None
    up_count: int
    down_count: int
    informative: bool


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 3:
        raise ActionAlignmentError(f"{name} requires at least three values")
    if not np.isfinite(array).all():
        raise ActionAlignmentError(f"{name} contains non-finite values")
    return array


def _pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = _finite_vector(left, "left correlation input")
    right = _finite_vector(right, "right correlation input")
    if left.shape != right.shape:
        raise ActionAlignmentError(
            f"correlation inputs differ in shape: {left.shape} != {right.shape}"
        )
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def fisher_z_correlation(left: np.ndarray, right: np.ndarray) -> float:
    correlation = np.clip(
        _pearson_correlation(left, right),
        -0.999999,
        0.999999,
    )
    return float(np.arctanh(correlation))


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = _finite_vector(values, "rank input")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    return _pearson_correlation(_rankdata(left), _rankdata(right))


def signed_flow_projection_spearman(
    action_xyz: np.ndarray,
    flow_xy: np.ndarray,
    projection: np.ndarray,
) -> float:
    action = np.asarray(action_xyz, dtype=np.float64)
    flow = np.asarray(flow_xy, dtype=np.float64)
    projection = np.asarray(projection, dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 3:
        raise ActionAlignmentError(
            f"action_xyz must have shape [N, 3], got {action.shape}"
        )
    if flow.shape != (action.shape[0], 2):
        raise ActionAlignmentError(
            f"flow_xy must have shape [{action.shape[0]}, 2], got {flow.shape}"
        )
    if projection.shape != (2, 3):
        raise ActionAlignmentError(
            f"projection must have shape [2, 3], got {projection.shape}"
        )
    if not (
        np.isfinite(action).all()
        and np.isfinite(flow).all()
        and np.isfinite(projection).all()
    ):
        raise ActionAlignmentError("flow projection inputs contain non-finite values")
    projected = action @ projection.T
    return _spearman_correlation(projected.reshape(-1), flow.reshape(-1))


def _ridge_cross_validated_predictions(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim not in (1, 2) or x.shape[0] != y.shape[0]:
        raise ActionAlignmentError(
            f"invalid ridge projection shapes: features={x.shape} targets={y.shape}"
        )
    if x.shape[0] < 8 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ActionAlignmentError(
            "ridge projection needs at least eight finite aligned samples"
        )
    ridge = float(ridge)
    if not (ridge > 0.0):
        raise ActionAlignmentError(f"ridge must be positive, got {ridge}")
    y_matrix = y[:, None] if y.ndim == 1 else y
    predictions = np.empty_like(y_matrix, dtype=np.float64)
    sample_indices = np.arange(x.shape[0])
    for holdout_parity in (0, 1):
        holdout = sample_indices % 2 == holdout_parity
        train = ~holdout
        train_x = np.concatenate(
            [x[train], np.ones((int(train.sum()), 1), dtype=np.float64)],
            axis=1,
        )
        holdout_x = np.concatenate(
            [x[holdout], np.ones((int(holdout.sum()), 1), dtype=np.float64)],
            axis=1,
        )
        regularizer = np.eye(train_x.shape[1], dtype=np.float64) * ridge
        regularizer[-1, -1] = 0.0
        coefficients = np.linalg.solve(
            train_x.T @ train_x + regularizer,
            train_x.T @ y_matrix[train],
        )
        predictions[holdout] = holdout_x @ coefficients
    return predictions[:, 0] if y.ndim == 1 else predictions


def cross_validated_ridge_flow_spearman(
    action_xyz: np.ndarray,
    flow_xy: np.ndarray,
    *,
    ridge: float = 1e-3,
) -> float:
    action = np.asarray(action_xyz, dtype=np.float64)
    flow = np.asarray(flow_xy, dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 3:
        raise ActionAlignmentError(
            f"action_xyz must have shape [N, 3], got {action.shape}"
        )
    if flow.shape != (action.shape[0], 2):
        raise ActionAlignmentError(
            f"flow_xy must have shape [{action.shape[0]}, 2], got {flow.shape}"
        )
    predicted = _ridge_cross_validated_predictions(
        action,
        flow,
        ridge=ridge,
    )
    return _spearman_correlation(predicted.reshape(-1), flow.reshape(-1))


def cross_validated_ridge_depth_fisher_z(
    action_xyz: np.ndarray,
    depth_delta: np.ndarray,
    *,
    ridge: float = 1e-3,
) -> float:
    action = np.asarray(action_xyz, dtype=np.float64)
    depth = np.asarray(depth_delta, dtype=np.float64).reshape(-1)
    if action.ndim != 2 or action.shape[1] != 3:
        raise ActionAlignmentError(
            f"action_xyz must have shape [N, 3], got {action.shape}"
        )
    if depth.shape != (action.shape[0],):
        raise ActionAlignmentError(
            f"depth_delta must have shape [{action.shape[0]}], got {depth.shape}"
        )
    predicted = _ridge_cross_validated_predictions(
        action,
        depth,
        ridge=ridge,
    )
    return fisher_z_correlation(predicted, depth)


def negative_first_motion_lag_score(
    action_magnitude: np.ndarray,
    observed_magnitude: np.ndarray,
    *,
    max_lag: int,
) -> float:
    action = _finite_vector(action_magnitude, "action magnitude")
    observed = _finite_vector(observed_magnitude, "observed magnitude")
    if action.shape != observed.shape:
        raise ActionAlignmentError(
            f"motion magnitudes differ in shape: {action.shape} != {observed.shape}"
        )
    max_lag = int(max_lag)
    if max_lag < 0 or action.size <= 2 * max_lag + 2:
        raise ActionAlignmentError(
            f"invalid max_lag={max_lag} for sequence length {action.size}"
        )

    candidates: list[tuple[float, int]] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aligned_action = action[-lag:]
            aligned_observed = observed[:lag]
        elif lag > 0:
            aligned_action = action[:-lag]
            aligned_observed = observed[lag:]
        else:
            aligned_action = action
            aligned_observed = observed
        candidates.append(
            (_pearson_correlation(aligned_action, aligned_observed), lag)
        )
    _, best_lag = max(
        candidates,
        key=lambda item: (item[0], -abs(item[1]), -item[1]),
    )
    return -float(abs(best_lag))


def grip_transition_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
) -> GripTransitionMetrics:
    predicted_binary = (
        _finite_vector(predicted, "predicted grip") >= 0.5
    ).astype(np.int8)
    observed_binary = (
        _finite_vector(observed, "observed grip") >= 0.5
    ).astype(np.int8)
    if predicted_binary.shape != observed_binary.shape:
        raise ActionAlignmentError(
            f"grip sequences differ in shape: "
            f"{predicted_binary.shape} != {observed_binary.shape}"
        )
    predicted_delta = np.diff(predicted_binary)
    observed_delta = np.diff(observed_binary)
    scores: list[float] = []
    recalls: dict[int, float | None] = {}
    counts: dict[int, int] = {}
    for transition in (-1, 1):
        predicted_event = predicted_delta == transition
        observed_event = observed_delta == transition
        observed_count = int(np.count_nonzero(observed_event))
        counts[transition] = observed_count
        if observed_count == 0:
            recalls[transition] = None
            continue
        true_positive = int(np.count_nonzero(predicted_event & observed_event))
        false_positive = int(np.count_nonzero(predicted_event & ~observed_event))
        false_negative = observed_count - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(
            0.0 if denominator == 0 else 2.0 * true_positive / denominator
        )
        recalls[transition] = true_positive / observed_count
    informative = bool(scores)
    return GripTransitionMetrics(
        macro_f1=float(np.mean(scores)) if informative else 0.0,
        up_recall=recalls[1],
        down_recall=recalls[-1],
        up_count=counts[1],
        down_count=counts[-1],
        informative=informative,
    )


def _grip_transition_f1(predicted: np.ndarray, observed: np.ndarray) -> float:
    return grip_transition_metrics(predicted, observed).macro_f1


def _block_permutation(
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    block_size: int = 2,
) -> np.ndarray:
    values = np.asarray(values)
    blocks = [
        values[start : start + block_size]
        for start in range(0, values.shape[0], block_size)
    ]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[int(index)] for index in order], axis=0)


def _circular_shift(
    values: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values)
    shift = int(rng.integers(1, values.shape[0]))
    return np.roll(values, shift=shift, axis=0)


def _aligned_state_grip(
    state_grip: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    grip = np.asarray(state_grip, dtype=np.float64).reshape(-1)
    if grip.size == target_indices.size:
        aligned = grip
    elif target_indices.size and grip.size > int(target_indices.max()):
        aligned = grip[target_indices]
    else:
        raise ActionAlignmentError(
            "state_grip must be target-aligned or cover all target frame indices"
        )
    if not np.isfinite(aligned).all():
        raise ActionAlignmentError("state_grip contains non-finite values")
    return aligned


def _validated_action_frame_indices(
    mapping: Mapping[int, Sequence[int]],
    *,
    offsets: Sequence[int],
    target_size: int,
    action_count: int,
    label: str,
) -> dict[int, np.ndarray]:
    if set(mapping) != set(offsets):
        raise ActionAlignmentError(f"{label} must cover offsets -2..2 exactly")
    result: dict[int, np.ndarray] = {}
    for offset in offsets:
        indices = np.asarray(mapping[offset], dtype=np.int64)
        if indices.shape != (int(target_size),):
            raise ActionAlignmentError(
                f"{label}[{offset}] must have shape [{target_size}]"
            )
        if int(indices.min()) < 0 or int(indices.max()) >= int(action_count):
            raise ActionAlignmentError(f"{label}[{offset}] leaves action bounds")
        result[offset] = indices
    return result


def build_clip_offset_evidence(
    *,
    clip_id: str,
    actions: np.ndarray,
    signals: AlignmentSignals,
    projection: np.ndarray | None,
    action_frame_indices_by_offset: Mapping[int, Sequence[int]],
    offsets: Iterable[int],
    null_repeats: int,
    seed: int,
    frozen_projections: FrozenProjectionSet | None = None,
) -> list[ClipOffsetEvidence]:
    action_array = np.asarray(actions, dtype=np.float64)
    if action_array.ndim != 2 or action_array.shape[1] < 7:
        raise ActionAlignmentError(
            f"actions must have shape [N, >=7], got {action_array.shape}"
        )
    if not np.isfinite(action_array).all():
        raise ActionAlignmentError("actions contain non-finite values")
    target = np.asarray(signals.target_frame_indices, dtype=np.int64)
    if target.ndim != 1 or target.size < 8:
        raise ActionAlignmentError("at least eight target frame indices are required")
    if np.any(np.diff(target) != 1):
        raise ActionAlignmentError("target frame indices must be contiguous")
    offset_values = tuple(int(value) for value in offsets)
    if offset_values != tuple(range(-2, 3)):
        raise ActionAlignmentError("offsets must be exactly (-2, -1, 0, 1, 2)")
    null_repeats = int(null_repeats)
    if null_repeats < 32:
        raise ActionAlignmentError("null_repeats must be at least 32")
    action_indices_by_offset = _validated_action_frame_indices(
        action_frame_indices_by_offset,
        offsets=offset_values,
        target_size=int(target.size),
        action_count=int(action_array.shape[0]),
        label=f"action_frame_indices_by_offset for {clip_id}",
    )

    flow = np.asarray(signals.flow_vectors, dtype=np.float64)
    depth = np.asarray(signals.depth_delta, dtype=np.float64).reshape(-1)
    if flow.shape != (target.size, 2):
        raise ActionAlignmentError(
            f"flow_vectors must have shape [{target.size}, 2], got {flow.shape}"
        )
    if depth.shape != (target.size,):
        raise ActionAlignmentError(
            f"depth_delta must have shape [{target.size}], got {depth.shape}"
        )
    if not np.isfinite(flow).all() or not np.isfinite(depth).all():
        raise ActionAlignmentError("flow/depth signals contain non-finite values")

    pose = None
    if signals.state_pose_delta is not None:
        pose = np.asarray(signals.state_pose_delta, dtype=np.float64)
        if (
            pose.ndim != 2
            or pose.shape not in ((target.size, 3), (target.size, 6))
            or not np.isfinite(pose).all()
        ):
            raise ActionAlignmentError(
                f"state_pose_delta must have shape [{target.size}, 3 or 6]"
            )
    grip = None
    if signals.state_grip is not None:
        grip = _aligned_state_grip(signals.state_grip, target)

    projection_array = (
        None if projection is None else np.asarray(projection, dtype=np.float64)
    )
    if projection_array is not None and frozen_projections is not None:
        raise ActionAlignmentError(
            "legacy projection and frozen projections are mutually exclusive"
        )
    if frozen_projections is not None and set(
        frozen_projections.by_offset
    ) != set(offset_values):
        raise ActionAlignmentError("frozen projections must cover offsets -2..2")
    seed_sequence = np.random.SeedSequence(int(seed))
    child_sequences = seed_sequence.spawn(len(offset_values))
    rows: list[ClipOffsetEvidence] = []
    for offset, child_seed in zip(offset_values, child_sequences, strict=True):
        candidate = action_array[action_indices_by_offset[offset]]
        rng = np.random.default_rng(child_seed)
        modalities: dict[str, ModalityEvidence] = {}

        def add_modality(
            name: str,
            family: str,
            observed: float,
            nulls: list[float],
            informative: bool = True,
            *,
            robot_mask_motion_coverage: float | None = None,
            valid_depth_coverage: float | None = None,
            support_count: int | None = None,
            support_fraction: float | None = None,
            minimum_support_count: int | None = None,
            fallback_used: bool = False,
        ) -> None:
            modalities[name] = ModalityEvidence(
                family=family,
                source_class=(
                    "proprioceptive"
                    if family in {"state", "gripper"}
                    else "exteroceptive"
                ),
                observed=float(observed),
                null_samples=tuple(float(value) for value in nulls),
                direction=1,
                scale_epsilon=0.05,
                informative=bool(informative),
                robot_mask_motion_coverage=robot_mask_motion_coverage,
                valid_depth_coverage=valid_depth_coverage,
                support_count=support_count,
                support_fraction=support_fraction,
                minimum_support_count=minimum_support_count,
                fallback_used=bool(fallback_used),
            )

        if pose is not None:
            add_modality(
                "state_pose_xyz_fisher_z",
                "state",
                fisher_z_correlation(candidate[:, :3], pose[:, :3]),
                [
                    fisher_z_correlation(
                        _block_permutation(candidate[:, :3], rng),
                        pose[:, :3],
                    )
                    for _ in range(null_repeats)
                ],
            )
            if pose.shape[1] == 6:
                add_modality(
                    "state_pose_rpy_fisher_z",
                    "state",
                    fisher_z_correlation(candidate[:, 3:6], pose[:, 3:6]),
                    [
                        fisher_z_correlation(
                            _block_permutation(candidate[:, 3:6], rng),
                            pose[:, 3:6],
                        )
                        for _ in range(null_repeats)
                    ],
                )
        if frozen_projections is not None:
            frozen_projection = frozen_projections.by_offset[offset]
            flow_name = "flow_frozen_ridge_spearman"
            depth_name = "depth_frozen_ridge_fisher_z"

            def flow_score(action_xyz: np.ndarray) -> float:
                return score_frozen_offset_projection(
                    action_xyz,
                    flow,
                    depth,
                    frozen_projection,
                )[0]

            def depth_score(action_xyz: np.ndarray) -> float:
                return score_frozen_offset_projection(
                    action_xyz,
                    flow,
                    depth,
                    frozen_projection,
                )[1]

            flow_weights = np.asarray(
                frozen_projection.flow_weights,
                dtype=np.float64,
            )
            flow_bias = np.asarray(
                frozen_projection.flow_bias,
                dtype=np.float64,
            )
            candidate_motion = np.linalg.norm(
                candidate[:, :3] @ flow_weights + flow_bias,
                axis=1,
            )
        elif projection_array is None:
            flow_name = "flow_cv_ridge_spearman"
            depth_name = "depth_cv_ridge_fisher_z"

            def flow_score(action_xyz: np.ndarray) -> float:
                return cross_validated_ridge_flow_spearman(
                    action_xyz,
                    flow,
                    ridge=1e-3,
                )

            def depth_score(action_xyz: np.ndarray) -> float:
                return cross_validated_ridge_depth_fisher_z(
                    action_xyz,
                    depth,
                    ridge=1e-3,
                )

            candidate_motion = np.linalg.norm(candidate[:, :3], axis=1)
        else:
            flow_name = "flow_projection_spearman"
            depth_name = "depth_fisher_z"

            def flow_score(action_xyz: np.ndarray) -> float:
                return signed_flow_projection_spearman(
                    action_xyz,
                    flow,
                    projection_array,
                )

            def depth_score(action_xyz: np.ndarray) -> float:
                return fisher_z_correlation(action_xyz[:, 2], depth)

            candidate_motion = np.linalg.norm(
                candidate[:, :3] @ projection_array.T,
                axis=1,
            )

        add_modality(
            flow_name,
            "flow",
            flow_score(candidate[:, :3]),
            [
                flow_score(_block_permutation(candidate[:, :3], rng))
                for _ in range(null_repeats)
            ],
            informative=signals.flow_informative,
            robot_mask_motion_coverage=signals.robot_mask_motion_coverage,
            valid_depth_coverage=signals.valid_depth_coverage,
            support_count=signals.flow_support_count,
            support_fraction=signals.flow_support_fraction,
            minimum_support_count=signals.flow_support_threshold,
            fallback_used=signals.flow_fallback_used,
        )
        observed_motion = np.linalg.norm(flow, axis=1)
        add_modality(
            "first_motion_lag",
            "first_motion",
            negative_first_motion_lag_score(
                candidate_motion,
                observed_motion,
                max_lag=2,
            ),
            [
                negative_first_motion_lag_score(
                    _circular_shift(candidate_motion, rng),
                    observed_motion,
                    max_lag=2,
                )
                for _ in range(null_repeats)
            ],
            informative=signals.flow_informative,
        )
        add_modality(
            depth_name,
            "geometry",
            depth_score(candidate[:, :3]),
            [
                depth_score(_block_permutation(candidate[:, :3], rng))
                for _ in range(null_repeats)
            ],
            informative=signals.geometry_informative,
            robot_mask_motion_coverage=signals.robot_mask_motion_coverage,
            valid_depth_coverage=signals.valid_depth_coverage,
            support_count=signals.geometry_support_count,
            support_fraction=signals.geometry_support_fraction,
            minimum_support_count=signals.geometry_support_threshold,
            fallback_used=signals.geometry_fallback_used,
        )
        if grip is not None:
            grip_metrics = grip_transition_metrics(candidate[:, 6], grip)
            add_modality(
                "grip_transition_f1",
                "gripper",
                grip_metrics.macro_f1,
                [
                    _grip_transition_f1(
                        _circular_shift(candidate[:, 6], rng),
                        grip,
                    )
                    for _ in range(null_repeats)
                ],
                informative=grip_metrics.informative,
            )
        rows.append(
            ClipOffsetEvidence(
                clip_id=str(clip_id),
                offset=offset,
                modalities=modalities,
            )
        )
    return rows


def _fit_affine_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim not in (1, 2) or x.shape[0] != y.shape[0]:
        raise ActionAlignmentError(
            f"invalid frozen projection shapes: features={x.shape} targets={y.shape}"
        )
    if x.shape[0] < 32 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ActionAlignmentError(
            "frozen projection requires at least 32 finite samples"
        )
    ridge = float(ridge)
    if not (ridge > 0.0):
        raise ActionAlignmentError("frozen projection ridge must be positive")
    matrix_y = y[:, None] if y.ndim == 1 else y
    design = np.concatenate(
        (x, np.ones((x.shape[0], 1), dtype=np.float64)),
        axis=1,
    )
    regularizer = np.eye(design.shape[1], dtype=np.float64) * ridge
    regularizer[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + regularizer,
        design.T @ matrix_y,
    )
    return coefficients[:-1], coefficients[-1]


def fit_frozen_offset_projections(
    clips: Sequence[ProjectionCalibrationClip],
    *,
    offsets: Iterable[int],
    ridge: float = 1e-3,
    minimum_clips: int = 32,
) -> FrozenProjectionSet:
    offset_values = tuple(int(value) for value in offsets)
    if offset_values != tuple(range(-2, 3)):
        raise ActionAlignmentError("projection offsets must be exactly -2..2")
    ordered = sorted(
        (clip for clip in clips if bool(clip.source_informative)),
        key=lambda clip: str(clip.clip_id),
    )
    clip_ids = tuple(str(clip.clip_id) for clip in ordered)
    if len(clip_ids) < int(minimum_clips) or len(set(clip_ids)) != len(clip_ids):
        raise ActionAlignmentError(
            "frozen projection requires unique independent calibration clips"
        )

    features: dict[int, list[np.ndarray]] = {
        offset: [] for offset in offset_values
    }
    flow_targets: list[np.ndarray] = []
    depth_targets: list[np.ndarray] = []
    for clip in ordered:
        actions = np.asarray(clip.actions, dtype=np.float64)
        target = np.asarray(clip.target_frame_indices, dtype=np.int64)
        flow = np.asarray(clip.flow_vectors, dtype=np.float64)
        depth = np.asarray(clip.depth_delta, dtype=np.float64).reshape(-1)
        if (
            actions.ndim != 2
            or actions.shape[1] < 7
            or target.ndim != 1
            or target.size < 8
            or np.any(np.diff(target) != 1)
            or flow.shape != (target.size, 2)
            or depth.shape != (target.size,)
        ):
            raise ActionAlignmentError(
                f"invalid calibration clip geometry: {clip.clip_id}"
            )
        if not (
            np.isfinite(actions).all()
            and np.isfinite(flow).all()
            and np.isfinite(depth).all()
        ):
            raise ActionAlignmentError(
                f"non-finite calibration clip: {clip.clip_id}"
            )
        action_indices_by_offset = _validated_action_frame_indices(
            clip.action_frame_indices_by_offset,
            offsets=offset_values,
            target_size=int(target.size),
            action_count=int(actions.shape[0]),
            label=f"calibration action_frame_indices_by_offset for {clip.clip_id}",
        )
        for offset in offset_values:
            features[offset].append(actions[action_indices_by_offset[offset], :3])
        flow_targets.append(flow)
        depth_targets.append(depth)

    stacked_flow = np.concatenate(flow_targets, axis=0)
    stacked_depth = np.concatenate(depth_targets, axis=0)
    by_offset: dict[int, FrozenOffsetProjection] = {}
    for offset in offset_values:
        stacked_action = np.concatenate(features[offset], axis=0)
        flow_weights, flow_bias = _fit_affine_ridge(
            stacked_action,
            stacked_flow,
            ridge=ridge,
        )
        depth_weights, depth_bias = _fit_affine_ridge(
            stacked_action,
            stacked_depth,
            ridge=ridge,
        )
        by_offset[offset] = FrozenOffsetProjection(
            offset=offset,
            flow_weights=tuple(
                tuple(float(value) for value in row)
                for row in flow_weights
            ),
            flow_bias=tuple(float(value) for value in flow_bias.reshape(-1)),
            depth_weights=tuple(
                float(value) for value in depth_weights.reshape(-1)
            ),
            depth_bias=float(depth_bias.reshape(-1)[0]),
        )
    return FrozenProjectionSet(
        calibration_clip_ids=clip_ids,
        by_offset=by_offset,
        ridge=float(ridge),
    )


def score_frozen_offset_projection(
    action_xyz: np.ndarray,
    flow_xy: np.ndarray,
    depth_delta: np.ndarray,
    projection: FrozenOffsetProjection,
) -> tuple[float, float]:
    action = np.asarray(action_xyz, dtype=np.float64)
    flow = np.asarray(flow_xy, dtype=np.float64)
    depth = np.asarray(depth_delta, dtype=np.float64).reshape(-1)
    flow_weights = np.asarray(projection.flow_weights, dtype=np.float64)
    flow_bias = np.asarray(projection.flow_bias, dtype=np.float64)
    depth_weights = np.asarray(projection.depth_weights, dtype=np.float64)
    if (
        action.ndim != 2
        or action.shape[1] != 3
        or flow.shape != (action.shape[0], 2)
        or depth.shape != (action.shape[0],)
        or flow_weights.shape != (3, 2)
        or flow_bias.shape != (2,)
        or depth_weights.shape != (3,)
    ):
        raise ActionAlignmentError("invalid frozen projection scoring shapes")
    if not (
        np.isfinite(action).all()
        and np.isfinite(flow).all()
        and np.isfinite(depth).all()
    ):
        raise ActionAlignmentError("non-finite frozen projection score inputs")
    predicted_flow = action @ flow_weights + flow_bias
    predicted_depth = action @ depth_weights + float(projection.depth_bias)
    return (
        _spearman_correlation(predicted_flow.reshape(-1), flow.reshape(-1)),
        fisher_z_correlation(predicted_depth, depth),
    )
