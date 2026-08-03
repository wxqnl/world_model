"""Embodiment-aware high-rate action alignment for native WM3D-V7 5B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from .scale5b_contracts import ContractError, EmbodimentSpec


@dataclass(frozen=True)
class RawActionSeries:
    timestamps: np.ndarray
    values: np.ndarray
    valid: np.ndarray | None = None

    def validate(self) -> None:
        if self.timestamps.ndim != 1:
            raise ContractError("action timestamps must be one-dimensional")
        if self.values.ndim != 2 or self.values.shape[0] != self.timestamps.shape[0]:
            raise ContractError("action values must be [N,D] aligned to timestamps")
        if self.timestamps.size < 2:
            raise ContractError("action series needs at least two samples")
        if not np.isfinite(self.timestamps).all():
            raise ContractError("action timestamps contain non-finite values")
        if not np.all(np.diff(self.timestamps) > 0):
            raise ContractError("action timestamps must be strictly increasing")
        if self.valid is not None:
            if self.valid.shape != self.values.shape:
                raise ContractError("action valid mask shape mismatch")
            if not np.isfinite(self.values[self.valid]).all():
                raise ContractError("valid action entries contain NaN or Inf")
        elif not np.isfinite(self.values).all():
            raise ContractError("action series contains non-finite values")


@dataclass(frozen=True)
class ActionNormalization:
    center: np.ndarray
    scale: np.ndarray
    clip: float = 5.0

    def validate(self, dimension: int) -> None:
        if self.center.shape != (dimension,) or self.scale.shape != (dimension,):
            raise ContractError("action normalization shape mismatch")
        if not np.isfinite(self.center).all() or not np.isfinite(self.scale).all():
            raise ContractError("action normalization is non-finite")
        if np.any(self.scale <= 0) or float(self.clip) <= 0:
            raise ContractError("action normalization scale/clip must be positive")


def robust_action_normalization(
    values: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
) -> ActionNormalization:
    if values.ndim != 2:
        raise ContractError("normalization values must be [N,D]")
    if valid is None:
        valid = np.isfinite(values)
    if valid.shape != values.shape:
        raise ContractError("normalization valid mask shape mismatch")
    center = np.zeros(values.shape[1], dtype=np.float64)
    scale = np.ones(values.shape[1], dtype=np.float64)
    for dimension in range(values.shape[1]):
        selected = values[valid[:, dimension], dimension]
        selected = selected[np.isfinite(selected)]
        if selected.size < 32:
            raise ContractError(
                f"action dimension {dimension} has only {selected.size} valid samples"
            )
        median = np.median(selected)
        low, high = np.quantile(selected, [quantile_low, quantile_high])
        center[dimension] = median
        scale[dimension] = max((high - low) / 2.0, 1.0e-6)
    return ActionNormalization(center=center, scale=scale)


def _interpolate_series(
    series: RawActionSeries,
    query: np.ndarray,
    *,
    discrete: bool,
) -> tuple[np.ndarray, np.ndarray]:
    series.validate()
    inside = (query >= series.timestamps[0]) & (query <= series.timestamps[-1])
    output = np.zeros((query.size, series.values.shape[1]), dtype=np.float32)
    valid = np.zeros_like(output, dtype=bool)
    if discrete:
        right = np.searchsorted(series.timestamps, query, side="left")
        right = np.clip(right, 0, series.timestamps.size - 1)
        left = np.clip(right - 1, 0, series.timestamps.size - 1)
        choose_right = np.abs(series.timestamps[right] - query) < np.abs(
            series.timestamps[left] - query
        )
        indices = np.where(choose_right, right, left)
        output[:] = series.values[indices]
        if series.valid is None:
            valid[:] = inside[:, None]
        else:
            valid[:] = series.valid[indices] & inside[:, None]
    else:
        for dimension in range(series.values.shape[1]):
            output[:, dimension] = np.interp(
                query,
                series.timestamps,
                series.values[:, dimension],
            )
        if series.valid is None:
            valid[:] = inside[:, None]
        else:
            right = np.searchsorted(series.timestamps, query, side="right")
            right = np.clip(right, 1, series.timestamps.size - 1)
            left = right - 1
            valid[:] = series.valid[left] & series.valid[right] & inside[:, None]
    return output, valid


def align_grouped_actions(
    *,
    visual_timestamps: np.ndarray,
    group_series: Mapping[str, RawActionSeries],
    embodiment: EmbodimentSpec,
    normalizations: Mapping[str, ActionNormalization],
    max_groups: int,
    max_action_dim: int,
    action_substeps: int,
    feature_fps: float = 5.0,
) -> dict[str, torch.Tensor]:
    """Align 15-30 Hz actions into S substeps for every 5 Hz visual frame."""

    if visual_timestamps.ndim != 1 or visual_timestamps.size == 0:
        raise ContractError("visual timestamps must be a non-empty vector")
    if not np.all(np.diff(visual_timestamps) > 0):
        raise ContractError("visual timestamps must be strictly increasing")
    if len(embodiment.action_groups) > max_groups:
        raise ContractError("embodiment exceeds max action groups")
    substep_offsets = np.arange(action_substeps, dtype=np.float64) / (
        feature_fps * action_substeps
    )
    query = (visual_timestamps[:, None] + substep_offsets[None]).reshape(-1)
    values = np.zeros(
        (
            visual_timestamps.size,
            max_groups,
            action_substeps,
            max_action_dim,
        ),
        dtype=np.float32,
    )
    dim_mask = np.zeros_like(values, dtype=bool)
    contact = np.zeros(
        (visual_timestamps.size, max_groups, action_substeps),
        dtype=np.float32,
    )
    contact_mask = np.zeros_like(contact, dtype=bool)
    group_ids = np.zeros(max_groups, dtype=np.int64)
    group_mask = np.zeros(max_groups, dtype=bool)

    for slot, group in enumerate(embodiment.action_groups):
        if group.name not in group_series:
            raise ContractError(f"missing raw action group {group.name}")
        if group.name not in normalizations:
            raise ContractError(f"missing normalization for group {group.name}")
        dimension = len(group.dimensions)
        normalization = normalizations[group.name]
        normalization.validate(dimension)
        discrete = "grip" in group.control_mode or "discrete" in group.control_mode
        aligned, valid = _interpolate_series(
            group_series[group.name], query, discrete=discrete
        )
        normalized = (aligned - normalization.center) / normalization.scale
        normalized = np.clip(normalized, -normalization.clip, normalization.clip)
        normalized = normalized.reshape(
            visual_timestamps.size, action_substeps, dimension
        )
        valid = valid.reshape(visual_timestamps.size, action_substeps, dimension)
        values[:, slot, :, :dimension] = normalized
        dim_mask[:, slot, :, :dimension] = valid
        group_ids[slot] = int(group.group_id)
        group_mask[slot] = True
        if discrete:
            contact[:, slot] = (normalized[..., -1] > 0.0).astype(np.float32)
            contact_mask[:, slot] = valid[..., -1]

    return {
        "action_values": torch.from_numpy(values),
        "action_dim_mask": torch.from_numpy(dim_mask),
        "contact": torch.from_numpy(contact),
        "contact_mask": torch.from_numpy(contact_mask),
        "action_group_ids": torch.from_numpy(group_ids),
        "action_group_mask": torch.from_numpy(group_mask),
    }


def align_auxiliary_tokens(
    *,
    visual_timestamps: np.ndarray,
    modality_series: Mapping[str, RawActionSeries],
    embodiment: EmbodimentSpec,
    normalizations: Mapping[str, ActionNormalization],
    max_aux_tokens: int,
    aux_dim: int,
    max_aux_type_id: int,
) -> dict[str, torch.Tensor]:
    """Pack heterogeneous context sensors into deterministic D256 tokens.

    The first ``max_aux_type_id`` channels are a one-hot modality identity.
    Each modality's normalized values and per-dimension validity bits follow.
    Only values at visual timestamps are emitted; the high-rate raw signal is
    never allowed to expose future context to the world model.
    """

    if visual_timestamps.ndim != 1 or visual_timestamps.size == 0:
        raise ContractError("auxiliary visual timestamps must be non-empty")
    if not np.all(np.diff(visual_timestamps) > 0):
        raise ContractError("auxiliary visual timestamps must be increasing")
    if len(embodiment.auxiliary_modalities) > int(max_aux_tokens):
        raise ContractError("embodiment exceeds max auxiliary tokens")
    if int(max_aux_type_id) >= int(aux_dim):
        raise ContractError("auxiliary type region consumes the whole token")

    tokens = np.zeros(
        (visual_timestamps.size, max_aux_tokens, aux_dim),
        dtype=np.float32,
    )
    mask = np.zeros(
        (visual_timestamps.size, max_aux_tokens),
        dtype=bool,
    )
    for slot, modality in enumerate(embodiment.auxiliary_modalities):
        if modality.name not in modality_series:
            raise ContractError(f"missing raw auxiliary modality {modality.name}")
        if modality.name not in normalizations:
            raise ContractError(
                f"missing normalization for auxiliary modality {modality.name}"
            )
        dimension = len(modality.dimensions)
        if int(max_aux_type_id) + 2 * dimension > int(aux_dim):
            raise ContractError(
                f"auxiliary modality {modality.name} does not fit D{aux_dim}"
            )
        normalization = normalizations[modality.name]
        normalization.validate(dimension)
        aligned, valid = _interpolate_series(
            modality_series[modality.name],
            visual_timestamps,
            discrete=bool(modality.discrete),
        )
        normalized = (aligned - normalization.center) / normalization.scale
        normalized = np.clip(
            normalized,
            -normalization.clip,
            normalization.clip,
        )
        normalized = np.where(valid, normalized, 0.0)
        tokens[:, slot, int(modality.type_id)] = 1.0
        start = int(max_aux_type_id)
        tokens[:, slot, start : start + dimension] = normalized
        tokens[:, slot, start + dimension : start + 2 * dimension] = valid.astype(
            np.float32
        )
        mask[:, slot] = valid.any(axis=-1)

    return {
        "aux_tokens": torch.from_numpy(tokens),
        "aux_mask": torch.from_numpy(mask),
    }
