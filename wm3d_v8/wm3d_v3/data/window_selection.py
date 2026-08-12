"""Select native WM3D windows only from recorded source timestamps.

The selector may thin a high-rate observation stream to cover a configured
physical horizon, but it never interpolates a state or rewrites its time.  It
also returns one observed leading boundary, which is required to assign all T
history-action intervals without overlapping the K future intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class WindowSelectionError(ValueError):
    pass


@dataclass(frozen=True)
class ObservedWorldWindow:
    leading_boundary_index: int
    context_indices: np.ndarray
    future_indices: np.ndarray
    world_indices: np.ndarray
    action_boundary_indices: np.ndarray
    world_times_s: np.ndarray
    action_boundaries_s: np.ndarray


def _clock(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size < 3:
        raise WindowSelectionError("world-state clock must contain at least 3 samples")
    if not np.isfinite(result).all() or np.any(np.diff(result) <= 0):
        raise WindowSelectionError(
            "world-state timestamps must be finite and strictly increasing"
        )
    return result


def _nearest_monotonic(
    clock: np.ndarray,
    candidates: np.ndarray,
    targets: np.ndarray,
    *,
    pin_last: int | None = None,
) -> np.ndarray:
    """Choose distinct ordered observed rows nearest to physical targets."""

    count = int(targets.size)
    if candidates.size < count:
        raise WindowSelectionError(
            f"only {candidates.size} observed states for {count} requested samples"
        )
    output = np.empty(count, dtype=np.int64)
    lower_position = 0
    for slot, target in enumerate(targets):
        remaining = count - slot - 1
        upper_position = candidates.size - remaining - 1
        if slot == count - 1 and pin_last is not None:
            positions = np.flatnonzero(candidates == pin_last)
            if positions.size != 1:
                raise WindowSelectionError("pinned observed state is not a candidate")
            position = int(positions[0])
            if not lower_position <= position <= upper_position:
                raise WindowSelectionError("pinned observed state violates ordering")
        else:
            region = candidates[lower_position : upper_position + 1]
            distances = np.abs(clock[region] - target)
            position = lower_position + int(np.argmin(distances))
        output[slot] = candidates[position]
        lower_position = position + 1
    if np.any(np.diff(output) <= 0):
        raise AssertionError("internal selector produced non-monotonic indices")
    return output


def select_observed_world_window(
    timestamps_s: Sequence[float],
    *,
    anchor_index: int,
    context_samples: int,
    future_samples: int,
    context_horizon_s: float,
    future_horizon_s: float,
    minimum_horizon_coverage: float,
    future_offsets_s: Sequence[float] | None = None,
) -> ObservedWorldWindow:
    """Select T context + K future states and a real leading boundary.

    ``anchor_index`` is the final context state and the policy-chunk start.  It
    must be an actual source row.  All returned indices point into the input
    clock; there is no synthetic time or nearest-state fallback at runtime.
    """

    clock = _clock(timestamps_s)
    anchor_index = int(anchor_index)
    context_samples = int(context_samples)
    future_samples = int(future_samples)
    if not 0 <= anchor_index < clock.size:
        raise WindowSelectionError("anchor_index is outside the observed clock")
    if context_samples < 1 or future_samples < 1:
        raise WindowSelectionError("context/future sample counts must be positive")
    if context_horizon_s <= 0 or future_horizon_s <= 0:
        raise WindowSelectionError("context/future horizons must be positive")
    if not 0 < minimum_horizon_coverage <= 1:
        raise WindowSelectionError("minimum_horizon_coverage must be in (0,1]")

    anchor = float(clock[anchor_index])
    context_start = anchor - float(context_horizon_s)
    context_candidates = np.flatnonzero(
        (clock >= context_start) & (np.arange(clock.size) <= anchor_index)
    )
    context_targets = np.linspace(
        context_start, anchor, context_samples, dtype=np.float64
    )
    context = _nearest_monotonic(
        clock, context_candidates, context_targets, pin_last=anchor_index
    )
    leading = int(context[0]) - 1
    if leading < 0:
        raise WindowSelectionError(
            "no observed leading boundary before the first context state"
        )
    context_coverage = anchor - float(clock[context[0]])
    if context_samples > 1 and context_coverage + 1.0e-12 < (
        context_horizon_s * minimum_horizon_coverage
    ):
        raise WindowSelectionError(
            f"context covers only {context_coverage:.9f}s of "
            f"{context_horizon_s:.9f}s requested horizon"
        )

    if future_offsets_s is None:
        offsets = np.linspace(
            future_horizon_s / future_samples,
            future_horizon_s,
            future_samples,
            dtype=np.float64,
        )
    else:
        offsets = np.asarray(future_offsets_s, dtype=np.float64)
        if (
            offsets.shape != (future_samples,)
            or not np.isfinite(offsets).all()
            or np.any(offsets <= 0)
            or np.any(np.diff(offsets) <= 0)
        ):
            raise WindowSelectionError(
                "future_offsets_s must contain K finite, positive, increasing offsets"
            )
        if not np.isclose(
            offsets[-1], future_horizon_s, rtol=0.0, atol=1.0e-12
        ):
            raise WindowSelectionError(
                "last future offset must equal future_horizon_s"
            )
    future_stop = anchor + float(offsets[-1])
    future_candidates = np.flatnonzero(
        (np.arange(clock.size) > anchor_index) & (clock <= future_stop)
    )
    future_targets = anchor + offsets
    future = _nearest_monotonic(clock, future_candidates, future_targets)
    future_coverage = float(clock[future[-1]]) - anchor
    if future_coverage + 1.0e-12 < (
        future_horizon_s * minimum_horizon_coverage
    ):
        raise WindowSelectionError(
            f"future covers only {future_coverage:.9f}s of "
            f"{future_horizon_s:.9f}s requested horizon"
        )

    world = np.concatenate((context, future))
    boundaries = np.concatenate((np.asarray([leading], dtype=np.int64), world))
    return ObservedWorldWindow(
        leading_boundary_index=leading,
        context_indices=context,
        future_indices=future,
        world_indices=world,
        action_boundary_indices=boundaries,
        world_times_s=clock[world].copy(),
        action_boundaries_s=clock[boundaries].copy(),
    )
