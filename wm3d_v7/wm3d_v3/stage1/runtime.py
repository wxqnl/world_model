"""Stage1 update cadence, serialized rolling execution, and Wan checkpointing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import wraps
from types import MethodType
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint


PHASE_LENGTH = 32
FORMAL_ROLLING_START_UPDATE = 512
GENERATED_CONTEXT_START_PROBABILITY = 0.25
GENERATED_CONTEXT_END_PROBABILITY = 0.75
GENERATED_CONTEXT_RAMP_EVENTS = (
    (3072 - FORMAL_ROLLING_START_UPDATE) // PHASE_LENGTH
) + 1


class Stage1Phase(str, Enum):
    """The one global Stage1 expensive-work phase."""

    NORMAL = "normal"
    DECODED_RGB = "decoded_rgb"
    ACTION_PROBE = "action_probe"
    ROLLING = "rolling"


def _require_non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(value: int, *, name: str) -> int:
    value = _require_non_negative_int(value, name=name)
    if value == 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def phase_for_update(
    update_id: int,
    *,
    rolling_start_update: int = FORMAL_ROLLING_START_UPDATE,
) -> Stage1Phase:
    """Return the sole phase for a 1-based update identifier."""

    update_id = _require_positive_int(update_id, name="update_id")
    rolling_start_update = _require_positive_int(
        rolling_start_update,
        name="rolling_start_update",
    )

    if update_id >= rolling_start_update and update_id % PHASE_LENGTH == 0:
        return Stage1Phase.ROLLING
    if update_id % PHASE_LENGTH == 8:
        return Stage1Phase.DECODED_RGB
    if update_id % PHASE_LENGTH == 16:
        return Stage1Phase.ACTION_PROBE
    return Stage1Phase.NORMAL


def _completed_rolling_events(
    completed_steps: int,
    *,
    rolling_start_update: int,
) -> int:
    first_event = (
        (rolling_start_update + PHASE_LENGTH - 1) // PHASE_LENGTH
    ) * PHASE_LENGTH
    if completed_steps < first_event:
        return 0
    return ((completed_steps - first_event) // PHASE_LENGTH) + 1


@dataclass
class RuntimeState:
    """Resume-safe state whose next update is always completed_steps + 1."""

    completed_steps: int = 0
    rolling_events_completed: int | None = None
    rolling_start_update: int = FORMAL_ROLLING_START_UPDATE

    def __post_init__(self) -> None:
        self.completed_steps = _require_non_negative_int(
            self.completed_steps,
            name="completed_steps",
        )
        self.rolling_start_update = _require_positive_int(
            self.rolling_start_update,
            name="rolling_start_update",
        )
        expected_events = _completed_rolling_events(
            self.completed_steps,
            rolling_start_update=self.rolling_start_update,
        )
        if self.rolling_events_completed is None:
            self.rolling_events_completed = expected_events
        else:
            self.rolling_events_completed = _require_non_negative_int(
                self.rolling_events_completed,
                name="rolling_events_completed",
            )
            if self.rolling_events_completed != expected_events:
                raise ValueError(
                    "rolling_events_completed does not match completed_steps "
                    "and rolling_start_update"
                )

    @property
    def update_id(self) -> int:
        return self.completed_steps + 1

    @property
    def phase(self) -> Stage1Phase:
        return phase_for_update(
            self.update_id,
            rolling_start_update=self.rolling_start_update,
        )

    @property
    def rolling_event_number(self) -> int | None:
        if self.phase is not Stage1Phase.ROLLING:
            return None
        assert self.rolling_events_completed is not None
        return self.rolling_events_completed + 1

    @property
    def generated_context_probability(self) -> float | None:
        event_number = self.rolling_event_number
        if event_number is None:
            return None
        return generated_context_probability(event_number)

    def skip_update(self, update_id: int | None = None) -> int:
        """Record no progress; a skipped batch retries the same update ID."""

        if update_id is not None and update_id != self.update_id:
            raise ValueError(
                f"cannot skip update {update_id}; current update is {self.update_id}"
            )
        return self.update_id

    def complete_update(self, update_id: int | None = None) -> int:
        """Commit exactly the current update and return completed_steps."""

        if update_id is not None and update_id != self.update_id:
            raise ValueError(
                f"cannot complete update {update_id}; current update is "
                f"{self.update_id}"
            )
        if self.phase is Stage1Phase.ROLLING:
            assert self.rolling_events_completed is not None
            self.rolling_events_completed += 1
        self.completed_steps += 1
        return self.completed_steps


def generated_context_probability(rolling_event_number: int) -> float:
    """Return the rolling-event schedule from 0.25 to 0.75."""

    event_number = _require_positive_int(
        rolling_event_number,
        name="rolling_event_number",
    )
    if event_number >= GENERATED_CONTEXT_RAMP_EVENTS:
        return GENERATED_CONTEXT_END_PROBABILITY
    progress = (event_number - 1) / (GENERATED_CONTEXT_RAMP_EVENTS - 1)
    return GENERATED_CONTEXT_START_PROBABILITY + progress * (
        GENERATED_CONTEXT_END_PROBABILITY - GENERATED_CONTEXT_START_PROBABILITY
    )


SerializedCallback = Callable[[], Any]


def run_serialized_pair(
    *,
    a_forward: SerializedCallback,
    a_backward: SerializedCallback,
    a_release: SerializedCallback,
    roll_forward: SerializedCallback,
    roll_release: SerializedCallback,
    b_forward: SerializedCallback,
    b_backward: SerializedCallback,
    b_release: SerializedCallback,
    optimizer_step: SerializedCallback,
) -> None:
    """Run paired A/roll/B work without allowing the A and B graphs to coexist."""

    try:
        a_forward()
        a_backward()
    finally:
        a_release()

    try:
        with torch.no_grad():
            roll_forward()
    finally:
        roll_release()

    try:
        b_forward()
        b_backward()
    finally:
        b_release()

    optimizer_step()


_CHECKPOINT_MARKER = "_wm3d_stage1_activation_checkpointed"


def _checkpoint_block_call(block: nn.Module) -> None:
    if bool(getattr(block, _CHECKPOINT_MARKER, False)):
        return

    original_call_impl = block._call_impl

    @wraps(original_call_impl)
    def checkpointed_call_impl(
        _block: nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not torch.is_grad_enabled():
            return original_call_impl(*args, **kwargs)
        return activation_checkpoint(
            original_call_impl,
            *args,
            use_reentrant=False,
            **kwargs,
        )

    block._call_impl = MethodType(checkpointed_call_impl, block)
    setattr(block, _CHECKPOINT_MARKER, True)


def apply_wan_activation_checkpointing(transformer: nn.Module) -> int:
    """Checkpoint every Wan block forward before the transformer is FSDP-wrapped.

    Blocks are modified in place so their identity and injector-facing attributes
    remain unchanged. The non-reentrant checkpoint call re-executes each real
    block forward during backward.
    """

    if hasattr(transformer, "_fsdp_wrapped_module"):
        raise RuntimeError(
            "Wan activation checkpointing must be applied before FSDP wrapping"
        )

    blocks = getattr(transformer, "blocks", None)
    if blocks is None:
        raise RuntimeError(
            "Could not locate Wan transformer.blocks for activation checkpointing"
        )
    try:
        block_list = tuple(blocks)
    except TypeError as exc:
        raise RuntimeError("Wan transformer.blocks must be iterable") from exc
    if not block_list or not all(isinstance(block, nn.Module) for block in block_list):
        raise RuntimeError(
            "Wan transformer.blocks must contain at least one torch module"
        )

    for block in block_list:
        _checkpoint_block_call(block)

    setattr(transformer, "_wm3d_activation_checkpoint", True)
    setattr(transformer, "_wm3d_activation_checkpoint_use_reentrant", False)
    return len(block_list)


__all__ = [
    "FORMAL_ROLLING_START_UPDATE",
    "GENERATED_CONTEXT_END_PROBABILITY",
    "GENERATED_CONTEXT_RAMP_EVENTS",
    "GENERATED_CONTEXT_START_PROBABILITY",
    "PHASE_LENGTH",
    "RuntimeState",
    "Stage1Phase",
    "apply_wan_activation_checkpointing",
    "generated_context_probability",
    "phase_for_update",
    "run_serialized_pair",
]
