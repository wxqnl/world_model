"""Reversible packing of audited native action columns into fixed group slots.

Only recorded constants may be implicit. The reader rechecks them on every
episode; changing a declared constant is an error, never a clipped/dropped
control. All nonconstant values keep their native numeric representation.
"""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import numpy as np


class ActionLayoutError(ValueError):
    pass


def validate_action_layout(value: Mapping[str, Any], groups: Sequence[Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"key", "width", "constant_columns"}:
        raise ActionLayoutError("raw action layout fields mismatch")
    if not isinstance(value["key"], str) or not value["key"]:
        raise ActionLayoutError("raw action layout needs a field key")
    width = value["width"]
    if type(width) is not int or width <= 0:
        raise ActionLayoutError("raw action layout width must be positive")
    constants = value["constant_columns"]
    if not isinstance(constants, dict):
        raise ActionLayoutError("constant columns must be an explicit mapping")
    constant_ids = []
    for key, number in constants.items():
        if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key:
            raise ActionLayoutError("constant column IDs must be canonical integers")
        index = int(key)
        if not 0 <= index < width or isinstance(number, bool) or not np.isfinite(number):
            raise ActionLayoutError("constant action column/value is invalid")
        constant_ids.append(index)
    columns = []
    for group in groups:
        if group.action_transform != "identity" or group.action_alignment != "as_recorded":
            raise ActionLayoutError("lossless layout must preserve action values and row timing")
        if group.action_value_mask is not None and not all(group.action_value_mask):
            raise ActionLayoutError("lossless layout cannot mask a native action dimension")
        for term in group.action:
            if term.key != value["key"] or any(x != 1 for x in term.scale) or any(x != 0 for x in term.offset):
                raise ActionLayoutError("lossless layout permits column placement only")
            columns.extend(term.columns)
    if (len(set(columns)) != len(columns) or set(columns) & set(constant_ids)
            or set(columns) | set(constant_ids) != set(range(width))):
        raise ActionLayoutError("every native action column must appear exactly once or be a declared constant")
    return {"key": value["key"], "width": width, "constant_columns": dict(constants)}


def check_action_constants(accessor: Any, layout: Mapping[str, Any]) -> np.ndarray:
    value = np.asarray(accessor.array(layout["key"]))
    if value.ndim != 2 or value.shape[1] != layout["width"] or not np.isfinite(value).all():
        raise ActionLayoutError("native action field width/finite contract changed")
    for key, number in layout["constant_columns"].items():
        if not np.all(value[:, int(key)] == number):
            raise ActionLayoutError(f"declared constant action column {key} changed")
    return value


def reconstruct_native_actions(
    grouped_values: Mapping[str, np.ndarray], *, layout: Mapping[str, Any], groups: Sequence[Any]
) -> np.ndarray:
    """Inverse on unnormalized values; normalization is inverted beforehand."""
    layout = validate_action_layout(layout, groups)
    if set(grouped_values) != {g.group for g in groups}:
        raise ActionLayoutError("reconstruction group membership differs")
    arrays = {name: np.asarray(v) for name, v in grouped_values.items()}
    shapes = {v.shape[:-1] for v in arrays.values()}
    if len(shapes) != 1 or not all(v.ndim >= 1 and np.isfinite(v).all() for v in arrays.values()):
        raise ActionLayoutError("reconstruction shapes or values are invalid")
    shape = next(iter(shapes))
    dtype = np.result_type(*(x.dtype for x in arrays.values()))
    out = np.empty((*shape, layout["width"]), dtype=dtype)
    for key, number in layout["constant_columns"].items():
        out[..., int(key)] = number
    for group in groups:
        columns = tuple(c for term in group.action for c in term.columns)
        value = arrays[group.group]
        if value.shape[-1] != len(columns):
            raise ActionLayoutError("reconstruction group width differs")
        out[..., columns] = value
    return out
