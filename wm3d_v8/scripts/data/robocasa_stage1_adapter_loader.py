#!/usr/bin/env python3
"""Minimal sealed action-adapter loader for the RoboCasa Stage1 replay.

This module intentionally has no model, encoder, cache, or training imports.  The
replay authority snapshots it together with the canonical V7 action dataclass so
the legacy simulator producer cannot import ambient project code.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from dataclasses import dataclass


@dataclass(frozen=True)
class ActionAdapter:
    """Duck-compatible immutable subset consumed by the pinned V7 functions."""

    source: str
    source_frame: str
    translation_unit_scale: float | tuple[float, float, float]
    rotation_unit_scale: float | tuple[float, float, float]
    rotation_repr: str
    gripper_index: int = 6
    gripper_open_value: float = -1.0
    gripper_closed_value: float = 1.0
    is_delta: bool = True
    nominal_hz: float = 5.0
    base_from_source_rotation: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    adapter_version: str = "wm3d_v7_base_delta_axisangle_gripclose_v1"


def _stable_json(path: Path) -> dict:
    raw = Path(path)
    if raw.is_symlink():
        raise RuntimeError("action audit must not be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(raw, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("action audit must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        def identity(value: os.stat_result) -> tuple[int, int, int, int]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
            )
        if identity(before) != identity(after):
            raise RuntimeError("action audit changed while being read")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(b"".join(chunks))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("action audit is not valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError("action audit must be a JSON object")
    return value


def _load_adapter(
    audit_path: Path, *, allow_legacy_proof_audit: bool
) -> ActionAdapter:
    payload = _stable_json(audit_path)
    factual = payload.get("factual_action_audit")
    legacy = payload.get("audit")
    factual_passed = isinstance(factual, dict) and factual.get("passed") is True
    legacy_passed = isinstance(legacy, dict) and legacy.get("passed") is True
    if not factual_passed and not (allow_legacy_proof_audit and legacy_passed):
        raise RuntimeError(
            "formal replay requires factual_action_audit.passed=true; "
            "counterfactual replay is audited independently"
        )
    adapter = payload.get("adapter")
    if not isinstance(adapter, dict):
        raise RuntimeError("action audit lacks an adapter object")
    try:
        return ActionAdapter(**adapter)
    except (TypeError, ValueError) as error:
        raise RuntimeError("action audit adapter contract is invalid") from error
