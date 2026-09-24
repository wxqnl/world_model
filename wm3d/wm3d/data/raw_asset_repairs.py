"""Explicit, checkpoint-bound RGB asset repairs over an immutable data seal.

The original asset remains available and is still verified. Only listed RGB
videos may read an independently verified, frame-aligned upstream replacement.
This is a data revision, not an execution-only hotfix or a hash bypass.
"""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

ENVIRONMENT = "WM3D_RAW_ASSET_REPAIRS"
SCHEMA = "wm3d_raw_rgb_asset_repairs_v1"


@lru_cache(maxsize=4)
def _read_contract(filename: str) -> dict[str, Any]:
    path = Path(filename)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("raw RGB repair contract must be an absolute regular file")
    value = json.loads(path.read_text())
    required = {"schema", "repair_id", "upstream", "user_approved",
                "adoption_run_root", "adoption_checkpoint_step", "assets"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("raw RGB repair contract fields differ")
    if value["schema"] != SCHEMA or value["user_approved"] is not True:
        raise ValueError("raw RGB repair contract is not approved")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]+", value["repair_id"]):
        raise ValueError("raw RGB repair identity is invalid")
    if not str(value["upstream"]).startswith("https://"):
        raise ValueError("raw RGB repair lacks its upstream provenance")
    if not Path(value["adoption_run_root"]).is_absolute():
        raise ValueError("raw RGB repair adoption run must be absolute")
    if type(value["adoption_checkpoint_step"]) is not int or value["adoption_checkpoint_step"] < 0:
        raise ValueError("raw RGB repair adoption step is invalid")
    assets = value["assets"]
    if not isinstance(assets, list) or not assets:
        raise ValueError("raw RGB repair assets are empty")
    seen = set()
    fields = {"source_root", "relative_path", "sealed_digest",
              "replacement_relative_path", "replacement_digest",
              "decoded_frames", "time_base", "timestamps_unchanged"}
    for item in assets:
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError("raw RGB repair asset fields differ")
        relative = Path(item["relative_path"])
        if (not Path(item["source_root"]).is_absolute() or relative.is_absolute()
                or ".." in relative.parts or relative.suffix != ".mp4"
                or not relative.parts or relative.parts[0] != "videos"):
            raise ValueError("raw RGB repair must name a source-local RGB video")
        expected = Path(".wm3d_rgb_repairs") / value["repair_id"] / relative
        if item["replacement_relative_path"] != str(expected):
            raise ValueError("raw RGB replacement escaped its approved namespace")
        identity = (item["source_root"], item["relative_path"])
        if identity in seen:
            raise ValueError("raw RGB repair asset is duplicated")
        seen.add(identity)
        if any(not re.fullmatch(r"[0-9a-f]{64}", str(item[key]))
               for key in ("sealed_digest", "replacement_digest")):
            raise ValueError("raw RGB repair asset integrity binding is invalid")
        if item["sealed_digest"] == item["replacement_digest"]:
            raise ValueError("raw RGB repair must be an explicit data revision")
        if (type(item["decoded_frames"]) is not int or item["decoded_frames"] <= 0
                or not re.fullmatch(r"[1-9][0-9]*/[1-9][0-9]*", item["time_base"])
                or item["timestamps_unchanged"] is not True):
            raise ValueError("raw RGB repair lacks complete frame/PTS verification")
    return value


def active_raw_asset_repairs() -> dict[str, Any] | None:
    filename = os.environ.get(ENVIRONMENT, "").strip()
    return _read_contract(filename) if filename else None


def repair_resume_expectation(run_root: str, step: int) -> Mapping[str, Any] | None:
    repair = active_raw_asset_repairs()
    if repair is None:
        return None
    if Path(run_root) != Path(repair["adoption_run_root"]):
        raise ValueError("raw RGB repair belongs to a different formal run")
    adopted_at = repair["adoption_checkpoint_step"]
    if step < adopted_at:
        raise ValueError("cannot silently apply RGB repair to an earlier checkpoint")
    # Exactly one approved pre-repair checkpoint may have no addendum. All
    # subsequent checkpoints must retain the full, identical repair contract.
    return None if step == adopted_at else repair
