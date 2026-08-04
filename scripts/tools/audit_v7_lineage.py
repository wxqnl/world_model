#!/usr/bin/env python3
"""核验当前 5B 预设与清理前 V7 native 实现的可追溯关系。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wm3d.models.wm3d import WM3D, config_from_mapping  # noqa: E402
from wm3d.training.runtime import assert_dependency_boundary  # noqa: E402


SOURCE_REF = "7241146891a61225a1c38947c57193967a9c11e9"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source(repo: Path, path: str) -> str:
    return _git(repo, "show", f"{SOURCE_REF}:{path}") + "\n"


def _identity(text: str) -> str:
    return text


def _normalize_model(text: str) -> str:
    return (
        text.replace("Native WM3D 5B core.", "WM3D core.")
        .replace("Native5BConfig", "WM3DConfig")
        .replace("NativeWM3D5B", "WM3D")
    )


def _normalize_features(text: str) -> str:
    return (
        text.replace("native WM3D-V7 5B", "WM3D")
        .replace("Native5BVGGTEncoder", "VGGTFeatureEncoder")
        .replace("native5b", "wm3d")
    )


def _normalize_action(text: str) -> str:
    return text.replace("native WM3D-V7 5B", "WM3D").replace(
        ".scale5b_contracts", ".contracts"
    )


def _normalize_loss(text: str) -> str:
    return (
        text.replace("WM3D-V7 5B", "WM3D")
        .replace("Native5BLossConfig", "WM3DLossConfig")
        .replace("native5b_loss", "wm3d_loss")
        .replace("native5b", "wm3d")
    )


LINEAGE_FILES: tuple[tuple[str, str, str, Callable[[str], str]], ...] = (
    (
        "wm3d_v7/wm3d_v3/models/native5b.py",
        "wm3d/models/wm3d.py",
        "225b34e9ef65895398ceb97e1c57ba164929cb77",
        _normalize_model,
    ),
    (
        "wm3d_v7/wm3d_v3/encoders/native5b_vggt.py",
        "wm3d/encoders/vggt_features.py",
        "096a024a2272b83fadca9b361802a1c778ac5b43",
        _normalize_features,
    ),
    (
        "wm3d_v7/wm3d_v3/encoders/vggt_encoder.py",
        "wm3d/encoders/vggt_encoder.py",
        "bf7e14ea32521fce73df5965417e49b820c0f5f1",
        _identity,
    ),
    (
        "wm3d_v7/wm3d_v3/data/scale5b_action.py",
        "wm3d/data/action.py",
        "684c64ade66bb94767df5d73c6d5a4586d8f9f66",
        _normalize_action,
    ),
    (
        "wm3d_v7/wm3d_v3/training/scale5b_loss.py",
        "wm3d/training/loss.py",
        "62927c415b1084a024003068e00c7912a7fcb2c1",
        _normalize_loss,
    ),
)


def audit(repo: Path, config_path: Path) -> dict[str, object]:
    repo = repo.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    if _git(repo, "rev-parse", "--is-inside-work-tree") != "true":
        raise RuntimeError("V7 血统审计要求从完整 Git clone 运行")
    _git(repo, "merge-base", "--is-ancestor", SOURCE_REF, "HEAD")

    files: dict[str, object] = {}
    for source_path, current_path, expected_blob, normalize in LINEAGE_FILES:
        actual_blob = _git(repo, "rev-parse", f"{SOURCE_REF}:{source_path}")
        if actual_blob != expected_blob:
            raise RuntimeError(
                f"V7 anchor blob 漂移：{source_path} {actual_blob} != {expected_blob}"
            )
        source_text = _source(repo, source_path)
        current_text = (repo / current_path).read_text(encoding="utf-8")
        normalized = normalize(source_text)
        if normalized != current_text:
            raise RuntimeError(f"当前实现偏离 V7 anchor：{current_path}")
        files[current_path] = {
            "anchor_path": source_path,
            "anchor_blob_sha1": actual_blob,
            "current_sha256": _sha256(current_text),
            "exact_after_declared_rename": True,
        }

    source_config_path = (
        "wm3d_v7/configs/scale5b/wm3d_v7_native5b_h200.template.yaml"
    )
    source_config_blob = _git(
        repo, "rev-parse", f"{SOURCE_REF}:{source_config_path}"
    )
    if source_config_blob != "d68715fe60db0771ac9e397af2d6548d83774a47":
        raise RuntimeError("V7 5B 配置 anchor blob 漂移")
    source_config = yaml.safe_load(_source(repo, source_config_path))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "wm3d_v7_pretrain_config_v1":
        raise RuntimeError("训练配置不是 V7 schema")
    inherited_sections = (
        "model",
        "data",
        "distributed",
        "optimizer",
        "schedule",
        "train",
        "loss",
    )
    for section in inherited_sections:
        if config.get(section) != source_config.get(section):
            raise RuntimeError(f"5B 配置的 V7 anchor 段发生漂移：{section}")

    boundary_paths = [
        path
        for base in (repo / "wm3d", repo / "scripts", repo / "environments")
        for path in base.rglob("*.py")
    ]
    boundary_paths.extend((repo / "configs").rglob("*.yaml"))
    boundary_paths.extend((repo / "configs").rglob("*.json"))
    assert_dependency_boundary(boundary_paths)

    with torch.device("meta"):
        model = WM3D(config_from_mapping(config["model"]))
    counts = model.parameter_counts()
    expected_parameters = int(config["model_budget"]["expected_parameters"])
    if counts["total"] != expected_parameters:
        raise RuntimeError(
            f"参数预算漂移：{counts['total']} != {expected_parameters}"
        )
    owners = {
        "native_world_state": hasattr(model, "state_blocks"),
        "grouped_action": hasattr(model, "action_blocks")
        and hasattr(model, "action_head"),
        "bidirectional_bridges": hasattr(model, "bridges"),
        "explicit_rgb": hasattr(model, "rgb_head"),
        "explicit_depth_point_camera": hasattr(model, "geometry_head"),
    }
    if not all(owners.values()):
        raise RuntimeError(f"V7 native ownership contract 不完整：{owners}")

    return {
        "schema": "wm3d_v7_lineage_audit_v1",
        "pass": True,
        "source_ref": SOURCE_REF,
        "files": files,
        "config": {
            "path": config_path.relative_to(repo).as_posix(),
            "anchor_path": source_config_path,
            "anchor_blob_sha1": source_config_blob,
            "inherited_sections_exact": list(inherited_sections),
        },
        "dependency_boundary": "pass",
        "owners": owners,
        "parameters": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    config = args.config or args.repo_root / "configs/train/5b_h200.yaml"
    print(json.dumps(audit(args.repo_root, config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
