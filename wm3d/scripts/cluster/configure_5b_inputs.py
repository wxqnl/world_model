#!/usr/bin/env python3
"""Discover a local WM3D 5B model/data bundle and write one site file.

The command is intentionally transport-neutral.  It inspects files already on
shared storage; it does not care whether they arrived through ModelScope,
Hugging Face, rsync, or another approved transfer path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
MODEL_PROFILE = ROOT / "configs/model/native_5b_v8_action_owned_transport.yaml"
VGGT_PROFILE = ROOT / "configs/encoder/vggt_native_p144.yaml"
TASK_PROFILE = ROOT / "configs/encoder/task_qwen3_vl_embedding_2b.yaml"
SITE_TEMPLATE = ROOT / "configs/cluster/h200_5b_direct.env.example"
AGIBOT_CHECKER = ROOT / "scripts/data/check_existing_agibot2026.py"
AGIBOT_PREFIXES = (
    "ImitationLearning",
    "RichInteraction",
    "ReinforcementLearning",
)
SKIP_SCAN_DIRECTORIES = {
    ".git",
    "__pycache__",
    "blobs",
    "data",
    "images",
    "videos",
}


def _real_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{label} must be a real directory: {path}")
    return path.resolve(strict=True)


def _profile_value(path: Path, key: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:\s*([^#\s]+)\s*(?:#.*)?$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1)
    raise RuntimeError(f"missing {key} in {path}")


def _walk_directories(root: Path, maximum_depth: int) -> Iterable[Path]:
    base_depth = len(root.parts)
    visited = 0
    for current, directories, _ in os.walk(root, followlinks=False):
        path = Path(current)
        visited += 1
        if visited > 50_000:
            raise RuntimeError(
                f"input scan exceeded 50000 directories; pass a narrower root: {root}"
            )
        depth = len(path.parts) - base_depth
        if depth >= maximum_depth:
            directories[:] = []
        else:
            directories[:] = [
                name
                for name in directories
                if name not in SKIP_SCAN_DIRECTORIES
                and not (path / name).is_symlink()
            ]
        yield path


def _one_candidate(candidates: list[Path], label: str, root: Path) -> Path:
    unique = sorted({path.resolve(strict=True) for path in candidates})
    if not unique:
        raise RuntimeError(f"cannot find {label} under {root}")
    if len(unique) > 1:
        rows = "\n  ".join(str(path) for path in unique[:8])
        raise RuntimeError(
            f"found multiple {label} candidates; pass a narrower model root:\n  {rows}"
        )
    return unique[0]


def _has_model_weights(root: Path) -> bool:
    names = {
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    }
    return any((root / name).is_file() for name in names)


def _discover_models(model_root: Path) -> dict[str, Path]:
    vggt_revision = _profile_value(VGGT_PROFILE, "model_revision")
    task_revision = _profile_value(TASK_PROFILE, "model_revision")
    sources: list[Path] = []
    vggt_models: list[Path] = []
    task_models: list[Path] = []
    for path in _walk_directories(model_root, maximum_depth=9):
        if (
            (path / "vggt/models/vggt.py").is_file()
            and (path / "vggt/models/aggregator.py").is_file()
        ):
            sources.append(path)
        if (
            path.name == vggt_revision
            and (path / "config.json").is_file()
            and _has_model_weights(path)
        ):
            vggt_models.append(path)
        if (
            path.name == task_revision
            and (path / "modules.json").is_file()
            and (path / "config.json").is_file()
            and _has_model_weights(path)
        ):
            task_models.append(path)
    return {
        "vggt_source": _one_candidate(sources, "VGGT source", model_root),
        "vggt_model": _one_candidate(vggt_models, "VGGT-1B snapshot", model_root),
        "task_model": _one_candidate(
            task_models, "Qwen3-VL-Embedding-2B snapshot", model_root
        ),
    }


def _find_agibot_root(data_root: Path) -> Path | None:
    candidates: list[Path] = []
    for path in _walk_directories(data_root, maximum_depth=8):
        if all((path / prefix).is_dir() for prefix in AGIBOT_PREFIXES):
            candidates.append(path)
    unique = sorted({path.resolve(strict=True) for path in candidates})
    if not unique:
        return None
    minimum_depth = min(len(path.parts) for path in unique)
    shallow = [path for path in unique if len(path.parts) == minimum_depth]
    if len(shallow) != 1:
        rows = "\n  ".join(str(path) for path in shallow[:8])
        raise RuntimeError(
            f"found multiple AgiBotWorld2026 roots; pass a narrower data root:\n  {rows}"
        )
    return shallow[0]


def _probe_agibot(snapshot: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            sys.executable,
            str(AGIBOT_CHECKER),
            "--snapshot-root",
            str(snapshot),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "AgiBotWorld2026 layout check failed:\n" + (result.stderr or result.stdout)
        )
    return json.loads(result.stdout)


def _candidate_profile(path: Path) -> bool:
    if path.is_symlink() or not path.is_file() or path.suffix not in {".yaml", ".yml"}:
        return False
    if path.stat().st_size > 8 * 1024 * 1024:
        return False
    payload = path.read_text(encoding="utf-8", errors="replace")
    return (
        "schema: wm3d_v8_data_profile_v4" in payload
        and "__MATERIALIZE_REQUIRED__" not in payload
    )


def _find_data_profile(data_root: Path, default_work_root: Path) -> Path | None:
    direct = [
        data_root / "control/public_robot_oxe.yaml",
        data_root / "public_robot_oxe.yaml",
        default_work_root / "control/public_robot_oxe.yaml",
    ]
    for path in direct:
        if _candidate_profile(path):
            return path.resolve(strict=True)
    candidates: list[Path] = []
    for directory in _walk_directories(data_root, maximum_depth=4):
        for name in ("public_robot_oxe.yaml", "data_profile.yaml"):
            path = directory / name
            if _candidate_profile(path):
                candidates.append(path)
    if not candidates:
        return None
    return _one_candidate(candidates, "materialized WM3D data profile", data_root)


def _profile_path_mismatches(profile: Path) -> list[str]:
    pattern = re.compile(
        r"^\s*(raw_root|adapter_config|manifest):\s*['\"]?([^'\"#]+?)['\"]?\s*(?:#.*)?$"
    )
    mismatches: list[str] = []
    for line in profile.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        role, value = match.groups()
        path = Path(value.strip())
        if not path.is_absolute():
            mismatches.append(f"{role} is not absolute: {value.strip()}")
        elif role == "raw_root" and not path.is_dir():
            mismatches.append(f"missing raw_root: {path}")
        elif role != "raw_root" and not path.is_file():
            mismatches.append(f"missing {role}: {path}")
    return mismatches


def _replace_assignment(payload: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
    replacement = f"{key}={value}"
    updated, count = pattern.subn(lambda _: replacement, payload, count=1)
    if count != 1:
        raise RuntimeError(f"site template does not contain exactly one {key} assignment")
    return updated


def _shell_path(path: Path) -> str:
    return shlex.quote(str(path))


def _write_site(
    output: Path,
    *,
    work_root: Path,
    data_root: Path,
    data_profile: Path | None,
    models: dict[str, Path],
) -> str:
    payload = SITE_TEMPLATE.read_text(encoding="utf-8")
    assignments = {
        "WORK_ROOT": _shell_path(work_root),
        "RAW_ROOT": _shell_path(data_root),
        "WM3D_VGGT_SOURCE_ROOT": _shell_path(models["vggt_source"]),
        "WM3D_VGGT_MODEL_SNAPSHOT": _shell_path(models["vggt_model"]),
        "QWEN3_VL_EMBEDDING_PATH": _shell_path(models["task_model"]),
        "DATA_PROFILE": _shell_path(
            data_profile or (work_root / "control/public_robot_oxe.yaml")
        ),
        "MASTER_ADDR": "${MASTER_ADDR:-127.0.0.1}",
    }
    for key, value in assignments.items():
        payload = _replace_assignment(payload, key, value)
    encoded = payload.encode("utf-8")
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        if output.is_file() and not output.is_symlink() and output.read_bytes() == encoded:
            output.chmod(0o600)
            return "verified-skip"
        raise FileExistsError(f"refusing to overwrite a different site file: {output}")
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.chmod(0o600)
    os.replace(temporary, output)
    return "created"


def _metadata_state(
    work_root: Path,
    data_profile: Path | None,
    profile_mismatches: list[str],
) -> tuple[str, list[str]]:
    if data_profile is None:
        return "RAW_COMPATIBLE", ["materialized data profile"]
    if profile_mismatches:
        return "PROFILE_PATH_MISMATCH", profile_mismatches
    required = {
        "task bank": work_root / "cache/native_p144/task_bank/index.jsonl",
        "task manifest": work_root / "cache/native_p144/cache_tasks.jsonl",
        "episode index": work_root / "cache/native_p144/episode_index.jsonl",
        "window index": work_root / "cache/native_p144/window_index_5b.jsonl",
        "normalization": work_root / "cache/native_p144/grouped_normalization_5b.json",
        "metadata seal": work_root
        / "streaming_metadata/native_p144/metadata_seal_5b.json",
    }
    missing = [label for label, path in required.items() if not path.is_file()]
    return ("PROFILE_READY" if missing else "TRAIN_METADATA_READY"), missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--site-output", type=Path)
    args = parser.parse_args()

    model_root = _real_directory(args.model_root, "model root")
    data_root = _real_directory(args.data_root, "data root")
    provisional_work = (args.work_root or Path("/data/wm3d")).absolute()
    data_profile = _find_data_profile(data_root, provisional_work)
    if args.work_root is not None:
        work_root = args.work_root.absolute()
    elif data_profile is not None and data_profile.parent.name == "control":
        work_root = data_profile.parent.parent
    else:
        work_root = provisional_work
    work_root.mkdir(parents=True, exist_ok=True)
    if work_root.is_symlink() or not work_root.is_dir():
        raise RuntimeError(f"work root must be a real directory: {work_root}")

    models = _discover_models(model_root)
    agibot_root = _find_agibot_root(data_root)
    agibot_evidence = _probe_agibot(agibot_root) if agibot_root is not None else None
    if data_profile is None and agibot_evidence is None:
        raise RuntimeError(
            "data root contains neither a materialized WM3D data profile nor a "
            "compatible AgiBotWorld2026 tree"
        )

    site_output = args.site_output or work_root / "control/5b_canary1k.env"
    site_status = _write_site(
        site_output,
        work_root=work_root,
        data_root=data_root,
        data_profile=data_profile,
        models=models,
    )
    profile_mismatches = (
        _profile_path_mismatches(data_profile) if data_profile is not None else []
    )
    data_state, missing_metadata = _metadata_state(
        work_root, data_profile, profile_mismatches
    )
    environment = work_root / "envs/wm3d-cu128/environment_receipt.json"
    runtime = work_root / "control/runtime_5b_canary1k.yaml"
    ready_to_train = (
        data_state == "TRAIN_METADATA_READY"
        and environment.is_file()
        and runtime.is_file()
    )
    print(
        json.dumps(
            {
                "schema": "wm3d_v8_5b_local_input_check_v1",
                "input_check": "PASS",
                "models": {
                    "vggt_source": "PASS",
                    "vggt_model": "PASS",
                    "task_model": "PASS",
                },
                "data_source": "transport_neutral_existing_files",
                "agibot_layout": "PASS" if agibot_evidence is not None else "NOT_SCANNED",
                "data_state": data_state,
                "missing_training_metadata": missing_metadata,
                "environment_ready": environment.is_file(),
                "runtime_ready": runtime.is_file(),
                "ready_to_train": ready_to_train,
                "site": str(site_output.absolute()),
                "site_status": site_status,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
