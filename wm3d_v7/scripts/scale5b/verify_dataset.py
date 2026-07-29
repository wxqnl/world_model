#!/usr/bin/env python3
"""Control-plane or full-payload verifier for a sealed V7 5B dataset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

import torch

from wm3d_v3.data.scale5b_contracts import (
    canonical_sha256,
    load_contract,
    load_seal,
    resolve_real_directory,
    resolve_regular_file,
    sha256_file,
    verify_dataset_seal,
)
from wm3d_v3.data.scale5b_dataset import (
    Native5BSourceDataset,
    WindowLoaderConfig,
)

PART_SCHEMA = "wm3d_v7_native5b_encoded_part_v2"
PART_COMMIT_SCHEMA = "wm3d_v7_native5b_encoded_part_commit_v2"
PART_PAYLOAD_FILES = {
    "features.safetensors",
    "actions.safetensors",
    "rgb.jpgpack",
    "windows.parquet",
}
PART_NAME_RE = re.compile(r"^part-([0-9]{5})-([0-9]{6})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("control", "deep"), default="control")
    parser.add_argument("--sample-windows-per-source", type=int, default=2)
    return parser.parse_args()


def _verify_part_payload(root: Path, relative: str) -> dict[str, int]:
    manifest_path = resolve_regular_file(root, relative)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    part = manifest_path.parent
    commit_path = resolve_regular_file(part, "COMMITTED.json")
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    match = PART_NAME_RE.fullmatch(part.name)
    if (
        match is None
        or manifest.get("schema") != PART_SCHEMA
        or manifest.get("part_name") != part.name
        or int(manifest.get("worker_shard_id", -1)) != int(match.group(1))
        or int(manifest.get("part_index", -1)) != int(match.group(2))
        or int(manifest.get("worker_num_shards", -1)) <= int(match.group(1))
        or commit.get("schema") != PART_COMMIT_SCHEMA
        or commit.get("part_name") != part.name
        or sha256_file(manifest_path) != commit.get("manifest_sha256")
        or canonical_sha256(manifest)
        != commit.get("manifest_content_sha256")
        or not isinstance(files, dict)
        or set(files) != PART_PAYLOAD_FILES
    ):
        raise ValueError(f"deep part contract failed: {part}")
    expected_entries = PART_PAYLOAD_FILES | {
        "manifest.json",
        "COMMITTED.json",
    }
    actual_entries = {path.name for path in part.iterdir()}
    if actual_entries != expected_entries:
        raise ValueError(
            f"{part}: file set mismatch "
            f"missing={sorted(expected_entries - actual_entries)} "
            f"extra={sorted(actual_entries - expected_entries)}"
        )
    errors: list[str] = []
    total = 0
    for name, evidence in sorted(files.items()):
        try:
            path = resolve_regular_file(part, name)
            info = os.lstat(path)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{part / name}: unsafe or missing: {exc}")
            continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            errors.append(f"{path}: not a regular file")
            continue
        total += int(info.st_size)
        if int(info.st_size) != int(evidence["size"]):
            errors.append(f"{path}: size mismatch")
        elif sha256_file(path) != evidence["sha256"]:
            errors.append(f"{path}: sha256 mismatch")
    if errors:
        raise ValueError("deep payload verification failed:\n" + "\n".join(errors))
    return {"files": len(files), "bytes": total}


def _committed_part_names(root: Path) -> set[str]:
    parts_root = resolve_real_directory(
        root / "payload" / "parts",
        "encoded payload parts root",
    )
    result: set[str] = set()
    for path in parts_root.iterdir():
        if path.name.startswith("."):
            continue
        info = os.lstat(path)
        if (
            PART_NAME_RE.fullmatch(path.name) is None
            or stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
        ):
            raise ValueError(f"unexpected non-committed payload entry: {path}")
        result.add(path.name)
    return result


def _sample_windows(
    root: Path,
    *,
    count: int,
) -> dict[str, Any]:
    if count <= 0:
        return {}
    contract = load_contract(
        resolve_regular_file(root, "control/dataset_contract.json")
    )
    config = WindowLoaderConfig(
        rgb_decode_indices=(3, 7, 11, 15),
        memory_slots=12,
        memory_stride_frames=25,
        row_group_cache_size=2,
        task_cache_size=128,
        strict_shapes=True,
    )
    result = {}
    discrete_group_ids = {
        int(group.group_id)
        for embodiment in contract.embodiments
        for group in embodiment.action_groups
        if "grip" in group.control_mode or "discrete" in group.control_mode
    }
    for source in contract.source_order:
        dataset = Native5BSourceDataset(
            root,
            contract,
            source_name=source,
            split="val",
            config=config,
        )
        indices = sorted(
            {
                round(position * (len(dataset) - 1) / max(1, count - 1))
                for position in range(min(count, len(dataset)))
            }
        )
        samples = []
        for index in indices:
            sample = dataset[index]
            bad = [
                name
                for name, value in sample.items()
                if isinstance(value, torch.Tensor)
                and value.is_floating_point()
                and not bool(torch.isfinite(value).all())
            ]
            if bad:
                raise ValueError(f"{source}[{index}] has non-finite tensors {bad}")
            active_group_ids = {
                int(group_id)
                for group_id, active in zip(
                    sample["action_group_ids"].tolist(),
                    sample["action_group_mask"].tolist(),
                    strict=True,
                )
                if active
            }
            contact_valid = int(sample["target_contact_mask"].sum())
            expects_contact = bool(
                active_group_ids.intersection(discrete_group_ids)
            )
            if expects_contact != (contact_valid > 0):
                raise ValueError(
                    f"{source}[{index}] contact mask disagrees with "
                    "discrete action groups"
                )
            samples.append(
                {
                    "index": index,
                    "world_shape": list(sample["world_tokens"].shape),
                    "rgb_shape": list(sample["target_rgb"].shape),
                    "action_valid": int(sample["target_action_dim_mask"].sum()),
                    "contact_valid": contact_valid,
                }
            )
        result[source] = samples
    return result


def main() -> None:
    args = parse_args()
    root = resolve_real_directory(args.dataset_root, "dataset root")
    report = verify_dataset_seal(root)
    if not report["pass"]:
        raise ValueError("control-plane verification failed:\n" + "\n".join(report["errors"]))
    output: dict[str, Any] = {"pass": True, "mode": args.mode, "control": report}
    if args.mode == "deep":
        seal = load_seal(
            resolve_regular_file(root, "receipts/dataset_seal.json")
        )
        manifest_paths = sorted(
            relative
            for relative in seal.payload_manifest_files
            if relative.endswith("/manifest.json")
        )
        totals = {"parts": 0, "files": 0, "bytes": 0}
        sealed_parts = {Path(relative).parent.name for relative in manifest_paths}
        actual_parts = _committed_part_names(root)
        if actual_parts != sealed_parts:
            raise ValueError(
                "sealed payload part set mismatch: "
                f"missing={sorted(sealed_parts - actual_parts)} "
                f"extra={sorted(actual_parts - sealed_parts)}"
            )
        for relative in manifest_paths:
            value = _verify_part_payload(root, relative)
            totals["parts"] += 1
            totals["files"] += value["files"]
            totals["bytes"] += value["bytes"]
        output["payload"] = totals
    output["sample_windows"] = _sample_windows(
        root,
        count=args.sample_windows_per_source,
    )
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
