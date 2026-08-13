#!/usr/bin/env python3
"""Materialize one fail-closed WM3D source manifest from an audited adapter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import uuid

import yaml

from wm3d.data.grouped_robot import ActionGroupSpec, EmbodimentSpec
from wm3d.data.manifest_contract import sha256_file
from wm3d.data.source_adapters import load_adapter_contract
from wm3d.data.source_inventory import (
    scan_lerobot_source,
    validate_written_inventory,
)


ADAPTER_AUDIT_SCHEMA = "wm3d_v8_source_adapter_audit_receipt_v1"


def _episode_indices(path: Path | None) -> tuple[tuple[int, ...] | None, str | None]:
    if path is None:
        return None, None
    safe = path.resolve(strict=True)
    if safe.is_symlink() or not safe.is_file():
        raise RuntimeError("episode index file must be a regular file")
    result: list[int] = []
    for line_number, line in enumerate(safe.read_text(encoding="utf-8").splitlines(), 1):
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        try:
            index = int(value)
        except ValueError as exc:
            raise RuntimeError(
                f"{safe}:{line_number}: episode index is not an integer"
            ) from exc
        if index < 0:
            raise RuntimeError(f"{safe}:{line_number}: episode index is negative")
        result.append(index)
    if not result or len(result) != len(set(result)):
        raise RuntimeError("episode index file must contain unique non-negative values")
    return tuple(result), sha256_file(safe)


def _publish(path: Path, payload: bytes) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite non-identical inventory: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _embodiment_from_template(path: Path, source_name: str) -> tuple[EmbodimentSpec, str]:
    safe = path.resolve(strict=True)
    value = yaml.safe_load(safe.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "wm3d_v8_data_profile_v4":
        raise RuntimeError("data template schema must be wm3d_v8_data_profile_v4")
    sources = value.get("sources")
    if not isinstance(sources, list):
        raise RuntimeError("data template sources must be a list")
    matches = [row for row in sources if isinstance(row, dict) and row.get("name") == source_name]
    if len(matches) != 1:
        raise RuntimeError(f"data template must contain source {source_name!r} exactly once")
    embodiment_name = str(matches[0].get("embodiment", ""))
    raw_embodiments = value.get("embodiments")
    if not isinstance(raw_embodiments, list):
        raise RuntimeError("data template embodiments must be a list")
    candidates = [
        row
        for row in raw_embodiments
        if isinstance(row, dict) and row.get("name") == embodiment_name
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"source embodiment {embodiment_name!r} is missing/duplicated")
    raw = candidates[0]
    if set(raw) != {"name", "embodiment_id", "groups"}:
        raise RuntimeError("embodiment fields mismatch")
    groups = []
    required = {
        "name", "group_id", "action_semantics", "state_semantics",
        "action_frame", "state_frame", "composition_operators",
    }
    for group in raw["groups"]:
        if not isinstance(group, dict) or set(group) != required:
            raise RuntimeError("embodiment group fields mismatch")
        groups.append(
            ActionGroupSpec(
                name=str(group["name"]),
                group_id=int(group["group_id"]),
                action_semantics=tuple(str(item) for item in group["action_semantics"]),
                state_semantics=tuple(str(item) for item in group["state_semantics"]),
                action_frame=str(group["action_frame"]),
                state_frame=str(group["state_frame"]),
                composition_operators=tuple(
                    str(item) for item in group["composition_operators"]
                ),
            )
        )
    return (
        EmbodimentSpec(
            name=embodiment_name,
            embodiment_id=int(raw["embodiment_id"]),
            groups=tuple(groups),
        ),
        sha256_file(safe),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-template", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--adapter-contract", type=Path, required=True)
    parser.add_argument("--adapter-contract-sha256", required=True)
    parser.add_argument("--adapter-audit-receipt", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=3407)
    parser.add_argument("--train-fraction", type=float, default=0.98)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--default-task", default="")
    parser.add_argument(
        "--episode-index-file",
        type=Path,
        help=(
            "可选的显式 episode_index 白名单；用于审计子集/烟测，文件 SHA "
            "会写入 receipt，正式全量训练应省略。"
        ),
    )
    args = parser.parse_args()

    embodiment, template_sha = _embodiment_from_template(
        args.data_template, args.source
    )
    root = args.raw_root
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"source raw_root is not an absolute real directory: {root}")
    root = root.resolve(strict=True)
    adapter = load_adapter_contract(
        args.adapter_contract,
        expected_sha256=args.adapter_contract_sha256,
    )
    audit_path = args.adapter_audit_receipt.resolve(strict=True)
    if audit_path.is_symlink() or not audit_path.is_file():
        raise RuntimeError("adapter audit receipt must be a regular file")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        not isinstance(audit, dict)
        or audit.get("schema") != ADAPTER_AUDIT_SCHEMA
        or audit.get("source") != args.source
        or audit.get("adapter_contract_sha256") != adapter.sha256
        or audit.get("data_template_sha256") != template_sha
        or audit.get("structural_checks") != "pass"
        or audit.get("semantic_review") != "operator_confirmed_fail_closed"
    ):
        raise RuntimeError("adapter audit receipt does not authorize this inventory")
    episode_indices, episode_indices_sha = _episode_indices(args.episode_index_file)
    rows, receipt = scan_lerobot_source(
        root=root,
        source=args.source,
        embodiment=embodiment,
        adapter=adapter,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        default_task=args.default_task,
        episode_indices=episode_indices,
    )
    manifest_payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    _publish(args.output_manifest, manifest_payload)
    manifest = args.output_manifest.absolute().resolve(strict=True)
    validation = validate_written_inventory(
        manifest,
        source=args.source,
        embodiment=embodiment,
    )
    final_receipt = {
        **receipt,
        "data_template_path": str(args.data_template.resolve(strict=True)),
        "data_template_sha256": template_sha,
        "raw_root": str(root),
        "adapter_contract_path": str(adapter.path),
        "adapter_contract_sha256": adapter.sha256,
        "adapter_audit_receipt_path": str(audit_path),
        "adapter_audit_receipt_sha256": sha256_file(audit_path),
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "manifest_validation": validation,
        "episode_index_file_path": (
            None
            if args.episode_index_file is None
            else str(args.episode_index_file.resolve(strict=True))
        ),
        "episode_index_file_sha256": episode_indices_sha,
    }
    _publish(
        args.output_receipt,
        (json.dumps(final_receipt, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    print(json.dumps(final_receipt, sort_keys=True))


if __name__ == "__main__":
    main()
