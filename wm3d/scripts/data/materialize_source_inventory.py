#!/usr/bin/env python3
"""Materialize one fail-closed WM3D source manifest from an audited adapter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import uuid

import yaml

from wm3d.data.file_binding import bind_stat_asset, regular_asset, verify_stat_asset
from wm3d.data.grouped_robot import ActionGroupSpec, EmbodimentSpec
from wm3d.data.manifest_contract import sha256_file
from wm3d.data.source_adapters import load_adapter_contract
from wm3d.data.source_inventory import (
    scan_lerobot_source,
    validate_written_inventory,
)


ADAPTER_AUDIT_SCHEMA = "wm3d_v8_source_adapter_audit_receipt_v1"
ADMITTED_ADAPTER_AUDIT_SCHEMA = "wm3d_admitted_source_adapter_audit_receipt_v1"
ADMITTED_EPISODE_CATALOG_SCHEMA = "wm3d_admitted_episode_catalog_v1"


def _bound_json(path: Path, *, admission_id: str) -> tuple[dict, dict]:
    relative = path.absolute().as_posix().lstrip("/")
    safe = regular_asset(Path("/"), relative)
    binding = bind_stat_asset(Path("/"), relative, admission_id=admission_id)
    value = json.loads(safe.read_text(encoding="utf-8"))
    verify_stat_asset(Path("/"), relative, binding)
    if not isinstance(value, dict):
        raise RuntimeError("admitted JSON must be a mapping")
    return value, binding


def _admitted_catalog(path: Path, *, admission_id: str, source: str) -> tuple[dict, dict]:
    value, binding = _bound_json(path, admission_id=admission_id)
    if (set(value) != {"schema", "admission_id", "source", "episodes"}
            or value["schema"] != ADMITTED_EPISODE_CATALOG_SCHEMA
            or value["admission_id"] != admission_id or value["source"] != source):
        raise RuntimeError("admitted episode catalog identity/schema mismatch")
    records = value["episodes"]
    if not isinstance(records, dict) or not records:
        raise RuntimeError("admitted episode catalog must not be empty")
    result = {}
    for key, row in records.items():
        if not key.isdecimal() or str(int(key)) != key:
            raise RuntimeError("episode catalog keys must be canonical non-negative integers")
        result[int(key)] = row
    return result, binding



def _episode_indices(path: Path | None) -> tuple[tuple[int, ...] | None, str | None]:
    if path is None:
        return None, None
    safe = path.resolve(strict=True)
    if safe.is_symlink() or not safe.is_file():
        raise RuntimeError("episode index file must be a regular file")
    result: list[int] = []
    for line_number, line in enumerate(
        safe.read_text(encoding="utf-8").splitlines(), 1
    ):
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


def _episode_ranges(path: Path | None) -> tuple[dict | None, str | None]:
    if path is None:
        return None, None
    safe = path.resolve(strict=True)
    if safe.is_symlink() or not safe.is_file():
        raise RuntimeError("episode range file must be a regular file")
    value = json.loads(safe.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise RuntimeError("episode range file must contain a non-empty JSON mapping")
    return value, sha256_file(safe)


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


def _embodiment_from_template(
    path: Path, source_name: str, *, admission_id: str | None = None
) -> tuple[EmbodimentSpec, str | dict]:
    relative = path.absolute().as_posix().lstrip("/")
    safe = (
        path.resolve(strict=True) if admission_id is None
        else regular_asset(Path("/"), relative)
    )
    binding = (
        None if admission_id is None
        else bind_stat_asset(Path("/"), relative, admission_id=admission_id)
    )
    value = yaml.safe_load(safe.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "wm3d_v8_data_profile_v4":
        raise RuntimeError("data template schema must be wm3d_v8_data_profile_v4")
    sources = value.get("sources")
    if not isinstance(sources, list):
        raise RuntimeError("data template sources must be a list")
    matches = [
        row
        for row in sources
        if isinstance(row, dict) and row.get("name") == source_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"data template must contain source {source_name!r} exactly once"
        )
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
        raise RuntimeError(
            f"source embodiment {embodiment_name!r} is missing/duplicated"
        )
    raw = candidates[0]
    if set(raw) != {"name", "embodiment_id", "groups"}:
        raise RuntimeError("embodiment fields mismatch")
    groups = []
    required = {
        "name",
        "group_id",
        "action_semantics",
        "state_semantics",
        "action_frame",
        "state_frame",
        "composition_operators",
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
    if binding is not None:
        verify_stat_asset(Path("/"), relative, binding)
    return (
        EmbodimentSpec(
            name=embodiment_name,
            embodiment_id=int(raw["embodiment_id"]),
            groups=tuple(groups),
        ),
        sha256_file(safe) if binding is None else binding,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-template", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--adapter-contract", type=Path, required=True)
    binding_args = parser.add_mutually_exclusive_group(required=True)
    binding_args.add_argument("--adapter-contract-sha256")
    binding_args.add_argument("--adapter-contract-binding", type=Path)
    parser.add_argument("--admission-id")
    parser.add_argument("--episode-catalog", type=Path)
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
    parser.add_argument(
        "--episode-range-file",
        type=Path,
        help=(
            "可选的 episode_index→[[start,stop),...] JSON；用于 DROID "
            "非 idle 片段，文件 SHA 会写入 receipt，指定后对所选 episode fail closed。"
        ),
    )
    args = parser.parse_args()

    admitted = args.admission_id is not None
    if admitted:
        if (not args.admission_id.strip() or args.adapter_contract_binding is None
                or args.episode_catalog is None
                or args.episode_index_file is not None
                or args.episode_range_file is not None):
            raise RuntimeError(
                "admitted mode requires explicit binding/catalog and whole episodes"
            )
    elif args.adapter_contract_binding is not None or args.episode_catalog is not None:
        raise RuntimeError("stat binding/catalog requires explicit --admission-id")
    embodiment, template_binding = _embodiment_from_template(
        args.data_template, args.source, admission_id=args.admission_id
    )
    root = args.raw_root
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"source raw_root is not an absolute real directory: {root}")
    root = root.resolve(strict=True)
    catalog = None
    catalog_binding = None
    audit_binding = None
    if admitted:
        expected_adapter_binding, _ = _bound_json(
            args.adapter_contract_binding, admission_id=args.admission_id
        )
        adapter = load_adapter_contract(
            args.adapter_contract, expected_binding=expected_adapter_binding
        )
        catalog, catalog_binding = _admitted_catalog(
            args.episode_catalog, admission_id=args.admission_id, source=args.source
        )
        audit, audit_binding = _bound_json(
            args.adapter_audit_receipt, admission_id=args.admission_id
        )
        audit_path = args.adapter_audit_receipt.absolute()
        authorized = (
            audit.get("schema") == ADMITTED_ADAPTER_AUDIT_SCHEMA
            and audit.get("admission_id") == args.admission_id
            and audit.get("adapter_contract_binding") == adapter.file_binding
            and audit.get("data_template_binding") == template_binding
            and audit.get("episode_catalog_binding") == catalog_binding
        )
        episode_indices = tuple(sorted(catalog))
        episode_ranges = None
        episode_indices_sha = episode_ranges_sha = None
    else:
        adapter = load_adapter_contract(
            args.adapter_contract, expected_sha256=args.adapter_contract_sha256
        )
        audit_path = args.adapter_audit_receipt.resolve(strict=True)
        if audit_path.is_symlink() or not audit_path.is_file():
            raise RuntimeError("adapter audit receipt must be a regular file")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        authorized = (
            isinstance(audit, dict)
            and audit.get("schema") == ADAPTER_AUDIT_SCHEMA
            and audit.get("adapter_contract_sha256") == adapter.sha256
            and audit.get("data_template_sha256") == template_binding
        )
        episode_indices, episode_indices_sha = _episode_indices(args.episode_index_file)
        episode_ranges, episode_ranges_sha = _episode_ranges(args.episode_range_file)
    if (not authorized or audit.get("source") != args.source
            or audit.get("structural_checks") != "pass"
            or audit.get("semantic_review") != "operator_confirmed_fail_closed"):
        raise RuntimeError("adapter audit receipt does not authorize this inventory")
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
        episode_ranges=episode_ranges,
        admission_id=args.admission_id,
        episode_catalog=catalog,
    )
    manifest_payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")
    _publish(args.output_manifest, manifest_payload)
    manifest = args.output_manifest.absolute().resolve(strict=True)
    validation = validate_written_inventory(
        manifest,
        source=args.source,
        embodiment=embodiment,
        admission_id=args.admission_id,
    )
    if admitted:
        # Recheck all small contracts after the scan and bind the published output.
        for path, expected in (
            (args.data_template, template_binding),
            (args.episode_catalog, catalog_binding),
            (audit_path, audit_binding),
        ):
            verify_stat_asset(Path("/"), path.absolute().as_posix().lstrip("/"), expected)
        final_receipt = {
            **receipt,
            "data_template_path": str(args.data_template.absolute()),
            "data_template_binding": template_binding,
            "adapter_contract_path": str(adapter.path),
            "adapter_audit_receipt_path": str(audit_path),
            "adapter_audit_receipt_binding": audit_binding,
            "episode_catalog_path": str(args.episode_catalog.absolute()),
            "episode_catalog_binding": catalog_binding,
            "manifest_path": str(manifest),
            "manifest_binding": bind_stat_asset(
                Path("/"), manifest.as_posix().lstrip("/"), admission_id=args.admission_id
            ),
            "manifest_validation": validation,
        }
    else:
        final_receipt = {
            **receipt,
            "data_template_path": str(args.data_template.resolve(strict=True)),
            "data_template_sha256": template_binding,
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
            "episode_range_file_path": (
                None
                if args.episode_range_file is None
                else str(args.episode_range_file.resolve(strict=True))
            ),
            "episode_range_file_sha256": episode_ranges_sha,
        }
    _publish(
        args.output_receipt,
        (json.dumps(final_receipt, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    print(json.dumps(final_receipt, sort_keys=True))


if __name__ == "__main__":
    main()
