#!/usr/bin/env python3
"""Import an audited V7 residual plan into the standard V8 source ABI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

import yaml

from wm3d_v3.data.grouped_robot import ActionGroupSpec, EmbodimentSpec
from wm3d_v3.data.legacy_residual_inventory import import_legacy_residual_plan
from wm3d_v3.data.manifest_contract import sha256_file
from wm3d_v3.data.source_adapters import load_adapter_contract
from wm3d_v3.data.source_inventory import validate_written_inventory


ADAPTER_AUDIT_SCHEMA = "wm3d_v8_source_adapter_audit_receipt_v1"
DATA_PROFILE_SCHEMA = "wm3d_v8_data_profile_v4"


def _safe_input(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file")
    return path.resolve(strict=True)


def _profile_contract(
    path: Path, source_name: str
) -> tuple[EmbodimentSpec, tuple[str, ...], str, Path]:
    safe = _safe_input(path, label="data template")
    value = yaml.safe_load(safe.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != DATA_PROFILE_SCHEMA:
        raise RuntimeError(f"data template schema must be {DATA_PROFILE_SCHEMA}")
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
        raise RuntimeError(f"source embodiment {embodiment_name!r} is missing/duplicated")
    raw = candidates[0]
    if set(raw) != {"name", "embodiment_id", "groups"}:
        raise RuntimeError("embodiment fields mismatch")
    required_group = {
        "name",
        "group_id",
        "action_semantics",
        "state_semantics",
        "action_frame",
        "state_frame",
        "composition_operators",
    }
    groups: list[ActionGroupSpec] = []
    for group in raw["groups"]:
        if not isinstance(group, dict) or set(group) != required_group:
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
    representation = value.get("cache_representation")
    if not isinstance(representation, dict):
        raise RuntimeError("data template cache_representation must be a mapping")
    raw_slots = representation.get("view_slots")
    if (
        not isinstance(raw_slots, list)
        or not raw_slots
        or any(not isinstance(item, str) or not item for item in raw_slots)
        or len(raw_slots) != len(set(raw_slots))
        or int(representation.get("num_views", -1)) != len(raw_slots)
        or representation.get("missing_view_policy") != "mask_without_duplication"
    ):
        raise RuntimeError("data template does not declare a valid masked view vocabulary")
    return (
        EmbodimentSpec(
            name=embodiment_name,
            embodiment_id=int(raw["embodiment_id"]),
            groups=tuple(groups),
        ),
        tuple(raw_slots),
        sha256_file(safe),
        safe,
    )


def _audit_receipt(
    path: Path,
    *,
    source: str,
    adapter_sha: str,
    template_sha: str,
) -> tuple[Path, str]:
    safe = _safe_input(path, label="adapter audit receipt")
    value = json.loads(safe.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema") != ADAPTER_AUDIT_SCHEMA
        or value.get("source") != source
        or value.get("adapter_contract_sha256") != adapter_sha
        or value.get("data_template_sha256") != template_sha
        or value.get("structural_checks") != "pass"
        or value.get("semantic_review") != "operator_confirmed_fail_closed"
    ):
        raise RuntimeError("adapter audit receipt does not authorize this import")
    return safe, sha256_file(safe)


def _publish(path: Path, payload: bytes) -> None:
    target = path.absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_file() and not target.is_symlink() and target.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite non-identical inventory: {target}")
    temporary = target.with_name(
        f".{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_publishable(path: Path, payload: bytes) -> None:
    target = path.absolute()
    if target.exists() or target.is_symlink():
        if target.is_file() and not target.is_symlink() and target.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite non-identical inventory: {target}")


def _validate_manifest_payload(
    *,
    output_path: Path,
    payload: bytes,
    source: str,
    embodiment: EmbodimentSpec,
) -> dict[str, object]:
    output = output_path.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.name}.validate.{os.getpid()}.{uuid.uuid4().hex}"
    )
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        return validate_written_inventory(
            temporary, source=source, embodiment=embodiment
        )
    finally:
        temporary.unlink(missing_ok=True)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "严格重读 V7 legacy_v7_formal residual plan 的真实 Parquet/video，"
            "并生成标准 V8 source manifest/receipt；不会复制或生成 cache。"
        )
    )
    parser.add_argument("--legacy-plan", type=Path, required=True)
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
    return parser.parse_args()


def main() -> None:
    args = _args()
    embodiment, view_slots, template_sha, template_path = _profile_contract(
        args.data_template, args.source
    )
    adapter = load_adapter_contract(
        args.adapter_contract,
        expected_sha256=args.adapter_contract_sha256,
    )
    audit_path, audit_sha = _audit_receipt(
        args.adapter_audit_receipt,
        source=args.source,
        adapter_sha=adapter.sha256,
        template_sha=template_sha,
    )
    rows, receipt = import_legacy_residual_plan(
        plan_path=args.legacy_plan,
        raw_root=args.raw_root,
        source=args.source,
        embodiment=embodiment,
        adapter=adapter,
        view_slots=view_slots,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    manifest_payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    validation = _validate_manifest_payload(
        output_path=args.output_manifest,
        payload=manifest_payload,
        source=args.source,
        embodiment=embodiment,
    )
    manifest_path = args.output_manifest.absolute()
    final_receipt = {
        **receipt,
        "data_template_path": str(template_path),
        "data_template_sha256": template_sha,
        "adapter_contract_path": str(adapter.path),
        "adapter_contract_sha256": adapter.sha256,
        "adapter_audit_receipt_path": str(audit_path),
        "adapter_audit_receipt_sha256": audit_sha,
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "manifest_validation": validation,
    }
    receipt_payload = (
        json.dumps(final_receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    # Preflight the pair before publishing either file.  A stale/conflicting
    # receipt must not leave a newly published manifest behind (and vice
    # versa).  Each final creation is still atomic/no-clobber.
    _assert_publishable(args.output_manifest, manifest_payload)
    _assert_publishable(args.output_receipt, receipt_payload)
    _publish(args.output_manifest, manifest_payload)
    _publish(args.output_receipt, receipt_payload)
    print(json.dumps(final_receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
