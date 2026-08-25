#!/usr/bin/env python3
"""Structurally audit, then explicitly approve one source adapter.

This command cannot infer units, frames, gripper polarity or control semantics.
It records the operator's explicit review and proves that the approved mapping
is structurally compatible with every inspected LeRobot root.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import uuid

import yaml

from scripts.data.materialize_source_inventory import _embodiment_from_template
from wm3d.data.manifest_contract import sha256_file
from wm3d.data.source_adapters import load_adapter_contract


AUDIT_SCHEMA = "wm3d_v8_raw_schema_audit_v1"
CANDIDATE_SCHEMA = "wm3d_v8_source_adapter_candidate_v1"
RECEIPT_SCHEMA = "wm3d_v8_source_adapter_audit_receipt_v1"
CONFIRM = "I_VERIFIED_FIELDS_UNITS_FRAMES_GRIPPER_GROUPS_AND_NATIVE_CLOCKS"


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve(strict=True)


def _publish(path: Path, value: dict) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite non-identical receipt: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema-audit", type=Path, required=True)
    parser.add_argument("--adapter-candidate", type=Path, required=True)
    parser.add_argument("--adapter-contract", type=Path, required=True)
    parser.add_argument("--adapter-contract-sha256", required=True)
    parser.add_argument("--data-template", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.confirm != CONFIRM:
        raise RuntimeError(
            "formal adapter audit requires the exact --confirm literal printed in --help/source"
        )
    if not args.operator.strip():
        raise RuntimeError("--operator cannot be blank")

    audit_path = _regular(args.schema_audit, "schema audit")
    candidate_path = _regular(args.adapter_candidate, "adapter candidate")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if (
        not isinstance(audit, dict)
        or audit.get("schema") != AUDIT_SCHEMA
        or not audit.get("all_roots_inspected")
        or not audit.get("homogeneous")
        or not audit.get("roots")
    ):
        raise RuntimeError("schema audit is partial, heterogeneous or invalid")
    if (
        not isinstance(candidate, dict)
        or candidate.get("schema") != CANDIDATE_SCHEMA
        or candidate.get("formal_adapter") is not False
        or audit.get("adapter_candidate_sha256") != sha256_file(candidate_path)
        or Path(str(audit.get("adapter_candidate_path", ""))).resolve(strict=True)
        != candidate_path
    ):
        raise RuntimeError("adapter candidate is not the one bound by the schema audit")

    embodiment, template_sha = _embodiment_from_template(args.data_template, args.source)
    adapter = load_adapter_contract(
        args.adapter_contract,
        expected_sha256=args.adapter_contract_sha256,
    )
    specs = {group.name: group for group in embodiment.groups}
    mappings = {group.group: group for group in adapter.groups}
    if set(specs) != set(mappings):
        raise RuntimeError("adapter groups differ from the selected embodiment")
    for name, spec in specs.items():
        mapping = mappings[name]
        raw_action_width = sum(len(term.columns) for term in mapping.action)
        action_width = (
            raw_action_width if mapping.action_transform == "identity" else 7
        )
        if action_width != spec.action_dim:
            raise RuntimeError(f"{name}: adapter action width differs from embodiment")
        raw_state_width = sum(len(term.columns) for term in mapping.state)
        if mapping.state_transform == "identity":
            state_width = raw_state_width
        elif mapping.state_transform == "zero":
            state_width = 0
        else:
            state_width = 10
        if state_width != spec.state_dim:
            raise RuntimeError(f"{name}: adapter state width differs from embodiment")

    template = yaml.safe_load(_regular(args.data_template, "data template").read_text())
    slots = set(template["cache_representation"]["view_slots"])
    if not {view.name for view in adapter.views}.issubset(slots):
        raise RuntimeError("adapter maps a view outside data-profile canonical slots")

    required_fields = set(adapter.required_array_keys)
    required_views = {view.key for view in adapter.views}
    for root in audit["roots"]:
        samples = root.get("sample_data")
        if not isinstance(samples, list) or not samples:
            raise RuntimeError("audited root has no Parquet schema sample")
        for sample in samples:
            available = set(sample.get("columns", {}))
            missing = required_fields - available
            if missing:
                raise RuntimeError(
                    f"{root.get('relative_root')}: adapter fields absent from sample: "
                    f"{sorted(missing)}"
                )
            for mapping in adapter.groups:
                for term in (*mapping.action, *mapping.state):
                    shape = sample["columns"][term.key]
                    width = shape.get("list_size")
                    if width is None:
                        observed_widths = shape.get("observed_list_widths")
                        observed_rows = int(shape.get("observed_list_rows", 0))
                        observed_nulls = int(shape.get("observed_list_null_rows", 0))
                        if shape.get("arrow_type") in {
                            "bool",
                            "double",
                            "float",
                            "int8",
                            "int16",
                            "int32",
                            "int64",
                            "uint8",
                            "uint16",
                            "uint32",
                            "uint64",
                        }:
                            width = 1
                        elif (
                            not isinstance(observed_widths, list)
                            or len(observed_widths) != 1
                            or observed_rows <= 0
                            or observed_nulls != 0
                        ):
                            raise RuntimeError(
                                f"{root.get('relative_root')}: {term.key} has no "
                                "single non-null observed payload width"
                            )
                        else:
                            width = int(observed_widths[0])
                    if max(term.columns) >= int(width):
                        raise RuntimeError(
                            f"{root.get('relative_root')}: {term.key} does not expose "
                            f"requested columns {term.columns}"
                        )
        declared = root.get("declared_features")
        if not isinstance(declared, dict) or not required_views.issubset(declared):
            raise RuntimeError(
                f"{root.get('relative_root')}: adapter RGB keys are absent from info features"
            )

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "source": args.source,
        "operator": args.operator.strip(),
        "explicit_confirmation": CONFIRM,
        "schema_audit_path": str(audit_path),
        "schema_audit_sha256": sha256_file(audit_path),
        "adapter_candidate_path": str(candidate_path),
        "adapter_candidate_sha256": sha256_file(candidate_path),
        "adapter_contract_path": str(adapter.path),
        "adapter_contract_sha256": adapter.sha256,
        "data_template_path": str(_regular(args.data_template, "data template")),
        "data_template_sha256": template_sha,
        "upstream_receipt_sha256": audit["upstream_receipt_sha256"],
        "roots_audited": int(audit["roots_inspected"]),
        "structural_checks": "pass",
        "semantic_review": "operator_confirmed_fail_closed",
    }
    _publish(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
