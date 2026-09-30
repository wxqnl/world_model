#!/usr/bin/env python3
"""Bind audited source inventory receipts into one immutable data profile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import uuid

import yaml

from wm3d.data.file_binding import bind_stat_asset, verify_stat_asset
from wm3d.data.manifest_contract import (
    ADMITTED_CACHE_INDEX_SCHEMA, ADMITTED_DATA_PROFILE_SCHEMA,
    load_data_profile, sha256_file,
)


SHA_RE = re.compile(r"[0-9a-f]{64}")


def _assignments(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        name, separator, path = raw.partition("=")
        if not separator or not name or name in result:
            raise ValueError(f"invalid/duplicate --inventory {raw!r}")
        result[name] = Path(path).resolve(strict=True)
    return result


def _publish(path: Path, payload: bytes) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite non-identical data profile: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bind(path: Path, admission_id: str):
    return bind_stat_asset(Path("/"), path.absolute().as_posix().lstrip("/"),
                           admission_id=admission_id)


def _verify(path: Path, binding, admission_id: str):
    if not isinstance(binding, dict) or binding.get("admission_id") != admission_id:
        raise RuntimeError("profile input admission identity mismatch")
    return verify_stat_asset(Path("/"), path.absolute().as_posix().lstrip("/"), binding)


def _read_bound(path: Path, admission_id: str):
    binding = _bind(path, admission_id)
    value = json.loads(path.read_text(encoding="utf-8"))
    _verify(path, binding, admission_id)
    return value, binding


def _materialize_admitted(args, template: Path, value, template_binding):
    admission_id = args.admission_id
    if not admission_id or not admission_id.strip():
        raise RuntimeError("admission identity must be non-empty")
    sources = value.get("sources")
    if not isinstance(sources, list) or not sources or any(not isinstance(s, dict) for s in sources):
        raise RuntimeError("admitted template must declare sources")
    names = [s.get("name") for s in sources]
    if any(not isinstance(n, str) or not n.strip() for n in names) or len(set(names)) != len(names):
        raise RuntimeError("admitted template source names empty/duplicated")
    new_names = {s["name"] for s in sources if s.get("origin") == "admitted"}
    if any(s.get("origin") not in {"admitted", "inherited_legacy"} for s in sources):
        raise RuntimeError("admitted template sources must explicitly declare their origin")
    receipts = _assignments(args.inventory)
    if set(receipts) != new_names:
        raise RuntimeError("inventory receipt set must exactly match newly admitted sources")
    receipt_bindings = {}
    for source in sources:
        name = source["name"]
        if source["origin"] == "inherited_legacy":
            # Metadata lineage is checked against the original profile and seal
            # by load_data_profile below. Never re-run the old source inventory.
            for field, binding_field in (("manifest", "manifest_binding"),
                                         ("adapter_config", "adapter_contract_binding")):
                candidate = Path(source[field])
                if not candidate.is_absolute():
                    candidate = template.parent / candidate
                source[field] = str(candidate.absolute())
                observed = _bind(candidate, admission_id)
                if binding_field in source and source[binding_field] != observed:
                    raise RuntimeError(f"{name}: inherited metadata identity changed")
                source[binding_field] = observed
            continue
        if "manifest_sha256" in source or "adapter_contract_sha256" in source:
            raise RuntimeError(f"{name}: new stat source must not have fabricated SHA fields")
        rp = receipts[name]
        receipt, rb = _read_bound(rp, admission_id)
        if (receipt.get("schema") != "wm3d_admitted_source_inventory_receipt_v1"
                or receipt.get("source") != name or receipt.get("admission_id") != admission_id
                or receipt.get("data_template_binding") != template_binding):
            raise RuntimeError(f"{name}: inventory source/template/admission mismatch")
        bound_paths = []
        for field, binding_field in (
            ("adapter_audit_receipt_path", "adapter_audit_receipt_binding"),
            ("episode_catalog_path", "episode_catalog_binding"),
            ("manifest_path", "manifest_binding"),
            ("adapter_contract_path", "adapter_contract_binding"),
        ):
            candidate = Path(receipt[field])
            _verify(candidate, receipt[binding_field], admission_id)
            bound_paths.append((candidate, receipt[binding_field]))
        audit = json.loads(Path(receipt["adapter_audit_receipt_path"]).read_text())
        if (audit.get("schema") != "wm3d_admitted_source_adapter_audit_receipt_v1"
                or audit.get("source") != name or audit.get("admission_id") != admission_id
                or audit.get("semantic_review") != "operator_confirmed_fail_closed"
                or audit.get("structural_checks") != "pass"
                or audit.get("adapter_contract_binding") != receipt["adapter_contract_binding"]
                or audit.get("data_template_binding") != template_binding
                or audit.get("episode_catalog_binding") != receipt["episode_catalog_binding"]):
            raise RuntimeError(f"{name}: physical adapter audit is missing/mismatched")
        validation = receipt.get("manifest_validation", {})
        if not isinstance(validation, dict) or int(validation.get("episodes", 0)) <= 0:
            raise RuntimeError(f"{name}: inventory has no validated episodes")
        source.update(
            raw_root=str(Path(receipt["raw_root"]).absolute()),
            manifest=str(Path(receipt["manifest_path"]).absolute()),
            manifest_binding=receipt["manifest_binding"],
            adapter_config=str(Path(receipt["adapter_contract_path"]).absolute()),
            adapter_contract_binding=receipt["adapter_contract_binding"],
        )
        for candidate, binding in bound_paths:
            _verify(candidate, binding, admission_id)
        _verify(rp, rb, admission_id)
        receipt_bindings[name] = dict(path=str(rp), binding=rb)
    value["schema"] = ADMITTED_DATA_PROFILE_SCHEMA
    value["admission_id"] = admission_id
    value["cache"]["schema"] = ADMITTED_CACHE_INDEX_SCHEMA
    value["cache"]["resume"] = "receipt_and_stat"
    _verify(template, template_binding, admission_id)
    payload = yaml.safe_dump(value, sort_keys=False, allow_unicode=True).encode("utf-8")
    output = args.output.absolute()
    # Validate the complete prospective profile before publishing its final name.
    # Relative metadata paths have already been made absolute above.
    temporary = output.with_name(f".{output.name}.checking.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        _publish(temporary, payload)
        profile = load_data_profile(
            temporary, expected_binding=_bind(temporary, admission_id),
            verify_source_manifests=True,
        )
        _publish(output, payload)
    finally:
        temporary.unlink(missing_ok=True)
    profile_binding = _bind(output, admission_id)
    final_receipt = {
        "schema": "wm3d_admitted_data_profile_receipt_v1",
        "admission_id": admission_id, "template_path": str(template),
        "template_binding": template_binding, "data_profile_path": str(output),
        "data_profile_binding": profile_binding,
        "source_inventory_receipts": receipt_bindings,
        "inherited": profile.inherited,
        "source_count": len(profile.sources),
        "new_source_count": len(new_names),
        "inherited_source_count": len(profile.sources) - len(new_names),
        "embodiment_count": len(profile.embodiments),
        "SHA_scans": False,
        "training_ready": False,
    }
    _publish(args.receipt, (json.dumps(final_receipt, sort_keys=True, indent=2)+"\n").encode())
    print(json.dumps(final_receipt, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--admission-id")
    parser.add_argument("--inventory", action="append", default=[], metavar="SOURCE=RECEIPT")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    template = args.template.absolute()
    template_binding = None if args.admission_id is None else _bind(template, args.admission_id)
    value = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "wm3d_v8_data_profile_v4":
        raise RuntimeError("data profile template schema mismatch")
    if args.admission_id is not None:
        _verify(template, template_binding, args.admission_id)
        _materialize_admitted(args, template, value, template_binding)
        return
    sources = value.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("data profile template has no sources")
    receipts = _assignments(args.inventory)
    names = {str(source.get("name")) for source in sources if isinstance(source, dict)}
    if set(receipts) != names:
        raise RuntimeError(
            f"inventory receipt set must exactly match sources: "
            f"missing={sorted(names-set(receipts))} unknown={sorted(set(receipts)-names)}"
        )
    receipt_sha_by_source: dict[str, str] = {}
    for source in sources:
        if not isinstance(source, dict):
            raise RuntimeError("source entry must be a mapping")
        name = str(source["name"])
        receipt_path = receipts[name]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema") != "wm3d_v8_source_inventory_receipt_v1":
            raise RuntimeError(f"{name}: inventory receipt schema mismatch")
        if receipt.get("source") != name:
            raise RuntimeError(f"{name}: inventory receipt source mismatch")
        expected_template_sha = receipt.get("data_template_sha256")
        if expected_template_sha != sha256_file(template):
            raise RuntimeError(f"{name}: inventory was built from a different data template")
        for field in ("adapter_contract_sha256", "manifest_sha256"):
            if SHA_RE.fullmatch(str(receipt.get(field, ""))) is None:
                raise RuntimeError(f"{name}: receipt {field} is invalid")
        audit_receipt = Path(
            str(receipt.get("adapter_audit_receipt_path", ""))
        ).resolve(strict=True)
        if (
            sha256_file(audit_receipt)
            != receipt.get("adapter_audit_receipt_sha256")
        ):
            raise RuntimeError(f"{name}: adapter audit receipt SHA drift")
        audit = json.loads(audit_receipt.read_text(encoding="utf-8"))
        if (
            audit.get("schema") != "wm3d_v8_source_adapter_audit_receipt_v1"
            or audit.get("source") != name
            or audit.get("adapter_contract_sha256")
            != receipt["adapter_contract_sha256"]
            or audit.get("semantic_review") != "operator_confirmed_fail_closed"
        ):
            raise RuntimeError(f"{name}: adapter semantic audit is missing or mismatched")
        manifest = Path(str(receipt["manifest_path"])).resolve(strict=True)
        adapter = Path(str(receipt["adapter_contract_path"])).resolve(strict=True)
        if sha256_file(manifest) != receipt["manifest_sha256"]:
            raise RuntimeError(f"{name}: manifest SHA drift")
        if sha256_file(adapter) != receipt["adapter_contract_sha256"]:
            raise RuntimeError(f"{name}: adapter SHA drift")
        source["raw_root"] = str(Path(str(receipt["raw_root"])).resolve(strict=True))
        source["adapter_config"] = str(adapter)
        source["adapter_contract_sha256"] = str(receipt["adapter_contract_sha256"])
        source["manifest"] = str(manifest)
        source["manifest_sha256"] = str(receipt["manifest_sha256"])
        receipt_sha_by_source[name] = sha256_file(receipt_path)
    payload = yaml.safe_dump(value, sort_keys=False, allow_unicode=True).encode("utf-8")
    _publish(args.output, payload)
    output = args.output.absolute().resolve(strict=True)
    profile = load_data_profile(output, verify_source_manifests=True)
    final_receipt = {
        "schema": "wm3d_v8_materialized_data_profile_receipt_v1",
        "template_path": str(template),
        "template_sha256": sha256_file(template),
        "data_profile_path": str(output),
        "data_profile_sha256": profile.profile_sha256,
        "source_inventory_receipt_sha256_by_name": receipt_sha_by_source,
        "source_count": len(profile.sources),
        "embodiment_count": len(profile.embodiments),
    }
    _publish(
        args.receipt,
        (json.dumps(final_receipt, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    print(json.dumps(final_receipt, sort_keys=True))


if __name__ == "__main__":
    main()
