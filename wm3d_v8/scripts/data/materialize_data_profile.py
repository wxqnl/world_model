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

from wm3d_v3.data.manifest_contract import load_data_profile, sha256_file


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--inventory", action="append", default=[], metavar="SOURCE=RECEIPT")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    template = args.template.resolve(strict=True)
    value = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "wm3d_v8_data_profile_v4":
        raise RuntimeError("data profile template schema mismatch")
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
