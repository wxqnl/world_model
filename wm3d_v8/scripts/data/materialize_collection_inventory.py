#!/usr/bin/env python3
"""Materialize one V8 source manifest from a sealed LeRobot collection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.data.materialize_source_inventory import (
    ADAPTER_AUDIT_SCHEMA,
    _embodiment_from_template,
    _publish,
)
from wm3d_v3.data.manifest_contract import canonical_sha256, sha256_file
from wm3d_v3.data.source_adapters import load_adapter_contract
from wm3d_v3.data.source_inventory import (
    INVENTORY_RECEIPT_SCHEMA,
    deterministic_split,
    scan_lerobot_source,
    validate_written_inventory,
)


COLLECTION_SCHEMA = "wm3d_v8_lerobot_collection_receipt_v1"


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve(strict=True)


def _root(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{label} must be a real directory: {path}")
    return path.resolve(strict=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-template", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--collection-receipt", type=Path, required=True)
    parser.add_argument("--adapter-contract", type=Path, required=True)
    parser.add_argument("--adapter-contract-sha256", required=True)
    parser.add_argument("--adapter-audit-receipt", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=3407)
    parser.add_argument("--train-fraction", type=float, default=0.98)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--default-task", default="")
    args = parser.parse_args()

    collection_root = _root(args.collection_root, "collection root")
    collection_receipt_path = _regular(args.collection_receipt, "collection receipt")
    collection = json.loads(collection_receipt_path.read_text(encoding="utf-8"))
    roots = tuple(_root(Path(str(item)), "LeRobot root") for item in collection.get("lerobot_roots", []))
    if (
        not isinstance(collection, dict)
        or collection.get("schema") != COLLECTION_SCHEMA
        or not roots
        or len(roots) != len(set(roots))
        or int(collection.get("lerobot_root_count", -1)) != len(roots)
        or any(collection_root not in root.parents for root in roots)
    ):
        raise RuntimeError("collection receipt does not bind a closed LeRobot collection")

    embodiment, template_sha = _embodiment_from_template(args.data_template, args.source)
    adapter = load_adapter_contract(
        args.adapter_contract,
        expected_sha256=args.adapter_contract_sha256,
    )
    audit_path = _regular(args.adapter_audit_receipt, "adapter audit receipt")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("schema") != ADAPTER_AUDIT_SCHEMA
        or audit.get("source") != args.source
        or audit.get("adapter_contract_sha256") != adapter.sha256
        or audit.get("data_template_sha256") != template_sha
        or audit.get("upstream_receipt_sha256") != sha256_file(collection_receipt_path)
        or int(audit.get("roots_audited", -1)) != len(roots)
        or audit.get("structural_checks") != "pass"
        or audit.get("semantic_review") != "operator_confirmed_fail_closed"
    ):
        raise RuntimeError("adapter audit does not authorize this collection inventory")

    rows: list[dict] = []
    child_receipts: list[dict] = []
    for child in sorted(roots):
        relative_root = child.relative_to(collection_root)
        namespace = hashlib.sha256(relative_root.as_posix().encode()).hexdigest()[:16]
        child_rows, child_receipt = scan_lerobot_source(
            root=child,
            source=args.source,
            embodiment=embodiment,
            adapter=adapter,
            split_seed=args.split_seed,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            default_task=args.default_task,
        )
        for raw in child_rows:
            row = dict(raw)
            old_identity = str(row["episode_id"])
            suffix = old_identity.rsplit(":", 1)[-1]
            identity = f"{args.source}:{namespace}:{suffix}"
            row["episode_id"] = identity
            row["split"] = deterministic_split(
                args.source,
                identity,
                seed=args.split_seed,
                train_fraction=args.train_fraction,
                validation_fraction=args.validation_fraction,
            )
            row["payload"] = (relative_root / str(row["payload"])).as_posix()
            row["assets"] = [
                {**asset, "path": (relative_root / str(asset["path"])).as_posix()}
                for asset in row["assets"]
            ]
            rows.append(row)
        child_receipts.append(
            {
                **child_receipt,
                "raw_root": relative_root.as_posix(),
                "episode_namespace": namespace,
            }
        )

    rows.sort(key=lambda row: str(row["episode_id"]))
    identities = {str(row["episode_id"]) for row in rows}
    if not rows or len(identities) != len(rows):
        raise RuntimeError("collection inventory is empty or has duplicate episode identities")
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    _publish(args.output_manifest, payload)
    manifest = args.output_manifest.absolute().resolve(strict=True)
    validation = validate_written_inventory(
        manifest,
        source=args.source,
        embodiment=embodiment,
    )
    receipt = {
        "schema": INVENTORY_RECEIPT_SCHEMA,
        "source": args.source,
        "raw_root": str(collection_root),
        "collection_receipt_path": str(collection_receipt_path),
        "collection_receipt_sha256": sha256_file(collection_receipt_path),
        "data_template_path": str(args.data_template.resolve(strict=True)),
        "data_template_sha256": template_sha,
        "adapter_contract_path": str(adapter.path),
        "adapter_contract_sha256": adapter.sha256,
        "adapter_audit_receipt_path": str(audit_path),
        "adapter_audit_receipt_sha256": sha256_file(audit_path),
        "episode_count": len(rows),
        "split_count": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "val", "test")
        },
        "duration_s": sum(float(row["duration_s"]) for row in rows),
        "canonical_rows_sha256": canonical_sha256(rows),
        "child_inventory_receipts_sha256": canonical_sha256(child_receipts),
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "manifest_validation": validation,
    }
    _publish(
        args.output_receipt,
        (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
