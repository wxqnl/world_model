#!/usr/bin/env python3
"""Close and merge the formal 16-shard V8 Stage0 causal cache.

This command never repairs, replaces, or deletes cache artifacts.  It verifies
the immutable producer receipts and shard indices, then publishes deterministic
merged indices with no-clobber semantics.  Full NPZ SHA/schema validation is
performed by the subsequent formal seal/preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OXE_SCHEMA = "wm3d_v8_stage0_causal_dual_view_oxe_producer_v1"
ROBO_SCHEMA = "wm3d_v8_stage0_causal_dual_view_robocasa_producer_v1"
ROW_SCHEMA = "wm3d_v8_stage0_causal_dual_view_v1"
REPRESENTATION = "wm3d_v8_vggt_observed_context_target_split_v1"
CODEC_SHA256 = "a8b0103e936fdb97658a5325400a996452d9896685e9a5af1618ba38ada34470"
CODEC_GATE_SHA256 = "1880f19c5a03a39f3b9e593d2f5bbf84582b45ced749c7b8b6ff3a287ca9dd21"
VGGT_REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"
CONFIRM = "EXECUTE_WM3D_V8_STAGE0_WORLD16_FINALIZE"
REPORT_SCHEMA = "wm3d_v8_stage0_causal_dual_view_world16_closure_v1"


@dataclass(frozen=True)
class FamilySpec:
    name: str
    kind: str
    shards: int
    global_count: int
    manifest_sha256: str
    source: str | None = None
    split: str | None = None
    partition: str | None = None


FAMILIES = (
    FamilySpec(
        "oxe_droid_action_train",
        "oxe",
        16,
        130458,
        "c8abc8ffc6d0da2e60760254bbfcb50a4dcf3eaa26b06f6482fc853ca8432a95",
        source="oxe_droid_action",
        split="train",
    ),
    FamilySpec(
        "oxe_droid_action_val",
        "oxe",
        16,
        4044,
        "c8abc8ffc6d0da2e60760254bbfcb50a4dcf3eaa26b06f6482fc853ca8432a95",
        source="oxe_droid_action",
        split="val",
    ),
    FamilySpec(
        "oxe_bridge_action_train",
        "oxe",
        16,
        26037,
        "8be14e8797ee73cf54d036944ab351bd83f703bf4eb0be1979d996cad3fe059e",
        source="oxe_bridge_action",
        split="train",
    ),
    FamilySpec(
        "oxe_bridge_action_val",
        "oxe",
        16,
        799,
        "8be14e8797ee73cf54d036944ab351bd83f703bf4eb0be1979d996cad3fe059e",
        source="oxe_bridge_action",
        split="val",
    ),
    FamilySpec(
        "robocasa_atomic",
        "robocasa",
        14,
        6548,
        "0030cfa88f33eed3d96ca3138fd1f7ca3d3b3de16a2f8ec770998bbc27444585",
        partition="atomic",
    ),
    FamilySpec(
        "robocasa_composite",
        "robocasa",
        16,
        57879,
        "a4fd5c8b3b9523e6ee68118b6fd567217e8b87d8ad7d1c7213508251e98ad329",
        partition="composite",
    ),
    FamilySpec(
        "robocasa_mg",
        "robocasa",
        16,
        40000,
        "a3963c87805519ea110b15820630daabbfbcebe1a03efa15a6c9e49b8924aec8",
        partition="mg",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _publish_no_clobber(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = _sha256_bytes(data)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"existing output is non-identical: {path}")
        return digest
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise FileExistsError(f"concurrent output is non-identical: {path}")
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: row is not a mapping")
        rows.append(row)
    return rows


def _report_paths(
    manifest_dir: Path, spec: FamilySpec, shard: int
) -> tuple[Path, Path]:
    suffix = f"shard-{shard:05d}-of-{spec.shards:05d}"
    index = manifest_dir / f"{spec.name}.{suffix}.jsonl"
    return index, index.with_suffix(".report.json")


def _validate_common_row(path: Path, number: int, row: dict[str, Any]) -> str:
    label = f"{path}:{number}"
    expected = {
        "schema": ROW_SCHEMA,
        "representation": REPRESENTATION,
        "context_future_leakage": False,
        "target_usage": "supervision_only",
        "geometry_coordinate_frame": "first_observed_camera",
        "T": 16,
        "k": 8,
        "P": 64,
        "token_D": 2048,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"{label}: {key}={row.get(key)!r}, expected {value!r}")
    artifact = str(row.get("path") or "")
    if not artifact or not artifact.endswith(".npz"):
        raise ValueError(f"{label}: invalid artifact path")
    digest = str(row.get("artifact_sha256") or "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{label}: invalid artifact SHA256")
    return artifact


def collect_family(
    manifest_dir: Path, spec: FamilySpec
) -> tuple[bytes, dict[str, Any], set[str]]:
    merged: list[dict[str, Any]] = []
    identities: set[tuple[Any, ...]] = set()
    artifacts: set[str] = set()
    report_evidence: list[dict[str, Any]] = []
    selection_digests: set[str] = set()
    config_digests: set[str] = set()
    total = 0
    for shard in range(spec.shards):
        index, report_path = _report_paths(manifest_dir, spec, shard)
        if not index.is_file() or not report_path.is_file():
            raise FileNotFoundError(f"missing shard output: {index} / {report_path}")
        report = json.loads(report_path.read_text())
        expected_schema = OXE_SCHEMA if spec.kind == "oxe" else ROBO_SCHEMA
        if report.get("schema") != expected_schema or report.get("pass") is not True:
            raise ValueError(f"{report_path}: producer receipt did not pass")
        if report.get("manifest_sha256") != spec.manifest_sha256:
            raise ValueError(f"{report_path}: source manifest SHA mismatch")
        if report.get("codec_sha256") != CODEC_SHA256:
            raise ValueError(f"{report_path}: codec SHA mismatch")
        if report.get("representation") != REPRESENTATION:
            raise ValueError(f"{report_path}: representation mismatch")
        if report.get("selection_sha256"):
            selection_digests.add(str(report["selection_sha256"]))
        config_digest = str(report.get("config_sha256") or "")
        if len(config_digest) != 64 or any(
            c not in "0123456789abcdef" for c in config_digest
        ):
            raise ValueError(f"{report_path}: invalid producer config SHA")
        config_digests.add(config_digest)
        if spec.kind == "oxe":
            if report.get("source") != spec.source or report.get("split") != spec.split:
                raise ValueError(f"{report_path}: OXE source/split mismatch")
            if (
                int(report.get("shard_id", -1)) != shard
                or int(report.get("num_shards", -1)) != spec.shards
            ):
                raise ValueError(f"{report_path}: OXE shard identity mismatch")
            if int(report.get("selected_global", -1)) != spec.global_count:
                raise ValueError(f"{report_path}: OXE global count mismatch")
            count = int(report.get("encoded", -1))
            if int(report.get("selected_shard", -2)) != count:
                raise ValueError(f"{report_path}: selected/encoded mismatch")
            if report.get("vggt_revision") != VGGT_REVISION:
                raise ValueError(f"{report_path}: VGGT revision mismatch")
        else:
            sharding = report.get("sharding") or {}
            teacher = report.get("geometry_teacher") or {}
            if report.get("v7_source") != spec.partition:
                raise ValueError(f"{report_path}: partition mismatch")
            if (
                int(sharding.get("shard_id", -1)) != shard
                or int(sharding.get("num_shards", -1)) != spec.shards
            ):
                raise ValueError(f"{report_path}: RoboCasa shard identity mismatch")
            if int(report.get("global_selected_clips", -1)) != spec.global_count:
                raise ValueError(f"{report_path}: global clip count mismatch")
            count = int(report.get("clips", -1))
            if report.get("causal_dual_view") is not True:
                raise ValueError(f"{report_path}: causal dual-view receipt is false")
            if report.get("codec_downstream_report_sha256") != CODEC_GATE_SHA256:
                raise ValueError(f"{report_path}: codec downstream gate SHA mismatch")
            if int(sharding.get("assigned_clips", -2)) != count:
                raise ValueError(f"{report_path}: assigned/encoded clip mismatch")
            if (
                report.get("paired_views") is not True
                or report.get("task_embedding_real") is not True
            ):
                raise ValueError(f"{report_path}: paired/task contract mismatch")
            if report.get("rgb_sidecar_coverage_passed") is not True:
                raise ValueError(f"{report_path}: RGB sidecar coverage failed")
            if report.get("factual_action_audit_passed") is not True:
                raise ValueError(f"{report_path}: factual action audit failed")
            if teacher.get("revision") != VGGT_REVISION:
                raise ValueError(f"{report_path}: VGGT revision mismatch")
            if teacher.get("pseudo_teacher") is not True:
                raise ValueError(f"{report_path}: VGGT teacher contract mismatch")
        observed_index_sha = sha256_file(index)
        if observed_index_sha != report.get("index_sha256"):
            raise ValueError(f"{report_path}: index SHA mismatch")
        rows = _read_jsonl(index)
        if len(rows) != count:
            raise ValueError(f"{index}: rows={len(rows)}, receipt count={count}")
        for number, row in enumerate(rows, 1):
            artifact = _validate_common_row(index, number, row)
            if artifact in artifacts:
                raise ValueError(f"{index}:{number}: duplicate artifact path")
            artifacts.add(artifact)
            if spec.kind == "oxe":
                if (
                    row.get("source") != spec.source
                    or row.get("split") != spec.split
                    or row.get("paired_views") is not False
                ):
                    raise ValueError(f"{index}:{number}: OXE row identity mismatch")
                identity = (row.get("clip_id"), row.get("start"))
                if not identity[0] or not isinstance(identity[1], int):
                    raise ValueError(f"{index}:{number}: invalid OXE identity")
            else:
                if (
                    row.get("v7_source") != spec.partition
                    or row.get("paired_views") is not True
                ):
                    raise ValueError(
                        f"{index}:{number}: RoboCasa row identity mismatch"
                    )
                identity = (row.get("clip_hash"),)
                if not identity[0]:
                    raise ValueError(f"{index}:{number}: invalid clip hash")
            if identity in identities:
                raise ValueError(
                    f"{index}:{number}: duplicate row identity {identity!r}"
                )
            identities.add(identity)
            merged.append(row)
        total += count
        report_evidence.append(
            {
                "path": str(report_path.resolve()),
                "sha256": sha256_file(report_path),
                "index_path": str(index.resolve()),
                "index_sha256": observed_index_sha,
                "count": count,
                "shard_id": shard,
            }
        )
    if total != spec.global_count or len(merged) != spec.global_count:
        raise ValueError(
            f"{spec.name}: closure count={total}, expected={spec.global_count}"
        )
    if len(selection_digests) != 1:
        raise ValueError(f"{spec.name}: selection SHA is not uniform")
    if len(config_digests) != 1:
        raise ValueError(f"{spec.name}: producer config SHA is not uniform")
    if spec.kind == "oxe":
        merged.sort(key=lambda row: (str(row["clip_id"]), int(row["start"])))
    else:
        merged.sort(key=lambda row: (str(row["split"]), str(row["clip_hash"])))
    encoded = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in merged
    ).encode()
    evidence = {
        "kind": spec.kind,
        "expected_count": spec.global_count,
        "actual_count": total,
        "shards": spec.shards,
        "selection_sha256": next(iter(selection_digests)),
        "producer_config_sha256": next(iter(config_digests)),
        "merged_index_sha256": _sha256_bytes(encoded),
        "producer_reports": report_evidence,
    }
    return encoded, evidence, artifacts


def evaluate(
    manifest_dir: Path, output_dir: Path
) -> tuple[dict[str, Any], dict[Path, bytes]]:
    families: dict[str, Any] = {}
    outputs: dict[Path, bytes] = {}
    all_artifacts: set[str] = set()
    for spec in FAMILIES:
        data, evidence, artifacts = collect_family(manifest_dir, spec)
        overlap = all_artifacts & artifacts
        if overlap:
            raise ValueError(
                f"{spec.name}: artifact reused across source families: "
                f"{sorted(overlap)[:3]}"
            )
        all_artifacts.update(artifacts)
        output = (output_dir / f"{spec.name}.jsonl").resolve()
        evidence["merged_index_path"] = str(output)
        families[spec.name] = evidence
        outputs[output] = data
    expected_total = sum(spec.global_count for spec in FAMILIES)
    if len(all_artifacts) != expected_total:
        raise ValueError(
            f"global artifact closure={len(all_artifacts)}, expected={expected_total}"
        )
    report = {
        "schema": REPORT_SCHEMA,
        "pass": True,
        "artifact_count": len(all_artifacts),
        "expected_artifact_count": expected_total,
        "families": families,
    }
    return report, outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mode", choices=("dry-run", "execute"), required=True)
    args = parser.parse_args()

    report, outputs = evaluate(args.manifest_dir.resolve(), args.output_dir.resolve())
    report["mode"] = args.mode
    report["mutated"] = False
    if args.mode == "execute":
        if os.environ.get("WM3D_V8_STAGE0_WORLD16_FINALIZE") != CONFIRM:
            raise SystemExit(f"set WM3D_V8_STAGE0_WORLD16_FINALIZE={CONFIRM}")
        for path, data in outputs.items():
            observed = _publish_no_clobber(path, data)
            if observed != report["families"][path.stem]["merged_index_sha256"]:
                raise RuntimeError(f"published index digest mismatch: {path}")
        report["mutated"] = True
        payload = json.dumps(report, indent=2, sort_keys=True).encode() + b"\n"
        _publish_no_clobber(args.report.resolve(), payload)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
