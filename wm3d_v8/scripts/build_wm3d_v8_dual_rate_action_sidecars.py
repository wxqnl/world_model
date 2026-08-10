#!/usr/bin/env python3
"""Build action-only 20 Hz sidecars from sealed V8 RoboCasa archives.

This command never recomputes or mutates VGGT/RGB/depth/point artifacts.  It
extracts the already-stored raw controller stream, applies the exact pinned
partition adapter, proves four controller commands reproduce every sealed
5 Hz action, and publishes no-clobber sidecars plus train-only statistics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wm3d_v3.data.v7_action_contract import (
    ActionAdapter,
    canonicalize_dense_action,
    resample_canonical_actions,
)
from wm3d_v3.data.v8_action_contract import (
    ACTION_DIM,
    POLICY_HZ,
    V8_ACTION_SIDECAR_INDEX_SCHEMA,
    V8_ACTION_SIDECAR_SCHEMA,
    V8_ACTION_STATS_SCHEMA,
)


LOWER_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _parse_bindings(values: list[str], *, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        key, item = key.strip(), item.strip()
        if not separator or not key or not item or key in result:
            raise ValueError(f"invalid or duplicate {label} binding {value!r}")
        result[key] = item
    return result


def _load_adapters(
    paths: dict[str, str], expected_sha256: dict[str, str]
) -> tuple[dict[str, ActionAdapter], dict[str, str]]:
    if set(paths) != set(expected_sha256):
        raise ValueError(
            "adapter audit path/SHA keys differ: "
            f"paths={sorted(paths)} sha={sorted(expected_sha256)}"
        )
    adapters: dict[str, ActionAdapter] = {}
    digests: dict[str, str] = {}
    for source in sorted(paths):
        path = Path(paths[source]).resolve()
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"adapter audit is missing/not regular: {path}")
        observed = sha256_file(path)
        if observed != expected_sha256[source]:
            raise RuntimeError(
                f"adapter audit digest mismatch for {source}: "
                f"observed={observed} expected={expected_sha256[source]}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not bool((payload.get("factual_action_audit") or {}).get("passed")):
            raise RuntimeError(f"adapter factual audit did not pass: {path}")
        adapter = ActionAdapter(**payload["adapter"])
        if float(adapter.nominal_hz) != float(POLICY_HZ):
            raise RuntimeError(
                f"{source} adapter nominal_hz={adapter.nominal_hz} != {POLICY_HZ}"
            )
        adapters[source] = adapter
        digests[source] = observed
    return adapters, digests


def _load_index_rows(paths: Iterable[Path]) -> list[dict]:
    rows: dict[str, dict] = {}
    origins: dict[str, str] = {}
    for index_path in paths:
        index_path = Path(index_path).resolve()
        if index_path.is_symlink() or not index_path.is_file():
            raise FileNotFoundError(index_path)
        with index_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                clip_hash = str(row.get("clip_hash", ""))
                if len(clip_hash) != 64:
                    raise RuntimeError(
                        f"invalid clip_hash {index_path}:{line_number}: {clip_hash!r}"
                    )
                if clip_hash in rows:
                    raise RuntimeError(
                        "duplicate clip is forbidden across action sidecar inputs: "
                        f"clip={clip_hash} first={origins[clip_hash]} "
                        f"second={index_path}:{line_number}"
                    )
                rows[clip_hash] = row
                origins[clip_hash] = f"{index_path}:{line_number}"
    if not rows:
        raise RuntimeError("input action sidecar indices are empty")
    return [rows[key] for key in sorted(rows)]


class _RunningPoseStats:
    def __init__(self) -> None:
        self.count = 0
        self.mean = np.zeros(6, dtype=np.float64)
        self.m2 = np.zeros(6, dtype=np.float64)

    def update(self, pose: np.ndarray) -> None:
        values = np.asarray(pose, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all():
            raise RuntimeError(f"invalid pose stats input: {values.shape}")
        if len(values) == 0:
            return
        batch_count = len(values)
        batch_mean = values.mean(axis=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(axis=0)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = self.m2 + batch_m2 + np.square(delta) * (
            self.count * batch_count / total
        )
        self.count = total

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count <= 0:
            raise RuntimeError("train split contributed no controller actions")
        variance = self.m2 / max(1, self.count)
        std = np.sqrt(np.maximum(variance, 1.0e-12))
        return self.mean.astype(np.float32), std.astype(np.float32)


def _arrays_equal(left: dict[str, np.ndarray], right_path: Path) -> bool:
    try:
        with np.load(right_path, allow_pickle=False) as archive:
            if set(archive.files) != set(left):
                return False
            return all(np.array_equal(np.asarray(archive[key]), value) for key, value in left.items())
    except Exception:
        return False


def _publish_npz_no_clobber(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not _arrays_equal(arrays, path):
            raise FileExistsError(f"existing sidecar is non-identical: {path}")
        return sha256_file(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not _arrays_equal(arrays, path):
                raise FileExistsError(f"raced with non-identical sidecar: {path}")
        return sha256_file(path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_bytes_no_clobber(path: Path, payload: bytes) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"existing file is non-identical: {path}")
        return sha256_file(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(f"raced with non-identical file: {path}")
        return sha256_file(path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_one(
    row: dict,
    *,
    adapter: ActionAdapter,
    adapter_sha256: str,
    output_root: Path,
    composition_atol: float,
    dry_run: bool,
) -> tuple[dict, np.ndarray]:
    source = str(row.get("v7_source", ""))
    clip_hash = str(row["clip_hash"])
    archive_path = Path(row["path"]).resolve()
    if archive_path.is_symlink() or not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    expected_archive_sha256 = str(row.get("artifact_sha256", ""))
    if LOWER_HEX64.fullmatch(expected_archive_sha256) is None:
        raise RuntimeError(
            f"source archive SHA256 is missing/invalid for {clip_hash}: "
            f"{expected_archive_sha256!r}"
        )
    observed_archive_sha256 = sha256_file(archive_path)
    if observed_archive_sha256 != expected_archive_sha256:
        raise RuntimeError(
            f"source archive digest mismatch for {clip_hash}: "
            f"observed={observed_archive_sha256} "
            f"expected={expected_archive_sha256}"
        )
    if str(row.get("action_audit_sha256", "")) != adapter_sha256:
        raise RuntimeError(f"row adapter digest mismatch for {clip_hash}")
    with np.load(archive_path, allow_pickle=False) as archive:
        for key, expected in (
            ("clip_hash", clip_hash),
            ("split", row["split"]),
            ("source", row["source"]),
            ("action_audit_sha256", adapter_sha256),
        ):
            if str(np.asarray(archive[key]).item()) != str(expected):
                raise RuntimeError(f"archive identity mismatch {key}: {clip_hash}")
        if float(np.asarray(archive["source_control_hz"]).item()) != float(POLICY_HZ):
            raise RuntimeError(f"source control rate is not 20 Hz: {clip_hash}")
        if float(np.asarray(archive["model_control_hz"]).item()) != 5.0:
            raise RuntimeError(f"world control rate is not 5 Hz: {clip_hash}")
        raw = np.asarray(archive["raw_actions"], dtype=np.float32)
        world = np.asarray(archive["actions"], dtype=np.float32)
        native_indices = np.asarray(archive["native_frame_indices"], dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] < 12 or not np.isfinite(raw).all():
        raise RuntimeError(f"raw RoboCasa action must be finite [N,>=12]: {clip_hash}")
    if world.ndim != 2 or world.shape[1] != ACTION_DIM or not np.isfinite(world).all():
        raise RuntimeError(f"sealed world action must be finite [M,7]: {clip_hash}")
    if native_indices.shape != (len(raw),):
        raise RuntimeError(f"native action frame identity mismatch: {clip_hash}")
    dense7 = np.concatenate((raw[:, 5:11], raw[:, 11:12]), axis=1)
    fine = canonicalize_dense_action(dense7, adapter)
    used_count = len(world) * 4
    if len(fine) < used_count or len(fine) - used_count >= 4:
        raise RuntimeError(
            f"20/5 Hz cardinality mismatch fine={len(fine)} world={len(world)}: {clip_hash}"
        )
    fine = fine[:used_count]
    composed = resample_canonical_actions(fine, source_hz=20.0, target_hz=5.0)
    if not np.allclose(composed, world, rtol=0.0, atol=composition_atol):
        max_error = np.max(np.abs(composed - world), axis=0)
        raise RuntimeError(
            f"20 Hz composition mismatch {clip_hash}: {max_error.tolist()}"
        )
    sidecar_path = (output_root / str(row["split"]) / f"{clip_hash}.npz").resolve()
    arrays = {
        "schema": np.asarray(V8_ACTION_SIDECAR_SCHEMA),
        "clip_hash": np.asarray(clip_hash),
        "split": np.asarray(str(row["split"])),
        "source": np.asarray(str(row["source"])),
        "v7_source": np.asarray(source),
        "source_archive_path": np.asarray(str(archive_path)),
        "source_archive_sha256": np.asarray(observed_archive_sha256),
        "action_audit_sha256": np.asarray(adapter_sha256),
        "adapter_version": np.asarray(adapter.adapter_version),
        "policy_hz": np.asarray(POLICY_HZ, dtype=np.int64),
        "world_hz": np.asarray(5, dtype=np.int64),
        "fine_actions": fine.astype(np.float32),
        "native_frame_indices": native_indices[:used_count].astype(np.int64),
        "dropped_tail_actions": np.asarray(len(raw) - used_count, dtype=np.int64),
        "composition_max_abs_by_dim": np.max(
            np.abs(composed - world), axis=0
        ).astype(np.float32),
    }
    artifact_sha256 = (
        "dry-run"
        if dry_run
        else _publish_npz_no_clobber(sidecar_path, arrays)
    )
    output_row = {
        "schema": V8_ACTION_SIDECAR_INDEX_SCHEMA,
        "clip_hash": clip_hash,
        "split": str(row["split"]),
        "source": str(row["source"]),
        "v7_source": source,
        "path": str(sidecar_path),
        "artifact_sha256": artifact_sha256,
        "source_archive_path": str(archive_path),
        "source_archive_sha256": observed_archive_sha256,
        "action_audit_sha256": adapter_sha256,
        "fine_action_count": int(len(fine)),
        "world_action_count": int(len(world)),
        "dropped_tail_actions": int(len(raw) - used_count),
    }
    return output_row, fine[:, :6]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-index", action="append", required=True, type=Path)
    parser.add_argument(
        "--adapter-audit",
        action="append",
        default=[],
        metavar="PARTITION=PATH",
        required=True,
    )
    parser.add_argument(
        "--adapter-audit-sha256",
        action="append",
        default=[],
        metavar="PARTITION=SHA256",
        required=True,
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-index", required=True, type=Path)
    parser.add_argument("--output-stats", required=True, type=Path)
    parser.add_argument("--composition-atol", type=float, default=2.0e-5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    audit_paths = _parse_bindings(args.adapter_audit, label="adapter-audit")
    audit_sha = _parse_bindings(
        args.adapter_audit_sha256, label="adapter-audit-sha256"
    )
    adapters, adapter_digests = _load_adapters(audit_paths, audit_sha)
    rows = _load_index_rows(args.input_index)
    unknown = sorted({str(row.get("v7_source", "")) for row in rows} - set(adapters))
    if unknown:
        raise RuntimeError(f"input rows have no pinned adapter: {unknown}")

    stats = _RunningPoseStats()
    output_rows: list[dict] = []
    for position, row in enumerate(rows, 1):
        source = str(row["v7_source"])
        output_row, pose = _build_one(
            row,
            adapter=adapters[source],
            adapter_sha256=adapter_digests[source],
            output_root=args.output_root,
            composition_atol=float(args.composition_atol),
            dry_run=bool(args.dry_run),
        )
        output_rows.append(output_row)
        if str(row["split"]) == "train":
            stats.update(pose)
        if position == 1 or position % 1000 == 0 or position == len(rows):
            print(
                f"[v8-action-sidecar] {position}/{len(rows)} "
                f"clip={row['clip_hash']} split={row['split']} source={source}",
                flush=True,
            )

    mean, std = stats.arrays()
    input_index_digests = {
        str(Path(path).resolve()): sha256_file(Path(path).resolve())
        for path in args.input_index
    }
    stats_arrays = {
        "schema": np.asarray(V8_ACTION_STATS_SCHEMA),
        "split": np.asarray("train"),
        "mean": mean,
        "std": std,
        "count": np.asarray(stats.count, dtype=np.int64),
        "world_hz": np.asarray(5, dtype=np.int64),
        "policy_hz": np.asarray(POLICY_HZ, dtype=np.int64),
        "input_indices_json": np.asarray(
            json.dumps(input_index_digests, sort_keys=True, separators=(",", ":"))
        ),
        "adapter_audits_json": np.asarray(
            json.dumps(adapter_digests, sort_keys=True, separators=(",", ":"))
        ),
    }
    index_payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in output_rows
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema": V8_ACTION_SIDECAR_INDEX_SCHEMA,
                    "dry_run": True,
                    "rows": len(output_rows),
                    "train_action_count": stats.count,
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                },
                sort_keys=True,
            )
        )
        return
    stats_sha = _publish_npz_no_clobber(args.output_stats.resolve(), stats_arrays)
    index_sha = _publish_bytes_no_clobber(args.output_index.resolve(), index_payload)
    print(
        json.dumps(
            {
                "schema": V8_ACTION_SIDECAR_INDEX_SCHEMA,
                "dry_run": False,
                "rows": len(output_rows),
                "train_action_count": stats.count,
                "index": str(args.output_index.resolve()),
                "index_sha256": index_sha,
                "stats": str(args.output_stats.resolve()),
                "stats_sha256": stats_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
