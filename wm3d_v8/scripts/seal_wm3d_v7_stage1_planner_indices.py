#!/usr/bin/env python3
"""Deterministically merge and seal sharded V7 Stage1-P indices."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SCHEMAS = {
    "candidates": "wm3d_v7_stage1_planner_candidates_v1",
    "runtime": "wm3d_v7_stage1_planner_same_root_runtime_v1",
    "cache": "wm3d_v7_stage1_planner_branch_compact_v2",
}
ROLES = {
    "candidates": (
        "direct", "flow_0", "flow_1", "flow_2", "flow_3",
        "grip_open", "grip_close", "arm_hold", "pose_reverse", "pose_half",
    ),
    "runtime": (
        "factual_teacher", "direct", "flow_0", "flow_1", "flow_2", "flow_3",
        "grip_open", "grip_close", "arm_hold", "pose_reverse", "pose_half",
    ),
    "cache": (
        "factual_teacher", "direct", "flow_0", "flow_1", "flow_2", "flow_3",
        "grip_open", "grip_close", "arm_hold", "pose_reverse", "pose_half",
    ),
}


def sha256_file(path: Path, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=tuple(SCHEMAS), required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-roots", type=int, default=0)
    args = parser.parse_args()
    if len(set(args.input)) != len(args.input):
        raise SystemExit("duplicate input index")
    rows = []
    roots: set[str] = set()
    source_shas: set[str] = set()
    split_groups = {split: set() for split in ("train", "val", "test")}
    path_key = "candidate_path" if args.kind == "candidates" else "path"
    for input_path in args.input:
        if not input_path.is_file() or input_path.name.endswith(".partial"):
            raise RuntimeError(f"unsealed input index: {input_path}")
        for line_number, line in enumerate(input_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            root_id = str(row.get("root_id", ""))
            if row.get("schema") != SCHEMAS[args.kind]:
                raise RuntimeError(f"{input_path}:{line_number}: schema mismatch")
            if not root_id or root_id in roots:
                raise RuntimeError(f"blank/duplicate root_id: {root_id!r}")
            roots.add(root_id)
            if tuple(row.get("branch_roles") or ()) != ROLES[args.kind]:
                raise RuntimeError(f"branch role mismatch: {root_id}")
            payload = Path(str(row.get(path_key, "")))
            if not payload.is_file():
                raise FileNotFoundError(payload)
            split = str(row.get("split", ""))
            if split not in split_groups or not row.get("split_group"):
                raise RuntimeError(f"split provenance mismatch: {root_id}")
            split_groups[split].add(str(row["split_group"]))
            source_sha = str(row.get("stage0_checkpoint_sha256", ""))
            if len(source_sha) != 64:
                raise RuntimeError(f"Stage0 checkpoint SHA missing: {root_id}")
            source_shas.add(source_sha)
            if len(str(row.get("root_context_sha256", ""))) != 64:
                raise RuntimeError(f"root-context SHA missing: {root_id}")
            if row.get("future_observation_leakage") is not False:
                raise RuntimeError(f"future leakage contract missing: {root_id}")
            if args.kind in {"runtime", "cache"}:
                if row.get("same_root_current_runtime_exact") is not True:
                    raise RuntimeError(f"same-root exactness missing: {root_id}")
                if row.get("pseudo_outcomes") is not False:
                    raise RuntimeError(f"pseudo outcomes forbidden: {root_id}")
            if args.kind == "cache" and row.get("all_branch_native_geometry") is not True:
                raise RuntimeError(f"all-branch geometry missing: {root_id}")
            if args.kind == "cache" and row.get("context_source") != (
                "current_pinned_robocasa_runtime_causal_replay"
            ):
                raise RuntimeError(f"real causal T16 context missing: {root_id}")
            rows.append(row)
    if len(source_shas) != 1:
        raise RuntimeError(f"shards bind to different Stage0 checkpoints: {source_shas}")
    if args.expected_roots > 0 and len(rows) != args.expected_roots:
        raise RuntimeError(f"root count {len(rows)} != expected {args.expected_roots}")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_groups[left] & split_groups[right]
        if overlap:
            raise RuntimeError(f"split_group leakage {left}/{right}: {sorted(overlap)[:4]}")
    rows.sort(key=lambda row: (row["split"], row["task"], row["root_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    report = {
        "schema": "wm3d_v7_stage1_planner_index_seal_v1",
        "kind": args.kind,
        "roots": len(rows),
        "splits": {split: sum(row["split"] == split for row in rows) for split in split_groups},
        "stage0_checkpoint_sha256": next(iter(source_shas)),
        "input_indices": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in args.input
        ],
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "passed": True,
    }
    args.output.with_suffix(".seal.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
