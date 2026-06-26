#!/usr/bin/env python3
"""Plan node-local cache shards for v5 Stage0 NVMe training.

Each node gets whole clips, not random windows. That keeps RGB/action/depth
base files and window-native3D shards aligned on the same local NVMe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _safe(cid: str) -> str:
    return cid.replace("/", "__")


def _n_windows(n_frames: int, T: int, k: int, stride: int) -> int:
    win = int(T) + int(k)
    if int(n_frames) < win:
        return 0
    return (int(n_frames) - win) // int(stride) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--nodes", default="node43,node41,node44,node42")
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()

    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
    if not nodes:
        raise ValueError("--nodes cannot be empty")
    records = []
    with args.manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            nw = _n_windows(int(rec.get("n_frames") or 0), args.T, args.k, args.stride)
            if nw <= 0:
                continue
            records.append((nw, int(rec.get("n_frames") or 0), rec))

    buckets = [{"node": node, "windows": 0, "frames": 0, "records": []} for node in nodes]
    for nw, nf, rec in sorted(records, key=lambda x: (x[0], x[1]), reverse=True):
        bucket = min(buckets, key=lambda b: (b["windows"], b["frames"]))
        bucket["windows"] += int(nw)
        bucket["frames"] += int(nf)
        bucket["records"].append(rec)

    args.out.mkdir(parents=True, exist_ok=True)
    summary_lines = ["node\tclips\tframes\twindows\n"]
    for bucket in buckets:
        node = str(bucket["node"])
        manifest_path = args.out / f"{node}.manifest.jsonl"
        filelist_path = args.out / f"{node}.base_files.null"
        with manifest_path.open("w", encoding="utf-8") as mf, filelist_path.open("wb") as fl:
            for rec in bucket["records"]:
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cid = _safe(str(rec["clip_id"]))
                for rel in (
                    f"actions/{cid}.npy",
                    f"rgb_256/{cid}.npy",
                    f"vggt_geom/{cid}.npz",
                    f"qwen_taskemb/{cid}.npy",
                ):
                    fl.write(rel.encode("utf-8") + b"\0")
            fl.write(b"action_stats_oxe_droid20k_stage1_world_v1.npz\0")
        summary_lines.append(
            f"{node}\t{len(bucket['records'])}\t{bucket['frames']}\t{bucket['windows']}\n"
        )
    (args.out / "summary.tsv").write_text("".join(summary_lines), encoding="utf-8")
    print("".join(summary_lines), end="", flush=True)


if __name__ == "__main__":
    main()
