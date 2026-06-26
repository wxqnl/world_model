#!/usr/bin/env python3
"""Build random-access tar shards for WM3D window geometry npz cache.

The output is a directory of uncompressed tar files plus index.tsv:
    member_name<TAB>shard_file<TAB>offset_data<TAB>size

Do not compress these shards. The dataset seeks directly to offset_data and
reads one npz, so compressed tar files would destroy random access.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time


def _shard_for(name: str, n: int) -> int:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % int(n)


def _iter_npz_files(src: Path):
    with os.scandir(src) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(".npz"):
                yield entry


def _safe(cid: str) -> str:
    return cid.replace("/", "__")


def _iter_manifest_window_names(manifest: Path, T: int, k: int, stride: int):
    win = int(T) + int(k)
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            n_frames = int(rec.get("n_frames") or 0)
            if n_frames < win:
                continue
            cid = _safe(str(rec["clip_id"]))
            for start in range(0, n_frames - win + 1, int(stride)):
                yield f"{cid}__start_{int(start):06d}.npz"


def _write_member_lists(
    src: Path,
    out: Path,
    num_shards: int,
    *,
    selected_shards: set[int] | None = None,
    shard_strategy: str = "hash",
    manifest: Path | None = None,
    T: int = 16,
    k: int = 8,
    stride: int = 4,
    log_every: int = 10000,
) -> tuple[list[int], int]:
    list_dir = out / "_lists"
    list_dir.mkdir(parents=True, exist_ok=True)
    if selected_shards is None:
        selected_shards = set(range(num_shards))
    handles = {
        idx: (list_dir / f"window_geom_{idx:04d}.lst").open("wb")
        for idx in sorted(selected_shards)
    }
    counts = [0 for _ in range(num_shards)]
    total = 0
    selected_total = 0
    try:
        if manifest is not None and shard_strategy == "contiguous":
            all_names = list(_iter_manifest_window_names(manifest, T=T, k=k, stride=stride))
            chunk = max(1, (len(all_names) + num_shards - 1) // num_shards)
            indexed_names = (
                (min(i // chunk, num_shards - 1), name)
                for i, name in enumerate(all_names)
            )
        elif manifest is not None:
            indexed_names = (
                (_shard_for(name, num_shards), name)
                for name in _iter_manifest_window_names(manifest, T=T, k=k, stride=stride)
            )
        else:
            indexed_names = (
                (_shard_for(entry.name, num_shards), entry.name)
                for entry in _iter_npz_files(src)
            )
        start_t = time.time()
        for idx, name in indexed_names:
            counts[idx] += 1
            total += 1
            if idx in selected_shards:
                handles[idx].write(name.encode("utf-8") + b"\0")
                selected_total += 1
            if log_every > 0 and total % log_every == 0:
                elapsed = max(1e-6, time.time() - start_t)
                print(
                    f"listed_files={total} selected_files={selected_total} "
                    f"rate_files_s={total/elapsed:.1f}",
                    flush=True,
                )
    finally:
        for handle in handles.values():
            handle.close()
    return counts, selected_total


def _build_one_shard(task: tuple[str, str, int]) -> tuple[int, int, int]:
    src_s, out_s, idx = task
    src = Path(src_s)
    out = Path(out_s)
    list_path = out / "_lists" / f"window_geom_{idx:04d}.lst"
    tar_name = f"window_geom_{idx:04d}.tar"
    tar_path = out / tar_name
    index_part = out / f"index_{idx:04d}.tsv"

    if list_path.stat().st_size == 0:
        index_part.write_text("", encoding="utf-8")
        return idx, 0, 0

    tmp_tar = tar_path.with_suffix(".tar.tmp")
    if tmp_tar.exists():
        tmp_tar.unlink()
    if tar_path.exists():
        tar_path.unlink()

    subprocess.run(
        [
            "tar",
            "--format=gnu",
            "--null",
            "-C",
            str(src),
            "-cf",
            str(tmp_tar),
            "-T",
            str(list_path),
        ],
        check=True,
    )
    tmp_tar.replace(tar_path)

    rows: list[str] = []
    files = 0
    bytes_in = 0
    with tarfile.open(tar_path, "r:") as tf:
        for member in tf:
            if not member.isfile():
                continue
            files += 1
            bytes_in += int(member.size)
            rows.append(
                f"{Path(member.name).name}\t{tar_name}\t{int(member.offset_data)}\t{int(member.size)}\n"
            )
    index_part.write_text("".join(rows), encoding="utf-8")
    return idx, files, bytes_in


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--num-shards", type=int, default=128)
    ap.add_argument("--shard-mod", type=int, default=1)
    ap.add_argument("--shard-rank", type=int, default=0)
    ap.add_argument("--shard-strategy", choices=("hash", "contiguous"), default="hash")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--backend", choices=("gnu-tar", "python"), default="gnu-tar")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log-every", type=int, default=10000)
    args = ap.parse_args()

    src = args.src
    out = args.out
    if not src.is_dir():
        raise FileNotFoundError(f"source dir not found: {src}")
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    tmp_index = out / "index.tsv.tmp"
    final_index = out / "index.tsv"
    tar_handles: dict[int, tarfile.TarFile] = {}
    counts = [0 for _ in range(args.num_shards)]
    bytes_in = 0
    start_t = time.time()

    if args.backend == "gnu-tar":
        if args.shard_mod <= 0:
            raise ValueError("--shard-mod must be positive")
        if args.shard_rank < 0 or args.shard_rank >= args.shard_mod:
            raise ValueError("--shard-rank must be in [0, shard_mod)")
        selected_shards = {
            idx for idx in range(args.num_shards)
            if idx % int(args.shard_mod) == int(args.shard_rank)
        }
        counts, listed = _write_member_lists(
            src,
            out,
            args.num_shards,
            selected_shards=selected_shards,
            shard_strategy=args.shard_strategy,
            manifest=args.manifest,
            T=args.T,
            k=args.k,
            stride=args.stride,
            log_every=args.log_every,
        )
        print(
            f"selected_files={listed} selected_shards={len(selected_shards)} "
            f"all_nonempty_shards={sum(1 for c in counts if c)} "
            f"elapsed_s={time.time() - start_t:.1f}",
            flush=True,
        )
        tasks = [(str(src), str(out), idx) for idx in sorted(selected_shards) if counts[idx]]
        done_files = 0
        with ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as pool:
            futures = [pool.submit(_build_one_shard, task) for task in tasks]
            for fut in as_completed(futures):
                idx, files, shard_bytes = fut.result()
                done_files += files
                bytes_in += shard_bytes
                elapsed = max(1e-6, time.time() - start_t)
                print(
                    f"built_shard={idx:04d} shard_files={files} "
                    f"done_files={done_files}/{listed} input_gb={bytes_in/1024**3:.1f} "
                    f"rate_files_s={done_files/elapsed:.1f} rate_mb_s={bytes_in/1024**2/elapsed:.1f}",
                    flush=True,
                )

        tmp_index = out / "index.tsv.tmp"
        final_index = out / "index.tsv"
        with tmp_index.open("w", encoding="utf-8") as index_f:
            index_f.write("# member\tshard\toffset_data\tsize\n")
            for idx in range(args.num_shards):
                part = out / f"index_{idx:04d}.tsv"
                if part.exists():
                    with part.open("r", encoding="utf-8") as part_f:
                        shutil.copyfileobj(part_f, index_f)
        tmp_index.replace(final_index)
        (out / "summary.txt").write_text(
            (
                "num_shards={}\nfiles={}\ninput_bytes={}\nnonempty_shards={}\n"
                "backend=gnu-tar\njobs={}\nshard_mod={}\nshard_rank={}\nselected_shards={}\n"
                "shard_strategy={}\n"
            ).format(
                args.num_shards,
                listed,
                bytes_in,
                sum(1 for c in counts if c),
                args.jobs,
                args.shard_mod,
                args.shard_rank,
                ",".join(str(i) for i in sorted(selected_shards)),
                args.shard_strategy,
            ),
            encoding="utf-8",
        )
        print(f"done files={listed} input_gb={bytes_in/1024**3:.1f} out={out}", flush=True)
        return

    def get_tar(idx: int) -> tarfile.TarFile:
        tf = tar_handles.get(idx)
        if tf is None:
            tf = tarfile.open(out / f"window_geom_{idx:04d}.tar", "w", format=tarfile.GNU_FORMAT)
            tar_handles[idx] = tf
        return tf

    try:
        with tmp_index.open("w", encoding="utf-8") as index_f:
            index_f.write("# member\tshard\toffset_data\tsize\n")
            for i, entry in enumerate(_iter_npz_files(src), start=1):
                name = entry.name
                shard_idx = _shard_for(name, args.num_shards)
                tf = get_tar(shard_idx)
                path = Path(entry.path)
                info = tf.gettarinfo(str(path), arcname=name)
                offset_data = int(tf.offset) + tarfile.BLOCKSIZE
                with path.open("rb") as f:
                    tf.addfile(info, f)
                shard_name = f"window_geom_{shard_idx:04d}.tar"
                index_f.write(f"{name}\t{shard_name}\t{offset_data}\t{int(info.size)}\n")
                counts[shard_idx] += 1
                bytes_in += int(info.size)
                if args.log_every > 0 and i % args.log_every == 0:
                    elapsed = max(1e-6, time.time() - start_t)
                    print(
                        f"built_files={i} input_gb={bytes_in/1024**3:.1f} "
                        f"rate_files_s={i/elapsed:.1f} rate_mb_s={bytes_in/1024**2/elapsed:.1f}",
                        flush=True,
                    )
    finally:
        for tf in tar_handles.values():
            tf.close()

    tmp_index.replace(final_index)
    (out / "summary.txt").write_text(
        "num_shards={}\nfiles={}\ninput_bytes={}\nnonempty_shards={}\n".format(
            args.num_shards, sum(counts), bytes_in, sum(1 for c in counts if c)
        ),
        encoding="utf-8",
    )
    print(f"done files={sum(counts)} input_gb={bytes_in/1024**3:.1f} out={out}", flush=True)


if __name__ == "__main__":
    main()
