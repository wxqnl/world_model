"""Cache LeRobot DROID episodes into the WM3D cache format.

This is a bridge from the HF/LeRobot layout:

    data/chunk-XXX/file-YYY.parquet
    videos/<camera>/chunk-XXX/file-YYY.mp4

to the existing WM3D training cache:

    vggt_pooled/<clip_id>.npy
    vggt_geom/<clip_id>.npz
    rgb_256/<clip_id>.npy
    actions/<clip_id>.npy
    qwen_taskemb/<clip_id>.npy
    manifest.jsonl

The script intentionally supports a small max_episodes smoke run. Full DROID
conversion should be launched as a sharded cache job after this path is verified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from wm3d_v3.data.manifest import OXEClipRecord, write_manifest
from wm3d_v3.encoders.qwen_vl_encoder import QwenVLEmbed
from wm3d_v3.encoders.vggt_encoder import VGGTEncoder
from scripts.cache_geom_utils import (
    GEOM_EXTRA_KEYS,
    atomic_savez_compressed,
    existing_npz_payload,
    validate_actions_npy,
    validate_geom_npz,
    validate_pooled_npy,
    validate_qwen_npy,
    validate_rgb_npy,
)


def safe_id(clip_id: str) -> str:
    return clip_id.replace("/", "__")


def resize_image_batch(imgs: np.ndarray, size: int) -> torch.Tensor:
    t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
    if t.shape[-1] != size or t.shape[-2] != size:
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return t


def _append_chunk(chunks: dict[str, list[np.ndarray]], out: dict, key: str) -> None:
    if key not in out:
        return
    chunks[key].append(out[key][0].detach().cpu().numpy().astype(np.float16))


def _as_vec(x: Any, n: int | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if n is not None:
        out = np.zeros(n, dtype=np.float32)
        m = min(n, arr.size)
        out[:m] = arr.reshape(-1)[:m]
        return out
    return arr.reshape(-1).astype(np.float32)


def action_from_row(row: dict[str, Any]) -> np.ndarray:
    """Return a 7D action vector.

    Prefer LeRobot's normalized/packed `action` column when it is 7D or longer.
    Fall back to cartesian position + gripper position.
    """
    a = row.get("action")
    if a is not None:
        avec = _as_vec(a)
        if avec.size >= 7:
            return avec[:7].astype(np.float32)
    cart = _as_vec(row.get("action.cartesian_position", []), 6)
    grip = _as_vec(row.get("action.gripper_position", 0.0), 1)
    return np.concatenate([cart, grip], axis=0).astype(np.float32)


def read_episode_table(parquet_path: Path) -> list[dict[str, Any]]:
    cols = [
        "episode_index",
        "frame_index",
        "timestamp",
        "is_first",
        "is_last",
        "language_instruction",
        "language_instruction_2",
        "language_instruction_3",
        "action",
        "action.cartesian_position",
        "action.gripper_position",
    ]
    table = pq.read_table(parquet_path, columns=cols)
    data = table.to_pylist()
    return data


def read_episode_metadata(root: Path, camera: str) -> list[dict[str, Any]]:
    meta_root = root / "meta" / "episodes"
    if not meta_root.exists():
        return []
    cols = [
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
        f"videos/{camera}/chunk_index",
        f"videos/{camera}/file_index",
        f"videos/{camera}/from_timestamp",
        f"videos/{camera}/to_timestamp",
    ]
    rows: list[dict[str, Any]] = []
    for p in sorted(meta_root.glob("chunk-*/file-*.parquet")):
        rows.extend(pq.read_table(p, columns=cols).to_pylist())
    return rows


def select_episodes(
    rows: list[dict[str, Any]],
    min_frames: int,
    max_frames_per_episode: int,
    max_episodes: int,
) -> list[tuple[int, int, int, str]]:
    """Return (episode_index, start_row, end_row_exclusive, task_text)."""
    by_ep: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_ep[int(row["episode_index"])].append(i)

    selected: list[tuple[int, int, int, str]] = []
    for ep_idx in sorted(by_ep):
        idxs = by_ep[ep_idx]
        if len(idxs) < min_frames:
            continue
        if idxs != list(range(idxs[0], idxs[-1] + 1)):
            continue
        start = idxs[0]
        end = min(idxs[-1] + 1, start + max_frames_per_episode)
        if end - start < min_frames:
            continue
        first = rows[start]
        task = (
            first.get("language_instruction")
            or first.get("language_instruction_2")
            or first.get("language_instruction_3")
            or "robot manipulation"
        )
        selected.append((ep_idx, start, end, str(task)))
        if len(selected) >= max_episodes:
            break
    return selected


def decode_selected_frames(video_path: Path, ranges: list[tuple[int, int]]) -> dict[int, np.ndarray]:
    """Decode selected contiguous row ranges using PyAV.

    Returns a map from absolute video frame index to RGB uint8 frame.
    """
    wanted: set[int] = set()
    max_needed = -1
    for start, end in ranges:
        wanted.update(range(start, end))
        max_needed = max(max_needed, end - 1)
    out: dict[int, np.ndarray] = {}
    if not wanted:
        return out
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i in wanted:
                out[i] = frame.to_ndarray(format="rgb24")
            if i >= max_needed:
                break
    missing = len(wanted) - len(out)
    if missing:
        raise RuntimeError(f"decoded {len(out)}/{len(wanted)} selected frames from {video_path}")
    return out


def decode_frame_indices(video_path: Path, indices: set[int]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    if not indices:
        return out
    max_needed = max(indices)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i in indices:
                out[i] = frame.to_ndarray(format="rgb24")
            if i >= max_needed:
                break
    missing = len(indices) - len(out)
    if missing:
        raise RuntimeError(f"decoded {len(out)}/{len(indices)} requested frames from {video_path}")
    return out


def cache_one_episode(
    *,
    clip_id: str,
    task_text: str,
    rows: list[dict[str, Any]],
    frame_map: dict[int, np.ndarray],
    start: int,
    end: int,
    frame_stride: int,
    row_indices: list[int] | None = None,
    video_indices: list[int] | None = None,
    cache_root: Path,
    enc: VGGTEncoder,
    qwen: QwenVLEmbed | None,
    batch_frames: int,
    write_rgb: bool,
    write_geom: bool,
    geom_extra: bool = True,
) -> OXEClipRecord:
    cid = safe_id(clip_id)
    pool_dir = cache_root / "vggt_pooled"
    geom_dir = cache_root / "vggt_geom"
    rgb_dir = cache_root / "rgb_256"
    act_dir = cache_root / "actions"
    qwen_dir = cache_root / "qwen_taskemb"
    for d in (pool_dir, geom_dir, rgb_dir, act_dir, qwen_dir):
        d.mkdir(parents=True, exist_ok=True)

    if row_indices is None:
        row_indices = list(range(start, end, max(1, int(frame_stride))))
    if video_indices is None:
        video_indices = row_indices
    frames = np.stack([frame_map[i] for i in video_indices], axis=0)
    actions = np.stack([action_from_row(rows[i]) for i in row_indices], axis=0).astype(np.float32)
    n_frames = int(frames.shape[0])

    pool_path = pool_dir / f"{cid}.npy"
    geom_path = geom_dir / f"{cid}.npz"
    rgb_path = rgb_dir / f"{cid}.npy"
    act_path = act_dir / f"{cid}.npy"
    qwen_path = qwen_dir / f"{cid}.npy"

    if not validate_actions_npy(act_path, expected_frames=n_frames):
        np.save(act_path, actions)
    if write_rgb and not validate_rgb_npy(rgb_path, expected_frames=n_frames):
        rgb256 = resize_image_batch(frames, 256)
        np.save(rgb_path, (rgb256.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy())

    need_pooled = not validate_pooled_npy(pool_path, expected_frames=n_frames)
    need_geom = write_geom and not validate_geom_npz(
        geom_path,
        expected_frames=n_frames,
        require_geom_extra=bool(geom_extra),
    )
    if need_pooled or need_geom:
        frames_224 = resize_image_batch(frames, 224)
        pooled_chunks: list[np.ndarray] = []
        geom_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
        for s in range(0, n_frames, batch_frames):
            chunk = frames_224[s : s + batch_frames].unsqueeze(0).to("cuda")
            out = enc(chunk)
            if need_pooled:
                pooled_chunks.append(out["pooled"][0].cpu().numpy().astype(np.float16))
            if need_geom:
                for key in ("depth", *GEOM_EXTRA_KEYS):
                    _append_chunk(geom_chunks, out, key)
        if need_pooled and pooled_chunks:
            np.save(pool_path, np.concatenate(pooled_chunks, axis=0))
        if need_geom:
            payload = existing_npz_payload(geom_path)
            for key, vals in geom_chunks.items():
                if vals:
                    payload[key] = np.concatenate(vals, axis=0)
            if payload:
                atomic_savez_compressed(geom_path, **payload)

    if qwen is not None and not validate_qwen_npy(qwen_path):
        emb = qwen.embed(task_text, Image.fromarray(frames[0]))
        np.save(qwen_path, emb.numpy().astype(np.float16))

    return OXEClipRecord(
        clip_id=clip_id,
        dataset="droid",
        tar_path="",
        pickle_member="",
        n_frames=n_frames,
        fps=max(1, int(round(15 / max(1, int(frame_stride))))),
        robot="franka",
        task_text=task_text,
        action_dim=7,
        action_kind="droid_action_7d",
        image_keys=["observation.images.exterior_1_left"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/data/Minko/datasets/hf_world_model_pretrain/lerobot__droid_1.0.1"))
    ap.add_argument("--cache_root", type=Path, default=Path("/data/Minko/datasets/cache/wm3d_v3_droid_smoke"))
    ap.add_argument("--out_manifest", type=Path, default=Path("manifests/droid_smoke_cached_rgb_geom_v1.jsonl"))
    ap.add_argument("--camera", default="observation.images.exterior_1_left")
    ap.add_argument("--max_files", type=int, default=1)
    ap.add_argument("--file_start", type=int, default=0)
    ap.add_argument("--file_stride", type=int, default=1)
    ap.add_argument("--episode_start", type=int, default=None)
    ap.add_argument("--episode_stride", type=int, default=None)
    ap.add_argument("--max_episodes", type=int, default=32)
    ap.add_argument("--min_frames", type=int, default=32)
    ap.add_argument("--max_frames_per_episode", type=int, default=96)
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--batch_frames", type=int, default=16)
    ap.add_argument("--skip_qwen", action="store_true")
    ap.add_argument("--no_rgb", action="store_true")
    ap.add_argument("--no_geom", action="store_true")
    ap.add_argument("--geom_extra", dest="geom_extra", action="store_true", default=True)
    ap.add_argument("--no_geom_extra", "--no-geom_extra", dest="geom_extra", action="store_false")
    args = ap.parse_args()

    all_files = sorted((args.root / "data").glob("chunk-*/file-*.parquet"))
    data_files = all_files[args.file_start :: max(1, int(args.file_stride))][: args.max_files]
    if not data_files:
        raise RuntimeError(f"no parquet files under {args.root / 'data'}")

    if args.no_geom:
        args.geom_extra = False
    enc = VGGTEncoder(device="cuda", return_depth=not args.no_geom, return_geom_extra=bool(args.geom_extra))
    qwen = None if args.skip_qwen else QwenVLEmbed()

    records: list[OXEClipRecord] = []
    remaining = int(args.max_episodes)
    meta_rows = read_episode_metadata(args.root, args.camera)
    if meta_rows:
        episode_start = int(args.file_start if args.episode_start is None else args.episode_start)
        episode_stride = int(args.file_stride if args.episode_stride is None else args.episode_stride)
        selected_meta: list[dict[str, Any]] = []
        for ordinal, meta in enumerate(sorted(meta_rows, key=lambda r: int(r["episode_index"]))):
            if ordinal % max(1, episode_stride) != episode_start:
                continue
            if int(meta.get("length", 0) or 0) < int(args.min_frames):
                continue
            selected_meta.append(meta)
            if len(selected_meta) >= remaining:
                break

        data_file_base: dict[tuple[int, int], int] = {}
        for meta in meta_rows:
            key = (int(meta["data/chunk_index"]), int(meta["data/file_index"]))
            start = int(meta["dataset_from_index"])
            data_file_base[key] = min(start, data_file_base.get(key, start))

        grouped: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
        for meta in selected_meta:
            grouped[(
                int(meta["data/chunk_index"]),
                int(meta["data/file_index"]),
                int(meta[f"videos/{args.camera}/chunk_index"]),
                int(meta[f"videos/{args.camera}/file_index"]),
            )].append(meta)

        for (data_chunk, data_file, video_chunk, video_file), metas in sorted(grouped.items()):
            if remaining <= 0:
                break
            parquet_path = args.root / "data" / f"chunk-{data_chunk:03d}" / f"file-{data_file:03d}.parquet"
            video_path = args.root / "videos" / args.camera / f"chunk-{video_chunk:03d}" / f"file-{video_file:03d}.mp4"
            if not parquet_path.exists() or not video_path.exists():
                print(f"[skip] missing data/video: {parquet_path} {video_path}")
                continue
            print(f"[file] parquet={parquet_path} video={video_path} episodes={len(metas)}")
            rows = read_episode_table(parquet_path)
            prepared: list[tuple[dict[str, Any], list[int], list[int], str]] = []
            wanted_video: set[int] = set()
            file_base = data_file_base.get((data_chunk, data_file), 0)
            for meta in metas:
                global_start = int(meta["dataset_from_index"])
                global_end = min(int(meta["dataset_to_index"]), global_start + int(args.max_frames_per_episode))
                local_start = max(0, global_start - file_base)
                local_end = min(len(rows), global_end - file_base)
                if local_end <= local_start:
                    print(
                        "[skip] metadata row range outside parquet: "
                        f"episode={meta['episode_index']} global={global_start}:{global_end} "
                        f"base={file_base} rows={len(rows)}"
                    )
                    continue
                row_indices = list(range(local_start, local_end, max(1, int(args.frame_stride))))
                if len(row_indices) < max(1, int(args.min_frames // max(1, int(args.frame_stride)))):
                    continue
                from_ts = float(meta[f"videos/{args.camera}/from_timestamp"])
                video_indices = [
                    int(round((from_ts + float(rows[i].get("timestamp", 0.0))) * 15.0))
                    for i in row_indices
                ]
                task_items = meta.get("tasks") or []
                task_text = next((str(t) for t in task_items if str(t).strip()), "robot manipulation")
                prepared.append((meta, row_indices, video_indices, task_text))
                wanted_video.update(video_indices)
            if not prepared:
                continue
            frame_map = decode_frame_indices(video_path, wanted_video)
            for meta, row_indices, video_indices, task_text in tqdm(prepared, desc="cache_droid"):
                ep_idx = int(meta["episode_index"])
                global_first = file_base + row_indices[0]
                global_end = file_base + row_indices[-1] + 1
                clip_id = (
                    f"droid/chunk-{data_chunk:03d}/data-{data_file:03d}_video-{video_file:03d}/"
                    f"episode_{ep_idx:06d}_{global_first:09d}_{global_end:09d}_s{max(1, int(args.frame_stride))}"
                )
                rec = cache_one_episode(
                    clip_id=clip_id,
                    task_text=task_text,
                    rows=rows,
                    frame_map=frame_map,
                    start=row_indices[0],
                    end=row_indices[-1] + 1,
                    frame_stride=args.frame_stride,
                    row_indices=row_indices,
                    video_indices=video_indices,
                    cache_root=args.cache_root,
                    enc=enc,
                    qwen=qwen,
                    batch_frames=args.batch_frames,
                    write_rgb=not args.no_rgb,
                    write_geom=not args.no_geom,
                    geom_extra=bool(args.geom_extra),
                )
                records.append(rec)
                write_manifest(args.out_manifest, records)
                remaining -= 1
                if remaining <= 0:
                    break
        write_manifest(args.out_manifest, records)
        print({"manifest": str(args.out_manifest), "records": len(records), "frames": sum(r.n_frames for r in records)})
        return

    for parquet_path in data_files:
        if remaining <= 0:
            break
        rel = parquet_path.relative_to(args.root / "data")
        video_path = args.root / "videos" / args.camera / rel.with_suffix(".mp4")
        if not video_path.exists():
            print(f"[skip] missing video: {video_path}")
            continue
        print(f"[file] parquet={parquet_path} video={video_path}")
        rows = read_episode_table(parquet_path)
        selected = select_episodes(
            rows,
            min_frames=args.min_frames,
            max_frames_per_episode=args.max_frames_per_episode,
            max_episodes=remaining,
        )
        if not selected:
            print(f"[skip] no suitable episodes: {parquet_path}")
            continue
        ranges = [(start, end) for _, start, end, _ in selected]
        frame_map = decode_selected_frames(video_path, ranges)
        for ep_idx, start, end, task_text in tqdm(selected, desc="cache_droid"):
            clip_id = (
                f"droid/{rel.parent.name}/{rel.stem}/"
                f"episode_{ep_idx:06d}_{start:09d}_{end:09d}_s{max(1, int(args.frame_stride))}"
            )
            rec = cache_one_episode(
                clip_id=clip_id,
                task_text=task_text,
                rows=rows,
                frame_map=frame_map,
                start=start,
                end=end,
                frame_stride=args.frame_stride,
                cache_root=args.cache_root,
                enc=enc,
                qwen=qwen,
                batch_frames=args.batch_frames,
                write_rgb=not args.no_rgb,
                write_geom=not args.no_geom,
                geom_extra=bool(args.geom_extra),
            )
            records.append(rec)
            remaining -= 1
            if remaining <= 0:
                break

    write_manifest(args.out_manifest, records)
    print({"manifest": str(args.out_manifest), "records": len(records), "frames": sum(r.n_frames for r in records)})


if __name__ == "__main__":
    main()
