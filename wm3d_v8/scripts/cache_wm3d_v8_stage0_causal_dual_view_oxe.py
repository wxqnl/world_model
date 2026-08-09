#!/usr/bin/env python3
"""Build atomic WM3D-V8 causal dual-view windows from sealed OXE caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wm3d_v3.data.manifest import OXEClipRecord, read_manifest
from wm3d_v3.data.v8_causal_dual_view import (
    CAUSAL_DUAL_VIEW_REPRESENTATION,
    CAUSAL_DUAL_VIEW_SCHEMA,
    causal_dual_view_metadata,
    encode_causal_dual_view,
    validate_causal_dual_view_archive,
)
from wm3d_v3.encoders.vggt_encoder import VGGTEncoder
from wm3d_v3.models.token_codec import PCATokenCodec, TokenCodecConfig
from wm3d_v3.stage1.action_window_geometry import VGGT_MODEL_REVISION


PRODUCER_SCHEMA = "wm3d_v8_stage0_causal_dual_view_oxe_producer_v1"


def _safe(clip_id: str) -> str:
    return clip_id.replace("/", "__")


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _window_path(output_root: Path, clip_id: str, start: int) -> Path:
    return Path(output_root) / f"{_safe(clip_id)}__start_{int(start):06d}.npz"


def _same_payload(path: Path, payload: dict[str, Any]) -> bool:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(payload):
                return False
            return all(
                np.array_equal(np.asarray(archive[key]), np.asarray(value))
                for key, value in payload.items()
            )
    except (OSError, ValueError, KeyError):
        return False


def _publish_archive(path: Path, payload: dict[str, Any]) -> str:
    """Atomically publish one immutable archive, accepting only exact replay."""

    T = int(np.asarray(payload["context_frames"]).item())
    k = int(np.asarray(payload["future_frames"]).item())
    validate_causal_dual_view_archive(payload, T=T, k=k, paired_views=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not _same_payload(path, payload):
            raise FileExistsError(f"existing causal archive is non-identical: {path}")
        return _sha256_file(path)

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary archive already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(
                handle,
                **{key: np.asarray(value) for key, value in payload.items()},
            )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not _same_payload(path, payload):
                raise FileExistsError(
                    f"existing causal archive is non-identical: {path}"
                )
        if not _same_payload(path, payload):
            raise RuntimeError(f"published causal archive failed replay check: {path}")
        return _sha256_file(path)
    finally:
        temporary.unlink(missing_ok=True)


def _resize_frames(frames: np.ndarray, size: int = 224) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("RGB cache must be uint8 [N,H,W,3]")
    tensor = torch.from_numpy(np.asarray(frames)).permute(0, 3, 1, 2).float()
    tensor.div_(255.0)
    if tensor.shape[-2:] != (size, size):
        tensor = F.interpolate(
            tensor,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return tensor


@torch.inference_mode()
def _encode_record_windows(
    record: OXEClipRecord,
    rgb: np.ndarray,
    *,
    starts: Iterable[int],
    encoder: torch.nn.Module,
    codec: Any,
    T: int,
    k: int,
    source: str,
    split: str,
    input_manifest_sha256: str,
    selection_sha256: str,
    config_sha256: str,
    rgb_sha256: str,
    action_sha256: str,
    task_sha256: str,
) -> list[tuple[int, dict[str, np.ndarray]]]:
    starts = [int(start) for start in starts]
    if starts != sorted(set(starts)):
        raise ValueError("OXE starts must be sorted and unique")
    if not starts:
        raise ValueError("OXE selection contains no starts")
    for name, value in (
        ("input_manifest_sha256", input_manifest_sha256),
        ("selection_sha256", selection_sha256),
        ("config_sha256", config_sha256),
        ("rgb_sha256", rgb_sha256),
        ("action_sha256", action_sha256),
        ("task_sha256", task_sha256),
    ):
        if len(value) != 64:
            raise ValueError(f"{name} must be a SHA256")
    if len(rgb) < T + k:
        raise ValueError(f"RGB cache for {record.clip_id} is too short")
    images = _resize_frames(np.asarray(rgb))
    output: list[tuple[int, dict[str, np.ndarray]]] = []
    for start in starts:
        if start < 0 or start + T + k > len(images):
            raise ValueError(
                f"window outside RGB cache: {record.clip_id} start={start}"
            )
        payload = {
            **causal_dual_view_metadata(T=T, k=k),
            "producer_schema": np.asarray(PRODUCER_SCHEMA),
            "clip_id": np.asarray(record.clip_id),
            "dataset": np.asarray(record.dataset),
            "source": np.asarray(source),
            "split": np.asarray(split),
            "start": np.asarray(start, dtype=np.int64),
            "task_text": np.asarray(record.task_text),
            "input_manifest_sha256": np.asarray(input_manifest_sha256),
            "selection_sha256": np.asarray(selection_sha256),
            "config_sha256": np.asarray(config_sha256),
            "rgb_sha256": np.asarray(rgb_sha256),
            "action_sha256": np.asarray(action_sha256),
            "task_sha256": np.asarray(task_sha256),
            **encode_causal_dual_view(
                images[start : start + T + k],
                encoder=encoder,
                codec=codec,
                T=T,
                k=k,
            ),
        }
        payload["token_count"] = np.asarray(
            payload["context_codes"].shape[-2], dtype=np.int64
        )
        payload["token_dim"] = np.asarray(2048, dtype=np.int64)
        payload["latent_dim"] = np.asarray(
            payload["context_codes"].shape[-1], dtype=np.int64
        )
        validate_causal_dual_view_archive(
            payload,
            T=T,
            k=k,
            paired_views=False,
        )
        output.append((start, payload))
    return output


def _index_row(
    *,
    record: OXEClipRecord,
    source: str,
    split: str,
    start: int,
    path: Path,
    artifact_sha256: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    T = int(np.asarray(payload["context_frames"]).item())
    k = int(np.asarray(payload["future_frames"]).item())
    summary = validate_causal_dual_view_archive(
        payload,
        T=T,
        k=k,
        paired_views=False,
    )
    identity = {
        "clip_id": record.clip_id,
        "dataset": record.dataset,
        "source": source,
        "split": split,
        "start": int(start),
        "T": T,
        "k": k,
        "input_manifest_sha256": str(
            np.asarray(payload["input_manifest_sha256"]).item()
        ),
        "selection_sha256": str(np.asarray(payload["selection_sha256"]).item()),
        "config_sha256": str(np.asarray(payload["config_sha256"]).item()),
        "rgb_sha256": str(np.asarray(payload["rgb_sha256"]).item()),
        "action_sha256": str(np.asarray(payload["action_sha256"]).item()),
        "task_sha256": str(np.asarray(payload["task_sha256"]).item()),
    }
    return {
        "schema": CAUSAL_DUAL_VIEW_SCHEMA,
        "representation": CAUSAL_DUAL_VIEW_REPRESENTATION,
        "context_future_leakage": False,
        "target_usage": "supervision_only",
        "geometry_coordinate_frame": "first_observed_camera",
        **identity,
        "P": int(summary["token_count"]),
        "token_D": 2048,
        "latent_dim": int(summary["latent_dim"]),
        "path": str(Path(path).resolve()),
        "artifact_sha256": artifact_sha256,
        "window_identity_sha256": _json_sha256(identity),
    }


def _validate_index_selection(
    expected: Iterable[tuple[str, int]],
    rows: Iterable[dict[str, Any]],
) -> None:
    expected_keys = [(str(clip_id), int(start)) for clip_id, start in expected]
    actual_keys = [
        (str(row.get("clip_id")), int(row.get("start", -1)))
        for row in rows
    ]
    if (
        len(expected_keys) != len(set(expected_keys))
        or len(actual_keys) != len(set(actual_keys))
        or sorted(expected_keys) != sorted(actual_keys)
    ):
        raise ValueError(
            "selection closure mismatch: "
            f"expected={sorted(expected_keys)} actual={sorted(actual_keys)}"
        )


def _atomic_publish_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = text.encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"existing report is non-identical: {path}")
        return _sha256_file(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise FileExistsError(f"existing report is non-identical: {path}")
        return _sha256_file(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_codec(path: Path, device: torch.device) -> PCATokenCodec:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    codec = PCATokenCodec(
        TokenCodecConfig(
            token_dim=int(payload["token_dim"]),
            latent_dim=int(payload["latent_dim"]),
        )
    )
    codec.set_basis(payload["mean"], payload["components"])
    return codec.to(device).eval()


def _stable_shard(clip_id: str, start: int, num_shards: int) -> int:
    digest = hashlib.sha256(f"{clip_id}\0{int(start)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def _shard_path(path: Path, shard_id: int, num_shards: int) -> Path:
    if num_shards == 1:
        return path
    return path.with_name(
        f"{path.stem}.shard-{shard_id:05d}-of-{num_shards:05d}{path.suffix}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--codec-downstream-report", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--rgb-subdir", default="rgb_256")
    parser.add_argument("--action-subdir", default="actions")
    parser.add_argument("--task-subdir", default="qwen_taskemb")
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.T <= 0 or args.k <= 0 or args.stride <= 0:
        raise SystemExit("T, k, and stride must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("invalid shard-id/num-shards")
    downstream = json.loads(args.codec_downstream_report.read_text())
    if downstream.get("formal_cache_allowed") is not True:
        raise SystemExit("codec downstream gate did not pass")

    manifest_sha = _sha256_file(args.manifest)
    codec_sha = _sha256_file(args.codec)
    records = sorted(read_manifest(args.manifest), key=lambda row: row.clip_id)
    candidates: list[tuple[OXEClipRecord, int]] = []
    for record in records:
        for start in range(
            0,
            int(record.n_frames) - int(args.T) - int(args.k) + 1,
            int(args.stride),
        ):
            candidates.append((record, start))
    if args.max_windows > 0:
        candidates = candidates[: int(args.max_windows)]
    if not candidates:
        raise SystemExit("selection contains no OXE windows")
    selection_identity = [
        {
            "clip_id": record.clip_id,
            "start": start,
            "source": args.source,
            "split": args.split,
        }
        for record, start in candidates
    ]
    selection_sha = _json_sha256(selection_identity)
    config_identity = {
        "schema": PRODUCER_SCHEMA,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha,
        "cache_root": str(args.cache_root.resolve()),
        "codec_sha256": codec_sha,
        "source": args.source,
        "split": args.split,
        "T": args.T,
        "k": args.k,
        "stride": args.stride,
        "rgb_subdir": args.rgb_subdir,
        "action_subdir": args.action_subdir,
        "task_subdir": args.task_subdir,
        "vggt_revision": VGGT_MODEL_REVISION,
    }
    config_sha = _json_sha256(config_identity)
    assigned = [
        (record, start)
        for record, start in candidates
        if _stable_shard(record.clip_id, start, args.num_shards) == args.shard_id
    ]
    if not assigned:
        raise SystemExit("this shard has no selected OXE windows")

    device = torch.device(args.device)
    codec = _load_codec(args.codec, device)
    encoder = VGGTEncoder(
        device=str(device),
        return_depth=True,
        return_depth_conf=True,
        return_geom_extra=True,
        model_revision=VGGT_MODEL_REVISION,
        local_files_only=True,
    )
    by_clip: dict[str, tuple[OXEClipRecord, list[int]]] = {}
    for record, start in assigned:
        if record.clip_id not in by_clip:
            by_clip[record.clip_id] = (record, [])
        by_clip[record.clip_id][1].append(start)

    rows: list[dict[str, Any]] = []
    for clip_id in sorted(by_clip):
        record, starts = by_clip[clip_id]
        safe = _safe(clip_id)
        rgb_path = args.cache_root / args.rgb_subdir / f"{safe}.npy"
        action_path = args.cache_root / args.action_subdir / f"{safe}.npy"
        task_path = args.cache_root / args.task_subdir / f"{safe}.npy"
        for path, label in (
            (rgb_path, "RGB"),
            (action_path, "action"),
            (task_path, "task"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"missing sealed {label} cache: {path}")
        rgb = np.load(rgb_path, mmap_mode="r")
        actions = np.load(action_path, mmap_mode="r")
        task = np.load(task_path, mmap_mode="r")
        if len(actions) < max(starts) + args.T + args.k - 1:
            raise ValueError(f"action cache is too short: {clip_id}")
        if np.asarray(task).shape != (2048,) or not np.isfinite(task).all():
            raise ValueError(f"task cache is invalid: {clip_id}")
        encoded = _encode_record_windows(
            record,
            rgb,
            starts=sorted(starts),
            encoder=encoder,
            codec=codec,
            T=args.T,
            k=args.k,
            source=args.source,
            split=args.split,
            input_manifest_sha256=manifest_sha,
            selection_sha256=selection_sha,
            config_sha256=config_sha,
            rgb_sha256=_sha256_file(rgb_path),
            action_sha256=_sha256_file(action_path),
            task_sha256=_sha256_file(task_path),
        )
        for start, payload in encoded:
            path = _window_path(args.output_root, clip_id, start)
            artifact_sha = _publish_archive(path, payload)
            rows.append(
                _index_row(
                    record=record,
                    source=args.source,
                    split=args.split,
                    start=start,
                    path=path,
                    artifact_sha256=artifact_sha,
                    payload=payload,
                )
            )

    _validate_index_selection(
        [(record.clip_id, start) for record, start in assigned],
        rows,
    )
    rows.sort(key=lambda row: (row["clip_id"], row["start"]))
    index_path = _shard_path(args.index, args.shard_id, args.num_shards)
    index_text = "".join(
        json.dumps(row, sort_keys=True) + "\n"
        for row in rows
    )
    index_sha = _atomic_publish_text(index_path, index_text)
    report = {
        "schema": PRODUCER_SCHEMA,
        "pass": True,
        "source": args.source,
        "split": args.split,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "selected_global": len(candidates),
        "selected_shard": len(assigned),
        "encoded": len(rows),
        "manifest_sha256": manifest_sha,
        "selection_sha256": selection_sha,
        "config_sha256": config_sha,
        "index": str(index_path.resolve()),
        "index_sha256": index_sha,
        "codec_sha256": codec_sha,
        "vggt_revision": VGGT_MODEL_REVISION,
    }
    report_path = index_path.with_suffix(".report.json")
    _atomic_publish_text(
        report_path,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
