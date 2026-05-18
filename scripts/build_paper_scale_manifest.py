"""Build `data/manifests/paper_scale.json` + `data/manifests/val_ids.json`.

Scans extracted-OXE directories (one per sub-dataset) and the cached DROID-100
clips and emits the clip manifest that drives Phase 1/2 paper-scale training.

Per `docs/DATA_ACQUISITION_SPEC.md` §2 target sizes are:

    bridge        12 000 episodes
    fractal       8 000
    kuka          4 000
    taco_play     3 000  (optional; included if its extracted dir exists)
    jaco_play     1 000
    droid_100      (all available, up to 2 000)

Selection is deterministic given a seed: per sub-dataset, episodes are sorted
by their numeric index and we take the first N. The validation split is
chosen by stable hash on `clip_id` (10% per sub-dataset, capped at 3 000
total), so re-running the script with the same seed and the same on-disk
extraction yields the same split.

Output schema (one entry per clip, identical to `data/manifests/set_a.json`):

    {
      "clip_id": "oxe_bridge_000123",
      "set": "A",
      "frames": ["data/raw/oxe_extracted/bridge/episode_00123/frame_0000.jpg", ...],
      "fps": 5,
      "meta": {
        "dataset": "bridge",
        "task": "Place the can to the left of the pot.",
        "episode_index": 123,
        "n_frames": 38
      },
      "gt_depth": null
    }

Notes
-----
* Episodes shorter than `--min_frames` (default 8 = predictor window) are skipped.
* Episodes longer than `--max_frames` (default 128) keep all their JPEGs on disk;
  the cache step trims via `cache.max_frames_per_clip`.
* `meta.episode_index` is a *global* index into the per-dataset extraction
  output, not the original OXE shard offset. That matches the on-disk
  `episode_NNNNN/` layout.
* The script does not move files. It only reads what `extract_oxe_frames.py`
  has already written, plus the existing DROID frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path

log = logging.getLogger("build_manifest")


DEFAULT_TARGETS = {
    "bridge": 12_000,
    "fractal20220817_data": 8_000,
    "kuka": 4_000,
    "taco_play": 3_000,
    "jaco_play": 1_000,
}

DEFAULT_FPS = {
    "bridge": 5,
    "fractal20220817_data": 3,
    "kuka": 10,
    "taco_play": 15,
    "jaco_play": 10,
    "droid_100": 15,
}


EPISODE_DIR_RE = re.compile(r"^episode_(\d+)$")


def _stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:12], 16)


def scan_oxe_dataset(
    extracted_root: Path,
    dataset: str,
    target_n: int,
    min_frames: int,
) -> list[dict]:
    """Return up to `target_n` clip dicts for one OXE sub-dataset."""
    ds_dir = extracted_root / dataset
    if not ds_dir.is_dir():
        log.info("  %s: no extracted dir, skipping", dataset)
        return []

    ep_dirs: list[tuple[int, Path]] = []
    for d in ds_dir.iterdir():
        if not d.is_dir():
            continue
        m = EPISODE_DIR_RE.match(d.name)
        if not m:
            continue
        ep_dirs.append((int(m.group(1)), d))
    ep_dirs.sort()
    log.info("  %s: scanning %d candidate episodes (target %d)", dataset, len(ep_dirs), target_n)

    clips: list[dict] = []
    for ep_idx, d in ep_dirs:
        meta_path = d / "episode.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as e:  # pragma: no cover - defensive
            log.warning("  %s/episode_%05d: bad meta json: %s", dataset, ep_idx, e)
            continue
        n_frames = int(meta.get("n_frames", 0))
        if n_frames < min_frames:
            continue
        # frame paths relative to the repo root, matching `data/manifests/set_a.json`
        # style. extracted_root is passed in as a relative path so str(d / ...)
        # yields e.g. "data/raw/oxe_extracted/bridge/episode_00123/frame_0000.jpg".
        frames = [str(d / f"frame_{fi:04d}.jpg") for fi in range(n_frames)]
        clip_id = f"oxe_{dataset.replace('20220817_data', '').rstrip('_')}_{ep_idx:06d}"
        clips.append(
            {
                "clip_id": clip_id,
                "set": "A",
                "frames": frames,
                "fps": DEFAULT_FPS.get(dataset, 10),
                "meta": {
                    "dataset": dataset,
                    "task": meta.get("task", ""),
                    "episode_index": ep_idx,
                    "n_frames": n_frames,
                },
                "gt_depth": None,
            }
        )
        if len(clips) >= target_n:
            break

    log.info("  %s: selected %d episodes", dataset, len(clips))
    return clips


def scan_droid(
    droid_frames_root: Path,
    droid_meta_path: Path | None,
    target_n: int,
    min_frames: int,
) -> list[dict]:
    """DROID-100 clips already on disk as `episode_NNNN/frame_NNNN.jpg`."""
    if not droid_frames_root.is_dir():
        log.info("  droid_100: no frames dir, skipping")
        return []

    # Optional task strings from droid_100 lerobot meta. Cheap if it exists.
    task_by_ep: dict[int, str] = {}
    if droid_meta_path and droid_meta_path.exists():
        try:
            for line in droid_meta_path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if "episode_index" in row and "task" in row:
                    task_by_ep[int(row["episode_index"])] = str(row["task"])
        except Exception as e:  # pragma: no cover - defensive
            log.warning("droid task meta parse failed: %s", e)

    ep_dirs: list[tuple[int, Path]] = []
    for d in droid_frames_root.iterdir():
        if not d.is_dir():
            continue
        m = EPISODE_DIR_RE.match(d.name)
        if not m:
            continue
        ep_dirs.append((int(m.group(1)), d))
    ep_dirs.sort()

    clips: list[dict] = []
    for ep_idx, d in ep_dirs:
        frames = sorted(d.glob("frame_*.jpg"))
        n = len(frames)
        if n < min_frames:
            continue
        rel = [str(p) for p in frames]
        clips.append(
            {
                "clip_id": f"droid_{ep_idx:04d}",
                "set": "A",
                "frames": rel,
                "fps": DEFAULT_FPS["droid_100"],
                "meta": {
                    "dataset": "droid_100",
                    "task": task_by_ep.get(ep_idx, ""),
                    "episode_index": ep_idx,
                    "n_frames": n,
                },
                "gt_depth": None,
            }
        )
        if len(clips) >= target_n:
            break

    log.info("  droid_100: selected %d episodes", len(clips))
    return clips


def split_val(
    clips: list[dict],
    val_ratio: float,
    val_cap: int,
    seed: int,
) -> tuple[list[dict], list[str]]:
    """Pick a deterministic val set, capped at `val_cap` total."""
    # Per sub-dataset, hash clip_ids and take the lowest-hash `ceil(ratio*n)` ids.
    by_ds: dict[str, list[dict]] = {}
    for c in clips:
        by_ds.setdefault(c["meta"]["dataset"], []).append(c)
    val_ids: set[str] = set()
    for ds, ds_clips in by_ds.items():
        ranked = sorted(
            ds_clips, key=lambda c: _stable_hash(f"{seed}:{c['clip_id']}")
        )
        k = max(1, int(round(len(ranked) * val_ratio)))
        for c in ranked[:k]:
            val_ids.add(c["clip_id"])
    # Cap the global val set at val_cap (drop the highest-hash held-out ids).
    if len(val_ids) > val_cap:
        held = sorted(val_ids, key=lambda cid: _stable_hash(f"{seed}:{cid}"))
        val_ids = set(held[:val_cap])
    val_list = sorted(val_ids)
    log.info("val set: %d clips", len(val_list))
    return clips, val_list


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extracted_root", default="data/raw/oxe_extracted")
    parser.add_argument("--droid_frames_root", default="data/raw/droid_frames")
    parser.add_argument("--droid_meta", default="data/raw/droid_100/meta/episodes.jsonl")
    parser.add_argument(
        "--out_manifest", default="data/manifests/paper_scale.json"
    )
    parser.add_argument("--out_val_ids", default="data/manifests/val_ids.json")
    parser.add_argument(
        "--targets",
        type=json.loads,
        default=json.dumps(DEFAULT_TARGETS),
        help="JSON dict of {dataset: max_episodes}. Defaults match the data acquisition spec.",
    )
    parser.add_argument("--droid_target", type=int, default=2000)
    parser.add_argument("--min_frames", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--val_cap", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    t0 = time.time()
    extracted_root = Path(args.extracted_root)
    droid_frames_root = Path(args.droid_frames_root)
    droid_meta = Path(args.droid_meta) if args.droid_meta else None

    all_clips: list[dict] = []
    log.info("scanning OXE sub-datasets under %s", extracted_root)
    for ds, target in args.targets.items():
        all_clips.extend(
            scan_oxe_dataset(extracted_root, ds, target, args.min_frames)
        )
    log.info("scanning DROID-100 under %s", droid_frames_root)
    all_clips.extend(
        scan_droid(droid_frames_root, droid_meta, args.droid_target, args.min_frames)
    )

    log.info("total clips before split: %d", len(all_clips))
    _, val_ids = split_val(all_clips, args.val_ratio, args.val_cap, args.seed)

    out_manifest = Path(args.out_manifest)
    out_val_ids = Path(args.out_val_ids)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(all_clips, indent=2))
    out_val_ids.write_text(json.dumps(val_ids, indent=2))

    train_n = len(all_clips) - len(val_ids)
    log.info("wrote %s (%d clips: %d train / %d val) in %.1fs", out_manifest, len(all_clips), train_n, len(val_ids), time.time() - t0)
    log.info("wrote %s (%d val ids)", out_val_ids, len(val_ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
