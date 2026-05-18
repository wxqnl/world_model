"""Extract frames + actions/state from Open X-Embodiment HF-mirror tars.

Per `docs/DATA_ACQUISITION_SPEC.md` §3 and §5, paper-scale Phase 2 needs
per-episode JPEG frames on disk so the existing `src/phase1/cache.py` VGGT pass
can consume them. The `jxu124/OpenX-Embodiment` HF mirror ships each
sub-dataset as `*.tar` shards of `sample_NNNNN.data.pickle` files, one pickle =
one episode.

This script unpacks one or more shards into

    {out_root}/{dataset}/episode_{XXXXX}/frame_{YYYY}.jpg
    {out_root}/{dataset}/episode_{XXXXX}/episode.npz

The npz sidecar holds a unified 7-dim action [wv_x, wv_y, wv_z, rd_x, rd_y,
rd_z, gripper] and a 7-dim state (zeros where the sub-dataset has no state
slot). Per-episode language goes into a JSON sidecar.

Action conventions per sub-dataset (paper note: actions go unused by the
default `vggt_noact` predictor; we still record them so we can change our mind
later without re-running extraction):

    bridge        wv(3) + rotation_delta(3) + open_gripper(bool→0/1)
    fractal       wv(3) + rotation_delta(3) + gripper_closedness_action(1)
    kuka          wv(3) + rotation_delta(3) + gripper_closedness_action(1)
    jaco_play     wv(3) +     zeros(3)      + gripper_closedness_action(1)
    taco_play     (filled in when we inspect a downloaded shard)

Resumable: if `episode.npz` already exists for an episode the extractor skips
it without re-decoding JPEGs. Safe to interrupt; safe to parallelize across
sub-datasets (each writes to a disjoint directory).
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import pickle
import re
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


log = logging.getLogger("extract_oxe")


# ----------------------------------------------------------------- action mux
def _to_float_array(v: Any) -> np.ndarray:
    """Coerce a numpy scalar / 0-d / 1-d into a 1-d float32 vector."""
    a = np.asarray(v).astype(np.float32, copy=False).ravel()
    return a


def _action_7d(dataset: str, action: dict[str, Any]) -> np.ndarray:
    """Return a 7-dim action vector for one step.

    All sub-datasets in our subset have at least world_vector(3) and a gripper
    signal; only jaco_play lacks rotation_delta. We store zeros there and let
    the predictor see a per-dataset mean/std bias picked up by normalization
    (`src/phase1/dataset.py::compute_action_stats`).
    """
    wv = _to_float_array(action.get("world_vector", np.zeros(3, np.float32)))[:3]
    if wv.size < 3:
        wv = np.pad(wv, (0, 3 - wv.size))

    rd = _to_float_array(action.get("rotation_delta", np.zeros(3, np.float32)))[:3]
    if rd.size < 3:
        rd = np.pad(rd, (0, 3 - rd.size))

    if "open_gripper" in action:
        grip = np.float32(bool(action["open_gripper"]))
    elif "gripper_closedness_action" in action:
        grip = _to_float_array(action["gripper_closedness_action"])[:1]
        grip = float(grip[0]) if grip.size else 0.0
    else:
        grip = 0.0

    return np.concatenate([wv, rd, np.asarray([grip], np.float32)], axis=0).astype(
        np.float32, copy=False
    )


def _state_7d(dataset: str, obs: dict[str, Any]) -> np.ndarray:
    """Return a 7-dim state vector for one step (zeros if unavailable).

    bridge has `obs.state` already at 7 dims. fractal/kuka don't have a single
    "state" slot; we synthesize one from `base_pose_tool_reached` (typically
    7-dim end-effector pose) when present, else zeros. jaco_play has
    end_effector_cartesian_pos(3) + joint_pos(6+); we pack [eef_pos(3), 0, 0,
    0, gripper_proxy=0]. None of this is used by `vggt_noact`, so the fidelity
    bar is "don't crash downstream loaders."
    """
    if "state" in obs:
        s = _to_float_array(obs["state"])
        if s.size >= 7:
            return s[:7].astype(np.float32, copy=False)
        return np.pad(s, (0, 7 - s.size)).astype(np.float32, copy=False)
    if "base_pose_tool_reached" in obs:
        s = _to_float_array(obs["base_pose_tool_reached"])
        if s.size >= 7:
            return s[:7].astype(np.float32, copy=False)
        return np.pad(s, (0, 7 - s.size)).astype(np.float32, copy=False)
    if "end_effector_cartesian_pos" in obs:
        eef = _to_float_array(obs["end_effector_cartesian_pos"])[:3]
        return np.concatenate(
            [eef, np.zeros(7 - eef.size, np.float32)], axis=0
        ).astype(np.float32, copy=False)
    return np.zeros(7, dtype=np.float32)


def _bytes_to_str(v: Any) -> str:
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:
            return ""
    if isinstance(v, str):
        return v
    return ""


# -------------------------------------------------------- per-tar extractor
SHARD_RE = re.compile(r"_(\d{5})\.tar$")


def _shard_index(tar_name: str) -> int:
    m = SHARD_RE.search(tar_name)
    return int(m.group(1)) if m else 0


def extract_tar(
    tar_path: Path,
    dataset: str,
    out_dataset_dir: Path,
    episodes_per_shard_offset: int,
    force: bool = False,
    jpeg_only_first_camera: bool = True,
) -> dict[str, Any]:
    """Extract one .tar shard. Returns a summary dict."""
    t0 = time.time()
    out_dataset_dir.mkdir(parents=True, exist_ok=True)
    n_eps_done = 0
    n_eps_skip = 0
    n_eps_fail = 0
    n_frames_total = 0

    with tarfile.open(tar_path, "r") as t:
        members = sorted(
            (m for m in t.getmembers() if m.name.endswith(".data.pickle")),
            key=lambda m: m.name,
        )
        for local_idx, m in enumerate(members):
            global_ep = episodes_per_shard_offset + local_idx
            ep_dir = out_dataset_dir / f"episode_{global_ep:05d}"
            ep_npz = ep_dir / "episode.npz"
            ep_json = ep_dir / "episode.json"
            if ep_npz.exists() and ep_json.exists() and not force:
                n_eps_skip += 1
                continue

            try:
                f = t.extractfile(m)
                if f is None:
                    n_eps_fail += 1
                    continue
                obj = pickle.load(f)
                steps = obj["steps"]
                image_keys = obj.get("image_list", ["image"]) or ["image"]
                if jpeg_only_first_camera:
                    image_keys = image_keys[:1]

                ep_dir.mkdir(parents=True, exist_ok=True)
                actions = np.zeros((len(steps), 7), dtype=np.float32)
                states = np.zeros((len(steps), 7), dtype=np.float32)
                task_str = ""
                for fi, step in enumerate(steps):
                    obs = step["observation"]
                    # frame (first camera only by default)
                    img_bytes = obs.get(image_keys[0])
                    if not isinstance(img_bytes, (bytes, bytearray)):
                        # rare: some shards may have decoded arrays; encode jpeg
                        n_eps_fail += 1
                        break
                    (ep_dir / f"frame_{fi:04d}.jpg").write_bytes(img_bytes)
                    actions[fi] = _action_7d(dataset, step["action"])
                    states[fi] = _state_7d(dataset, obs)
                    if fi == 0:
                        task_str = _bytes_to_str(
                            obs.get("natural_language_instruction", "")
                        ).strip()
                else:
                    # only runs if the for-loop didn't break
                    np.savez_compressed(ep_npz, actions=actions, states=states)
                    ep_json.write_text(
                        json.dumps(
                            {
                                "dataset": dataset,
                                "episode_index": global_ep,
                                "n_frames": len(steps),
                                "task": task_str,
                                "shard": tar_path.name,
                            },
                            indent=2,
                        )
                    )
                    n_eps_done += 1
                    n_frames_total += len(steps)
            except Exception as e:  # pragma: no cover - defensive
                log.warning("episode %d/%s failed: %s", global_ep, tar_path.name, e)
                n_eps_fail += 1
                continue

    return {
        "tar": tar_path.name,
        "n_done": n_eps_done,
        "n_skip": n_eps_skip,
        "n_fail": n_eps_fail,
        "n_frames": n_frames_total,
        "elapsed": round(time.time() - t0, 1),
    }


# ----------------------------------------------------------------- planning
def plan_tars_for_dataset(
    raw_root: Path, dataset: str, max_tars: int | None
) -> list[Path]:
    d = raw_root / dataset
    if not d.is_dir():
        return []
    tars = sorted(d.glob("*.tar"))
    if max_tars is not None:
        tars = tars[:max_tars]
    return tars


def episodes_per_tar(tar_path: Path) -> int:
    """Cheap header-only count of pickle entries in a tar (no extraction)."""
    n = 0
    with tarfile.open(tar_path, "r") as t:
        for m in t:
            if m.name.endswith(".data.pickle"):
                n += 1
    return n


# ----------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw_root",
        default="/mnt/data/user01/world_model_data/oxe_hf",
        help="root with one subdir per OXE sub-dataset (each holds .tar shards)",
    )
    parser.add_argument(
        "--out_root",
        default="data/raw/oxe_extracted",
        help="output root (one subdir per sub-dataset)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["bridge", "fractal20220817_data", "kuka", "taco_play", "jaco_play"],
    )
    parser.add_argument(
        "--max_tars_per_dataset",
        type=int,
        default=None,
        help="only process the first N tars per sub-dataset (debug)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="parallel tar workers (each tar is ~750 MB; 4 is safe on most disks)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--summary_path",
        default="data/raw/oxe_extracted/_extract_summary.json",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Plan: for each sub-dataset, count episodes in already-extracted dirs to
    # decide the next episode_index offset per shard. We use a fixed
    # episodes-per-tar grid by reading shard headers once.
    plan: list[tuple[Path, str, Path, int]] = []
    log.info("planning shards from %s", raw_root)
    for ds in args.datasets:
        tars = plan_tars_for_dataset(raw_root, ds, args.max_tars_per_dataset)
        if not tars:
            log.info("  %s: no tars on disk yet, skipping", ds)
            continue
        out_ds = out_root / ds
        offset = 0
        for tar in tars:
            n_ep = episodes_per_tar(tar)
            plan.append((tar, ds, out_ds, offset))
            offset += n_ep
        log.info("  %s: %d tars, planned %d episodes", ds, len(tars), offset)

    if not plan:
        log.error("no work to do (no tars found under %s)", raw_root)
        return 1

    summary = {"shards": [], "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    log.info("extracting %d shards with %d workers", len(plan), args.workers)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(extract_tar, tar, ds, out_ds, off, args.force): (tar, ds)
            for (tar, ds, out_ds, off) in plan
        }
        for i, fut in enumerate(as_completed(futures), 1):
            tar, ds = futures[fut]
            try:
                info = fut.result()
            except Exception as e:  # pragma: no cover
                log.error("shard %s failed: %s", tar.name, e)
                info = {"tar": tar.name, "error": str(e)}
            info["dataset"] = ds
            summary["shards"].append(info)
            log.info(
                "[%d/%d] %s/%s  done=%s skip=%s fail=%s frames=%s  %.1fs",
                i,
                len(plan),
                ds,
                tar.name,
                info.get("n_done"),
                info.get("n_skip"),
                info.get("n_fail"),
                info.get("n_frames"),
                info.get("elapsed", -1),
            )

    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_path).write_text(json.dumps(summary, indent=2))
    log.info("summary → %s", args.summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
