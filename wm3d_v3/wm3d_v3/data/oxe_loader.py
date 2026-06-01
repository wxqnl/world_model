"""Read OXE tar shards (one episode per .data.pickle), yield (manifest, decoded_episode)."""
from __future__ import annotations
import io
import pickle
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from .manifest import OXEClipRecord


DATASET_ROBOT = {
    "bridge": "widowx",
    "fractal20220817_data": "google_robot",
    "taco_play": "franka",
    "jaco_play": "jaco",
    "kuka": "kuka",
}


@dataclass
class OXEEpisode:
    """Decoded episode (loaded into RAM)."""
    clip_id: str
    images: np.ndarray            # [T, H, W, 3] uint8
    actions: np.ndarray           # [T, 7] float32 (canonical)
    states: np.ndarray            # [T, S] float32 (raw)
    task_text: str
    image_key: str                # which image stream was used


def _decode_image(b) -> np.ndarray:
    if isinstance(b, np.ndarray) and b.dtype == np.uint8:
        return b
    if isinstance(b, bytes):
        img = Image.open(io.BytesIO(b)).convert("RGB")
        return np.asarray(img)
    raise TypeError(f"unrecognized image {type(b)}")


def _pick_image_key(obs: dict) -> str | None:
    for k in ("image", "image_primary", "rgb", "front_image",
              "image_static", "rgb_static", "wrist_image", "image_wrist", "rgb_gripper"):
        if k in obs:
            return k
    for k, v in obs.items():
        if isinstance(v, (bytes, bytearray, np.ndarray)) and "image" in k.lower():
            return k
    return None


def decode_episode(episode_dict: dict, clip_id: str, dataset: str) -> OXEEpisode | None:
    """Convert raw OXE pickle dict into normalized OXEEpisode."""
    from .action_normalize import normalize_action
    steps = episode_dict["steps"]
    if len(steps) < 8:
        return None
    obs0 = steps[0]["observation"]
    image_key = _pick_image_key(obs0)
    if image_key is None:
        return None
    images = []
    actions = []
    states = []
    task_text = ""
    for step in steps:
        obs = step["observation"]
        if image_key not in obs:
            return None
        images.append(_decode_image(obs[image_key]))
        try:
            actions.append(normalize_action(step["action"], dataset))
        except Exception:
            return None
        if "state" in obs:
            states.append(np.asarray(obs["state"], dtype=np.float32).reshape(-1))
        if not task_text and "natural_language_instruction" in obs:
            instr = obs["natural_language_instruction"]
            if isinstance(instr, bytes):
                task_text = instr.decode("utf-8", errors="replace").strip()
            else:
                task_text = str(instr).strip()
    if not task_text:
        task_text = "robot manipulation"
    images_np = np.stack(images)
    actions_np = np.stack(actions)
    states_np = np.stack(states) if states else np.zeros((len(steps), 0), dtype=np.float32)
    return OXEEpisode(
        clip_id=clip_id,
        images=images_np,
        actions=actions_np,
        states=states_np,
        task_text=task_text,
        image_key=image_key,
    )


def iter_episodes_in_tar(
    tar_path: Path, dataset: str
) -> Iterator[tuple[str, dict]]:
    """Yield (member_name, raw_dict) per .data.pickle in tar."""
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith(".data.pickle"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            raw = pickle.load(f)
            yield member.name, raw


def quick_manifest_from_tar(tar_path: Path, dataset: str, shard_idx: int) -> list[OXEClipRecord]:
    """Build manifest records by partially reading each pickle in a tar.

    To keep this cheap, we read each pickle once (necessary to get n_frames
    + task_text + image key).  For 50 tars × ~100 episodes that's ~5000
    pickle loads; on local disk a few minutes per dataset.
    """
    records: list[OXEClipRecord] = []
    robot = DATASET_ROBOT.get(dataset, "unknown")
    for member_name, raw in iter_episodes_in_tar(tar_path, dataset):
        steps = raw.get("steps", [])
        if len(steps) < 8:
            continue
        obs0 = steps[0].get("observation", {})
        image_key = _pick_image_key(obs0)
        if image_key is None:
            continue
        task_text = ""
        for s in steps[:4]:
            instr = s.get("observation", {}).get("natural_language_instruction", b"")
            if isinstance(instr, bytes):
                instr = instr.decode("utf-8", errors="replace").strip()
            if instr:
                task_text = instr
                break
        if not task_text:
            task_text = "robot manipulation"
        # ep id: dataset / shardNNN / pickle name
        base = Path(member_name).stem.replace(".data", "")
        clip_id = f"{dataset}/{shard_idx:05d}/{base}"
        records.append(OXEClipRecord(
            clip_id=clip_id,
            dataset=dataset,
            tar_path=str(tar_path),
            pickle_member=member_name,
            n_frames=len(steps),
            fps=5,
            robot=robot,
            task_text=task_text,
            action_dim=7,
            action_kind="delta_xyz+rpy+gripper",
            image_keys=[image_key],
        ))
    return records
