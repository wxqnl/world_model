from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class OXEClipRecord:
    """One episode reference inside an OXE tar shard."""
    clip_id: str                # e.g. "bridge/00001/ep_000123"
    dataset: str                # bridge / fractal20220817 / taco_play / jaco_play
    tar_path: str
    pickle_member: str          # filename inside the tar
    n_frames: int
    fps: int                    # 5 for fractal, 5 for bridge (approx)
    robot: str                  # widowx / google_robot / franka / jaco
    task_text: str
    action_dim: int = 7
    action_kind: str = "delta_xyz+rpy+gripper"
    image_keys: list[str] = field(default_factory=list)


def write_manifest(path: str | Path, records: list[OXEClipRecord]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")


def read_manifest(path: str | Path) -> list[OXEClipRecord]:
    out: list[OXEClipRecord] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(OXEClipRecord(**d))
    return out
