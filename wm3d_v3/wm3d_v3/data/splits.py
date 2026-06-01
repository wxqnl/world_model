"""Dataset split helpers.

The training default is a historical random window split. Episode mode keeps
all windows from the same clip on the same side of the train/val boundary.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


@dataclass(frozen=True)
class ClipSplit:
    train_clip_ids: set[str]
    val_clip_ids: set[str]


def _field(record: Any, name: str) -> Any:
    if isinstance(record, dict):
        return record[name]
    return getattr(record, name)


def _normalise_ids(ids: Iterable[Any] | None) -> set[str] | None:
    if ids is None:
        return None
    return {str(clip_id) for clip_id in ids}


def _clip_ids(records: Sequence[Any]) -> set[str]:
    return {str(_field(record, "clip_id")) for record in records}


def _heldout_clip_ids(records: Sequence[Any], heldout_dataset: str | None) -> set[str]:
    if not heldout_dataset:
        return set()
    return {
        str(_field(record, "clip_id"))
        for record in records
        if str(_field(record, "dataset")) == str(heldout_dataset)
    }


def episode_split(
    records: Sequence[Any],
    *,
    val_frac: float,
    seed: int,
    train_clip_ids: Iterable[Any] | None = None,
    val_clip_ids: Iterable[Any] | None = None,
    heldout_dataset: str | None = None,
) -> ClipSplit:
    """Return deterministic train/val clip-id sets for episode-level splitting.

    ``heldout_dataset`` clips are always placed in validation. When explicit
    train/val ids are not provided, ``val_frac`` is applied to the remaining
    non-heldout clip ids after sorting and seeded shuffling.
    """
    if not 0.0 <= float(val_frac) <= 1.0:
        raise ValueError(f"val_frac must be in [0, 1], got {val_frac}")

    all_ids = _clip_ids(records)
    train_ids = _normalise_ids(train_clip_ids)
    val_ids = _normalise_ids(val_clip_ids)
    heldout_ids = _heldout_clip_ids(records, heldout_dataset)

    if train_ids is not None or val_ids is not None:
        train = set(train_ids) if train_ids is not None else all_ids - set(val_ids or ())
        val = set(val_ids) if val_ids is not None else all_ids - train
        overlap = train & val
        if overlap:
            sample = ", ".join(sorted(overlap)[:5])
            raise ValueError(f"train_clip_ids and val_clip_ids overlap: {sample}")
        val |= heldout_ids
        train -= val
        return ClipSplit(train_clip_ids=train, val_clip_ids=val)

    candidates = sorted(all_ids - heldout_ids)
    rng = random.Random(int(seed))
    rng.shuffle(candidates)
    if not candidates or float(val_frac) == 0.0:
        n_val = 0
    else:
        n_val = max(1, int(len(candidates) * float(val_frac)))
    val = set(candidates[:n_val]) | heldout_ids
    train = set(candidates[n_val:])
    return ClipSplit(train_clip_ids=train, val_clip_ids=val)


def random_window_indices(n: int, *, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Return ``(train_indices, val_indices)`` for the legacy window split."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 <= float(val_frac) <= 1.0:
        raise ValueError(f"val_frac must be in [0, 1], got {val_frac}")
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=generator).tolist()
    n_val = max(1, int(n * float(val_frac)))
    return perm[n_val:], perm[:n_val]


def load_clip_split_file(path: str | Path) -> dict[str, list[str]]:
    """Load a JSON/YAML split file with train/val clip id lists."""
    split_path = Path(path)
    suffix = split_path.suffix.lower()
    text = split_path.read_text()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text)
    else:
        raise ValueError(f"unsupported split file type: {split_path}")
    if not isinstance(data, dict):
        raise ValueError(f"split file must contain a mapping: {split_path}")
    train = data.get("train_clip_ids", data.get("train"))
    val = data.get("val_clip_ids", data.get("val"))
    return {
        "train_clip_ids": list(train or []),
        "val_clip_ids": list(val or []),
    }


def data_split_config(data_cfg: dict) -> dict:
    """Return normalized ``data.split`` config.

    A bare string/path is treated as a split file. Missing config preserves the
    historical random-window behavior unless explicit split keys are present.
    """
    split_cfg = data_cfg.get("split") or {}
    if isinstance(split_cfg, (str, Path)):
        return {"file": str(split_cfg)}
    if not isinstance(split_cfg, dict):
        raise ValueError("data.split must be a mapping or split-file path")
    return dict(split_cfg)


def split_value(data_cfg: dict, split_cfg: dict, key: str, default=None):
    return split_cfg.get(key, data_cfg.get(key, default))


def explicit_clip_ids_from_config(data_cfg: dict, split_cfg: dict) -> tuple[list[str] | None, list[str] | None]:
    """Return explicit train/val clip ids from inline config or split file."""
    train_ids = split_cfg.get("train_clip_ids")
    val_ids = split_cfg.get("val_clip_ids")
    split_file = data_cfg.get("split_file") or split_cfg.get("file") or split_cfg.get("path")
    if split_file:
        file_split = load_clip_split_file(split_file)
        train_ids = train_ids if train_ids is not None else file_split["train_clip_ids"]
        val_ids = val_ids if val_ids is not None else file_split["val_clip_ids"]
    return train_ids, val_ids


def split_mode_from_config(data_cfg: dict) -> str:
    """Return split mode, inferring episode mode for explicit clip/dataset splits."""
    split_cfg = data_split_config(data_cfg)
    has_episode_keys = (
        data_cfg.get("split_file")
        or any(k in split_cfg for k in ("file", "path", "train_clip_ids", "val_clip_ids", "heldout_dataset"))
    )
    return split_cfg.get("mode", "episode" if has_episode_keys else "random_window")


def split_records(records: Sequence[Any], data_cfg: dict) -> tuple[list[Any], list[Any]]:
    """Split manifest records according to ``data.split`` episode settings."""
    split_cfg = data_split_config(data_cfg)
    mode = split_mode_from_config(data_cfg)
    if mode != "episode":
        raise ValueError(f"split_records requires episode mode, got {mode}")
    train_ids, val_ids = explicit_clip_ids_from_config(data_cfg, split_cfg)
    clip_split = episode_split(
        records,
        val_frac=float(split_value(data_cfg, split_cfg, "val_frac", 0.0)),
        seed=int(split_value(data_cfg, split_cfg, "seed", 0)),
        train_clip_ids=train_ids,
        val_clip_ids=val_ids,
        heldout_dataset=split_value(data_cfg, split_cfg, "heldout_dataset"),
    )
    train_records = [r for r in records if str(_field(r, "clip_id")) in clip_split.train_clip_ids]
    val_records = [r for r in records if str(_field(r, "clip_id")) in clip_split.val_clip_ids]
    return train_records, val_records
