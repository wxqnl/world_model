"""Dataset for Phase 1 predictive training.

Reads cached .npz shards written by :mod:`src.phase1.cache` and yields tuples of
the form::

    (context_tokens,   # [C, P, D]   past C frames
     context_actions,  # [C, A]
     target_token,     # [P, D]      frame t+1
     target_action,    # [A]         action applied at t
     metadata)

Splits are episode-level (indices given in the config).

Notes
-----
* All tokens are stored as int16 bit-views of bf16. We reconstruct bf16 on load
  and cast to fp32 for the model (the head is small; fp32 is fine).
* Depth and extrinsic are also accessible for evaluation, but not part of the
  training inputs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_tokens(arr: np.ndarray) -> torch.Tensor:
    # int16 -> bf16 -> fp32
    return torch.from_numpy(arr).view(torch.bfloat16).to(torch.float32)


@dataclass
class Shard:
    path: Path
    meta: dict
    n_frames: int
    episode_index: int


def discover_shards(cache_dir: str | Path) -> list[Shard]:
    cache = Path(cache_dir)
    out: list[Shard] = []
    for p in sorted(cache.glob("*.npz")):
        meta = json.loads(p.with_suffix(".json").read_text())
        out.append(Shard(path=p, meta=meta,
                         n_frames=int(meta["n_frames"]),
                         episode_index=int(meta["episode_index"])))
    return out


class NextTokenPairs(Dataset):
    """(context → next) pairs across all shards.

    For a clip of length N with context length C, we emit indices
    t = C-1, C, ..., N-2 so that (t-C+1 .. t) is the context and t+1 is the target.
    """

    def __init__(
        self,
        shards: list[Shard],
        context_len: int,
        token_pool: str = "mean",
        normalize_actions: bool = True,
        action_stats: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.shards = shards
        self.C = context_len
        self.token_pool = token_pool
        self.index: list[tuple[int, int]] = []  # (shard_idx, t)
        for si, sh in enumerate(shards):
            N = sh.n_frames
            for t in range(self.C - 1, N - 1):
                self.index.append((si, t))
        self._shard_cache: dict[int, dict[str, np.ndarray]] = {}
        self.normalize_actions = normalize_actions
        self.action_stats = action_stats

    def _load_shard(self, si: int) -> dict[str, np.ndarray]:
        if si in self._shard_cache:
            return self._shard_cache[si]
        data = np.load(self.shards[si].path)
        # force arrays into memory and close the npz handle
        loaded = {k: np.asarray(data[k]) for k in data.files}
        data.close()
        self._shard_cache[si] = loaded
        return loaded

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        si, t = self.index[i]
        sh = self._shard_cache.get(si) or self._load_shard(si)
        C = self.C
        ctx_tokens = _load_tokens(sh["tokens"][t - C + 1 : t + 1])   # [C, P, D]
        tgt_tokens = _load_tokens(sh["tokens"][t + 1])                # [P, D]
        ctx_actions = torch.from_numpy(sh["actions"][t - C + 1 : t + 1]).float()
        tgt_action = torch.from_numpy(sh["actions"][t]).float()
        if self.normalize_actions and self.action_stats is not None:
            mean, std = self.action_stats
            mean_t = torch.from_numpy(mean).float()
            std_t = torch.from_numpy(std).float()
            ctx_actions = (ctx_actions - mean_t) / std_t
            tgt_action = (tgt_action - mean_t) / std_t
        out = {
            "ctx_tokens": ctx_tokens,
            "tgt_tokens": tgt_tokens,
            "ctx_actions": ctx_actions,
            "tgt_action": tgt_action,
            "shard_idx": torch.tensor(si, dtype=torch.long),
            "t": torch.tensor(t, dtype=torch.long),
        }
        if self.token_pool == "mean":
            out["ctx_tokens"] = out["ctx_tokens"].mean(dim=1)   # [C, D]
            out["tgt_tokens"] = out["tgt_tokens"].mean(dim=0)   # [D]
        return out


def compute_action_stats(shards: list[Shard]) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    for sh in shards:
        a = np.load(sh.path)["actions"]
        xs.append(a)
    X = np.concatenate(xs, axis=0)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def split_shards(
    shards: list[Shard],
    val_episode_ids: list[int],
) -> tuple[list[Shard], list[Shard]]:
    train, val = [], []
    vs = set(val_episode_ids)
    for sh in shards:
        (val if sh.episode_index in vs else train).append(sh)
    return train, val
