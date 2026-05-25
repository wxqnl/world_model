"""Phase 2 dataset: sliding-window token sequences + task text.

Yields dicts with keys:
    z         [T, P, D]  full-grid VGGT tokens for T consecutive frames
    init      [P, D]     first frame of the window (generator's init conditioning)
    text      str        task instruction (empty-string clips get a placeholder)
    shard_idx, t         bookkeeping

Reuses the Phase 1 cache (`results/phase1/cache_tokens/*.npz`) — no re-extraction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.phase1.dataset import Shard, discover_shards, _load_tokens


def _load_manifest_tasks(manifest: str | Path) -> dict[int, str]:
    m = json.loads(Path(manifest).read_text())
    return {c["meta"]["episode_index"]: c["meta"].get("task", "") for c in m}


class GenerativeWindows(Dataset):
    def __init__(
        self,
        shards: list[Shard],
        seq_len: int,
        stride: int,
        task_by_ep: dict[int, str],
        empty_placeholder: str,
    ):
        self.shards = shards
        self.T = seq_len
        self.stride = stride
        self.task_by_ep = task_by_ep
        self.empty = empty_placeholder
        self.index: list[tuple[int, int]] = []
        for si, sh in enumerate(shards):
            N = sh.n_frames
            for t0 in range(0, N - seq_len + 1, stride):
                self.index.append((si, t0))
        self._cache: dict[int, np.ndarray] = {}

    def _tokens(self, si: int) -> np.ndarray:
        if si not in self._cache:
            d = np.load(self.shards[si].path)
            self._cache[si] = np.asarray(d["tokens"])
            d.close()
        return self._cache[si]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        si, t0 = self.index[i]
        toks = self._tokens(si)[t0 : t0 + self.T]
        z = _load_tokens(toks)                        # [T, P, D] fp32
        init = z[0]                                   # [P, D]
        ep = self.shards[si].episode_index
        task = self.task_by_ep.get(ep, "") or ""
        if not task.strip():
            task = self.empty
        return {
            "z": z,
            "init": init,
            "text": task,
            "shard_idx": torch.tensor(si, dtype=torch.long),
            "t0": torch.tensor(t0, dtype=torch.long),
        }


def collate(batch: list[dict]) -> dict:
    out = {
        "z": torch.stack([b["z"] for b in batch]),
        "init": torch.stack([b["init"] for b in batch]),
        "text": [b["text"] for b in batch],
        "shard_idx": torch.stack([b["shard_idx"] for b in batch]),
        "t0": torch.stack([b["t0"] for b in batch]),
    }
    return out


def build_datasets(cfg: dict) -> tuple[GenerativeWindows, GenerativeWindows]:
    shards = discover_shards(cfg["data"]["cache_dir"])
    val_eps = set(cfg["data"]["val_episode_ids"])
    train_sh = [s for s in shards if s.episode_index not in val_eps]
    val_sh = [s for s in shards if s.episode_index in val_eps]
    tasks = _load_manifest_tasks(cfg["data"]["manifest"])
    T = cfg["data"]["seq_len"]; S = cfg["data"]["stride"]
    ph = cfg["data"]["empty_task_placeholder"]
    return (
        GenerativeWindows(train_sh, T, S, tasks, ph),
        GenerativeWindows(val_sh, T, S, tasks, ph),
    )
