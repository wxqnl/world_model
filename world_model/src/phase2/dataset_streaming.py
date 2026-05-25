"""Streaming variant of Phase 2 sliding-window dataset for paper-scale training.

Same shape/contents as :class:`src.phase2.dataset.GenerativeWindows` but as an
:class:`IterableDataset` so we never hold the full token cache in RAM. See
:mod:`src.phase1.dataset_streaming` for the partitioning rationale; logic here
mirrors it.

Reuses the same npz shards Phase 1 produced.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.phase1.dataset import Shard, _load_tokens, discover_shards


def _load_manifest_tasks(manifest: str | Path) -> dict[int, str]:
    m = json.loads(Path(manifest).read_text())
    return {c["meta"]["episode_index"]: c["meta"].get("task", "") for c in m}


class StreamingGenerativeWindows(IterableDataset):
    """Per-shard streaming sliding-window dataset.

    Yields the same dict layout as ``GenerativeWindows`` so the existing
    ``collate`` works unchanged.
    """

    def __init__(
        self,
        shards: list[Shard],
        seq_len: int,
        stride: int,
        task_by_ep: dict[int, str],
        empty_placeholder: str,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ):
        self.all_shards = list(shards)
        self.T = int(seq_len)
        self.stride = int(stride)
        self.task_by_ep = task_by_ep
        self.empty = empty_placeholder
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _assigned(self, worker_id: int, num_workers: int) -> list[Shard]:
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.all_shards)))
        rng.shuffle(order)
        rank_slice = order[self.rank::self.world_size]
        worker_slice = rank_slice[worker_id::max(1, num_workers)]
        return [self.all_shards[i] for i in worker_slice]

    def __len__(self) -> int:
        total = 0
        for sh in self.all_shards:
            total += max(0, (sh.n_frames - self.T) // self.stride + 1)
        return max(1, total // self.world_size)

    def _emit(self, sh: Shard, rng: random.Random) -> Iterator[dict]:
        data = np.load(sh.path)
        try:
            tokens = np.asarray(data["tokens"])  # [N, P, D]
        finally:
            data.close()
        N = sh.n_frames
        positions = list(range(0, N - self.T + 1, self.stride))
        rng.shuffle(positions)
        ep = sh.episode_index
        task = self.task_by_ep.get(ep, "") or ""
        if not task.strip():
            task = self.empty
        for t0 in positions:
            z = _load_tokens(tokens[t0:t0 + self.T])  # [T, P, D] fp32
            yield {
                "z": z,
                "init": z[0],
                "text": task,
                "shard_idx": torch.tensor(0, dtype=torch.long),
                "t0": torch.tensor(t0, dtype=torch.long),
            }

    def __iter__(self) -> Iterator[dict]:
        wi = get_worker_info()
        worker_id = 0 if wi is None else wi.id
        num_workers = 1 if wi is None else wi.num_workers
        shards = self._assigned(worker_id, num_workers)
        rng = random.Random(self.seed + self.epoch * 7919 + self.rank * 31 + worker_id)
        order = list(range(len(shards)))
        rng.shuffle(order)
        for si in order:
            yield from self._emit(shards[si], rng)


def build_streaming_datasets(
    cfg: dict, rank: int, world_size: int,
) -> tuple[StreamingGenerativeWindows, StreamingGenerativeWindows]:
    shards = discover_shards(cfg["data"]["cache_dir"])
    val_eps = set(cfg["data"]["val_episode_ids"])
    train_sh = [s for s in shards if s.episode_index not in val_eps]
    val_sh = [s for s in shards if s.episode_index in val_eps]
    tasks = _load_manifest_tasks(cfg["data"]["manifest"])
    T = cfg["data"]["seq_len"]; S = cfg["data"]["stride"]
    ph = cfg["data"]["empty_task_placeholder"]
    seed = int(cfg.get("seed", 0))
    return (
        StreamingGenerativeWindows(train_sh, T, S, tasks, ph,
                                   rank=rank, world_size=world_size, seed=seed),
        # Validation runs on rank 0 only (full set), so world_size=1 there.
        StreamingGenerativeWindows(val_sh, T, S, tasks, ph,
                                   rank=0, world_size=1, seed=seed),
    )
