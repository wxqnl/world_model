"""Streaming variant of Phase 1 dataset for paper-scale training.

`NextTokenPairs` (in :mod:`src.phase1.dataset`) loads every shard fully into RAM
on first access. That works for the 30-clip prototype but not for the
~30K-clip / 750 GB paper-scale token cache.

This module exposes :class:`StreamingNextTokenPairs`, an :class:`IterableDataset`
that:

* shards the shard list across DDP ranks (deterministic by rank),
* further shards across DataLoader workers within each rank,
* opens one shard at a time, emits its (context, target) pairs, then closes it,
* shuffles shard order per epoch using a deterministic RNG seeded from
  (epoch, rank, worker_id).

Shard format is identical to the prototype (npz with int16-bf16 tokens), so it
reuses :func:`src.phase1.dataset._load_tokens` and :func:`discover_shards`.

Usage::

    ds = StreamingNextTokenPairs(shards, context_len=8, token_pool="mean",
                                 normalize_actions=True, action_stats=stats)
    ds.set_epoch(epoch)              # call before each epoch
    dl = DataLoader(ds, batch_size=B, num_workers=W)
"""
from __future__ import annotations

import random
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .dataset import Shard, _load_tokens


class StreamingNextTokenPairs(IterableDataset):
    """Per-shard streaming next-token-pairs dataset.

    Notes
    -----
    * Length is approximate (sum of valid pair-positions over assigned shards
      for *this* rank). Used by trainers for progress bars; the actual epoch
      length depends on dataloader workers and drop_last semantics.
    * Each shard is opened once, fully iterated, then dropped before the next
      shard is opened. Memory footprint is one shard per worker.
    * Pair emission order within a shard is randomized per epoch.
    """

    def __init__(
        self,
        shards: list[Shard],
        context_len: int,
        token_pool: str = "mean",
        normalize_actions: bool = True,
        action_stats: tuple[np.ndarray, np.ndarray] | None = None,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ):
        self.all_shards = list(shards)
        self.C = int(context_len)
        self.token_pool = token_pool
        self.normalize_actions = bool(normalize_actions)
        self.action_stats = action_stats
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _assigned_shards(self, worker_id: int, num_workers: int) -> list[Shard]:
        # rank-level partition first, then worker-level partition within rank.
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.all_shards)))
        rng.shuffle(order)
        rank_slice = order[self.rank::self.world_size]
        worker_slice = rank_slice[worker_id::max(1, num_workers)]
        return [self.all_shards[i] for i in worker_slice]

    def __len__(self) -> int:
        # rough lower bound: total pairs / world_size
        total = 0
        for sh in self.all_shards:
            total += max(0, sh.n_frames - self.C)
        return max(1, total // self.world_size)

    def _emit_shard(self, sh: Shard, rng: random.Random) -> Iterator[dict]:
        data = np.load(sh.path)
        try:
            tokens_arr = np.asarray(data["tokens"])     # [N, P, D] int16-bf16
            actions_arr = np.asarray(data["actions"])   # [N, A]
        finally:
            data.close()

        N = sh.n_frames
        positions = list(range(self.C - 1, N - 1))
        rng.shuffle(positions)

        if self.normalize_actions and self.action_stats is not None:
            mean, std = self.action_stats
            mean_t = torch.from_numpy(mean).float()
            std_t = torch.from_numpy(std).float()
        else:
            mean_t = std_t = None

        for t in positions:
            ctx_tokens = _load_tokens(tokens_arr[t - self.C + 1 : t + 1])  # [C, P, D]
            tgt_tokens = _load_tokens(tokens_arr[t + 1])                    # [P, D]
            ctx_actions = torch.from_numpy(actions_arr[t - self.C + 1 : t + 1]).float()
            tgt_action = torch.from_numpy(actions_arr[t]).float()
            if mean_t is not None:
                ctx_actions = (ctx_actions - mean_t) / std_t
                tgt_action = (tgt_action - mean_t) / std_t

            sample = {
                "ctx_tokens": ctx_tokens,
                "tgt_tokens": tgt_tokens,
                "ctx_actions": ctx_actions,
                "tgt_action": tgt_action,
            }
            if self.token_pool == "mean":
                sample["ctx_tokens"] = sample["ctx_tokens"].mean(dim=1)  # [C, D]
                sample["tgt_tokens"] = sample["tgt_tokens"].mean(dim=0)  # [D]
            yield sample

    def __iter__(self) -> Iterator[dict]:
        wi = get_worker_info()
        worker_id = 0 if wi is None else wi.id
        num_workers = 1 if wi is None else wi.num_workers
        shards = self._assigned_shards(worker_id, num_workers)
        rng = random.Random(self.seed + self.epoch * 7919 + self.rank * 31 + worker_id)
        order = list(range(len(shards)))
        rng.shuffle(order)
        for si in order:
            yield from self._emit_shard(shards[si], rng)
