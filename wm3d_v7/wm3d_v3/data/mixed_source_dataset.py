"""Source-homogeneous mono/dual-view mixing for WM3D-v7 pretraining.

The legacy OXE cache stores full 2048-D VGGT tokens while the compact V7
cache stores 384-D codec tokens. They are both valid model inputs, but cannot
be stacked in the same PyTorch batch. This module keeps one optimization run
and one scheduler while making every individual batch source-homogeneous.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch
from torch.utils.data import Dataset, Sampler, Subset


class MixedSourceWindowDataset(Dataset):
    """Concatenate named datasets and attach a stable source contract.

    Mono sources receive a zero wrist tensor and an all-false wrist mask.
    Supplying the tensors, instead of omitting them, keeps the multiview fuser
    in the DDP graph on mono batches.
    """

    def __init__(
        self,
        sources: Sequence[tuple[str, Dataset]],
        *,
        mono_sources: Sequence[str] = (),
    ) -> None:
        if not sources:
            raise ValueError("mixed dataset needs at least one source")
        names = [str(name) for name, _dataset in sources]
        if len(names) != len(set(names)):
            raise ValueError(f"mixed dataset source names must be unique: {names}")
        self.datasets = tuple(dataset for _name, dataset in sources)
        self.source_names = tuple(names)
        self.source_id_by_name = {name: index for index, name in enumerate(names)}
        self.mono_sources = frozenset(str(name) for name in mono_sources)
        unknown_mono = self.mono_sources.difference(self.source_id_by_name)
        if unknown_mono:
            raise ValueError(f"unknown mono sources: {sorted(unknown_mono)}")

        self.source_spans: dict[str, tuple[int, int]] = {}
        offset = 0
        for name, dataset in sources:
            length = int(len(dataset))
            if length <= 0:
                raise ValueError(f"mixed dataset source {name!r} is empty")
            self.source_spans[str(name)] = (offset, offset + length)
            offset += length
        self.total_length = offset

    def __len__(self) -> int:
        return self.total_length

    def _locate(self, sample_index: int) -> tuple[int, int]:
        index = int(sample_index)
        if index < 0:
            index += self.total_length
        if index < 0 or index >= self.total_length:
            raise IndexError(sample_index)
        for source_id, source_name in enumerate(self.source_names):
            start, stop = self.source_spans[source_name]
            if start <= index < stop:
                return source_id, index - start
        raise RuntimeError(f"failed to locate mixed sample {sample_index}")

    def __getitem__(self, sample_index: int) -> dict:
        source_id, local_index = self._locate(sample_index)
        source_name = self.source_names[source_id]
        sample = dict(self.datasets[source_id][local_index])
        if "s_in" not in sample:
            raise KeyError(f"mixed source {source_name!r} sample has no s_in")
        if source_name in self.mono_sources:
            if "s_wrist" in sample or "view_mask" in sample:
                raise ValueError(
                    f"mono source {source_name!r} unexpectedly supplied multiview fields"
                )
            state = sample["s_in"]
            if not isinstance(state, torch.Tensor) or state.ndim != 3:
                raise ValueError(
                    f"mono source {source_name!r} s_in must be [T,P,D], got "
                    f"{getattr(state, 'shape', None)}"
                )
            view_mask = torch.zeros((state.shape[0], 2), dtype=torch.bool)
            view_mask[:, 0] = True
            sample["s_wrist"] = torch.zeros_like(state)
            sample["view_mask"] = view_mask
        sample["source_id"] = torch.tensor(source_id, dtype=torch.long)
        return sample

    def set_epoch(self, epoch: int) -> None:
        visited: set[int] = set()
        for dataset in self.datasets:
            target = dataset
            while isinstance(target, Subset):
                target = target.dataset
            if id(target) in visited:
                continue
            visited.add(id(target))
            if hasattr(target, "set_epoch"):
                target.set_epoch(int(epoch))


def partition_v7_compact_dataset(
    dataset: Dataset,
    source_names: Sequence[str],
    *,
    record_field: str = "dataset",
) -> dict[str, Subset]:
    """Partition compact windows by a record field without opening archives."""

    if not hasattr(dataset, "index") or not hasattr(dataset, "records"):
        raise TypeError("compact partitioning requires dataset.index and dataset.records")
    requested = tuple(str(name) for name in source_names)
    buckets: dict[str, list[int]] = {name: [] for name in requested}
    for sample_index, (record_index, _start) in enumerate(dataset.index):
        record = dataset.records[int(record_index)]
        value = (
            record.get(record_field)
            if isinstance(record, dict)
            else getattr(record, record_field)
        )
        value = str(value)
        if value in buckets:
            buckets[value].append(sample_index)
    missing = [name for name, indices in buckets.items() if not indices]
    if missing:
        raise ValueError(
            f"compact dataset has no windows for {record_field}={missing}; "
            f"requested={requested}"
        )
    return {name: Subset(dataset, indices) for name, indices in buckets.items()}


class _PermutationStream:
    """Deterministic no-replacement stream that reshuffles on exhaustion."""

    def __init__(self, length: int, generator: torch.Generator) -> None:
        self.length = int(length)
        self.generator = generator
        self.permutation = torch.empty(0, dtype=torch.long)
        self.position = 0

    def take(self, count: int) -> list[int]:
        remaining = int(count)
        output: list[int] = []
        while remaining > 0:
            if self.position >= int(self.permutation.numel()):
                self.permutation = torch.randperm(
                    self.length, generator=self.generator
                )
                self.position = 0
            available = int(self.permutation.numel()) - self.position
            take = min(remaining, available)
            output.extend(
                self.permutation[self.position : self.position + take].tolist()
            )
            self.position += take
            remaining -= take
        return output


class SourceHomogeneousDistributedBatchSampler(Sampler[list[int]]):
    """Emit exact-ratio, source-homogeneous batches for every DDP rank.

    Source cycle counts are integer exposure counts. For example one OXE and
    four RoboCasa slots is exactly 20:80. The order is reshuffled within every
    cycle using a rank-independent generator, so all ranks execute the same
    source graph on each optimizer step. ``batches_per_source_group`` repeats
    the selected source for a complete gradient-accumulation group, preventing
    one optimizer update from mixing source-specific training contracts.
    Sample permutations are also shared and then disjointly sliced across
    ranks.
    """

    def __init__(
        self,
        dataset: MixedSourceWindowDataset,
        source_cycle_counts: Mapping[str, int],
        *,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        num_batches: int | None = None,
        seed: int = 0,
        batches_per_source_group: int = 1,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.batches_per_source_group = int(batches_per_source_group)
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_replicas <= 0 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("invalid distributed sampler rank/world")
        if self.batches_per_source_group <= 0:
            raise ValueError("batches_per_source_group must be positive")

        normalized: dict[str, int] = {}
        for source_name, raw_count in source_cycle_counts.items():
            name = str(source_name)
            if name not in dataset.source_spans:
                raise ValueError(f"source cycle references unknown source {name!r}")
            count = int(raw_count)
            if count <= 0:
                raise ValueError(
                    f"source cycle count must be positive: {name}={raw_count}"
                )
            normalized[name] = count
        missing = set(dataset.source_names).difference(normalized)
        if missing:
            raise ValueError(f"source cycle omits dataset sources: {sorted(missing)}")
        self.source_cycle_counts = normalized
        self.cycle_source_ids = tuple(
            dataset.source_id_by_name[name]
            for name in dataset.source_names
            for _ in range(normalized[name])
        )
        default_batches = math.ceil(
            len(dataset) / (self.batch_size * self.num_replicas)
        )
        self.num_batches = (
            int(num_batches) if num_batches is not None else default_batches
        )
        if self.num_batches <= 0:
            raise ValueError("num_batches must be positive")
        global_batch = self.batch_size * self.num_replicas
        for name, (start, stop) in dataset.source_spans.items():
            if stop - start < global_batch:
                raise ValueError(
                    f"source {name!r} has {stop - start} samples, fewer than global "
                    f"batch {global_batch}"
                )

    @property
    def source_fractions(self) -> dict[str, float]:
        denominator = float(sum(self.source_cycle_counts.values()))
        return {
            name: float(count) / denominator
            for name, count in self.source_cycle_counts.items()
        }

    def __iter__(self):
        schedule_generator = torch.Generator()
        schedule_generator.manual_seed(self.seed + self.epoch * 1_000_003)
        streams: dict[str, _PermutationStream] = {}
        for source_id, source_name in enumerate(self.dataset.source_names):
            generator = torch.Generator()
            generator.manual_seed(
                self.seed + self.epoch * 1_000_003 + (source_id + 1) * 10_007
            )
            start, stop = self.dataset.source_spans[source_name]
            streams[source_name] = _PermutationStream(stop - start, generator)

        emitted = 0
        global_batch = self.batch_size * self.num_replicas
        while emitted < self.num_batches:
            order = torch.randperm(
                len(self.cycle_source_ids), generator=schedule_generator
            ).tolist()
            for cycle_index in order:
                if emitted >= self.num_batches:
                    break
                source_id = self.cycle_source_ids[cycle_index]
                source_name = self.dataset.source_names[source_id]
                source_start, _source_stop = self.dataset.source_spans[source_name]
                for _ in range(self.batches_per_source_group):
                    if emitted >= self.num_batches:
                        break
                    global_local_indices = streams[source_name].take(global_batch)
                    rank_begin = self.rank * self.batch_size
                    rank_indices = global_local_indices[
                        rank_begin : rank_begin + self.batch_size
                    ]
                    yield [source_start + index for index in rank_indices]
                    emitted += 1

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
