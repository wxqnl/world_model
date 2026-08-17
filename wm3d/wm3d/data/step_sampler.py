"""Step-addressed exact-ratio sampler shared by every WM3D model profile."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from bisect import bisect_right
import hashlib
from math import gcd

from torch.utils.data import Sampler


class SamplingContractError(ValueError):
    pass


def _seed64(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def _cycle_permutation(length: int, seed: int, cycle: int) -> tuple[int, ...]:
    values = list(range(int(length)))
    state = _seed64(seed, "source-cycle", cycle)
    for index in range(len(values) - 1, 0, -1):
        state = _splitmix64(state)
        target = state % (index + 1)
        values[index], values[target] = values[target], values[index]
    return tuple(values)


def _coprime_multiplier(length: int, seed: int) -> int:
    if length <= 1:
        return 0
    candidate = int(seed % length) | 1
    for _ in range(length):
        if gcd(candidate, length) == 1:
            return candidate
        candidate = (candidate + 2) % length or 1
    raise SamplingContractError(f"failed to find coprime multiplier for {length}")


@dataclass(frozen=True)
class StepAddress:
    optimizer_step: int
    source_name: str
    source_occurrence: int
    cycle: int
    cycle_position: int


class ExactSourceSchedule:
    """Exact integer schedule with a deterministic shuffle inside each cycle."""

    def __init__(
        self,
        source_order: Sequence[str],
        source_weights: Mapping[str, int],
        *,
        seed: int,
    ) -> None:
        self.source_order = tuple(str(value) for value in source_order)
        if not self.source_order or len(set(self.source_order)) != len(self.source_order):
            raise SamplingContractError("source_order must be non-empty and unique")
        if set(self.source_order) != set(source_weights):
            raise SamplingContractError("source weights must exactly match source_order")
        self.weights = {name: int(source_weights[name]) for name in self.source_order}
        if any(value <= 0 for value in self.weights.values()):
            raise SamplingContractError("all source weights must be positive integers")
        self.seed = int(seed)
        expanded: list[int] = []
        for source_id, source_name in enumerate(self.source_order):
            expanded.extend([source_id] * self.weights[source_name])
        self.expanded_source_ids = tuple(expanded)
        self.cycle_length = len(expanded)

    @property
    def fractions(self) -> dict[str, float]:
        return {
            name: self.weights[name] / self.cycle_length for name in self.source_order
        }

    def address(self, optimizer_step: int) -> StepAddress:
        step = int(optimizer_step)
        if step < 0:
            raise SamplingContractError("optimizer step must be non-negative")
        cycle, position = divmod(step, self.cycle_length)
        order = _cycle_permutation(self.cycle_length, self.seed, cycle)
        source_ids = tuple(self.expanded_source_ids[index] for index in order)
        source_id = source_ids[position]
        source_name = self.source_order[source_id]
        occurrence = cycle * self.weights[source_name] + sum(
            earlier == source_id for earlier in source_ids[:position]
        )
        return StepAddress(step, source_name, occurrence, cycle, position)


class AffinePermutation:
    """O(1) no-replacement stream for one source."""

    def __init__(self, length: int, *, seed: int, source_name: str) -> None:
        self.length = int(length)
        self.seed = int(seed)
        self.source_name = str(source_name)
        if self.length <= 0:
            raise SamplingContractError("permutation length must be positive")

    def at(self, ordinal: int) -> int:
        ordinal = int(ordinal)
        if ordinal < 0:
            raise SamplingContractError("sample ordinal must be non-negative")
        epoch, position = divmod(ordinal, self.length)
        epoch_seed = _seed64(self.seed, self.source_name, "epoch", epoch)
        if self.length == 1:
            return 0
        multiplier = _coprime_multiplier(self.length, epoch_seed)
        offset = _splitmix64(epoch_seed) % self.length
        return (multiplier * position + offset) % self.length


class EpisodeLocalPermutation:
    """A deterministic no-replacement permutation that keeps episodes contiguous.

    The episode order and each episode's starting window change every epoch,
    while adjacent global positions stay in the same episode whenever possible.
    This is critical for amortizing raw decode/encoding over all of an
    episode's windows without changing exact-resume addressing.
    """

    def __init__(
        self,
        *,
        source_start: int,
        source_stop: int,
        episode_spans: Sequence[tuple[int, int]],
        seed: int,
        source_name: str,
    ) -> None:
        self.source_start = int(source_start)
        self.length = int(source_stop) - self.source_start
        self.seed = int(seed)
        self.source_name = str(source_name)
        self.spans = tuple((int(start), int(stop)) for start, stop in episode_spans)
        cursor = self.source_start
        for start, stop in self.spans:
            if start != cursor or stop <= start:
                raise SamplingContractError(
                    f"episode spans for {source_name} are not contiguous"
                )
            cursor = stop
        if cursor != int(source_stop) or not self.spans:
            raise SamplingContractError(
                f"episode spans for {source_name} do not cover its source span"
            )
        self._cached_epoch: int | None = None
        self._ordered_spans: tuple[tuple[int, int, int], ...] = ()
        self._cumulative: tuple[int, ...] = ()

    def _epoch_layout(self, epoch: int) -> None:
        if self._cached_epoch == epoch:
            return
        count = len(self.spans)
        permutation = AffinePermutation(
            count,
            seed=_seed64(self.seed, self.source_name, "episode-order"),
            source_name=f"{self.source_name}:episode:{epoch}",
        )
        ordered: list[tuple[int, int, int]] = []
        cumulative: list[int] = []
        total = 0
        for position in range(count):
            span_index = permutation.at(epoch * count + position)
            start, stop = self.spans[span_index]
            span_length = stop - start
            rotation = _splitmix64(
                _seed64(self.seed, self.source_name, "window", epoch, span_index)
            ) % span_length
            ordered.append((start, stop, int(rotation)))
            total += span_length
            cumulative.append(total)
        if total != self.length:
            raise AssertionError("episode-local permutation lost source windows")
        self._cached_epoch = epoch
        self._ordered_spans = tuple(ordered)
        self._cumulative = tuple(cumulative)

    def at(self, absolute_position: int) -> int:
        if absolute_position < 0:
            raise SamplingContractError("permutation position must be non-negative")
        epoch, position = divmod(int(absolute_position), self.length)
        self._epoch_layout(epoch)
        episode_index = bisect_right(self._cumulative, position)
        previous = 0 if episode_index == 0 else self._cumulative[episode_index - 1]
        start, stop, rotation = self._ordered_spans[episode_index]
        within = position - previous
        return start - self.source_start + ((within + rotation) % (stop - start))


class RankEpisodeLocalPermutation:
    """Episode-local stream with a stable, disjoint episode set per rank.

    A global contiguous episode stream makes every rank land in the same raw
    episode.  Each rank then materializes the full episode but consumes only
    its small slice of windows.  Assigning whole episodes to ranks keeps the
    global sample set disjoint while allowing each materialization to be reused
    for every window from that episode.
    """

    def __init__(
        self,
        *,
        source_start: int,
        source_stop: int,
        episode_spans: Sequence[tuple[int, int]],
        world_size: int,
        rank: int,
        minimum_windows_per_rank: int,
        seed: int,
        source_name: str,
    ) -> None:
        self.source_start = int(source_start)
        self.seed = int(seed)
        self.source_name = str(source_name)
        self.world_size = int(world_size)
        self.rank = int(rank)
        spans = tuple((int(start), int(stop)) for start, stop in episode_spans)
        cursor = self.source_start
        for start, stop in spans:
            if start != cursor or stop <= start:
                raise SamplingContractError(
                    f"episode spans for {source_name} are not contiguous"
                )
            cursor = stop
        if cursor != int(source_stop) or not spans:
            raise SamplingContractError(
                f"episode spans for {source_name} do not cover its source span"
            )
        if len(spans) < self.world_size:
            raise SamplingContractError(
                f"source {source_name} has {len(spans)} episodes for "
                f"{self.world_size} ranks"
            )

        # Group similarly sized episodes before assigning one member of each
        # group to every rank.  All ranks then enter/leave raw episodes at
        # approximately the same time instead of waiting for a random long
        # encoder job on one straggling rank.
        indices_by_length: dict[int, list[int]] = {}
        for span_index, (start, stop) in enumerate(spans):
            indices_by_length.setdefault(stop - start, []).append(span_index)
        ordered_indices: list[int] = []
        for span_length in sorted(indices_by_length):
            indices = indices_by_length[span_length]
            count = len(indices)
            order_seed = _seed64(
                self.seed, self.source_name, "episode-length", span_length
            )
            multiplier = _coprime_multiplier(count, order_seed)
            offset = _splitmix64(order_seed) % count if count > 1 else 0
            ordered_indices.extend(
                indices[(multiplier * position + offset) % count]
                if count > 1
                else indices[0]
                for position in range(count)
            )

        groups: list[tuple[tuple[int, int] | None, ...]] = []
        windows_by_rank = [0] * self.world_size
        for group_index, group_start in enumerate(
            range(0, len(ordered_indices), self.world_size)
        ):
            members: list[tuple[int, int] | None] = [None] * self.world_size
            rank_shift = _splitmix64(
                _seed64(self.seed, self.source_name, "episode-rank", group_index)
            ) % self.world_size
            for position, span_index in enumerate(
                ordered_indices[group_start : group_start + self.world_size]
            ):
                owner = (position + rank_shift) % self.world_size
                start, stop = spans[span_index]
                members[owner] = (start, stop)
                windows_by_rank[owner] += stop - start
            groups.append(tuple(members))
        if min(windows_by_rank) < int(minimum_windows_per_rank):
            raise SamplingContractError(
                f"source {source_name} cannot provide {minimum_windows_per_rank} "
                "disjoint local windows to every rank"
            )

        self.groups = tuple(groups)
        self.spans = tuple(
            group[self.rank]
            for group in self.groups
            if group[self.rank] is not None
        )
        self.length = windows_by_rank[self.rank]
        self._cached_epoch: int | None = None
        self._ordered_spans: tuple[tuple[int, int, int], ...] = ()
        self._cumulative: tuple[int, ...] = ()

    def _epoch_layout(self, epoch: int) -> None:
        if self._cached_epoch == epoch:
            return
        count = len(self.groups)
        epoch_seed = _seed64(
            self.seed, self.source_name, "rank-episode-group-order", epoch
        )
        multiplier = _coprime_multiplier(count, epoch_seed)
        offset = _splitmix64(epoch_seed) % count if count > 1 else 0
        ordered: list[tuple[int, int, int]] = []
        cumulative: list[int] = []
        total = 0
        for position in range(count):
            group_index = (multiplier * position + offset) % count if count > 1 else 0
            span = self.groups[group_index][self.rank]
            if span is None:
                continue
            start, stop = span
            span_length = stop - start
            rotation = _splitmix64(
                _seed64(
                    self.seed,
                    self.source_name,
                    "rank-window",
                    self.rank,
                    epoch,
                    start,
                    stop,
                )
            ) % span_length
            ordered.append((start, stop, int(rotation)))
            total += span_length
            cumulative.append(total)
        if total != self.length:
            raise AssertionError("rank episode-local permutation lost source windows")
        self._cached_epoch = epoch
        self._ordered_spans = tuple(ordered)
        self._cumulative = tuple(cumulative)

    def at(self, absolute_position: int) -> int:
        if absolute_position < 0:
            raise SamplingContractError("permutation position must be non-negative")
        epoch, position = divmod(int(absolute_position), self.length)
        self._epoch_layout(epoch)
        episode_index = bisect_right(self._cumulative, position)
        previous = 0 if episode_index == 0 else self._cumulative[episode_index - 1]
        start, stop, rotation = self._ordered_spans[episode_index]
        within = position - previous
        return start - self.source_start + ((within + rotation) % (stop - start))


class StepAddressedBatchSampler(Sampler[list[int]]):
    """Reconstruct local batches from the optimizer step without cursor state."""

    def __init__(
        self,
        source_spans: Mapping[str, tuple[int, int]],
        source_order: Sequence[str],
        source_weights: Mapping[str, int],
        *,
        world_size: int,
        rank: int,
        micro_batch_size: int,
        gradient_accumulation: int,
        start_optimizer_step: int,
        num_optimizer_steps: int,
        seed: int,
        source_episode_spans: Mapping[
            str, Sequence[tuple[int, int]]
        ] | None = None,
    ) -> None:
        self.source_spans = {
            str(name): (int(span[0]), int(span[1])) for name, span in source_spans.items()
        }
        self.schedule = ExactSourceSchedule(source_order, source_weights, seed=seed)
        if set(self.source_spans) != set(self.schedule.source_order):
            raise SamplingContractError("source spans must exactly match scheduled sources")
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.micro_batch_size = int(micro_batch_size)
        self.gradient_accumulation = int(gradient_accumulation)
        self.start_optimizer_step = int(start_optimizer_step)
        self.num_optimizer_steps = int(num_optimizer_steps)
        self.seed = int(seed)
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise SamplingContractError("invalid rank/world size")
        if self.micro_batch_size <= 0 or self.gradient_accumulation <= 0:
            raise SamplingContractError("batch size and accumulation must be positive")
        if self.start_optimizer_step < 0 or self.num_optimizer_steps <= 0:
            raise SamplingContractError("invalid optimizer-step interval")
        self.global_micro_batch = self.world_size * self.micro_batch_size
        self.global_batch = self.global_micro_batch * self.gradient_accumulation
        self._rank_episode_partitioned_sources: set[str] = set()
        self._permutations: dict[
            str,
            AffinePermutation | EpisodeLocalPermutation | RankEpisodeLocalPermutation,
        ] = {}
        for source_name, (start, stop) in self.source_spans.items():
            length = stop - start
            if start < 0 or stop <= start:
                raise SamplingContractError(f"invalid span for {source_name}")
            if length < self.global_batch:
                raise SamplingContractError(
                    f"source {source_name} has {length} windows, below global batch "
                    f"{self.global_batch}; a no-replacement optimizer step cannot be formed"
                )
            if source_episode_spans is None:
                self._permutations[source_name] = AffinePermutation(
                    length, seed=self.seed, source_name=source_name
                )
            else:
                if set(source_episode_spans) != set(self.source_spans):
                    raise SamplingContractError(
                        "episode-local spans must exactly match scheduled sources"
                    )
                spans = source_episode_spans[source_name]
                if len(spans) < self.world_size:
                    self._permutations[source_name] = EpisodeLocalPermutation(
                        source_start=start,
                        source_stop=stop,
                        episode_spans=spans,
                        seed=self.seed,
                        source_name=source_name,
                    )
                    continue
                try:
                    permutation = RankEpisodeLocalPermutation(
                        source_start=start,
                        source_stop=stop,
                        episode_spans=spans,
                        world_size=self.world_size,
                        rank=self.rank,
                        minimum_windows_per_rank=(
                            self.micro_batch_size * self.gradient_accumulation
                        ),
                        seed=self.seed,
                        source_name=source_name,
                    )
                except SamplingContractError as exc:
                    if "disjoint local windows" not in str(exc):
                        raise
                    self._permutations[source_name] = EpisodeLocalPermutation(
                        source_start=start,
                        source_stop=stop,
                        episode_spans=spans,
                        seed=self.seed,
                        source_name=source_name,
                    )
                else:
                    self._permutations[source_name] = permutation
                    self._rank_episode_partitioned_sources.add(source_name)

    def __len__(self) -> int:
        return self.num_optimizer_steps * self.gradient_accumulation

    def describe_step(self, optimizer_step: int) -> dict[str, int | str]:
        address = self.schedule.address(optimizer_step)
        start, stop = self.source_spans[address.source_name]
        return {
            "optimizer_step": address.optimizer_step,
            "source_name": address.source_name,
            "source_occurrence": address.source_occurrence,
            "source_length": stop - start,
            "global_batch": self.global_batch,
            "cycle": address.cycle,
            "cycle_position": address.cycle_position,
        }

    def __iter__(self) -> Iterator[list[int]]:
        stop_step = self.start_optimizer_step + self.num_optimizer_steps
        for step in range(self.start_optimizer_step, stop_step):
            address = self.schedule.address(step)
            source_start, _ = self.source_spans[address.source_name]
            permutation = self._permutations[address.source_name]
            rank_episode_partitioned = (
                address.source_name in self._rank_episode_partitioned_sources
            )
            if rank_episode_partitioned:
                local_batch = self.micro_batch_size * self.gradient_accumulation
                step_base = address.source_occurrence * local_batch
            else:
                step_base = address.source_occurrence * self.global_batch
            for micro_step in range(self.gradient_accumulation):
                if rank_episode_partitioned:
                    rank_base = step_base + micro_step * self.micro_batch_size
                else:
                    rank_base = (
                        step_base
                        + micro_step * self.global_micro_batch
                        + self.rank * self.micro_batch_size
                    )
                yield [
                    source_start + permutation.at(rank_base + local_index)
                    for local_index in range(self.micro_batch_size)
                ]
