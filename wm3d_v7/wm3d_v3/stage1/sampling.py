from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import random
from typing import Any, Iterator, Mapping, Protocol, Sequence

from torch.utils.data import Sampler

from wm3d_v3.stage1.action_contract import canonical_dataset_name


DEFAULT_STAGE1_DOMAIN_MASSES: dict[str, float] = {
    "droid": 0.30,
    "bridge": 0.25,
    "fractal20220817_data": 0.15,
    "taco_play": 0.15,
    "jaco_play": 0.10,
    "kuka": 0.05,
}


class Stage1SamplingError(ValueError):
    pass


class _Record(Protocol):
    dataset: str
    clip_id: str


class _WindowDataset(Protocol):
    records: Sequence[_Record]
    index: Sequence[tuple[int, int]]


def _validate_masses(domain_masses: Mapping[str, float]) -> dict[str, float]:
    if not domain_masses:
        raise Stage1SamplingError("domain masses cannot be empty")
    canonical: dict[str, float] = {}
    for raw_domain, raw_mass in domain_masses.items():
        domain = canonical_dataset_name(raw_domain)
        mass = float(raw_mass)
        if not math.isfinite(mass) or mass <= 0.0:
            raise Stage1SamplingError(
                f"domain mass must be finite and positive: {raw_domain}={raw_mass}"
            )
        if domain in canonical:
            raise Stage1SamplingError(f"duplicate canonical domain mass: {domain}")
        canonical[domain] = mass
    if not math.isclose(sum(canonical.values()), 1.0, abs_tol=1e-9):
        raise Stage1SamplingError(
            f"domain masses must sum to 1.0, got {sum(canonical.values())}"
        )
    return canonical


class HierarchicalWindowDistributedSampler(Sampler[int]):
    """Draw domain -> clip -> window without length-based hidden weighting."""

    def __init__(
        self,
        dataset: _WindowDataset,
        *,
        num_samples: int,
        domain_masses: Mapping[str, float] = DEFAULT_STAGE1_DOMAIN_MASSES,
        seed: int = 0,
        rank: int = 0,
        num_replicas: int = 1,
    ) -> None:
        if num_samples <= 0:
            raise Stage1SamplingError("num_samples must be positive")
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise Stage1SamplingError(
                f"invalid distributed layout: rank={rank} replicas={num_replicas}"
            )
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.rank = int(rank)
        self.num_replicas = int(num_replicas)
        self.epoch = 0
        self.cursor = 0
        self.domain_masses = _validate_masses(domain_masses)

        grouped: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for sample_index, (record_index, _) in enumerate(dataset.index):
            try:
                record = dataset.records[record_index]
            except IndexError as exc:
                raise Stage1SamplingError(
                    f"window index references missing record: {record_index}"
                ) from exc
            domain = canonical_dataset_name(record.dataset)
            if domain in self.domain_masses:
                grouped[domain][str(record.clip_id)].append(sample_index)

        missing = sorted(set(self.domain_masses) - set(grouped))
        if missing:
            raise Stage1SamplingError(
                f"missing configured domains with usable windows: {missing}"
            )
        self._windows = {
            domain: {
                clip_id: tuple(sample_indices)
                for clip_id, sample_indices in sorted(clips.items())
            }
            for domain, clips in sorted(grouped.items())
        }
        self._domains = tuple(self.domain_masses)
        self._weights = tuple(self.domain_masses[domain] for domain in self._domains)
        digest = hashlib.sha256()
        for record_index, window_start in dataset.index:
            record = dataset.records[record_index]
            digest.update(
                (
                    f"{canonical_dataset_name(record.dataset)}\0"
                    f"{record.clip_id}\0{window_start}\n"
                ).encode("utf-8")
            )
        self._dataset_fingerprint = digest.hexdigest()

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.cursor = 0

    def _draw_stream(self, count: int) -> list[int]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        stream: list[int] = []
        for domain in rng.choices(self._domains, weights=self._weights, k=count):
            clips = self._windows[domain]
            clip_id = rng.choice(tuple(clips))
            stream.append(rng.choice(clips[clip_id]))
        return stream

    def __iter__(self) -> Iterator[int]:
        global_stream = self._draw_stream(self.num_samples * self.num_replicas)
        rank_stream = global_stream[self.rank :: self.num_replicas]
        for position in range(self.cursor, self.num_samples):
            self.cursor = position + 1
            yield rank_stream[position]

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "seed": self.seed,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "num_samples": self.num_samples,
            "rank": self.rank,
            "num_replicas": self.num_replicas,
            "domain_masses": dict(self.domain_masses),
            "dataset_fingerprint": self._dataset_fingerprint,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "version": 1,
            "seed": self.seed,
            "num_samples": self.num_samples,
            "rank": self.rank,
            "num_replicas": self.num_replicas,
            "domain_masses": dict(self.domain_masses),
            "dataset_fingerprint": self._dataset_fingerprint,
        }
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if state.get(key) != value
        }
        if mismatches:
            raise Stage1SamplingError(
                f"primary sampler state is incompatible: {mismatches}"
            )
        cursor = int(state.get("cursor", -1))
        if not 0 <= cursor <= self.num_samples:
            raise Stage1SamplingError(
                f"primary sampler cursor is outside [0, {self.num_samples}]: {cursor}"
            )
        self.epoch = int(state.get("epoch", 0))
        self.cursor = cursor
