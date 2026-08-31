from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterator, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from wm3d_v3.stage1.action_contract import canonical_dataset_name


PAIR_OFFSET = 8
WINDOW_FRAMES = 24
OVERLAP_FRAMES = 16
RETAINED_FRAMES = 8
DEFAULT_MIN_COSINE = 0.999
DEFAULT_MAX_NORMALIZED_MAD = 0.08
METRIC_COMPARISON_EPS = 1e-7

DEFAULT_STAGE1_PAIR_DOMAIN_MASSES: dict[str, float] = {
    "droid": 0.30,
    "bridge": 0.35,
    "fractal20220817_data": 0.15,
    "taco_play": 0.10,
    "jaco_play": 0.10,
}


class PairedK8Error(ValueError):
    pass


class _Record(Protocol):
    dataset: str
    clip_id: str


class _WindowConfig(Protocol):
    cache_root: Path
    window_geom_subdir: str
    use_window_tokens: bool
    T: int
    k: int


class _WindowShardReader(Protocol):
    def open_npz(self, name: str) -> Any: ...


class _BaseWindowDataset(Protocol):
    records: Sequence[_Record]
    index: Sequence[tuple[int, int]]
    cfg: _WindowConfig
    _window_shards: _WindowShardReader | None

    def __getitem__(self, sample_index: int) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class PairSectionGauge:
    mean_cosine: float
    normalized_mad: float
    passed: bool


@dataclass(frozen=True)
class PairGauge:
    combined: PairSectionGauge | None
    retained: PairSectionGauge | None
    forecast: PairSectionGauge | None
    accepted: bool
    rejection_reasons: tuple[str, ...]
    dataset: str = ""
    clip_id: str = ""
    start_a: int = -1
    start_b: int = -1


@dataclass(frozen=True)
class PairRecord:
    a_sample_index: int
    b_sample_index: int
    record_index: int
    start_a: int
    start_b: int
    dataset: str
    clip_id: str
    gauge: PairGauge


PairedK8Entry = PairRecord


@dataclass(frozen=True)
class PairSamplingAudit:
    draws: int
    domain_fractions: dict[str, float]
    max_clip_deviation: float
    max_start_deviation: float
    deterministic_rank_partition: bool
    rejection_adjusted_coverage: dict[str, float]
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.violations


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_frozen_cache_identity(
    base: _BaseWindowDataset,
    report: Mapping[str, Any],
) -> None:
    identity = report.get("cache_identity")
    if not isinstance(identity, Mapping):
        raise PairedK8Error("frozen pair report has no cache identity")
    shard_index = identity.get("shard_index")
    shard_root = identity.get("shard_root")
    referenced_shards = identity.get("referenced_shards")
    if not isinstance(shard_index, Mapping) or not isinstance(shard_root, Mapping):
        raise PairedK8Error("frozen pair report has incomplete shard identity")
    if not isinstance(referenced_shards, list) or not referenced_shards:
        raise PairedK8Error("frozen pair report has no referenced shard identities")

    cfg_index = Path(getattr(base.cfg, "window_geom_shard_index", ""))
    cfg_root = Path(getattr(base.cfg, "window_geom_shard_root", ""))
    expected_index = Path(str(shard_index.get("resolved_path") or shard_index.get("path")))
    expected_root = Path(str(shard_root.get("resolved_path") or shard_root.get("path")))
    if cfg_index.resolve(strict=True) != expected_index.resolve(strict=True):
        raise PairedK8Error("frozen pair report shard index differs from runtime")
    if cfg_root.resolve(strict=True) != expected_root.resolve(strict=True):
        raise PairedK8Error("frozen pair report shard root differs from runtime")
    if _sha256_file(cfg_index) != str(shard_index.get("sha256")):
        raise PairedK8Error("frozen pair report shard index digest differs from runtime")

    for expected in referenced_shards:
        if not isinstance(expected, Mapping):
            raise PairedK8Error("invalid referenced shard identity")
        path = Path(str(expected.get("resolved_path") or expected.get("path")))
        stat = path.stat()
        actual = {
            "device": int(stat.st_dev),
            "inode": int(stat.st_ino),
            "mtime_ns": int(stat.st_mtime_ns),
            "size_bytes": int(stat.st_size),
        }
        required = {key: int(expected[key]) for key in actual}
        if actual != required:
            raise PairedK8Error(f"frozen pair cache shard identity changed: {path}")


def _pairs_from_frozen_report(
    base: _BaseWindowDataset,
    report_path: str | Path,
    *,
    pair_offset: int,
    domain_masses: Mapping[str, float],
    min_cosine: float,
    max_normalized_mad: float,
) -> tuple[tuple[PairRecord, ...], tuple[PairGauge, ...]]:
    path = Path(report_path)
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("passed") is not True or report.get("status") != "passed":
        raise PairedK8Error("frozen pair report is not passed")
    settings = report.get("settings")
    if not isinstance(settings, Mapping):
        raise PairedK8Error("frozen pair report has no settings")
    expected_masses = _validate_domain_masses(domain_masses)
    report_masses = _validate_domain_masses(settings.get("domain_masses") or {})
    if report_masses != expected_masses:
        raise PairedK8Error("frozen pair report domain masses differ from runtime")
    expected_settings = {
        "pair_offset": int(pair_offset),
        "min_cosine": float(min_cosine),
        "max_normalized_mad": float(max_normalized_mad),
        "window_frames": WINDOW_FRAMES,
    }
    for key, expected in expected_settings.items():
        if settings.get(key) != expected:
            raise PairedK8Error(
                f"frozen pair report setting differs: {key}={settings.get(key)!r} "
                f"expected={expected!r}"
            )
    table = report.get("pair_table")
    if not isinstance(table, Mapping) or not isinstance(table.get("entries"), list):
        raise PairedK8Error("frozen pair report has no exact pair table")
    entries = table["entries"]
    if table.get("entries_sha256") != _sha256_json(entries):
        raise PairedK8Error("frozen pair table digest is invalid")
    _validate_frozen_cache_identity(base, report)

    logical_windows: dict[tuple[str, int], tuple[int, int]] = {}
    for sample_index, raw_entry in enumerate(base.index):
        record_index, start = int(raw_entry[0]), int(raw_entry[1])
        record = base.records[record_index]
        key = (str(record.clip_id), start)
        previous = logical_windows.get(key)
        if previous is not None:
            previous_record = base.records[previous[1]]
            if previous_record != record:
                raise PairedK8Error(f"conflicting duplicate logical window: {key}")
            continue
        logical_windows[key] = (sample_index, record_index)

    passed_section = PairSectionGauge(
        mean_cosine=1.0,
        normalized_mad=0.0,
        passed=True,
    )
    pairs: list[PairRecord] = []
    gauges: list[PairGauge] = []
    seen: set[tuple[str, int, int]] = set()
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise PairedK8Error("invalid frozen pair entry")
        dataset = canonical_dataset_name(str(raw.get("dataset")))
        if dataset not in expected_masses:
            continue
        clip_id = str(raw.get("clip_id"))
        start_a = int(raw.get("start_a"))
        start_b = int(raw.get("start_b"))
        if start_b != start_a + PAIR_OFFSET:
            raise PairedK8Error(f"invalid frozen pair offset: {clip_id} {start_a}/{start_b}")
        pair_key = (clip_id, start_a, start_b)
        if pair_key in seen:
            raise PairedK8Error(f"duplicate frozen pair entry: {pair_key}")
        seen.add(pair_key)
        a = logical_windows.get((clip_id, start_a))
        b = logical_windows.get((clip_id, start_b))
        if a is None or b is None:
            continue
        if a[1] != b[1]:
            raise PairedK8Error(f"frozen pair resolves across records: {pair_key}")
        record = base.records[a[1]]
        if canonical_dataset_name(str(record.dataset)) != dataset:
            raise PairedK8Error(f"frozen pair dataset differs from runtime: {pair_key}")
        gauge = PairGauge(
            combined=passed_section,
            retained=passed_section,
            forecast=passed_section,
            accepted=True,
            rejection_reasons=(),
            dataset=dataset,
            clip_id=clip_id,
            start_a=start_a,
            start_b=start_b,
        )
        pairs.append(
            PairRecord(
                a_sample_index=a[0],
                b_sample_index=b[0],
                record_index=a[1],
                start_a=start_a,
                start_b=start_b,
                dataset=dataset,
                clip_id=clip_id,
                gauge=gauge,
            )
        )
        gauges.append(gauge)
    accepted_domains = {pair.dataset for pair in pairs}
    missing = sorted(set(expected_masses) - accepted_domains)
    if missing:
        raise PairedK8Error(
            f"frozen pair report has zero local accepted pairs for domains={missing}"
        )
    return tuple(pairs), tuple(gauges)


def _validate_domain_masses(domain_masses: Mapping[str, float]) -> dict[str, float]:
    if not domain_masses:
        raise PairedK8Error("pair domain masses cannot be empty")
    canonical: dict[str, float] = {}
    for raw_domain, raw_mass in domain_masses.items():
        domain = canonical_dataset_name(raw_domain)
        mass = float(raw_mass)
        if not math.isfinite(mass) or mass <= 0.0:
            raise PairedK8Error(
                f"pair domain mass must be finite and positive: {raw_domain}={raw_mass}"
            )
        if domain in canonical:
            raise PairedK8Error(f"duplicate canonical pair domain mass: {domain}")
        canonical[domain] = mass
    total = sum(canonical.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise PairedK8Error(f"pair domain masses must sum to 1.0, got {total}")
    return canonical


def _safe_clip_id(clip_id: str) -> str:
    return str(clip_id).replace("/", "__")


def _window_member_name(clip_id: str, start: int) -> str:
    return f"{_safe_clip_id(clip_id)}__start_{int(start):06d}.npz"


def _scalar_int(archive: Any, key: str) -> int | None:
    if key not in archive.files:
        return None
    value = np.asarray(archive[key])
    if value.size != 1:
        raise PairedK8Error(f"window cache metadata {key!r} must be scalar")
    return int(value.reshape(-1)[0])


def _pooled_from_archive(
    archive: Any,
    *,
    member_name: str,
    start: int,
    expected_t: int,
    expected_k: int,
) -> Tensor:
    cached_start = _scalar_int(archive, "start")
    if cached_start is not None and cached_start != start:
        raise PairedK8Error(
            f"window cache start mismatch for {member_name}: "
            f"expected={start} cached={cached_start}"
        )
    cached_t = _scalar_int(archive, "T")
    if cached_t is not None and cached_t != expected_t:
        raise PairedK8Error(
            f"window cache T mismatch for {member_name}: "
            f"expected={expected_t} cached={cached_t}"
        )
    cached_k = _scalar_int(archive, "k")
    if cached_k is not None and cached_k != expected_k:
        raise PairedK8Error(
            f"window cache k mismatch for {member_name}: "
            f"expected={expected_k} cached={cached_k}"
        )

    pooled_key = next(
        (key for key in ("pooled", "vggt_pooled") if key in archive.files),
        None,
    )
    if pooled_key is None:
        raise PairedK8Error(f"window cache has no pooled tokens: {member_name}")
    pooled = np.asarray(archive[pooled_key])
    expected_frames = expected_t + expected_k
    if pooled.ndim < 2 or pooled.shape[0] != expected_frames:
        raise PairedK8Error(
            f"window cache must contain exactly {expected_frames} pooled-token "
            f"frames: {member_name} shape={pooled.shape}"
        )
    if not np.issubdtype(pooled.dtype, np.number):
        raise PairedK8Error(
            f"window pooled tokens must be numeric: {member_name} dtype={pooled.dtype}"
        )
    pooled_f32 = np.array(pooled, dtype=np.float32, copy=True)
    if not np.isfinite(pooled_f32).all():
        raise PairedK8Error(
            f"window pooled tokens contain non-finite values: {member_name}"
        )
    return torch.from_numpy(pooled_f32)


def load_window_pooled_tokens(
    dataset: _BaseWindowDataset,
    record_index: int | None = None,
    start: int | None = None,
    *,
    sample_index: int | None = None,
) -> Tensor:
    """Read one 24-frame pooled-token window from the real NPZ/tar cache."""

    if sample_index is not None:
        if record_index is not None or start is not None:
            raise PairedK8Error("provide sample_index or record_index/start, not both")
        try:
            raw_record_index, raw_start = dataset.index[int(sample_index)]
        except (IndexError, TypeError) as exc:
            raise PairedK8Error(f"invalid window sample index: {sample_index}") from exc
        record_index, start = int(raw_record_index), int(raw_start)
    if record_index is None or start is None:
        raise PairedK8Error("record_index and start are required")
    record_index = int(record_index)
    start = int(start)
    if record_index < 0 or record_index >= len(dataset.records):
        raise PairedK8Error(f"invalid record index: {record_index}")

    cfg = getattr(dataset, "cfg", None)
    if cfg is None:
        raise PairedK8Error("paired-K8 gauge requires a cache-backed dataset with cfg")
    if not bool(getattr(cfg, "use_window_tokens", False)):
        raise PairedK8Error(
            "paired-K8 gauge requires real window pooled-token cache "
            "(use_window_tokens=True)"
        )
    expected_t = int(getattr(cfg, "T", 0))
    expected_k = int(getattr(cfg, "k", 0))
    if expected_t != 16 or expected_k != 8:
        raise PairedK8Error(
            f"paired-K8 gauge requires T=16 and k=8, got T={expected_t} k={expected_k}"
        )

    record = dataset.records[record_index]
    member_name = _window_member_name(record.clip_id, start)
    cache_root = Path(cfg.cache_root)
    subdir = str(cfg.window_geom_subdir)
    local_path = cache_root / subdir / member_name

    archive = None
    try:
        if local_path.is_file():
            archive = np.load(local_path, allow_pickle=False)
        else:
            shard_reader = getattr(dataset, "_window_shards", None)
            if shard_reader is not None:
                archive = shard_reader.open_npz(member_name)
        if archive is None:
            raise PairedK8Error(
                f"missing real window pooled-token cache: {local_path} "
                f"(tar member {member_name!r} also unavailable)"
            )
        return _pooled_from_archive(
            archive,
            member_name=member_name,
            start=start,
            expected_t=expected_t,
            expected_k=expected_k,
        )
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, PairedK8Error):
            raise
        raise PairedK8Error(
            f"failed to read window pooled-token cache {member_name}: {exc}"
        ) from exc
    finally:
        if archive is not None and hasattr(archive, "close"):
            archive.close()


def _as_window_tensor(tokens: Tensor | np.ndarray, name: str) -> Tensor:
    try:
        tensor = torch.as_tensor(tokens, dtype=torch.float32, device="cpu")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PairedK8Error(f"{name} pooled tokens are not tensor-like") from exc
    if tensor.ndim < 2 or tensor.shape[0] != WINDOW_FRAMES:
        raise PairedK8Error(
            f"{name} must contain exactly 24 cached frames; "
            f"got shape={tuple(tensor.shape)}"
        )
    if tensor[0].numel() == 0:
        raise PairedK8Error(f"{name} pooled-token frames cannot be empty")
    if not torch.isfinite(tensor).all():
        raise PairedK8Error(f"{name} pooled tokens contain non-finite values")
    return tensor


def _section_gauge(
    left: Tensor,
    right: Tensor,
    *,
    min_cosine: float,
    max_normalized_mad: float,
) -> PairSectionGauge:
    if left.shape != right.shape:
        raise PairedK8Error(
            f"overlap token shapes differ: {tuple(left.shape)} != {tuple(right.shape)}"
        )
    if torch.equal(left, right):
        return PairSectionGauge(
            mean_cosine=1.0,
            normalized_mad=0.0,
            passed=True,
        )

    left_flat = left.reshape(left.shape[0], -1)
    right_flat = right.reshape(right.shape[0], -1)
    dot = (left_flat * right_flat).sum(dim=1)
    left_norm = torch.linalg.vector_norm(left_flat, dim=1)
    right_norm = torch.linalg.vector_norm(right_flat, dim=1)
    norm_product = left_norm * right_norm
    cosine = torch.zeros_like(dot)
    nonzero = norm_product > 0
    cosine[nonzero] = dot[nonzero] / norm_product[nonzero]
    both_zero = (left_norm == 0) & (right_norm == 0)
    cosine[both_zero] = 1.0
    mean_cosine = float(cosine.clamp(-1.0, 1.0).mean().item())

    mad = (left_flat - right_flat).abs().mean()
    magnitude = 0.5 * (left_flat.abs().mean() + right_flat.abs().mean())
    if float(magnitude.item()) == 0.0:
        normalized_mad = 0.0 if float(mad.item()) == 0.0 else math.inf
    else:
        normalized_mad = float((mad / magnitude).item())
    passed = (
        math.isfinite(mean_cosine)
        and math.isfinite(normalized_mad)
        and mean_cosine + METRIC_COMPARISON_EPS >= min_cosine
        and normalized_mad <= max_normalized_mad + METRIC_COMPARISON_EPS
    )
    return PairSectionGauge(
        mean_cosine=mean_cosine,
        normalized_mad=normalized_mad,
        passed=passed,
    )


def compare_pair_tokens(
    tokens_a: Tensor | np.ndarray,
    tokens_b: Tensor | np.ndarray,
    *,
    min_cosine: float = DEFAULT_MIN_COSINE,
    max_normalized_mad: float = DEFAULT_MAX_NORMALIZED_MAD,
) -> PairGauge:
    """Gauge A[8:24] against B[0:16], including both eight-frame halves."""

    min_cosine = float(min_cosine)
    max_normalized_mad = float(max_normalized_mad)
    if not math.isfinite(min_cosine) or not -1.0 <= min_cosine <= 1.0:
        raise PairedK8Error(f"min_cosine must be finite in [-1, 1]: {min_cosine}")
    if not math.isfinite(max_normalized_mad) or max_normalized_mad < 0.0:
        raise PairedK8Error(
            f"max_normalized_mad must be finite and non-negative: {max_normalized_mad}"
        )

    a = _as_window_tensor(tokens_a, "A")
    b = _as_window_tensor(tokens_b, "B")
    if a.shape[1:] != b.shape[1:]:
        raise PairedK8Error(
            f"A/B pooled-token tails differ: {tuple(a.shape[1:])} != "
            f"{tuple(b.shape[1:])}"
        )

    overlap_a = a[8:24]
    overlap_b = b[0:16]
    combined = _section_gauge(
        overlap_a,
        overlap_b,
        min_cosine=min_cosine,
        max_normalized_mad=max_normalized_mad,
    )
    retained = _section_gauge(
        overlap_a[:RETAINED_FRAMES],
        overlap_b[:RETAINED_FRAMES],
        min_cosine=min_cosine,
        max_normalized_mad=max_normalized_mad,
    )
    forecast = _section_gauge(
        overlap_a[RETAINED_FRAMES:OVERLAP_FRAMES],
        overlap_b[RETAINED_FRAMES:OVERLAP_FRAMES],
        min_cosine=min_cosine,
        max_normalized_mad=max_normalized_mad,
    )

    reasons: list[str] = []
    for section_name, section in (
        ("combined", combined),
        ("retained", retained),
        ("forecast", forecast),
    ):
        if not math.isfinite(section.mean_cosine):
            reasons.append(f"{section_name}.mean_cosine_non_finite")
        elif section.mean_cosine + METRIC_COMPARISON_EPS < min_cosine:
            reasons.append(f"{section_name}.mean_cosine<{min_cosine:.6f}")
        if not math.isfinite(section.normalized_mad):
            reasons.append(f"{section_name}.normalized_mad_non_finite")
        elif section.normalized_mad > max_normalized_mad + METRIC_COMPARISON_EPS:
            reasons.append(f"{section_name}.normalized_mad>{max_normalized_mad:.6f}")

    return PairGauge(
        combined=combined,
        retained=retained,
        forecast=forecast,
        accepted=combined.passed and retained.passed and forecast.passed,
        rejection_reasons=tuple(reasons),
    )


def _cache_failure_gauge(
    *,
    dataset: str,
    clip_id: str,
    start_a: int,
    start_b: int,
    error: Exception,
) -> PairGauge:
    message = " ".join(str(error).split())
    return PairGauge(
        combined=None,
        retained=None,
        forecast=None,
        accepted=False,
        rejection_reasons=(f"cache_error:{type(error).__name__}:{message}",),
        dataset=dataset,
        clip_id=clip_id,
        start_a=start_a,
        start_b=start_b,
    )


def build_pair_table(
    base: _BaseWindowDataset,
    *,
    pair_offset: int = PAIR_OFFSET,
    domain_masses: Mapping[str, float] = DEFAULT_STAGE1_PAIR_DOMAIN_MASSES,
    min_cosine: float = DEFAULT_MIN_COSINE,
    max_normalized_mad: float = DEFAULT_MAX_NORMALIZED_MAD,
    progress_every: int = 0,
) -> tuple[tuple[PairRecord, ...], tuple[PairGauge, ...]]:
    """Build and gauge exact s/s+8 candidates, returning only accepted records."""

    if int(pair_offset) != PAIR_OFFSET:
        raise PairedK8Error(
            f"paired-K8 only connects starts exactly 8 frames apart; "
            f"got pair_offset={pair_offset}"
        )
    canonical_masses = _validate_domain_masses(domain_masses)
    allowed_domains = set(canonical_masses)
    progress_every = int(progress_every)
    if progress_every < 0:
        raise PairedK8Error("progress_every must be non-negative")

    cfg = getattr(base, "cfg", None)
    if cfg is None:
        raise PairedK8Error("paired-K8 requires a cache-backed base dataset with cfg")
    if int(getattr(cfg, "T", 0)) != 16 or int(getattr(cfg, "k", 0)) != 8:
        raise PairedK8Error("paired-K8 requires base cache contract T=16 and k=8")
    if not bool(getattr(cfg, "use_window_tokens", False)):
        raise PairedK8Error(
            "paired-K8 requires use_window_tokens=True; dataset samples are "
            "not a substitute for the real window cache gauge"
        )

    lookup: dict[tuple[int, int], int] = {}
    for sample_index, raw_entry in enumerate(base.index):
        if len(raw_entry) < 2:
            raise PairedK8Error(
                f"invalid base window index entry at {sample_index}: {raw_entry}"
            )
        record_index, start = int(raw_entry[0]), int(raw_entry[1])
        if record_index < 0 or record_index >= len(base.records):
            raise PairedK8Error(
                f"base window references missing record: {record_index}"
            )
        key = (record_index, start)
        if key in lookup:
            raise PairedK8Error(
                f"duplicate base window: record={record_index} start={start}"
            )
        lookup[key] = sample_index

    pairs: list[PairRecord] = []
    gauges: list[PairGauge] = []
    candidate_counts = Counter({domain: 0 for domain in canonical_masses})
    accepted_counts = Counter({domain: 0 for domain in canonical_masses})
    token_cache: dict[tuple[int, int], Tensor] = {}
    cached_record_index: int | None = None
    processed_candidates = 0

    for (record_index, start_a), a_sample_index in sorted(lookup.items()):
        record = base.records[record_index]
        dataset = canonical_dataset_name(record.dataset)
        if dataset not in allowed_domains:
            continue
        start_b = start_a + PAIR_OFFSET
        b_sample_index = lookup.get((record_index, start_b))
        if b_sample_index is None:
            continue
        clip_id = str(record.clip_id)
        if cached_record_index != record_index:
            token_cache.clear()
            cached_record_index = record_index
        candidate_counts[dataset] += 1
        processed_candidates += 1
        try:
            def cached_tokens(start: int) -> Tensor:
                key = (record_index, start)
                tokens = token_cache.get(key)
                if tokens is None:
                    tokens = load_window_pooled_tokens(
                        base,
                        record_index=record_index,
                        start=start,
                    )
                    token_cache[key] = tokens
                return tokens

            tokens_a = cached_tokens(start_a)
            tokens_b = cached_tokens(start_b)
            gauge = replace(
                compare_pair_tokens(
                    tokens_a,
                    tokens_b,
                    min_cosine=min_cosine,
                    max_normalized_mad=max_normalized_mad,
                ),
                dataset=dataset,
                clip_id=clip_id,
                start_a=start_a,
                start_b=start_b,
            )
        except (PairedK8Error, OSError, RuntimeError, KeyError, ValueError) as exc:
            gauge = _cache_failure_gauge(
                dataset=dataset,
                clip_id=clip_id,
                start_a=start_a,
                start_b=start_b,
                error=exc,
            )
        for key in tuple(token_cache):
            if key[0] != record_index or key[1] < start_a:
                del token_cache[key]
        gauges.append(gauge)
        if gauge.accepted:
            accepted_counts[dataset] += 1
            pairs.append(
                PairRecord(
                    a_sample_index=a_sample_index,
                    b_sample_index=b_sample_index,
                    record_index=record_index,
                    start_a=start_a,
                    start_b=start_b,
                    dataset=dataset,
                    clip_id=clip_id,
                    gauge=gauge,
                )
            )
        if progress_every and processed_candidates % progress_every == 0:
            print(
                f"[pair_gauge] processed={processed_candidates} "
                f"accepted={sum(accepted_counts.values())}",
                flush=True,
            )

    zero_domains = [
        domain for domain in canonical_masses if accepted_counts[domain] == 0
    ]
    if zero_domains:
        reason_counts = Counter(
            reason
            for gauge in gauges
            if gauge.dataset in zero_domains
            for reason in gauge.rejection_reasons
        )
        raise PairedK8Error(
            "configured pair domains have zero accepted pairs; masses were not "
            f"renormalized: domains={zero_domains} "
            f"candidate_counts={dict(candidate_counts)} "
            f"accepted_counts={dict(accepted_counts)} "
            f"rejection_reasons={dict(reason_counts)}"
        )
    return tuple(pairs), tuple(gauges)


class PairedK8Dataset(Dataset[dict[str, Any]]):
    """Gauge and expose accepted adjacent K8 windows for rolling training."""

    def __init__(
        self,
        base: _BaseWindowDataset,
        *,
        pair_offset: int = PAIR_OFFSET,
        domain_masses: Mapping[str, float] = DEFAULT_STAGE1_PAIR_DOMAIN_MASSES,
        min_cosine: float = DEFAULT_MIN_COSINE,
        max_normalized_mad: float = DEFAULT_MAX_NORMALIZED_MAD,
        progress_every: int = 0,
        frozen_pair_report: str | Path | None = None,
    ) -> None:
        canonical_masses = _validate_domain_masses(domain_masses)
        if frozen_pair_report is None:
            pairs, gauges = build_pair_table(
                base,
                pair_offset=pair_offset,
                domain_masses=canonical_masses,
                min_cosine=min_cosine,
                max_normalized_mad=max_normalized_mad,
                progress_every=progress_every,
            )
        else:
            pairs, gauges = _pairs_from_frozen_report(
                base,
                frozen_pair_report,
                pair_offset=pair_offset,
                domain_masses=canonical_masses,
                min_cosine=min_cosine,
                max_normalized_mad=max_normalized_mad,
            )
        self.base = base
        self.domain_masses = canonical_masses
        self.pairs = pairs
        self.gauges = gauges
        self.records = base.records
        self.index = tuple((entry.record_index, entry.start_a) for entry in self.pairs)
        self.candidate_counts = {
            domain: sum(gauge.dataset == domain for gauge in gauges)
            for domain in canonical_masses
        }
        self.accepted_counts = {
            domain: sum(entry.dataset == domain for entry in pairs)
            for domain in canonical_masses
        }

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, pair_index: int) -> dict[str, Any]:
        entry = self.pairs[pair_index]
        return {
            "a": self.base[entry.a_sample_index],
            "b": self.base[entry.b_sample_index],
            "pair_index": pair_index,
            "pair_start_a": entry.start_a,
            "pair_start_b": entry.start_b,
        }


class HierarchicalPairDistributedSampler(Sampler[int]):
    """Draw domain -> accepted clip -> accepted start from one global stream."""

    def __init__(
        self,
        dataset: PairedK8Dataset,
        *,
        num_samples: int,
        domain_masses: Mapping[str, float] | None = None,
        seed: int = 0,
        rank: int = 0,
        num_replicas: int = 1,
    ) -> None:
        if int(num_samples) <= 0:
            raise PairedK8Error("num_samples must be positive")
        if int(num_replicas) <= 0 or not 0 <= int(rank) < int(num_replicas):
            raise PairedK8Error(
                f"invalid distributed layout: rank={rank} replicas={num_replicas}"
            )
        self.dataset = dataset
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.rank = int(rank)
        self.num_replicas = int(num_replicas)
        self.epoch = 0
        self.cursor = 0
        self.domain_masses = _validate_domain_masses(
            dataset.domain_masses if domain_masses is None else domain_masses
        )
        if self.domain_masses != dataset.domain_masses:
            raise PairedK8Error(
                "sampler domain masses must exactly match the gauged pair dataset"
            )

        grouped: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for pair_index, pair in enumerate(dataset.pairs):
            if pair.dataset in self.domain_masses:
                grouped[pair.dataset][pair.clip_id].append(pair_index)
        missing = [domain for domain in self.domain_masses if not grouped.get(domain)]
        if missing:
            raise PairedK8Error(
                "configured pair domains have zero accepted pairs; masses were "
                f"not renormalized: domains={missing}"
            )
        self._pairs = {
            domain: {
                clip_id: tuple(
                    sorted(
                        pair_indices,
                        key=lambda index: dataset.pairs[index].start_a,
                    )
                )
                for clip_id, pair_indices in sorted(clips.items())
            }
            for domain, clips in sorted(grouped.items())
        }
        self._domains = tuple(self.domain_masses)
        self._weights = tuple(self.domain_masses[domain] for domain in self._domains)
        digest = hashlib.sha256()
        for pair in dataset.pairs:
            digest.update(
                (
                    f"{pair.dataset}\0{pair.clip_id}\0{pair.start_a}\0{pair.start_b}\n"
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
        domains = rng.choices(self._domains, weights=self._weights, k=count)
        stream: list[int] = []
        for domain in domains:
            clips = self._pairs[domain]
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
            raise PairedK8Error(f"pair sampler state is incompatible: {mismatches}")
        cursor = int(state.get("cursor", -1))
        if not 0 <= cursor <= self.num_samples:
            raise PairedK8Error(
                f"pair sampler cursor is outside [0, {self.num_samples}]: {cursor}"
            )
        self.epoch = int(state.get("epoch", 0))
        self.cursor = cursor

    def audit(
        self,
        *,
        num_draws: int = 10_000,
        domain_tolerance: float = 0.02,
    ) -> PairSamplingAudit:
        if int(num_draws) != 10_000:
            raise PairedK8Error(
                f"formal pair sampler audit requires exactly 10000 draws; "
                f"got {num_draws}"
            )
        if not 0.0 < float(domain_tolerance) <= 1.0:
            raise PairedK8Error("domain_tolerance must be in (0, 1]")

        stream = self._draw_stream(int(num_draws))
        domain_counts: Counter[str] = Counter()
        clip_counts: Counter[tuple[str, str]] = Counter()
        start_counts: Counter[tuple[str, str, int]] = Counter()
        for pair_index in stream:
            pair = self.dataset.pairs[pair_index]
            domain_counts[pair.dataset] += 1
            clip_counts[(pair.dataset, pair.clip_id)] += 1
            start_counts[(pair.dataset, pair.clip_id, pair.start_a)] += 1

        fractions = {
            domain: domain_counts[domain] / num_draws for domain in self.domain_masses
        }
        violations: list[str] = []
        for domain, expected in self.domain_masses.items():
            if abs(fractions[domain] - expected) > domain_tolerance:
                violations.append(
                    f"domain_mass:{domain}:observed={fractions[domain]:.6f}:"
                    f"expected={expected:.6f}"
                )

        max_clip_deviation = 0.0
        max_start_deviation = 0.0
        for domain, clips in self._pairs.items():
            domain_count = domain_counts[domain]
            expected_clip = 1.0 / len(clips)
            clip_limit = max(
                domain_tolerance,
                5.0
                * math.sqrt(
                    expected_clip * (1.0 - expected_clip) / max(domain_count, 1)
                ),
            )
            for clip_id, pair_indices in clips.items():
                observed_clip = (
                    clip_counts[(domain, clip_id)] / domain_count
                    if domain_count
                    else 0.0
                )
                clip_deviation = abs(observed_clip - expected_clip)
                max_clip_deviation = max(
                    max_clip_deviation,
                    clip_deviation,
                )
                if clip_deviation > clip_limit:
                    violations.append(
                        f"clip_uniformity:{domain}:{clip_id}:"
                        f"deviation={clip_deviation:.6f}:"
                        f"limit={clip_limit:.6f}"
                    )

                clip_count = clip_counts[(domain, clip_id)]
                expected_start = 1.0 / len(pair_indices)
                start_limit = max(
                    domain_tolerance,
                    5.0
                    * math.sqrt(
                        expected_start * (1.0 - expected_start) / max(clip_count, 1)
                    ),
                )
                for pair_index in pair_indices:
                    start = self.dataset.pairs[pair_index].start_a
                    observed_start = (
                        start_counts[(domain, clip_id, start)] / clip_count
                        if clip_count
                        else 0.0
                    )
                    start_deviation = abs(observed_start - expected_start)
                    max_start_deviation = max(
                        max_start_deviation,
                        start_deviation,
                    )
                    if start_deviation > start_limit:
                        violations.append(
                            f"start_uniformity:{domain}:{clip_id}:{start}:"
                            f"deviation={start_deviation:.6f}:"
                            f"limit={start_limit:.6f}"
                        )

        partition_stream = self._draw_stream(num_draws * 2)
        rank_zero = partition_stream[0::2]
        rank_one = partition_stream[1::2]
        interleaved = [
            value for pair in zip(rank_zero, rank_one, strict=True) for value in pair
        ]
        deterministic_partition = (
            partition_stream == self._draw_stream(num_draws * 2)
            and interleaved == partition_stream
        )
        if not deterministic_partition:
            violations.append("deterministic_rank_partition")

        coverage = {
            domain: (
                self.dataset.accepted_counts[domain]
                / self.dataset.candidate_counts[domain]
                if self.dataset.candidate_counts[domain]
                else 0.0
            )
            for domain in self.domain_masses
        }
        for domain, ratio in coverage.items():
            if ratio <= 0.0:
                violations.append(f"rejection_adjusted_coverage:{domain}:zero")

        return PairSamplingAudit(
            draws=num_draws,
            domain_fractions=fractions,
            max_clip_deviation=max_clip_deviation,
            max_start_deviation=max_start_deviation,
            deterministic_rank_partition=deterministic_partition,
            rejection_adjusted_coverage=coverage,
            violations=tuple(violations),
        )


def build_paired_k8_sampler(
    dataset: PairedK8Dataset,
    *,
    num_samples: int,
    seed: int = 0,
    rank: int = 0,
    num_replicas: int = 1,
) -> HierarchicalPairDistributedSampler:
    return HierarchicalPairDistributedSampler(
        dataset,
        num_samples=num_samples,
        seed=seed,
        rank=rank,
        num_replicas=num_replicas,
    )


def build_native_rolling_context(
    s_b: Tensor,
    pred_a: Tensor,
    *,
    observed_frames: int = 8,
) -> Tensor:
    if (
        s_b.ndim < 3
        or pred_a.ndim != s_b.ndim
        or observed_frames <= 0
        or s_b.shape[1] != observed_frames * 2
        or pred_a.shape[1] != observed_frames
        or s_b.shape[0] != pred_a.shape[0]
        or s_b.shape[2:] != pred_a.shape[2:]
    ):
        raise PairedK8Error(
            "rolling context shapes must be Bx16x... and Bx8x... with matching "
            f"tails; got s_b={tuple(s_b.shape)} pred_a={tuple(pred_a.shape)}"
        )
    return torch.cat(
        (s_b[:, :observed_frames].detach(), pred_a.detach()),
        dim=1,
    )
