"""Fail-closed access to one raw episode and its ordinal-bound RGB streams."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .cache_tasks import CacheTask
from .manifest_contract import canonical_timestamp_sha256, sha256_file
from .source_adapters import AdapterContract, EpisodeAccessor


class EpisodeIOError(RuntimeError):
    pass


@dataclass(frozen=True)
class _VerifiedAsset:
    expected_sha256: str
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int


class VerifiedAssetStore:
    """Verify a raw immutable asset once per long-lived cache worker.

    A LeRobot Parquet/video shard commonly backs many episode tasks.  Hashing
    that entire shard for every episode creates an unbounded read
    amplification.  This store performs the full SHA256 verification on first
    use and subsequently accepts the cached verification only while all
    mutation-relevant stat fields are unchanged.  Any replacement or write
    forces a fresh hash and therefore still fails closed against the digest
    sealed in the source manifest.
    """

    def __init__(self) -> None:
        self._verified: dict[Path, _VerifiedAsset] = {}

    @staticmethod
    def _fingerprint(path: Path, expected_sha256: str) -> _VerifiedAsset:
        stat = path.stat()
        return _VerifiedAsset(
            expected_sha256=expected_sha256,
            device=int(stat.st_dev),
            inode=int(stat.st_ino),
            size_bytes=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
            ctime_ns=int(stat.st_ctime_ns),
        )

    def verify(self, root: Path, relative: str, expected_sha256: str) -> Path:
        root = Path(root).resolve(strict=True)
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise EpisodeIOError(f"episode asset is not a regular file: {candidate}")
        path = candidate.resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise EpisodeIOError(f"episode asset escapes source root: {path}") from exc
        current = self._fingerprint(path, expected_sha256)
        cached = self._verified.get(path)
        if cached == current:
            return path
        observed = sha256_file(path)
        if observed != expected_sha256:
            raise EpisodeIOError(
                f"episode asset SHA mismatch {relative}: "
                f"{observed} != {expected_sha256}"
            )
        # Re-stat after hashing so a write racing with verification cannot be
        # silently accepted as immutable evidence.
        after = self._fingerprint(path, expected_sha256)
        if after != current:
            raise EpisodeIOError(
                f"episode asset changed while its SHA was being verified: {relative}"
            )
        self._verified[path] = after
        return path


def select_episode_cache_rows(
    observation_times_s: Sequence[float], *, minimum_separation_s: float
) -> np.ndarray:
    """Greedily thin to recorded rows while preserving both episode endpoints."""

    clock = np.asarray(observation_times_s, dtype=np.float64)
    if (
        clock.ndim != 1
        or clock.size < 2
        or not np.isfinite(clock).all()
        or np.any(np.diff(clock) <= 0)
    ):
        raise EpisodeIOError("observation clock must be finite/strictly increasing")
    if not np.isfinite(minimum_separation_s) or minimum_separation_s < 0:
        raise EpisodeIOError("minimum cache-state separation must be non-negative")
    if minimum_separation_s == 0:
        return np.arange(clock.size, dtype=np.int64)
    rows = [0]
    for row in range(1, len(clock) - 1):
        if clock[row] - clock[rows[-1]] + 1.0e-12 >= minimum_separation_s:
            rows.append(row)
    if rows[-1] != len(clock) - 1:
        rows.append(len(clock) - 1)
    result = np.asarray(rows, dtype=np.int64)
    if np.any(np.diff(result) <= 0):
        raise AssertionError("internal cache-state selector produced duplicate rows")
    return result


def _safe_asset(
    root: Path,
    relative: str,
    expected_sha256: str,
    *,
    verifier: VerifiedAssetStore | None = None,
) -> Path:
    return (verifier or VerifiedAssetStore()).verify(
        root, relative, expected_sha256
    )


class ParquetEpisodeAccessor(EpisodeAccessor):
    """Read only row groups intersecting a SHA-bound episode slice."""

    def __init__(self, path: Path, *, row_start: int, row_stop: int):
        import pyarrow.parquet as pq

        self.path = Path(path)
        self.parquet = pq.ParquetFile(self.path)
        self.start = int(row_start)
        self.stop = int(row_stop)
        if (
            self.start < 0
            or self.stop <= self.start
            or self.stop > self.parquet.metadata.num_rows
        ):
            raise EpisodeIOError(
                f"invalid Parquet episode slice [{self.start},{self.stop})"
            )
        self._cache: dict[str, np.ndarray] = {}

    def array(self, key: str) -> np.ndarray:
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if key not in self.parquet.schema_arrow.names:
            raise EpisodeIOError(f"raw Parquet payload misses field {key!r}")
        values: list[object] = []
        row_group_start = 0
        for row_group in range(self.parquet.metadata.num_row_groups):
            count = self.parquet.metadata.row_group(row_group).num_rows
            row_group_stop = row_group_start + count
            left = max(self.start, row_group_start)
            right = min(self.stop, row_group_stop)
            if left < right:
                column = self.parquet.read_row_group(row_group, columns=[key]).column(0)
                values.extend(
                    column.slice(left - row_group_start, right - left).to_pylist()
                )
            row_group_start = row_group_stop
            if row_group_start >= self.stop:
                break
        result = np.asarray(values)
        if result.shape[0] != self.stop - self.start:
            raise EpisodeIOError(f"Parquet field {key!r} episode slice is incomplete")
        self._cache[key] = result
        return result


class NpzEpisodeAccessor(EpisodeAccessor):
    def __init__(self, path: Path, *, row_start: int, row_stop: int):
        self.path = Path(path)
        self.start = int(row_start)
        self.stop = int(row_stop)
        if self.start < 0 or self.stop <= self.start:
            raise EpisodeIOError("invalid NPZ episode slice")
        self.archive = np.load(self.path, allow_pickle=False)

    def array(self, key: str) -> np.ndarray:
        if key not in self.archive.files:
            raise EpisodeIOError(f"raw NPZ payload misses field {key!r}")
        value = np.asarray(self.archive[key])
        if value.ndim < 1 or self.stop > value.shape[0]:
            raise EpisodeIOError(f"NPZ field {key!r} cannot satisfy episode slice")
        return value[self.start : self.stop]


def open_episode_accessor(
    *,
    task: CacheTask,
    source_root: Path,
    adapter: AdapterContract,
    asset_verifier: VerifiedAssetStore | None = None,
) -> EpisodeAccessor:
    primary = [asset for asset in task.assets if asset[0] == "primary_payload"]
    if len(primary) != 1:
        raise EpisodeIOError("cache task must bind exactly one primary payload")
    _role, relative, digest = primary[0]
    if relative != task.payload or digest != task.payload_sha256:
        raise EpisodeIOError("primary payload binding differs from cache task")
    path = _safe_asset(
        source_root, relative, digest, verifier=asset_verifier
    )
    kwargs = {"row_start": task.payload_row_start, "row_stop": task.payload_row_stop}
    if adapter.raw_format in {"lerobot_parquet_video", "agibot_parquet_video"}:
        return ParquetEpisodeAccessor(path, **kwargs)
    if adapter.raw_format == "npz":
        return NpzEpisodeAccessor(path, **kwargs)
    raise EpisodeIOError(f"unsupported episode format {adapter.raw_format!r}")


@dataclass(frozen=True)
class DecodedEpisodeVideo:
    frames: np.ndarray
    recorded_pts_s: np.ndarray
    pts_sha256: str
    segment_kind: str


def _decode_segment(
    path: Path,
    *,
    segment_kind: str,
    start_s: float | None,
    stop_s: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    import av

    if segment_kind not in {"entire_file", "recorded_pts_range"}:
        raise EpisodeIOError(f"unsupported video segment kind {segment_kind!r}")
    if segment_kind == "entire_file":
        if start_s is not None or stop_s is not None:
            raise EpisodeIOError("entire-file video cannot carry PTS bounds")
    elif (
        start_s is None
        or stop_s is None
        or not np.isfinite(start_s)
        or not np.isfinite(stop_s)
        or stop_s <= start_s
    ):
        raise EpisodeIOError("recorded PTS segment requires finite increasing bounds")

    frames: list[np.ndarray] = []
    pts_values: list[float] = []
    with av.open(str(path), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise EpisodeIOError(f"expected one video stream in {path}, found {len(streams)}")
        stream = streams[0]
        if segment_kind == "recorded_pts_range":
            # Seek backward to a keyframe then discard decoded frames before
            # the exact recorded half-open range.
            assert start_s is not None
            container.seek(
                int(start_s / float(stream.time_base)),
                stream=stream,
                backward=True,
                any_frame=False,
            )
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise EpisodeIOError(f"video frame has no recorded PTS: {path}")
            pts = float(frame.pts * frame.time_base)
            if segment_kind == "recorded_pts_range":
                assert start_s is not None and stop_s is not None
                if pts + 1.0e-12 < start_s:
                    continue
                if pts >= stop_s - 1.0e-12:
                    break
            array = frame.to_ndarray(format="rgb24")
            if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
                raise EpisodeIOError(f"decoded frame is not uint8 RGB: {path}")
            frames.append(array)
            pts_values.append(pts)
    if len(frames) < 2:
        raise EpisodeIOError(f"video segment has fewer than two frames: {path}")
    pts_array = np.asarray(pts_values, dtype=np.float64)
    if not np.isfinite(pts_array).all() or np.any(np.diff(pts_array) <= 0):
        raise EpisodeIOError(f"video PTS must be finite and strictly increasing: {path}")
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise EpisodeIOError(f"video resolution changes within episode: {path}")
    return np.stack(frames), pts_array


def decode_episode_views(
    *,
    task: CacheTask,
    source_root: Path,
    canonical_view_slots: Sequence[str],
    selected_observation_rows: Sequence[int],
    asset_verifier: VerifiedAssetStore | None = None,
) -> tuple[dict[str, DecodedEpisodeVideo], dict[str, Mapping[str, object]]]:
    """Decode each real view and bind it to observation rows by ordinal.

    MP4 PTS is retained as audit evidence, but is deliberately not used as a
    nearest-neighbour alignment clock: several real LeRobot releases carry
    one video frame per episode row while their encoded PTS cadence differs
    from the Parquet observation timestamp cadence.
    """

    slots = tuple(str(item) for item in canonical_view_slots)
    if not slots or len(set(slots)) != len(slots):
        raise EpisodeIOError("canonical view slots must be unique/non-empty")
    rows = np.asarray(selected_observation_rows, dtype=np.int64)
    if (
        rows.ndim != 1
        or rows.size < 2
        or rows[0] < 0
        or rows[-1] >= task.observation_samples
        or np.any(np.diff(rows) <= 0)
    ):
        raise EpisodeIOError("selected observation rows are invalid")
    asset_by_role = {role: (path, digest) for role, path, digest in task.assets}
    if len(asset_by_role) != len(task.assets):
        raise EpisodeIOError("cache task contains duplicate asset roles")
    output: dict[str, DecodedEpisodeVideo] = {}
    evidence: dict[str, Mapping[str, object]] = {}
    for name, role, segment_kind, start_s, stop_s in task.views:
        if name not in slots or name in output:
            raise EpisodeIOError(f"view {name!r} is unknown or duplicated")
        if role not in asset_by_role:
            raise EpisodeIOError(f"view {name!r} references missing asset role {role!r}")
        relative, digest = asset_by_role[role]
        path = _safe_asset(
            source_root, relative, digest, verifier=asset_verifier
        )
        frames, pts = _decode_segment(
            path,
            segment_kind=segment_kind,
            start_s=start_s,
            stop_s=stop_s,
        )
        if len(frames) != task.observation_samples:
            raise EpisodeIOError(
                f"view {name!r} has {len(frames)} decoded frames but episode has "
                f"{task.observation_samples} observation rows; ordinal binding failed"
            )
        digest_pts = canonical_timestamp_sha256(pts)
        output[name] = DecodedEpisodeVideo(
            frames=frames[rows].copy(),
            recorded_pts_s=pts,
            pts_sha256=digest_pts,
            segment_kind=segment_kind,
        )
        evidence[name] = {
            "asset_role": role,
            "asset_sha256": digest,
            "segment_kind": segment_kind,
            "decoded_frame_count": int(len(frames)),
            "selected_frame_count": int(len(rows)),
            "recorded_pts_start_s": float(pts[0]),
            "recorded_pts_end_s": float(pts[-1]),
            "recorded_pts_sha256": digest_pts,
            "binding": "episode_row_ordinal",
        }
    if not output:
        raise EpisodeIOError("episode contains no decodable real view")
    return output, evidence
