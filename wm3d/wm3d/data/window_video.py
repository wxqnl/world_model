"""Bounded random-access RGB decoding for direct VGGT windows.

The legacy cache worker deliberately decodes a complete episode before writing
its immutable payload.  Direct training must not inherit that behavior:
only the T+K ordinal rows for the current sample are decoded.  A small CPU
LRU retains exact packet PTS indices; Decord is the fast random-access decoder
when it supports the codec, with bounded PyAV seek/decode as the fallback.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence

import numpy as np

from .cache_tasks import CacheTask
from .episode_io import DecodedEpisodeVideo, EpisodeIOError, VerifiedAssetStore
from .manifest_contract import canonical_timestamp_sha256


@dataclass(frozen=True)
class VideoFrameIndex:
    pts: np.ndarray
    time_base_s: float

    @property
    def timestamps_s(self) -> np.ndarray:
        return self.pts.astype(np.float64) * self.time_base_s


class VideoTimestampIndexStore:
    """Small thread-safe LRU of exact frame PTS, never decoded RGB."""

    def __init__(self, maximum_assets: int = 64) -> None:
        if maximum_assets <= 0:
            raise EpisodeIOError("video timestamp index capacity must be positive")
        self.maximum_assets = int(maximum_assets)
        self._values: OrderedDict[Path, VideoFrameIndex] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.builds = 0
        self.evictions = 0

    @staticmethod
    def _build(path: Path) -> VideoFrameIndex:
        import av

        with av.open(str(path), mode="r") as container:
            streams = list(container.streams.video)
            if len(streams) != 1:
                raise EpisodeIOError(
                    f"expected one video stream in {path}, found {len(streams)}"
                )
            stream = streams[0]
            time_base_s = float(stream.time_base)
            pts_values = [
                int(packet.pts)
                for packet in container.demux(stream)
                if packet.pts is not None
            ]
        if len(pts_values) < 2 or len(set(pts_values)) != len(pts_values):
            raise EpisodeIOError(
                f"video packet PTS are empty or duplicated: {path}"
            )
        pts = np.asarray(sorted(pts_values), dtype=np.int64)
        timestamps = pts.astype(np.float64) * time_base_s
        if (
            not np.isfinite(time_base_s)
            or time_base_s <= 0
            or not np.isfinite(timestamps).all()
            or np.any(np.diff(timestamps) <= 0)
        ):
            raise EpisodeIOError(
                f"video frame timestamps are not finite/increasing: {path}"
            )
        pts.setflags(write=False)
        return VideoFrameIndex(pts=pts, time_base_s=time_base_s)

    def get(self, path: Path) -> VideoFrameIndex:
        path = Path(path)
        with self._lock:
            cached = self._values.pop(path, None)
            if cached is not None:
                self._values[path] = cached
                self.hits += 1
                return cached
        built = self._build(path)
        with self._lock:
            cached = self._values.pop(path, None)
            if cached is not None:
                self._values[path] = cached
                self.hits += 1
                return cached
            self._values[path] = built
            while len(self._values) > self.maximum_assets:
                self._values.popitem(last=False)
                self.evictions += 1
            self.builds += 1
            return built

    @property
    def metrics(self) -> Mapping[str, int]:
        with self._lock:
            return {
                "video_index_assets": len(self._values),
                "video_index_hits": self.hits,
                "video_index_builds": self.builds,
                "video_index_evictions": self.evictions,
            }


def _open_video_reader(path: Path) -> Any:
    import decord

    return decord.VideoReader(
        str(path),
        ctx=decord.cpu(0),
        num_threads=1,
    )


def _decode_selected_with_pyav(
    path: Path,
    target_pts: np.ndarray,
    *,
    expected_time_base_s: float,
) -> tuple[np.ndarray, int]:
    """Seek to the preceding keyframe and decode only through the last target."""

    import av

    wanted = {int(value) for value in np.asarray(target_pts, dtype=np.int64)}
    if not wanted:
        raise EpisodeIOError("random-access decode target is empty")
    first_pts = min(wanted)
    last_pts = max(wanted)
    frames_by_pts: dict[int, np.ndarray] = {}
    decoded_count = 0
    with av.open(str(path), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise EpisodeIOError(
                f"expected one video stream in {path}, found {len(streams)}"
            )
        stream = streams[0]
        observed_time_base_s = float(stream.time_base)
        if not np.isclose(
            observed_time_base_s,
            expected_time_base_s,
            rtol=0.0,
            atol=max(1.0e-15, abs(expected_time_base_s) * 1.0e-12),
        ):
            raise EpisodeIOError(f"video time base changed after indexing: {path}")
        container.seek(
            first_pts,
            stream=stream,
            backward=True,
            any_frame=False,
        )
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_pts = int(frame.pts)
            decoded_count += 1
            if frame_pts in wanted:
                array = frame.to_ndarray(format="rgb24")
                if (
                    array.ndim != 3
                    or array.shape[-1] != 3
                    or array.dtype != np.uint8
                ):
                    raise EpisodeIOError(
                        f"PyAV returned invalid uint8 RGB for {path}"
                    )
                frames_by_pts[frame_pts] = array
            if frame_pts > last_pts or len(frames_by_pts) == len(wanted):
                break
    missing = sorted(wanted - set(frames_by_pts))
    if missing:
        raise EpisodeIOError(
            f"PyAV random seek missed {len(missing)} requested frames in {path}"
        )
    return (
        np.stack([frames_by_pts[int(value)] for value in target_pts]),
        decoded_count,
    )


def _decode_selected_frames(
    path: Path,
    frame_index: VideoFrameIndex,
    selected_ordinals: np.ndarray,
) -> tuple[np.ndarray, str, int]:
    """Use Decord when supported, otherwise bounded PyAV seek/decode."""

    decord_error: Exception | None = None
    try:
        reader = _open_video_reader(path)
        if len(reader) != len(frame_index.pts):
            raise EpisodeIOError(
                f"Decord/PyAV frame cardinality differs for {path}"
            )
        raw = reader.get_batch(selected_ordinals.tolist())
        if hasattr(raw, "asnumpy"):
            raw = raw.asnumpy()
        frames = np.asarray(raw)
        if (
            frames.ndim != 4
            or frames.shape[0] != len(selected_ordinals)
            or frames.shape[-1] != 3
            or frames.dtype != np.uint8
        ):
            raise EpisodeIOError(
                f"Decord returned invalid uint8 RGB for {path}"
            )
        return frames, "decord", int(len(selected_ordinals))
    except Exception as exc:
        decord_error = exc

    target_pts = frame_index.pts[selected_ordinals]
    try:
        frames, decoded_count = _decode_selected_with_pyav(
            path,
            target_pts,
            expected_time_base_s=frame_index.time_base_s,
        )
    except Exception as exc:
        raise EpisodeIOError(
            f"both random-access video backends failed for {path}; "
            f"Decord error: {decord_error}"
        ) from exc
    return frames, "pyav_seek", decoded_count


def _segment_ordinals(
    *,
    timestamps: np.ndarray,
    segment_kind: str,
    start_s: float | None,
    stop_s: float | None,
    observation_samples: int,
    view_name: str,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    if segment_kind == "entire_file":
        if start_s is not None or stop_s is not None:
            raise EpisodeIOError("entire-file video cannot carry PTS bounds")
        ordinals = np.arange(len(timestamps), dtype=np.int64)
    elif segment_kind == "recorded_pts_range":
        if (
            start_s is None
            or stop_s is None
            or not np.isfinite(start_s)
            or not np.isfinite(stop_s)
            or stop_s <= start_s
        ):
            raise EpisodeIOError(
                "recorded PTS segment requires finite increasing bounds"
            )
        # Preserve the sealed full-episode decoder's exact half-open rule:
        # skip while ``pts + 1e-12 < start`` and stop at
        # ``pts >= stop - 1e-12``.  Nearest-frame snapping changes the
        # episode-row ordinal at large floating-point PTS offsets.
        start_ordinal = int(
            np.searchsorted(timestamps, float(start_s) - 1.0e-12, side="left")
        )
        stop_ordinal = int(
            np.searchsorted(timestamps, float(stop_s) - 1.0e-12, side="left")
        )
        if stop_ordinal <= start_ordinal:
            raise EpisodeIOError(
                f"view {view_name!r} has an empty/reversed recorded PTS range"
            )
        ordinals = np.arange(start_ordinal, stop_ordinal, dtype=np.int64)
    else:
        raise EpisodeIOError(
            f"unsupported video segment kind {segment_kind!r}"
        )
    container_count = int(len(ordinals))
    if container_count < 2:
        raise EpisodeIOError(
            f"view {view_name!r} segment has fewer than two frames"
        )
    trailing_dropped = 0
    trailing_repeated = 0
    delta = container_count - int(observation_samples)
    if delta == 1:
        ordinals = ordinals[:-1]
        trailing_dropped = 1
    elif delta == -1:
        ordinals = np.concatenate((ordinals, ordinals[-1:]))
        trailing_repeated = 1
    elif delta != 0:
        raise EpisodeIOError(
            f"view {view_name!r} has {container_count} indexed frames but episode "
            f"has {observation_samples} observation rows; ordinal binding failed"
        )
    segment_pts = timestamps[ordinals].copy()
    if trailing_repeated:
        step = float(
            timestamps[ordinals[-2]] - timestamps[ordinals[-3]]
        )
        if not np.isfinite(step) or step <= 0:
            raise EpisodeIOError("cannot synthesize one missing trailing PTS")
        segment_pts[-1] = segment_pts[-2] + step
    if (
        len(ordinals) != int(observation_samples)
        or not np.isfinite(segment_pts).all()
        or np.any(np.diff(segment_pts) <= 0)
    ):
        raise EpisodeIOError("window decoder produced an invalid ordinal map")
    return (
        ordinals,
        segment_pts,
        container_count,
        trailing_dropped,
        trailing_repeated,
    )


def decode_episode_window_views(
    *,
    task: CacheTask,
    source_root: Path,
    canonical_view_slots: Sequence[str],
    selected_observation_rows: Sequence[int],
    asset_verifier: VerifiedAssetStore,
    timestamp_indices: VideoTimestampIndexStore,
    decode_workers: int = 1,
) -> tuple[dict[str, DecodedEpisodeVideo], dict[str, Mapping[str, object]]]:
    """Decode only selected ordinal rows from every real camera stream."""

    slots = tuple(str(item) for item in canonical_view_slots)
    if not slots or len(set(slots)) != len(slots):
        raise EpisodeIOError("canonical view slots must be unique/non-empty")
    if decode_workers <= 0:
        raise EpisodeIOError("decode_workers must be positive")
    rows = np.asarray(selected_observation_rows, dtype=np.int64)
    if (
        rows.ndim != 1
        or rows.size < 1
        or rows[0] < 0
        or rows[-1] >= task.observation_samples
        or np.any(np.diff(rows) <= 0)
    ):
        raise EpisodeIOError("selected observation rows are invalid")
    assets = {role: (relative, digest) for role, relative, digest in task.assets}
    if len(assets) != len(task.assets):
        raise EpisodeIOError("cache task contains duplicate asset roles")
    jobs: list[tuple[str, str, str, float | None, float | None, Path, str]] = []
    for name, role, segment_kind, start_s, stop_s in task.views:
        if name not in slots or any(item[0] == name for item in jobs):
            raise EpisodeIOError(f"view {name!r} is unknown or duplicated")
        if role not in assets:
            raise EpisodeIOError(
                f"view {name!r} references missing asset role {role!r}"
            )
        relative, digest = assets[role]
        path = asset_verifier.verify(source_root, relative, digest)
        jobs.append(
            (name, role, segment_kind, start_s, stop_s, path, digest)
        )
    if not jobs:
        raise EpisodeIOError("episode contains no decodable real view")

    def decode_one(
        job: tuple[str, str, str, float | None, float | None, Path, str]
    ) -> tuple[str, DecodedEpisodeVideo, Mapping[str, object]]:
        name, role, segment_kind, start_s, stop_s, path, digest = job
        frame_index = timestamp_indices.get(path)
        timestamps = frame_index.timestamps_s
        (
            ordinals,
            segment_pts,
            container_count,
            trailing_dropped,
            trailing_repeated,
        ) = _segment_ordinals(
            timestamps=timestamps,
            segment_kind=segment_kind,
            start_s=start_s,
            stop_s=stop_s,
            observation_samples=task.observation_samples,
            view_name=name,
        )
        selected_ordinals = ordinals[rows]
        frames, decode_backend, codec_decoded_count = _decode_selected_frames(
            path,
            frame_index,
            selected_ordinals,
        )
        digest_pts = canonical_timestamp_sha256(segment_pts)
        decoded = DecodedEpisodeVideo(
            frames=frames.copy(),
            recorded_pts_s=segment_pts,
            pts_sha256=digest_pts,
            segment_kind=segment_kind,
        )
        evidence = {
            "asset_role": role,
            "asset_sha256": digest,
            "segment_kind": segment_kind,
            "decoded_frame_count": int(len(ordinals)),
            "container_decoded_frame_count": container_count,
            "trailing_frames_dropped": trailing_dropped,
            "trailing_frames_repeated": trailing_repeated,
            "selected_frame_count": int(len(rows)),
            "random_access_decoded_frame_count": int(len(rows)),
            "codec_decoded_frame_count": int(codec_decoded_count),
            "decode_backend": decode_backend,
            "recorded_pts_start_s": float(segment_pts[0]),
            "recorded_pts_end_s": float(segment_pts[-1]),
            "recorded_pts_sha256": digest_pts,
            "binding": "episode_row_ordinal_random_access",
        }
        return name, decoded, evidence

    workers = min(int(decode_workers), len(jobs))
    if workers == 1:
        decoded_jobs = [decode_one(job) for job in jobs]
    else:
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="wm3d-window-video"
        ) as executor:
            decoded_jobs = list(executor.map(decode_one, jobs))
    return (
        {name: decoded for name, decoded, _evidence in decoded_jobs},
        {name: evidence for name, _decoded, evidence in decoded_jobs},
    )
