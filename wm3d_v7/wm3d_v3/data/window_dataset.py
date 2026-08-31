"""Cache-backed window dataset for OXE training.

Each sample = (T_in input frames + k future frames) sliced from one episode.
Cache layout:
    cache/wm3d_v3/vggt_pooled/<safe_id>.npy        [n_frames, 64, 2048] fp16
    cache/wm3d_v3/vggt_geom/<safe_id>.npz          {"depth": [n,224,224] fp16,
                                                     optional extra VGGT geometry}
    cache/wm3d_v3/vggt_window_geom_p64/<safe_id>__start_<start>.npz
                                                     window-aligned VGGT pooled
                                                     tokens and native geometry
    cache/wm3d_v3/rgb_256/<safe_id>.npy            [n_frames, 256, 256, 3] uint8
    cache/wm3d_v3/actions/<safe_id>.npy            [n_frames, 7] fp32
    cache/wm3d_v3/qwen_taskemb/<safe_id>.npy       [2048] fp16
"""
from __future__ import annotations
from dataclasses import dataclass
import io
import hashlib
import os
from pathlib import Path
import stat
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from .manifest import OXEClipRecord
from wm3d_v3.stage1.action_cache import (
    DEFAULT_ACTION_CACHE_SUBDIR,
    read_stable_cache_bytes,
    ActionCacheResolutionError,
    resolve_action_cache,
    validate_formal_droid_cache_index,
)
from wm3d_v3.stage1.action_contract import (
    ActionContractBoundaryError,
    ActionContractCoverageError,
    ActionWindowResolution,
    UnknownDatasetAlias,
    action_contract_key,
    load_passed_contracts,
    resolve_action_window,
    validate_manifest_contract_coverage,
)


def _rss_mib() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except (OSError, IndexError, ValueError):
        pass
    return float("nan")


def _dataset_startup_log(event: str, started: float, **fields) -> None:
    now = time.monotonic()
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    elapsed = f" elapsed_s={now - started:.6f}" if event == "end" else ""
    print(
        f"[startup] stage=window_dataset.scan event={event} monotonic_s={now:.6f}{elapsed} "
        f"rank={int(os.environ.get('RANK', '0'))} rss_mib={_rss_mib():.1f} {details}".rstrip(),
        flush=True,
    )


@dataclass
class WindowConfig:
    T: int = 16
    k: int = 8
    stride: int = 4
    cache_root: Path = Path("/home/user01/Minko/datasets/cache/wm3d_v3")
    tokens_subdir: str = "vggt_pooled"  # set to "vggt_p256" for 16x16 grid
    action_stats: Path | None = None  # set to action_stats.npz to enable normalization
    manifest_path: Path | None = None
    action_contract_path: Path | None = None
    require_action_contract: bool = False
    default_action_frame_offset: int = 0
    droid_cache_index_path: Path | None = None
    require_task_emb: bool = False  # fail fast instead of silently using zero Qwen embeddings
    load_task_text: bool = False  # emit raw task_text string in sample (Hunyuan text-cond generation)
    load_rgb: bool = True
    load_geom: bool = True
    load_state_tgt: bool = True
    load_geom_extra: bool = False
    require_geom_extra: bool = False
    window_geom_subdir: str = "vggt_window_geom_p64"
    window_geom_shard_index: Path | None = None
    window_geom_shard_root: Path | None = None
    window_geom_shard_indices: tuple[Path, ...] | None = None
    window_geom_shard_roots: tuple[Path | None, ...] | None = None
    use_window_tokens: bool = False
    max_windows_per_episode: int = 0
    trust_window_geom_cache: bool = False
    # Skip repeated remote cache-header scans only when the input manifest is
    # content-addressed and was itself built by strict cache validation.
    trusted_manifest_fast_init: bool = False
    trusted_manifest_sha256: str | None = None
    allow_pseudo_progress_targets: bool = False
    require_progress: bool = False
    load_policy_state: bool = False
    require_policy_state: bool = False
    # Optional policy geom is lazy by default, so corrupt archives fail in
    # __getitem__. Enable this to open/read policy arrays during initialization.
    strict_policy_state_prescan: bool = False
    policy_lowdim_dim: int = 0
    policy_object_state_dim: int = 0
    policy_plan_state_dim: int = 0
    policy_action_history_len: int = 0
    policy_action_history_dim: int = 7


def _safe(cid: str) -> str:
    return cid.replace("/", "__")


def _window_geom_path(cache_root: Path, subdir: str, cid: str, start: int) -> Path:
    return cache_root / subdir / f"{cid}__start_{int(start):06d}.npz"


def _window_geom_name(cid: str, start: int) -> str:
    return f"{cid}__start_{int(start):06d}.npz"




def _open_regular_binary_nofollow(path: Path, label: str):
    raw = Path(path)
    if ".." in raw.parts:
        raise RuntimeError(f"{label} contains path traversal: {raw}")
    absolute = raw.absolute()
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor = os.open(absolute.anchor, directory_flags)
    try:
        for part in absolute.parts[1:-1]:
            next_descriptor = os.open(
                part,
                directory_flags,
                dir_fd=parent_descriptor,
            )
            os.close(parent_descriptor)
            parent_descriptor = next_descriptor
        descriptor = os.open(
            absolute.parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        raise RuntimeError(
            f"{label} is missing, replaced, or contains a symlink: {absolute}"
        ) from exc
    finally:
        os.close(parent_descriptor)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise RuntimeError(f"{label} is not a regular file: {absolute}")
    return os.fdopen(descriptor, "rb")
class _WindowGeomShardReader:

    """Random-access reader for local uncompressed tar shards.

    The index is a TSV with: member_name, shard_path, offset_data, size. Shards
    must be uncompressed tar files so workers can seek directly to one npz.
    """

    def __init__(self, index_path: Path, shard_root: Path | None = None):
        self.index_path = Path(index_path)
        self.shard_root = Path(shard_root) if shard_root is not None else self.index_path.parent
        self._index: dict[str, tuple[Path, int, int]] | None = None
        self._handles: dict[Path, object] = {}
        self._expected_index_sha256: str | None = None
        self._expected_shards: dict[str, dict[str, object]] = {}

    def bind_identity(
        self,
        *,
        index_sha256: str,
        shard_entries: list[dict[str, object]],
    ) -> None:
        if self._index is not None or self._handles:
            raise RuntimeError("cannot bind shard identity after cache access")
        if len(index_sha256) != 64:
            raise RuntimeError("invalid bound shard-index SHA256")
        expected: dict[str, dict[str, object]] = {}
        for entry in shard_entries:
            path = str(Path(str(entry.get("path", ""))).absolute())
            if not path or path in expected:
                raise RuntimeError("invalid or duplicate bound shard path")
            expected[path] = dict(entry)
        if not expected:
            raise RuntimeError("bound shard identity is empty")
        self._expected_index_sha256 = index_sha256
        self._expected_shards = expected

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def _load_index(self) -> dict[str, tuple[Path, int, int]]:
        if self._index is not None:
            return self._index
        index: dict[str, tuple[Path, int, int]] = {}
        raw_index = _open_regular_binary_nofollow(
            self.index_path,
            "window geometry shard index",
        )
        with raw_index:
            index_bytes = raw_index.read()
        if (
            self._expected_index_sha256 is not None
            and hashlib.sha256(index_bytes).hexdigest()
            != self._expected_index_sha256
        ):
            raise RuntimeError("window geometry shard-index digest mismatch")
        try:
            index_text = index_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError("window geometry shard index is not UTF-8") from exc
        with io.StringIO(index_text) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) != 4:
                    raise RuntimeError("invalid window geometry shard-index row")
                name, shard, offset, size = parts
                shard_path = Path(shard)
                if not shard_path.is_absolute():
                    shard_path = self.shard_root / shard_path
                index[name] = (shard_path, int(offset), int(size))
        self._index = index
        return index

    def has(self, name: str) -> bool:
        return name in self._load_index()

    def open_npz(self, name: str):
        entry = self._load_index().get(name)
        if entry is None:
            return None
        shard_path, offset, size = entry
        fh = self._handles.get(shard_path)
        absolute_shard = str(shard_path.absolute())
        expected = self._expected_shards.get(absolute_shard)
        if self._expected_shards and expected is None:
            raise RuntimeError(f"unbound window geometry shard: {absolute_shard}")
        if fh is None or fh.closed:
            fh = _open_regular_binary_nofollow(
                shard_path,
                "window geometry shard",
            )
            metadata = os.fstat(fh.fileno())
            if expected is not None:
                digest = hashlib.sha256()
                for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
                    digest.update(block)
                if (
                    digest.hexdigest() != expected.get("sha256")
                    or int(metadata.st_size) != int(expected.get("size_bytes", -1))
                    or int(metadata.st_ino) != int(expected.get("inode", -1))
                    or int(metadata.st_dev) != int(expected.get("device", -1))
                ):
                    fh.close()
                    raise RuntimeError(
                        f"window geometry shard identity mismatch: {absolute_shard}"
                    )
            self._handles[shard_path] = fh
        before = os.fstat(fh.fileno())
        fh.seek(offset)
        payload = fh.read(size)
        after = os.fstat(fh.fileno())
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            raise RuntimeError(
                f"window geometry shard changed during read: {absolute_shard}"
            )
        return np.load(io.BytesIO(payload))


class _CompositeWindowGeomShardReader:
    """Try multiple independent tar-shard indexes without merging filenames."""

    def __init__(self, readers: list[_WindowGeomShardReader]):
        if not readers:
            raise ValueError("Composite shard reader requires at least one reader")
        self.readers = readers

    def bind_identity(
        self,
        *,
        index_sha256: str,
        shard_entries: list[dict[str, object]],
    ) -> None:
        if len(self.readers) != 1:
            raise RuntimeError(
                "formal pair-gauge identity requires exactly one shard reader"
            )
        self.readers[0].bind_identity(
            index_sha256=index_sha256,
            shard_entries=shard_entries,
        )

    def has(self, name: str) -> bool:
        return any(reader.has(name) for reader in self.readers)

    def open_npz(self, name: str):
        for reader in self.readers:
            npz = reader.open_npz(name)
            if npz is not None:
                return npz
        return None


def _npz_first(npz, names: tuple[str, ...]):
    if npz is None:
        return None
    for name in names:
        if name in npz.files:
            return npz[name]
    return None


def _slice_optional(arr, start: int, end: int):
    if arr is None:
        return None
    return np.array(arr[start:end])


def _future_slice_optional(arr, T: int, k: int):
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.shape[0] >= T + k:
        return np.array(arr[T:T + k])
    if arr.shape[0] >= k:
        return np.array(arr[:k])
    return None


def _scalar_from_record(rec: OXEClipRecord, *names: str):
    for name in names:
        value = getattr(rec, name, None)
        if value is not None:
            return value
    return None


def _load_cache_array(cache_root: Path, subdir: str, cid: str):
    for suffix in (".npy", ".npz"):
        path = cache_root / subdir / f"{cid}{suffix}"
        if not path.exists():
            continue
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            try:
                key = subdir if subdir in arr.files else arr.files[0]
                return np.array(arr[key])
            finally:
                arr.close()
        return arr
    return None


def _select_frame_or_global(arr, frame_idx: int):
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim >= 2 and arr.shape[0] > frame_idx:
        return arr[frame_idx]
    return arr


def _resize_vector(arr, dim: int):
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if dim <= 0:
        return arr
    out = np.zeros(dim, dtype=np.float32)
    n = min(dim, arr.shape[0])
    out[:n] = arr[:n]
    return out


def _resize_action_history(arr, hist_len: int, action_dim: int):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, action_dim) if arr.size % max(1, action_dim) == 0 else arr.reshape(1, -1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])
    out = np.zeros((hist_len, action_dim), dtype=np.float32)
    if hist_len <= 0 or action_dim <= 0:
        return out
    arr = arr[-hist_len:, :action_dim]
    out[-arr.shape[0]:, :arr.shape[1]] = arr
    return out


def _action_history_from_actions(actions, start: int, T: int, hist_len: int, action_dim: int):
    end = start + T
    begin = max(0, end - hist_len)
    hist = np.asarray(actions[begin:end, :action_dim], dtype=np.float32)
    return _resize_action_history(hist, hist_len, action_dim)


def _cache_frame_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim < 1 or arr.shape[0] <= 0:
            return None
        return int(arr.shape[0])
    except Exception:
        return None


def _valid_len(arr, needed: int) -> bool:
    if arr is None:
        return False
    arr = np.asarray(arr)
    return arr.ndim >= 1 and int(arr.shape[0]) >= int(needed)


def _valid_world_points(arr, needed: int) -> bool:
    arr = np.asarray(arr) if arr is not None else None
    return arr is not None and arr.ndim == 4 and arr.shape[-1] == 3 and int(arr.shape[0]) >= int(needed)


def _valid_pose(arr, needed: int) -> bool:
    arr = np.asarray(arr) if arr is not None else None
    return arr is not None and arr.ndim >= 2 and int(arr.shape[0]) >= int(needed) and arr.shape[-1] > 0


def _valid_future_len(arr, T: int, k: int) -> bool:
    arr = np.asarray(arr) if arr is not None else None
    return arr is not None and arr.ndim >= 1 and int(arr.shape[0]) >= min(T + k, k)


def _valid_window_geom(path: Path, T: int, k: int, *, require_extra: bool, require_pooled: bool) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as d:
            if require_pooled:
                pooled = _npz_first(d, ("pooled", "vggt_pooled"))
                if not _valid_len(pooled, T + k):
                    return False
            if not require_extra:
                return True
            depth = _npz_first(d, ("depth", "depth_map"))
            point = _npz_first(d, ("point", "world_points", "points", "point_map"))
            point_conf = _npz_first(d, ("point_conf", "world_points_conf", "points_conf", "conf"))
            pose = _npz_first(d, ("pose", "pose_enc", "camera_pose"))
            depth_conf = _npz_first(d, ("depth_conf", "depth_confidence"))
            return (
                _valid_future_len(depth, T, k)
                and _valid_world_points(point, k)
                and _valid_future_len(point_conf, T, k)
                and _valid_pose(pose, k)
                and _valid_future_len(depth_conf, T, k)
            )
    except Exception:
        return False


def _valid_window_geom_npz(npz, T: int, k: int, *, require_extra: bool, require_pooled: bool) -> bool:
    try:
        if require_pooled:
            pooled = _npz_first(npz, ("pooled", "vggt_pooled"))
            if not _valid_len(pooled, T + k):
                return False
        if not require_extra:
            return True
        depth = _npz_first(npz, ("depth", "depth_map"))
        point = _npz_first(npz, ("point", "world_points", "points", "point_map"))
        point_conf = _npz_first(npz, ("point_conf", "world_points_conf", "points_conf", "conf"))
        pose = _npz_first(npz, ("pose", "pose_enc", "camera_pose"))
        depth_conf = _npz_first(npz, ("depth_conf", "depth_confidence"))
        return (
            _valid_future_len(depth, T, k)
            and _valid_world_points(point, k)
            and _valid_future_len(point_conf, T, k)
            and _valid_pose(pose, k)
            and _valid_future_len(depth_conf, T, k)
        )
    except Exception:
        return False


def _progress_from_geom_or_record(geom, rec: OXEClipRecord):
    progress_arr = _npz_first(geom, ("progress", "progress_tgt"))
    if progress_arr is not None:
        return np.asarray(progress_arr, dtype=np.float32)
    rec_progress = getattr(rec, "progress", None)
    if rec_progress is not None:
        return np.asarray(rec_progress, dtype=np.float32)
    return None


def _valid_progress(arr, needed: int) -> bool:
    if arr is None:
        return False
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    return arr.shape[0] >= int(needed) and bool(np.isfinite(arr).all()) and float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0


def _has_policy_state(cache_root: Path, geom, cid: str, sample_key: str, geom_keys: tuple[str, ...], dim: int, needed: int) -> bool:
    if dim <= 0:
        return True
    arr = _npz_first(geom, geom_keys)
    if arr is None:
        arr = _load_cache_array(cache_root, sample_key, cid)
    if arr is None:
        return False
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return False
    if arr.ndim >= 2 and arr.shape[0] not in (1, dim) and arr.shape[0] < int(needed):
        return False
    return arr.size > 0


class OXEWindowDataset(Dataset):
    """One sample = T input frames + k target frames.

    Optional policy-state geometry is loaded lazily at sample time. A corrupt
    geom archive therefore raises from ``__getitem__`` unless
    ``strict_policy_state_prescan`` is enabled for eager validation.
    """

    def __init__(self, records: list[OXEClipRecord], cfg: WindowConfig | None = None):
        scan_started = time.monotonic()
        self.cfg = cfg or WindowConfig()
        self.cfg.cache_root = Path(self.cfg.cache_root)
        _dataset_startup_log(
            "begin",
            scan_started,
            records=len(records),
            load_geom=self.cfg.load_geom,
            load_policy_state=self.cfg.load_policy_state,
            require_policy_state=self.cfg.require_policy_state,
            strict_policy_state_prescan=self.cfg.strict_policy_state_prescan,
        )
        if self.cfg.action_stats is not None:
            self.cfg.action_stats = Path(self.cfg.action_stats)
        if self.cfg.manifest_path is not None:
            self.cfg.manifest_path = Path(self.cfg.manifest_path)
        if self.cfg.trusted_manifest_fast_init:
            if self.cfg.manifest_path is None or not self.cfg.trusted_manifest_sha256:
                raise ValueError(
                    "trusted_manifest_fast_init requires manifest_path and "
                    "trusted_manifest_sha256"
                )
            manifest_bytes = self.cfg.manifest_path.read_bytes()
            observed_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            if observed_manifest_sha256 != self.cfg.trusted_manifest_sha256:
                raise RuntimeError(
                    "trusted fast-init manifest digest mismatch: "
                    f"observed={observed_manifest_sha256} "
                    f"expected={self.cfg.trusted_manifest_sha256}"
                )
        if self.cfg.droid_cache_index_path is not None:
            self.cfg.droid_cache_index_path = Path(
                self.cfg.droid_cache_index_path
            )
        has_droid = any(
            str(record.dataset).strip().lower() == "droid"
            for record in records
        )
        droid_index_bytes: bytes | None = None
        droid_index_sha256: str | None = None
        if self.cfg.require_action_contract and has_droid:
            if self.cfg.droid_cache_index_path is None:
                raise ActionCacheResolutionError(
                    "formal DROID Stage1 requires droid_cache_index"
                )
            droid_index_bytes = read_stable_cache_bytes(
                self.cfg.droid_cache_index_path,
                "finalized DROID cache index",
            )
            droid_index_sha256 = hashlib.sha256(droid_index_bytes).hexdigest()
        if self.cfg.action_contract_path is not None:
            self.cfg.action_contract_path = Path(self.cfg.action_contract_path)
            self._action_frame_offsets = load_passed_contracts(
                self.cfg.action_contract_path,
                expected_droid_cache_sha256=droid_index_sha256,
            )
        else:
            self._action_frame_offsets = {}
        if self.cfg.require_action_contract:
            if self.cfg.action_contract_path is None:
                raise ActionContractCoverageError(
                    "require_action_contract=True but action_contract_path is unset"
                )
            validate_manifest_contract_coverage(records, self._action_frame_offsets)
        if self.cfg.require_action_contract and has_droid:
            assert self.cfg.droid_cache_index_path is not None
            assert droid_index_bytes is not None
            assert droid_index_sha256 is not None
            self._droid_cache_index_records = validate_formal_droid_cache_index(
                records,
                cache_root=self.cfg.cache_root,
                index_path=self.cfg.droid_cache_index_path,
                index_payload_bytes=droid_index_bytes,
                expected_index_sha256=droid_index_sha256,
            )
        else:
            self._droid_cache_index_records = {}
        self._action_contract_excluded_windows = 0
        if self.cfg.window_geom_shard_index is not None:
            self.cfg.window_geom_shard_index = Path(self.cfg.window_geom_shard_index)
        if self.cfg.window_geom_shard_root is not None:
            self.cfg.window_geom_shard_root = Path(self.cfg.window_geom_shard_root)
        shard_indices: list[Path] = []
        shard_roots: list[Path | None] = []
        if self.cfg.window_geom_shard_index is not None:
            shard_indices.append(Path(self.cfg.window_geom_shard_index))
            shard_roots.append(Path(self.cfg.window_geom_shard_root) if self.cfg.window_geom_shard_root is not None else None)
        if self.cfg.window_geom_shard_indices is not None:
            raw_roots = tuple(self.cfg.window_geom_shard_roots or ())
            for idx, index_path in enumerate(self.cfg.window_geom_shard_indices):
                shard_indices.append(Path(index_path))
                root = raw_roots[idx] if idx < len(raw_roots) else None
                shard_roots.append(Path(root) if root is not None else None)
        readers = [
            _WindowGeomShardReader(index_path, root_path)
            for index_path, root_path in zip(shard_indices, shard_roots)
            if Path(index_path).exists()
        ]
        if len(readers) == 1:
            self._window_shards = readers[0]
        elif len(readers) > 1:
            self._window_shards = _CompositeWindowGeomShardReader(readers)
        else:
            self._window_shards = None
        self.act_mean: np.ndarray | None = None
        self.act_std: np.ndarray | None = None
        if self.cfg.action_stats is not None and Path(self.cfg.action_stats).exists():
            d = np.load(self.cfg.action_stats)
            self.act_mean = d["mean"][:6].astype(np.float32)
            self.act_std = d["std"][:6].astype(np.float32)
        self.records = []
        self._usable_frames: list[int] = []
        self._valid_starts: list[list[int] | None] = []
        missing_task_emb: list[str] = []
        init_errors: list[str] = []
        init_geom_opened = 0
        win = self.cfg.T + self.cfg.k
        trusted_window_fast_init = (
            self.cfg.use_window_tokens
            and self.cfg.trust_window_geom_cache
            and not self.cfg.load_policy_state
            and not self.cfg.require_progress
            and self._window_shards is None
        )
        trusted_manifest_shard_fast_init = (
            self.cfg.trusted_manifest_fast_init
            and self.cfg.use_window_tokens
            and self.cfg.trust_window_geom_cache
            and not self.cfg.load_policy_state
            and not self.cfg.require_progress
            and not self.cfg.require_action_contract
            and self._window_shards is not None
        )
        for r in records:
            cid = _safe(r.clip_id)
            record_frame_count = int(getattr(r, "n_frames", 0) or 0)
            if trusted_manifest_shard_fast_init:
                if record_frame_count < win:
                    continue
                starts = list(
                    range(0, record_frame_count - win + 1, self.cfg.stride)
                )
                if self.cfg.max_windows_per_episode and len(starts) > int(
                    self.cfg.max_windows_per_episode
                ):
                    starts = starts[: int(self.cfg.max_windows_per_episode)]
                starts = [
                    start
                    for start in starts
                    if self._window_shards.has(_window_geom_name(cid, start))
                ]
                starts = self._filter_action_starts(
                    r,
                    starts,
                    record_frame_count,
                )
                if not starts:
                    continue
                self.records.append(r)
                self._usable_frames.append(record_frame_count)
                self._valid_starts.append(starts)
                continue
            action_cache = resolve_action_cache(
                r,
                cache_root=self.cfg.cache_root,
                formal_stage1=self.cfg.require_action_contract,
            )
            validated_droid = self._droid_cache_index_records.get(r.clip_id)
            if validated_droid is not None:
                action_len = int(validated_droid.actions.shape[0])
            else:
                action_len = _cache_frame_count(action_cache.path)
                if action_len is None:
                    if action_cache.cache_subdir != DEFAULT_ACTION_CACHE_SUBDIR:
                        raise FileNotFoundError(
                            "missing required dedicated action cache for "
                            f"{r.clip_id}: {action_cache.path}"
                        )
                    continue
            action_valid_count = action_cache.valid_action_count(action_len)
            if trusted_window_fast_init:
                usable = record_frame_count or action_len
                if usable < win:
                    continue
                if self.cfg.require_task_emb and not (self.cfg.cache_root / "qwen_taskemb" / f"{cid}.npy").exists():
                    missing_task_emb.append(r.clip_id)
                    continue
                starts = list(range(0, usable - win + 1, self.cfg.stride))
                if self.cfg.max_windows_per_episode and len(starts) > int(self.cfg.max_windows_per_episode):
                    starts = starts[: int(self.cfg.max_windows_per_episode)]
                starts = self._filter_action_starts(r, starts, action_valid_count)
                if not starts:
                    continue
                self.records.append(r)
                self._usable_frames.append(usable)
                self._valid_starts.append(starts)
                continue
            if self.cfg.use_window_tokens:
                token_len = record_frame_count or action_len
            else:
                token_len = _cache_frame_count(self.cfg.cache_root / self.cfg.tokens_subdir / f"{cid}.npy")
                if token_len is None:
                    continue
            lengths = [token_len, record_frame_count or token_len]
            if self.cfg.load_rgb:
                rgb_len = _cache_frame_count(self.cfg.cache_root / "rgb_256" / f"{cid}.npy")
                if rgb_len is None:
                    continue
                lengths.append(rgb_len)
            geom_path = self.cfg.cache_root / "vggt_geom" / f"{cid}.npz"
            geom = None
            try:
                skip_init_geom_decode = (
                    self.cfg.load_geom
                    and self.cfg.use_window_tokens
                    and self.cfg.trust_window_geom_cache
                    and not self.cfg.require_progress
                )
                # Optional policy state remains a sample-time input, but it does
                # not determine record validity and need not decode every geom
                # archive during this initialization prescan.
                need_init_geom_file = (
                    self.cfg.load_geom
                    or self.cfg.require_progress
                    or (
                        self.cfg.load_policy_state
                        and (self.cfg.require_policy_state or self.cfg.strict_policy_state_prescan)
                    )
                )
                if skip_init_geom_decode or not need_init_geom_file or not geom_path.exists():
                    geom = None
                else:
                    geom = np.load(geom_path)
                    init_geom_opened += 1
                if self.cfg.load_geom:
                    if skip_init_geom_decode:
                        if not geom_path.exists():
                            continue
                    elif geom is None:
                        continue
                    else:
                        depth_arr = _npz_first(geom, ("depth", "depth_map"))
                        if not _valid_len(depth_arr, win):
                            if self.cfg.require_geom_extra:
                                init_errors.append(f"{r.clip_id}: depth missing/short in vggt_geom")
                            continue
                        lengths.append(int(np.asarray(depth_arr).shape[0]))
                        point_arr = _npz_first(geom, ("world_points", "point", "points", "point_map"))
                        point_conf_arr = _npz_first(geom, ("world_points_conf", "point_conf", "points_conf", "conf"))
                        pose_arr = _npz_first(geom, ("pose_enc", "pose", "camera_pose"))
                        depth_conf_arr = _npz_first(geom, ("depth_conf", "depth_confidence"))
                        if self.cfg.require_geom_extra and not (self.cfg.load_geom_extra or self.cfg.use_window_tokens):
                            missing = []
                            if not _valid_world_points(point_arr, win):
                                missing.append("world_points")
                            if not _valid_len(point_conf_arr, win):
                                missing.append("world_points_conf")
                            if not _valid_pose(pose_arr, win):
                                missing.append("pose_enc")
                            if not _valid_len(depth_conf_arr, win):
                                missing.append("depth_conf")
                            if missing:
                                init_errors.append("{}: require_geom_extra missing/short {}".format(r.clip_id, ", ".join(missing)))
                                continue
                        for extra_arr in (point_arr, point_conf_arr, pose_arr, depth_conf_arr):
                            if extra_arr is not None and np.asarray(extra_arr).ndim >= 1:
                                lengths.append(int(np.asarray(extra_arr).shape[0]))
                if self.cfg.load_policy_state and (
                    self.cfg.require_policy_state or self.cfg.strict_policy_state_prescan
                ):
                    policy_specs = (
                        ("lowdim_state", ("lowdim_state", "lowdim", "robot_state"), self.cfg.policy_lowdim_dim),
                        ("object_state", ("object_state", "objects_state"), self.cfg.policy_object_state_dim),
                        ("plan_state", ("plan_state", "stage_state", "task_plan_state"), self.cfg.policy_plan_state_dim),
                    )
                    for sample_key, geom_keys, dim in policy_specs:
                        has_policy_state = _has_policy_state(
                            self.cfg.cache_root,
                            geom,
                            cid,
                            sample_key,
                            geom_keys,
                            dim,
                            win,
                        )
                        if self.cfg.require_policy_state and not has_policy_state:
                            init_errors.append(f"{r.clip_id}: missing {sample_key}")
                if self.cfg.require_progress:
                    progress_arr = _progress_from_geom_or_record(geom, r)
                    if progress_arr is None and self.cfg.allow_pseudo_progress_targets:
                        pass
                    elif not _valid_progress(progress_arr, win):
                        init_errors.append(f"{r.clip_id}: require_progress missing, short, or outside [0, 1]")
            finally:
                if geom is not None:
                    geom.close()
            if self.cfg.require_task_emb and not (self.cfg.cache_root / "qwen_taskemb" / f"{cid}.npy").exists():
                missing_task_emb.append(r.clip_id)
                continue
            usable = min(lengths)
            if usable < win:
                continue
            valid_starts = None
            if self.cfg.load_geom_extra or self.cfg.require_geom_extra or self.cfg.use_window_tokens:
                valid_starts = []
                starts = list(range(0, usable - win + 1, self.cfg.stride))
                if self.cfg.max_windows_per_episode and len(starts) > int(self.cfg.max_windows_per_episode):
                    starts = starts[: int(self.cfg.max_windows_per_episode)]
                for start in starts:
                    path = _window_geom_path(self.cfg.cache_root, self.cfg.window_geom_subdir, cid, start)
                    name = _window_geom_name(cid, start)
                    if self.cfg.trust_window_geom_cache:
                        if path.exists() or (self._window_shards is not None and self._window_shards.has(name)):
                            valid_starts.append(start)
                    elif _valid_window_geom(
                        path,
                        self.cfg.T,
                        self.cfg.k,
                        require_extra=self.cfg.require_geom_extra or self.cfg.load_geom_extra,
                        require_pooled=self.cfg.use_window_tokens,
                    ):
                        valid_starts.append(start)
                    elif self._window_shards is not None and self._window_shards.has(name):
                        with self._window_shards.open_npz(name) as d:
                            if _valid_window_geom_npz(
                                d,
                                self.cfg.T,
                                self.cfg.k,
                                require_extra=self.cfg.require_geom_extra or self.cfg.load_geom_extra,
                                require_pooled=self.cfg.use_window_tokens,
                            ):
                                valid_starts.append(start)
                if not valid_starts:
                    if self._window_shards is not None and self.cfg.trust_window_geom_cache:
                        # Node-sharded tar caches may share or symlink a broader
                        # base cache than the local window shard owns.
                        continue
                    if self.cfg.require_geom_extra or self.cfg.use_window_tokens:
                        init_errors.append(f"{r.clip_id}: no usable window geom cache in {self.cfg.window_geom_subdir}")
                    continue
            if valid_starts is None:
                valid_starts = list(range(0, usable - win + 1, self.cfg.stride))
                if self.cfg.max_windows_per_episode and len(valid_starts) > int(self.cfg.max_windows_per_episode):
                    valid_starts = valid_starts[: int(self.cfg.max_windows_per_episode)]
            valid_starts = self._filter_action_starts(
                r,
                valid_starts,
                action_valid_count,
            )
            if not valid_starts:
                continue
            self.records.append(r)
            self._usable_frames.append(usable)
            self._valid_starts.append(valid_starts)
        if missing_task_emb:
            preview = ", ".join(missing_task_emb[:5])
            more = "" if len(missing_task_emb) <= 5 else f", ... +{len(missing_task_emb) - 5} more"
            raise RuntimeError(
                "require_task_emb=True but Qwen task embeddings are missing for "
                f"{len(missing_task_emb)} cached clips: {preview}{more}"
            )
        if init_errors:
            preview = "; ".join(init_errors[:5])
            more = "" if len(init_errors) <= 5 else f"; ... +{len(init_errors) - 5} more"
            raise RuntimeError(preview + more)
        self.index: list[tuple[int, int]] = []
        for i, usable in enumerate(self._usable_frames):
            starts = self._valid_starts[i]
            if starts is None:
                starts = list(range(0, usable - win + 1, self.cfg.stride))
            for start in starts:
                self.index.append((i, start))
        _dataset_startup_log(
            "end",
            scan_started,
            records=len(self.records),
            windows=len(self.index),
            action_contract_excluded_windows=self._action_contract_excluded_windows,
            init_geom_opened=init_geom_opened,
            optional_policy_prescan_skipped=(
                self.cfg.load_policy_state
                and not self.cfg.require_policy_state
                and not self.cfg.strict_policy_state_prescan
            ),
            policy_state_sample_load=self.cfg.load_policy_state,
            trusted_manifest_fast_init=trusted_manifest_shard_fast_init,
        )

    def _action_offset_for(self, record: OXEClipRecord) -> int:
        if not self.cfg.require_action_contract and not self._action_frame_offsets:
            return int(self.cfg.default_action_frame_offset)
        key = action_contract_key(record)
        offset = self._action_frame_offsets.get(key)
        if offset is None:
            if self.cfg.require_action_contract:
                raise ActionContractCoverageError(f"missing passed action contract for {key}")
            return 0
        return int(offset)

    def _resolve_action_window(self, record: OXEClipRecord, start: int, action_len: int):
        try:
            return resolve_action_window(
                record,
                start=start,
                T=self.cfg.T,
                k=self.cfg.k,
                offset=self._action_offset_for(record),
                n_action_frames=action_len,
            )
        except UnknownDatasetAlias:
            if self.cfg.require_action_contract or self._action_frame_offsets:
                raise
            target_indices = tuple(
                range(start + self.cfg.T, start + self.cfg.T + self.cfg.k)
            )
            action_indices = target_indices
            previous_gripper_index = action_indices[0] - 1
            if (
                previous_gripper_index < 0
                or max(target_indices) >= int(record.n_frames)
                or max(action_indices) >= int(action_len)
            ):
                raise ActionContractBoundaryError(record.clip_id, start, 0)
            raw_key = "|".join(
                (
                    str(record.dataset).strip().lower(),
                    format(float(record.fps), ".8g"),
                    str(record.action_kind).strip(),
                )
            )
            return ActionWindowResolution(
                contract_key=raw_key,
                action_frame_offset=0,
                target_frame_indices=target_indices,
                action_frame_indices=action_indices,
                previous_gripper_index=previous_gripper_index,
            )

    def _filter_action_starts(
        self,
        record: OXEClipRecord,
        starts: list[int],
        action_len: int,
    ) -> list[int]:
        valid: list[int] = []
        for start in starts:
            try:
                self._resolve_action_window(record, start, action_len)
            except ActionContractBoundaryError:
                self._action_contract_excluded_windows += 1
                continue
            valid.append(start)
        return valid

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        ri, start = self.index[i]
        rec = self.records[ri]
        cid = _safe(rec.clip_id)
        T, k = self.cfg.T, self.cfg.k
        window_geom = None
        window_geom_path = _window_geom_path(self.cfg.cache_root, self.cfg.window_geom_subdir, cid, start)
        if self.cfg.load_geom_extra or self.cfg.require_geom_extra or self.cfg.use_window_tokens:
            if window_geom_path.exists():
                window_geom = np.load(window_geom_path)
            elif self._window_shards is not None:
                window_geom = self._window_shards.open_npz(_window_geom_name(cid, start))
        pooled = None if self.cfg.use_window_tokens else np.load(self.cfg.cache_root / self.cfg.tokens_subdir / f"{cid}.npy", mmap_mode="r")
        need_geom_file = self.cfg.load_geom or self.cfg.load_policy_state or self.cfg.require_progress
        geom_path = self.cfg.cache_root / "vggt_geom" / f"{cid}.npz"
        geom = np.load(geom_path) if need_geom_file and geom_path.exists() else None
        rgb = np.load(self.cfg.cache_root / "rgb_256" / f"{cid}.npy", mmap_mode="r") if self.cfg.load_rgb else None
        action_cache = resolve_action_cache(
            rec,
            cache_root=self.cfg.cache_root,
            formal_stage1=self.cfg.require_action_contract,
        )
        validated_droid = self._droid_cache_index_records.get(rec.clip_id)
        if validated_droid is not None:
            actions = validated_droid.actions
        else:
            actions = np.load(action_cache.path, mmap_mode="r")
        action_valid_count = action_cache.valid_action_count(actions.shape[0])
        qwen_p = self.cfg.cache_root / "qwen_taskemb" / f"{cid}.npy"
        if qwen_p.exists():
            qwen = np.load(qwen_p)
        else:
            if self.cfg.require_task_emb:
                raise FileNotFoundError(f"missing required Qwen task embedding: {qwen_p}")
            qwen = np.zeros(2048, dtype=np.float16)

        if self.cfg.load_geom:
            depth_arr = _npz_first(geom, ("depth", "depth_map"))
        else:
            depth_arr = None
        window_depth_arr = None
        if self.cfg.load_geom_extra or self.cfg.require_geom_extra or self.cfg.use_window_tokens:
            window_depth_arr = _npz_first(window_geom, ("depth", "depth_map"))
            point_arr = _npz_first(window_geom, ("point", "world_points", "points", "point_map"))
            point_conf_arr = _npz_first(window_geom, ("point_conf", "world_points_conf", "points_conf", "conf"))
            pose_arr = _npz_first(window_geom, ("pose", "pose_enc", "camera_pose"))
            pose_conf_arr = _npz_first(window_geom, ("pose_conf", "pose_confidence"))
            depth_conf_arr = _npz_first(window_geom, ("depth_conf", "depth_confidence"))
        elif self.cfg.load_geom:
            point_arr = _npz_first(geom, ("world_points", "point", "points", "point_map"))
            point_conf_arr = _npz_first(geom, ("world_points_conf", "point_conf", "points_conf", "conf"))
            pose_arr = _npz_first(geom, ("pose_enc", "pose", "camera_pose"))
            pose_conf_arr = _npz_first(geom, ("pose_conf", "pose_confidence"))
            depth_conf_arr = _npz_first(geom, ("depth_conf", "depth_confidence"))
        else:
            point_arr = point_conf_arr = pose_arr = pose_conf_arr = depth_conf_arr = None
        if self.cfg.require_geom_extra and (point_arr is None or pose_arr is None):
            missing = []
            if point_arr is None:
                missing.append("world_points/point")
            if pose_arr is None:
                missing.append("pose_enc/pose")
            raise KeyError(f"require_geom_extra=True but vggt_geom missing {', '.join(missing)} for {rec.clip_id}")

        end = start + T + k
        required_lengths = []
        if pooled is not None:
            required_lengths.append(pooled.shape[0])
        if rgb is not None:
            required_lengths.append(rgb.shape[0])
        if depth_arr is not None:
            required_lengths.append(depth_arr.shape[0])
        for extra_arr in (point_arr, point_conf_arr, pose_arr, pose_conf_arr, depth_conf_arr):
            if extra_arr is not None and np.asarray(extra_arr).shape[0] >= T + k:
                required_lengths.append(extra_arr.shape[0])
        if any(length < end for length in required_lengths):
            raise RuntimeError(f"cache window shorter than indexed range for {rec.clip_id}: end={end} lengths={required_lengths}")
        pooled_len = T + k if self.cfg.load_state_tgt else T
        if self.cfg.use_window_tokens:
            pooled_arr = _npz_first(window_geom, ("pooled", "vggt_pooled"))
            if pooled_arr is None or np.asarray(pooled_arr).shape[0] < pooled_len:
                raise KeyError(f"use_window_tokens=True but window geom missing pooled tokens for {rec.clip_id} start={start}")
            pooled_w = np.array(pooled_arr[:pooled_len])
        else:
            pooled_w = np.array(pooled[start : start + pooled_len])
        rgb_w = np.array(rgb[start : start + T + k]) if rgb is not None else None
        depth_w = _slice_optional(depth_arr, start, start + T + k)
        depth_tgt_w = depth_w[T:] if depth_w is not None else None
        if self.cfg.load_geom_extra or self.cfg.require_geom_extra or self.cfg.use_window_tokens:
            window_depth_tgt_w = _future_slice_optional(window_depth_arr, T, k)
            if window_depth_tgt_w is not None:
                depth_tgt_w = window_depth_tgt_w
            point_tgt_w = _future_slice_optional(point_arr, T, k)
            point_conf_tgt_w = _future_slice_optional(point_conf_arr, T, k)
            pose_tgt_w = _future_slice_optional(pose_arr, T, k)
            pose_conf_tgt_w = _future_slice_optional(pose_conf_arr, T, k)
            depth_conf_tgt_w = _future_slice_optional(depth_conf_arr, T, k)
        else:
            point_w = _slice_optional(point_arr, start, start + T + k)
            point_conf_w = _slice_optional(point_conf_arr, start, start + T + k)
            pose_w = _slice_optional(pose_arr, start, start + T + k)
            pose_conf_w = _slice_optional(pose_conf_arr, start, start + T + k)
            depth_conf_w = _slice_optional(depth_conf_arr, start, start + T + k)
            point_tgt_w = point_w[T:] if point_w is not None else None
            point_conf_tgt_w = point_conf_w[T:] if point_conf_w is not None else None
            pose_tgt_w = pose_w[T:] if pose_w is not None else None
            pose_conf_tgt_w = pose_conf_w[T:] if pose_conf_w is not None else None
            depth_conf_tgt_w = depth_conf_w[T:] if depth_conf_w is not None else None
        action_resolution = self._resolve_action_window(rec, start, action_valid_count)
        act_w = np.array(actions[list(action_resolution.action_frame_indices)])
        prev_grip = np.array(
            [actions[action_resolution.previous_gripper_index, 6]],
            dtype=np.float32,
        )
        frame_ids = np.arange(start + T, start + T + k, dtype=np.float32)
        denom = np.float32(max(1, rec.n_frames - 1))
        pseudo_progress_tgt = np.clip(frame_ids / denom, 0.0, 1.0).astype(np.float32)
        if self.act_mean is not None:
            action_tgt_norm = (act_w[:, :6] - self.act_mean) / self.act_std
        else:
            action_tgt_norm = act_w[:, :6].astype(np.float32)
        sample = {
            "s_in":            torch.from_numpy(pooled_w[:T]).float(),
            "action_tgt":      torch.from_numpy(act_w).float(),
            "action_tgt_norm": torch.from_numpy(action_tgt_norm).float(),
            "action_prev_grip": torch.from_numpy(prev_grip).float(),
            "c":               torch.from_numpy(np.asarray(qwen, dtype=np.float16)).float(),
            "clip_id":         rec.clip_id,
            "start":           start,
            "dataset":         rec.dataset,
            "action_cache_subdir": action_cache.cache_subdir,
            "action_valid_count": action_valid_count,
            "action_contract_key": action_resolution.contract_key,
            "action_frame_offset": action_resolution.action_frame_offset,
            "action_frame_indices": torch.tensor(
                action_resolution.action_frame_indices, dtype=torch.long
            ),
        }
        if self.cfg.load_state_tgt:
            sample["s_tgt"] = torch.from_numpy(pooled_w[T:]).float()
        if depth_w is not None:
            sample["depth_in"] = torch.from_numpy(depth_w[:T]).float()
        if depth_tgt_w is not None:
            sample["depth_tgt"] = torch.from_numpy(depth_tgt_w).float()
        if depth_conf_tgt_w is not None:
            sample["depth_conf_tgt"] = torch.from_numpy(depth_conf_tgt_w).float()
        if point_tgt_w is not None:
            sample["point_tgt"] = torch.from_numpy(point_tgt_w).float()
        if point_conf_tgt_w is not None:
            sample["point_conf_tgt"] = torch.from_numpy(point_conf_tgt_w).float()
        if pose_tgt_w is not None:
            sample["pose_geom_tgt"] = torch.from_numpy(pose_tgt_w).float()
        if pose_conf_tgt_w is not None:
            sample["pose_geom_conf_tgt"] = torch.from_numpy(pose_conf_tgt_w).float()
        if rgb_w is not None:
            sample["rgb_in"] = torch.from_numpy(rgb_w[:T]).float() / 255.0
            sample["rgb_tgt"] = torch.from_numpy(rgb_w[T:]).float() / 255.0

        if self.cfg.load_policy_state:
            frame_idx = start + T - 1
            policy_specs = (
                ("lowdim_state", ("lowdim_state", "lowdim", "robot_state"), self.cfg.policy_lowdim_dim),
                ("object_state", ("object_state", "objects_state"), self.cfg.policy_object_state_dim),
                ("plan_state", ("plan_state", "stage_state", "task_plan_state"), self.cfg.policy_plan_state_dim),
            )
            for sample_key, geom_keys, dim in policy_specs:
                arr = _npz_first(geom, geom_keys)
                if arr is None:
                    arr = _load_cache_array(self.cfg.cache_root, sample_key, cid)
                vec = _resize_vector(_select_frame_or_global(arr, frame_idx), dim)
                if vec is not None:
                    sample[sample_key] = torch.from_numpy(vec).float()
                elif self.cfg.require_policy_state and dim > 0:
                    raise KeyError(f"require_policy_state=True but missing {sample_key} for {rec.clip_id}")
            if self.cfg.policy_action_history_len > 0:
                hist_arr = _npz_first(geom, ("action_history", "action_hist"))
                if hist_arr is None:
                    hist_arr = _load_cache_array(self.cfg.cache_root, "action_history", cid)
                if hist_arr is not None:
                    hist_arr = _select_frame_or_global(hist_arr, frame_idx)
                    hist = _resize_action_history(
                        hist_arr,
                        self.cfg.policy_action_history_len,
                        self.cfg.policy_action_history_dim,
                    )
                else:
                    hist = _action_history_from_actions(
                        actions,
                        start,
                        T,
                        self.cfg.policy_action_history_len,
                        self.cfg.policy_action_history_dim,
                    )
                sample["action_history"] = torch.from_numpy(hist).float()

        progress_arr = _npz_first(geom, ("progress", "progress_tgt"))
        if progress_arr is not None:
            sample["progress_tgt"] = torch.from_numpy(np.array(progress_arr[start + T : start + T + k], dtype=np.float32)).float()
        elif getattr(rec, "progress", None) is not None:
            sample["progress_tgt"] = torch.tensor(rec.progress[start + T : start + T + k], dtype=torch.float32)
        elif self.cfg.allow_pseudo_progress_targets:
            sample["progress_tgt"] = torch.from_numpy(pseudo_progress_tgt).float()

        terminal_arr = _npz_first(geom, ("terminal_success", "terminal_success_tgt", "success"))
        terminal_value = None
        if terminal_arr is not None:
            terminal_value = terminal_arr[start + T + k - 1] if np.ndim(terminal_arr) > 0 else terminal_arr
        else:
            terminal_value = _scalar_from_record(rec, "terminal_success", "terminal_success_tgt", "success")
        if terminal_value is not None:
            sample["terminal_success_tgt"] = torch.tensor(float(terminal_value), dtype=torch.float32)
        elif self.cfg.allow_pseudo_progress_targets:
            sample["terminal_success_tgt"] = torch.tensor(float(pseudo_progress_tgt[-1]), dtype=torch.float32)

        plaus_arr = _npz_first(geom, ("plausibility", "plausibility_tgt"))
        plaus_value = None
        if plaus_arr is not None:
            plaus_value = plaus_arr[start + T + k - 1] if np.ndim(plaus_arr) > 0 else plaus_arr
        else:
            plaus_value = _scalar_from_record(rec, "plausibility", "plausibility_tgt")
        if plaus_value is not None:
            sample["plausibility_tgt"] = torch.tensor(float(plaus_value), dtype=torch.float32)
        elif self.cfg.allow_pseudo_progress_targets:
            sample["plausibility_tgt"] = torch.tensor(1.0, dtype=torch.float32)
        if self.cfg.load_task_text:
            sample["task_text"] = str(getattr(rec, "task_text", "") or "")
        return sample
