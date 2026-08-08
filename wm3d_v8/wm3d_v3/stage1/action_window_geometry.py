from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Iterator, Mapping as MappingABC
from contextlib import contextmanager
from io import BytesIO
import fcntl
import gc
import hashlib
import json
import os
import platform
import re
from pathlib import Path
import stat
import subprocess
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

from wm3d_v3.data.manifest import OXEClipRecord
from wm3d_v3.stage1.action_cache import ActionCacheResolutionError,validate_formal_droid_cache_index
from wm3d_v3.stage1.action_contract import action_contract_key, canonical_dataset_name
from wm3d_v3.stage1.action_contract_evidence import FORMAL_DROID_CONTRACT_KEY, FORMAL_OXE_CONTRACT_KEYS
from wm3d_v3.stage1.action_contract_split import ActionContractSplitError, frozen_contract_split_from_mapping
from wm3d_v3.stage1.action_evidence_sources import (
    EVIDENCE_CANDIDATE_OFFSETS, EVIDENCE_CONTEXT_LENGTH,
    TEMPORAL_WINDOW_BINDING_SCHEMA, TEMPORAL_WINDOW_DERIVATION,
    bind_temporal_window, safe_clip_id,
)
from wm3d_v3.stage1.immutable_artifact import (
    ImmutableArtifactConflict,
    publish_immutable_bytes,
)

SCHEMA_VERSION = "wm3d_v6_stage1_action_window_geometry_index_v1"
ARTIFACT_TYPE = "stage1_action_evidence_continuous_window_vggt_geometry"
ROLES = ("calibration", "qualification", "confirmation")
WINDOW_LENGTH = 17
TARGET_LENGTH = 16
SPLIT_SCHEMA = "wm3d_v6_action_contract_split_v1"
SPLIT_SEED = 1729
SPLIT_DERIVATION = "sha256(seed|contract_key|independence_group_id)"
SPLIT_TEMPORAL_BINDING_SCHEMA = "wm3d_v6_action_temporal_window_binding_spec_v1"
EXPECTED_CONTRACT_KEYS = frozenset((*FORMAL_OXE_CONTRACT_KEYS, FORMAL_DROID_CONTRACT_KEY))
EXPECTED_WINDOWS = 96 * 6
VGGT_MODEL_REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"
VGGT_SOURCE_ROOT = "/data/world_model_workspace/world_model/vggt"
VGGT_SOURCE_COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"
FORMAL_RUNTIME = {
    "python": "3.10.12",
    "torch": "2.7.1+cu128",
    "torch_cuda": "12.8",
    "torch_cudnn": "90701",
    "numpy": "1.26.4",
    "transformers": "5.9.0",
    "nvidia_driver": "570.211.01",
}


class ActionWindowGeometryError(RuntimeError):
    pass


GeometryIdentity = tuple[str, str, str, int]


@dataclass(frozen=True)
class ValidatedGeometryIndex(MappingABC[GeometryIdentity, Mapping[str, Any]]):
    index_path: str
    index_sha256: str
    metadata: Mapping[str, Any]
    entries: Mapping[GeometryIdentity, Mapping[str, Any]]

    def __getitem__(self, key: GeometryIdentity) -> Mapping[str, Any]:
        return self.entries[key]

    def __iter__(self) -> Iterator[GeometryIdentity]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


@dataclass(frozen=True, order=True)
class ActionWindowCandidate:
    contract_key: str
    role: str
    clip_id: str
    start: int
    group_id: str
    legal_starts_sha256: str
    selection_sha256: str
    action_path: str = ""
    action_sha256: str = ""
    action_shape: tuple[int, ...] = ()
    action_dtype: str = ""
    droid_cache_index_path: str | None = None
    droid_cache_index_sha256: str | None = None
    droid_cache_root: str | None = None

    @property
    def target_indices(self) -> tuple[int, ...]:
        return tuple(range(self.start + 1, self.start + WINDOW_LENGTH))

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return (self.start, *self.target_indices)


@dataclass(frozen=True)
class GeometryConfig:
    model_name: str = "facebook/VGGT-1B"
    dtype: str = "bfloat16"
    input_size: int = 224
    context_length: int = 1
    target_length: int = TARGET_LENGTH
    input_range: str = "uint8_to_float_0_1"
    resize: str = "bilinear_antialias_align_corners_false"
    outputs: tuple[str, ...] = ("depth", "depth_conf")
    model_revision: str = VGGT_MODEL_REVISION
    vggt_source_commit: str = VGGT_SOURCE_COMMIT

    def validate(self) -> None:
        if (self.context_length, self.target_length) != (1, 16):
            raise ActionWindowGeometryError("window must remain predecessor+16 targets")
        if self.input_size != 224:
            raise ActionWindowGeometryError("VGGT input_size must remain 224")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ActionWindowGeometryError(f"unsupported dtype: {self.dtype}")
        if self.outputs != ("depth", "depth_conf"):
            raise ActionWindowGeometryError("outputs must be depth and depth_conf")

        if self.model_name != "facebook/VGGT-1B":
            raise ActionWindowGeometryError("formal VGGT model must remain facebook/VGGT-1B")
        if self.model_revision != VGGT_MODEL_REVISION:
            raise ActionWindowGeometryError(
                f"formal VGGT revision must remain {VGGT_MODEL_REVISION}"
            )
        if self.vggt_source_commit != VGGT_SOURCE_COMMIT:
            raise ActionWindowGeometryError(
                f"formal VGGT source commit must remain {VGGT_SOURCE_COMMIT}"
            )

def _canonical_bytes(payload: Any, pretty: bool = False) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _read_regular_bytes(path: str | Path, label: str) -> bytes:
    file_path = Path(path)
    _regular_file(file_path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(file_path, flags)
    except OSError as exc:
        raise ActionWindowGeometryError(f"cannot open {label}: {file_path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ActionWindowGeometryError(f"{label} changed type: {file_path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 4 * 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise ActionWindowGeometryError(f"{label} changed while reading: {file_path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(_read_regular_bytes(path, "hashed file")).hexdigest()


def _input_fingerprint(path: str | Path, label: str) -> dict[str, Any]:
    file_path = Path(path).absolute()
    payload = _read_regular_bytes(file_path, label)
    metadata = os.stat(file_path, follow_symlinks=False)
    return {
        "path": str(file_path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "size": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
        "ctime_ns": int(metadata.st_ctime_ns),
    }


def _assert_inputs_unchanged(fingerprints: Mapping[str, Mapping[str, Any]]) -> None:
    for label, expected in fingerprints.items():
        current = _input_fingerprint(str(expected["path"]), label)
        if current != expected:
            raise ActionWindowGeometryError(f"{label} changed during geometry build")


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(f"{array.dtype}|{array.shape}|".encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(_read_regular_bytes(path, label).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ActionWindowGeometryError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ActionWindowGeometryError(f"{label} must be a JSON mapping")
    return payload


def _regular_file(path: Path, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as exc:
        raise ActionWindowGeometryError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ActionWindowGeometryError(f"{label} must be a regular non-symlink file: {path}")


def _manifest_records(path: Path) -> dict[str, Any]:
    try:
        lines = _read_regular_bytes(path, "source manifest").decode("utf-8").splitlines()
        known = set(OXEClipRecord.__dataclass_fields__)
        records = [
            OXEClipRecord(**{key: value for key, value in json.loads(line).items() if key in known})
            for raw in lines
            if (line := raw.strip())
        ]
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ActionWindowGeometryError(f"invalid source manifest: {path}") from exc
    by_id: dict[str, Any] = {}
    safe_ids: dict[str, str] = {}
    for record in records:
        if record.clip_id in by_id:
            raise ActionWindowGeometryError(f"duplicate manifest clip_id: {record.clip_id}")
        safe = safe_clip_id(record.clip_id)
        previous = safe_ids.get(safe)
        if previous is not None:
            raise ActionWindowGeometryError(f"safe-id collision: {previous!r} and {record.clip_id!r} -> {safe!r}")
        by_id[record.clip_id] = record
        safe_ids[safe] = record.clip_id
    return by_id


def _validate_temporal_spec(value: Any) -> None:
    expected = {
        "schema_version": SPLIT_TEMPORAL_BINDING_SCHEMA,
        "binding_schema_version": TEMPORAL_WINDOW_BINDING_SCHEMA,
        "candidate_offsets": list(EVIDENCE_CANDIDATE_OFFSETS),
        "context_length": EVIDENCE_CONTEXT_LENGTH,
        "start_derivation": TEMPORAL_WINDOW_DERIVATION,
        "resolver": {"path": "wm3d_v3/stage1/action_contract.py", "symbol": "resolve_action_window"},
    }
    if value != expected:
        raise ActionWindowGeometryError("split temporal_window_binding mismatch")


def _derive_group(
    split: Any, records: Mapping[str, Any], cache_root: Path,
    droid_sources: Mapping[str, Mapping[str, Any]],
    droid_binding: tuple[str | None, str | None, str | None],
) -> list[ActionWindowCandidate]:
    result: list[ActionWindowCandidate] = []
    droid_index_path,droid_index_sha,droid_root=droid_binding
    for role in ROLES:
        for clip_id in getattr(split, f"{role}_clip_ids"):
            record = records.get(clip_id)
            if record is None:
                raise ActionWindowGeometryError(f"split clip absent from manifest: {clip_id}")
            if action_contract_key(record) != split.contract_key:
                raise ActionWindowGeometryError(f"manifest contract mismatch for {clip_id}")
            safe = safe_clip_id(clip_id)
            rgb_path = cache_root / "rgb_256" / f"{safe}.npy"
            if (
                canonical_dataset_name(record.dataset) == "droid"
                and clip_id in droid_sources
            ):
                source=droid_sources[clip_id]["actions"]
                action_path=Path(str(source["path"])).absolute()
                index_path,index_sha,index_root=droid_index_path,droid_index_sha,droid_root
            elif canonical_dataset_name(record.dataset) == "droid":
                raise ActionWindowGeometryError(f"DROID finalized cache source missing: {clip_id}")
            else:
                action_path=cache_root/"actions"/f"{safe}.npy"
                source=None
                index_path=index_sha=index_root=None
            _regular_file(rgb_path, "RGB cache"); _regular_file(action_path, "action cache")
            try:
                rgb = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
                actions = np.load(action_path, mmap_mode="r", allow_pickle=False)
            except Exception as exc:
                raise ActionWindowGeometryError(f"invalid RGB/action cache for {clip_id}") from exc
            action_sha=sha256_file(action_path)
            if source is not None and action_sha!=str(source["sha256"]):
                raise ActionWindowGeometryError(f"DROID action hash mismatch: {clip_id}")
            group_id = split.clip_to_group_id[clip_id]
            binding = bind_temporal_window(
                record, group_id=group_id, seed=SPLIT_SEED,
                usable_frames=min(int(record.n_frames), int(rgb.shape[0])),
                n_action_frames=int(actions.shape[0]), target_length=TARGET_LENGTH,
            )
            result.append(ActionWindowCandidate(
                split.contract_key, role, clip_id, int(binding.start), group_id,
                binding.legal_starts_sha256, binding.selection_sha256,
                str(action_path),action_sha,tuple(int(v) for v in actions.shape),
                str(actions.dtype),index_path,index_sha,index_root,
            ))
            del rgb,actions
    return result


def load_frozen_temporal_candidates(
    path: str | Path, *, manifest_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    droid_cache_index: str | Path | None = None,
) -> tuple[ActionWindowCandidate, ...]:
    candidate_path = Path(path).absolute()
    payload = _read_json(candidate_path, "frozen split temporal candidate")
    if payload.get("schema_version") != SPLIT_SCHEMA:
        raise ActionWindowGeometryError(f"unexpected split schema: {payload.get('schema_version')!r}")
    if payload.get("immutable") is not True or payload.get("seed") != SPLIT_SEED:
        raise ActionWindowGeometryError("split must be immutable with seed 1729")
    if payload.get("derivation") != SPLIT_DERIVATION:
        raise ActionWindowGeometryError("split derivation mismatch")
    _validate_temporal_spec(payload.get("temporal_window_binding"))
    if manifest_path is None or cache_root is None:
        raise ActionWindowGeometryError("manifest_path and cache_root are required")
    manifest = Path(manifest_path).absolute()
    source_manifest = payload.get("source_manifest")
    if not isinstance(source_manifest, dict) or set(source_manifest) != {"path", "sha256"}:
        raise ActionWindowGeometryError("split source_manifest binding is invalid")
    if Path(str(source_manifest["path"])).resolve() != manifest:
        raise ActionWindowGeometryError("split source_manifest path mismatch")
    if source_manifest["sha256"] != sha256_file(manifest):
        raise ActionWindowGeometryError("split source_manifest hash mismatch")
    records = _manifest_records(manifest)
    groups = payload.get("groups")
    if not isinstance(groups, dict) or set(groups) != EXPECTED_CONTRACT_KEYS:
        raise ActionWindowGeometryError("split must contain exactly the six fixed contracts")
    if droid_cache_index is None:
        raise ActionWindowGeometryError("formal six-domain geometry requires droid_cache_index")
    droid_sources: Mapping[str, Mapping[str, Any]] = {}
    droid_binding: tuple[str | None, str | None, str | None] = (
        None,
        None,
        None,
    )
    if droid_cache_index is not None:
        droid_index_path = Path(droid_cache_index).absolute()
        droid_index = _read_json(droid_index_path, "DROID finalized cache index")
        droid_root = Path(str(droid_index.get("cache_root", ""))).absolute()
        droid_split = frozen_contract_split_from_mapping(
            groups[FORMAL_DROID_CONTRACT_KEY]
        )
        droid_records = [
            records[clip_id]
            for role in ROLES
            for clip_id in getattr(droid_split, f"{role}_clip_ids")
        ]
        try:
            droid_sources = validate_formal_droid_cache_index(
                droid_records,
                cache_root=droid_root,
                index_path=droid_index_path,
            )
        except ActionCacheResolutionError as exc:
            raise ActionWindowGeometryError(str(exc)) from exc
        droid_binding = (
            str(droid_index_path),
            sha256_file(droid_index_path),
            str(droid_root),
        )
    result: list[ActionWindowCandidate] = []
    try:
        for key in sorted(groups):
            raw_group = groups[key]
            if not isinstance(raw_group, dict):
                raise ActionWindowGeometryError(f"invalid split group: {key}")
            split = frozen_contract_split_from_mapping(raw_group)
            if split.contract_key != key:
                raise ActionWindowGeometryError(f"split contract key mismatch: {key}")
            result.extend(
                _derive_group(
                    split,
                    records,
                    Path(cache_root).resolve(),
                    droid_sources,
                    droid_binding,
                )
            )
    except ActionContractSplitError as exc:
        raise ActionWindowGeometryError(str(exc)) from exc
    identities = [(x.contract_key, x.role, x.clip_id, x.start) for x in result]
    if len(result) != EXPECTED_WINDOWS or len(set(identities)) != EXPECTED_WINDOWS:
        raise ActionWindowGeometryError(f"split must derive exactly {EXPECTED_WINDOWS} unique windows, got {len(result)}")
    return tuple(result)

def _deterministic_npz(payload: Mapping[str, np.ndarray]) -> bytes:
    destination = BytesIO()
    with zipfile.ZipFile(
        destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for key in sorted(payload):
            stream = BytesIO()
            np.lib.format.write_array(
                stream, np.asarray(payload[key]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, stream.getvalue(), compresslevel=6)
    return destination.getvalue()


def _output_name(item: ActionWindowCandidate) -> str:
    digest = hashlib.sha256(item.clip_id.encode()).hexdigest()[:20]
    return f"{digest}__start_{item.start:08d}.npz"


def _source_window(item: ActionWindowCandidate, root: Path, hashes: dict[Path, str]):
    path = root / "rgb_256" / f"{safe_clip_id(item.clip_id)}.npy"
    _regular_file(path, "source RGB")
    try:
        rgb = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise ActionWindowGeometryError(f"invalid source RGB: {path}") from exc
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ActionWindowGeometryError(f"source RGB must be uint8 [N,H,W,3]: {path}")
    if item.start + WINDOW_LENGTH > rgb.shape[0]:
        raise ActionWindowGeometryError(
            f"window exceeds source RGB: {item.clip_id} start={item.start}"
        )
    window = np.ascontiguousarray(rgb[item.start : item.start + WINDOW_LENGTH])
    return window, {
        "path": str(path.resolve()),
        "sha256": hashes.setdefault(path, sha256_file(path)),
        "window_sha256": _array_sha256(window),
        "shape": list(rgb.shape),
        "dtype": str(rgb.dtype),
    }


def _to_input(windows: Sequence[np.ndarray], size: int) -> torch.Tensor:
    tensor = (
        torch.from_numpy(np.stack(windows)).permute(0, 1, 4, 2, 3).float().div_(255)
    )
    batch, frames = tensor.shape[:2]
    flat = tensor.reshape(batch * frames, 3, *tensor.shape[-2:])
    if flat.shape[-2:] != (size, size):
        flat = F.interpolate(
            flat, (size, size), mode="bilinear", align_corners=False, antialias=True
        )
    return flat.reshape(batch, frames, 3, size, size)


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def _geometry_batch(value: Any, key: str, batch: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 5 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 4 or array.shape[:2] != (batch, WINDOW_LENGTH):
        raise ActionWindowGeometryError(
            f"VGGT {key} must be [B,17,H,W], got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ActionWindowGeometryError(f"VGGT {key} contains non-finite values")
    return array.astype(np.float16, copy=False)


def _infer(encoder: Any, pending: Sequence[tuple], size: int) -> list[tuple]:
    if len(pending)!=1:
        raise ActionWindowGeometryError(
            "VGGT formal inference accepts exactly one independent 17-frame window"
        )
    input_tensor=None; output=None
    try:
        input_tensor=_to_input([pending[0][1]],size)
        output=encoder(input_tensor)
        if not isinstance(output,Mapping) or not {"depth","depth_conf"}.issubset(output):
            raise ActionWindowGeometryError("VGGT forward did not return depth and depth_conf")
        depth=_geometry_batch(output["depth"],"depth",1)
        confidence=_geometry_batch(output["depth_conf"],"depth_conf",1)
        if depth.shape!=confidence.shape:
            raise ActionWindowGeometryError("VGGT depth/depth_conf shape mismatch")
        item=pending[0]
        return [(item[0],depth[0],confidence[0],item[2])]
    except BaseException as exc:
        if not _is_oom(exc):
            raise
        item=pending[0][0]
        del output,input_tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise ActionWindowGeometryError(
            f"VGGT OOM for one 17-frame window: {item.clip_id} start={item.start}: {exc}"
        ) from exc


def _state_content_sha256(model: Any, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    if model is None or not hasattr(model, "state_dict"):
        return digest.hexdigest()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().contiguous()
        digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}|".encode("utf-8"))
        raw = tensor.view(torch.uint8).reshape(-1)
        for start in range(0, int(raw.numel()), int(chunk_bytes)):
            chunk = raw[start : start + int(chunk_bytes)].cpu()
            digest.update(memoryview(chunk.numpy()).cast("B"))
    return digest.hexdigest()


def _model_identity(encoder: Any, config: GeometryConfig) -> dict[str, Any]:
    model = getattr(encoder, "model", None)
    model_config = getattr(model, "config", None)
    raw_config = model_config.to_dict() if model_config is not None and hasattr(model_config, "to_dict") else None
    state_manifest = [] if model is None or not hasattr(model, "state_dict") else [
        (name, list(value.shape), str(value.dtype)) for name, value in sorted(model.state_dict().items())
    ]
    name_or_path = getattr(model_config, "_name_or_path", None)
    requested_revision = getattr(encoder, "model_revision", None)
    revision = getattr(encoder, "model_resolved_revision", None)
    snapshot_path = Path(str(getattr(encoder, "model_snapshot_path", "")))
    source_root = Path(str(getattr(encoder, "vggt_source_root", "")))
    source_file = Path(str(getattr(encoder, "vggt_source_file", "")))
    if requested_revision != config.model_revision:
        raise ActionWindowGeometryError("VGGT requested revision mismatch")
    if revision != config.model_revision:
        raise ActionWindowGeometryError(
            f"VGGT resolved revision mismatch: {revision!r} != {config.model_revision!r}"
        )
    if not snapshot_path.is_dir() or snapshot_path.name != config.model_revision:
        raise ActionWindowGeometryError("VGGT resolved snapshot path mismatch")
    if source_root != Path(VGGT_SOURCE_ROOT) or not source_root.is_dir():
        raise ActionWindowGeometryError("VGGT executed source root mismatch")
    try:
        source_file.resolve(strict=True).relative_to(source_root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ActionWindowGeometryError("VGGT executed source file mismatch") from exc
    encoder_model_name = str(getattr(encoder, "model_name", config.model_name))
    if encoder_model_name != config.model_name:
        raise ActionWindowGeometryError(
            f"VGGT encoder model mismatch: {encoder_model_name!r} != {config.model_name!r}"
        )
    execution_device = torch.device(getattr(encoder, "device", "cpu"))
    if execution_device.type != "cuda" or not torch.cuda.is_available():
        raise ActionWindowGeometryError("formal VGGT execution requires a CUDA device")
    device_index = (
        torch.cuda.current_device()
        if execution_device.index is None else int(execution_device.index)
    )
    properties = torch.cuda.get_device_properties(device_index)
    capability = (int(properties.major), int(properties.minor))
    if capability != (9, 0) or "H100" not in str(properties.name):
        raise ActionWindowGeometryError(
            f"formal VGGT execution device mismatch: {properties.name} {capability}"
        )
    if raw_config is None:
        raw_config = {
            "model_name": config.model_name,
            "resolved_revision": revision,
            "resolved_snapshot_path": str(snapshot_path),
        }
    if name_or_path is None:
        name_or_path = str(snapshot_path)
    execution_identity = {
        "requested": str(execution_device),
        "visible_index": device_index,
        "uuid": str(properties.uuid),
        "name": str(properties.name),
        "capability": list(capability),
        "total_memory": int(properties.total_memory),
    }
    content_sha = _state_content_sha256(model)
    parameter_count = None if model is None or not hasattr(model, "parameters") else sum(int(v.numel()) for v in model.parameters())
    buffer_count = None if model is None or not hasattr(model, "buffers") else sum(int(v.numel()) for v in model.buffers())
    return {
        "requested_model_name": config.model_name,
        "requested_revision": requested_revision,
        "resolved_revision": revision,
        "resolved_snapshot_path": str(snapshot_path),
        "executed_vggt_source_root": str(source_root),
        "executed_vggt_source_file": str(source_file),
        "execution_device": execution_identity,
        "encoder_class": f"{encoder.__class__.__module__}.{encoder.__class__.__qualname__}",
        "model_class": None if model is None else f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "model_name_or_path": name_or_path,
        "model_commit_hash": getattr(model_config, "_commit_hash", None),
        "model_config_sha256": None if raw_config is None else _payload_sha256(raw_config),
        "state_manifest_sha256": _payload_sha256(state_manifest),
        "state_content_sha256": content_sha,
        "parameter_content_sha256": content_sha,
        "parameter_count": parameter_count,
        "buffer_count": buffer_count,
    }

def _run_git(source_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(source_root), *args),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ActionWindowGeometryError(
            f"cannot resolve VGGT source provenance at {source_root}: {exc}"
        ) from exc
    return result.stdout.strip()


def _code_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    relatives = (
        "wm3d_v3/stage1/action_window_geometry.py",
        "wm3d_v3/encoders/vggt_encoder.py",
        "scripts/build_stage1_action_window_geometry.py",
        "wm3d_v3/data/manifest.py",
        "wm3d_v3/stage1/action_contract.py",
        "wm3d_v3/stage1/action_contract_evidence.py",
        "wm3d_v3/stage1/action_contract_split.py",
        "wm3d_v3/stage1/action_evidence_sources.py",
        "wm3d_v3/stage1/action_cache.py",
        "wm3d_v3/stage1/droid_interval_action.py",
        "wm3d_v3/stage1/immutable_artifact.py",
    )
    files = [{"path": name, "sha256": sha256_file(root / name)} for name in relatives]
    source_root = Path(VGGT_SOURCE_ROOT)
    try:
        source_metadata = os.lstat(source_root)
    except FileNotFoundError as exc:
        raise ActionWindowGeometryError(f"missing VGGT source root: {source_root}") from exc
    if stat.S_ISLNK(source_metadata.st_mode) or not stat.S_ISDIR(source_metadata.st_mode):
        raise ActionWindowGeometryError(
            f"VGGT source root must be a non-symlink directory: {source_root}"
        )
    source_head = _run_git(source_root, "rev-parse", "HEAD")
    if source_head != VGGT_SOURCE_COMMIT:
        raise ActionWindowGeometryError(
            f"VGGT source commit mismatch: {source_head!r} != {VGGT_SOURCE_COMMIT!r}"
        )
    source_status = _run_git(
        source_root, "status", "--porcelain", "--untracked-files=all"
    )
    if source_status:
        raise ActionWindowGeometryError("VGGT source tree must be clean")
    tracked = tuple(name for name in _run_git(source_root, "ls-files").splitlines() if name)
    if not tracked:
        raise ActionWindowGeometryError("VGGT source tree has no tracked files")
    source_files: list[dict[str, str]] = []
    for name in tracked:
        source_path = source_root / name
        _regular_file(source_path, "tracked VGGT source")
        source_files.append({"path": name, "sha256": sha256_file(source_path)})
    external_vggt = {
        "root": str(source_root),
        "commit": source_head,
        "clean": True,
        "files": source_files,
        "tree_sha256": _payload_sha256(source_files),
    }
    import transformers
    driver_result = subprocess.run(
        ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        check=True, capture_output=True, text=True,
    )
    driver_versions = sorted(set(driver_result.stdout.split()))
    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "torch_cudnn": str(torch.backends.cudnn.version()),
        "numpy": str(np.__version__),
        "transformers": str(transformers.__version__),
        "nvidia_driver": ",".join(driver_versions),
    }
    if runtime != FORMAL_RUNTIME:
        raise ActionWindowGeometryError(f"formal VGGT runtime mismatch: {runtime}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise ActionWindowGeometryError("formal VGGT runtime requires a visible CUDA GPU")
    devices=[]
    for index in range(torch.cuda.device_count()):
        capability=tuple(int(value) for value in torch.cuda.get_device_capability(index))
        name=str(torch.cuda.get_device_name(index))
        if capability!=(9,0) or "H100" not in name:
            raise ActionWindowGeometryError(f"formal VGGT device mismatch: {name} {capability}")
        devices.append({"index":index,"name":name,"capability":list(capability),
                        "total_memory":int(torch.cuda.get_device_properties(index).total_memory)})
    runtime_flags={
        "deterministic_algorithms":bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark":bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic":bool(torch.backends.cudnn.deterministic),
    }
    identity = {"files": files, "external_vggt": external_vggt, "runtime": runtime,
                "visible_devices":devices,"runtime_flags":runtime_flags}
    return {**identity, "tree_sha256": _payload_sha256(identity)}

def _validate_npz(path: Path, entry: Mapping[str, Any]) -> None:
    _regular_file(path,"indexed geometry output")
    output=entry.get("output")
    if not isinstance(output,dict) or sha256_file(path)!=output.get("sha256"):
        raise ActionWindowGeometryError(f"geometry output hash conflict: {path}")
    required={"depth","depth_conf","start","predecessor_index","target_indices","frame_indices",
              "source_rgb_sha256","source_rgb_window_sha256","source_action_sha256",
              "model_identity_sha256","config_sha256","code_sha256"}
    try:
        with np.load(path,allow_pickle=False) as payload:
            if set(payload.files)!=required:
                raise ActionWindowGeometryError(f"geometry NPZ key conflict: {path}")
            arrays={key:np.asarray(payload[key]).copy() for key in payload.files}
    except ActionWindowGeometryError:
        raise
    except Exception as exc:
        raise ActionWindowGeometryError(f"invalid geometry NPZ: {path}") from exc
    depth,confidence=arrays["depth"],arrays["depth_conf"]
    if (depth.dtype!=np.float16 or confidence.dtype!=np.float16 or depth.ndim!=3
            or depth.shape[0]!=17 or confidence.shape!=depth.shape
            or not np.isfinite(depth).all() or not np.isfinite(confidence).all()):
        raise ActionWindowGeometryError(f"geometry NPZ shape/dtype/finite conflict: {path}")
    if output.get("shape")!=list(depth.shape) or output.get("dtype")!="float16":
        raise ActionWindowGeometryError(f"geometry output metadata conflict: {path}")
    for key,expected in {"start":entry["start"],"predecessor_index":entry["predecessor_index"]}.items():
        value=arrays[key]
        if value.shape!=() or value.dtype!=np.int64 or int(value)!=int(expected):
            raise ActionWindowGeometryError(f"geometry NPZ integer scalar conflict: {key}")
    for key,expected in {"target_indices":entry["target_indices"],"frame_indices":entry["frame_indices"]}.items():
        value=arrays[key]
        if value.dtype!=np.int64 or value.ndim!=1 or value.tolist()!=expected:
            raise ActionWindowGeometryError(f"geometry NPZ index conflict: {key}")
    if entry["predecessor_index"]!=entry["start"] or entry["frame_indices"]!=list(range(entry["start"],entry["start"]+17)):
        raise ActionWindowGeometryError("geometry entry temporal identity conflict")
    checks={"source_rgb_sha256":entry["source_rgb"]["sha256"],
            "source_rgb_window_sha256":entry["source_rgb"]["window_sha256"],
            "source_action_sha256":entry["source_action"]["sha256"],
            "model_identity_sha256":entry["model_identity_sha256"],
            "config_sha256":entry["config_sha256"],"code_sha256":entry["code_sha256"]}
    for key,expected in checks.items():
        value=arrays[key]
        if (value.shape!=() or value.dtype.kind not in ("U","S") or str(value.item())!=str(expected)
                or re.fullmatch(r"[0-9a-f]{64}",str(value.item())) is None):
            raise ActionWindowGeometryError(f"geometry NPZ identity conflict: {key}")
    if (_array_sha256(depth)!=output.get("depth_sha256")
            or _array_sha256(confidence)!=output.get("depth_conf_sha256")):
        raise ActionWindowGeometryError(f"geometry array hash conflict: {path}")


def _identity(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    return (entry.get("contract_key"), entry.get("role"), entry.get("clip_id"), entry.get("start"))


def _validate_index(path: Path, expected: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    payload = _read_json(path, "geometry index")
    for key in (
        "schema_version", "artifact_type", "immutable", "candidate_artifact",
        "source_manifest", "droid_cache_index", "config", "config_sha256", "code", "code_sha256",
        "model_identity", "model_identity_sha256",
    ):
        if payload.get(key) != expected.get(key):
            raise ActionWindowGeometryError(f"geometry index conflict in {key}: {path}")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != EXPECTED_WINDOWS:
        raise ActionWindowGeometryError(f"geometry index entry count conflict: {path}")
    expected_entries = expected["entries"]
    expected_ids = [_identity(row) for row in expected_entries]
    actual_ids = [_identity(row) for row in entries]
    if len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != set(expected_ids):
        raise ActionWindowGeometryError("geometry index identity set conflict")
    outputs = [entry.get("output") for entry in entries]
    output_paths = [str(value.get("path")) for value in outputs if isinstance(value, dict)]
    if len(output_paths) != len(entries) or len(set(output_paths)) != len(output_paths):
        raise ActionWindowGeometryError("geometry index output paths are not unique")
    if payload.get("output_hashes_sha256") != _payload_sha256(outputs):
        raise ActionWindowGeometryError(f"geometry index output hash conflict: {path}")
    actual_paths = {item.name for item in output_root.glob("*.npz")}
    if actual_paths != set(output_paths):
        missing = sorted(set(output_paths) - actual_paths)
        unexpected = sorted(actual_paths - set(output_paths))
        raise ActionWindowGeometryError(f"geometry output inventory conflict; missing={missing} unexpected={unexpected}")
    by_id = {_identity(row): row for row in expected_entries}
    for entry in entries:
        identity = _identity(entry)
        expected_entry = by_id[identity]
        for key in (
            "group_id", "legal_starts_sha256", "selection_sha256",
            "predecessor_index", "target_indices", "frame_indices", "source_rgb", "source_action",
            "config_sha256", "code_sha256",
        ):
            if entry.get(key) != expected_entry.get(key):
                raise ActionWindowGeometryError(f"geometry index conflict in {key}: {identity}")
        if entry.get("model_identity_sha256") != expected["model_identity_sha256"]:
            raise ActionWindowGeometryError(f"geometry index model identity conflict: {identity}")
        relative = Path(str(entry.get("output", {}).get("path", "")))
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ActionWindowGeometryError(f"unsafe indexed output path: {relative}")
        _validate_npz(output_root / relative, entry)
    return {"status": "skipped", "window_count": len(entries), "index": str(path), "index_sha256": sha256_file(path)}


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _consumer_path(path: str | Path, label: str, *, directory: bool = False) -> Path:
    raw = Path(path)
    if ".." in raw.parts:
        raise ActionWindowGeometryError(f"{label} contains path traversal: {raw}")
    absolute = raw.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as exc:
            raise ActionWindowGeometryError(f"missing {label}: {absolute}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ActionWindowGeometryError(f"{label} path contains symlink: {current}")
    metadata = os.lstat(absolute)
    expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if not expected:
        kind = "directory" if directory else "regular file"
        raise ActionWindowGeometryError(f"{label} must be a non-symlink {kind}: {absolute}")
    return absolute


def _bound_path(value: Any, expected: Path, label: str) -> bool:
    if not isinstance(value, str):
        return False
    raw = Path(value)
    return raw.is_absolute() and ".." not in raw.parts and raw == expected


def _hash_field(value: Any) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _validate_geometry_provenance(payload: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if payload.get("config") != config or payload.get("config_sha256") != _payload_sha256(config):
        raise ActionWindowGeometryError("geometry index formal config/hash conflict")

    code = payload.get("code")
    if not isinstance(code, dict) or set(code) != {
        "files", "external_vggt", "runtime", "visible_devices", "runtime_flags", "tree_sha256",
    }:
        raise ActionWindowGeometryError("geometry index code identity schema conflict")
    tree = {key: value for key, value in code.items() if key != "tree_sha256"}
    if code["tree_sha256"] != _payload_sha256(tree) or payload.get("code_sha256") != code["tree_sha256"]:
        raise ActionWindowGeometryError("geometry index code hash conflict")
    files = code["files"]
    if not isinstance(files, list) or not files:
        raise ActionWindowGeometryError("geometry index code file inventory conflict")
    registered_files = {
        "wm3d_v3/stage1/action_window_geometry.py",
        "wm3d_v3/encoders/vggt_encoder.py",
        "scripts/build_stage1_action_window_geometry.py",
        "wm3d_v3/data/manifest.py",
        "wm3d_v3/stage1/action_contract.py",
        "wm3d_v3/stage1/action_contract_evidence.py",
        "wm3d_v3/stage1/action_contract_split.py",
        "wm3d_v3/stage1/action_evidence_sources.py",
        "wm3d_v3/stage1/action_cache.py",
        "wm3d_v3/stage1/droid_interval_action.py",
        "wm3d_v3/stage1/immutable_artifact.py",
    }
    if {row.get("path") for row in files if isinstance(row, dict)} != registered_files:
        raise ActionWindowGeometryError("geometry index registered code inventory conflict")
    seen: set[str] = set()
    for row in files:
        if (not isinstance(row, dict) or set(row) != {"path", "sha256"}
                or not isinstance(row["path"], str) or Path(row["path"]).is_absolute()
                or ".." in Path(row["path"]).parts or row["path"] in seen
                or not _hash_field(row["sha256"])):
            raise ActionWindowGeometryError("geometry index code file identity conflict")
        seen.add(row["path"])
    external = code["external_vggt"]
    if not isinstance(external, dict) or set(external) != {
        "root", "commit", "clean", "files", "tree_sha256",
    }:
        raise ActionWindowGeometryError("geometry index VGGT source schema conflict")
    source_files = external["files"]
    if (external["root"] != VGGT_SOURCE_ROOT or external["commit"] != VGGT_SOURCE_COMMIT
            or external["clean"] is not True or not isinstance(source_files, list)
            or external["tree_sha256"] != _payload_sha256(source_files)):
        raise ActionWindowGeometryError("geometry index registered VGGT source conflict")
    source_seen: set[str] = set()
    for row in source_files:
        if (not isinstance(row, dict) or set(row) != {"path", "sha256"}
                or not isinstance(row["path"], str) or Path(row["path"]).is_absolute()
                or ".." in Path(row["path"]).parts or row["path"] in source_seen
                or not _hash_field(row["sha256"])):
            raise ActionWindowGeometryError("geometry index VGGT source file conflict")
        source_seen.add(row["path"])
    if code["runtime"] != FORMAL_RUNTIME:
        raise ActionWindowGeometryError("geometry index formal runtime identity conflict")
    devices = code["visible_devices"]
    if (not isinstance(devices, list) or not devices
            or any(not isinstance(item, dict)
                   or item.get("capability") != [9, 0]
                   or "H100" not in str(item.get("name", ""))
                   for item in devices)):
        raise ActionWindowGeometryError("geometry index formal device identity conflict")
    flags = code["runtime_flags"]
    if (not isinstance(flags, dict)
            or set(flags) != {"deterministic_algorithms", "cudnn_benchmark", "cudnn_deterministic"}
            or any(not isinstance(value, bool) for value in flags.values())):
        raise ActionWindowGeometryError("geometry index runtime flags conflict")

    identity = payload.get("model_identity")
    required = {
        "requested_model_name", "requested_revision", "resolved_revision",
        "resolved_snapshot_path", "executed_vggt_source_root", "executed_vggt_source_file",
        "execution_device",
        "encoder_class", "model_class", "model_name_or_path", "model_commit_hash",
        "model_config_sha256", "state_manifest_sha256", "state_content_sha256",
        "parameter_content_sha256", "parameter_count", "buffer_count",
    }
    if not isinstance(identity, dict) or set(identity) != required:
        raise ActionWindowGeometryError("geometry index model identity schema conflict")
    snapshot = Path(str(identity["resolved_snapshot_path"]))
    source_file = Path(str(identity["executed_vggt_source_file"]))
    try:
        source_file.relative_to(Path(VGGT_SOURCE_ROOT))
    except ValueError as exc:
        raise ActionWindowGeometryError("geometry index registered model source conflict") from exc
    if (identity["requested_model_name"] != config["model_name"]
            or identity["requested_revision"] != VGGT_MODEL_REVISION
            or identity["resolved_revision"] != VGGT_MODEL_REVISION
            or identity["model_commit_hash"] not in (None, VGGT_MODEL_REVISION)
            or not snapshot.is_absolute() or ".." in snapshot.parts
            or snapshot.name != VGGT_MODEL_REVISION
            or identity["executed_vggt_source_root"] != VGGT_SOURCE_ROOT
            or source_file != Path(VGGT_SOURCE_ROOT) / "vggt/models/vggt.py"
            or identity["model_name_or_path"] not in {config["model_name"], str(snapshot)}
            or not source_file.is_absolute() or ".." in source_file.parts):
        raise ActionWindowGeometryError("geometry index registered model identity conflict")
    execution = identity["execution_device"]
    if not isinstance(execution, dict) or set(execution) != {
        "requested", "visible_index", "uuid", "name", "capability", "total_memory",
    }:
        raise ActionWindowGeometryError("geometry index execution device schema conflict")
    requested = execution["requested"]
    visible_index = execution["visible_index"]
    if (not isinstance(requested, str)
            or re.fullmatch(r"cuda(?::[0-9]+)?", requested) is None
            or not isinstance(visible_index, int) or isinstance(visible_index, bool)
            or visible_index < 0
            or (":" in requested and int(requested.rsplit(":", 1)[1]) != visible_index)
            or not isinstance(execution["uuid"], str) or not execution["uuid"].strip()
            or not isinstance(execution["name"], str) or "H100" not in execution["name"]
            or execution["capability"] != [9, 0]
            or not isinstance(execution["total_memory"], int)
            or isinstance(execution["total_memory"], bool) or execution["total_memory"] <= 0):
        raise ActionWindowGeometryError("geometry index formal execution device conflict")
    for key in ("model_config_sha256", "state_manifest_sha256", "state_content_sha256",
                "parameter_content_sha256"):
        if not _hash_field(identity[key]):
            raise ActionWindowGeometryError(f"geometry index model hash conflict: {key}")
    if identity["parameter_content_sha256"] != identity["state_content_sha256"]:
        raise ActionWindowGeometryError("geometry index model content hash conflict")
    for key in ("parameter_count", "buffer_count"):
        value = identity[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ActionWindowGeometryError(f"geometry index model count conflict: {key}")
    model_sha = _payload_sha256(identity)
    if payload.get("model_identity_sha256") != model_sha:
        raise ActionWindowGeometryError("geometry index model identity hash conflict")


def load_validated_geometry_index(
    index_path: str | Path, *, split_temporal_candidate: str | Path,
    manifest_path: str | Path, cache_root: str | Path,
    droid_cache_index: str | Path,
) -> ValidatedGeometryIndex:
    index = _consumer_path(index_path, "geometry index")
    split = _consumer_path(split_temporal_candidate, "split temporal candidate")
    manifest = _consumer_path(manifest_path, "source manifest")
    root = _consumer_path(cache_root, "common cache root", directory=True)
    droid = _consumer_path(droid_cache_index, "DROID finalized cache index")
    output_root = _consumer_path(index.parent, "geometry output root", directory=True)
    fingerprints = {
        "geometry index": _input_fingerprint(index, "geometry index"),
        "split temporal candidate": _input_fingerprint(split, "split temporal candidate"),
        "source manifest": _input_fingerprint(manifest, "source manifest"),
        "DROID finalized cache index": _input_fingerprint(droid, "DROID finalized cache index"),
    }
    candidates = load_frozen_temporal_candidates(
        split, manifest_path=manifest, cache_root=root, droid_cache_index=droid,
    )
    payload = _read_json(index, "geometry index")
    required_top = {
        "schema_version", "artifact_type", "immutable", "candidate_artifact",
        "source_manifest", "droid_cache_index", "config", "config_sha256",
        "code", "code_sha256", "model_identity", "model_identity_sha256",
        "entries", "output_hashes_sha256",
    }
    if set(payload) != required_top or payload.get("schema_version") != SCHEMA_VERSION:
        raise ActionWindowGeometryError("geometry index schema conflict")
    if payload.get("artifact_type") != ARTIFACT_TYPE or payload.get("immutable") is not True:
        raise ActionWindowGeometryError("geometry index must be the immutable formal artifact")
    split_binding = payload.get("candidate_artifact")
    manifest_binding = payload.get("source_manifest")
    droid_binding = payload.get("droid_cache_index")
    candidate_droid = {
        (item.droid_cache_index_path, item.droid_cache_index_sha256, item.droid_cache_root)
        for item in candidates if item.droid_cache_index_path is not None
    }
    if len(candidate_droid) != 1:
        raise ActionWindowGeometryError("geometry DROID binding is not unique")
    droid_path, droid_sha, droid_root = next(iter(candidate_droid))
    if (not isinstance(split_binding, dict) or set(split_binding) != {"path", "sha256"}
            or not _bound_path(split_binding.get("path"), split, "candidate artifact")
            or split_binding.get("sha256") != fingerprints["split temporal candidate"]["sha256"]):
        raise ActionWindowGeometryError("geometry index candidate path/hash conflict")
    if (not isinstance(manifest_binding, dict) or set(manifest_binding) != {"path", "sha256"}
            or not _bound_path(manifest_binding.get("path"), manifest, "source manifest")
            or manifest_binding.get("sha256") != fingerprints["source manifest"]["sha256"]):
        raise ActionWindowGeometryError("geometry index manifest path/hash conflict")
    if (not isinstance(droid_binding, dict) or set(droid_binding) != {"path", "sha256", "cache_root"}
            or not _bound_path(droid_binding.get("path"), droid, "DROID index")
            or droid_binding.get("sha256") != fingerprints["DROID finalized cache index"]["sha256"]
            or droid_binding != {"path": droid_path, "sha256": droid_sha, "cache_root": droid_root}):
        raise ActionWindowGeometryError("geometry index DROID path/hash binding conflict")
    formal_config = asdict(GeometryConfig())
    formal_config["outputs"] = list(GeometryConfig().outputs)
    formal_config["batch_windows"] = 1
    GeometryConfig().validate()
    _validate_geometry_provenance(payload, formal_config)

    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != EXPECTED_WINDOWS:
        raise ActionWindowGeometryError("geometry index entry count conflict")
    actual_ids = [_identity(entry) if isinstance(entry, dict) else (None, None, None, None) for entry in entries]
    expected_ids = [(item.contract_key, item.role, item.clip_id, item.start) for item in candidates]
    if len(set(actual_ids)) != EXPECTED_WINDOWS or set(actual_ids) != set(expected_ids):
        raise ActionWindowGeometryError("geometry index identity set conflict")
    by_id = {identity: entry for identity, entry in zip(actual_ids, entries, strict=True)}
    hashes: dict[Path, str] = {}
    output_names: set[str] = set()
    validated: dict[GeometryIdentity, Mapping[str, Any]] = {}
    for item in candidates:
        identity = (item.contract_key, item.role, item.clip_id, item.start)
        entry = by_id[identity]
        _, source_rgb = _source_window(item, root, hashes)
        expected = {
            "contract_key": item.contract_key, "role": item.role, "clip_id": item.clip_id,
            "start": item.start, "group_id": item.group_id,
            "legal_starts_sha256": item.legal_starts_sha256,
            "selection_sha256": item.selection_sha256,
            "predecessor_index": item.start, "target_indices": list(item.target_indices),
            "frame_indices": list(item.frame_indices), "source_rgb": source_rgb,
            "source_action": {"path": item.action_path, "sha256": item.action_sha256,
                              "shape": list(item.action_shape), "dtype": item.action_dtype},
            "config_sha256": payload["config_sha256"], "code_sha256": payload["code_sha256"],
            "model_identity_sha256": payload["model_identity_sha256"],
        }
        if set(entry) != {*expected, "output"} or any(entry.get(key) != value for key, value in expected.items()):
            raise ActionWindowGeometryError(f"geometry index entry binding conflict: {identity}")
        output = entry.get("output")
        name = _output_name(item)
        if (not isinstance(output, dict)
                or set(output) != {"path", "sha256", "depth_sha256", "depth_conf_sha256", "shape", "dtype"}
                or output.get("path") != name or name in output_names):
            raise ActionWindowGeometryError(f"geometry index output identity conflict: {identity}")
        relative = Path(name)
        if relative.is_absolute() or len(relative.parts) != 1 or ".." in relative.parts:
            raise ActionWindowGeometryError(f"unsafe indexed geometry output path: {name}")
        output_names.add(name)
        _validate_npz(output_root / relative, entry)
        validated[identity] = _deep_freeze(entry)
    actual_outputs = {path.name for path in output_root.iterdir() if path.name.endswith(".npz")}
    if actual_outputs != output_names:
        raise ActionWindowGeometryError("geometry output inventory conflict")
    if payload.get("output_hashes_sha256") != _payload_sha256([entry["output"] for entry in entries]):
        raise ActionWindowGeometryError("geometry index output hash inventory conflict")
    _assert_inputs_unchanged(fingerprints)
    metadata = _deep_freeze({key: value for key, value in payload.items() if key != "entries"})
    return ValidatedGeometryIndex(
        index_path=str(index), index_sha256=fingerprints["geometry index"]["sha256"],
        metadata=metadata, entries=MappingProxyType(validated),
    )


@contextmanager
def _output_lock(output_root: Path):
    try:
        metadata = os.lstat(output_root)
    except FileNotFoundError:
        output_root.mkdir(parents=True)
    else:
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ActionWindowGeometryError(f"output_root must be a non-symlink directory: {output_root}")
    lock_path = output_root / ".action_window_geometry.lock"
    if lock_path.is_symlink():
        raise ActionWindowGeometryError(f"lock destination is a symlink: {lock_path}")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise ActionWindowGeometryError(f"cannot open geometry lock: {lock_path}") from exc
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _output_metadata(path: Path) -> dict[str, Any]:
    _regular_file(path, "geometry candidate")
    try:
        with np.load(path, allow_pickle=False) as payload:
            depth = np.asarray(payload["depth"])
            confidence = np.asarray(payload["depth_conf"])
    except Exception as exc:
        raise ActionWindowGeometryError(f"invalid resumable geometry candidate: {path}") from exc
    return {
        "path": path.name, "sha256": sha256_file(path),
        "depth_sha256": _array_sha256(depth),
        "depth_conf_sha256": _array_sha256(confidence),
        "shape": list(depth.shape), "dtype": str(depth.dtype),
    }


def build_action_window_geometry(
    *, split_temporal_candidate: str | Path, cache_root: str | Path,
    output_root: str | Path, encoder_factory: Callable[[], Any],
    config: GeometryConfig = GeometryConfig(), batch_windows: int = 1,
    manifest_path: str | Path | None = None,
    droid_cache_index: str | Path | None = None,
    index_name: str = "index.json",
) -> dict[str, Any]:
    config.validate()
    if int(batch_windows)!=1:
        raise ActionWindowGeometryError(
            "formal geometry requires one independent 17-frame window per forward"
        )
    if manifest_path is None:
        raise ActionWindowGeometryError("manifest_path is required")
    candidate_path=Path(split_temporal_candidate).absolute()
    manifest=Path(manifest_path).absolute()
    cache_root=Path(cache_root).absolute()
    output_root=Path(output_root).absolute()
    if len(Path(index_name).parts)!=1 or index_name in {"",".",".."}:
        raise ActionWindowGeometryError("index_name must be a plain filename")
    with _output_lock(output_root):
        if droid_cache_index is None:
            raise ActionWindowGeometryError("droid_cache_index is required")
        input_fingerprints={
            "split temporal candidate":_input_fingerprint(candidate_path,"split temporal candidate"),
            "source manifest":_input_fingerprint(manifest,"source manifest"),
            "DROID finalized cache index":_input_fingerprint(droid_cache_index,"DROID finalized cache index"),
        }
        candidates=load_frozen_temporal_candidates(
            candidate_path,manifest_path=manifest,cache_root=cache_root,
            droid_cache_index=droid_cache_index,
        )
        _assert_inputs_unchanged(input_fingerprints)
        code=_code_identity()
        config_payload=asdict(config); config_payload["outputs"]=list(config.outputs)
        config_payload["batch_windows"]=1
        config_sha,code_sha=_payload_sha256(config_payload),code["tree_sha256"]
        encoder=encoder_factory()
        model_identity=_model_identity(encoder,config)
        model_sha=_payload_sha256(model_identity)
        _assert_inputs_unchanged(input_fingerprints)
        droid_bindings={(item.droid_cache_index_path,item.droid_cache_index_sha256,item.droid_cache_root)
                        for item in candidates if item.droid_cache_index_path is not None}
        if len(droid_bindings)!=1:
            raise ActionWindowGeometryError("DROID finalized cache binding must be unique")
        droid_path,droid_sha,droid_root=next(iter(droid_bindings))
        expected={
            "schema_version":SCHEMA_VERSION,"artifact_type":ARTIFACT_TYPE,"immutable":True,
            "candidate_artifact":{"path":str(candidate_path),"sha256":sha256_file(candidate_path)},
            "source_manifest":{"path":str(manifest),"sha256":sha256_file(manifest)},
            "droid_cache_index":{"path":droid_path,"sha256":droid_sha,"cache_root":droid_root},
            "config":config_payload,"config_sha256":config_sha,
            "code":code,"code_sha256":code_sha,
            "model_identity":model_identity,"model_identity_sha256":model_sha,
            "entries":[],
        }
        index_path=output_root/index_name
        index_exists=index_path.exists() or index_path.is_symlink()
        output_names={_output_name(item) for item in candidates}
        if len(output_names)!=len(candidates):
            raise ActionWindowGeometryError("geometry output-name collision")
        unexpected=sorted(item.name for item in output_root.glob("*.npz") if item.name not in output_names)
        if unexpected:
            raise ActionWindowGeometryError(f"unexpected geometry candidates without index: {unexpected}")
        hashes={}; completed=[]
        for item in candidates:
            window,source=_source_window(item,cache_root,hashes)
            source_action={"path":item.action_path,"sha256":item.action_sha256,
                           "shape":list(item.action_shape),"dtype":item.action_dtype}
            base={
                "contract_key":item.contract_key,"role":item.role,"clip_id":item.clip_id,
                "start":item.start,"group_id":item.group_id,
                "legal_starts_sha256":item.legal_starts_sha256,
                "selection_sha256":item.selection_sha256,
                "predecessor_index":item.start,"target_indices":list(item.target_indices),
                "frame_indices":list(item.frame_indices),"source_rgb":source,
                "source_action":source_action,"config_sha256":config_sha,
                "code_sha256":code_sha,"model_identity_sha256":model_sha,
            }
            expected["entries"].append({k:v for k,v in base.items() if k!="model_identity_sha256"})
            path=output_root/_output_name(item)
            if index_exists:
                del window
                continue
            if path.exists() or path.is_symlink():
                entry={**base,"output":_output_metadata(path)}
                _validate_npz(path,entry); completed.append(entry); del window
                continue
            inferred=_infer(encoder,[(item,window,source)],config.input_size)
            del window
            inferred_item,depth,confidence,inferred_source=inferred[0]
            if inferred_item!=item or inferred_source!=source:
                raise ActionWindowGeometryError("single-window inference identity drift")
            arrays={
                "depth":depth,"depth_conf":confidence,
                "start":np.asarray(item.start,np.int64),
                "predecessor_index":np.asarray(item.start,np.int64),
                "target_indices":np.asarray(item.target_indices,np.int64),
                "frame_indices":np.asarray(item.frame_indices,np.int64),
                "source_rgb_sha256":np.asarray(source["sha256"]),
                "source_rgb_window_sha256":np.asarray(source["window_sha256"]),
                "source_action_sha256":np.asarray(item.action_sha256),
                "model_identity_sha256":np.asarray(model_sha),
                "config_sha256":np.asarray(config_sha),"code_sha256":np.asarray(code_sha),
            }
            try:
                publish_immutable_bytes(path,_deterministic_npz(arrays))
            except ImmutableArtifactConflict as exc:
                raise ActionWindowGeometryError(str(exc)) from exc
            entry={**base,"output":_output_metadata(path)}
            _validate_npz(path,entry); completed.append(entry)
        if index_exists:
            _assert_inputs_unchanged(input_fingerprints)
            return _validate_index(index_path,expected,output_root)
        if len(completed)!=EXPECTED_WINDOWS:
            raise ActionWindowGeometryError(
                f"geometry build completed {len(completed)} of {EXPECTED_WINDOWS} windows"
            )
        index_payload={
            **{k:v for k,v in expected.items() if k!="entries"},
            "entries":completed,
            "output_hashes_sha256":_payload_sha256([item["output"] for item in completed]),
        }
        _assert_inputs_unchanged(input_fingerprints)
        try:
            result=publish_immutable_bytes(index_path,_canonical_bytes(index_payload,pretty=True))
        except ImmutableArtifactConflict as exc:
            raise ActionWindowGeometryError(str(exc)) from exc
        return {"status":"built","window_count":len(completed),
                "index":str(index_path),"index_sha256":result.sha256}
