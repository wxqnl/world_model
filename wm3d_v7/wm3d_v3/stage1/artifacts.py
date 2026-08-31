from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any

import torch
from torch import nn


CANDIDATE_WEIGHT_FILES = ("wm.pt", "wan_control.pt", "wan_trainable.pt")
CANDIDATE_FILES = frozenset((*CANDIDATE_WEIGHT_FILES, "candidate.json"))
CANDIDATE_SCHEMA_VERSION = 1
RESUME_SCHEMA_VERSION = 1
_STEP_PATTERN = re.compile(r"^step_(\d{8})$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ArtifactError(RuntimeError):
    """Raised when a Stage1 artifact is partial, mutable, or misidentified."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _require_sha256(value: object, label: str) -> str:
    digest = str(value)
    if not _SHA256_PATTERN.fullmatch(digest):
        raise ArtifactError(f"{label} must be a lowercase SHA256 digest")
    return digest


def _absolute_without_resolving(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def reject_symlink_components(path: str | Path) -> None:
    candidate = _absolute_without_resolving(Path(path))
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current /= component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactError(f"artifact path contains a symlink: {current}")


def require_directory_no_symlink(path: str | Path) -> Path:
    candidate = Path(path)
    reject_symlink_components(candidate)
    try:
        metadata = os.lstat(candidate)
    except FileNotFoundError as exc:
        raise ArtifactError(f"artifact directory does not exist: {candidate}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ArtifactError(f"artifact path is not a directory: {candidate}")
    return candidate


def require_regular_file_no_symlink(path: str | Path) -> Path:
    candidate = Path(path)
    reject_symlink_components(candidate)
    try:
        metadata = os.lstat(candidate)
    except FileNotFoundError as exc:
        raise ArtifactError(f"artifact file does not exist: {candidate}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ArtifactError(f"artifact path is not a regular file: {candidate}")
    return candidate


def _open_regular_no_follow(path: Path) -> int:
    require_regular_file_no_symlink(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArtifactError(f"cannot safely open artifact file: {path}") from exc
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode):
        os.close(descriptor)
        raise ArtifactError(f"artifact changed type while opening: {path}")
    return descriptor


def read_bytes_no_follow(path: str | Path) -> bytes:
    candidate = Path(path)
    descriptor = _open_regular_no_follow(candidate)
    with os.fdopen(descriptor, "rb") as stream:
        return stream.read()


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    descriptor = _open_regular_no_follow(candidate)
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(path: str | Path) -> None:
    directory = require_directory_no_symlink(path)
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_tree(path: str | Path) -> None:
    root = require_directory_no_symlink(path)
    directories: list[Path] = []
    for current, child_directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories.append(current_path)
        for name in child_directories:
            child = current_path / name
            metadata = os.lstat(child)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ArtifactError(f"invalid directory in artifact tree: {child}")
        for name in files:
            file_path = require_regular_file_no_symlink(current_path / name)
            descriptor = os.open(file_path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for directory in reversed(directories):
        fsync_directory(directory)


def _remove_tree(path: Path) -> None:
    if not os.path.lexists(path):
        return
    for current, directories, files in os.walk(path, topdown=False):
        for name in files:
            try:
                os.chmod(Path(current) / name, 0o600, follow_symlinks=False)
            except (FileNotFoundError, NotImplementedError):
                pass
        for name in directories:
            child = Path(current) / name
            try:
                os.chmod(child, 0o700, follow_symlinks=False)
            except (FileNotFoundError, NotImplementedError):
                pass
    try:
        os.chmod(path, 0o700, follow_symlinks=False)
    except (FileNotFoundError, NotImplementedError):
        pass
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _legacy_rename_no_replace(source: str | Path, destination: str | Path) -> None:
    source_path = Path(source)
    destination_path = Path(destination)
    if os.path.lexists(destination_path):
        raise ArtifactError(f"artifact destination already exists: {destination_path}")

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source_path),
            -100,
            os.fsencode(destination_path),
            1,
        )
        if result == 0:
            return
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise ArtifactError(
                f"artifact destination already exists: {destination_path}"
            )
        if error not in {errno.ENOSYS, errno.EINVAL}:
            raise ArtifactError(
                f"atomic artifact rename failed: {source_path} -> {destination_path}"
            ) from OSError(error, os.strerror(error))

    if os.path.lexists(destination_path):
        raise ArtifactError(f"artifact destination already exists: {destination_path}")
    try:
        os.rename(source_path, destination_path)
    except FileExistsError as exc:
        raise ArtifactError(
            f"artifact destination already exists: {destination_path}"
        ) from exc


def _write_bytes_fsync(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def atomic_write_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(destination.parent)
    if os.path.lexists(destination) and not overwrite:
        raise ArtifactError(f"artifact destination already exists: {destination}")

    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp.",
        dir=destination.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(dict(payload)))
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            reject_symlink_components(destination)
            os.replace(temporary, destination)
        else:
            rename_no_replace(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


@dataclass(frozen=True)
class CandidateIdentity:
    update_id: int
    wm_sha256: str
    wan_control_sha256: str
    wan_trainable_sha256: str
    config_sha256: str
    contract_sha256: str
    unique_manifest_sha256: str
    source_digest: str
    stage0_parent_sha256: str

    def __post_init__(self) -> None:
        if int(self.update_id) <= 0:
            raise ArtifactError("candidate update_id must be positive")
        for field_name in (
            "wm_sha256",
            "wan_control_sha256",
            "wan_trainable_sha256",
            "config_sha256",
            "contract_sha256",
            "unique_manifest_sha256",
            "source_digest",
            "stage0_parent_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)

    @property
    def weight_sha256(self) -> dict[str, str]:
        return {
            "wm.pt": self.wm_sha256,
            "wan_control.pt": self.wan_control_sha256,
            "wan_trainable.pt": self.wan_trainable_sha256,
        }

    def as_tuple(self) -> tuple[object, ...]:
        return (
            self.update_id,
            self.wm_sha256,
            self.wan_control_sha256,
            self.wan_trainable_sha256,
            self.config_sha256,
            self.contract_sha256,
            self.unique_manifest_sha256,
            self.source_digest,
            self.stage0_parent_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_id": self.update_id,
            "weight_sha256": self.weight_sha256,
            "config_sha256": self.config_sha256,
            "contract_sha256": self.contract_sha256,
            "unique_manifest_sha256": self.unique_manifest_sha256,
            "source_digest": self.source_digest,
            "stage0_parent_sha256": self.stage0_parent_sha256,
        }

    @property
    def candidate_id(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CandidateIdentity:
        weights = payload.get("weight_sha256")
        if not isinstance(weights, Mapping) or set(weights) != set(
            CANDIDATE_WEIGHT_FILES
        ):
            raise ArtifactError("candidate identity has an invalid weight hash set")
        try:
            return cls(
                update_id=int(payload["update_id"]),
                wm_sha256=str(weights["wm.pt"]),
                wan_control_sha256=str(weights["wan_control.pt"]),
                wan_trainable_sha256=str(weights["wan_trainable.pt"]),
                config_sha256=str(payload["config_sha256"]),
                contract_sha256=str(payload["contract_sha256"]),
                unique_manifest_sha256=str(payload["unique_manifest_sha256"]),
                source_digest=str(payload["source_digest"]),
                stage0_parent_sha256=str(payload["stage0_parent_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError("candidate identity is incomplete") from exc


@dataclass(frozen=True)
class CandidateArtifact:
    path: Path
    identity: CandidateIdentity


@dataclass(frozen=True)
class StrictLoadReport:
    loaded_files: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not (self.missing_keys or self.unexpected_keys or self.shape_mismatches)


@dataclass(frozen=True)
class LoadedCandidate:
    path: Path
    identity: CandidateIdentity
    states: dict[str, Mapping[str, Any]]
    strict_load_report: StrictLoadReport | None = None


def _state_mapping(
    value: nn.Module | Mapping[str, Any], label: str
) -> Mapping[str, Any]:
    state = value.state_dict() if isinstance(value, nn.Module) else value
    if not isinstance(state, Mapping) or not state:
        raise ArtifactError(f"{label} state must be a nonempty mapping")
    if any(not isinstance(key, str) for key in state):
        raise ArtifactError(f"{label} state keys must be strings")
    return state


def _write_torch_state(path: Path, state: Mapping[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        torch.save(dict(state), stream)
        stream.flush()
        os.fsync(stream.fileno())


def _load_torch_state(path: Path, *, map_location: Any = "cpu") -> Mapping[str, Any]:
    descriptor = _open_regular_no_follow(path)
    try:
        with os.fdopen(descriptor, "rb") as stream:
            state = torch.load(
                stream,
                map_location=map_location,
                weights_only=True,
            )
    except Exception as exc:
        raise ArtifactError(
            f"strict weight reload failed for {path.name}: {exc}"
        ) from exc
    if not isinstance(state, Mapping) or not state:
        raise ArtifactError(f"strict weight reload produced invalid {path.name} state")
    if any(not isinstance(key, str) for key in state):
        raise ArtifactError(
            f"strict weight reload found non-string keys in {path.name}"
        )
    return state


def _read_candidate_manifest(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(read_bytes_no_follow(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("candidate.json is not valid canonical JSON") from exc
    if not isinstance(payload, Mapping):
        raise ArtifactError("candidate.json must be an object")
    if payload.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise ArtifactError("unsupported candidate schema version")
    if payload.get("artifact_type") != "wm3d_stage1_candidate":
        raise ArtifactError("candidate.json has the wrong artifact type")
    return payload


def _strict_reload_candidate(
    candidate_dir: str | Path,
    *,
    expected_identity: CandidateIdentity | None = None,
    modules: Mapping[str, nn.Module] | None = None,
    strict_loader: Any = None,
    map_location: Any = "cpu",
) -> LoadedCandidate:
    directory = require_directory_no_symlink(candidate_dir)
    actual_names = {entry.name for entry in os.scandir(directory)}
    if actual_names != CANDIDATE_FILES:
        missing = sorted(CANDIDATE_FILES - actual_names)
        extra = sorted(actual_names - CANDIDATE_FILES)
        raise ArtifactError(
            f"candidate must contain the strict three-file set; "
            f"missing={missing}, extra={extra}"
        )

    manifest = _read_candidate_manifest(directory / "candidate.json")
    identity_payload = manifest.get("identity")
    if not isinstance(identity_payload, Mapping):
        raise ArtifactError("candidate.json has no identity object")
    identity = CandidateIdentity.from_dict(identity_payload)
    if manifest.get("candidate_id") != identity.candidate_id:
        raise ArtifactError("candidate_id does not match the identity tuple")
    if expected_identity is not None and identity != expected_identity:
        raise ArtifactError("candidate identity does not match the expected identity")

    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(CANDIDATE_WEIGHT_FILES):
        raise ArtifactError("candidate.json has an invalid three-file inventory")

    states: dict[str, Mapping[str, Any]] = {}
    for filename in CANDIDATE_WEIGHT_FILES:
        entry = files[filename]
        if not isinstance(entry, Mapping):
            raise ArtifactError(f"invalid candidate inventory entry: {filename}")
        expected_hash = _require_sha256(entry.get("sha256"), f"{filename} SHA256")
        if expected_hash != identity.weight_sha256[filename]:
            raise ArtifactError(f"{filename} hash disagrees with candidate identity")
        file_path = require_regular_file_no_symlink(directory / filename)
        actual_hash = sha256_file(file_path)
        if actual_hash != expected_hash:
            raise ArtifactError(
                f"SHA256 mismatch for {filename}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        if int(entry.get("size_bytes", -1)) != file_path.stat().st_size:
            raise ArtifactError(f"size mismatch for {filename}")
        states[filename] = _load_torch_state(
            file_path,
            map_location=map_location,
        )

    report = _validate_schema_load(
        states,
        modules=modules,
        strict_loader=strict_loader,
    )
    if modules is not None:
        if set(modules) != set(CANDIDATE_WEIGHT_FILES):
            raise ArtifactError("strict target modules must match the three-file set")
        for filename in CANDIDATE_WEIGHT_FILES:
            try:
                modules[filename].load_state_dict(states[filename], strict=True)
            except Exception as exc:
                raise ArtifactError(
                    f"strict state reload failed for {filename}: {exc}"
                ) from exc

    return LoadedCandidate(path=directory, identity=identity, states=states, strict_load_report=report)


def load_candidate(
    candidate_dir: str | Path,
    *,
    expected_identity: CandidateIdentity | None = None,
    modules: Mapping[str, nn.Module] | None = None,
    strict_loader: Any = None,
    map_location: Any = "cpu",
) -> LoadedCandidate:
    return _strict_reload_candidate(
        candidate_dir,
        expected_identity=expected_identity,
        modules=modules,
        strict_loader=strict_loader,
        map_location=map_location,
    )


def save_candidate(
    candidates_root: str | Path,
    *,
    update_id: int,
    wm_state: nn.Module | Mapping[str, Any],
    wan_control_state: nn.Module | Mapping[str, Any],
    wan_trainable_state: nn.Module | Mapping[str, Any],
    config_sha256: str,
    contract_sha256: str,
    unique_manifest_sha256: str,
    source_digest: str,
    stage0_parent_sha256: str,
) -> CandidateArtifact:
    if int(update_id) <= 0:
        raise ArtifactError("candidate update_id must be positive")
    root = Path(candidates_root)
    root.mkdir(parents=True, exist_ok=True)
    require_directory_no_symlink(root)
    destination = root / f"step_{int(update_id):08d}"
    if os.path.lexists(destination):
        raise ArtifactError(f"candidate already exists and is immutable: {destination}")

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".step_{int(update_id):08d}.tmp.",
            dir=root,
        )
    )
    if temporary.stat().st_dev != root.stat().st_dev:
        _remove_tree(temporary)
        raise ArtifactError(
            "candidate temporary directory is not on the same filesystem"
        )

    states = {
        "wm.pt": _state_mapping(wm_state, "wm"),
        "wan_control.pt": _state_mapping(wan_control_state, "wan_control"),
        "wan_trainable.pt": _state_mapping(
            wan_trainable_state,
            "wan_trainable",
        ),
    }
    try:
        for filename in CANDIDATE_WEIGHT_FILES:
            _write_torch_state(temporary / filename, states[filename])

        identity = CandidateIdentity(
            update_id=int(update_id),
            wm_sha256=sha256_file(temporary / "wm.pt"),
            wan_control_sha256=sha256_file(temporary / "wan_control.pt"),
            wan_trainable_sha256=sha256_file(temporary / "wan_trainable.pt"),
            config_sha256=config_sha256,
            contract_sha256=contract_sha256,
            unique_manifest_sha256=unique_manifest_sha256,
            source_digest=source_digest,
            stage0_parent_sha256=stage0_parent_sha256,
        )
        inventory = {
            filename: {
                "sha256": identity.weight_sha256[filename],
                "size_bytes": (temporary / filename).stat().st_size,
            }
            for filename in CANDIDATE_WEIGHT_FILES
        }
        manifest = {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "artifact_type": "wm3d_stage1_candidate",
            "candidate_id": identity.candidate_id,
            "identity": identity.to_dict(),
            "files": inventory,
        }
        _write_bytes_fsync(
            temporary / "candidate.json",
            canonical_json_bytes(manifest),
        )
        fsync_tree(temporary)
        _strict_reload_candidate(temporary, expected_identity=identity)

        for filename in CANDIDATE_FILES:
            os.chmod(temporary / filename, 0o444)
        os.chmod(temporary, 0o555)
        fsync_tree(temporary)
        rename_no_replace(temporary, destination)
        fsync_directory(root)
        return CandidateArtifact(path=destination, identity=identity)
    except Exception:
        _remove_tree(temporary)
        raise


class Stage1CheckpointComposite(nn.Module):
    """Checkpoint-only owner for the three Stage1 FSDP roots."""

    def __init__(
        self,
        wm_root: nn.Module,
        wan_transformer_root: nn.Module,
        nested_adapter_root: nn.Module,
    ) -> None:
        super().__init__()
        roots = (wm_root, wan_transformer_root, nested_adapter_root)
        if any(not isinstance(root, nn.Module) for root in roots):
            raise ArtifactError("checkpoint roots must be torch modules")
        if len({id(root) for root in roots}) != 3:
            raise ArtifactError("checkpoint roots must be distinct module objects")
        self.wm_root = wm_root
        self.wan_transformer_root = wan_transformer_root
        self.nested_adapter_root = nested_adapter_root


@dataclass(frozen=True)
class ResumeState:
    completed_steps: int
    epoch: int
    rolling_event_counter: int
    scheduler_state: Mapping[str, Any]
    sampler_state: Mapping[str, Any]
    cpu_rng_state: torch.Tensor
    cuda_rng_states: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if self.completed_steps < 0:
            raise ArtifactError("completed_steps cannot be negative")
        if self.epoch < 0:
            raise ArtifactError("epoch cannot be negative")
        if self.rolling_event_counter < 0:
            raise ArtifactError("rolling_event_counter cannot be negative")
        if self.cpu_rng_state.device.type != "cpu":
            raise ArtifactError("CPU RNG state must be a CPU tensor")

    @classmethod
    def capture(
        cls,
        *,
        completed_steps: int,
        epoch: int,
        rolling_event_counter: int,
        scheduler: Any,
        sampler: Any,
    ) -> ResumeState:
        if not hasattr(scheduler, "state_dict"):
            raise ArtifactError("scheduler must provide state_dict()")
        if not hasattr(sampler, "state_dict"):
            raise ArtifactError("sampler must provide state_dict()")
        cuda_states: Sequence[torch.Tensor] = ()
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()
        return cls(
            completed_steps=int(completed_steps),
            epoch=int(epoch),
            rolling_event_counter=int(rolling_event_counter),
            scheduler_state=copy.deepcopy(scheduler.state_dict()),
            sampler_state=copy.deepcopy(sampler.state_dict()),
            cpu_rng_state=torch.get_rng_state().clone(),
            cuda_rng_states=tuple(state.cpu().clone() for state in cuda_states),
        )

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "completed_steps": self.completed_steps,
            "epoch": self.epoch,
            "rolling_event_counter": self.rolling_event_counter,
            "scheduler_state": copy.deepcopy(dict(self.scheduler_state)),
            "sampler_state": copy.deepcopy(dict(self.sampler_state)),
            "cpu_rng_state": self.cpu_rng_state.clone(),
            "cuda_rng_states": [state.clone() for state in self.cuda_rng_states],
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> ResumeState:
        try:
            cpu_rng_state = payload["cpu_rng_state"]
            cuda_rng_states = payload["cuda_rng_states"]
            if not isinstance(cpu_rng_state, torch.Tensor):
                raise TypeError("cpu_rng_state")
            if not isinstance(cuda_rng_states, Sequence):
                raise TypeError("cuda_rng_states")
            return cls(
                completed_steps=int(payload["completed_steps"]),
                epoch=int(payload["epoch"]),
                rolling_event_counter=int(payload["rolling_event_counter"]),
                scheduler_state=copy.deepcopy(payload["scheduler_state"]),
                sampler_state=copy.deepcopy(payload["sampler_state"]),
                cpu_rng_state=cpu_rng_state.cpu().clone(),
                cuda_rng_states=tuple(state.cpu().clone() for state in cuda_rng_states),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError("invalid Stage1 resume state") from exc


def restore_resume_state(
    state: ResumeState,
    *,
    scheduler: Any,
    sampler: Any,
    restore_rng: bool = True,
) -> dict[str, int]:
    if not hasattr(scheduler, "load_state_dict"):
        raise ArtifactError("scheduler must provide load_state_dict()")
    if not hasattr(sampler, "load_state_dict"):
        raise ArtifactError("sampler must provide load_state_dict()")
    scheduler.load_state_dict(copy.deepcopy(dict(state.scheduler_state)))
    sampler.load_state_dict(copy.deepcopy(dict(state.sampler_state)))
    if restore_rng:
        torch.set_rng_state(state.cpu_rng_state.clone())
        if state.cuda_rng_states:
            if not torch.cuda.is_available():
                raise ArtifactError(
                    "resume contains CUDA RNG state but CUDA is unavailable"
                )
            if len(state.cuda_rng_states) != torch.cuda.device_count():
                raise ArtifactError(
                    "CUDA RNG state count does not match visible devices"
                )
            torch.cuda.set_rng_state_all([rng.clone() for rng in state.cuda_rng_states])
    return {
        "completed_steps": state.completed_steps,
        "epoch": state.epoch,
        "rolling_event_counter": state.rolling_event_counter,
    }


def build_dcp_state_dict(
    checkpoint: Stage1CheckpointComposite,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    from torch.distributed.checkpoint.state_dict import get_state_dict

    model_state, optimizer_state = get_state_dict(checkpoint, optimizer)
    return {
        "model": model_state,
        "optimizer": optimizer_state,
    }


def restore_dcp_state_dict(
    checkpoint: Stage1CheckpointComposite,
    optimizer: torch.optim.Optimizer,
    payload: Mapping[str, Any],
) -> None:
    from torch.distributed.checkpoint.state_dict import set_state_dict

    if set(payload) != {"model", "optimizer"}:
        raise ArtifactError("DCP payload must contain model and optimizer state")
    try:
        incompatible = set_state_dict(
            checkpoint,
            optimizer,
            model_state_dict=payload["model"],
            optim_state_dict=payload["optimizer"],
        )
    except Exception as exc:
        raise ArtifactError(f"strict DCP state restore failed: {exc}") from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ArtifactError(
            "strict DCP model restore reported incompatible keys: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def _distributed_context(process_group: Any) -> tuple[bool, int, int]:
    import torch.distributed as distributed

    active = distributed.is_available() and distributed.is_initialized()
    if not active:
        return False, 0, 1
    return (
        True,
        distributed.get_rank(process_group),
        distributed.get_world_size(process_group),
    )


def _gather_rank_states(
    state: ResumeState,
    *,
    process_group: Any,
) -> list[dict[str, Any]]:
    import torch.distributed as distributed

    active, _, world_size = _distributed_context(process_group)
    local = state.to_state_dict()
    if not active:
        return [local]
    gathered: list[dict[str, Any] | None] = [None] * world_size
    distributed.all_gather_object(gathered, local, group=process_group)
    if any(item is None for item in gathered):
        raise ArtifactError("failed to gather every rank-local resume state")
    return [item for item in gathered if item is not None]


def _resume_inventory(root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            child = current_path / directory
            metadata = os.lstat(child)
            if stat.S_ISLNK(metadata.st_mode):
                raise ArtifactError(f"resume tree contains a symlink: {child}")
        for name in files:
            path = current_path / name
            if path.name in {"resume.json", "COMPLETE"}:
                continue
            relative = path.relative_to(root).as_posix()
            inventory[relative] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    return dict(sorted(inventory.items()))


def _legacy_validate_resume_inventory(path: Path) -> Mapping[str, Any]:
    directory = require_directory_no_symlink(path)
    metadata_path = require_regular_file_no_symlink(directory / "resume.json")
    try:
        metadata = json.loads(read_bytes_no_follow(metadata_path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("resume.json is invalid") from exc
    if not isinstance(metadata, Mapping):
        raise ArtifactError("resume.json must be an object")
    if metadata.get("schema_version") != RESUME_SCHEMA_VERSION:
        raise ArtifactError("unsupported resume schema version")
    if metadata.get("artifact_type") != "wm3d_stage1_dcp_resume":
        raise ArtifactError("resume.json has the wrong artifact type")
    expected = metadata.get("files")
    if not isinstance(expected, Mapping):
        raise ArtifactError("resume.json has no file inventory")
    actual = _resume_inventory(directory)
    if set(actual) != set(expected):
        raise ArtifactError("resume file inventory is incomplete or has extra files")
    for relative, entry in expected.items():
        if not isinstance(entry, Mapping):
            raise ArtifactError(f"invalid resume inventory entry: {relative}")
        expected_hash = _require_sha256(entry.get("sha256"), relative)
        if actual[relative]["sha256"] != expected_hash:
            raise ArtifactError(f"SHA256 mismatch for resume file: {relative}")
        if actual[relative]["size_bytes"] != int(entry.get("size_bytes", -1)):
            raise ArtifactError(f"size mismatch for resume file: {relative}")
    return metadata


def _load_resume_payload(
    path: Path,
    *,
    checkpoint: Stage1CheckpointComposite,
    optimizer: torch.optim.Optimizer,
    local_state_template: ResumeState,
    process_group: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], Mapping[str, Any]]:
    import torch.distributed.checkpoint as dcp

    metadata = _validate_resume_inventory(path)
    rank_count = int(metadata.get("rank_count", 0))
    active, rank, world_size = _distributed_context(process_group)
    if rank_count != world_size:
        raise ArtifactError(
            f"resume rank count {rank_count} does not match runtime {world_size}"
        )
    payload = build_dcp_state_dict(checkpoint, optimizer)
    rank_states = [
        copy.deepcopy(local_state_template.to_state_dict()) for _ in range(rank_count)
    ]
    dcp_state = {
        **payload,
        "rank_states": rank_states,
    }
    try:
        dcp.load(
            dcp_state,
            checkpoint_id=path / "dcp",
            process_group=process_group if active else None,
            no_dist=not active,
        )
    except Exception as exc:
        raise ArtifactError(f"strict DCP reload failed: {exc}") from exc
    if rank >= len(rank_states):
        raise ArtifactError("resume does not contain this rank's state")
    return payload, rank_states, metadata


def _legacy_save_resume(
    resume_root: str | Path,
    *,
    update_id: int,
    checkpoint: Stage1CheckpointComposite,
    optimizer: torch.optim.Optimizer,
    state: ResumeState,
    process_group: Any = None,
    keep: int = 2,
) -> Path:
    import torch.distributed as distributed
    import torch.distributed.checkpoint as dcp

    if update_id <= 0 or state.completed_steps != update_id:
        raise ArtifactError("resume update_id must equal completed_steps")
    if keep < 1:
        raise ArtifactError("resume retention must keep at least one snapshot")
    root = Path(resume_root)
    active, rank, _ = _distributed_context(process_group)
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        require_directory_no_symlink(root)
        destination = root / f"step_{update_id:08d}"
        if os.path.lexists(destination):
            raise ArtifactError(f"resume snapshot already exists: {destination}")
        temporary_value = tempfile.mkdtemp(
            prefix=f".step_{update_id:08d}.tmp.",
            dir=root,
        )
    else:
        destination = root / f"step_{update_id:08d}"
        temporary_value = ""

    if active:
        values = [temporary_value]
        distributed.broadcast_object_list(values, src=0, group=process_group)
        temporary_value = values[0]
    temporary = Path(temporary_value)
    rank_states = _gather_rank_states(state, process_group=process_group)
    try:
        dcp_state = {
            **build_dcp_state_dict(checkpoint, optimizer),
            "rank_states": rank_states,
        }
        dcp.save(
            dcp_state,
            checkpoint_id=temporary / "dcp",
            process_group=process_group if active else None,
            no_dist=not active,
        )
        if active:
            distributed.barrier(group=process_group)
        if rank == 0:
            inventory = _resume_inventory(temporary)
            metadata = {
                "schema_version": RESUME_SCHEMA_VERSION,
                "artifact_type": "wm3d_stage1_dcp_resume",
                "update_id": update_id,
                "rank_count": len(rank_states),
                "files": inventory,
            }
            _write_bytes_fsync(
                temporary / "resume.json",
                canonical_json_bytes(metadata),
            )
            fsync_tree(temporary)
        if active:
            distributed.barrier(group=process_group)

        loaded_payload, loaded_rank_states, metadata = _load_resume_payload(
            temporary,
            checkpoint=checkpoint,
            optimizer=optimizer,
            local_state_template=state,
            process_group=process_group,
        )
        restore_dcp_state_dict(checkpoint, optimizer, loaded_payload)
        loaded_local = ResumeState.from_state_dict(loaded_rank_states[rank])
        if loaded_local.completed_steps != update_id:
            raise ArtifactError("strict resume reload changed completed_steps")
        if int(metadata["update_id"]) != update_id:
            raise ArtifactError("resume metadata update_id mismatch")

        if active:
            distributed.barrier(group=process_group)
        if rank == 0:
            if temporary.stat().st_dev != root.stat().st_dev:
                raise ArtifactError(
                    "resume temporary directory is not on the same filesystem"
                )
            rename_no_replace(temporary, destination)
            fsync_directory(root)
            prune_resume_snapshots(root, keep=keep)
        if active:
            distributed.barrier(group=process_group)
        return destination
    except Exception:
        if active:
            try:
                distributed.barrier(group=process_group)
            except Exception:
                pass
        if rank == 0:
            _remove_tree(temporary)
        raise


def _legacy_load_resume(
    resume_dir: str | Path,
    *,
    checkpoint: Stage1CheckpointComposite,
    optimizer: torch.optim.Optimizer,
    state_template: ResumeState,
    scheduler: Any,
    sampler: Any,
    process_group: Any = None,
) -> ResumeState:
    _, rank, _ = _distributed_context(process_group)
    payload, rank_states, metadata = _load_resume_payload(
        Path(resume_dir),
        checkpoint=checkpoint,
        optimizer=optimizer,
        local_state_template=state_template,
        process_group=process_group,
    )
    restore_dcp_state_dict(checkpoint, optimizer, payload)
    state = ResumeState.from_state_dict(rank_states[rank])
    if state.completed_steps != int(metadata["update_id"]):
        raise ArtifactError("resume state and metadata update IDs differ")
    restore_resume_state(state, scheduler=scheduler, sampler=sampler)
    return state


def _legacy_prune_resume_snapshots(
    resume_root: str | Path,
    *,
    keep: int = 2,
) -> tuple[Path, ...]:
    if keep < 1:
        raise ArtifactError("resume retention must keep at least one snapshot")
    root = Path(resume_root)
    if not root.exists():
        return ()
    require_directory_no_symlink(root)
    snapshots: list[tuple[int, Path]] = []
    for child in root.iterdir():
        match = _STEP_PATTERN.fullmatch(child.name)
        if match is None:
            continue
        require_directory_no_symlink(child)
        snapshots.append((int(match.group(1)), child))
    snapshots.sort()
    removed: list[Path] = []
    for _, path in snapshots[:-keep]:
        _remove_tree(path)
        removed.append(path)
    if removed:
        fsync_directory(root)
    return tuple(removed)

def _validate_schema_load(
    states: Mapping[str, Mapping[str, Any]],
    *,
    modules: Mapping[str, nn.Module] | None,
    strict_loader: Any,
) -> StrictLoadReport | None:
    if modules is not None and strict_loader is not None:
        raise ArtifactError("provide target modules or a strict loader, not both")
    if modules is not None:
        if set(modules) != set(CANDIDATE_WEIGHT_FILES):
            raise ArtifactError("strict target modules must match the three-file set")
        for filename in CANDIDATE_WEIGHT_FILES:
            try:
                incompatible = modules[filename].load_state_dict(
                    states[filename], strict=True
                )
            except Exception as exc:
                raise ArtifactError(
                    f"strict state reload failed for {filename}: {exc}"
                ) from exc
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise ArtifactError(
                    f"strict state reload failed for {filename}: "
                    f"missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
        return StrictLoadReport(
            loaded_files=tuple(sorted(CANDIDATE_WEIGHT_FILES)),
            missing_keys=(),
            unexpected_keys=(),
            shape_mismatches=(),
        )
    if strict_loader is None:
        return None
    try:
        raw_report = strict_loader(states)
    except Exception as exc:
        raise ArtifactError(f"injected strict loader failed: {exc}") from exc
    if isinstance(raw_report, StrictLoadReport):
        report = raw_report
    elif isinstance(raw_report, Mapping):
        try:
            report = StrictLoadReport(
                loaded_files=tuple(str(item) for item in raw_report["loaded_files"]),
                missing_keys=tuple(str(item) for item in raw_report["missing_keys"]),
                unexpected_keys=tuple(
                    str(item) for item in raw_report["unexpected_keys"]
                ),
                shape_mismatches=tuple(
                    str(item) for item in raw_report["shape_mismatches"]
                ),
            )
        except (KeyError, TypeError) as exc:
            raise ArtifactError("strict loader returned an invalid report") from exc
    else:
        raise ArtifactError("strict loader must return a StrictLoadReport")
    if set(report.loaded_files) != set(CANDIDATE_WEIGHT_FILES):
        raise ArtifactError("strict loader did not validate the exact three-file set")
    if not report.passed:
        raise ArtifactError(
            "strict loader reported schema mismatch: "
            f"missing={report.missing_keys}, unexpected={report.unexpected_keys}, "
            f"shape={report.shape_mismatches}"
        )
    return report

def _renameat2_no_replace(source: Path, destination: Path) -> bool:
    libc = ctypes.CDLL(None, use_errno=True)
    operation = getattr(libc, "renameat2", None)
    if operation is None:
        return False
    operation.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    operation.restype = ctypes.c_int
    result = operation(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return True
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ArtifactError(f"artifact destination already exists: {destination}")
    if error in {errno.ENOSYS, errno.EINVAL}:
        return False
    raise ArtifactError(
        f"atomic artifact rename failed: {source} -> {destination}"
    ) from OSError(error, os.strerror(error))


def rename_no_replace(source: str | Path, destination: str | Path) -> None:
    source_path = Path(source)
    destination_path = Path(destination)
    if os.path.lexists(destination_path):
        raise ArtifactError(f"artifact destination already exists: {destination_path}")
    if _renameat2_no_replace(source_path, destination_path):
        return
    raise ArtifactError(
        "atomic artifact rename requires renameat2 RENAME_NOREPLACE support"
    )

@dataclass(frozen=True)
class ResumeRetentionResult:
    retained: tuple[Path, ...]
    removed: tuple[Path, ...]
    quarantined: tuple[Path, ...]


def _resume_identity_payload(
    *,
    update_id: int,
    rank_count: int,
    files: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "update_id": int(update_id),
        "rank_count": int(rank_count),
        "files": dict(files),
    }


def _commit_resume_snapshot(
    path: str | Path,
    *,
    update_id: int,
    rank_count: int,
) -> Mapping[str, Any]:
    directory = require_directory_no_symlink(path)
    if os.path.lexists(directory / "resume.json") or os.path.lexists(
        directory / "COMPLETE"
    ):
        raise ArtifactError("resume snapshot is already committed")
    inventory = _resume_inventory(directory)
    identity_payload = _resume_identity_payload(
        update_id=update_id,
        rank_count=rank_count,
        files=inventory,
    )
    resume_id = hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()
    metadata = {
        "schema_version": RESUME_SCHEMA_VERSION,
        "artifact_type": "wm3d_stage1_dcp_resume",
        **identity_payload,
        "resume_id": resume_id,
    }
    _write_bytes_fsync(
        directory / "resume.json",
        canonical_json_bytes(metadata),
    )
    _write_bytes_fsync(
        directory / "COMPLETE",
        f"{resume_id}\n".encode("ascii"),
    )
    fsync_tree(directory)
    return metadata


def _validate_resume_inventory(path: Path) -> Mapping[str, Any]:
    directory = require_directory_no_symlink(path)
    metadata_path = require_regular_file_no_symlink(directory / "resume.json")
    complete_path = require_regular_file_no_symlink(directory / "COMPLETE")
    try:
        metadata = json.loads(read_bytes_no_follow(metadata_path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("resume.json is invalid") from exc
    if not isinstance(metadata, Mapping):
        raise ArtifactError("resume.json must be an object")
    if metadata.get("schema_version") != RESUME_SCHEMA_VERSION:
        raise ArtifactError("unsupported resume schema version")
    if metadata.get("artifact_type") != "wm3d_stage1_dcp_resume":
        raise ArtifactError("resume.json has the wrong artifact type")
    match = _STEP_PATTERN.fullmatch(directory.name)
    if match is not None and int(match.group(1)) != int(metadata.get("update_id", -1)):
        raise ArtifactError("resume directory and update identity differ")
    expected = metadata.get("files")
    if not isinstance(expected, Mapping):
        raise ArtifactError("resume.json has no file inventory")
    actual = _resume_inventory(directory)
    if set(actual) != set(expected):
        raise ArtifactError("resume file inventory is incomplete or has extra files")
    for relative, entry in expected.items():
        if not isinstance(entry, Mapping):
            raise ArtifactError(f"invalid resume inventory entry: {relative}")
        if actual[relative] != {
            "sha256": _require_sha256(entry.get("sha256"), relative),
            "size_bytes": int(entry.get("size_bytes", -1)),
        }:
            raise ArtifactError(f"hash or size mismatch for resume file: {relative}")
    identity_payload = _resume_identity_payload(
        update_id=int(metadata["update_id"]),
        rank_count=int(metadata["rank_count"]),
        files=expected,
    )
    resume_id = hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()
    if metadata.get("resume_id") != resume_id:
        raise ArtifactError("resume identity hash mismatch")
    if read_bytes_no_follow(complete_path) != f"{resume_id}\n".encode("ascii"):
        raise ArtifactError("resume COMPLETE marker does not match identity")
    return metadata


def prune_resume_snapshots(
    resume_root: str | Path,
    *,
    keep: int = 2,
) -> ResumeRetentionResult:
    if keep < 1:
        raise ArtifactError("resume retention must keep at least one snapshot")
    root = Path(resume_root)
    if not root.exists():
        return ResumeRetentionResult((), (), ())
    require_directory_no_symlink(root)
    valid: list[tuple[int, Path]] = []
    quarantined: list[Path] = []
    quarantine_root = root / "quarantine"
    for child in sorted(root.iterdir()):
        match = _STEP_PATTERN.fullmatch(child.name)
        if match is None:
            continue
        try:
            _validate_resume_inventory(child)
        except (ArtifactError, OSError):
            quarantine_root.mkdir(parents=True, exist_ok=True)
            destination = quarantine_root / f"{child.name}.invalid"
            suffix = 0
            while os.path.lexists(destination):
                suffix += 1
                destination = quarantine_root / f"{child.name}.invalid.{suffix}"
            rename_no_replace(child, destination)
            quarantined.append(destination)
        else:
            valid.append((int(match.group(1)), child))
    valid.sort()
    retained = tuple(path for _, path in valid[-keep:])
    removed: list[Path] = []
    for _, path in valid[:-keep]:
        _remove_tree(path)
        removed.append(path)
    if removed or quarantined:
        fsync_directory(root)
    return ResumeRetentionResult(
        retained=retained,
        removed=tuple(removed),
        quarantined=tuple(quarantined),
    )

def _collective_call(label: str, operation: Any, process_group: Any) -> Any:
    import torch.distributed as distributed

    active, rank, world_size = _distributed_context(process_group)
    try:
        result = operation()
        error = None
    except Exception as exc:
        result = None
        error = f"{type(exc).__name__}: {exc}"
    if active:
        errors: list[str | None] = [None] * world_size
        distributed.all_gather_object(errors, error, group=process_group)
    else:
        errors = [error]
    failures = [f"rank{index}={message}" for index, message in enumerate(errors) if message]
    if failures:
        raise ArtifactError(f"{label} failed consistently: {'; '.join(failures)}")
    return result


def _rank0_call(label: str, operation: Any, process_group: Any) -> Any:
    import torch.distributed as distributed

    active, rank, _ = _distributed_context(process_group)
    envelope: list[Any] = [None]
    if rank == 0:
        try:
            envelope[0] = {"ok": True, "value": operation()}
        except Exception as exc:
            envelope[0] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    if active:
        distributed.broadcast_object_list(envelope, src=0, group=process_group)
    if not envelope[0]["ok"]:
        raise ArtifactError(f"{label} failed on rank0: {envelope[0]['error']}")
    return envelope[0]["value"]


def _restore_loaded_resume_state(
    *,
    checkpoint: Stage1CheckpointComposite,
    optimizer: torch.optim.Optimizer,
    payload: Mapping[str, Any],
    rank_states: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    rank: int,
    expected_update_id: int | None = None,
    scheduler: Any | None = None,
    sampler: Any | None = None,
) -> ResumeState:
    restore_dcp_state_dict(checkpoint, optimizer, payload)
    if rank >= len(rank_states):
        raise ArtifactError("resume does not contain this rank's state")
    state = ResumeState.from_state_dict(rank_states[rank])
    metadata_update_id = int(metadata["update_id"])
    if expected_update_id is not None and metadata_update_id != expected_update_id:
        raise ArtifactError("resume metadata update_id mismatch")
    if expected_update_id is not None and state.completed_steps != expected_update_id:
        raise ArtifactError("strict resume reload changed completed_steps")
    if state.completed_steps != metadata_update_id:
        raise ArtifactError("resume state and metadata update IDs differ")
    if (scheduler is None) != (sampler is None):
        raise ArtifactError("scheduler and sampler must be provided together")
    if scheduler is not None and sampler is not None:
        restore_resume_state(state, scheduler=scheduler, sampler=sampler)
    return state


def save_resume(
    resume_root: str | Path,
    *,
    update_id: int,
    checkpoint: Stage1CheckpointComposite,
    optimizer: torch.optim.Optimizer,
    state: ResumeState,
    process_group: Any = None,
    keep: int = 2,
) -> Path:
    import torch.distributed.checkpoint as dcp

    if update_id <= 0 or state.completed_steps != update_id:
        raise ArtifactError("resume update_id must equal completed_steps")
    if keep < 1:
        raise ArtifactError("resume retention must keep at least one snapshot")
    root = Path(resume_root)
    active, rank, _ = _distributed_context(process_group)
    destination = root / f"step_{update_id:08d}"

    def prepare() -> str:
        root.mkdir(parents=True, exist_ok=True)
        require_directory_no_symlink(root)
        if os.path.lexists(destination):
            raise ArtifactError(f"resume snapshot already exists: {destination}")
        return tempfile.mkdtemp(prefix=f".step_{update_id:08d}.tmp.", dir=root)

    temporary = Path(_rank0_call("resume prepare", prepare, process_group))
    try:
        rank_states = _gather_rank_states(state, process_group=process_group)
        dcp_state = {
            **build_dcp_state_dict(checkpoint, optimizer),
            "rank_states": rank_states,
        }
        _collective_call(
            "DCP save",
            lambda: dcp.save(
                dcp_state,
                checkpoint_id=temporary / "dcp",
                process_group=process_group if active else None,
                no_dist=not active,
            ),
            process_group,
        )
        _rank0_call(
            "resume commit",
            lambda: dict(
                _commit_resume_snapshot(
                    temporary,
                    update_id=update_id,
                    rank_count=len(rank_states),
                )
            ),
            process_group,
        )
        _collective_call(
            "resume inventory validation",
            lambda: _validate_resume_inventory(temporary),
            process_group,
        )
        loaded_payload, loaded_rank_states, metadata = _collective_call(
            "DCP strict reload",
            lambda: _load_resume_payload(
                temporary,
                checkpoint=checkpoint,
                optimizer=optimizer,
                local_state_template=state,
                process_group=process_group,
            ),
            process_group,
        )
        _collective_call(
            "resume continuity",
            lambda: _restore_loaded_resume_state(
                checkpoint=checkpoint,
                optimizer=optimizer,
                payload=loaded_payload,
                rank_states=loaded_rank_states,
                metadata=metadata,
                rank=rank,
                expected_update_id=update_id,
            ),
            process_group,
        )

        def publish() -> str:
            if temporary.stat().st_dev != root.stat().st_dev:
                raise ArtifactError(
                    "resume temporary directory is not on the same filesystem"
                )
            rename_no_replace(temporary, destination)
            fsync_directory(root)
            prune_resume_snapshots(root, keep=keep)
            return str(destination)

        published = _rank0_call("resume publish", publish, process_group)
        return Path(published)
    except Exception:
        if os.path.lexists(temporary):
            try:
                _rank0_call(
                    "resume cleanup",
                    lambda: _remove_tree(temporary),
                    process_group,
                )
            except ArtifactError:
                pass
        raise


def load_resume(
    resume_dir: str | Path,
    *,
    checkpoint: Stage1CheckpointComposite,
    optimizer: torch.optim.Optimizer,
    state_template: ResumeState,
    scheduler: Any,
    sampler: Any,
    process_group: Any = None,
) -> ResumeState:
    path = Path(resume_dir)
    _, rank, _ = _distributed_context(process_group)
    _collective_call(
        "resume inventory validation",
        lambda: _validate_resume_inventory(path),
        process_group,
    )
    payload, rank_states, metadata = _collective_call(
        "DCP load",
        lambda: _load_resume_payload(
            path,
            checkpoint=checkpoint,
            optimizer=optimizer,
            local_state_template=state_template,
            process_group=process_group,
        ),
        process_group,
    )
    return _collective_call(
        "resume restore",
        lambda: _restore_loaded_resume_state(
            checkpoint=checkpoint,
            optimizer=optimizer,
            payload=payload,
            rank_states=rank_states,
            metadata=metadata,
            rank=rank,
            expected_update_id=int(metadata["update_id"]),
            scheduler=scheduler,
            sampler=sampler,
        ),
        process_group,
    )
