"""Transactional, sharded, exact-resume checkpoints for WM3D V8.

The manager is shared by DDP and FSDP2.  It never materializes a full model on
rank zero.  DCP writes model/optimizer shards; integrity hashing and validation
are also partitioned across ranks so checkpoint cost scales with the cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import stat
import uuid
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)


CHECKPOINT_SCHEMA = "wm3d_v8_distributed_checkpoint_v2"
COMMIT_SCHEMA = "wm3d_v8_distributed_checkpoint_commit_v2"
_LOAD_SNAPSHOT_DIRECTORY = ".wm3d_checkpoint_load_snapshots"
_LOAD_SNAPSHOT_FREE_RESERVE_BYTES = 1 << 30
_COPY_CHUNK_BYTES = 16 << 20


class CheckpointIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResumeExpectations:
    step: int
    run_lineage: str
    runtime_config_sha256: str
    data_closure_sha256: str
    model_contract_sha256: str
    world_size: int
    shard_degree: int
    distributed_strategy: str
    global_batch_size: int
    topology_contract_sha256: str
    allow_topology_reshard: bool = False
    extra_immutable_metadata: Mapping[str, Any] | None = None


def checkpoint_name(step: int) -> str:
    step = int(step)
    if step < 0:
        raise CheckpointIntegrityError("checkpoint step must be non-negative")
    return f"step_{step:08d}"


def sha256_file(path: Path, *, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: object, *, exclusive: bool = True) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    payload = (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if exclusive:
        try:
            os.link(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        temporary.unlink()
    else:
        os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_torch_save(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink()
    _fsync_directory(path.parent)


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None
        ),
    }


def _restore_rng_state(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if value.get("torch_cuda") is not None:
        torch.cuda.set_rng_state(value["torch_cuda"])


def _regular_files(root: Path) -> list[Path]:
    result: list[Path] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in dirnames:
            path = directory_path / dirname
            if stat.S_ISLNK(os.lstat(path).st_mode):
                raise CheckpointIntegrityError(f"symlink in checkpoint: {path}")
        for filename in filenames:
            path = directory_path / filename
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise CheckpointIntegrityError(f"non-regular checkpoint file: {path}")
            result.append(path)
    return sorted(result)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_absolute_directory_no_symlinks(path: Path) -> int:
    """Pin an existing absolute directory chain without following symlinks."""

    absolute = Path(os.path.abspath(path))
    if not absolute.is_absolute():
        raise CheckpointIntegrityError("checkpoint snapshot path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            if not component or component in {".", ".."}:
                raise CheckpointIntegrityError(
                    "checkpoint snapshot path has an invalid component"
                )
            child = os.open(component, flags, dir_fd=descriptor)
            child_info = os.fstat(child)
            if not stat.S_ISDIR(child_info.st_mode):
                os.close(child)
                raise CheckpointIntegrityError(
                    "checkpoint snapshot path has a non-directory ancestor"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _copy_regular_no_clobber(source: Path, destination: Path) -> None:
    """Copy one regular file through pinned descriptors into a new pathname."""

    source_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    destination_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
    )
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
        destination_flags |= os.O_NOFOLLOW
    source_descriptor = os.open(source, source_flags)
    destination_descriptor: int | None = None
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CheckpointIntegrityError(
                f"checkpoint snapshot source is not regular: {source}"
            )
        destination_descriptor = os.open(destination, destination_flags, 0o600)
        digest = hashlib.sha256()
        copied = 0
        while True:
            block = os.read(source_descriptor, _COPY_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise OSError("checkpoint snapshot copy made no progress")
                view = view[written:]
                copied += written
        os.fsync(destination_descriptor)
        os.fchmod(destination_descriptor, 0o400)
        after = os.fstat(source_descriptor)
        if _stat_identity(before) != _stat_identity(after) or copied != before.st_size:
            raise CheckpointIntegrityError(
                f"checkpoint file changed while snapshotting: {source}"
            )
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        os.close(source_descriptor)
    if sha256_file(destination) != digest.hexdigest():
        raise CheckpointIntegrityError(
            f"checkpoint snapshot copy digest mismatch: {destination}"
        )


def _snapshot_parent(path: Path) -> Path:
    """Create/open the private namespace through a pinned real parent."""

    base = Path(os.path.abspath(path.parent))
    if path.name != _LOAD_SNAPSHOT_DIRECTORY:
        raise CheckpointIntegrityError("unexpected checkpoint snapshot parent name")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    base_descriptor = _open_absolute_directory_no_symlinks(base)
    try:
        try:
            os.mkdir(path.name, mode=0o700, dir_fd=base_descriptor)
            os.fsync(base_descriptor)
        except FileExistsError:
            pass
        parent_descriptor = os.open(path.name, flags, dir_fd=base_descriptor)
        try:
            info = os.fstat(parent_descriptor)
            observed = os.stat(path.name, dir_fd=base_descriptor, follow_symlinks=False)
            if (
                not stat.S_ISDIR(info.st_mode)
                or _stat_identity(info) != _stat_identity(observed)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o700
            ):
                raise CheckpointIntegrityError(
                    "checkpoint snapshot parent is not private and stable"
                )
        finally:
            os.close(parent_descriptor)
    finally:
        os.close(base_descriptor)
    return base / path.name


def _new_snapshot_container(snapshot_parent: Path) -> Path:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    parent_descriptor = os.open(snapshot_parent, flags)
    try:
        for _ in range(32):
            name = f"load-{os.getpid()}-{uuid.uuid4().hex}"
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
            except FileExistsError:
                continue
            container_descriptor = os.open(name, flags, dir_fd=parent_descriptor)
            try:
                info = os.fstat(container_descriptor)
                observed = os.stat(
                    name, dir_fd=parent_descriptor, follow_symlinks=False
                )
                if (
                    not stat.S_ISDIR(info.st_mode)
                    or _stat_identity(info) != _stat_identity(observed)
                ):
                    raise CheckpointIntegrityError(
                        "checkpoint snapshot container identity changed"
                    )
            finally:
                os.close(container_descriptor)
            os.fsync(parent_descriptor)
            return snapshot_parent / name
    finally:
        os.close(parent_descriptor)
    raise CheckpointIntegrityError("could not allocate checkpoint snapshot container")


def _fsync_tree_directories(root: Path) -> None:
    directories = [Path(directory) for directory, _, _ in os.walk(root)]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        _fsync_directory(directory)


def _make_private_checkpoint_snapshot(
    source: Path,
    *,
    snapshot_parent: Path,
) -> tuple[Path, Path]:
    """Copy a checkpoint into a no-clobber, job-private load namespace.

    The copy deliberately does not use reflinks: not all formal filesystems
    support FICLONE, and a private byte copy is the portable isolation boundary.
    The caller must validate the copied controls and every manifest payload
    before passing the returned path to DCP.
    """

    if source.is_symlink():
        raise CheckpointIntegrityError("checkpoint cannot be a symlink")
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise CheckpointIntegrityError("checkpoint must be a real directory")
    source_files = _regular_files(source)
    if not source_files:
        raise CheckpointIntegrityError("checkpoint contains no files")
    total_bytes = sum(os.lstat(path).st_size for path in source_files)

    snapshot_parent = _snapshot_parent(snapshot_parent)
    available = shutil.disk_usage(snapshot_parent).free
    required = total_bytes + _LOAD_SNAPSHOT_FREE_RESERVE_BYTES
    if available < required:
        raise CheckpointIntegrityError(
            "insufficient free space for private checkpoint snapshot: "
            f"available={available} required={required} payload={total_bytes}"
        )

    container = _new_snapshot_container(snapshot_parent).resolve(strict=True)
    snapshot = container / source.name
    try:
        snapshot.mkdir(mode=0o700)
        for source_file in source_files:
            relative = _safe_relative(source_file.relative_to(source).as_posix())
            destination = snapshot / relative
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if destination.parent.is_symlink():
                raise CheckpointIntegrityError(
                    f"checkpoint snapshot destination escaped: {destination}"
                )
            _copy_regular_no_clobber(source_file, destination)
        _fsync_tree_directories(snapshot)
        for directory, _, _ in os.walk(snapshot, topdown=False):
            os.chmod(directory, 0o500, follow_symlinks=False)
        _fsync_directory(container)
        _fsync_directory(snapshot_parent)
    except Exception:
        _remove_private_checkpoint_snapshot(container)
        raise
    return container, snapshot


def _remove_private_checkpoint_snapshot(container: Path) -> None:
    """Best-effort writable transition followed by complete private cleanup."""

    if not container.exists() and not container.is_symlink():
        return
    if container.is_symlink():
        raise CheckpointIntegrityError("checkpoint snapshot container became a symlink")
    for directory, dirnames, filenames in os.walk(container, topdown=False):
        directory_path = Path(directory)
        for filename in filenames:
            path = directory_path / filename
            if path.is_symlink():
                raise CheckpointIntegrityError(
                    f"symlink appeared in checkpoint snapshot: {path}"
                )
            os.chmod(path, 0o600, follow_symlinks=False)
        for dirname in dirnames:
            path = directory_path / dirname
            if path.is_symlink():
                raise CheckpointIntegrityError(
                    f"symlink appeared in checkpoint snapshot: {path}"
                )
            os.chmod(path, 0o700, follow_symlinks=False)
        os.chmod(directory_path, 0o700, follow_symlinks=False)
    parent = container.parent
    shutil.rmtree(container)
    _fsync_directory(parent)


def _safe_relative(relative: str) -> str:
    path = Path(relative)
    if not relative or path.is_absolute() or ".." in path.parts:
        raise CheckpointIntegrityError(f"unsafe checkpoint path {relative!r}")
    normalized = path.as_posix()
    if normalized != relative:
        raise CheckpointIntegrityError(f"non-canonical checkpoint path {relative!r}")
    return normalized


def _collective_error(phase: str, error: Optional[Exception]) -> None:
    local = (
        None
        if error is None
        else {
            "rank": dist.get_rank(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    failures = [item for item in gathered if item is not None]
    if failures:
        raise CheckpointIntegrityError(
            f"checkpoint {phase} failed collectively: "
            + json.dumps(failures, sort_keys=True)
        )


def _distributed_manifest(
    root: Path,
    *,
    exclusions: set[str],
) -> dict[str, dict[str, Any]]:
    """Hash checkpoint files in parallel and return the merged rank0 manifest."""

    files = [
        path
        for path in _regular_files(root)
        if path.relative_to(root).as_posix() not in exclusions
    ]
    local: dict[str, dict[str, Any]] = {}
    error: Optional[Exception] = None
    try:
        for index, path in enumerate(files):
            if index % dist.get_world_size() != dist.get_rank():
                continue
            relative = path.relative_to(root).as_posix()
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            local[relative] = {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    except Exception as exc:
        error = exc
    _collective_error("parallel manifest hashing", error)
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    merged: dict[str, dict[str, Any]] = {}
    for shard in gathered:
        for relative, evidence in shard.items():
            if relative in merged:
                raise CheckpointIntegrityError(
                    f"checkpoint manifest contains duplicate path {relative}"
                )
            merged[relative] = evidence
    if len(merged) != len(files):
        raise CheckpointIntegrityError(
            f"parallel manifest lost files: {len(merged)} != {len(files)}"
        )
    return dict(sorted(merged.items()))


def _distributed_verify_payload(
    root: Path,
    files: Mapping[str, Mapping[str, Any]],
) -> None:
    errors: list[str] = []
    ordered = sorted(files.items())
    for index, (relative, evidence) in enumerate(ordered):
        if index % dist.get_world_size() != dist.get_rank():
            continue
        try:
            relative = _safe_relative(relative)
            path = root / relative
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise CheckpointIntegrityError("not a regular file")
            expected_size = int(evidence["size"])
            if info.st_size != expected_size:
                raise CheckpointIntegrityError(
                    f"size {info.st_size} != {expected_size}"
                )
            digest = sha256_file(path)
            if digest != evidence["sha256"]:
                raise CheckpointIntegrityError(
                    f"sha256 {digest} != {evidence['sha256']}"
                )
        except Exception as exc:
            errors.append(f"{relative}: {exc}")
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, errors)
    failures = [item for shard in gathered for item in shard]
    if failures:
        raise CheckpointIntegrityError(
            "checkpoint payload verification failed:\n" + "\n".join(failures)
        )


def _validate_metadata(
    metadata: Mapping[str, Any],
    expected: ResumeExpectations,
) -> str:
    """Validate a resume and return ``exact`` or ``topology_reshard``.

    A topology change is intentionally a different contract from an exact
    resume: DCP may reshard model/optimizer tensors, but per-rank RNG streams
    cannot be continued one-for-one when the rank set changes.  Same-topology
    resumes therefore always require the exact materialized runtime digest;
    setting ``allow_topology_reshard`` must never weaken that check.
    """

    exact = {
        "step": int(expected.step),
        "run_lineage": expected.run_lineage,
        "data_closure_sha256": expected.data_closure_sha256,
        "model_contract_sha256": expected.model_contract_sha256,
        "distributed_strategy": expected.distributed_strategy,
        "global_batch_size": int(expected.global_batch_size),
    }
    errors = [
        f"{name}: {metadata.get(name)!r} != {value!r}"
        for name, value in exact.items()
        if metadata.get(name) != value
    ]
    extra = expected.extra_immutable_metadata
    if extra is not None:
        if not isinstance(extra, Mapping):
            errors.append("extra_immutable_metadata expectation must be a mapping")
        else:
            for name, value in extra.items():
                if not isinstance(name, str) or not name:
                    errors.append(
                        "extra_immutable_metadata expectation keys must be "
                        "non-empty strings"
                    )
                elif name not in metadata:
                    errors.append(f"{name}: missing != {value!r}")
                elif metadata[name] != value:
                    errors.append(f"{name}: {metadata[name]!r} != {value!r}")
    saved_world_size = int(metadata.get("world_size", -1))
    saved_shard_degree = int(metadata.get("shard_degree", -1))
    topology_changed = saved_world_size != int(expected.world_size)
    if not topology_changed:
        if metadata.get("runtime_config_sha256") != expected.runtime_config_sha256:
            errors.append(
                "runtime_config_sha256: "
                f"{metadata.get('runtime_config_sha256')!r} != "
                f"{expected.runtime_config_sha256!r}"
            )
        if saved_shard_degree != int(expected.shard_degree):
            errors.append(
                f"shard_degree: {saved_shard_degree} != {expected.shard_degree}"
            )
        mode = "exact"
    else:
        mode = "topology_reshard"
        if not expected.allow_topology_reshard:
            errors.append(
                f"world_size: {saved_world_size} != {expected.world_size}; "
                "allow_topology_reshard is false"
            )
        if (
            metadata.get("distributed_strategy") != "fsdp2"
            or expected.distributed_strategy != "fsdp2"
        ):
            errors.append("topology reshard is supported only for FSDP2")
        if saved_shard_degree != int(expected.shard_degree):
            errors.append(
                "topology reshard requires an unchanged shard mesh degree: "
                f"{saved_shard_degree} != {expected.shard_degree}"
            )
        if (
            saved_shard_degree <= 1
            or saved_world_size <= 0
            or saved_world_size % saved_shard_degree
            or int(expected.world_size) % int(expected.shard_degree)
        ):
            errors.append("saved/current FSDP2 shard meshes are not compatible")
        if metadata.get("topology_contract_sha256") != expected.topology_contract_sha256:
            errors.append(
                "topology_contract_sha256: "
                f"{metadata.get('topology_contract_sha256')!r} != "
                f"{expected.topology_contract_sha256!r}"
            )
        sampler_progress = metadata.get("sampler_progress")
        if not isinstance(sampler_progress, dict) or int(
            sampler_progress.get("next_optimizer_step", -1)
        ) != int(expected.step):
            errors.append("topology reshard lacks exact global sampler progress")
    if errors:
        raise CheckpointIntegrityError(
            "resume metadata mismatch:\n" + "\n".join(errors)
        )
    return mode


class DistributedCheckpointManager:
    def __init__(self, root: Path):
        self.root = Path(root)

    def save(
        self,
        *,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metadata: Mapping[str, Any],
        rank_state: Mapping[str, Any],
    ) -> Path:
        if not dist.is_initialized():
            raise CheckpointIntegrityError("distributed checkpoint requires process group")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        final = self.root / checkpoint_name(step)
        launch: list[Any] = [None]
        if rank == 0:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                if self.root.is_symlink():
                    raise CheckpointIntegrityError("checkpoint root cannot be a symlink")
                if final.exists():
                    raise FileExistsError(final)
                launch[0] = {"ok": True, "token": uuid.uuid4().hex}
            except Exception as exc:
                launch[0] = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        dist.broadcast_object_list(launch, src=0)
        if not launch[0]["ok"]:
            raise CheckpointIntegrityError(f"checkpoint precheck failed: {launch[0]}")
        temporary = self.root / (
            f".{checkpoint_name(step)}.incomplete.{launch[0]['token']}"
        )
        if rank == 0:
            temporary.mkdir(mode=0o750)
            (temporary / "rank_state").mkdir(mode=0o750)
        dist.barrier()

        options = StateDictOptions(full_state_dict=False, cpu_offload=False)
        model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
        error: Optional[Exception] = None
        try:
            dcp.save(
                {"model": model_state, "optimizer": optimizer_state},
                checkpoint_id=temporary / "distcp",
            )
        except Exception as exc:
            error = exc
        _collective_error("DCP save", error)

        error = None
        try:
            _atomic_torch_save(
                temporary / "rank_state" / f"rank_{rank:05d}.pt",
                {"rng": _capture_rng_state(), "progress": dict(rank_state)},
            )
        except Exception as exc:
            error = exc
        _collective_error("rank state save", error)

        reserved = {"schema", "step", "world_size"}
        conflict = sorted(reserved & set(metadata))
        if conflict:
            raise CheckpointIntegrityError(
                f"caller metadata cannot set reserved fields: {conflict}"
            )
        complete_metadata = {
            **dict(metadata),
            "schema": CHECKPOINT_SCHEMA,
            "step": int(step),
            "world_size": world_size,
        }
        publication: list[Any] = [None]
        if rank == 0:
            try:
                _atomic_json(temporary / "metadata.json", complete_metadata)
                publication[0] = {"ok": True}
            except Exception as exc:
                publication[0] = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        dist.broadcast_object_list(publication, src=0)
        if not publication[0]["ok"]:
            raise CheckpointIntegrityError(
                f"checkpoint metadata publication failed: {publication[0]}"
            )
        dist.barrier()

        files = _distributed_manifest(
            temporary, exclusions={"MANIFEST.json", "COMMITTED.json"}
        )
        publication = [None]
        if rank == 0:
            try:
                manifest = {
                    "schema": CHECKPOINT_SCHEMA,
                    "step": int(step),
                    "files": files,
                }
                _atomic_json(temporary / "MANIFEST.json", manifest)
                commit = {
                    "schema": COMMIT_SCHEMA,
                    "step": int(step),
                    "run_lineage": complete_metadata["run_lineage"],
                    "metadata_sha256": sha256_file(temporary / "metadata.json"),
                    "manifest_sha256": sha256_file(temporary / "MANIFEST.json"),
                    "manifest_content_sha256": canonical_sha256(manifest),
                }
                _atomic_json(temporary / "COMMITTED.json", commit)
                os.replace(temporary, final)
                _fsync_directory(self.root)
                publication[0] = {"ok": True}
            except Exception as exc:
                publication[0] = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        dist.broadcast_object_list(publication, src=0)
        if not publication[0]["ok"]:
            raise CheckpointIntegrityError(
                f"checkpoint commit failed: {publication[0]}"
            )
        dist.barrier()
        return final

    def _verify_controls(self, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        if path.is_symlink():
            raise CheckpointIntegrityError("checkpoint cannot be a symlink")
        path = path.resolve(strict=True)
        if not path.is_dir():
            raise CheckpointIntegrityError("checkpoint must be a real directory")
        match = re.fullmatch(r"step_([0-9]{8})", path.name)
        if match is None:
            raise CheckpointIntegrityError("checkpoint must use an explicit numbered name")
        expected_step = int(match.group(1))
        controls = {
            name: path / name
            for name in ("metadata.json", "MANIFEST.json", "COMMITTED.json")
        }
        for name, control in controls.items():
            info = os.lstat(control)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise CheckpointIntegrityError(f"invalid control file {name}")
        metadata = json.loads(controls["metadata.json"].read_text(encoding="utf-8"))
        manifest = json.loads(controls["MANIFEST.json"].read_text(encoding="utf-8"))
        commit = json.loads(controls["COMMITTED.json"].read_text(encoding="utf-8"))
        if metadata.get("schema") != CHECKPOINT_SCHEMA:
            raise CheckpointIntegrityError("metadata schema mismatch")
        if manifest.get("schema") != CHECKPOINT_SCHEMA:
            raise CheckpointIntegrityError("manifest schema mismatch")
        if commit.get("schema") != COMMIT_SCHEMA:
            raise CheckpointIntegrityError("commit schema mismatch")
        observed_steps = {
            int(metadata.get("step", -1)),
            int(manifest.get("step", -1)),
            int(commit.get("step", -1)),
        }
        if observed_steps != {expected_step}:
            raise CheckpointIntegrityError(
                f"checkpoint step identity mismatch: {observed_steps} != {expected_step}"
            )
        if commit.get("run_lineage") != metadata.get("run_lineage"):
            raise CheckpointIntegrityError("commit/metadata lineage mismatch")
        if sha256_file(controls["metadata.json"]) != commit.get("metadata_sha256"):
            raise CheckpointIntegrityError("metadata digest mismatch")
        if sha256_file(controls["MANIFEST.json"]) != commit.get("manifest_sha256"):
            raise CheckpointIntegrityError("manifest digest mismatch")
        if canonical_sha256(manifest) != commit.get("manifest_content_sha256"):
            raise CheckpointIntegrityError("canonical manifest digest mismatch")
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise CheckpointIntegrityError("checkpoint manifest contains no payload")
        expected_files = set(files) | {"MANIFEST.json", "COMMITTED.json"}
        actual_files = {
            item.relative_to(path).as_posix() for item in _regular_files(path)
        }
        if actual_files != expected_files:
            raise CheckpointIntegrityError(
                "checkpoint file set mismatch: "
                f"missing={sorted(expected_files - actual_files)} "
                f"extra={sorted(actual_files - expected_files)}"
            )
        return metadata, files

    def inspect_committed(
        self,
        *,
        path: Path,
        expected: ResumeExpectations,
    ) -> dict[str, Any]:
        """Validate immutable controls before a launch qualification is published.

        Payload shards are still verified collectively by :meth:`load`; this
        method only exposes the already strict, rank-0-safe control verdict and
        the COMMITTED digest needed to bind the launch to an explicit source.
        """

        checkpoint_path = Path(path).resolve(strict=True)
        root = self.root.resolve(strict=True)
        if checkpoint_path.parent != root:
            raise CheckpointIntegrityError(
                f"checkpoint escaped run root: {checkpoint_path}"
            )
        metadata, _ = self._verify_controls(checkpoint_path)
        resume_mode = _validate_metadata(metadata, expected)
        return {
            "path": str(checkpoint_path),
            "step": int(metadata["step"]),
            "committed_sha256": sha256_file(checkpoint_path / "COMMITTED.json"),
            "saved_world_size": int(metadata["world_size"]),
            "saved_shard_degree": int(metadata["shard_degree"]),
            "resume_mode": resume_mode,
        }

    def inspect_committed_collective(
        self,
        *,
        path: Path,
        expected: ResumeExpectations,
        require_exact: bool = False,
    ) -> dict[str, Any]:
        """Broadcast one rank-0 committed-control inspection verdict."""

        if not dist.is_initialized():
            raise CheckpointIntegrityError(
                "distributed checkpoint inspection requires process group"
            )
        verdict: list[Any] = [None]
        if dist.get_rank() == 0:
            try:
                source = self.inspect_committed(path=path, expected=expected)
                if require_exact and source["resume_mode"] != "exact":
                    raise CheckpointIntegrityError(
                        "checkpoint inspection requires exact topology"
                    )
                verdict[0] = {"ok": True, "source": source}
            except Exception as exc:
                verdict[0] = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        dist.broadcast_object_list(verdict, src=0)
        if not verdict[0]["ok"]:
            raise CheckpointIntegrityError(
                f"checkpoint inspection failed: {verdict[0]}"
            )
        return dict(verdict[0]["source"])

    def _snapshot_for_load(
        self,
        *,
        checkpoint_path: Path,
        expected: ResumeExpectations,
        require_exact: bool,
    ) -> tuple[Path, Path, dict[str, Any], str]:
        """Collectively create and validate one private load snapshot."""

        verdict: list[Any] = [None]
        if dist.get_rank() == 0:
            container: Path | None = None
            try:
                snapshot_parent = self.root.parent / _LOAD_SNAPSHOT_DIRECTORY
                container, snapshot = _make_private_checkpoint_snapshot(
                    checkpoint_path,
                    snapshot_parent=snapshot_parent,
                )
                metadata, files = self._verify_controls(snapshot)
                resume_mode = _validate_metadata(metadata, expected)
                if require_exact and resume_mode != "exact":
                    raise CheckpointIntegrityError(
                        "checkpoint load snapshot requires exact topology"
                    )
                verdict[0] = {
                    "ok": True,
                    "container": str(container),
                    "snapshot": str(snapshot),
                    "metadata": metadata,
                    "files": files,
                    "resume_mode": resume_mode,
                }
            except Exception as exc:
                if container is not None:
                    try:
                        _remove_private_checkpoint_snapshot(container)
                    except Exception:
                        pass
                verdict[0] = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        dist.broadcast_object_list(verdict, src=0)
        if not verdict[0]["ok"]:
            raise CheckpointIntegrityError(
                f"checkpoint snapshot creation failed: {verdict[0]}"
            )
        container = Path(verdict[0]["container"])
        snapshot = Path(verdict[0]["snapshot"])
        metadata = dict(verdict[0]["metadata"])
        resume_mode = str(verdict[0]["resume_mode"])
        try:
            _distributed_verify_payload(snapshot, verdict[0]["files"])
        except Exception:
            self._cleanup_load_snapshot(container)
            raise
        return container, snapshot, metadata, resume_mode

    @staticmethod
    def _cleanup_load_snapshot(container: Path) -> None:
        error: Optional[Exception] = None
        dist.barrier()
        if dist.get_rank() == 0:
            try:
                _remove_private_checkpoint_snapshot(container)
            except Exception as exc:
                error = exc
        _collective_error("private snapshot cleanup", error)

    def load(
        self,
        *,
        path: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        expected: ResumeExpectations,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not dist.is_initialized():
            raise CheckpointIntegrityError("distributed checkpoint requires process group")
        checkpoint_path = Path(path).resolve(strict=True)
        root = self.root.resolve(strict=True)
        if checkpoint_path.parent != root:
            raise CheckpointIntegrityError(
                f"resume checkpoint escaped run root: {checkpoint_path}"
            )
        container, snapshot, metadata, resume_mode = self._snapshot_for_load(
            checkpoint_path=checkpoint_path,
            expected=expected,
            require_exact=False,
        )
        try:
            options = StateDictOptions(full_state_dict=False, cpu_offload=False)
            error: Optional[Exception] = None
            try:
                model_state, optimizer_state = get_state_dict(
                    model, optimizer, options=options
                )
            except Exception as exc:
                error = exc
            _collective_error("state-dict preparation", error)
            state = {"model": model_state, "optimizer": optimizer_state}
            error = None
            try:
                dcp.load(state, checkpoint_id=snapshot / "distcp")
            except Exception as exc:
                error = exc
            _collective_error("DCP load", error)
            error = None
            try:
                set_state_dict(
                    model,
                    optimizer,
                    model_state_dict=state["model"],
                    optim_state_dict=state["optimizer"],
                    options=options,
                )
            except Exception as exc:
                error = exc
            _collective_error("model/optimizer restore", error)

            rank_state_path = (
                snapshot / "rank_state" / f"rank_{dist.get_rank():05d}.pt"
            )
            progress: dict[str, Any] = {}
            error = None
            try:
                if resume_mode == "exact":
                    if not rank_state_path.is_file():
                        raise CheckpointIntegrityError(
                            f"missing exact per-rank state {rank_state_path}"
                        )
                    rank_state = torch.load(
                        rank_state_path, map_location="cpu", weights_only=False
                    )
                    _restore_rng_state(rank_state["rng"])
                    progress = dict(rank_state["progress"])
                elif resume_mode == "topology_reshard":
                    # A changed rank set has no one-to-one RNG continuation.  All
                    # ranks, including reused rank numbers, start a newly bound
                    # deterministic stream; never mix old and new RNG states.
                    if "initial_seed" not in metadata:
                        raise CheckpointIntegrityError(
                            "topology reshard requires initial_seed metadata"
                        )
                    seed = (
                        int(metadata["initial_seed"])
                        + int(expected.step) * 1_000_003
                        + dist.get_rank()
                    )
                    random.seed(seed)
                    np.random.seed(seed % (2**32))
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    progress = dict(metadata["sampler_progress"])
                else:
                    raise CheckpointIntegrityError(
                        f"unknown resume mode {resume_mode!r}"
                    )
            except Exception as exc:
                error = exc
            _collective_error("rank state restore", error)
            return metadata, progress
        finally:
            self._cleanup_load_snapshot(container)

    def load_model_for_evaluation(
        self,
        *,
        path: Path,
        model: torch.nn.Module,
        expected: ResumeExpectations,
    ) -> dict[str, Any]:
        """Strictly load only model shards from a committed checkpoint.

        Offline evaluation must not fabricate an optimizer merely to consume a
        training checkpoint.  It does, however, use exactly the same control,
        manifest, lineage, topology, and DCP validation as training resume.
        Evaluation is same-topology only; changing topology remains an explicit
        training-reshard operation with new RNG streams.
        """

        if expected.allow_topology_reshard:
            raise CheckpointIntegrityError(
                "offline evaluation cannot request topology reshard"
            )
        if not dist.is_initialized():
            raise CheckpointIntegrityError("distributed checkpoint requires process group")
        checkpoint_path = Path(path).resolve(strict=True)
        root = self.root.resolve(strict=True)
        if checkpoint_path.parent != root:
            raise CheckpointIntegrityError(
                f"evaluation checkpoint escaped run root: {checkpoint_path}"
            )
        container, snapshot, metadata, _mode = self._snapshot_for_load(
            checkpoint_path=checkpoint_path,
            expected=expected,
            require_exact=True,
        )
        try:
            options = StateDictOptions(full_state_dict=False, cpu_offload=False)
            error: Optional[Exception] = None
            try:
                model_state = get_model_state_dict(model, options=options)
            except Exception as exc:
                error = exc
            _collective_error("evaluation state-dict preparation", error)
            error = None
            try:
                dcp.load({"model": model_state}, checkpoint_id=snapshot / "distcp")
                set_model_state_dict(model, model_state, options=options)
            except Exception as exc:
                error = exc
            _collective_error("evaluation model restore", error)
            return metadata
        finally:
            self._cleanup_load_snapshot(container)
