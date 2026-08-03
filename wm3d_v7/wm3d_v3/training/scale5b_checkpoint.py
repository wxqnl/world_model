"""Transactional FSDP2/DCP checkpoints for native WM3D-V7 5B."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import re
import stat
import uuid
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)

from wm3d_v3.data.scale5b_contracts import (
    ContractError,
    atomic_write_json,
    canonical_sha256,
    resolve_regular_file,
    safe_relative_path,
    sha256_file,
)


CHECKPOINT_SCHEMA = "wm3d_v7_native5b_fsdp2_checkpoint_v1"
COMMIT_SCHEMA = "wm3d_v7_native5b_checkpoint_commit_v1"


class CheckpointIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResumeExpectations:
    step: int
    run_lineage: str
    config_sha256: str
    dataset_receipt_sha256: str
    world_size: int
    shard_degree: int
    allow_topology_reshard: bool = False


def checkpoint_name(step: int) -> str:
    step = int(step)
    if step < 0:
        raise CheckpointIntegrityError("checkpoint step must be non-negative")
    return f"step_{step:08d}"


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


def _fsync_tree(path: Path) -> None:
    """Durably flush every regular payload and directory before publication."""

    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise CheckpointIntegrityError(
            f"checkpoint tree is not a real directory: {root}"
        )
    files = _regular_files(root)
    for file_path in files:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(file_path, flags)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise CheckpointIntegrityError(
                    f"checkpoint payload is not regular: {file_path}"
                )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    directories = {root}
    directories.update(file_path.parent for file_path in files)
    for directory in sorted(
        directories,
        key=lambda value: len(value.relative_to(root).parts),
        reverse=True,
    ):
        _fsync_directory(directory)


def _collective_phase_error(phase: str, error: Exception | None) -> None:
    """Propagate one-rank filesystem failures instead of stranding peers."""

    rank = dist.get_rank()
    local = (
        None
        if error is None
        else {
            "rank": rank,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    failures = [value for value in gathered if value is not None]
    if failures:
        raise CheckpointIntegrityError(
            f"checkpoint {phase} failed collectively: "
            + json.dumps(failures, sort_keys=True)
        )


def _atomic_torch_save(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
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
    files: list[Path] = []
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
            files.append(path)
    return sorted(files)


def _file_manifest(root: Path, *, exclusions: set[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in _regular_files(root):
        relative = path.relative_to(root).as_posix()
        if relative in exclusions:
            continue
        result[relative] = {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def _validate_metadata(
    metadata: Mapping[str, Any],
    expected: ResumeExpectations,
) -> None:
    fields = {
        "step": int(expected.step),
        "run_lineage": expected.run_lineage,
        "config_sha256": expected.config_sha256,
        "dataset_receipt_sha256": expected.dataset_receipt_sha256,
    }
    errors = [
        f"{name}: {metadata.get(name)!r} != {value!r}"
        for name, value in fields.items()
        if metadata.get(name) != value
    ]
    saved_world = int(metadata.get("world_size", -1))
    saved_shard = int(metadata.get("shard_degree", -1))
    if not expected.allow_topology_reshard:
        if saved_world != int(expected.world_size):
            errors.append(f"world_size: {saved_world} != {expected.world_size}")
        if saved_shard != int(expected.shard_degree):
            errors.append(f"shard_degree: {saved_shard} != {expected.shard_degree}")
    if errors:
        raise CheckpointIntegrityError(
            "resume metadata mismatch:\n" + "\n".join(errors)
        )


class Native5BCheckpointManager:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def save(
        self,
        *,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metadata: Mapping[str, Any],
    ) -> Path:
        if not dist.is_initialized():
            raise CheckpointIntegrityError(
                "distributed checkpoint requires process group"
            )
        rank = dist.get_rank()
        world = dist.get_world_size()
        final = self.root / checkpoint_name(step)
        launch = [None]
        if rank == 0:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                if self.root.is_symlink():
                    raise CheckpointIntegrityError(
                        "checkpoint root may not be a symlink"
                    )
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
            raise CheckpointIntegrityError(
                f"checkpoint publication precheck failed: {launch[0]}"
            )
        temporary = self.root / (
            f".{checkpoint_name(step)}.incomplete.{launch[0]['token']}"
        )
        if rank == 0:
            temporary.mkdir(mode=0o750)
            (temporary / "rng").mkdir(mode=0o750)
        dist.barrier()

        options = StateDictOptions(full_state_dict=False, cpu_offload=False)
        model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
        dcp_error: Exception | None = None
        try:
            dcp.save(
                {"model": model_state, "optimizer": optimizer_state},
                checkpoint_id=temporary / "distcp",
            )
        except Exception as exc:
            dcp_error = exc
        _collective_phase_error("DCP payload write", dcp_error)

        rng_error: Exception | None = None
        try:
            _atomic_torch_save(
                temporary / "rng" / f"rank_{rank:05d}.pt",
                _capture_rng_state(),
            )
        except Exception as exc:
            rng_error = exc
        _collective_phase_error("RNG payload write", rng_error)

        publication: list[Any] = [None]
        if rank == 0:
            complete_metadata = {
                **dict(metadata),
                "schema": CHECKPOINT_SCHEMA,
                "step": int(step),
                "world_size": world,
            }
            try:
                if int(complete_metadata.get("step", -1)) != int(step):
                    raise CheckpointIntegrityError("metadata step was overwritten")
                # DCP does not promise a POSIX durability barrier.  Flush its
                # payload before hashing it into the immutable manifest.
                _fsync_tree(temporary)
                atomic_write_json(
                    temporary / "metadata.json", complete_metadata, exclusive=True
                )
                manifest = _file_manifest(
                    temporary, exclusions={"MANIFEST.json", "COMMITTED.json"}
                )
                manifest_value = {
                    "schema": CHECKPOINT_SCHEMA,
                    "step": int(step),
                    "files": manifest,
                }
                atomic_write_json(
                    temporary / "MANIFEST.json", manifest_value, exclusive=True
                )
                commit = {
                    "schema": COMMIT_SCHEMA,
                    "step": int(step),
                    "metadata_sha256": sha256_file(temporary / "metadata.json"),
                    "manifest_sha256": sha256_file(temporary / "MANIFEST.json"),
                    "manifest_content_sha256": canonical_sha256(manifest_value),
                    "run_lineage": complete_metadata["run_lineage"],
                }
                atomic_write_json(temporary / "COMMITTED.json", commit, exclusive=True)
                _fsync_tree(temporary)
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
                f"checkpoint publication failed: {publication[0]}"
            )
        dist.barrier()
        return final

    def verify(self, path: Path) -> dict[str, Any]:
        path = Path(path)
        if path.is_symlink():
            raise CheckpointIntegrityError("checkpoint may not be a symlink")
        path = path.resolve(strict=True)
        if not path.is_dir():
            raise CheckpointIntegrityError("checkpoint must be a real directory")
        commit_path = path / "COMMITTED.json"
        manifest_path = path / "MANIFEST.json"
        metadata_path = path / "metadata.json"
        for required in (commit_path, manifest_path, metadata_path):
            if required.is_symlink() or not required.is_file():
                raise CheckpointIntegrityError(
                    f"missing checkpoint control file {required}"
                )
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if commit.get("schema") != COMMIT_SCHEMA:
            raise CheckpointIntegrityError("checkpoint commit schema mismatch")
        if manifest.get("schema") != CHECKPOINT_SCHEMA:
            raise CheckpointIntegrityError("checkpoint manifest schema mismatch")
        if metadata.get("schema") != CHECKPOINT_SCHEMA:
            raise CheckpointIntegrityError("checkpoint metadata schema mismatch")
        match = re.fullmatch(r"step_([0-9]{8})", path.name)
        if match is None:
            raise CheckpointIntegrityError(
                "checkpoint directory is not explicitly numbered"
            )
        expected_step = int(match.group(1))
        observed_steps = {
            int(commit.get("step", -1)),
            int(manifest.get("step", -1)),
            int(metadata.get("step", -1)),
        }
        if observed_steps != {expected_step}:
            raise CheckpointIntegrityError(
                f"checkpoint step identity mismatch: {sorted(observed_steps)} "
                f"!= {expected_step}"
            )
        if commit.get("run_lineage") != metadata.get("run_lineage"):
            raise CheckpointIntegrityError(
                "checkpoint commit/metadata run lineage mismatch"
            )
        if sha256_file(metadata_path) != commit.get("metadata_sha256"):
            raise CheckpointIntegrityError("checkpoint metadata digest mismatch")
        if sha256_file(manifest_path) != commit.get("manifest_sha256"):
            raise CheckpointIntegrityError("checkpoint manifest digest mismatch")
        if canonical_sha256(manifest) != commit.get("manifest_content_sha256"):
            raise CheckpointIntegrityError("checkpoint canonical manifest mismatch")
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise CheckpointIntegrityError("checkpoint manifest has no files")
        reserved = {"MANIFEST.json", "COMMITTED.json"}.intersection(files)
        if reserved:
            raise CheckpointIntegrityError(
                f"checkpoint manifest contains reserved files {sorted(reserved)}"
            )
        errors: list[str] = []
        for relative, evidence in sorted(files.items()):
            try:
                safe_relative_path(relative)
                file_path = resolve_regular_file(path, relative)
                info = os.lstat(file_path)
            except (ContractError, FileNotFoundError, OSError) as exc:
                errors.append(f"{relative}: {exc}")
                continue
            if info.st_size != int(evidence["size"]):
                errors.append(f"{relative}: size {info.st_size} != {evidence['size']}")
                continue
            digest = sha256_file(file_path)
            if digest != evidence["sha256"]:
                errors.append(f"{relative}: sha256 {digest} != {evidence['sha256']}")
        expected_files = set(files) | {
            "MANIFEST.json",
            "COMMITTED.json",
        }
        actual_files = {
            item.relative_to(path).as_posix() for item in _regular_files(path)
        }
        if actual_files != expected_files:
            errors.append(
                "checkpoint file set mismatch: "
                f"missing={sorted(expected_files - actual_files)} "
                f"extra={sorted(actual_files - expected_files)}"
            )
        if errors:
            raise CheckpointIntegrityError(
                "checkpoint payload verification failed:\n" + "\n".join(errors)
            )
        return metadata

    def load(
        self,
        *,
        path: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        expected: ResumeExpectations,
    ) -> dict[str, Any]:
        rank = dist.get_rank()
        checkpoint_path = Path(path).resolve(strict=True)
        checkpoint_root = self.root.resolve(strict=True)
        if checkpoint_path.parent != checkpoint_root:
            raise CheckpointIntegrityError(
                f"resume checkpoint escaped run root: {checkpoint_path}"
            )
        result: list[Any] = [None]
        if rank == 0:
            try:
                metadata = self.verify(path)
                _validate_metadata(metadata, expected)
                result[0] = {"ok": True, "metadata": metadata}
            except Exception as exc:
                result[0] = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        dist.broadcast_object_list(result, src=0)
        if not result[0]["ok"]:
            raise CheckpointIntegrityError(
                f"rank0 checkpoint verification failed: {result[0]}"
            )
        metadata = result[0]["metadata"]
        options = StateDictOptions(full_state_dict=False, cpu_offload=False)
        model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
        state = {"model": model_state, "optimizer": optimizer_state}
        dcp_error: Exception | None = None
        try:
            dcp.load(state, checkpoint_id=Path(path) / "distcp")
        except Exception as exc:
            dcp_error = exc
        _collective_phase_error("DCP payload load", dcp_error)
        state_error: Exception | None = None
        try:
            set_state_dict(
                model,
                optimizer,
                model_state_dict=state["model"],
                optim_state_dict=state["optimizer"],
                options=options,
            )
        except Exception as exc:
            state_error = exc
        _collective_phase_error("model/optimizer restore", state_error)
        rng_path = Path(path) / "rng" / f"rank_{rank:05d}.pt"
        rng_error: Exception | None = None
        try:
            if not rng_path.is_file():
                if expected.allow_topology_reshard:
                    seed = (
                        int(metadata["initial_seed"])
                        + int(expected.step) * 1_000_003
                        + rank
                    )
                    random.seed(seed)
                    np.random.seed(seed % (2**32))
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                else:
                    raise CheckpointIntegrityError(
                        f"missing exact RNG state {rng_path}"
                    )
            else:
                _restore_rng_state(
                    torch.load(
                        rng_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                )
        except Exception as exc:
            rng_error = exc
        _collective_phase_error("per-rank RNG restore", rng_error)
        dist.barrier()
        return metadata
