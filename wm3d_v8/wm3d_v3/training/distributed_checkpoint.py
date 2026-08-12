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
        verification: list[Any] = [None]
        if dist.get_rank() == 0:
            try:
                metadata, files = self._verify_controls(checkpoint_path)
                resume_mode = _validate_metadata(metadata, expected)
                verification[0] = {
                    "ok": True,
                    "metadata": metadata,
                    "files": files,
                    "resume_mode": resume_mode,
                }
            except Exception as exc:
                verification[0] = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        dist.broadcast_object_list(verification, src=0)
        if not verification[0]["ok"]:
            raise CheckpointIntegrityError(
                f"rank0 checkpoint verification failed: {verification[0]}"
            )
        metadata = verification[0]["metadata"]
        resume_mode = str(verification[0]["resume_mode"])
        _distributed_verify_payload(checkpoint_path, verification[0]["files"])

        options = StateDictOptions(full_state_dict=False, cpu_offload=False)
        model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
        state = {"model": model_state, "optimizer": optimizer_state}
        error: Optional[Exception] = None
        try:
            dcp.load(state, checkpoint_id=checkpoint_path / "distcp")
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
            checkpoint_path / "rank_state" / f"rank_{dist.get_rank():05d}.pt"
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
                raise CheckpointIntegrityError(f"unknown resume mode {resume_mode!r}")
        except Exception as exc:
            error = exc
        _collective_error("rank state restore", error)
        dist.barrier()
        return metadata, progress

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
        verification: list[Any] = [None]
        if dist.get_rank() == 0:
            try:
                metadata, files = self._verify_controls(checkpoint_path)
                mode = _validate_metadata(metadata, expected)
                if mode != "exact":
                    raise CheckpointIntegrityError(
                        "offline evaluation requires same-topology checkpoint"
                    )
                verification[0] = {
                    "ok": True,
                    "metadata": metadata,
                    "files": files,
                }
            except Exception as exc:
                verification[0] = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        dist.broadcast_object_list(verification, src=0)
        if not verification[0]["ok"]:
            raise CheckpointIntegrityError(
                f"rank0 checkpoint verification failed: {verification[0]}"
            )
        _distributed_verify_payload(checkpoint_path, verification[0]["files"])
        options = StateDictOptions(full_state_dict=False, cpu_offload=False)
        model_state = get_model_state_dict(model, options=options)
        error: Optional[Exception] = None
        try:
            dcp.load({"model": model_state}, checkpoint_id=checkpoint_path / "distcp")
            set_model_state_dict(model, model_state, options=options)
        except Exception as exc:
            error = exc
        _collective_error("evaluation model restore", error)
        dist.barrier()
        return dict(verification[0]["metadata"])
