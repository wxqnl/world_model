"""Immutable per-launch qualification for unified WM3D V8 jobs.

The materialized runtime and run contract describe stable training semantics.
Resource preflight receipts, rank identities, and resume sources are deliberately
launch-scoped: a later exact resume must consume a new receipt without changing
the stable run contract.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time
import uuid
from typing import Any, Mapping

from wm3d_v3.training.distributed_checkpoint import canonical_sha256, sha256_file
from wm3d_v3.training.resource_preflight import (
    ResourcePreflightError,
    validate_current_rank_identities,
    validate_resource_receipt,
)


LAUNCH_QUALIFICATION_SCHEMA = "wm3d_v8_launch_qualification_v1"
_LAUNCH_KINDS = {"fresh", "exact_resume", "topology_reshard", "eval"}
_HEX_64 = re.compile(r"[0-9a-f]{64}")


class LaunchQualificationError(RuntimeError):
    pass


def resource_contract_sha256(resources: Mapping[str, Any] | None) -> str:
    """Hash the immutable resource policy, never a volatile receipt."""

    return canonical_sha256(None if resources is None else dict(resources))


def verify_clean_runtime_checkout(repo: Path, expected_commit: str) -> str:
    """Require the exact clean repository that materialized the runtime."""

    repository = Path(repo).resolve(strict=True)
    try:
        subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        head = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        if head != expected_commit:
            raise LaunchQualificationError(
                f"runtime code commit mismatch: {head} != {expected_commit}"
            )
        dirty = subprocess.check_output(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except LaunchQualificationError:
        raise
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LaunchQualificationError("unable to verify runtime git checkout") from exc
    if dirty:
        preview = dirty.splitlines()[:8]
        raise LaunchQualificationError(
            "runtime code checkout is dirty or contains untracked files: "
            + "; ".join(preview)
        )
    return head


def build_launch_qualification(
    *,
    launch_kind: str,
    runtime_config_sha256: str,
    run_contract: Mapping[str, Any],
    resources: Mapping[str, Any] | None,
    resource_preflight: Mapping[str, Any] | None,
    rank_identities: list[Mapping[str, Any]],
    world_size: int,
    local_world_size: int,
    distributed_strategy: str,
    shard_degree: int,
    source_checkpoint: Mapping[str, Any] | None,
    created_unix_ns: int | None = None,
) -> dict[str, Any]:
    if launch_kind not in _LAUNCH_KINDS:
        raise LaunchQualificationError(f"unknown launch kind {launch_kind!r}")
    has_source = source_checkpoint is not None
    if launch_kind == "fresh" and has_source:
        raise LaunchQualificationError("fresh launch cannot bind a source checkpoint")
    if launch_kind != "fresh" and not has_source:
        raise LaunchQualificationError(f"{launch_kind} requires a source checkpoint")
    if source_checkpoint is not None:
        expected_mode = (
            "exact"
            if launch_kind in {"exact_resume", "eval"}
            else "topology_reshard"
        )
        if source_checkpoint.get("resume_mode") != expected_mode:
            raise LaunchQualificationError("source checkpoint resume mode mismatch")
    if world_size <= 0 or local_world_size <= 0 or world_size % local_world_size:
        raise LaunchQualificationError("launch topology is invalid")
    identities = [dict(value) for value in rank_identities]
    expected_ranks = list(range(world_size))
    if [value.get("rank") for value in identities] != expected_ranks:
        raise LaunchQualificationError("launch rank identity closure is invalid")
    identity_keys = {"rank", "hostname", "local_rank", "gpu_uuid"}
    if any(set(value) != identity_keys for value in identities):
        raise LaunchQualificationError("launch rank identity fields mismatch")
    if resources is None and resource_preflight is not None:
        raise LaunchQualificationError("resource-free runtime cannot bind a receipt")
    if resources is not None and resource_preflight is None:
        raise LaunchQualificationError("resource-qualified runtime requires a receipt")
    return {
        "schema": LAUNCH_QUALIFICATION_SCHEMA,
        "created_unix_ns": int(time.time_ns() if created_unix_ns is None else created_unix_ns),
        "launch_kind": launch_kind,
        "runtime_config_sha256": runtime_config_sha256,
        "run_contract_sha256": canonical_sha256(dict(run_contract)),
        "resource_contract_sha256": resource_contract_sha256(resources),
        "resource_preflight": (
            None if resource_preflight is None else dict(resource_preflight)
        ),
        "topology": {
            "world_size": int(world_size),
            "local_world_size": int(local_world_size),
            "distributed_strategy": str(distributed_strategy),
            "shard_degree": int(shard_degree),
        },
        "rank_identities": identities,
        "source_checkpoint": (
            None if source_checkpoint is None else dict(source_checkpoint)
        ),
    }


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise LaunchQualificationError(f"{label} cannot be a symlink")
    resolved = path.resolve(strict=True)
    info = os.lstat(resolved)
    if not stat.S_ISREG(info.st_mode):
        raise LaunchQualificationError(f"{label} must be a regular file")
    return resolved


def validate_launch_qualification(
    value: Mapping[str, Any],
    *,
    launch_kind: str,
    runtime_config_sha256: str,
    run_contract: Mapping[str, Any],
    resources: Mapping[str, Any] | None,
    rank_identities: list[Mapping[str, Any]],
    world_size: int,
    local_world_size: int,
    distributed_strategy: str,
    shard_degree: int,
    source_checkpoint: Mapping[str, Any] | None,
    now_unix_ns: int | None = None,
) -> None:
    required = {
        "schema",
        "created_unix_ns",
        "launch_kind",
        "runtime_config_sha256",
        "run_contract_sha256",
        "resource_contract_sha256",
        "resource_preflight",
        "topology",
        "rank_identities",
        "source_checkpoint",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise LaunchQualificationError("launch qualification fields mismatch")
    if value.get("schema") != LAUNCH_QUALIFICATION_SCHEMA:
        raise LaunchQualificationError("launch qualification schema mismatch")
    created = value.get("created_unix_ns")
    if isinstance(created, bool) or not isinstance(created, int) or created <= 0:
        raise LaunchQualificationError("launch qualification timestamp is invalid")
    expected = build_launch_qualification(
        launch_kind=launch_kind,
        runtime_config_sha256=runtime_config_sha256,
        run_contract=run_contract,
        resources=resources,
        resource_preflight=value.get("resource_preflight"),
        rank_identities=rank_identities,
        world_size=world_size,
        local_world_size=local_world_size,
        distributed_strategy=distributed_strategy,
        shard_degree=shard_degree,
        source_checkpoint=source_checkpoint,
        created_unix_ns=created,
    )
    for field in required - {"resource_preflight"}:
        if value.get(field) != expected[field]:
            raise LaunchQualificationError(
                f"launch qualification {field} mismatch"
            )

    receipt_evidence = value.get("resource_preflight")
    if resources is not None:
        if not isinstance(receipt_evidence, dict) or set(receipt_evidence) != {
            "path",
            "sha256",
            "created_unix_ns",
        }:
            raise LaunchQualificationError("resource receipt evidence is malformed")
        receipt_path = _regular_file(
            Path(str(receipt_evidence["path"])), "resource receipt"
        )
        if sha256_file(receipt_path) != receipt_evidence.get("sha256"):
            raise LaunchQualificationError("resource receipt SHA mismatch")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        try:
            observed_created = validate_resource_receipt(
                receipt,
                resources=resources,
                runtime_config_sha256=runtime_config_sha256,
                world_size=world_size,
                now_unix_ns=now_unix_ns,
            )
        except ResourcePreflightError as exc:
            raise LaunchQualificationError(
                f"resource receipt is invalid: {exc}"
            ) from exc
        if observed_created != receipt_evidence.get("created_unix_ns"):
            raise LaunchQualificationError("resource receipt timestamp mismatch")
        try:
            validate_current_rank_identities(
                receipt,
                [
                    {
                        name: identity[name]
                        for name in ("hostname", "local_rank", "gpu_uuid")
                    }
                    for identity in rank_identities
                ],
            )
        except ResourcePreflightError as exc:
            raise LaunchQualificationError(
                f"resource receipt identity is invalid: {exc}"
            ) from exc
    elif receipt_evidence is not None:
        raise LaunchQualificationError("unexpected resource receipt evidence")

    bound_source = value.get("source_checkpoint")
    if bound_source is not None:
        if not isinstance(bound_source, dict) or set(bound_source) != {
            "path",
            "step",
            "committed_sha256",
            "saved_world_size",
            "saved_shard_degree",
            "resume_mode",
        }:
            raise LaunchQualificationError("source checkpoint evidence is malformed")
        checkpoint_path = Path(str(bound_source["path"]))
        if re.fullmatch(r"step_[0-9]{8}", checkpoint_path.name) is None:
            raise LaunchQualificationError("source checkpoint is not explicitly numbered")
        commit_path = _regular_file(checkpoint_path / "COMMITTED.json", "checkpoint commit")
        if sha256_file(commit_path) != bound_source.get("committed_sha256"):
            raise LaunchQualificationError("source checkpoint COMMITTED SHA mismatch")


def _payload_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def publish_launch_qualification(
    root: Path, value: Mapping[str, Any]
) -> tuple[Path, str]:
    payload = _payload_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    created = int(value["created_unix_ns"])
    directory = Path(root) / "launch_qualifications"
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise LaunchQualificationError("launch qualification root cannot be a symlink")
    path = directory / f"launch_{created}_{digest}.json"
    temporary = directory / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise LaunchQualificationError(
            f"refusing to overwrite launch qualification {path}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return path, digest


def load_published_launch_qualification(path: Path, expected_sha256: str) -> dict[str, Any]:
    if _HEX_64.fullmatch(expected_sha256) is None:
        raise LaunchQualificationError("launch qualification expected SHA is invalid")
    resolved = _regular_file(Path(path), "launch qualification")
    if sha256_file(resolved) != expected_sha256:
        raise LaunchQualificationError("launch qualification SHA mismatch")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LaunchQualificationError("launch qualification payload is not an object")
    return value
