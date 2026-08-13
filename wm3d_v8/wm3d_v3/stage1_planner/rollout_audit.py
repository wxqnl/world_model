"""Strict, fail-closed contract for audited real RoboCasa rollouts."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any
import uuid

from wm3d_v3.data.manifest_contract import SHA256_RE, canonical_sha256


ROLLOUT_AUDIT_SCHEMA = "wm3d_v8_robocasa_real_rollout_audit_v3"
ROLLOUT_AUDIT_FIELDS = {
    "schema", "code_commit", "runtime_root",
    "launch_receipt_path", "launch_receipt_sha256",
    "runtime_generator_path", "runtime_generator_sha256",
    "replay_helper_path", "replay_helper_sha256",
    "action_audit_path", "action_audit_sha256",
    "candidate_index_path", "candidate_index_sha256",
    "candidate_index_seal_path", "candidate_index_seal_sha256",
    "replay_authority_path", "replay_authority_sha256",
    "selection_manifest_path", "selection_manifest_sha256",
    "source_roots", "source_metadata_sha256",
    "simulator_revision", "simulator_revision_sha256", "camera_order",
    "simulator_action_order", "source_action_order",
    "simulator_action_period_seconds", "selection_count", "rows",
    "rows_sha256", "passed",
}
ROLLOUT_AUDIT_ROW_FIELDS = {
    "split", "source", "root_id", "episode_id", "t0", "task_text",
    "runtime_payload_path", "runtime_payload_sha256",
    "runtime_index_shard_path", "runtime_index_shard_sha256",
    "runtime_index_row_sha256", "root_context_path", "root_context_sha256",
    "root_state_sha256", "model_xml_sha256", "ep_meta_sha256",
    "candidate_seed", "candidate_payload_sha256",
    "candidate_index_row_sha256", "source_action_slice_sha256",
    "factual_simulator_action_source_byte_exact",
    "candidate_actions_executed_exact", "real_simulator_outcomes",
    "future_observation_leakage", "outcome_indices",
    "future_offsets_seconds", "branch_rgb_indices",
    "source_future_row_offsets", "candidate_count",
    "replay_authority_row_sha256",
}
_ROW_TRUE_GATES = {
    "factual_simulator_action_source_byte_exact",
    "candidate_actions_executed_exact",
    "real_simulator_outcomes",
}
_SPLITS = {"train", "val", "test"}
_SIMULATOR_REVISION_FIELDS = {
    "source_repo", "source_revision", "robocasa_commit",
    "robocasa_dataset_version", "robosuite_version", "robosuite_commit",
    "mujoco_version",
}
_SIMULATOR_ACTION_ORDER = [
    "eef_position3", "eef_rotation3", "gripper_close1", "base_motion4",
    "control_mode1",
]
_SOURCE_ACTION_ORDER = [
    "base_motion4", "control_mode1", "eef_position3", "eef_rotation3",
    "gripper_close1",
]


class RolloutAuditError(ValueError):
    pass


_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _directory_identity(value: os.stat_result) -> tuple[int, int]:
    return int(value.st_dev), int(value.st_ino)


def _open_absolute_directory(
    path: Path,
    *,
    create: bool,
    label: str,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Open an absolute directory chain without ever following a symlink.

    Each child is opened relative to the already-open parent.  The returned
    descriptor therefore keeps the selected directory pinned even if a path
    component is concurrently renamed or replaced.
    """
    absolute = _absolute_path(path)
    if not absolute.is_absolute():
        raise RolloutAuditError(f"{label} must be absolute: {path}")
    try:
        descriptor = os.open("/", _DIRECTORY_FLAGS)
    except OSError as error:  # pragma: no cover - an unusable host filesystem
        raise RolloutAuditError("cannot open filesystem root") from error
    identities = [_directory_identity(os.fstat(descriptor))]
    try:
        for component in absolute.parts[1:]:
            if not component or component in {".", ".."}:
                raise RolloutAuditError(f"{label} has an invalid path component")
            if create:
                try:
                    os.mkdir(component, mode=0o750, dir_fd=descriptor)
                except FileExistsError:
                    pass
                else:
                    os.fsync(descriptor)
            try:
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except OSError as error:
                try:
                    entry = os.stat(
                        component, dir_fd=descriptor, follow_symlinks=False
                    )
                except OSError:
                    entry = None
                if entry is not None and stat.S_ISLNK(entry.st_mode):
                    raise RolloutAuditError(
                        f"{label} must not be a symlink or use a symlink "
                        f"ancestor: {absolute}"
                    ) from error
                raise RolloutAuditError(
                    f"{label} contains a symlink, missing, or non-directory "
                    f"ancestor: {absolute}"
                ) from error
            child_stat = os.fstat(child)
            if not stat.S_ISDIR(child_stat.st_mode):
                os.close(child)
                raise RolloutAuditError(
                    f"{label} contains a non-directory ancestor: {absolute}"
                )
            os.close(descriptor)
            descriptor = child
            identities.append(_directory_identity(child_stat))
        return descriptor, tuple(identities)
    except BaseException:
        os.close(descriptor)
        raise


def _reopen_matching_directory(
    path: Path,
    expected: tuple[tuple[int, int], ...],
    *,
    label: str,
) -> int:
    descriptor, observed = _open_absolute_directory(
        path, create=False, label=label
    )
    if observed != expected:
        os.close(descriptor)
        raise RolloutAuditError(
            f"{label} ancestor was replaced while the path was in use: {path}"
        )
    return descriptor


def _read_regular_at(
    parent_descriptor: int,
    filename: str,
    *,
    label: str,
) -> tuple[bytes, str, os.stat_result]:
    try:
        descriptor = os.open(filename, _READ_FLAGS, dir_fd=parent_descriptor)
    except OSError as error:
        try:
            entry = os.stat(
                filename, dir_fd=parent_descriptor, follow_symlinks=False
            )
        except OSError:
            entry = None
        if entry is not None and stat.S_ISLNK(entry.st_mode):
            raise RolloutAuditError(
                f"{label} must not be a symlink: {filename}"
            ) from error
        raise RolloutAuditError(f"cannot open {label}: {filename}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RolloutAuditError(f"{label} must be a regular file: {filename}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 16 << 20)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise RolloutAuditError(f"{label} changed while it was read: {filename}")
        try:
            pinned = os.stat(
                filename, dir_fd=parent_descriptor, follow_symlinks=False
            )
        except OSError as error:
            raise RolloutAuditError(
                f"{label} was removed while it was read: {filename}"
            ) from error
        if (
            pinned.st_dev != before.st_dev
            or pinned.st_ino != before.st_ino
            or not stat.S_ISREG(pinned.st_mode)
        ):
            raise RolloutAuditError(
                f"{label} was replaced while it was read: {filename}"
            )
        return b"".join(chunks), digest.hexdigest(), before
    finally:
        os.close(descriptor)


class TrustedOutputRoot:
    """A pinned, no-symlink output namespace for mkdir/read/publish operations."""

    def __init__(self, path: Path, *, label: str = "trusted output root"):
        self.path = _absolute_path(path)
        self.label = label
        self._descriptor, self._identities = _open_absolute_directory(
            self.path, create=True, label=label
        )
        self._closed = False
        self._verify_namespace()

    def __enter__(self) -> TrustedOutputRoot:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError):
            pass

    def close(self) -> None:
        if not self._closed:
            os.close(self._descriptor)
            self._closed = True

    def _verify_namespace(self) -> None:
        if self._closed:
            raise RolloutAuditError(f"{self.label} descriptor is closed")
        current = _reopen_matching_directory(
            self.path, self._identities, label=self.label
        )
        os.close(current)

    def _relative_parts(self, target: Path) -> tuple[Path, tuple[str, ...]]:
        absolute = _absolute_path(target)
        try:
            relative = absolute.relative_to(self.path)
        except ValueError as error:
            raise RolloutAuditError(
                f"output path escapes {self.label}: {absolute}"
            ) from error
        parts = tuple(relative.parts)
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise RolloutAuditError(f"output path is invalid: {absolute}")
        return absolute, parts

    def _open_relative_directory(
        self,
        parts: tuple[str, ...],
        *,
        create: bool,
        label: str,
    ) -> tuple[int, tuple[tuple[int, int], ...]]:
        descriptor = os.dup(self._descriptor)
        identities = [_directory_identity(os.fstat(descriptor))]
        try:
            for component in parts:
                if create:
                    try:
                        os.mkdir(component, mode=0o750, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                    else:
                        os.fsync(descriptor)
                try:
                    child = os.open(
                        component, _DIRECTORY_FLAGS, dir_fd=descriptor
                    )
                except OSError as error:
                    try:
                        entry = os.stat(
                            component,
                            dir_fd=descriptor,
                            follow_symlinks=False,
                        )
                    except OSError:
                        entry = None
                    if entry is not None and stat.S_ISLNK(entry.st_mode):
                        raise RolloutAuditError(
                            f"{label} must not be a symlink or use a symlink "
                            f"component: {component}"
                        ) from error
                    raise RolloutAuditError(
                        f"{label} contains a symlink, missing, or non-directory "
                        f"component: {component}"
                    ) from error
                os.close(descriptor)
                descriptor = child
                identities.append(_directory_identity(os.fstat(child)))
            return descriptor, tuple(identities)
        except BaseException:
            os.close(descriptor)
            raise

    def _reopen_matching_relative_directory(
        self,
        parts: tuple[str, ...],
        expected: tuple[tuple[int, int], ...],
        *,
        label: str,
    ) -> int:
        descriptor, observed = self._open_relative_directory(
            parts, create=False, label=label
        )
        if observed != expected:
            os.close(descriptor)
            raise RolloutAuditError(
                f"{label} ancestor was replaced while the path was in use"
            )
        return descriptor

    def mkdir(self, target: Path, *, label: str) -> Path:
        absolute, parts = self._relative_parts(target)
        self._verify_namespace()
        descriptor, identities = self._open_relative_directory(
            parts, create=True, label=label
        )
        os.close(descriptor)
        self._verify_namespace()
        current = self._reopen_matching_relative_directory(
            parts, identities, label=label
        )
        os.close(current)
        return absolute

    def pin_directory(
        self, target: Path, *, label: str
    ) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
        """Create and return a namespace identity that can span a subprocess."""
        _absolute, parts = self._relative_parts(target)
        self._verify_namespace()
        descriptor, identities = self._open_relative_directory(
            parts, create=True, label=label
        )
        os.close(descriptor)
        return parts, identities

    def verify_pinned_directory(
        self,
        pin: tuple[tuple[str, ...], tuple[tuple[int, int], ...]],
        *,
        label: str,
    ) -> None:
        """Fail if a pinned child directory or any ancestor was replaced."""
        parts, identities = pin
        self._verify_namespace()
        descriptor = self._reopen_matching_relative_directory(
            parts, identities, label=label
        )
        os.close(descriptor)

    def read(self, target: Path, *, label: str) -> tuple[Path, bytes, str]:
        absolute, parts = self._relative_parts(target)
        self._verify_namespace()
        parent, parent_identities = self._open_relative_directory(
            parts[:-1], create=False, label=label
        )
        try:
            payload, digest, before = _read_regular_at(
                parent, parts[-1], label=label
            )
        finally:
            os.close(parent)
        self._verify_namespace()
        current_parent = self._reopen_matching_relative_directory(
            parts[:-1], parent_identities, label=label
        )
        try:
            current = os.stat(
                parts[-1], dir_fd=current_parent, follow_symlinks=False
            )
            if (
                current.st_dev != before.st_dev
                or current.st_ino != before.st_ino
                or not stat.S_ISREG(current.st_mode)
            ):
                raise RolloutAuditError(
                    f"{label} was replaced while it was read: {absolute}"
                )
        except OSError as error:
            raise RolloutAuditError(
                f"{label} path changed while it was read: {absolute}"
            ) from error
        finally:
            os.close(current_parent)
        return absolute, payload, digest

    def publish(
        self,
        target: Path,
        payload: bytes,
        *,
        label: str,
        mode: int = 0o640,
    ) -> Path:
        if type(mode) is not int or mode < 0 or mode & ~0o777:
            raise RolloutAuditError(f"invalid publish mode for {label}")
        absolute, parts = self._relative_parts(target)
        self._verify_namespace()
        parent, parent_identities = self._open_relative_directory(
            parts[:-1], create=True, label=label
        )
        temporary = f".{parts[-1]}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        temporary_created = False
        try:
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(temporary, flags, mode, dir_fd=parent)
            temporary_created = True
            try:
                os.fchmod(descriptor, mode)
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:  # pragma: no cover - kernel contract
                        raise RolloutAuditError(f"short write for {label}")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            try:
                os.link(
                    temporary,
                    parts[-1],
                    src_dir_fd=parent,
                    dst_dir_fd=parent,
                    follow_symlinks=False,
                )
            except FileExistsError:
                try:
                    existing_stat = os.stat(
                        parts[-1], dir_fd=parent, follow_symlinks=False
                    )
                except OSError as error:
                    raise FileExistsError(
                        f"cannot inspect existing {label}: {absolute}"
                    ) from error
                if stat.S_ISLNK(existing_stat.st_mode):
                    raise FileExistsError(
                        f"refusing to overwrite {label} symlink: {absolute}"
                    )
                existing, _digest, _before = _read_regular_at(
                    parent, parts[-1], label=label
                )
                if existing != payload:
                    raise
            os.fsync(parent)
        finally:
            if temporary_created:
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass
            os.close(parent)
        self._verify_namespace()
        current = self._reopen_matching_relative_directory(
            parts[:-1], parent_identities, label=label
        )
        os.close(current)
        return absolute


def _sha(value: object, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise RolloutAuditError(f"{label} must be a lowercase SHA256 string")
    return value


def read_regular_bytes(path: Path, label: str) -> tuple[Path, bytes, str]:
    """Read/hash one regular file via a pinned, no-symlink ancestor chain."""
    absolute = _absolute_path(path)
    parent, identities = _open_absolute_directory(
        absolute.parent, create=False, label=label
    )
    try:
        payload, digest, before = _read_regular_at(
            parent, absolute.name, label=label
        )
    finally:
        os.close(parent)
    current_parent = _reopen_matching_directory(
        absolute.parent, identities, label=label
    )
    try:
        current = os.stat(
            absolute.name, dir_fd=current_parent, follow_symlinks=False
        )
        if (
            current.st_dev != before.st_dev
            or current.st_ino != before.st_ino
            or not stat.S_ISREG(current.st_mode)
        ):
            raise RolloutAuditError(
                f"{label} was replaced while it was read: {absolute}"
            )
    except OSError as error:
        raise RolloutAuditError(
            f"{label} path changed while it was read: {absolute}"
        ) from error
    finally:
        os.close(current_parent)
    return absolute, payload, digest


def verify_regular_sha(path: Path, expected: object, label: str) -> Path:
    resolved, _payload, observed = read_regular_bytes(path, label)
    if observed != _sha(expected, f"{label} SHA"):
        raise RolloutAuditError(f"{label} SHA mismatch: {observed} != {expected}")
    return resolved


def publish_no_clobber(path: Path, payload: bytes, label: str) -> Path:
    """Atomically publish via a pinned parent; never follow an ancestor symlink."""
    absolute = _absolute_path(path)
    with TrustedOutputRoot(absolute.parent, label=f"{label} parent") as output:
        return output.publish(absolute, payload, label=label)


def validate_rollout_audit(
    audit: object,
    *,
    expected_code_commit: str,
    verify_referents: bool = True,
) -> dict[str, Any]:
    if not isinstance(audit, dict) or set(audit) != ROLLOUT_AUDIT_FIELDS:
        raise RolloutAuditError("rollout audit top-level fields mismatch")
    if audit["schema"] != ROLLOUT_AUDIT_SCHEMA or audit["passed"] is not True:
        raise RolloutAuditError("rollout audit schema/pass mismatch")
    if audit["code_commit"] != expected_code_commit:
        raise RolloutAuditError("rollout audit code commit differs from Stage0 runtime")
    from wm3d_v3.stage1_planner.replay_authority import load_replay_authority

    authority_sha = _sha(
        audit["replay_authority_sha256"], "replay authority SHA"
    )
    authority_rows: dict[str, dict[str, Any]] = {}
    if verify_referents:
        authority, observed_authority_sha = load_replay_authority(
            Path(audit["replay_authority_path"]),
            expected_code_commit=expected_code_commit,
        )
        if observed_authority_sha != authority_sha:
            raise RolloutAuditError("replay authority SHA mismatch")
        if (
            audit["selection_manifest_path"]
            != authority["selection_manifest_path"]
            or audit["selection_manifest_sha256"]
            != authority["selection_manifest_sha256"]
        ):
            raise RolloutAuditError("rollout audit selection manifest mismatch")
        authority_rows = {row["root_id"]: row for row in authority["rows"]}
    _sha(audit["selection_manifest_sha256"], "selection manifest SHA")
    runtime_root = Path(str(audit["runtime_root"]))
    if runtime_root.is_symlink() or not runtime_root.is_dir():
        raise RolloutAuditError("rollout audit runtime root is invalid")
    for name in (
        "launch_receipt", "runtime_generator", "replay_helper", "action_audit",
        "candidate_index", "candidate_index_seal",
    ):
        _sha(audit[f"{name}_sha256"], f"{name} SHA")
        if verify_referents:
            verify_regular_sha(
                Path(audit[f"{name}_path"]), audit[f"{name}_sha256"], name
            )
    source_roots = audit["source_roots"]
    source_metadata = audit["source_metadata_sha256"]
    if (
        not isinstance(source_roots, dict)
        or not source_roots
        or not isinstance(source_metadata, dict)
        or set(source_metadata) != set(source_roots)
    ):
        raise RolloutAuditError("rollout audit source closure mismatch")
    for source, root_value in source_roots.items():
        if not isinstance(source, str) or not source:
            raise RolloutAuditError("rollout audit source name is invalid")
        root = Path(str(root_value))
        if root.is_symlink() or not root.is_dir():
            raise RolloutAuditError(f"rollout audit source root is invalid: {root}")
        metadata = source_metadata[source]
        expected_metadata = {"info.json", "modality.json", "embodiment.json", "episodes.jsonl"}
        if not isinstance(metadata, dict) or set(metadata) != expected_metadata:
            raise RolloutAuditError("rollout audit source metadata fields mismatch")
        for filename, expected in metadata.items():
            _sha(expected, f"{source} {filename} SHA")
            if verify_referents:
                verify_regular_sha(
                    root / "meta" / filename, expected, f"{source} {filename}"
                )
    revision = audit["simulator_revision"]
    if (
        not isinstance(revision, dict)
        or set(revision) != _SIMULATOR_REVISION_FIELDS
        or any(not isinstance(value, str) or not value for value in revision.values())
    ):
        raise RolloutAuditError("rollout audit simulator revision fields are invalid")
    if canonical_sha256(revision) != _sha(
        audit["simulator_revision_sha256"], "simulator revision SHA"
    ):
        raise RolloutAuditError("rollout audit simulator revision SHA mismatch")
    cameras = audit["camera_order"]
    if (
        not isinstance(cameras, list)
        or not cameras
        or any(not isinstance(value, str) or not value for value in cameras)
        or len(set(cameras)) != len(cameras)
        or audit["simulator_action_order"] != _SIMULATOR_ACTION_ORDER
        or audit["source_action_order"] != _SOURCE_ACTION_ORDER
    ):
        raise RolloutAuditError("rollout audit camera/action ordering is invalid")
    counts = audit["selection_count"]
    rows = audit["rows"]
    if (
        not isinstance(counts, dict)
        or set(counts) != _SPLITS
        or any(type(counts[split]) is not int or counts[split] <= 0 for split in _SPLITS)
        or not isinstance(rows, list)
        or not rows
        or sum(counts.values()) != len(rows)
    ):
        raise RolloutAuditError("rollout audit selection counts are invalid")
    if canonical_sha256(rows) != _sha(audit["rows_sha256"], "rollout rows SHA"):
        raise RolloutAuditError("rollout audit rows SHA mismatch")
    observed_counts = {split: 0 for split in _SPLITS}
    seen_roots: set[str] = set()
    for row_number, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != ROLLOUT_AUDIT_ROW_FIELDS:
            raise RolloutAuditError(f"rollout audit row {row_number} fields mismatch")
        split = row["split"]
        source = row["source"]
        if split not in _SPLITS or source not in source_roots:
            raise RolloutAuditError(f"rollout audit row {row_number} split/source invalid")
        if not isinstance(row["task_text"], str) or not row["task_text"]:
            raise RolloutAuditError(f"rollout audit row {row_number} task is invalid")
        observed_counts[split] += 1
        root_id = _sha(row["root_id"], f"row {row_number} root id")
        if root_id in seen_roots:
            raise RolloutAuditError("rollout audit contains duplicate roots")
        seen_roots.add(root_id)
        for field in (
            "runtime_payload_sha256", "runtime_index_shard_sha256",
            "runtime_index_row_sha256", "root_context_sha256", "root_state_sha256",
            "model_xml_sha256", "ep_meta_sha256", "candidate_payload_sha256",
            "candidate_index_row_sha256", "source_action_slice_sha256",
            "replay_authority_row_sha256",
        ):
            _sha(row[field], f"row {row_number} {field}")
        if any(row[field] is not True for field in _ROW_TRUE_GATES):
            raise RolloutAuditError(f"rollout audit row {row_number} gate failed")
        if row["future_observation_leakage"] is not False:
            raise RolloutAuditError(f"rollout audit row {row_number} leaks future observations")
        for field in ("episode_id", "t0", "candidate_seed", "candidate_count"):
            if type(row[field]) is not int or row[field] < 0:
                raise RolloutAuditError(f"rollout audit row {row_number} {field} invalid")
        if row["candidate_count"] < 2:
            raise RolloutAuditError("rollout audit requires at least two candidates")
        sequence_fields = (
            "outcome_indices", "future_offsets_seconds", "branch_rgb_indices",
            "source_future_row_offsets",
        )
        sequences = [row[field] for field in sequence_fields]
        if any(not isinstance(value, list) or not value for value in sequences):
            raise RolloutAuditError(f"rollout audit row {row_number} offsets invalid")
        if len({len(value) for value in sequences}) != 1:
            raise RolloutAuditError(f"rollout audit row {row_number} offset lengths differ")
        if any(
            type(value) is not int or value < 0
            for field in ("outcome_indices", "branch_rgb_indices", "source_future_row_offsets")
            for value in row[field]
        ):
            raise RolloutAuditError(f"rollout audit row {row_number} integer offsets invalid")
        for field in ("outcome_indices", "branch_rgb_indices", "source_future_row_offsets"):
            values = row[field]
            if any(right <= left for left, right in zip(values, values[1:])):
                raise RolloutAuditError(
                    f"rollout audit row {row_number} integer offsets are not increasing"
                )
        future = row["future_offsets_seconds"]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in future
        ) or any(right <= left for left, right in zip(future, future[1:])):
            raise RolloutAuditError(f"rollout audit row {row_number} future offsets invalid")
        for field in ("runtime_payload", "runtime_index_shard", "root_context"):
            if verify_referents:
                verify_regular_sha(
                    Path(row[f"{field}_path"]), row[f"{field}_sha256"],
                    f"row {row_number} {field}",
                )
        if verify_referents:
            authority_row = authority_rows.get(root_id)
            if authority_row is None:
                raise RolloutAuditError(
                    f"rollout audit row {row_number} lacks replay authority"
                )
            if canonical_sha256(authority_row) != row["replay_authority_row_sha256"]:
                raise RolloutAuditError(
                    f"rollout audit row {row_number} replay authority row mismatch"
                )
            authority_bindings = {
                "split": split,
                "root_id": root_id,
                "episode_id": row["episode_id"],
                "t0": row["t0"],
                "candidate_seed": row["candidate_seed"],
                "candidate_payload_sha256": row["candidate_payload_sha256"],
                "root_context_sha256": row["root_context_sha256"],
                "legacy_runtime_payload_sha256": row["runtime_payload_sha256"],
                "legacy_runtime_index_shard_sha256": row[
                    "runtime_index_shard_sha256"
                ],
                "legacy_runtime_index_row_sha256": row[
                    "runtime_index_row_sha256"
                ],
                "candidate_count": row["candidate_count"],
            }
            if any(
                authority_row.get(name) != expected
                for name, expected in authority_bindings.items()
            ):
                raise RolloutAuditError(
                    f"rollout audit row {row_number} differs from replay authority"
                )
    if observed_counts != counts:
        raise RolloutAuditError("rollout audit row split counts mismatch")
    if verify_referents and set(authority_rows) != seen_roots:
        raise RolloutAuditError("rollout audit/replay authority coverage mismatch")
    period = audit["simulator_action_period_seconds"]
    if (
        isinstance(period, bool)
        or not isinstance(period, (int, float))
        or not math.isfinite(period)
        or period <= 0
    ):
        raise RolloutAuditError("rollout audit simulator cadence is invalid")
    return audit


def load_rollout_audit(
    path: Path,
    *,
    expected_code_commit: str,
) -> tuple[dict[str, Any], str]:
    _resolved, payload, digest = read_regular_bytes(path, "rollout audit")
    try:
        audit = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RolloutAuditError("rollout audit is not valid JSON") from error
    return validate_rollout_audit(
        audit,
        expected_code_commit=expected_code_commit,
        verify_referents=True,
    ), digest
