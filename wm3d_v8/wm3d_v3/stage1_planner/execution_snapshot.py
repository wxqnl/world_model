"""Private, copy-only execution snapshots for Stage1 simulator replay.

The snapshot is fully materialized before a child starts.  The child receives
only snapshot paths; original paths remain provenance and are never used as a
pre/post mutation detector.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any

from wm3d_v3.data.manifest_contract import SHA256_RE, canonical_sha256
from wm3d_v3.stage1_planner.rollout_audit import (
    RolloutAuditError,
    TrustedOutputRoot,
    read_regular_bytes,
)


EXECUTION_SNAPSHOT_SCHEMA = "wm3d_v8_robocasa_stage1_execution_snapshot_v1"
EXECUTION_SNAPSHOT_FIELDS = {
    "schema", "root", "copy_method", "required_bytes", "available_bytes",
    "safety_margin_bytes", "rows", "rows_sha256", "file_count",
    "total_bytes", "passed",
}
EXECUTION_SNAPSHOT_ROW_FIELDS = {
    "kind", "provenance_path", "provenance_resolved_path",
    "provenance_sha256", "snapshot_path", "snapshot_sha256", "size",
    "mode", "copy_method",
}
_DEFAULT_SAFETY_MARGIN = 1 << 30


class ExecutionSnapshotError(RolloutAuditError):
    pass


class PinnedExecutionPath:
    """An inherited descriptor alias immune to path rename/replacement."""

    def __init__(self, path: Path, *, directory: bool, label: str):
        self.path = _absolute(path)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        if directory:
            flags |= getattr(os, "O_DIRECTORY", 0)
        try:
            self.fd = os.open(self.path, flags)
        except OSError as error:
            raise ExecutionSnapshotError(f"cannot pin {label}: {self.path}") from error
        observed = os.fstat(self.fd)
        expected = stat.S_ISDIR if directory else stat.S_ISREG
        if not expected(observed.st_mode):
            os.close(self.fd)
            raise ExecutionSnapshotError(f"{label} has the wrong file type")
        self.directory = directory
        self.label = label

    def __enter__(self) -> PinnedExecutionPath:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError):
            pass

    def alias(self, relative: str | Path | None = None) -> str:
        if self.fd < 0:
            raise ExecutionSnapshotError(f"{self.label} anchor is closed")
        base = f"/proc/self/fd/{self.fd}"
        if relative is None:
            return base
        if not self.directory:
            raise ExecutionSnapshotError("a regular-file anchor has no children")
        parts = Path(relative).parts
        if not parts or Path(relative).is_absolute() or ".." in parts:
            raise ExecutionSnapshotError("execution alias relative path is invalid")
        return f"{base}/{Path(relative).as_posix()}"

    def read_regular(self, relative: str | Path, *, label: str) -> tuple[bytes, str]:
        """Read a child without resolving the snapshot's mutable named path."""
        if self.fd < 0:
            raise ExecutionSnapshotError(f"{self.label} anchor is closed")
        if not self.directory:
            raise ExecutionSnapshotError("a regular-file anchor has no children")
        relative_path = Path(relative)
        parts = relative_path.parts
        if (
            not parts
            or relative_path.is_absolute()
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise ExecutionSnapshotError(f"{label} relative path is invalid")
        parent = os.dup(self.fd)
        try:
            for component in parts[:-1]:
                child = os.open(
                    component,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent,
                )
                os.close(parent)
                parent = child
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent,
            )
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise ExecutionSnapshotError(f"{label} must be a regular file")
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
                    raise ExecutionSnapshotError(f"{label} changed while read")
                current = os.stat(
                    parts[-1], dir_fd=parent, follow_symlinks=False
                )
                if (
                    current.st_dev != before.st_dev
                    or current.st_ino != before.st_ino
                    or not stat.S_ISREG(current.st_mode)
                ):
                    raise ExecutionSnapshotError(f"{label} was replaced while read")
                return b"".join(chunks), digest.hexdigest()
            finally:
                os.close(descriptor)
        except OSError as error:
            raise ExecutionSnapshotError(f"cannot read {label}") from error
        finally:
            os.close(parent)


class ReadOnlyBindMount:
    """Keep one private input subtree immutable while a child consumes it."""

    def __init__(self, path: Path, *, label: str):
        self.path = _absolute(path)
        self.label = label
        self._mounted = False
        self._mount_identity: tuple[int, int] | None = None
        self._mount_id: int | None = None
        if self.path.is_symlink() or not self.path.is_dir():
            raise ExecutionSnapshotError(f"{label} must be a real directory")

    def __enter__(self) -> ReadOnlyBindMount:
        source = PinnedExecutionPath(
            self.path, directory=True, label=f"{self.label} source"
        )
        try:
            subprocess.run(
                ["mount", "--bind", source.alias(), str(self.path)],
                check=True,
                capture_output=True,
                text=True,
                pass_fds=(source.fd,),
            )
            self._mounted = True
            subprocess.run(
                ["mount", "-o", "remount,bind,ro", str(self.path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            if self._mounted:
                subprocess.run(
                    ["umount", str(self.path)], capture_output=True, text=True
                )
                self._mounted = False
            raise ExecutionSnapshotError(
                f"cannot seal {self.label} read-only"
            ) from error
        finally:
            source.close()
        try:
            metadata = os.stat(self.path, follow_symlinks=False)
            self._mount_identity = (metadata.st_dev, metadata.st_ino)
            self._mount_id = self._mount_info(str(self.path))[0]
            self.verify(target=str(self.path))
        except Exception:
            subprocess.run(
                ["umount", str(self.path)], capture_output=True, text=True
            )
            self._mounted = False
            self._mount_identity = None
            self._mount_id = None
            raise
        return self

    def _mount_info(
        self,
        target: str,
        *,
        pass_fds: tuple[int, ...] = (),
    ) -> tuple[int, set[str]]:
        try:
            result = subprocess.run(
                [
                    "findmnt", "--first-only", "-J", "-o", "ID,OPTIONS",
                    "--target", target,
                ],
                check=True,
                capture_output=True,
                text=True,
                pass_fds=pass_fds,
            )
            payload = json.loads(result.stdout)
            filesystems = payload["filesystems"]
            if len(filesystems) != 1:
                raise ValueError("ambiguous mount result")
            row = filesystems[0]
            return int(row["id"]), set(str(row["options"]).split(","))
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            subprocess.CalledProcessError,
        ) as error:
            raise ExecutionSnapshotError(
                f"cannot inspect {self.label} mount"
            ) from error

    def verify(
        self,
        *,
        target: str | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> None:
        if (
            not self._mounted
            or self._mount_identity is None
            or self._mount_id is None
        ):
            raise ExecutionSnapshotError(f"{self.label} mount is not active")
        inspected_target = target or str(self.path)
        metadata = os.stat(inspected_target, follow_symlinks=False)
        if (metadata.st_dev, metadata.st_ino) != self._mount_identity:
            raise ExecutionSnapshotError(f"{self.label} mount identity changed")
        mount_id, options = self._mount_info(
            inspected_target,
            pass_fds=pass_fds,
        )
        if mount_id != self._mount_id:
            raise ExecutionSnapshotError(f"{self.label} mount identity changed")
        if "ro" not in options:
            raise ExecutionSnapshotError(f"{self.label} mount is not read-only")

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(
        self,
        *,
        target: str | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> None:
        if not self._mounted:
            return
        result = subprocess.run(
            ["umount", target or str(self.path)],
            capture_output=True,
            text=True,
            pass_fds=pass_fds,
        )
        if result.returncode != 0:
            raise ExecutionSnapshotError(
                f"cannot unmount {self.label}: {result.stderr.strip()}"
            )
        self._mounted = False
        self._mount_identity = None
        self._mount_id = None


class WritableBindMount:
    """Temporarily re-enable writes for one directory inside sealed inputs."""

    def __init__(self, path: Path, *, label: str):
        self.path = _absolute(path)
        self.label = label
        self._mounted = False
        self._source: PinnedExecutionPath | None = None
        self._mount_id: int | None = None
        self._baseline: dict[str, str] | None = None
        if self.path.is_symlink() or not self.path.is_dir():
            raise ExecutionSnapshotError(f"{label} must be a real directory")

    @staticmethod
    def _files(path: Path) -> dict[str, str]:
        result: dict[str, str] = {}
        for item in sorted(
            path.rglob("*"), key=lambda value: value.relative_to(path).as_posix()
        ):
            metadata = item.lstat()
            relative = item.relative_to(path).as_posix()
            if stat.S_ISDIR(metadata.st_mode):
                result[relative + "/"] = "directory"
                continue
            if item.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise ExecutionSnapshotError(
                    "writable replay directory contains a non-regular entry"
                )
            payload = item.read_bytes()
            result[relative] = hashlib.sha256(payload).hexdigest()
        return result

    def __enter__(self) -> WritableBindMount:
        self._baseline = self._files(self.path)
        self._source = PinnedExecutionPath(
            self.path, directory=True, label=f"{self.label} source"
        )
        target = self._source.alias()
        try:
            subprocess.run(
                ["mount", "--bind", target, target],
                check=True,
                capture_output=True,
                text=True,
                pass_fds=(self._source.fd,),
            )
            self._mounted = True
            subprocess.run(
                ["mount", "-o", "remount,bind,rw,nodev,nosuid,noexec", target],
                check=True,
                capture_output=True,
                text=True,
                pass_fds=(self._source.fd,),
            )
            self._mount_id, options = ReadOnlyBindMount._mount_info(
                self, target, pass_fds=(self._source.fd,)
            )
            if not {"rw", "nodev", "nosuid", "noexec"}.issubset(options):
                raise ExecutionSnapshotError(f"{self.label} writable mount is unsafe")
        except BaseException:
            if self._mounted:
                subprocess.run(
                    ["umount", target],
                    capture_output=True,
                    text=True,
                    pass_fds=(self._source.fd,),
                )
            self._mounted = False
            self._source.close()
            self._source = None
            raise
        return self

    def verify(self) -> None:
        if not self._mounted or self._source is None or self._mount_id is None:
            raise ExecutionSnapshotError(f"{self.label} writable mount is not active")
        mount_id, options = ReadOnlyBindMount._mount_info(
            self, self._source.alias(), pass_fds=(self._source.fd,)
        )
        if (
            mount_id != self._mount_id
            or not {"rw", "nodev", "nosuid", "noexec"}.issubset(options)
        ):
            raise ExecutionSnapshotError(f"{self.label} writable mount changed")

    def close(self) -> None:
        if not self._mounted:
            return
        validation_error: BaseException | None = None
        try:
            self.verify()
            if self._files(self.path) != self._baseline:
                raise ExecutionSnapshotError(
                    f"{self.label} was not restored after replay"
                )
        except BaseException as error:
            validation_error = error
        assert self._source is not None
        result = subprocess.run(
            ["umount", self._source.alias()],
            capture_output=True,
            text=True,
            pass_fds=(self._source.fd,),
        )
        if result.returncode != 0:
            raise ExecutionSnapshotError(
                f"cannot unmount {self.label}: {result.stderr.strip()}"
            )
        self._mounted = False
        self._mount_id = None
        self._source.close()
        self._source = None
        if validation_error is not None:
            raise validation_error

    def __exit__(self, *_args: object) -> None:
        self.close()


def enter_private_mount_namespace() -> None:
    """Re-exec this authority command in a non-propagating mount namespace."""
    marker = "WM3D_STAGE1_REPLAY_PRIVATE_MOUNT_NAMESPACE"
    if os.environ.get(marker) == "1":
        current = os.stat("/proc/self/ns/mnt").st_ino
        parent = os.stat(f"/proc/{os.getppid()}/ns/mnt").st_ino
        if current == parent:
            raise ExecutionSnapshotError(
                "replay mount-namespace marker was set outside a private namespace"
            )
        try:
            subprocess.run(
                ["mount", "--make-rprivate", "/"],
                check=True,
                capture_output=True,
                text=True,
            )
            mountinfo = Path("/proc/self/mountinfo").read_text().splitlines()
        except (OSError, subprocess.CalledProcessError) as error:
            raise ExecutionSnapshotError(
                "cannot make the replay mount namespace private"
            ) from error
        for line in mountinfo:
            fields = line.split()
            try:
                separator = fields.index("-")
            except ValueError as error:
                raise ExecutionSnapshotError("invalid mountinfo entry") from error
            optional = fields[6:separator]
            if any(
                value.startswith(("shared:", "master:", "propagate_from:"))
                for value in optional
            ):
                raise ExecutionSnapshotError(
                    "replay mount namespace still permits propagation"
                )
        return
    environment = dict(os.environ)
    environment[marker] = "1"
    try:
        os.execvpe(
            "unshare",
            [
                "unshare",
                "--mount",
                "--propagation",
                "private",
                "--",
                sys.executable,
                *sys.argv,
            ],
            environment,
        )
    except OSError as error:
        raise ExecutionSnapshotError(
            "cannot enter the private replay mount namespace"
        ) from error


def _sha(value: object, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise ExecutionSnapshotError(f"{label} must be a lowercase SHA256 string")
    return value


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def scan_regular_tree(
    root: Path,
    *,
    label: str,
    exclude_python_cache: bool = False,
    materialize_file_symlinks: bool = False,
) -> list[dict[str, Any]]:
    """Seal deterministic membership for a no-symlink regular-file tree."""
    root = _absolute(root)
    if root.is_symlink() or not root.is_dir():
        raise ExecutionSnapshotError(f"{label} root must be a real directory")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        if exclude_python_cache and (
            "__pycache__" in relative.parts or path.suffix == ".pyc"
        ):
            continue
        source = path
        if path.is_symlink():
            if not materialize_file_symlinks:
                raise ExecutionSnapshotError(f"{label} contains a symlink: {relative}")
            try:
                source = path.resolve(strict=True)
            except OSError as error:
                raise ExecutionSnapshotError(
                    f"{label} contains a broken symlink: {relative}"
                ) from error
            if source.is_symlink() or not source.is_file():
                raise ExecutionSnapshotError(
                    f"{label} symlink does not resolve to a regular file: {relative}"
                )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ExecutionSnapshotError(
                f"{label} contains a non-regular entry: {relative}"
            )
        resolved, payload, digest = read_regular_bytes(source, f"{label} file")
        row = {
            "path": (
                relative.as_posix()
                if path.is_symlink()
                else resolved.relative_to(root).as_posix()
            ),
            "size": len(payload),
            "sha256": digest,
        }
        if path.is_symlink():
            row["source_path"] = str(resolved)
        rows.append(row)
    if not rows:
        raise ExecutionSnapshotError(f"{label} tree is empty")
    return rows


class ExecutionSnapshotPlan:
    """Collect, preflight, copy, fsync, and seal one private snapshot."""

    def __init__(self, output: TrustedOutputRoot, root: Path):
        self.output = output
        self.root = _absolute(root)
        try:
            self.root.relative_to(output.path)
        except ValueError as error:
            raise ExecutionSnapshotError("snapshot root escapes trusted output") from error
        self.output.mkdir(self.root, label="execution snapshot root")
        self._specs: dict[str, dict[str, Any]] = {}
        self._preflight: dict[str, int] | None = None

    def add_file(
        self,
        source: Path,
        target: Path,
        *,
        expected_sha256: str,
        size: int,
        kind: str,
        mode: int = 0o640,
        provenance_path: Path | None = None,
    ) -> Path:
        if self._preflight is not None:
            raise ExecutionSnapshotError("snapshot plan is already frozen")
        if type(kind) is not str or not kind:
            raise ExecutionSnapshotError("snapshot kind must be non-empty")
        if type(size) is not int or size < 0:
            raise ExecutionSnapshotError("snapshot size must be non-negative")
        digest = _sha(expected_sha256, "snapshot provenance SHA")
        source = _absolute(source)
        provenance = _absolute(
            source if provenance_path is None else provenance_path
        )
        target = _absolute(target)
        try:
            relative = target.relative_to(self.root).as_posix()
        except ValueError as error:
            raise ExecutionSnapshotError("snapshot target escapes snapshot root") from error
        spec = {
            "source": source,
            "provenance_path": provenance,
            "target": target,
            "expected_sha256": digest,
            "size": size,
            "kind": kind,
            "mode": mode,
        }
        previous = self._specs.get(relative)
        if previous is not None and previous != spec:
            raise ExecutionSnapshotError(f"snapshot target collision: {relative}")
        self._specs[relative] = spec
        return target

    def add_verified_file(
        self,
        source: Path,
        target: Path,
        *,
        kind: str,
        mode: int = 0o640,
    ) -> tuple[Path, str]:
        resolved, payload, digest = read_regular_bytes(source, kind)
        return self.add_file(
            resolved,
            target,
            expected_sha256=digest,
            size=len(payload),
            kind=kind,
            mode=mode,
            provenance_path=source,
        ), digest

    def add_tree(
        self,
        source_root: Path,
        target_root: Path,
        rows: list[dict[str, Any]],
        *,
        kind: str,
    ) -> None:
        source_root = _absolute(source_root)
        target_root = _absolute(target_root)
        observed: set[str] = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) not in (
                {"path", "size", "sha256"},
                {"path", "size", "sha256", "source_path"},
            ):
                raise ExecutionSnapshotError(f"{kind} tree row fields mismatch")
            relative = row["path"]
            if (
                type(relative) is not str
                or not relative
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or relative in observed
            ):
                raise ExecutionSnapshotError(f"{kind} tree row path is invalid")
            observed.add(relative)
            self.add_file(
                Path(row.get("source_path", source_root / relative)),
                target_root / relative,
                expected_sha256=row["sha256"],
                size=row["size"],
                kind=kind,
                provenance_path=source_root / relative,
            )

    @property
    def required_bytes(self) -> int:
        return sum(spec["size"] for spec in self._specs.values())

    def preflight(self, *, safety_margin_bytes: int = _DEFAULT_SAFETY_MARGIN) -> dict[str, int]:
        if self._preflight is not None:
            return dict(self._preflight)
        if not self._specs:
            raise ExecutionSnapshotError("execution snapshot plan is empty")
        if type(safety_margin_bytes) is not int or safety_margin_bytes < 0:
            raise ExecutionSnapshotError("snapshot safety margin is invalid")
        filesystem = os.statvfs(self.output.path)
        available = int(filesystem.f_bavail) * int(filesystem.f_frsize)
        required = self.required_bytes
        margin = max(safety_margin_bytes, required // 20)
        if available < required + margin:
            raise ExecutionSnapshotError(
                "insufficient space for copy-only execution snapshot: "
                f"available={available} required={required} margin={margin}"
            )
        self._preflight = {
            "available_bytes": available,
            "required_bytes": required,
            "safety_margin_bytes": margin,
        }
        return dict(self._preflight)

    def seal(self, manifest_path: Path) -> tuple[dict[str, Any], str]:
        preflight = self.preflight()
        rows: list[dict[str, Any]] = []
        for relative in sorted(self._specs):
            spec = self._specs[relative]
            resolved, payload, digest = read_regular_bytes(
                spec["source"], f'{spec["kind"]} provenance'
            )
            if digest != spec["expected_sha256"] or len(payload) != spec["size"]:
                raise ExecutionSnapshotError(
                    f'{spec["kind"]} provenance changed before snapshot'
                )
            self.output.publish(
                spec["target"], payload, label=f'{spec["kind"]} snapshot',
                mode=spec["mode"],
            )
            snapshot, copied, snapshot_sha = self.output.read(
                spec["target"], label=f'{spec["kind"]} snapshot verification'
            )
            if snapshot_sha != digest or copied != payload:
                raise ExecutionSnapshotError(f'{spec["kind"]} snapshot copy mismatch')
            rows.append({
                "kind": spec["kind"],
                "provenance_path": str(spec["provenance_path"]),
                "provenance_resolved_path": str(resolved),
                "provenance_sha256": digest,
                "snapshot_path": str(snapshot),
                "snapshot_sha256": snapshot_sha,
                "size": len(payload),
                "mode": spec["mode"],
                "copy_method": "copy",
            })
        manifest = {
            "schema": EXECUTION_SNAPSHOT_SCHEMA,
            "root": str(self.root),
            "copy_method": "copy",
            **preflight,
            "rows": rows,
            "rows_sha256": canonical_sha256(rows),
            "file_count": len(rows),
            "total_bytes": sum(row["size"] for row in rows),
            "passed": True,
        }
        payload = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        self.output.publish(manifest_path, payload, label="execution snapshot manifest")
        return manifest, hashlib.sha256(payload).hexdigest()


def validate_execution_snapshot(
    value: object,
    *,
    verify_provenance: bool,
    verify_snapshots: bool = True,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != EXECUTION_SNAPSHOT_FIELDS:
        raise ExecutionSnapshotError("execution snapshot fields mismatch")
    rows = value["rows"]
    if (
        value["schema"] != EXECUTION_SNAPSHOT_SCHEMA
        or value["copy_method"] != "copy"
        or value["passed"] is not True
        or type(value["root"]) is not str
        or not isinstance(rows, list)
        or not rows
        or type(value["file_count"]) is not int
        or value["file_count"] != len(rows)
        or canonical_sha256(rows) != _sha(value["rows_sha256"], "snapshot rows SHA")
    ):
        raise ExecutionSnapshotError("execution snapshot closure mismatch")
    for field in ("required_bytes", "available_bytes", "safety_margin_bytes", "total_bytes"):
        if type(value[field]) is not int or value[field] < 0:
            raise ExecutionSnapshotError(f"execution snapshot {field} is invalid")
    if value["total_bytes"] != sum(row.get("size", -1) for row in rows):
        raise ExecutionSnapshotError("execution snapshot byte total mismatch")
    if value["required_bytes"] != value["total_bytes"]:
        raise ExecutionSnapshotError("execution snapshot planned/copied bytes mismatch")
    root = _absolute(Path(value["root"]))
    if root.is_symlink() or not root.is_dir():
        raise ExecutionSnapshotError("execution snapshot root is not a real directory")
    snapshots: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != EXECUTION_SNAPSHOT_ROW_FIELDS:
            raise ExecutionSnapshotError("execution snapshot row fields mismatch")
        if (
            type(row["kind"]) is not str or not row["kind"]
            or row["copy_method"] != "copy"
            or type(row["size"]) is not int or row["size"] < 0
            or type(row["mode"]) is not int or row["mode"] < 0
        ):
            raise ExecutionSnapshotError("execution snapshot row is invalid")
        snapshot = _absolute(Path(row["snapshot_path"]))
        try:
            relative = snapshot.relative_to(root).as_posix()
        except ValueError as error:
            raise ExecutionSnapshotError("snapshot row escapes snapshot root") from error
        if relative in snapshots:
            raise ExecutionSnapshotError("execution snapshot contains duplicate target")
        snapshots.add(relative)
        provenance_sha = _sha(row["provenance_sha256"], "provenance SHA")
        snapshot_sha = _sha(row["snapshot_sha256"], "snapshot SHA")
        if provenance_sha != snapshot_sha:
            raise ExecutionSnapshotError("provenance/snapshot SHA mismatch")
        provenance_path = _absolute(Path(row["provenance_path"]))
        provenance_resolved = _absolute(Path(row["provenance_resolved_path"]))
        if verify_provenance and provenance_path != provenance_resolved:
            if (
                not provenance_path.is_symlink()
                or provenance_path.resolve(strict=True) != provenance_resolved
            ):
                raise ExecutionSnapshotError(
                    "execution provenance symlink target mismatch"
                )
        for enabled, path, digest, label in (
            (verify_provenance, provenance_resolved, provenance_sha, "provenance"),
            (verify_snapshots, row["snapshot_path"], snapshot_sha, "snapshot"),
        ):
            if enabled:
                _resolved, payload, observed = read_regular_bytes(Path(path), label)
                if observed != digest or len(payload) != row["size"]:
                    raise ExecutionSnapshotError(f"execution {label} mismatch")
    if verify_snapshots:
        current: set[str] = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ExecutionSnapshotError(
                    "execution snapshot contains a symlink"
                )
            if path.is_dir():
                continue
            if not path.is_file():
                raise ExecutionSnapshotError(
                    "execution snapshot contains a non-regular entry"
                )
            current.add(_absolute(path).relative_to(root).as_posix())
        if current != snapshots:
            raise ExecutionSnapshotError(
                "execution snapshot manifest membership mismatch"
            )
    return value
