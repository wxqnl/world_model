"""Explicit non-cryptographic file identity for approved data admissions.

This is not a content hash. It avoids reading payload bytes a second time and
detects ordinary replacement, truncation and modification through stat fields.
It cannot detect an adversarial same-inode, same-size write whose mtime is
restored. Only a separately approved admission may select this binding kind;
legacy SHA contracts are not converted or re-labelled.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


STAT_BINDING_KIND = "admitted_file_stat_v1"


class FileBindingError(ValueError):
    pass


def _regular_asset_stat(root: Path, relative: str):
    root = Path(root)
    rel = PurePosixPath(relative)
    if (not root.is_absolute() or not relative or not rel.parts or rel.is_absolute()
            or ".." in rel.parts or rel.as_posix() != relative):
        raise FileBindingError("asset requires a real absolute root and canonical relative path")
    # Walk relative components by descriptors: O_NOFOLLOW enforces the same
    # no-symlink contract without repeatedly resolving every deep NFS ancestor.
    descriptors=[]
    try:
        fd=os.open(root,os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(fd)
        proc=Path("/proc/self/fd")/str(fd)
        real_root=Path(os.readlink(proc)) if proc.exists() else root.resolve(strict=True)
        for i,part in enumerate(rel.parts):
            flags=getattr(os,"O_PATH",os.O_RDONLY) | os.O_NOFOLLOW
            if i<len(rel.parts)-1:flags|=os.O_DIRECTORY
            fd=os.open(part,flags,dir_fd=fd)
            descriptors.append(fd)
            st=os.fstat(fd)
            if stat.S_ISLNK(st.st_mode):
                raise FileBindingError("asset path contains a symlink")
        if not stat.S_ISREG(st.st_mode):
            raise FileBindingError("asset is not a regular file")
        return real_root/relative,st
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise FileBindingError("asset path is unavailable or contains a symlink") from exc
    finally:
        for fd in reversed(descriptors):os.close(fd)


def regular_asset(root: Path, relative: str) -> Path:
    return _regular_asset_stat(root,relative)[0]


def require_stat_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"kind", "admission_id", "size_bytes", "mtime_ns", "inode"}
    if not isinstance(value, dict) or set(value) != fields:
        raise FileBindingError("stat binding fields mismatch")
    if value["kind"] != STAT_BINDING_KIND:
        raise FileBindingError("unsupported file binding kind")
    if not isinstance(value["admission_id"], str) or not value["admission_id"].strip():
        raise FileBindingError("stat binding requires an explicit admission identity")
    for key in ("size_bytes", "mtime_ns", "inode"):
        if type(value[key]) is not int or value[key] <= 0:
            raise FileBindingError(f"stat binding {key} must be a positive integer")
    return dict(value)


def bind_stat_asset(root: Path, relative: str, *, admission_id: str) -> dict[str, Any]:
    """Capture identity after completed read/decode receipts have been admitted.

    The caller owns admission of the receipts, upstream revision, selected
    membership and physical adapter. This function does not certify those facts.
    Device is intentionally namespace-local and excluded from the portable
    contract; inode, file length and server mtime survive the NFS export.
    """
    path, st = _regular_asset_stat(root, relative)
    return require_stat_binding(dict(kind=STAT_BINDING_KIND,
        admission_id=admission_id, size_bytes=st.st_size,
        mtime_ns=st.st_mtime_ns, inode=st.st_ino))


def verify_stat_asset(root: Path, relative: str, binding: Mapping[str, Any]) -> Path:
    expected = require_stat_binding(binding)
    path, st = _regular_asset_stat(root, relative)
    observed = dict(kind=STAT_BINDING_KIND, admission_id=expected["admission_id"],
                    size_bytes=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino)
    if observed != expected:
        raise FileBindingError(f"admitted asset identity changed: {relative}")
    return path
