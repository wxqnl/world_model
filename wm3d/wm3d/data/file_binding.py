"""Explicit non-cryptographic file identity for approved data admissions.

This is not a content hash. It avoids reading payload bytes a second time and
detects ordinary replacement, truncation and modification through stat fields.
It cannot detect an adversarial same-inode, same-size write whose mtime is
restored. Only a separately approved admission may select this binding kind;
legacy SHA contracts are not converted or re-labelled.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Mapping


STAT_BINDING_KIND = "admitted_file_stat_v1"


class FileBindingError(ValueError):
    pass


def regular_asset(root: Path, relative: str) -> Path:
    root = Path(root)
    rel = PurePosixPath(relative)
    if (not root.is_absolute() or root.is_symlink() or not root.is_dir()
            or not relative or rel.is_absolute() or ".." in rel.parts
            or rel.as_posix() != relative):
        raise FileBindingError("asset requires a real absolute root and canonical relative path")
    root = root.resolve(strict=True)
    candidate = root
    for part in rel.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise FileBindingError(f"asset path contains a symlink: {candidate}")
    if not candidate.is_file():
        raise FileBindingError(f"asset is not a regular file: {candidate}")
    candidate.resolve(strict=True).relative_to(root)
    return candidate


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
    path = regular_asset(root, relative)
    st = path.stat()
    return require_stat_binding(dict(kind=STAT_BINDING_KIND,
        admission_id=admission_id, size_bytes=st.st_size,
        mtime_ns=st.st_mtime_ns, inode=st.st_ino))


def verify_stat_asset(root: Path, relative: str, binding: Mapping[str, Any]) -> Path:
    expected = require_stat_binding(binding)
    path = regular_asset(root, relative)
    observed = bind_stat_asset(root, relative, admission_id=expected["admission_id"])
    if observed != expected:
        raise FileBindingError(f"admitted asset identity changed: {relative}")
    return path
