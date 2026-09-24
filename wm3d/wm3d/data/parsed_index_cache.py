"""Reuse validated window-index objects without changing data or sampling.

The disk cache is a private, disposable accelerator. Source integrity is still
checked once in every process. Cached pickle accepts only CacheIndexEntry,
checks its CRC before deserialization, and is published atomically under flock.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import pickle
import stat
import struct
import tempfile
import threading
import time
import zlib

_FORMAT = "wm3d_parsed_window_index_v1"  # Bump if row validation changes.
_MEMORY = OrderedDict()
_MUTEX = threading.RLock()
_CHUNK = 16 << 20


def clear_parsed_index_memory_cache():
    """Release unselected rows after the trainer has assembled train and val."""
    with _MUTEX:
        _MEMORY.clear()


def _signature(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _private_directory(path):
    if path.is_symlink():
        raise ValueError("parsed-index cache directory must not be a symlink")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    s = path.stat()
    if s.st_uid != os.getuid() or stat.S_IMODE(s.st_mode) & 0o077:
        raise ValueError("parsed-index cache directory must be owner-only (0700)")


@contextmanager
def _lock(path, operation):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+b") as handle:
        s = os.fstat(handle.fileno())
        if s.st_uid != os.getuid() or stat.S_IMODE(s.st_mode) & 0o077:
            raise ValueError("parsed-index cache lock must be owner-only")
        fcntl.flock(handle, operation)
        yield


class _IndexUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("wm3d.data.manifest_contract", "CacheIndexEntry"):
            from .manifest_contract import CacheIndexEntry
            return CacheIndexEntry
        raise pickle.UnpicklingError("non-index object in parsed-index cache")


class _ChecksumWriter:
    def __init__(self, handle):
        self.handle, self.crc = handle, 0

    def write(self, data):
        self.crc = zlib.crc32(data, self.crc)
        return self.handle.write(data)


def _read(path, metadata):
    from .manifest_contract import CacheIndexEntry
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as handle:
        s = os.fstat(handle.fileno())
        if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or stat.S_IMODE(s.st_mode) & 0o077:
            raise ValueError("parsed-index cache file must be private and regular")
        header = handle.readline(8193)
        if len(header) > 8192 or not header.endswith(b"\n") or json.loads(header) != metadata:
            raise ValueError("parsed-index cache identity changed")
        start, end = handle.tell(), s.st_size - 4
        if end <= start:
            raise ValueError("incomplete parsed-index cache")
        crc = 0
        remaining = end - start
        while remaining:
            block = handle.read(min(remaining, _CHUNK))
            if not block:
                raise ValueError("truncated parsed-index cache")
            crc = zlib.crc32(block, crc)
            remaining -= len(block)
        if struct.pack("<I", crc) != handle.read(4):
            raise ValueError("parsed-index cache checksum mismatch")
        handle.seek(start)
        entries = _IndexUnpickler(handle).load()
        if handle.tell() != end:
            raise ValueError("unexpected parsed-index cache trailing data")
        if not isinstance(entries, tuple) or not entries or not all(isinstance(e, CacheIndexEntry) for e in entries):
            raise ValueError("invalid parsed-index cache entries")
        return entries


def _write(path, metadata, entries):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".building-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(metadata, sort_keys=True).encode() + b"\n")
            writer = _ChecksumWriter(handle)
            pickle.dump(entries, writer, protocol=5)
            handle.write(struct.pack("<I", writer.crc))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _try_read(path, metadata):
    try:
        return _read(path, metadata)
    except (OSError, ValueError, EOFError, pickle.UnpicklingError, AttributeError, TypeError):
        # A missing, interrupted, stale or corrupt cache never bypasses the
        # canonical parser. Its source has already passed the original check.
        return None


def load_parsed_window_index(path, expected_digest, *, verify_source, build):
    started = time.monotonic()
    signature = _signature(path)
    directory = os.environ.get("WM3D_INDEX_CACHE_DIR", "")
    key = (str(path), expected_digest, signature, directory)
    with _MUTEX:
        if key in _MEMORY:
            return _MEMORY[key]
        verify_source()
        if _signature(path) != signature:
            raise ValueError("window index changed during source verification")
        mode = "parsed"
        entries = None
        if directory:
            root = Path(directory).absolute()
            _private_directory(root)
            parent = root / _FORMAT / Path(*path.parts[1:])
            _private_directory(parent)
            cache, lock = parent / "entries.bin", parent / "build.lock"
            metadata = dict(format=_FORMAT, source=str(path), source_digest=expected_digest)
            while entries is None:
                with _lock(lock, fcntl.LOCK_SH):
                    entries = _try_read(cache, metadata)
                    failed_version = _signature(cache) if cache.exists() else None
                if entries is not None:
                    mode = "disk-hit"
                    break
                with _lock(lock, fcntl.LOCK_EX):
                    current_version = _signature(cache) if cache.exists() else None
                    if current_version != failed_version:
                        # A peer published while we waited. Release EX and
                        # deserialize under SH; never serialize all readers
                        # behind one expensive pickle load per rank.
                        continue
                    print(f"[index-cache] building {path}", flush=True)
                    entries = build()
                    if _signature(path) != signature:
                        raise ValueError("window index changed during parsing")
                    _write(cache, metadata, entries)
                    mode = "built"
        else:
            entries = build()
        if _signature(path) != signature:
            raise ValueError("window index changed during loading")
        _MEMORY.clear()
        _MEMORY[key] = entries
        print(f"[index-cache] {mode} rows={len(entries)} seconds={time.monotonic()-started:.2f} path={path}", flush=True)
        return entries
