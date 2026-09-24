from __future__ import annotations
import fcntl
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import time

import pytest
from wm3d.data import manifest_contract as mc
from wm3d.data import parsed_index_cache as pc


@pytest.fixture(autouse=True)
def clean_memory():
    pc.clear_parsed_index_memory_cache()
    yield
    pc.clear_parsed_index_memory_cache()


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "private-cache"
    monkeypatch.setenv("WM3D_INDEX_CACHE_DIR", str(root))
    path = tmp_path / "windows.jsonl"
    row = dict(schema=mc.CACHE_INDEX_SCHEMA, sample_id="sample-0", source="real-source",
               split="train", embodiment="panda", feature_shard="episode/features",
               feature_sha256="a"*64, robot_shard="episode/robot", robot_sha256="b"*64,
               rgb_pack="episode/rgb", rgb_pack_sha256="c"*64, leading_feature_row=0,
               context_feature_rows=[1, 3], future_feature_rows=[5, 8])
    path.write_text(json.dumps(row) + "\n")
    return path, root, row


def test_memory_and_disk_preserve_every_field_and_order(source, monkeypatch):
    path, root, row = source
    calls = []
    original = mc._parse_cache_index
    def parse(p):
        calls.append(p)
        return original(p)
    monkeypatch.setattr(mc, "_parse_cache_index", parse)
    digest = mc.sha256_file(path)
    first = mc.load_cache_index(path, expected_sha256=digest)
    assert mc.load_cache_index(path, expected_sha256=digest) is first
    pc.clear_parsed_index_memory_cache()
    second = mc.load_cache_index(path, expected_sha256=digest)
    assert first == second and len(calls) == 1
    assert second[0].context_feature_rows == (1, 3)
    assert second[0].future_feature_rows == (5, 8)


def test_source_change_is_not_hidden_by_memory_or_disk(source):
    path, root, row = source
    digest = mc.sha256_file(path)
    first = mc.load_cache_index(path, expected_sha256=digest)
    old_stat = path.stat()
    row["sample_id"] = "sample-1"
    path.write_text(json.dumps(row) + "\n")
    os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    with pytest.raises(mc.ManifestContractError, match="mismatch"):
        mc.load_cache_index(path, expected_sha256=digest)
    second = mc.load_cache_index(path, expected_sha256=mc.sha256_file(path))
    assert first != second and second[0].sample_id == "sample-1"


@pytest.mark.parametrize("damage", ["truncate", "flip"])
def test_corrupt_cache_rebuilds_from_validated_source(source, damage):
    path, root, _ = source
    digest = mc.sha256_file(path)
    expected = mc.load_cache_index(path, expected_sha256=digest)
    cache = next(root.rglob("entries.bin"))
    raw = bytearray(cache.read_bytes())
    if damage == "truncate":
        raw = raw[:-13]
    else:
        raw[-8] ^= 1
    cache.write_bytes(raw)
    pc.clear_parsed_index_memory_cache()
    assert mc.load_cache_index(path, expected_sha256=digest) == expected


def test_invalid_source_never_publishes_cache(source):
    path, root, row = source
    row["future_feature_rows"] = [3, 8]
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(mc.ManifestContractError, match="strictly increasing"):
        mc.load_cache_index(path, expected_sha256=mc.sha256_file(path))
    assert not list(root.rglob("entries.bin"))


def test_cache_rejects_non_index_pickle_globals():
    with pytest.raises(pickle.UnpicklingError, match="non-index"):
        pc._IndexUnpickler(io.BytesIO(pickle.dumps(Path("/not-a-window")))).load()


def test_cache_requires_private_directory(source):
    path, root, _ = source
    root.mkdir(mode=0o777)
    root.chmod(0o777)
    with pytest.raises(ValueError, match="owner-only"):
        mc.load_cache_index(path, expected_sha256=mc.sha256_file(path))


def test_concurrent_processes_build_once(source):
    path, root, _ = source
    digest = mc.sha256_file(path)
    ctx = mp.get_context("fork")
    calls = ctx.Value("i", 0)
    initial_read = ctx.Barrier(4)
    def worker():
        pc.clear_parsed_index_memory_cache()
        original_try, original_read = pc._try_read, pc._read
        first = True
        def guarded_read(cache, metadata):
            if cache.exists():
                with (cache.parent / "build.lock").open("rb") as probe:
                    fcntl.flock(probe, fcntl.LOCK_SH | fcntl.LOCK_NB)
            return original_read(cache, metadata)
        def synchronized_miss(cache, metadata):
            nonlocal first
            result = original_try(cache, metadata)
            if first:
                first = False
                initial_read.wait(10)
            return result
        pc._read, pc._try_read = guarded_read, synchronized_miss
        def build():
            with calls.get_lock():
                calls.value += 1
            time.sleep(0.1)
            return mc._parse_cache_index(path)
        result = pc.load_parsed_window_index(
            path, digest,
            verify_source=lambda: mc._verify_cache_index_source(path, digest),
            build=build,
        )
        assert result[0].sample_id == "sample-0"
    children = [ctx.Process(target=worker) for _ in range(4)]
    for child in children:
        child.start()
    for child in children:
        child.join(20)
        assert child.exitcode == 0
    assert calls.value == 1


def test_interrupted_publish_keeps_previous_cache(source, monkeypatch):
    path, root, _ = source
    digest = mc.sha256_file(path)
    expected = mc.load_cache_index(path, expected_sha256=digest)
    cache = next(root.rglob("entries.bin"))
    original = cache.read_bytes()
    def fail(*args):
        raise OSError("interrupted atomic publish")
    monkeypatch.setattr(pc.os, "replace", fail)
    with pytest.raises(OSError, match="interrupted"):
        pc._write(cache, {}, expected)
    assert cache.read_bytes() == original
    assert not list(cache.parent.glob(".building-*"))
