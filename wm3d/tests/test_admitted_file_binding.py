import os

import pytest
import yaml

from wm3d.data.file_binding import (
    FileBindingError, bind_stat_asset, require_stat_binding, verify_stat_asset,
)
from wm3d.data.source_adapters import AdapterContractError, load_adapter_contract


def forbid(*args, **kwargs):
    raise AssertionError("non-SHA admission must not invoke a hash or read payload bytes")


def test_stat_binding_does_not_read_asset_bytes(tmp_path, monkeypatch):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"real-file-for-stat-binding")
    monkeypatch.setattr(type(path), "open", forbid)
    binding = bind_stat_asset(tmp_path, path.name, admission_id="frozen-admission")
    assert verify_stat_asset(tmp_path, path.name, binding) == path
    assert "sha256" not in binding


@pytest.mark.parametrize("mutation", ["truncate", "replace", "mtime", "missing"])
def test_stat_binding_rejects_changed_identity(tmp_path, mutation):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"original")
    binding = bind_stat_asset(tmp_path, path.name, admission_id="frozen-admission")
    if mutation == "truncate":
        path.write_bytes(b"x")
    elif mutation == "replace":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(path.read_bytes())
        os.utime(replacement, ns=(binding["mtime_ns"], binding["mtime_ns"]))
        replacement.replace(path)
    elif mutation == "mtime":
        os.utime(path, ns=(binding["mtime_ns"] + 1_000_000, binding["mtime_ns"] + 1_000_000))
    else:
        path.unlink()
    with pytest.raises((FileBindingError, FileNotFoundError)):
        verify_stat_asset(tmp_path, path.name, binding)


def test_rejects_symlink_and_escape(tmp_path):
    path = tmp_path / "real"
    path.write_bytes(b"x")
    (tmp_path / "link").symlink_to(path)
    with pytest.raises(FileBindingError):
        bind_stat_asset(tmp_path, "link", admission_id="admission")
    with pytest.raises(FileBindingError):
        bind_stat_asset(tmp_path, "../real", admission_id="admission")


def test_binding_is_explicit_and_not_a_fabricated_digest(tmp_path):
    path = tmp_path / "real"
    path.write_bytes(b"x")
    binding = bind_stat_asset(tmp_path, path.name, admission_id="admission")
    for bad in [dict(binding, sha256="0"*64), dict(binding, admission_id=""),
                dict(binding, size_bytes=True), dict(binding, kind="sha256")]:
        with pytest.raises(FileBindingError):
            require_stat_binding(bad)


def test_adapter_reads_no_sha_and_keeps_semantic_validation(tmp_path, monkeypatch):
    term = dict(key="robot", columns=[0], scale=[1], offset=[0])
    value = dict(schema="wm3d_source_adapter_v6", name="physical_contract",
        raw_format="lerobot_parquet_video", observation_time_key="timestamp",
        views=[dict(name="head", key="images.head", color_order="rgb")],
        groups=[dict(group="arm", supervision="coarse_effect", action=[term], state=[term],
          action_time_key=None, state_time_key="timestamp", world_interval_index_key="frame_index",
          action_transform="identity", state_transform="identity", action_value_mask=[True],
          state_value_mask=[True], action_alignment="next_observation_drop_terminal")])
    path = tmp_path / "adapter.yaml"
    path.write_text(yaml.safe_dump(value))
    binding = bind_stat_asset(tmp_path, path.name, admission_id="adapter-admission")
    monkeypatch.setattr("wm3d.data.source_adapters.sha256_file", forbid)
    contract = load_adapter_contract(path, expected_binding=binding)
    assert contract.sha256 == ""
    assert contract.file_binding == binding
    assert contract.groups[0].action_alignment == "next_observation_drop_terminal"
    with pytest.raises(AdapterContractError, match="exactly one"):
        load_adapter_contract(path)
    with pytest.raises(AdapterContractError, match="exactly one"):
        load_adapter_contract(path, expected_binding=binding, expected_sha256="0"*64)
    value["groups"][0]["action_alignment"] = "guess"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(FileBindingError):
        load_adapter_contract(path, expected_binding=binding)
    new = bind_stat_asset(tmp_path, path.name, admission_id="adapter-admission")
    with pytest.raises(AdapterContractError):
        load_adapter_contract(path, expected_binding=new)


def test_legacy_digest_check_is_not_bypassed(tmp_path, monkeypatch):
    path = tmp_path / "legacy.yaml"
    path.write_text("schema: irrelevant")
    monkeypatch.setattr("wm3d.data.source_adapters.sha256_file", lambda _: "a" * 64)
    with pytest.raises(AdapterContractError, match="SHA mismatch"):
        load_adapter_contract(path, expected_sha256="b" * 64)


def test_production_parquet_reader_uses_admitted_binding(tmp_path, monkeypatch):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from types import SimpleNamespace
    from wm3d.data.episode_io import VerifiedAssetStore, open_admitted_episode_accessor, EpisodeIOError
    path = tmp_path / "episode.parquet"
    pq.write_table(pa.table({"timestamp": [0.0, 0.1, 0.2],
                            "robot": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}), path)
    binding = bind_stat_asset(tmp_path, path.name, admission_id="new-admission")
    monkeypatch.setattr("wm3d.data.episode_io.sha256_file", forbid)
    reader = open_admitted_episode_accessor(
        source_root=tmp_path, payload=path.name, payload_binding=binding,
        row_start=1, row_stop=3, adapter=SimpleNamespace(raw_format="lerobot_parquet_video"),
        asset_verifier=VerifiedAssetStore())
    assert np.array_equal(reader.array("robot"), [[3.0, 4.0], [5.0, 6.0]])
    assert np.array_equal(reader.array("timestamp"), [0.1, 0.2])
    with pytest.raises(EpisodeIOError, match="misses field"):
        reader.array("missing")
    os.utime(path, ns=(binding["mtime_ns"] + 1000000, binding["mtime_ns"] + 1000000))
    with pytest.raises(EpisodeIOError, match="identity changed"):
        reader.array("robot")  # cached arrays must not hide a changed payload


def test_production_stat_reader_refuses_legacy_repair_relabelling(tmp_path, monkeypatch):
    from wm3d.data.episode_io import VerifiedAssetStore, EpisodeIOError
    path = tmp_path / "payload.bin"
    path.write_bytes(b"original repaired source")
    binding = bind_stat_asset(tmp_path, path.name, admission_id="not-a-legacy-revision")
    monkeypatch.setattr("wm3d.data.episode_io.sha256_file", forbid)
    store = VerifiedAssetStore()
    store._repair_by_asset[(tmp_path.resolve(), path.name)] = {"sealed_digest": "a" * 64}
    with pytest.raises(EpisodeIOError, match="legacy repaired"):
        store.verify_admitted(tmp_path, path.name, binding)


@pytest.mark.parametrize("start,stop", [(0, 0), (-1, 1), (True, 2), (0, 4)])
def test_production_stat_reader_rejects_invalid_slice(tmp_path, monkeypatch, start, stop):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from types import SimpleNamespace
    from wm3d.data.episode_io import VerifiedAssetStore, open_admitted_episode_accessor, EpisodeIOError
    path = tmp_path / "episode.parquet"
    pq.write_table(pa.table({"timestamp": [0.0, 0.1, 0.2]}), path)
    binding = bind_stat_asset(tmp_path, path.name, admission_id="new-admission")
    monkeypatch.setattr("wm3d.data.episode_io.sha256_file", forbid)
    with pytest.raises(EpisodeIOError):
        open_admitted_episode_accessor(
            source_root=tmp_path, payload=path.name, payload_binding=binding,
            row_start=start, row_stop=stop, adapter=SimpleNamespace(raw_format="lerobot_parquet_video"),
            asset_verifier=VerifiedAssetStore())


def test_admitted_adapter_rejects_symlink_before_resolving(tmp_path):
    from wm3d.data.source_adapters import load_adapter_contract
    real = tmp_path / "real.yaml"
    real.write_text("this must not be parsed")
    alias = tmp_path / "alias.yaml"
    alias.symlink_to(real)
    binding = bind_stat_asset(tmp_path, real.name, admission_id="new-admission")
    with pytest.raises(FileBindingError, match="symlink"):
        load_adapter_contract(alias, expected_binding=binding)
