from __future__ import annotations

from types import SimpleNamespace


def _rec(clip_id: str, dataset: str = "bridge"):
    return SimpleNamespace(clip_id=clip_id, dataset=dataset)


def test_episode_split_keeps_clip_on_one_side():
    from wm3d_v3.data.splits import episode_split

    records = [
        _rec("bridge/ep1"),
        _rec("bridge/ep1"),
        _rec("bridge/ep2"),
        _rec("bridge/ep3"),
        _rec("bridge/ep4"),
    ]

    split = episode_split(records, val_frac=0.5, seed=7)

    assert split.train_clip_ids
    assert split.val_clip_ids
    assert split.train_clip_ids.isdisjoint(split.val_clip_ids)
    assert split.train_clip_ids | split.val_clip_ids == {
        "bridge/ep1",
        "bridge/ep2",
        "bridge/ep3",
        "bridge/ep4",
    }


def test_episode_split_is_deterministic_independent_of_record_order():
    from wm3d_v3.data.splits import episode_split

    records_a = [_rec("c3"), _rec("c1"), _rec("c2"), _rec("c4")]
    records_b = list(reversed(records_a))

    split_a = episode_split(records_a, val_frac=0.5, seed=123)
    split_b = episode_split(records_b, val_frac=0.5, seed=123)

    assert split_a == split_b


def test_episode_split_holds_out_dataset():
    from wm3d_v3.data.splits import episode_split

    records = [
        _rec("bridge/ep1", "bridge"),
        _rec("fractal/ep1", "fractal20220817"),
        _rec("bridge/ep2", "bridge"),
        _rec("fractal/ep2", "fractal20220817"),
    ]

    split = episode_split(records, val_frac=0.0, seed=0, heldout_dataset="fractal20220817")

    assert split.val_clip_ids == {"fractal/ep1", "fractal/ep2"}
    assert split.train_clip_ids == {"bridge/ep1", "bridge/ep2"}
    assert split.train_clip_ids.isdisjoint(split.val_clip_ids)


def test_episode_split_accepts_explicit_ids():
    from wm3d_v3.data.splits import episode_split

    records = [_rec("c1"), _rec("c2"), _rec("c3")]

    split = episode_split(records, val_frac=0.5, seed=0, val_clip_ids=["c2"])

    assert split.val_clip_ids == {"c2"}
    assert split.train_clip_ids == {"c1", "c3"}


def test_split_records_uses_episode_config_and_split_file(tmp_path):
    import json

    from wm3d_v3.data.splits import split_mode_from_config, split_records

    records = [_rec("c1"), _rec("c2"), _rec("c3")]
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps({"train": ["c1", "c3"], "val": ["c2"]}))
    data_cfg = {
        "split_file": str(split_file),
        "val_frac": 0.5,
        "seed": 0,
    }

    train_records, val_records = split_records(records, data_cfg)

    assert split_mode_from_config(data_cfg) == "episode"
    assert {r.clip_id for r in train_records} == {"c1", "c3"}
    assert {r.clip_id for r in val_records} == {"c2"}


def test_split_mode_defaults_to_random_window_without_episode_keys():
    from wm3d_v3.data.splits import split_mode_from_config

    assert split_mode_from_config({"val_frac": 0.1, "seed": 0}) == "random_window"


def test_random_window_indices_keep_legacy_minimum_val_window():
    from wm3d_v3.data.splits import random_window_indices

    train_idx, val_idx = random_window_indices(5, val_frac=0.0, seed=0)

    assert len(val_idx) == 1
    assert len(train_idx) == 4
    assert set(train_idx).isdisjoint(val_idx)


def test_build_datasets_defaults_to_legacy_random_window(monkeypatch):
    from torch.utils.data import Subset

    from wm3d_v3.training import train as train_mod

    records = [_rec(f"c{i}") for i in range(5)]

    class FakeWindowDataset:
        def __init__(self, records, cfg):
            self.records = records
            self.cfg = cfg

        def __len__(self):
            return 5

        def __getitem__(self, index):
            return index

    monkeypatch.setattr(train_mod, "read_manifest", lambda path: records)
    monkeypatch.setattr(train_mod, "OXEWindowDataset", FakeWindowDataset)

    cfg = {
        "data": {
            "manifest": "unused.jsonl",
            "cache_root": "/tmp/cache",
            "T": 2,
            "k": 1,
            "stride": 1,
            "val_frac": 0.0,
            "seed": 0,
        }
    }

    train_ds, val_ds = train_mod.build_datasets(cfg)

    assert isinstance(train_ds, Subset)
    assert isinstance(val_ds, Subset)
    assert len(train_ds) == 4
    assert len(val_ds) == 1
