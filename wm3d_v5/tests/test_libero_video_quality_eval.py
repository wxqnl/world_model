from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from wm3d_v3.eval.libero_video_quality_eval import LiberoCachedWorldVideoDataset


def _write_fake_libero_hdf5(path: Path) -> None:
    frames = np.zeros((5, 4, 4, 3), dtype=np.uint8)
    frames[0] = 10
    frames[1] = 20
    frames[2] = 30
    frames[3] = 40
    frames[4] = 50
    with h5py.File(path, "w") as h5:
        obs = h5.create_group("data").create_group("demo_0").create_group("obs")
        obs.create_dataset("agentview_rgb", data=frames)


def _write_fake_cache(path: Path) -> None:
    np.savez(
        path,
        s_in=np.ones((2, 64, 8), dtype=np.float16),
        c=np.ones((8,), dtype=np.float16),
        context_rgb=np.ones((3, 4, 4), dtype=np.float16) * 0.5,
        action_tgt=np.zeros((3, 7), dtype=np.float32),
        action_tgt_norm=np.zeros((3, 6), dtype=np.float32),
        lowdim_state=np.ones((12,), dtype=np.float32),
        action_history=np.ones((2, 7), dtype=np.float32),
    )


def test_libero_cached_dataset_loads_world_model_inputs_and_future_rgb(tmp_path):
    hdf5_path = tmp_path / "demo.hdf5"
    cache_path = tmp_path / "window_000000.npz"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_fake_libero_hdf5(hdf5_path)
    _write_fake_cache(cache_path)
    manifest_path.write_text(
        json.dumps(
            {
                "cache_path": str(cache_path),
                "hdf5_path": str(hdf5_path),
                "task_name": "LIBERO_TEST_TASK",
                "instruction": "do the test",
                "demo_id": "demo_0",
                "target_start": 3,
                "T": 2,
                "k": 3,
            }
        )
        + "\n"
    )

    ds = LiberoCachedWorldVideoDataset(manifest_path, rgb_size=4)
    item = ds[0]

    assert len(ds) == 1
    assert item["dataset"] == "libero"
    assert item["clip_id"] == "LIBERO_TEST_TASK/demo_0"
    assert item["start"] == 3
    assert item["s_in"].shape == (2, 64, 8)
    assert item["context_rgb"].shape == (3, 4, 4)
    assert item["lowdim_state"].shape == (12,)
    assert item["action_history"].shape == (2, 7)
    assert item["rgb_tgt"].shape == (3, 3, 4, 4)
    assert np.isclose(float(item["rgb_tgt"][0].mean()), 40.0 / 255.0)
    assert np.isclose(float(item["rgb_tgt"][1].mean()), 50.0 / 255.0)
    assert np.isclose(float(item["rgb_tgt"][2].mean()), 50.0 / 255.0)


def test_libero_cached_dataset_balances_tasks(tmp_path):
    hdf5_path = tmp_path / "demo.hdf5"
    _write_fake_libero_hdf5(hdf5_path)
    rows = []
    for task_id in range(2):
        for idx in range(3):
            cache_path = tmp_path / f"task{task_id}_{idx}.npz"
            _write_fake_cache(cache_path)
            rows.append(
                {
                    "cache_path": str(cache_path),
                    "hdf5_path": str(hdf5_path),
                    "task_name": f"TASK_{task_id}",
                    "instruction": "do it",
                    "demo_id": "demo_0",
                    "target_start": 0,
                    "T": 2,
                    "k": 3,
                }
            )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    ds = LiberoCachedWorldVideoDataset(
        manifest_path,
        rgb_size=4,
        balanced_tasks=True,
        max_windows_per_task=2,
        seed=7,
    )

    assert len(ds) == 4
    assert {row["task_name"] for row in ds.rows} == {"TASK_0", "TASK_1"}
