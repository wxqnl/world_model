from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.eval_worldarena_context_pyramid_val5 import (
    build_panel_audit,
    render_native_cache,
    run_generation,
)
from scripts.worldarena_context_pyramid_val import ProtocolError, locked_grid, variant_name


def _rows(task_count: int = 50) -> list[dict[str, object]]:
    return [
        {
            "id": f"task_{task_index:02d}/episode{episode}",
            "task": f"task_{task_index:02d}",
            "episode": episode,
            "video_file": f"/dataset/task_{task_index:02d}/episode{episode}.mp4",
            "hdf5_file": f"/dataset/task_{task_index:02d}/episode{episode}.hdf5",
            "instruction_file": f"/dataset/task_{task_index:02d}/episode{episode}.json",
        }
        for task_index in range(task_count)
        for episode in range(50)
    ]


def test_build_panel_audit_contains_only_val_rows_and_hashes() -> None:
    panel, audit = build_panel_audit(_rows())

    assert [row["episode"] for row in panel] == [36, 37, 38, 39, 36]
    assert audit["future_gt_used_for_inference"] is False
    assert audit["allowed_episodes"] == [36, 37, 38, 39]
    assert len(audit["manifest_row_sha256"]) == 5
    assert len(set(audit["manifest_row_sha256"])) == 5
    assert not any("episode4" in identity for identity in audit["ids"])


def test_render_native_cache_writes_baseline_plus_six_variants(tmp_path: Path) -> None:
    initial = np.zeros((48, 64, 3), dtype=np.uint8)
    native = np.zeros((2, 3, 64, 64), dtype=np.float32)
    calls: list[tuple[Path, int, int]] = []

    def fake_writer(frames, path, fps):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic")
        calls.append((path, len(frames), fps))

    written = render_native_cache(
        initial,
        native,
        tmp_path,
        "task_00_episode36.mp4",
        fps=10,
        video_writer=fake_writer,
    )

    assert set(written) == {
        "baseline",
        *(variant_name(config) for config in locked_grid()),
    }
    assert all(path.is_file() for path in written.values())
    assert len(calls) == 7
    assert all(frame_count == 3 and fps == 10 for _, frame_count, fps in calls)


def test_run_generation_validates_panel_before_loading_model(tmp_path: Path) -> None:
    events: list[str] = []
    config = {
        "manifest": str(tmp_path / "manifest.jsonl"),
        "checkpoint": str(tmp_path / "checkpoint.pt"),
        "output_root": str(tmp_path / "output"),
    }

    def invalid_manifest(_path):
        return _rows(task_count=49)

    def forbidden_loader(*_args, **_kwargs):
        events.append("loaded")
        raise AssertionError("loader must not run")

    with pytest.raises(ProtocolError, match="expected 50 tasks"):
        run_generation(
            config,
            physical_device=0,
            manifest_reader=invalid_manifest,
            checkpoint_loader=forbidden_loader,
        )

    assert events == []


def test_panel_audit_json_has_no_future_gt_inference_field() -> None:
    _panel, audit = build_panel_audit(_rows())
    encoded = json.dumps(audit, sort_keys=True)

    assert '"future_gt_used_for_inference": false' in encoded
    assert "future_gt_frames" not in encoded
