from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import cv2

from scripts.eval_worldarena_context_pyramid_val5 import (
    build_panel_audit,
    render_native_cache,
    run_generation,
    shard_panel,
)
from scripts.worldarena_context_pyramid_val import ProtocolError, locked_grid, variant_name
from scripts.run_worldarena_context_pyramid_val5 import (
    assert_no_test_episode_reference,
    build_gpu_assignments,
    prepare_variant_summary,
    validate_diagnostic_tree,
)


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


def test_three_shards_partition_locked_panel_without_overlap() -> None:
    panel, _audit = build_panel_audit(_rows())

    shards = [shard_panel(panel, index, 3) for index in range(3)]

    assert [len(shard) for shard in shards] == [2, 2, 1]
    assert sorted(row["id"] for shard in shards for row in shard) == sorted(
        row["id"] for row in panel
    )


def _write_mp4(path: Path, frames: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (16, 16)
    )
    assert writer.isOpened()
    try:
        for index in range(frames):
            writer.write(np.full((16, 16, 3), index * 20, dtype=np.uint8))
    finally:
        writer.release()


def _selected_panel(tmp_path: Path) -> tuple[list[dict[str, object]], Path]:
    tasks = ["task_00", "task_12", "task_24", "task_36", "task_49"]
    episodes = [36, 37, 38, 39, 36]
    prediction_dir = tmp_path / "rendered" / "baseline"
    panel: list[dict[str, object]] = []
    for task, episode in zip(tasks, episodes, strict=True):
        source = tmp_path / "source" / task
        video = source / f"episode{episode}.mp4"
        instruction = source / f"episode{episode}.json"
        _write_mp4(video)
        instruction.write_text(json.dumps({"unseen": [task.replace("_", " ")]}))
        _write_mp4(prediction_dir / f"{task}_episode{episode}.mp4")
        panel.append(
            {
                "id": f"{task}/episode{episode}",
                "task": task,
                "episode": episode,
                "video_file": str(video),
                "instruction_file": str(instruction),
            }
        )
    return panel, prediction_dir


def test_prepare_variant_summary_accepts_exact_val5_and_rejects_test_episode(
    tmp_path: Path,
) -> None:
    panel, prediction_dir = _selected_panel(tmp_path)
    summary = prepare_variant_summary(
        panel, prediction_dir, tmp_path / "prepared"
    )

    assert len(summary) == 5
    assert all(Path(item["gt_path"]).parts[-5].startswith("task_") for item in summary)
    assert all(Path(item["image"]).is_file() for item in summary)

    unsafe = [dict(row) for row in panel]
    unsafe[0]["episode"] = 40
    with pytest.raises(ProtocolError, match="episodes 36-39"):
        prepare_variant_summary(unsafe, prediction_dir, tmp_path / "bad")


def _synthetic_official_tree(root: Path, count: int) -> Path:
    for index in range(count):
        task = f"task_{index:02d}"
        gt = root / "gt_dataset" / task / "episode36" / "video"
        generated = (
            root
            / "generated_dataset"
            / task
            / "episode36"
            / "1"
            / "video"
        )
        gt.mkdir(parents=True, exist_ok=True)
        generated.mkdir(parents=True, exist_ok=True)
        (gt / "frame_00000.jpg").write_bytes(b"gt")
        (generated / "frame_00000.jpg").write_bytes(b"generated")
    return root


def test_validate_diagnostic_tree_requires_five_unique_videos(tmp_path: Path) -> None:
    tree = _synthetic_official_tree(tmp_path / "tree", count=4)
    with pytest.raises(ProtocolError, match="coverage"):
        validate_diagnostic_tree(tree, expected_count=5)


def test_assert_no_test_episode_reference_scans_nested_values() -> None:
    with pytest.raises(ProtocolError, match="test episode reference"):
        assert_no_test_episode_reference(
            {"nested": [{"video": "/x/task_episode40.mp4"}]}
        )


def test_gpu_assignments_keep_one_worker_per_node43_gpu() -> None:
    variants = ["baseline", *(variant_name(config) for config in locked_grid())]
    assignments = build_gpu_assignments(variants, (0, 1, 2, 3))

    assert set(assignments) == {0, 1, 2, 3}
    assert sorted(item for values in assignments.values() for item in values) == sorted(
        variants
    )
    assert all(values for values in assignments.values())
