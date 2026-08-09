from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.cache_robocasa365_v7_compact import (
    WorkItem,
    _filter_records_by_rgb_sidecar,
    _encode_causal_clip,
    _write_causal_item,
)
from scripts.cache_wm3d_v8_stage0_causal_dual_view_oxe import (
    _encode_record_windows,
    _index_row,
    _records_for_split,
    _publish_archive,
    _validate_index_selection,
    _window_path,
)
from wm3d_v3.data.manifest import OXEClipRecord
from wm3d_v3.data.v8_causal_dual_view import (
    validate_causal_dual_view_archive,
)


class FutureMixingEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.return_depth = True
        self.return_depth_conf = True
        self.return_geom_extra = True

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        _, frames, _, height, width = images.shape
        signal = images.float().mean(dim=(2, 3, 4))
        mixed = signal + signal.mean(dim=1, keepdim=True)
        pooled = mixed[:, :, None, None].repeat(1, 1, 4, 6)
        depth = mixed[:, :, None, None].repeat(1, 1, height, width)
        return {
            "pooled": pooled,
            "depth": depth,
            "depth_conf": torch.ones_like(depth),
            "world_points": torch.stack((depth, depth * 2, depth * 3), dim=-1),
            "world_points_conf": torch.ones_like(depth),
            "pose_enc": mixed[:, :, None].repeat(1, 1, 9),
        }


class IdentityCodec:
    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return value

    @staticmethod
    def quantize(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = torch.full(
            (*value.shape[:-2], 1, 1),
            1.0 / 127.0,
            dtype=torch.float32,
            device=value.device,
        )
        codes = torch.round(value / scale).clamp(-127, 127).to(torch.int8)
        return codes, scale.to(torch.float16)


def _frames(values: list[int]) -> list[np.ndarray]:
    return [
        np.full((8, 8, 3), value, dtype=np.uint8)
        for value in values
    ]


def _record() -> OXEClipRecord:
    return OXEClipRecord(
        clip_id="bridge/episode_1",
        dataset="bridge",
        tar_path="/sealed/bridge.tar",
        pickle_member="episode_1.pkl",
        n_frames=7,
        fps=5,
        robot="widowx",
        task_text="move the block",
    )


def test_robocasa_causal_builder_has_exact_sorted_window_axis() -> None:
    payload = _encode_causal_clip(
        _frames([0, 10, 20, 30, 40, 50, 60]),
        encoder=FutureMixingEncoder(),
        codec=IdentityCodec(),
        T=3,
        k=2,
        stride=2,
        keep_geometry=True,
    )

    assert payload["window_starts"].tolist() == [0, 2]
    assert payload["context_codes"].shape == (2, 3, 4, 6)
    assert payload["future_codes"].shape == (2, 2, 4, 6)
    assert payload["future_depth_patch"].shape == (2, 2, 8, 8)
    validate_causal_dual_view_archive(payload, T=3, k=2, paired_views=False)


def test_robocasa_context_bytes_ignore_future_pixels() -> None:
    first = _encode_causal_clip(
        _frames([0, 10, 20, 0, 0]),
        encoder=FutureMixingEncoder(),
        codec=IdentityCodec(),
        T=3,
        k=2,
        stride=2,
        keep_geometry=True,
    )
    changed = _encode_causal_clip(
        _frames([0, 10, 20, 255, 255]),
        encoder=FutureMixingEncoder(),
        codec=IdentityCodec(),
        T=3,
        k=2,
        stride=2,
        keep_geometry=True,
    )

    np.testing.assert_array_equal(first["context_codes"], changed["context_codes"])
    np.testing.assert_array_equal(first["context_scale"], changed["context_scale"])
    assert not np.array_equal(first["future_codes"], changed["future_codes"])


def test_robocasa_writer_emits_loader_ready_causal_archive(
    tmp_path: Path,
) -> None:
    action = SimpleNamespace(
        adapter_version="wm3d_v7_base_delta_axisangle_gripclose_v1",
        control_hz=20.0,
        raw_kind="joint",
        action_key="action",
    )
    record = SimpleNamespace(
        clip_hash="clip_hash",
        split="train",
        source="robocasa365",
        task_class="atomic",
        native_episode_id="episode_1",
        native_start_frame=0,
        native_end_frame=96,
        native_fps=20.0,
        embodiment_id="panda",
        task_text="move the block",
        action=action,
    )
    context = np.zeros((1, 16, 4, 6), dtype=np.int8)
    future = np.ones((1, 8, 4, 6), dtype=np.int8)
    item = WorkItem(
        record=record,
        metadata={},
        model_frame_indices=np.arange(24, dtype=np.int64) * 4,
        actions=np.zeros((24, 7), dtype=np.float32),
        raw_actions=np.zeros((96, 12), dtype=np.float32),
        native_frame_indices=np.arange(96, dtype=np.int64),
        model_timestamps=np.arange(24, dtype=np.float64) / 5.0,
        action_valid_mask=np.ones(24, dtype=np.bool_),
        rewards=np.zeros(24, dtype=np.float32),
        dones=np.zeros(24, dtype=np.bool_),
        anchor_codes=context,
        anchor_scale=np.ones((1, 16, 1, 1), dtype=np.float16),
        future_codes=future,
        future_scale=np.ones((1, 8, 1, 1), dtype=np.float16),
        window_starts=np.asarray([0], dtype=np.int64),
        wrist_codes=context.copy(),
        wrist_scale=np.ones((1, 16, 1, 1), dtype=np.float16),
        depth_patch=np.ones((1, 8, 8, 8), dtype=np.float16),
        depth_conf_patch=np.ones((1, 8, 8, 8), dtype=np.float16),
        point_patch=np.ones((1, 8, 8, 8, 3), dtype=np.float16),
        point_conf_patch=np.ones((1, 8, 8, 8), dtype=np.float16),
        pose_enc=np.ones((1, 8, 9), dtype=np.float16),
    )

    row = _write_causal_item(
        item,
        tmp_path,
        np.ones(2048, dtype=np.float16),
        action_audit_sha256="a" * 64,
        manifest_sha256="b" * 64,
        selection_sha256="c" * 64,
        config_sha256="d" * 64,
        codec_sha256="e" * 64,
        codec_downstream_report_sha256="f" * 64,
        v7_source="atomic",
    )

    with np.load(row["path"], allow_pickle=False) as archive:
        summary = validate_causal_dual_view_archive(
            archive,
            T=16,
            k=8,
            paired_views=True,
        )
        assert archive["window_starts"].tolist() == [0]
    assert summary["windows"] == 1
    assert row["schema"] == "wm3d_v8_stage0_causal_dual_view_v1"
    assert row["context_future_leakage"] is False
    assert row["v7_source"] == "atomic"
    assert len(row["artifact_sha256"]) == 64


def test_robocasa_causal_selection_intersects_rgb_sidecar(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "rgb_index.jsonl"
    sidecar.write_text(
        '{"schema":"wm3d_v7_rgb_sidecar_v1","clip_hash":"keep",'
        '"split":"train"}\n'
        '{"schema":"wm3d_v7_rgb_sidecar_v1","clip_hash":"other",'
        '"split":"val"}\n'
    )
    records = [
        SimpleNamespace(clip_hash="keep", split="train"),
        SimpleNamespace(clip_hash="drop", split="train"),
        SimpleNamespace(clip_hash="other", split="train"),
    ]

    filtered, digest = _filter_records_by_rgb_sidecar(records, sidecar)

    assert [record.clip_hash for record in filtered] == ["keep"]
    assert len(digest) == 64


def test_robocasa_causal_selection_rejects_bad_rgb_sidecar(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "rgb_index.jsonl"
    sidecar.write_text(
        '{"schema":"wrong","clip_hash":"keep","split":"train"}\n'
    )
    records = [SimpleNamespace(clip_hash="keep", split="train")]

    with pytest.raises(ValueError, match="RGB sidecar schema"):
        _filter_records_by_rgb_sidecar(records, sidecar)



def test_oxe_builder_publishes_exact_selection_without_clobber(
    tmp_path: Path,
) -> None:
    record = _record()
    encoded = _encode_record_windows(
        record,
        np.stack(_frames([0, 10, 20, 30, 40, 50, 60])),
        starts=[0, 2],
        encoder=FutureMixingEncoder(),
        codec=IdentityCodec(),
        T=3,
        k=2,
        source="Bridge",
        split="train",
        input_manifest_sha256="1" * 64,
        selection_sha256="2" * 64,
        config_sha256="3" * 64,
        rgb_sha256="4" * 64,
        action_sha256="5" * 64,
        task_sha256="6" * 64,
    )
    rows = []
    for start, payload in encoded:
        path = _window_path(tmp_path, record.clip_id, start)
        artifact_sha = _publish_archive(path, payload)
        rows.append(
            _index_row(
                record=record,
                source="Bridge",
                split="train",
                start=start,
                path=path,
                artifact_sha256=artifact_sha,
                payload=payload,
            )
        )

    expected = [(record.clip_id, 0), (record.clip_id, 2)]
    _validate_index_selection(expected, rows)
    assert all(Path(row["path"]).is_file() for row in rows)
    assert all(len(row["artifact_sha256"]) == 64 for row in rows)
    assert all(row["paired_views"] is False for row in rows)

    same_sha = _publish_archive(
        _window_path(tmp_path, record.clip_id, 0),
        encoded[0][1],
    )
    assert same_sha == rows[0]["artifact_sha256"]

    conflicting = dict(encoded[0][1])
    conflicting["future_codes"] = np.ones_like(conflicting["future_codes"])
    with pytest.raises(FileExistsError, match="non-identical"):
        _publish_archive(
            _window_path(tmp_path, record.clip_id, 0),
            conflicting,
        )


def test_oxe_producer_uses_training_episode_split_exactly() -> None:
    records = [
        OXEClipRecord(
            clip_id=f"bridge/episode_{index:03d}",
            dataset="bridge",
            tar_path="/sealed/bridge.tar",
            pickle_member=f"episode_{index:03d}.pkl",
            n_frames=32,
            fps=5,
            robot="widowx",
            task_text="move the block",
        )
        for index in range(20)
    ]
    train = _records_for_split(
        records, split="train", val_frac=0.20, seed=909
    )
    val = _records_for_split(
        records, split="val", val_frac=0.20, seed=909
    )

    assert len(train) == 16 and len(val) == 4
    assert {record.clip_id for record in train}.isdisjoint(
        record.clip_id for record in val
    )
    assert {record.clip_id for record in train + val} == {
        record.clip_id for record in records
    }


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [
            {"clip_id": "bridge/episode_1", "start": 0},
            {"clip_id": "bridge/episode_1", "start": 0},
        ],
        [
            {"clip_id": "bridge/episode_1", "start": 0},
            {"clip_id": "bridge/episode_1", "start": 4},
        ],
    ],
)
def test_oxe_selection_closure_rejects_missing_duplicate_or_extra(rows) -> None:
    with pytest.raises(ValueError, match="selection closure"):
        _validate_index_selection(
            [("bridge/episode_1", 0), ("bridge/episode_1", 2)],
            rows,
        )
