from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from wm3d_v3.data.manifest import OXEClipRecord
from wm3d_v3.data.window_dataset import OXEWindowDataset, WindowConfig
from wm3d_v3.data.v7_compact_dataset import (
    V7CompactDatasetConfig,
    V7CompactWindowDataset,
)
from wm3d_v3.training.train import (
    _window_config,
    apply_direct_policy_oxe_overrides,
    build_datasets,
)


SCHEMA = "wm3d_v8_stage0_causal_dual_view_v1"
REPRESENTATION = "wm3d_v8_vggt_observed_context_target_split_v1"


def _causal_metadata(*, leakage: bool = False) -> dict[str, np.ndarray]:
    return {
        "schema": np.asarray(SCHEMA),
        "representation": np.asarray(REPRESENTATION),
        "context_future_leakage": np.asarray(leakage),
        "target_usage": np.asarray("supervision_only"),
        "geometry_coordinate_frame": np.asarray("first_observed_camera"),
        "context_frames": np.asarray(3, dtype=np.int64),
        "future_frames": np.asarray(2, dtype=np.int64),
        "context_forward_frames": np.asarray(3, dtype=np.int64),
        "target_forward_frames": np.asarray(5, dtype=np.int64),
        "target_observed_outputs_discarded": np.asarray(3, dtype=np.int64),
    }


def _write_oxe_fixture(
    root: Path,
    *,
    window_root: Path | None = None,
    leakage: bool = False,
    schema: str = SCHEMA,
) -> OXEClipRecord:
    clip_id = "toy/episode"
    safe_id = "toy__episode"
    (root / "actions").mkdir(parents=True)
    causal_root = window_root or root
    (causal_root / "causal_windows").mkdir(parents=True)
    actions = np.zeros((5, 7), dtype=np.float32)
    actions[:, 0] = np.arange(5, dtype=np.float32)
    np.save(root / "actions" / f"{safe_id}.npy", actions)

    context_codes = np.zeros((3, 4, 6), dtype=np.int8)
    context_codes[:, :, 0] = np.asarray([2, 4, 6], dtype=np.int8)[:, None]
    future_codes = np.zeros((2, 4, 6), dtype=np.int8)
    future_codes[:, :, 0] = np.asarray([40, 60], dtype=np.int8)[:, None]
    payload = _causal_metadata(leakage=leakage)
    payload["schema"] = np.asarray(schema)
    payload.update(
        {
            "clip_id": np.asarray(clip_id),
            "start": np.asarray(0, dtype=np.int64),
            "context_codes": context_codes,
            "context_scale": np.full((3, 1, 1), 0.5, dtype=np.float16),
            "future_codes": future_codes,
            "future_scale": np.full((2, 1, 1), 0.25, dtype=np.float16),
            "future_depth_patch": np.full((2, 8, 8), 1.5, dtype=np.float16),
            "future_depth_conf_patch": np.full((2, 8, 8), 0.75, dtype=np.float16),
            "future_point_patch": np.full((2, 8, 8, 3), 2.5, dtype=np.float16),
            "future_point_conf_patch": np.full((2, 8, 8), 0.5, dtype=np.float16),
            "future_pose_enc": np.full((2, 9), 3.5, dtype=np.float16),
        }
    )
    np.savez_compressed(
        causal_root / "causal_windows" / f"{safe_id}__start_000000.npz",
        **payload,
    )
    return OXEClipRecord(
        clip_id=clip_id,
        dataset="toy",
        tar_path="unused.tar",
        pickle_member="unused.pkl",
        n_frames=5,
        fps=5,
        robot="test_robot",
        task_text="move",
    )


def _oxe_config(root: Path) -> WindowConfig:
    return WindowConfig(
        T=3,
        k=2,
        stride=1,
        cache_root=root,
        load_rgb=False,
        load_geom=False,
        load_state_tgt=True,
        load_geom_extra=True,
        require_geom_extra=True,
        window_geom_subdir="causal_windows",
        use_window_tokens=True,
        require_task_emb=False,
        causal_dual_view_required=True,
        causal_dual_view_representation=REPRESENTATION,
    )


def test_oxe_loader_keeps_observed_context_separate_from_future_targets(
    tmp_path: Path,
) -> None:
    """Catches concatenating target-forward tokens back into the input stream."""

    record = _write_oxe_fixture(tmp_path)
    dataset = OXEWindowDataset([record], _oxe_config(tmp_path))

    sample = dataset[0]

    assert sample["s_in"].shape == (3, 4, 6)
    assert "s_tgt" not in sample
    assert sample["s_tgt_codec"].shape == (2, 4, 6)
    assert sample["s_in"][:, 0, 0].tolist() == [1.0, 2.0, 3.0]
    assert sample["s_tgt_codec"][:, 0, 0].tolist() == [10.0, 15.0]
    assert sample["depth_tgt"].shape == (2, 8, 8)
    assert sample["point_tgt"].shape == (2, 8, 8, 3)
    assert sample["pose_geom_tgt"].shape == (2, 9)
    assert sample["action_tgt"][:, 0].tolist() == [3.0, 4.0]


def test_oxe_loader_reads_causal_windows_from_an_independent_cache_root(
    tmp_path: Path,
) -> None:
    """Catches replacing the base RGB/action cache root with the V8 window root."""

    base_root = tmp_path / "base"
    window_root = tmp_path / "v8_windows"
    record = _write_oxe_fixture(base_root, window_root=window_root)
    config = _oxe_config(base_root)
    config.window_geom_cache_root = window_root

    sample = OXEWindowDataset([record], config)[0]

    assert sample["s_in"].shape == (3, 4, 6)
    assert sample["action_tgt"][:, 0].tolist() == [3.0, 4.0]


def test_oxe_loader_skips_manifest_records_not_selected_by_sparse_causal_cache(
    tmp_path: Path,
) -> None:
    """Catches treating an intentionally sparse canary cache as corrupt data."""

    record = _write_oxe_fixture(tmp_path)
    missing = replace(record, clip_id="toy/missing")
    np.save(
        tmp_path / "actions" / "toy__missing.npy",
        np.zeros((5, 7), dtype=np.float32),
    )

    dataset = OXEWindowDataset([record, missing], _oxe_config(tmp_path))

    assert len(dataset) == 1
    assert dataset[0]["clip_id"] == "toy/episode"


@pytest.mark.parametrize(
    ("leakage", "schema", "message"),
    [
        (True, SCHEMA, "context_future_leakage"),
        (False, "wm3d_v7_compact_geom_v3", "causal dual-view schema"),
    ],
)
def test_oxe_loader_fails_closed_on_noncausal_window(
    tmp_path: Path,
    leakage: bool,
    schema: str,
    message: str,
) -> None:
    """Catches the causal loader silently accepting a future-leaking archive."""

    record = _write_oxe_fixture(tmp_path, leakage=leakage, schema=schema)

    with pytest.raises(ValueError, match=message):
        OXEWindowDataset([record], _oxe_config(tmp_path))

def test_window_config_threads_causal_dual_view_contract(tmp_path: Path) -> None:
    """Catches dropping the causal identity between YAML and the OXE loader."""

    config = _window_config(
        {
            "T": 3,
            "k": 2,
            "stride": 1,
            "cache_root": str(tmp_path),
            "window_geom_cache_root": str(tmp_path / "v8_windows"),
            "use_window_tokens": True,
            "causal_dual_view_required": True,
            "causal_dual_view_representation": REPRESENTATION,
        }
    )

    assert config.causal_dual_view_required is True
    assert config.causal_dual_view_representation == REPRESENTATION
    assert config.window_geom_cache_root == tmp_path / "v8_windows"


def test_direct_policy_override_allows_only_causal_loader_keys() -> None:
    """Catches mixed-source override plumbing rejecting the causal contract."""

    result = apply_direct_policy_oxe_overrides(
        {"manifest": "source.jsonl"},
        {
            "direct_policy_oxe_overrides": {
                "window_geom_cache_root": "/v8/windows",
                "causal_dual_view_required": True,
                "causal_dual_view_representation": REPRESENTATION,
            }
        },
    )

    assert result["causal_dual_view_required"] is True
    assert result["causal_dual_view_representation"] == REPRESENTATION
    assert result["window_geom_cache_root"] == "/v8/windows"


def _write_compact_fixture(
    root: Path,
    *,
    duplicate_starts: bool = False,
    schema: str = SCHEMA,
    split: str = "train",
    clip_hash: str = "compact_clip",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / f"{clip_hash}.npz"
    index_path = root / f"{clip_hash}.jsonl"
    starts = np.asarray([0, 0] if duplicate_starts else [0, 2], dtype=np.int64)
    context_codes = np.zeros((2, 3, 4, 6), dtype=np.int8)
    context_codes[0, :, :, 0] = np.asarray([2, 4, 6], dtype=np.int8)[:, None]
    context_codes[1, :, :, 0] = np.asarray([8, 10, 12], dtype=np.int8)[:, None]
    wrist_codes = context_codes + np.int8(2)
    future_codes = np.zeros((2, 2, 4, 6), dtype=np.int8)
    future_codes[0, :, :, 0] = np.asarray([40, 60], dtype=np.int8)[:, None]
    future_codes[1, :, :, 0] = np.asarray([80, 100], dtype=np.int8)[:, None]
    actions = np.zeros((7, 7), dtype=np.float32)
    actions[:, 0] = np.arange(7, dtype=np.float32)
    payload = _causal_metadata()
    payload["schema"] = np.asarray(schema)
    payload.update(
        {
            "clip_hash": np.asarray(clip_hash),
            "split": np.asarray(split),
            "source": np.asarray("robocasa365"),
            "action_adapter_version": np.asarray(
                "wm3d_v7_base_delta_axisangle_gripclose_v1"
            ),
            "action_audit_sha256": np.asarray("audit_sha"),
            "window_starts": starts,
            "context_codes": context_codes,
            "context_scale": np.full((2, 3, 1, 1), 0.5, dtype=np.float16),
            "wrist_context_codes": wrist_codes,
            "wrist_context_scale": np.full(
                (2, 3, 1, 1), 0.5, dtype=np.float16
            ),
            "future_codes": future_codes,
            "future_scale": np.full((2, 2, 1, 1), 0.25, dtype=np.float16),
            "future_depth_patch": np.full((2, 2, 8, 8), 1.5, dtype=np.float16),
            "future_depth_conf_patch": np.full(
                (2, 2, 8, 8), 0.75, dtype=np.float16
            ),
            "future_point_patch": np.full(
                (2, 2, 8, 8, 3), 2.5, dtype=np.float16
            ),
            "future_point_conf_patch": np.full(
                (2, 2, 8, 8), 0.5, dtype=np.float16
            ),
            "future_pose_enc": np.full((2, 2, 9), 3.5, dtype=np.float16),
            "task_text": np.asarray("move block"),
            "task_emb": np.ones(2048, dtype=np.float16),
            "action_valid_mask": np.ones(7, dtype=np.bool_),
            "actions": actions,
        }
    )
    np.savez_compressed(archive_path, **payload)
    row = {
        "schema": schema,
        "representation": REPRESENTATION,
        "context_future_leakage": False,
        "target_usage": "supervision_only",
        "geometry_coordinate_frame": "first_observed_camera",
        "clip_hash": clip_hash,
        "split": split,
        "source": "robocasa365",
        "path": str(archive_path),
        "model_frames": 7,
        "windows": len(starts),
        "window_starts": starts.tolist(),
        "paired_views": True,
        "action_valid": True,
        "action_adapter_version": "wm3d_v7_base_delta_axisangle_gripclose_v1",
        "action_audit_sha256": "audit_sha",
        "pseudo_outcomes": False,
        "geometry_teacher": {"pseudo_teacher": True, "confidence_stored": True},
    }
    index_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return index_path


def _compact_config(index_path: Path) -> V7CompactDatasetConfig:
    return V7CompactDatasetConfig(
        index_path=index_path,
        split="train",
        T=3,
        k=2,
        stride=1,
        require_action_stats=False,
        causal_dual_view_required=True,
        causal_dual_view_representation=REPRESENTATION,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compact_loader_maps_each_start_to_one_dual_view_window(
    tmp_path: Path,
) -> None:
    """Catches reconstructing W order from segments instead of window_starts."""

    dataset = V7CompactWindowDataset(
        _compact_config(_write_compact_fixture(tmp_path))
    )

    first = dataset[0]
    second = dataset[1]

    assert len(dataset) == 2
    assert first["start"] == 0 and second["start"] == 2
    assert first["s_in"][:, 0, 0].tolist() == [1.0, 2.0, 3.0]
    assert second["s_in"][:, 0, 0].tolist() == [4.0, 5.0, 6.0]
    assert first["s_wrist"][:, 0, 0].tolist() == [2.0, 3.0, 4.0]
    assert first["s_tgt_codec"][:, 0, 0].tolist() == [10.0, 15.0]
    assert second["s_tgt_codec"][:, 0, 0].tolist() == [20.0, 25.0]
    assert second["action_tgt"][:, 0].tolist() == [4.0, 5.0]
    assert second["depth_tgt"].shape == (2, 8, 8)
    assert second["point_tgt"].shape == (2, 8, 8, 3)


def test_compact_trusted_index_fast_init_is_digest_bound_and_lazy(
    tmp_path: Path,
) -> None:
    index_path = _write_compact_fixture(tmp_path)
    archive_path = Path(json.loads(index_path.read_text())["path"])
    config = replace(
        _compact_config(index_path),
        trusted_index_fast_init=True,
        trusted_index_sha256=_sha256(index_path),
    )

    archive_path.rename(archive_path.with_suffix(".hidden"))
    dataset = V7CompactWindowDataset(config)
    with pytest.raises(FileNotFoundError):
        dataset[0]

    with pytest.raises(RuntimeError, match="trusted compact index digest mismatch"):
        V7CompactWindowDataset(
            replace(config, trusted_index_sha256="0" * 64)
        )


def test_compact_trusted_index_fast_init_revalidates_sampled_payload(
    tmp_path: Path,
) -> None:
    index_path = _write_compact_fixture(tmp_path)
    row = json.loads(index_path.read_text())
    archive_path = Path(row["path"])
    config = replace(
        _compact_config(index_path),
        trusted_index_fast_init=True,
        trusted_index_sha256=_sha256(index_path),
    )
    dataset = V7CompactWindowDataset(config)

    with np.load(archive_path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["window_starts"] = np.asarray([0, 1], dtype=np.int64)
    np.savez_compressed(archive_path, **payload)

    with pytest.raises(ValueError, match="compact window_starts identity mismatch"):
        dataset[0]


@pytest.mark.parametrize(
    ("duplicate_starts", "schema", "message"),
    [
        (True, SCHEMA, "window_starts contains duplicates"),
        (False, "wm3d_v7_compact_geom_v3", "causal dual-view schema"),
    ],
)
def test_compact_loader_fails_closed_on_invalid_window_identity(
    tmp_path: Path,
    duplicate_starts: bool,
    schema: str,
    message: str,
) -> None:
    """Catches accepting ambiguous W identity or a legacy compact payload."""

    index_path = _write_compact_fixture(
        tmp_path, duplicate_starts=duplicate_starts, schema=schema
    )

    with pytest.raises(ValueError, match=message):
        V7CompactWindowDataset(_compact_config(index_path))


def test_build_datasets_threads_compact_causal_contract(tmp_path: Path) -> None:
    """Catches dropping compact causal mode in v7_mixed/v7_compact builders."""

    train_index = _write_compact_fixture(
        tmp_path / "train", split="train", clip_hash="train_clip"
    )
    val_index = _write_compact_fixture(
        tmp_path / "val", split="val", clip_hash="val_clip"
    )
    combined = tmp_path / "combined.jsonl"
    combined.write_text(
        train_index.read_text(encoding="utf-8")
        + val_index.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    train_dataset, val_dataset = build_datasets(
        {
            "model": {},
            "data": {
                "dataset_type": "v7_compact",
                "compact_index": str(combined),
                "T": 3,
                "k": 2,
                "stride": 1,
                "require_task_emb": True,
                "require_action_stats": False,
                "compact_causal_dual_view_required": True,
                "compact_causal_dual_view_representation": REPRESENTATION,
            },
        }
    )

    assert train_dataset.cfg.causal_dual_view_required is True
    assert val_dataset.cfg.causal_dual_view_required is True
    assert train_dataset.cfg.causal_dual_view_representation == REPRESENTATION
    assert val_dataset.cfg.causal_dual_view_representation == REPRESENTATION
