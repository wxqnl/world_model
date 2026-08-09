from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wm3d_v3.data.manifest import OXEClipRecord
from wm3d_v3.data.window_dataset import OXEWindowDataset, WindowConfig
from wm3d_v3.training.train import (
    _window_config,
    apply_direct_policy_oxe_overrides,
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
    leakage: bool = False,
    schema: str = SCHEMA,
) -> OXEClipRecord:
    clip_id = "toy/episode"
    safe_id = "toy__episode"
    (root / "actions").mkdir(parents=True)
    (root / "causal_windows").mkdir(parents=True)
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
        root / "causal_windows" / f"{safe_id}__start_000000.npz",
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
    assert sample["s_tgt"].shape == (2, 4, 6)
    assert sample["s_in"][:, 0, 0].tolist() == [1.0, 2.0, 3.0]
    assert sample["s_tgt"][:, 0, 0].tolist() == [10.0, 15.0]
    assert sample["depth_tgt"].shape == (2, 8, 8)
    assert sample["point_tgt"].shape == (2, 8, 8, 3)
    assert sample["pose_geom_tgt"].shape == (2, 9)
    assert sample["action_tgt"][:, 0].tolist() == [3.0, 4.0]


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
            "use_window_tokens": True,
            "causal_dual_view_required": True,
            "causal_dual_view_representation": REPRESENTATION,
        }
    )

    assert config.causal_dual_view_required is True
    assert config.causal_dual_view_representation == REPRESENTATION


def test_direct_policy_override_allows_only_causal_loader_keys() -> None:
    """Catches mixed-source override plumbing rejecting the causal contract."""

    result = apply_direct_policy_oxe_overrides(
        {"manifest": "source.jsonl"},
        {
            "direct_policy_oxe_overrides": {
                "causal_dual_view_required": True,
                "causal_dual_view_representation": REPRESENTATION,
            }
        },
    )
