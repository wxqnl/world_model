from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from wm3d.data.manifest_contract import sha256_file
from wm3d.training.offline_eval import (
    OfflineEvalError,
    declared_eval_coverage_lanes,
    validate_eval_coverage,
)


def _objective() -> SimpleNamespace:
    return SimpleNamespace(
        token_mse=1.0,
        token_cosine=0.1,
        rgb_l1=0.5,
        rgb_charbonnier=2.0,
        rgb_gradient=0.5,
        rgb_perceptual=0.1,
        depth_log=1.5,
        point=0.5,
        camera_pose=0.1,
        action_fine=2.0,
        action_coarse=1.0,
    )


def _metrics() -> dict[str, float]:
    return {
        "native_token_supervised_elements": 100.0,
        "rgb_supervised_elements": 100.0,
        "depth_supervised_elements": 100.0,
        "point_supervised_elements": 100.0,
        "camera_pose_supervised_elements": 100.0,
        "fine_supervised_dimensions": 100.0,
        "fine_continuous_supervised_dimensions": 90.0,
        "fine_binary_supervised_dimensions": 10.0,
        "coarse_supervised_dimensions": 100.0,
        "current_state_supervised_dimensions": 100.0,
    }


def test_eval_coverage_requires_every_enabled_lane() -> None:
    expected = frozenset(_metrics())
    assert validate_eval_coverage(_metrics(), expected_lanes=expected) == _metrics()
    for name in (
        "native_token_supervised_elements",
        "rgb_supervised_elements",
        "depth_supervised_elements",
        "point_supervised_elements",
        "camera_pose_supervised_elements",
        "fine_supervised_dimensions",
        "coarse_supervised_dimensions",
        "current_state_supervised_dimensions",
    ):
        metrics = _metrics()
        metrics[name] = 0.0
        with pytest.raises(OfflineEvalError, match="zero"):
            validate_eval_coverage(metrics, expected_lanes=expected)


def test_fine_only_profile_does_not_require_fabricated_coarse_coverage() -> None:
    metrics = _metrics()
    metrics["coarse_supervised_dimensions"] = 0.0
    expected = frozenset(set(metrics) - {"coarse_supervised_dimensions"})
    assert (
        validate_eval_coverage(metrics, expected_lanes=expected)[
            "coarse_supervised_dimensions"
        ]
        == 0.0
    )


def test_declared_coarse_lane_with_zero_coverage_is_rejected() -> None:
    metrics = _metrics()
    metrics["coarse_supervised_dimensions"] = 0.0
    with pytest.raises(OfflineEvalError, match="coarse_supervised_dimensions"):
        validate_eval_coverage(metrics, expected_lanes=frozenset(_metrics()))


def test_action_coverage_requires_real_continuous_dimensions() -> None:
    metrics = _metrics()
    metrics["fine_continuous_supervised_dimensions"] = 0.0
    with pytest.raises(OfflineEvalError, match="fine_continuous"):
        validate_eval_coverage(metrics, expected_lanes=frozenset(_metrics()))


def _adapter(path: Path, *, group: str, supervision: str) -> SimpleNamespace:
    interval = None if supervision == "fine_command" else "interval"
    action_time = "timestamp" if supervision == "fine_command" else None
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "wm3d_v8_source_adapter_v3",
                "name": group,
                "raw_format": "npz",
                "observation_time_key": "timestamp",
                "views": [{"name": "head", "key": "rgb"}],
                "groups": [
                    {
                        "group": group,
                        "supervision": supervision,
                        "action": [
                            {
                                "key": "action",
                                "columns": [0],
                                "scale": [1.0],
                                "offset": [0.0],
                            }
                        ],
                        "state": [
                            {
                                "key": "state",
                                "columns": [0],
                                "scale": [1.0],
                                "offset": [0.0],
                            }
                        ],
                        "action_time_key": action_time,
                        "state_time_key": "timestamp",
                        "world_interval_index_key": interval,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        adapter_config_path=path,
        adapter_contract_sha256=sha256_file(path),
    )


def test_declared_lanes_only_use_sources_active_in_eval_split(tmp_path: Path) -> None:
    fine = _adapter(tmp_path / "fine.yaml", group="fine_arm", supervision="fine_command")
    coarse = _adapter(
        tmp_path / "coarse.yaml", group="coarse_arm", supervision="coarse_effect"
    )
    fine.name = "fine_source"
    fine.embodiment = "fine_robot"
    coarse.name = "coarse_source"
    coarse.embodiment = "coarse_robot"
    profile = SimpleNamespace(
        sources=(fine, coarse),
        cache={
            "rgb_codec": "jpeg_pack",
            "depth_codec": "fp16",
            "point_codec": "fp16",
            "camera_pose_codec": "fp32",
        },
        embodiments={
            "fine_robot": SimpleNamespace(
                groups=(
                    SimpleNamespace(
                        name="fine_arm", action_semantics=("controller_command",)
                    ),
                )
            ),
            "coarse_robot": SimpleNamespace(
                groups=(
                    SimpleNamespace(
                        name="coarse_arm", action_semantics=("joint_delta_rad",)
                    ),
                )
            ),
        },
    )
    fine_only = declared_eval_coverage_lanes(
        profile,
        _objective(),
        active_source_names=frozenset({"fine_source"}),
    )
    assert "fine_supervised_dimensions" in fine_only
    assert "coarse_supervised_dimensions" not in fine_only
    metrics = _metrics()
    metrics["coarse_supervised_dimensions"] = 0.0
    validate_eval_coverage(metrics, expected_lanes=fine_only)

    coarse_active = declared_eval_coverage_lanes(
        profile,
        _objective(),
        active_source_names=frozenset({"coarse_source"}),
    )
    assert "coarse_supervised_dimensions" in coarse_active
    with pytest.raises(OfflineEvalError, match="coarse_supervised_dimensions"):
        validate_eval_coverage(metrics, expected_lanes=coarse_active)
