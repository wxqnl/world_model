from __future__ import annotations

import numpy as np


def _named(eef, cream, butter, basket):
    return {
        "robot0": {"eef_pos": list(eef)},
        "cream_cheese_1": {
            "pos": list(cream),
            "to_robot0_eef_pos": list(np.asarray(cream) - np.asarray(eef)),
        },
        "butter_1": {
            "pos": list(butter),
            "to_robot0_eef_pos": list(np.asarray(butter) - np.asarray(eef)),
        },
        "basket_1": {"pos": list(basket)},
    }


def test_extract_named_poses_groups_libero_keys():
    from wm3d_v3.benchmarks.libero_remote_runner import _extract_named_poses

    obs = {
        "robot0_eef_pos": np.asarray([0.0, 0.1, 0.2], dtype=np.float32),
        "cream_cheese_1_pos": np.asarray([0.2, 0.3, 0.4], dtype=np.float32),
        "cream_cheese_1_to_robot0_eef_pos": np.asarray([0.2, 0.2, 0.2], dtype=np.float32),
        "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
    }

    named = _extract_named_poses(obs)

    assert named["robot0"]["eef_pos"] == [0.0, 0.10000000149011612, 0.20000000298023224]
    assert named["cream_cheese_1"]["pos"] == [
        0.20000000298023224,
        0.30000001192092896,
        0.4000000059604645,
    ]
    assert named["cream_cheese_1"]["to_robot0_eef_pos"] == [
        0.20000000298023224,
        0.20000000298023224,
        0.20000000298023224,
    ]
    assert "agentview_image" not in named


def test_object_contact_eval_reports_contact_without_receptacle_progress():
    from wm3d_v3.benchmarks.libero_object_contact_eval import evaluate

    rollout = {
        "trace_schema_version": 2,
        "success_rate": 0.0,
        "results": [
            {
                "success": False,
                "steps": 3,
                "task_id": 1,
                "task_name": "synthetic",
                "instruction": "put both objects in basket",
                "step_trace": [
                    {"named_poses": _named((0.0, 0.0, 0.0), (0.4, 0.0, 0.0), (0.6, 0.0, 0.0), (1.0, 0.0, 0.0))},
                    {"named_poses": _named((0.39, 0.0, 0.0), (0.4, 0.0, 0.0), (0.6, 0.0, 0.0), (1.0, 0.0, 0.0))},
                    {"named_poses": _named((0.39, 0.0, 0.0), (0.4, 0.0, 0.0), (0.6, 0.0, 0.0), (1.0, 0.0, 0.0))},
                ],
            }
        ],
    }
    expert_trace = [
        _named((0.4, 0.0, 0.0), (0.4, 0.0, 0.0), (0.6, 0.0, 0.0), (1.0, 0.0, 0.0)),
        _named((1.0, 0.0, 0.0), (1.0, 0.02, 0.0), (1.0, -0.02, 0.0), (1.0, 0.0, 0.0)),
    ]

    report = evaluate(
        rollout,
        expert_trace=expert_trace,
        target_objects=["cream_cheese_1", "butter_1"],
        receptacle="basket_1",
        contact_threshold=0.08,
        receptacle_xy_threshold=0.14,
    )

    episode = report["episode_metrics"][0]
    assert report["stage_score_mean"] == 0.25
    assert episode["contact_objects_hit"] == 1
    assert episode["receptacle_objects_hit"] == 0
    assert episode["diagnosis"] == "contact_without_receptacle_progress"


def test_object_contact_eval_detects_missing_named_trace():
    from wm3d_v3.benchmarks.libero_object_contact_eval import evaluate

    report = evaluate(
        {"results": [{"success": False, "step_trace": [{"object_state": [0.0]}]}]},
        expert_trace=[],
        target_objects=["cream_cheese_1"],
        receptacle="basket_1",
        contact_threshold=0.08,
        receptacle_xy_threshold=0.14,
    )

    episode = report["episode_metrics"][0]
    assert episode["named_pose_steps"] == 0
    assert episode["diagnosis"] == "missing_named_pose_trace"
