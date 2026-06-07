from __future__ import annotations

import numpy as np
import torch


def test_video_condition_bundle_to_moves_tensor_fields():
    from wm3d_v3.video_backends import VideoConditionBundle

    bundle = VideoConditionBundle(
        context_rgb=torch.zeros(1, 3, 16, 16),
        action_cond=torch.ones(1, 2, 7),
        task_text=["pick object"],
        extra={"foo": torch.ones(1)},
    )

    moved = bundle.to("cpu")

    assert moved.context_rgb.device.type == "cpu"
    assert moved.action_cond is not None
    assert moved.action_cond.device.type == "cpu"
    assert moved.task_text == ["pick object"]
    assert moved.extra is not None
    assert moved.extra["foo"].device.type == "cpu"


def test_control_bundle_split_indices_are_deterministic():
    from scripts.cache_control_bundle import split_indices

    cfg = {"data": {"seed": 123, "val_frac": 0.2}}

    val_a = split_indices(20, cfg, "val")
    val_b = split_indices(20, cfg, "val")
    train = split_indices(20, cfg, "train")

    assert val_a == val_b
    assert len(val_a) == 4
    assert len(set(val_a) & set(train)) == 0
    assert sorted(val_a + train) == list(range(20))


def test_tensor_to_numpy_dtype_conversion():
    from scripts.cache_control_bundle import tensor_to_numpy

    arr = tensor_to_numpy(torch.ones(2, 3), np.float16)

    assert arr.shape == (2, 3)
    assert arr.dtype == np.float16


def test_hunyuan_video_backend_prompt_helpers():
    from wm3d_v3.video_backends import (
        VideoConditionBundle,
        align_hunyuan_video_length,
        summarize_bundle_for_prompt,
    )

    assert align_hunyuan_video_length(8) == 9
    assert align_hunyuan_video_length(9) == 9

    bundle = VideoConditionBundle(
        context_rgb=torch.zeros(1, 3, 16, 16),
        task_text=["pick up the red block"],
        motion_hint=torch.full((1, 8, 1, 4, 4), 0.2),
        contact_hint=torch.full((1, 8, 1, 4, 4), 0.1),
    )

    prompt = summarize_bundle_for_prompt(bundle)

    assert "pick up the red block" in prompt
    assert "visible object and gripper motion" in prompt
    assert "contacting" in prompt
