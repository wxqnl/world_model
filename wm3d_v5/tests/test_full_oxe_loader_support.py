from __future__ import annotations

import numpy as np


def test_pick_image_key_supports_taco_rgb_static():
    from wm3d_v3.data.oxe_loader import _pick_image_key

    obs = {
        "depth_static": np.zeros((2, 2), dtype=np.float32),
        "rgb_static": np.zeros((2, 2, 3), dtype=np.uint8),
        "rgb_gripper": np.zeros((2, 2, 3), dtype=np.uint8),
    }

    assert _pick_image_key(obs) == "rgb_static"


def test_pick_image_key_supports_taco_rgb_gripper_fallback():
    from wm3d_v3.data.oxe_loader import _pick_image_key

    obs = {
        "depth_gripper": np.zeros((2, 2), dtype=np.float32),
        "rgb_gripper": np.zeros((2, 2, 3), dtype=np.uint8),
    }

    assert _pick_image_key(obs) == "rgb_gripper"


def test_kuka_action_normalization_uses_world_rotation_and_closedness():
    from wm3d_v3.data.action_normalize import normalize_action

    action = {
        "world_vector": np.array([0.1, -0.2, 0.3], dtype=np.float32),
        "rotation_delta": np.array([0.01, -0.02, 0.03], dtype=np.float32),
        "gripper_closedness_action": np.array([1.0], dtype=np.float32),
    }

    out = normalize_action(action, "kuka")

    assert out.dtype == np.float32
    assert out.shape == (7,)
    assert np.allclose(out[:6], [0.1, -0.2, 0.3, 0.01, -0.02, 0.03])
    assert out[6] == 1.0


def test_kuka_action_normalization_open_gripper():
    from wm3d_v3.data.action_normalize import normalize_action

    action = {
        "world_vector": np.zeros(3, dtype=np.float32),
        "rotation_delta": np.zeros(3, dtype=np.float32),
        "gripper_closedness_action": np.array([0.0], dtype=np.float32),
    }

    out = normalize_action(action, "kuka")

    assert out[6] == 0.0
