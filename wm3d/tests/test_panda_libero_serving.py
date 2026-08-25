from __future__ import annotations

import numpy as np
import pytest
import torch

from wm3d.data.grouped_robot import ACTION_SEMANTIC_IDS
from wm3d.serving.panda_libero import (
    PANDA_LIBERO_POLICY_HORIZON,
    PANDA_ROBOCASA_LIBERO_ARM_GROUP_ID,
    PANDA_ROBOCASA_LIBERO_EMBODIMENT_ID,
    PandaLiberoContractError,
    panda_action_chunk_from_model_output,
    panda_libero_policy_inputs,
    panda_state_from_libero_observation,
)


def _semantics() -> torch.Tensor:
    return torch.tensor(
        [
            [
                ACTION_SEMANTIC_IDS["delta_position_m"],
                ACTION_SEMANTIC_IDS["delta_position_m"],
                ACTION_SEMANTIC_IDS["delta_position_m"],
                ACTION_SEMANTIC_IDS["delta_rotation_axis_angle_rad"],
                ACTION_SEMANTIC_IDS["delta_rotation_axis_angle_rad"],
                ACTION_SEMANTIC_IDS["delta_rotation_axis_angle_rad"],
                ACTION_SEMANTIC_IDS["absolute_gripper_close01"],
            ]
        ],
        dtype=torch.long,
    ).view(1, 1, 7)


def test_libero_observation_maps_to_panda_state_contract() -> None:
    state = panda_state_from_libero_observation(
        {
            "robot0_eef_pos": np.asarray((0.1, -0.2, 0.3), np.float32),
            "robot0_eef_quat": np.asarray((0.0, 0.0, 0.0, 1.0), np.float32),
            "robot0_gripper_qpos": np.asarray((0.04, -0.04), np.float32),
        }
    )
    np.testing.assert_allclose(state[:3], (0.1, -0.2, 0.3))
    np.testing.assert_allclose(state[3:9], (1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    assert state[9] == pytest.approx(0.0, abs=1.0e-6)


def test_model_action_chunk_maps_close_positive_to_libero_signed() -> None:
    action = torch.tensor(
        [
            [
                [
                    [0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.2],
                    [0.4, 0.5, 0.6, 0.04, 0.05, 0.06, 0.8],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    chunk = panda_action_chunk_from_model_output(
        {
            "policy_action": action,
            "policy_action_mask": torch.ones_like(action, dtype=torch.bool),
        },
        _semantics(),
        torch.tensor([[PANDA_ROBOCASA_LIBERO_ARM_GROUP_ID]]),
        torch.tensor([[True]]),
        torch.tensor([PANDA_ROBOCASA_LIBERO_EMBODIMENT_ID]),
    )
    assert chunk.canonical_close01.shape == (2, 7)
    np.testing.assert_array_equal(chunk.libero_signed[:, 6], (-1.0, 1.0))


def test_model_action_chunk_rejects_open_positive_semantic() -> None:
    semantics = _semantics()
    semantics[0, 0, 6] = ACTION_SEMANTIC_IDS["absolute_gripper_open01"]
    action = torch.zeros(1, 1, 1, 7)
    with pytest.raises(PandaLiberoContractError, match="close01"):
        panda_action_chunk_from_model_output(
            {
                "policy_action": action,
                "policy_action_mask": torch.ones_like(action, dtype=torch.bool),
            },
            semantics,
            torch.tensor([[PANDA_ROBOCASA_LIBERO_ARM_GROUP_ID]]),
            torch.tensor([[True]]),
            torch.tensor([PANDA_ROBOCASA_LIBERO_EMBODIMENT_ID]),
        )


def test_model_action_chunk_rejects_wrong_panda_identity() -> None:
    action = torch.zeros(1, 1, 1, 7)
    with pytest.raises(PandaLiberoContractError, match="shared Panda"):
        panda_action_chunk_from_model_output(
            {
                "policy_action": action,
                "policy_action_mask": torch.ones_like(action, dtype=torch.bool),
            },
            _semantics(),
            torch.tensor([[PANDA_ROBOCASA_LIBERO_ARM_GROUP_ID]]),
            torch.tensor([[True]]),
            torch.tensor([PANDA_ROBOCASA_LIBERO_EMBODIMENT_ID + 1]),
        )


def test_libero_policy_inputs_are_exact_k8_h1_panda_contract() -> None:
    history = np.zeros((4, 7), dtype=np.float32)
    history[:, 0] = (0.01, 0.02, 0.03, 0.04)
    history[:, 6] = (0.0, 0.0, 1.0, 1.0)
    packed = panda_libero_policy_inputs(
        np.asarray((0.1, -0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.25), np.float32),
        history,
        np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), np.float32),
        np.asarray((0.02, 0.02, 0.02, 0.1, 0.1, 0.1, 1.0), np.float32),
    ).model_kwargs()

    assert packed["embodiment_ids"].item() == PANDA_ROBOCASA_LIBERO_EMBODIMENT_ID
    assert packed["action_group_ids"][0, 0].item() == PANDA_ROBOCASA_LIBERO_ARM_GROUP_ID
    assert packed["action_group_mask"].sum().item() == 1
    query_mask = packed["policy_query_mask"][0, 0]
    assert query_mask.sum().item() == PANDA_LIBERO_POLICY_HORIZON
    torch.testing.assert_close(
        packed["policy_query_dt"][0, 0, :PANDA_LIBERO_POLICY_HORIZON],
        torch.arange(PANDA_LIBERO_POLICY_HORIZON) / 20.0,
    )
    torch.testing.assert_close(
        packed["history_coarse_action_values"][0, -4:, 0, :7],
        torch.from_numpy(history),
    )
    assert packed["history_fine_action_mask"].sum().item() == 0
    assert packed["future_factual_fine_action_mask"].sum().item() == 0
    assert packed["future_factual_coarse_action_mask"].sum().item() == 0


def test_libero_policy_inputs_reject_nonidentity_gripper_normalization() -> None:
    with pytest.raises(PandaLiberoContractError, match="identity normalization"):
        panda_libero_policy_inputs(
            np.zeros(10, dtype=np.float32),
            np.zeros((4, 7), dtype=np.float32),
            np.zeros(7, dtype=np.float32),
            np.ones(7, dtype=np.float32) * 2.0,
        )
