from __future__ import annotations

import numpy as np
import pytest
import torch

from wm3d.data.grouped_robot import ACTION_SEMANTIC_IDS
from wm3d.models.native_world_model import NativeWorldModel, NativeWorldModelConfig
from wm3d.serving.panda_libero import (
    PANDA_LIBERO_HISTORY_WORLD_INTERVALS,
    PANDA_LIBERO_POLICY_HISTORY,
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
    history = np.zeros((PANDA_LIBERO_POLICY_HISTORY, 7), dtype=np.float32)
    history[:, 0] = np.arange(1, PANDA_LIBERO_POLICY_HISTORY + 1) * 0.01
    history[:, 6] = np.arange(PANDA_LIBERO_POLICY_HISTORY) >= 8
    offset = np.asarray((0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), np.float32)
    scale = np.asarray((0.02, 0.02, 0.02, 0.1, 0.1, 0.1, 1.0), np.float32)
    state_offset = np.asarray((0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0), np.float32)
    state_scale = np.asarray((0.05, 1, 1, 1, 1, 1, 1, 1, 1, 1), np.float32)
    packed = panda_libero_policy_inputs(
        np.asarray((0.1, -0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.25), np.float32),
        history,
        offset,
        scale,
        state_offset,
        state_scale,
    ).model_kwargs()

    assert packed["policy_only"] is True
    assert "composition_operator_ids" not in packed
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
        packed["history_fine_action_values"][
            0, -PANDA_LIBERO_HISTORY_WORLD_INTERVALS :, 0, :4, :7
        ],
        torch.from_numpy(((history - offset[None]) / scale[None]).reshape(4, 4, 7)),
    )
    assert packed["history_fine_action_mask"].sum().item() == 16 * 7
    torch.testing.assert_close(
        packed["history_fine_action_dt"][0, -4:, 0, :4],
        (torch.arange(4) / 20.0).repeat(4, 1),
    )
    assert packed["history_fine_sample_mask"].sum().item() == 16
    assert packed["history_coarse_action_mask"].sum().item() == 0
    assert packed["future_factual_fine_action_mask"].sum().item() == 0
    assert packed["future_factual_coarse_action_mask"].sum().item() == 0
    assert packed["current_state_values"][0, 0, 0].item() == pytest.approx(1.0)
    torch.testing.assert_close(
        packed["state_normalization_offset"][0, 0, :10],
        torch.from_numpy(state_offset),
    )
    torch.testing.assert_close(
        packed["state_normalization_scale"][0, 0, :10],
        torch.from_numpy(state_scale),
    )


def test_panda_packer_runs_full_policy_model_and_action_consumer() -> None:
    cfg = NativeWorldModelConfig(
        T=4,
        P=4,
        K=8,
        token_dim=16,
        task_dim=12,
        num_views=2,
        state_hidden=32,
        state_layers=2,
        state_heads=4,
        state_ff_mult=2.0,
        action_hidden=24,
        action_layers=2,
        action_heads=4,
        action_ff_mult=2.0,
        bridge_layers_state=(1,),
        factual_v7_bridge_layers_state=(0,),
        bridge_heads=4,
        dynamics_layers=1,
        factual_v7_early_action_conditioning=True,
        factual_v7_early_action_scale=1.0,
        view_hidden=16,
        view_heads=4,
        view_ff_mult=2.0,
        max_action_groups=8,
        max_action_dim=16,
        max_state_dim=32,
        max_action_substeps=4,
        max_policy_queries=8,
        max_group_id=16,
        max_embodiments=8,
        max_action_semantic_id=16,
        max_state_semantic_id=16,
        time_fourier_dim=8,
        max_aux_tokens=2,
        aux_dim=8,
        max_aux_type_id=8,
        rgb_hidden=16,
        rgb_res_blocks=1,
        rgb_decode_chunk_size=1,
        rgb_size=16,
        rgb_decode_indices=tuple(range(8)),
        geom_hidden=16,
        activation_checkpointing=False,
    )
    packed = panda_libero_policy_inputs(
        np.zeros(10, dtype=np.float32),
        np.zeros((PANDA_LIBERO_POLICY_HISTORY, 7), dtype=np.float32),
        np.zeros(7, dtype=np.float32),
        np.ones(7, dtype=np.float32),
        np.zeros(10, dtype=np.float32),
        np.ones(10, dtype=np.float32),
        context_steps=cfg.T,
        world_horizon=cfg.K,
        max_groups=cfg.max_action_groups,
        max_action_dim=cfg.max_action_dim,
        max_state_dim=cfg.max_state_dim,
        max_substeps=cfg.max_action_substeps,
        max_policy_queries=cfg.max_policy_queries,
    )
    kwargs = packed.model_kwargs()
    kwargs.update(
        {
            "world_tokens": torch.randn(
                1, cfg.T, cfg.num_views, cfg.P, cfg.token_dim
            ),
            "view_mask": torch.ones(1, cfg.T, cfg.num_views, dtype=torch.bool),
            "world_times_s": torch.arange(
                cfg.T + cfg.K, dtype=torch.float32
            )[None],
            "task_embedding": torch.randn(1, cfg.task_dim),
        }
    )

    with torch.no_grad():
        output = NativeWorldModel(cfg).eval()(**kwargs)
    chunk = panda_action_chunk_from_model_output(
        output,
        packed.tensors["action_semantic_ids"],
        packed.tensors["action_group_ids"],
        packed.tensors["action_group_mask"],
        packed.tensors["embodiment_ids"],
    )

    assert chunk.canonical_close01.shape == (PANDA_LIBERO_POLICY_HORIZON, 7)
    assert chunk.libero_signed.shape == (PANDA_LIBERO_POLICY_HORIZON, 7)


def test_libero_policy_inputs_reject_nonidentity_gripper_normalization() -> None:
    with pytest.raises(PandaLiberoContractError, match="identity normalization"):
        panda_libero_policy_inputs(
            np.zeros(10, dtype=np.float32),
            np.zeros((PANDA_LIBERO_POLICY_HISTORY, 7), dtype=np.float32),
            np.zeros(7, dtype=np.float32),
            np.ones(7, dtype=np.float32) * 2.0,
            np.zeros(10, dtype=np.float32),
            np.ones(10, dtype=np.float32),
        )


def test_libero_policy_inputs_reject_old_h4_coarse_history_abi() -> None:
    with pytest.raises(PandaLiberoContractError, match="H16x7"):
        panda_libero_policy_inputs(
            np.zeros(10, dtype=np.float32),
            np.zeros((4, 7), dtype=np.float32),
            np.zeros(7, dtype=np.float32),
            np.ones(7, dtype=np.float32),
            np.zeros(10, dtype=np.float32),
            np.ones(10, dtype=np.float32),
        )


def test_libero_policy_inputs_reject_missing_state_calibration() -> None:
    state_scale = np.ones(10, dtype=np.float32)
    state_scale[0] = 0.0
    with pytest.raises(PandaLiberoContractError, match="state normalization"):
        panda_libero_policy_inputs(
            np.zeros(10, dtype=np.float32),
            np.zeros((PANDA_LIBERO_POLICY_HISTORY, 7), dtype=np.float32),
            np.zeros(7, dtype=np.float32),
            np.ones(7, dtype=np.float32),
            np.zeros(10, dtype=np.float32),
            state_scale,
        )
