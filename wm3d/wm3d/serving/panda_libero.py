"""Strict Panda/LIBERO serving boundary for the unified WM3D action head."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from wm3d.data.grouped_robot import (
    ACTION_SEMANTIC_IDS,
    COMPOSITION_OPERATOR_IDS,
    STATE_SEMANTIC_IDS,
)


PANDA_ROBOCASA_LIBERO_EMBODIMENT_ID = 2
PANDA_ROBOCASA_LIBERO_ARM_GROUP_ID = 12
PANDA_LIBERO_CONTROLLER_HZ = 20
PANDA_LIBERO_POLICY_HORIZON = 8
PANDA_LIBERO_EXECUTION_HORIZON = 1
PANDA_LIBERO_POLICY_HISTORY = 16
PANDA_LIBERO_HISTORY_WORLD_INTERVALS = 4


class PandaLiberoContractError(ValueError):
    """Raised when model or simulator data violates the Panda action ABI."""


@dataclass(frozen=True)
class PandaLiberoActionChunk:
    """Canonical model actions and the exact commands passed to LIBERO."""

    canonical_close01: np.ndarray
    libero_signed: np.ndarray


@dataclass(frozen=True)
class PandaLiberoPolicyInputs:
    """Action/state tensors for the unified Stage0 head at a LIBERO decision."""

    tensors: Mapping[str, torch.Tensor]

    def model_kwargs(self) -> dict[str, object]:
        # Composition operators supervise physical trajectory construction;
        # they are not an input to NativeWorldModel.forward.
        result: dict[str, object] = {
            name: value
            for name, value in self.tensors.items()
            if name != "composition_operator_ids"
        }
        result["policy_only"] = True
        return result


def _finite_vector(value: object, *, name: str, minimum: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.size < minimum or not np.isfinite(result).all():
        raise PandaLiberoContractError(
            f"{name} must contain at least {minimum} finite values"
        )
    return result


def _quaternion_xyzw_to_rotation6d(quaternion: np.ndarray) -> np.ndarray:
    value = quaternion[:4].astype(np.float64, copy=False)
    norm = float(np.linalg.norm(value))
    if norm < 1.0e-12:
        raise PandaLiberoContractError("robot0_eef_quat has zero norm")
    x, y, z, w = value / norm
    rotation = np.asarray(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ),
        dtype=np.float32,
    )
    return rotation[:, :2].T.reshape(6)


def panda_state_from_libero_observation(
    observation: Mapping[str, object],
    *,
    gripper_open_width_m: float = 0.08,
    gripper_closed_width_m: float = 0.0,
) -> np.ndarray:
    """Return canonical ``xyz + rotation6d + gripper_close01`` state."""

    if not np.isfinite(gripper_open_width_m) or not np.isfinite(gripper_closed_width_m):
        raise PandaLiberoContractError("gripper calibration must be finite")
    if gripper_open_width_m <= gripper_closed_width_m:
        raise PandaLiberoContractError("gripper open width must exceed closed width")
    try:
        position = _finite_vector(
            observation["robot0_eef_pos"], name="robot0_eef_pos", minimum=3
        )
        quaternion = _finite_vector(
            observation["robot0_eef_quat"], name="robot0_eef_quat", minimum=4
        )
        gripper = _finite_vector(
            observation["robot0_gripper_qpos"],
            name="robot0_gripper_qpos",
            minimum=2,
        )
    except KeyError as exc:
        raise PandaLiberoContractError(
            f"LIBERO observation is missing {exc.args[0]!r}"
        ) from exc
    aperture = float(abs(gripper[0] - gripper[1]))
    open_fraction = (aperture - float(gripper_closed_width_m)) / float(
        gripper_open_width_m - gripper_closed_width_m
    )
    close01 = 1.0 - float(np.clip(open_fraction, 0.0, 1.0))
    return np.concatenate(
        (
            position[:3],
            _quaternion_xyzw_to_rotation6d(quaternion),
            np.asarray((close01,), dtype=np.float32),
        )
    ).astype(np.float32, copy=False)


def panda_libero_policy_inputs(
    current_state: object,
    native_action_history_close01: object,
    action_normalization_offset: object,
    action_normalization_scale: object,
    state_normalization_offset: object,
    state_normalization_scale: object,
    *,
    context_steps: int = 16,
    world_horizon: int = 8,
    max_groups: int = 8,
    max_action_dim: int = 16,
    max_state_dim: int = 32,
    max_substeps: int = 128,
    max_policy_queries: int = 256,
    device: torch.device | str | None = None,
) -> PandaLiberoPolicyInputs:
    """Pack the documented V8 direct-policy ABI for the unified model.

    ``native_action_history_close01`` is the causal H16 sequence of completed
    20 Hz controller commands required by the V8 action contract.  The commands
    occupy the final four 5 Hz world intervals as four real fine substeps per
    interval; they are never composed into coarse effects.  The future K8
    queries remain direct 20 Hz controller commands.  No future action is
    inserted into the action-free world/policy trunk.
    """

    state = np.asarray(current_state, dtype=np.float32)
    history = np.asarray(native_action_history_close01, dtype=np.float32)
    offset = np.asarray(action_normalization_offset, dtype=np.float32)
    scale = np.asarray(action_normalization_scale, dtype=np.float32)
    state_offset = np.asarray(state_normalization_offset, dtype=np.float32)
    state_scale = np.asarray(state_normalization_scale, dtype=np.float32)
    if state.shape != (10,) or not np.isfinite(state).all():
        raise PandaLiberoContractError("current_state must be finite canonical [10]")
    if history.shape != (PANDA_LIBERO_POLICY_HISTORY, 7) or not np.isfinite(
        history
    ).all():
        raise PandaLiberoContractError(
            "native action history must be finite canonical H16x7"
        )
    if np.any((history[:, 6] < 0.0) | (history[:, 6] > 1.0)):
        raise PandaLiberoContractError("history gripper must be close01")
    if offset.shape != (7,) or scale.shape != (7,):
        raise PandaLiberoContractError("action normalization must be canonical [7]")
    if not np.isfinite(offset).all() or not np.isfinite(scale).all() or np.any(
        scale <= 0.0
    ):
        raise PandaLiberoContractError("action normalization must be finite and positive")
    if offset[6] != 0.0 or scale[6] != 1.0:
        raise PandaLiberoContractError(
            "absolute gripper close01 must use identity normalization"
        )
    if state_offset.shape != (10,) or state_scale.shape != (10,):
        raise PandaLiberoContractError("state normalization must be canonical [10]")
    if (
        not np.isfinite(state_offset).all()
        or not np.isfinite(state_scale).all()
        or np.any(state_scale <= 0.0)
    ):
        raise PandaLiberoContractError(
            "state normalization must be finite and positive"
        )
    if state_offset[9] != 0.0 or state_scale[9] != 1.0:
        raise PandaLiberoContractError(
            "gripper state close01 must use identity normalization"
        )
    if (
        context_steps < PANDA_LIBERO_HISTORY_WORLD_INTERVALS
        or world_horizon != 8
        or max_groups < 1
        or max_action_dim < 7
        or max_state_dim < 10
        or max_substeps < 4
        or max_policy_queries < PANDA_LIBERO_POLICY_HORIZON
    ):
        raise PandaLiberoContractError("model capacities cannot represent the LIBERO ABI")

    action_shape = (1, max_groups, max_action_dim)
    group_ids = torch.zeros((1, max_groups), dtype=torch.long, device=device)
    group_mask = torch.zeros((1, max_groups), dtype=torch.bool, device=device)
    action_semantics = torch.zeros(action_shape, dtype=torch.long, device=device)
    state_semantics = torch.zeros(
        (1, max_groups, max_state_dim), dtype=torch.long, device=device
    )
    composition = torch.zeros(action_shape, dtype=torch.long, device=device)
    group_ids[0, 0] = PANDA_ROBOCASA_LIBERO_ARM_GROUP_ID
    group_mask[0, 0] = True
    action_semantics[0, 0, :7] = torch.tensor(
        (
            ACTION_SEMANTIC_IDS["delta_position_m"],
            ACTION_SEMANTIC_IDS["delta_position_m"],
            ACTION_SEMANTIC_IDS["delta_position_m"],
            ACTION_SEMANTIC_IDS["delta_rotation_axis_angle_rad"],
            ACTION_SEMANTIC_IDS["delta_rotation_axis_angle_rad"],
            ACTION_SEMANTIC_IDS["delta_rotation_axis_angle_rad"],
            ACTION_SEMANTIC_IDS["absolute_gripper_close01"],
        ),
        dtype=torch.long,
        device=device,
    )
    state_semantics[0, 0, :10] = torch.tensor(
        (
            STATE_SEMANTIC_IDS["eef_position_m"],
            STATE_SEMANTIC_IDS["eef_position_m"],
            STATE_SEMANTIC_IDS["eef_position_m"],
            STATE_SEMANTIC_IDS["eef_rotation_6d"],
            STATE_SEMANTIC_IDS["eef_rotation_6d"],
            STATE_SEMANTIC_IDS["eef_rotation_6d"],
            STATE_SEMANTIC_IDS["eef_rotation_6d"],
            STATE_SEMANTIC_IDS["eef_rotation_6d"],
            STATE_SEMANTIC_IDS["eef_rotation_6d"],
            STATE_SEMANTIC_IDS["gripper_close01"],
        ),
        dtype=torch.long,
        device=device,
    )
    composition[0, 0, :7] = torch.tensor(
        (
            COMPOSITION_OPERATOR_IDS["sum"],
            COMPOSITION_OPERATOR_IDS["sum"],
            COMPOSITION_OPERATOR_IDS["sum"],
            COMPOSITION_OPERATOR_IDS["so3_axis_angle_base_left"],
            COMPOSITION_OPERATOR_IDS["so3_axis_angle_base_left"],
            COMPOSITION_OPERATOR_IDS["so3_axis_angle_base_left"],
            COMPOSITION_OPERATOR_IDS["logical_last"],
        ),
        dtype=torch.long,
        device=device,
    )

    normalized_history = (history - offset[None]) / scale[None]
    history_fine = torch.zeros(
        (
            1,
            context_steps,
            max_groups,
            max_substeps,
            max_action_dim,
        ),
        dtype=torch.float32,
        device=device,
    )
    history_fine_mask = torch.zeros_like(history_fine, dtype=torch.bool)
    history_fine_dt = torch.zeros(
        history_fine.shape[:-1], dtype=torch.float32, device=device
    )
    history_fine_sample_mask = torch.zeros_like(history_fine_dt, dtype=torch.bool)
    grouped_history = torch.as_tensor(
        normalized_history.reshape(PANDA_LIBERO_HISTORY_WORLD_INTERVALS, 4, 7),
        dtype=torch.float32,
        device=device,
    )
    history_fine[
        0, -PANDA_LIBERO_HISTORY_WORLD_INTERVALS :, 0, :4, :7
    ] = grouped_history
    history_fine_mask[
        0, -PANDA_LIBERO_HISTORY_WORLD_INTERVALS :, 0, :4, :7
    ] = True
    history_fine_dt[
        0, -PANDA_LIBERO_HISTORY_WORLD_INTERVALS :, 0, :4
    ] = torch.arange(4, dtype=torch.float32, device=device) / float(
        PANDA_LIBERO_CONTROLLER_HZ
    )
    history_fine_sample_mask[
        0, -PANDA_LIBERO_HISTORY_WORLD_INTERVALS :, 0, :4
    ] = True
    history_coarse = torch.zeros(
        (1, context_steps, max_groups, max_action_dim),
        dtype=torch.float32,
        device=device,
    )
    history_coarse_mask = torch.zeros_like(history_coarse, dtype=torch.bool)

    query_dt = torch.zeros(
        (1, max_groups, max_policy_queries), dtype=torch.float32, device=device
    )
    query_mask = torch.zeros_like(query_dt, dtype=torch.bool)
    query_dt[0, 0, :PANDA_LIBERO_POLICY_HORIZON] = torch.arange(
        PANDA_LIBERO_POLICY_HORIZON, dtype=torch.float32, device=device
    ) / float(PANDA_LIBERO_CONTROLLER_HZ)
    query_mask[0, 0, :PANDA_LIBERO_POLICY_HORIZON] = True

    current_values = torch.zeros(
        (1, max_groups, max_state_dim), dtype=torch.float32, device=device
    )
    current_mask = torch.zeros_like(current_values, dtype=torch.bool)
    current_values[0, 0, :10] = torch.as_tensor(
        (state - state_offset) / state_scale,
        dtype=torch.float32,
        device=device,
    )
    current_mask[0, 0, :10] = True

    norm_offset = torch.zeros(action_shape, dtype=torch.float32, device=device)
    norm_scale = torch.ones(action_shape, dtype=torch.float32, device=device)
    norm_offset[0, 0, :7] = torch.as_tensor(offset, device=device)
    norm_scale[0, 0, :7] = torch.as_tensor(scale, device=device)
    state_norm_offset = torch.zeros(
        (1, max_groups, max_state_dim), dtype=torch.float32, device=device
    )
    state_norm_scale = torch.ones_like(state_norm_offset)
    state_norm_offset[0, 0, :10] = torch.as_tensor(state_offset, device=device)
    state_norm_scale[0, 0, :10] = torch.as_tensor(state_scale, device=device)

    future_fine_shape = (
        1,
        world_horizon,
        max_groups,
        max_substeps,
        max_action_dim,
    )
    tensors = {
        "history_fine_action_values": history_fine,
        "history_fine_action_mask": history_fine_mask,
        "history_fine_action_dt": history_fine_dt,
        "history_fine_sample_mask": history_fine_sample_mask,
        "history_coarse_action_values": history_coarse,
        "history_coarse_action_mask": history_coarse_mask,
        "future_factual_fine_action_values": torch.zeros(
            future_fine_shape, dtype=torch.float32, device=device
        ),
        "future_factual_fine_action_mask": torch.zeros(
            future_fine_shape, dtype=torch.bool, device=device
        ),
        "future_factual_fine_action_dt": torch.zeros(
            future_fine_shape[:-1], dtype=torch.float32, device=device
        ),
        "future_factual_fine_sample_mask": torch.zeros(
            future_fine_shape[:-1], dtype=torch.bool, device=device
        ),
        "future_factual_coarse_action_values": torch.zeros(
            (1, world_horizon, max_groups, max_action_dim),
            dtype=torch.float32,
            device=device,
        ),
        "future_factual_coarse_action_mask": torch.zeros(
            (1, world_horizon, max_groups, max_action_dim),
            dtype=torch.bool,
            device=device,
        ),
        "action_group_ids": group_ids,
        "action_group_mask": group_mask,
        "action_semantic_ids": action_semantics,
        "composition_operator_ids": composition,
        "current_state_values": current_values,
        "current_state_mask": current_mask,
        "state_semantic_ids": state_semantics,
        "embodiment_ids": torch.tensor(
            (PANDA_ROBOCASA_LIBERO_EMBODIMENT_ID,), dtype=torch.long, device=device
        ),
        "policy_query_dt": query_dt,
        "policy_query_mask": query_mask,
        "action_normalization_offset": norm_offset,
        "action_normalization_scale": norm_scale,
        "state_normalization_offset": state_norm_offset,
        "state_normalization_scale": state_norm_scale,
    }
    return PandaLiberoPolicyInputs(tensors)


def panda_action_chunk_from_model_output(
    output: Mapping[str, torch.Tensor],
    action_semantic_ids: torch.Tensor,
    action_group_ids: torch.Tensor,
    action_group_mask: torch.Tensor,
    embodiment_ids: torch.Tensor,
    *,
    batch_index: int = 0,
    panda_group_id: int = PANDA_ROBOCASA_LIBERO_ARM_GROUP_ID,
    panda_embodiment_id: int = PANDA_ROBOCASA_LIBERO_EMBODIMENT_ID,
) -> PandaLiberoActionChunk:
    """Project the unified group-major action ABI to executable LIBERO 7D."""

    for key in ("policy_action", "policy_action_mask"):
        if key not in output:
            raise PandaLiberoContractError(f"model output is missing {key!r}")
    action = output["policy_action"]
    mask = output["policy_action_mask"].bool()
    if action.ndim != 4 or mask.shape != action.shape:
        raise PandaLiberoContractError("policy action and mask must both be [B,G,C,A]")
    if not 0 <= batch_index < action.shape[0]:
        raise PandaLiberoContractError("batch index is outside model output")
    expected_group_shape = (action.shape[0], action.shape[1])
    if action_group_ids.shape != expected_group_shape:
        raise PandaLiberoContractError("action_group_ids must be [B,G]")
    if action_group_mask.shape != expected_group_shape:
        raise PandaLiberoContractError("action_group_mask must be [B,G]")
    if embodiment_ids.shape != (action.shape[0],):
        raise PandaLiberoContractError("embodiment_ids must be [B]")
    if int(embodiment_ids[batch_index].item()) != int(panda_embodiment_id):
        raise PandaLiberoContractError(
            "LIBERO must use the shared Panda RoboCasa/LIBERO embodiment"
        )
    if action_semantic_ids.shape != (
        action.shape[0],
        action.shape[1],
        action.shape[3],
    ):
        raise PandaLiberoContractError("action_semantic_ids must be [B,G,A]")
    group_matches = (
        (
            action_group_mask[batch_index].bool()
            & action_group_ids[batch_index].eq(int(panda_group_id))
        )
        .nonzero(as_tuple=False)
        .flatten()
    )
    if group_matches.numel() != 1:
        raise PandaLiberoContractError("exactly one active Panda arm group is required")
    group_index = int(group_matches.item())
    expected_semantics = torch.tensor(
        (
            ACTION_SEMANTIC_IDS["delta_position_m"],
            ACTION_SEMANTIC_IDS["delta_position_m"],
            ACTION_SEMANTIC_IDS["delta_position_m"],
            ACTION_SEMANTIC_IDS["delta_rotation_axis_angle_rad"],
            ACTION_SEMANTIC_IDS["delta_rotation_axis_angle_rad"],
            ACTION_SEMANTIC_IDS["delta_rotation_axis_angle_rad"],
            ACTION_SEMANTIC_IDS["absolute_gripper_close01"],
        ),
        device=action_semantic_ids.device,
        dtype=action_semantic_ids.dtype,
    )
    observed_semantics = action_semantic_ids[
        batch_index, group_index, : expected_semantics.numel()
    ]
    if not torch.equal(observed_semantics, expected_semantics):
        raise PandaLiberoContractError(
            "Panda action semantics are not delta pose + close01"
        )
    valid = mask[batch_index, group_index, :, :7].all(dim=-1)
    if not bool(valid.any()):
        raise PandaLiberoContractError("Panda action chunk contains no valid query")
    if bool(mask[batch_index, group_index, :, :7].any(dim=-1).logical_xor(valid).any()):
        raise PandaLiberoContractError("Panda action query is only partially masked")
    canonical_tensor = action[batch_index, group_index, valid, :7]
    if not bool(torch.isfinite(canonical_tensor).all()):
        raise PandaLiberoContractError("Panda action chunk contains NaN/Inf")
    canonical = canonical_tensor.detach().float().cpu().numpy().copy()
    if np.any((canonical[:, 6] < 0.0) | (canonical[:, 6] > 1.0)):
        raise PandaLiberoContractError("Panda gripper output is not close01")
    libero = canonical.copy()
    libero[:, 6] = np.where(canonical[:, 6] > 0.5, 1.0, -1.0)
    canonical.setflags(write=False)
    libero.setflags(write=False)
    return PandaLiberoActionChunk(canonical, libero)
