from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from wm3d_v3.data.v7_action_contract import ActionAdapter
from wm3d_v3.models.action_stream import ActionConfig
from wm3d_v3.models.dual_stream import DualConfig
from wm3d_v3.models.joint_model import JointConfig, JointWorldModel
from wm3d_v3.models.state_stream import StateConfig
from wm3d_v3.stage1_planner.action_bridge import canonical_model_actions_to_simulator
from wm3d_v3.stage1_planner.candidates import DEFAULT_ROLES, build_candidate_set
from wm3d_v3.stage1_planner.dataset import SCHEMA, Stage1BranchDataset, Stage1BranchDatasetConfig
from wm3d_v3.stage1_planner.planner_head import NativePlannerConfig
from wm3d_v3.stage1_planner.system import NativePlanningSystem, Stage1SystemConfig


def test_action_bridge_is_exact_and_preserves_non_arm_template() -> None:
    adapter = ActionAdapter(
        source="robocasa",
        source_frame="base",
        translation_unit_scale=1.0,
        rotation_unit_scale=1.0,
        rotation_repr="axis_angle",
        gripper_index=6,
        nominal_hz=20.0,
    )
    actions = np.asarray(
        [
            [0.02, -0.01, 0.03, 0.10, 0.00, -0.05, -1.0],
            [-0.01, 0.02, 0.00, 0.00, 0.08, 0.00, 1.0],
        ],
        dtype=np.float32,
    )
    template = np.zeros((8, 12), dtype=np.float32)
    template[:, 7:] = np.arange(5, dtype=np.float32)[None]
    result = canonical_model_actions_to_simulator(
        actions,
        adapter,
        source_hz=20.0,
        target_hz=5.0,
        template=template,
        action_low=np.full(12, -10.0),
        action_high=np.full(12, 10.0),
    )
    np.testing.assert_allclose(result.reconstructed_canonical, actions, atol=2.0e-6)
    np.testing.assert_array_equal(result.simulator_actions[:, 7:], template[:, 7:])


def test_flow_candidates_cannot_take_gripper_ownership() -> None:
    direct = torch.randn(2, 8, 7)
    direct[..., 6] = torch.sigmoid(direct[..., 6])
    flow = torch.randn(2, 4, 8, 6)
    candidates = build_candidate_set(direct, flow)
    assert candidates.roles == DEFAULT_ROLES
    assert candidates.actions.shape == (2, 10, 8, 7)
    for index in range(1, 5):
        torch.testing.assert_close(candidates.actions[:, index, :, 6], direct[:, :, 6])


def _tiny_real_system() -> NativePlanningSystem:
    state = StateConfig(
        T=2,
        P=4,
        D=16,
        hidden=16,
        n_layers=1,
        n_heads=4,
        k=2,
        cond_dim=8,
        action_cond_dim=7,
    )
    action = ActionConfig(
        T=2,
        P=4,
        D=16,
        hidden=16,
        n_layers=1,
        n_heads=4,
        k=2,
        z_dim=8,
        cond_dim=8,
        action_cond_dim=7,
    )
    world = JointWorldModel(
        JointConfig(
            dual=DualConfig(
                state=state,
                action=action,
                xattn_layers_state=(0,),
                xattn_n_heads=4,
            ),
            enable_multiview_fuser=True,
            multiview_heads=4,
            enable_token_codec=False,
            action_proj_hidden=16,
            action_proj_layers=1,
            geom_hidden=32,
            geom_upsample_mode="resize_conv",
            enable_geom_extra=True,
            enable_pixel=False,
            enable_context_pixel=False,
        )
    )
    return NativePlanningSystem(
        world,
        Stage1SystemConfig(
            planner=NativePlannerConfig(
                token_dim=16,
                task_dim=8,
                hidden=16,
                layers=1,
                heads=4,
                mlp_mult=2,
                dropout=0.0,
                max_horizon=4,
                patches=4,
            ),
            candidate_microbatch=1,
            activation_checkpointing=False,
        ),
    )


def test_real_joint_world_model_rolls_multiple_native_chunks() -> None:
    torch.manual_seed(9)
    system = _tiny_real_system()
    context = torch.randn(1, 2, 4, 16)
    wrist = torch.randn_like(context)
    task = torch.randn(1, 8)
    actions = torch.randn(1, 2, 4, 7)
    actions[..., 6] = actions[..., 6].sigmoid()
    result = system(
        context,
        task,
        actions,
        wrist=wrist,
        view_mask=torch.ones(1, 2, 2, dtype=torch.bool),
        score_planner=True,
    )
    rollout = result["rollout"]
    assert rollout.tokens.shape == (1, 2, 4, 4, 16)
    assert rollout.depth.shape == (1, 2, 4, 224, 224)
    assert rollout.point.shape == (1, 2, 4, 224, 224, 3)
    assert rollout.pose.shape == (1, 2, 4, 9)
    assert result["planner"]["score"].shape == (1, 2)

    result["planner"]["score"].sum().backward()
    assert any(parameter.grad is not None for parameter in system.planner.parameters())
    assert all(parameter.grad is None for parameter in system.world.parameters())


def test_v2_dataset_requires_all_branch_native_evidence(tmp_path: Path) -> None:
    stats = tmp_path / "stats.npz"
    np.savez(stats, split=np.asarray("train"), mean=np.zeros(6), std=np.ones(6))
    payload = tmp_path / "root.npz"
    roles = ("factual_teacher", *DEFAULT_ROLES)
    np.savez_compressed(
        payload,
        schema=np.asarray(SCHEMA),
        root_id=np.asarray("root-1"),
        branch_roles=np.asarray(roles),
        anchor_codes=np.zeros((16, 64, 384), dtype=np.int8),
        anchor_scale=np.ones((16, 1, 1), dtype=np.float16),
        wrist_codes=np.zeros((16, 64, 384), dtype=np.int8),
        wrist_scale=np.ones((16, 1, 1), dtype=np.float16),
        root_context_sha256=np.asarray("b" * 64),
        branch_codes=np.zeros((11, 32, 64, 384), dtype=np.int8),
        branch_scales=np.ones((11, 32, 64, 1), dtype=np.float16),
        branch_actions_physical=np.zeros((11, 32, 7), dtype=np.float32),
        action_history_physical=np.zeros((4, 7), dtype=np.float32),
        branch_valid=np.ones(11, dtype=np.bool_),
        branch_rewards=np.zeros((11, 32), dtype=np.float32),
        branch_dones=np.zeros((11, 32), dtype=np.bool_),
        branch_success=np.zeros((11, 32), dtype=np.bool_),
        task_emb=np.ones(2048, dtype=np.float16),
        branch_depth_tgt=np.ones((11, 32, 8, 8), dtype=np.float16),
        branch_depth_conf_tgt=np.ones((11, 32, 8, 8), dtype=np.float16),
        branch_point_tgt=np.zeros((11, 32, 8, 8, 3), dtype=np.float16),
        branch_point_conf_tgt=np.ones((11, 32, 8, 8), dtype=np.float16),
        branch_pose_geom_tgt=np.zeros((11, 32, 9), dtype=np.float16),
        factual_index=np.asarray(0),
        direct_index=np.asarray(1),
    )
    index = tmp_path / "index.jsonl"
    index.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "root_id": "root-1",
                "path": str(payload),
                "split": "train",
                "split_group": "episode-1",
                "branch_roles": list(roles),
                "context_frames": 16,
                "future_frames": 32,
                "same_root_current_runtime_exact": True,
                "pseudo_outcomes": False,
                "future_observation_leakage": False,
                "context_source": "current_pinned_robocasa_runtime_causal_replay",
                "root_context_sha256": "b" * 64,
            }
        )
        + "\n"
    )
    dataset = Stage1BranchDataset(
        Stage1BranchDatasetConfig(index_path=index, split="train", action_stats=stats)
    )
    item = dataset[0]
    assert item["branch_s_tgt_codec"].shape == (11, 32, 64, 384)
    assert item["candidate_actions"].shape == (11, 32, 7)
    assert not bool(item["planning_mask"][0])
    assert bool(item["planning_mask"][1:].all())
