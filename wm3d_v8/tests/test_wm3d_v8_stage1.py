from __future__ import annotations

import numpy as np
import pytest
import torch

from wm3d_v3.stage1_planner.candidates import deterministic_action_cost
from wm3d_v3.stage1_planner.losses import planner_loss
from wm3d_v3.stage1_planner.planner_head import NativePlannerConfig, NativePlannerHead
from wm3d_v3.stage1_planner.planner_head import planning_score
from wm3d_v3.stage1_planner.rollout import multichunk_native_rollout
from wm3d_v3.stage1_planner.train import (
    StepAddressedBatchSampler,
    _atomic_checkpoint,
    _dataset_config,
    _validate_checkpoint_directory,
)


def _planner() -> NativePlannerHead:
    return NativePlannerHead(
        NativePlannerConfig(
            token_dim=16,
            task_dim=12,
            hidden=32,
            spatial_layers=1,
            temporal_layers=1,
            heads=4,
            mlp_mult=2,
            dropout=0.0,
            max_horizon=4,
            patches=4,
        )
    )


def test_planner_is_action_blind_and_factorized() -> None:
    torch.manual_seed(7)
    planner = _planner().eval()
    tokens = torch.randn(2, 3, 4, 4, 16)
    task = torch.randn(2, 12)
    depth = torch.rand(2, 3, 4, 2, 2)
    point = torch.randn(2, 3, 4, 2, 2, 3)
    pose = torch.randn(2, 3, 4, 9)
    outputs = planner(tokens, task, depth=depth, point=point, pose=pose)
    assert outputs["success_logit"].shape == (2, 3)
    assert outputs["progress_logit"].shape == (2, 3, 4)
    assert sum(parameter.numel() for parameter in planner.parameters()) < 200_000
    assert not any("action" in name for name, _ in planner.named_parameters())


def test_planner_loss_has_finite_nonzero_gradients() -> None:
    torch.manual_seed(11)
    planner = _planner().train()
    outputs = planner(
        torch.randn(1, 3, 4, 4, 16),
        torch.randn(1, 12),
        depth=torch.rand(1, 3, 4, 2, 2),
        point=torch.randn(1, 3, 4, 2, 2, 3),
        pose=torch.randn(1, 3, 4, 9),
    )
    outputs["score"] = planning_score(outputs, torch.zeros(1, 3))
    loss = planner_loss(
        outputs,
        branch_rewards=torch.tensor([[[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]]]).float(),
        branch_dones=torch.tensor([[[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]]).bool(),
        branch_success=torch.tensor([[[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]]]).bool(),
        branch_valid=torch.tensor([[False, True, True]]),
        uncertainty_target=torch.zeros(1, 3),
    )
    loss["loss"].backward()
    grads = [p.grad for p in planner.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(value).all() for value in grads)
    assert sum(float(value.abs().sum()) for value in grads) > 0.0


def test_physical_action_cost_rejects_v8_36d_condition() -> None:
    assert deterministic_action_cost(torch.zeros(2, 3, 4, 7)).shape == (2, 3)
    with pytest.raises(ValueError, match="physical candidate"):
        deterministic_action_cost(torch.zeros(2, 3, 4, 36))


def test_step_addressed_sampler_exact_suffix() -> None:
    common = dict(
        dataset_size=12,
        batch_size=1,
        accumulation_steps=2,
        seed=27081,
        rank=3,
        world_size=8,
    )
    whole = list(
        StepAddressedBatchSampler(start_step=0, stop_step=20, **common)
    )
    resumed = list(
        StepAddressedBatchSampler(start_step=7, stop_step=20, **common)
    )
    assert resumed == whole[7 * 2 :]


def test_dataset_config_requires_payload_sha_closure() -> None:
    cfg = _dataset_config(
        {
            "branch_index": "/tmp/branch.jsonl",
            "branch_index_sha256": "a" * 64,
            "branch_payload_sha256_manifest": "/tmp/branch-sha.jsonl",
            "branch_payload_sha256_manifest_sha256": "b" * 64,
            "runtime_index": "/tmp/runtime.jsonl",
            "runtime_index_sha256": "c" * 64,
            "action_stats": "/tmp/stats.npz",
            "action_stats_sha256": "d" * 64,
            "action_adapter_audit": "/tmp/audit.json",
            "action_adapter_audit_sha256": "e" * 64,
        },
        "train",
    )
    assert cfg.branch_payload_sha256_manifest.name == "branch-sha.jsonl"
    assert cfg.branch_payload_sha256_manifest_sha256 == "b" * 64
    assert cfg.verify_runtime_payload_sha256 is True


def test_stage1_numbered_checkpoints_are_no_clobber(tmp_path) -> None:
    ckpt_dir = tmp_path / "ckpt"
    _validate_checkpoint_directory(
        ckpt_dir, require_empty=True, resume=None
    )
    path = ckpt_dir / "step_00000025.pt"
    _atomic_checkpoint(path, {"step": 25})
    with pytest.raises(FileExistsError, match="overwrite"):
        _atomic_checkpoint(path, {"step": 25, "changed": True})
    with pytest.raises(FileExistsError, match="empty checkpoint"):
        _validate_checkpoint_directory(
            ckpt_dir, require_empty=True, resume=None
        )
    _validate_checkpoint_directory(
        ckpt_dir, require_empty=True, resume=path
    )
    later = ckpt_dir / "step_00000050.pt"
    _atomic_checkpoint(later, {"step": 50})
    with pytest.raises(FileExistsError, match="beyond the source"):
        _validate_checkpoint_directory(
            ckpt_dir, require_empty=True, resume=path
        )


class _StateCfg:
    action_cond_dim = 36
    k = 2
    T = 4


class _DualCfg:
    state = _StateCfg()


class _World:
    cfg = type("Cfg", (), {"dual": _DualCfg()})()


def test_rollout_rejects_non_v8_action_dimension() -> None:
    world = _World()
    with pytest.raises(ValueError, match="native core contract"):
        multichunk_native_rollout(
            world,
            torch.zeros(1, 4, 4, 8),
            torch.zeros(1, 8),
            torch.zeros(1, 2, 4, 7),
            include_geometry=False,
        )
