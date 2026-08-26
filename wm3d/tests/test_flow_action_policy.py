from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from tests.test_native_world_model import _batch, _tiny_config
from wm3d.data.grouped_robot import ACTION_SEMANTIC_IDS, COMPOSITION_OPERATOR_IDS
from wm3d.models.flow_action import (
    ContinuousFlowMatchScheduler,
    GroupedFlowActionHead,
)
from wm3d.models.model_factory import build_world_model, validate_model_profile
from wm3d.models.native_world_model import NativeWorldModel
from wm3d.training.distributed_runtime import _materialize_meta_shards
from wm3d.training.native_objective import (
    NativeObjectiveConfig,
    compute_native_objective,
)


def _flow_config():
    return replace(
        _tiny_config(),
        policy_action_mode="flow_matching",
        policy_flow_layers=2,
        policy_flow_inference_steps=4,
    )


def _head_inputs(cfg):
    batch = _batch(cfg)
    groups = cfg.max_action_groups
    queries = cfg.max_policy_queries
    query = torch.randn(2, queries, groups, cfg.action_hidden, requires_grad=True)
    target = torch.randn(2, groups, queries, cfg.max_action_dim)
    target[..., -1] = torch.tensor([0.0, 1.0]).view(2, 1, 1)
    mask = batch["policy_query_mask"][..., None] & batch["action_semantic_ids"][
        :, :, None
    ].ne(0)
    return batch, query, target, mask


def test_shifted_flow_scheduler_matches_wsa_endpoints_and_velocity_target() -> None:
    scheduler = ContinuousFlowMatchScheduler(1000, 5.0)
    unit = torch.tensor([0.0, 0.5, 1.0])
    sigma = scheduler.phi(unit, 5.0)
    torch.testing.assert_close(sigma[[0, 2]], torch.tensor([0.0, 1.0]))
    assert sigma[1].item() == pytest.approx(5.0 / 6.0)

    sample = torch.tensor([[1.0, -2.0]])
    noise = torch.tensor([[3.0, 4.0]])
    timestep = torch.tensor([250.0])
    torch.testing.assert_close(
        scheduler.add_noise(sample, noise, timestep),
        0.75 * sample + 0.25 * noise,
    )
    torch.testing.assert_close(scheduler.training_target(sample, noise), noise - sample)
    inference_t, deltas = scheduler.build_inference_schedule(
        10, shift=5.0, device=torch.device("cpu"), dtype=torch.float32
    )
    assert inference_t.shape == deltas.shape == (10,)
    assert inference_t[0].item() == pytest.approx(1000.0)
    assert deltas.sum().item() == pytest.approx(-1.0)


def test_grouped_flow_training_uses_velocity_target_and_preserves_binary_semantics() -> (
    None
):
    cfg = _flow_config()
    head = GroupedFlowActionHead(cfg).train()
    batch, query, target, target_mask = _head_inputs(cfg)
    noise = torch.full_like(target, 0.25)
    timestep = torch.tensor([250.0, 750.0])
    output = head(
        query,
        batch["action_semantic_ids"],
        batch["policy_query_mask"],
        batch["action_normalization_offset"],
        batch["action_normalization_scale"],
        target_action=target,
        target_action_mask=target_mask,
        flow_noise=noise,
        flow_timestep=timestep,
    )
    binary = output["policy_binary_mask"]
    continuous = output["policy_flow_continuous_mask"]
    expected_target = (noise - target) * continuous
    torch.testing.assert_close(output["policy_flow_target_velocity"], expected_target)
    assert not bool((continuous & binary).any())
    assert bool((output["policy_action_normalized"][binary] >= 0).all())
    assert bool((output["policy_action_normalized"][binary] <= 1).all())
    assert (
        output["policy_action_raw"][~output["policy_action_mask"]].count_nonzero() == 0
    )

    loss = (
        (output["policy_flow_velocity"] - output["policy_flow_target_velocity"])
        .square()[continuous]
        .mean()
    )
    loss.backward()
    assert query.grad is not None and query.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in head.blocks.parameters()
    )


def test_flow_inference_is_deterministic_with_fixed_noise_and_decodes_physical_units() -> (
    None
):
    cfg = _flow_config()
    head = GroupedFlowActionHead(cfg).eval()
    batch, query, target, target_mask = _head_inputs(cfg)
    query = query.detach()
    noise = torch.randn(
        2,
        cfg.max_action_groups,
        cfg.max_policy_queries,
        cfg.max_action_dim,
    )
    offset = batch["action_normalization_offset"].clone()
    scale = batch["action_normalization_scale"].clone()
    continuous_semantic = batch["action_semantic_ids"].eq(
        ACTION_SEMANTIC_IDS["delta_position_m"]
    )
    offset[continuous_semantic] = 0.2
    scale[continuous_semantic] = 0.5
    first = head(
        query,
        batch["action_semantic_ids"],
        batch["policy_query_mask"],
        offset,
        scale,
        flow_noise=noise,
    )
    second = head(
        query,
        batch["action_semantic_ids"],
        batch["policy_query_mask"],
        offset,
        scale,
        flow_noise=noise,
    )
    labeled_validation = head(
        query,
        batch["action_semantic_ids"],
        batch["policy_query_mask"],
        offset,
        scale,
        target_action=target,
        target_action_mask=target_mask,
        flow_noise=noise,
    )
    torch.testing.assert_close(first["policy_action"], second["policy_action"])
    torch.testing.assert_close(
        first["policy_action"], labeled_validation["policy_action"]
    )
    continuous = first["policy_action_mask"] & ~first["policy_binary_mask"]
    expected = (
        first["policy_action_normalized"] * scale[:, :, None] + offset[:, :, None]
    )
    torch.testing.assert_close(first["policy_action"][continuous], expected[continuous])


def test_full_v9_policy_stays_isolated_from_future_candidate_but_uses_context() -> None:
    cfg = _flow_config()
    torch.manual_seed(29)
    model = NativeWorldModel(cfg).eval()
    batch = _batch(cfg)
    noise = torch.randn(
        2,
        cfg.max_action_groups,
        cfg.max_policy_queries,
        cfg.max_action_dim,
    )
    baseline = model(**batch, policy_flow_noise=noise)
    changed_future = dict(batch)
    changed_future["future_factual_fine_action_values"] = (
        batch["future_factual_fine_action_values"] + 100.0
    )
    counterfactual = model(**changed_future, policy_flow_noise=noise)
    torch.testing.assert_close(
        baseline["policy_action_raw"],
        counterfactual["policy_action_raw"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        baseline["action_free_native_state"],
        counterfactual["action_free_native_state"],
        rtol=0,
        atol=0,
    )

    changed_context = dict(batch)
    changed_context["task_embedding"] = batch["task_embedding"] + 1.0
    conditioned = model(**changed_context, policy_flow_noise=noise)
    assert not torch.allclose(
        baseline["policy_action_normalized"],
        conditioned["policy_action_normalized"],
    )


def test_full_v9_training_forward_routes_targets_only_to_flow_owner() -> None:
    cfg = _flow_config()
    torch.manual_seed(31)
    model = NativeWorldModel(cfg).train()
    batch = _batch(cfg)
    target = torch.randn(
        2,
        cfg.max_action_groups,
        cfg.max_policy_queries,
        cfg.max_action_dim,
    )
    target[..., -1] = 1.0
    target_mask = batch["policy_query_mask"][..., None] & batch["action_semantic_ids"][
        :, :, None
    ].ne(0)
    noise = torch.randn_like(target)
    output = model(
        **batch,
        target_fine_action=target,
        target_fine_action_mask=target_mask,
        policy_flow_noise=noise,
        policy_flow_timestep=torch.tensor([200.0, 800.0]),
    )
    flow_mask = output["policy_flow_continuous_mask"]
    loss = (
        (output["policy_flow_velocity"] - output["policy_flow_target_velocity"])
        .square()[flow_mask]
        .mean()
    )
    loss.backward()
    assert model.action_head.action_input.weight.grad is not None
    assert model.action_head.action_input.weight.grad.abs().sum() > 0
    assert model.task_action.weight.grad is not None
    assert model.task_action.weight.grad.abs().sum() > 0


def test_native_objective_replaces_regression_instead_of_stacking_losses() -> None:
    velocity = torch.ones(1, 1, 2, 1, requires_grad=True)
    policy = torch.zeros_like(velocity)
    output = {
        "pred_tokens": torch.zeros(1, 1, 1, 2),
        "rgb": torch.empty(1, 0, 1, 3, 2, 2),
        "depth": torch.ones(1, 1, 1, 1),
        "point": torch.zeros(1, 1, 1, 1, 3),
        "camera_pose": torch.zeros(1, 1, 1, 9),
        "policy_action_raw": policy,
        "policy_action_normalized": policy,
        "policy_action": policy,
        "policy_action_mask": torch.ones_like(policy, dtype=torch.bool),
        "policy_gripper_mask": torch.zeros_like(policy, dtype=torch.bool),
        "policy_binary_mask": torch.zeros_like(policy, dtype=torch.bool),
        "policy_query_dt": torch.tensor([[[0.1, 0.2]]]),
        "policy_flow_velocity": velocity,
        "policy_flow_target_velocity": torch.zeros_like(velocity),
        "policy_flow_continuous_mask": torch.ones_like(velocity, dtype=torch.bool),
        "policy_flow_weight": torch.tensor([2.0]),
    }
    batch = {
        "target_tokens": torch.zeros(1, 1, 1, 2),
        "target_fine_action": torch.zeros_like(policy),
        "target_fine_action_mask": torch.ones_like(policy, dtype=torch.bool),
        "future_world_boundaries_dt": torch.tensor([[0.0, 0.3]]),
        "composition_operator_ids": torch.tensor(
            [[[COMPOSITION_OPERATOR_IDS["last"]]]]
        ),
        "target_coarse_action_normalized": torch.ones(1, 1, 1, 1),
        "target_coarse_action_mask": torch.ones(1, 1, 1, 1, dtype=torch.bool),
        "action_normalization_offset": torch.zeros(1, 1, 1),
        "action_normalization_scale": torch.ones(1, 1, 1),
    }
    losses = compute_native_objective(
        output=output,
        batch=batch,
        config=NativeObjectiveConfig(
            token_mse=0.0,
            token_cosine=0.0,
            depth_log=0.0,
            point=0.0,
            camera_pose=0.0,
            action_fine=1.0,
            action_coarse=1.0,
        ),
    )
    assert losses["action_fine_continuous"].item() == pytest.approx(2.0)
    assert losses["action_coarse"].item() == 0.0
    assert losses["action_coarse_metric"].item() > 0.0
    losses["total"].backward()
    torch.testing.assert_close(velocity.grad, torch.full_like(velocity, 2.0))


def test_v9_profiles_are_separate_and_meta_materializable() -> None:
    root = Path(__file__).resolve().parents[1]
    counts = {}
    for name in ("native_1b_v9_flow.yaml", "native_5b_v9_flow.yaml"):
        profile = yaml.safe_load((root / "configs/model" / name).read_text())
        validate_model_profile(profile)
        with torch.device("meta"):
            model = build_world_model(profile)
        assert isinstance(model.action_head, GroupedFlowActionHead)
        counts[name] = sum(parameter.numel() for parameter in model.parameters())
    assert counts["native_1b_v9_flow.yaml"] > 1_489_275_928
    assert counts["native_5b_v9_flow.yaml"] > 5_556_187_512


def test_flow_modules_reset_cleanly_from_meta_for_fsdp_initialization() -> None:
    cfg = _flow_config()
    with torch.device("meta"):
        model = NativeWorldModel(cfg)
    _materialize_meta_shards(model, torch.device("cpu"))
    assert not any(parameter.is_meta for parameter in model.parameters())
    assert not any(buffer.is_meta for buffer in model.buffers())
    model.eval()
    batch = _batch(cfg)
    noise = torch.zeros(
        2,
        cfg.max_action_groups,
        cfg.max_policy_queries,
        cfg.max_action_dim,
    )
    output = model(**batch, policy_flow_noise=noise)
    assert bool(torch.isfinite(output["policy_action"]).all())
