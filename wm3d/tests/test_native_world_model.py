from __future__ import annotations

from dataclasses import replace

import torch
import pytest
import yaml
from pathlib import Path
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)

from wm3d.data.grouped_robot import ACTION_SEMANTIC_IDS, STATE_SEMANTIC_IDS
from wm3d.models.native_world_model import (
    NativeWorldModel,
    NativeWorldModelConfig,
)
from wm3d.models.model_factory import build_world_model
from wm3d.models.model_factory import validate_model_data_compatibility
from wm3d.data.grouped_robot import ActionGroupSpec, EmbodimentSpec
from wm3d.training.gradient_ownership import (
    GradientOwnershipError,
    audit_gradient_ownership,
)


def _tiny_config() -> NativeWorldModelConfig:
    return NativeWorldModelConfig(
        T=2,
        P=4,
        K=2,
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
        bridge_heads=4,
        dynamics_layers=1,
        view_hidden=16,
        view_heads=4,
        view_ff_mult=2.0,
        max_action_groups=2,
        max_action_dim=7,
        max_state_dim=10,
        max_action_substeps=4,
        max_policy_queries=3,
        max_group_id=8,
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
        rgb_decode_indices=(0, 1),
        geom_hidden=16,
        activation_checkpointing=False,
    )


def _batch(cfg: NativeWorldModelConfig) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    batch = 2
    groups = cfg.max_action_groups
    substeps = cfg.max_action_substeps
    action_dim = cfg.max_action_dim
    state_dim = cfg.max_state_dim
    context_fine = torch.randn(batch, cfg.T, groups, substeps, action_dim)
    future_fine = torch.randn(batch, cfg.K, groups, substeps, action_dim)
    context_mask = torch.ones_like(context_fine, dtype=torch.bool)
    future_mask = torch.ones_like(future_fine, dtype=torch.bool)
    context_dt = torch.tensor([0.0, 0.031, 0.079, 0.11]).view(1, 1, 1, -1)
    context_dt = context_dt.expand(batch, cfg.T, groups, -1).clone()
    future_dt = context_dt[:, : cfg.K].clone()
    sample_mask = torch.ones(batch, cfg.T, groups, substeps, dtype=torch.bool)
    future_sample_mask = torch.ones(batch, cfg.K, groups, substeps, dtype=torch.bool)
    group_mask = torch.tensor([[True, True], [True, False]])
    context_mask[1, :, 1] = False
    future_mask[1, :, 1] = False
    sample_mask[1, :, 1] = False
    future_sample_mask[1, :, 1] = False
    action_semantic_ids = torch.tensor(
        [
            ACTION_SEMANTIC_IDS["delta_position_m"],
            ACTION_SEMANTIC_IDS["delta_position_m"],
            ACTION_SEMANTIC_IDS["delta_position_m"],
            ACTION_SEMANTIC_IDS["delta_rotation_axis_angle_rad"],
            ACTION_SEMANTIC_IDS["delta_rotation_axis_angle_rad"],
            ACTION_SEMANTIC_IDS["delta_rotation_axis_angle_rad"],
            ACTION_SEMANTIC_IDS["absolute_gripper_open01"],
        ]
    ).view(1, 1, action_dim).expand(batch, groups, -1).clone()
    state_semantic_ids = torch.tensor(
        [
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
        ]
    ).view(1, 1, state_dim).expand(batch, groups, -1).clone()
    query_dt = torch.tensor([0.01, 0.073, 0.231]).view(1, 1, -1)
    query_dt = query_dt.expand(batch, groups, -1).clone()
    query_mask = group_mask[..., None].expand(-1, -1, cfg.max_policy_queries).clone()
    current_mask = group_mask[..., None].expand(-1, -1, state_dim).clone()
    return {
        "world_tokens": torch.randn(
            batch, cfg.T, cfg.num_views, cfg.P, cfg.token_dim
        ),
        "view_mask": torch.ones(batch, cfg.T, cfg.num_views, dtype=torch.bool),
        "world_times_s": torch.tensor(
            [[0.0, 0.13, 0.37, 0.82], [1.0, 1.08, 1.4, 2.1]]
        ),
        "task_embedding": torch.randn(batch, cfg.task_dim),
        "history_fine_action_values": context_fine,
        "history_fine_action_mask": context_mask,
        "history_fine_action_dt": context_dt,
        "history_fine_sample_mask": sample_mask,
        "history_coarse_action_values": torch.zeros(
            batch, cfg.T, groups, action_dim
        ),
        "history_coarse_action_mask": torch.zeros(
            batch, cfg.T, groups, action_dim, dtype=torch.bool
        ),
        "future_factual_fine_action_values": future_fine,
        "future_factual_fine_action_mask": future_mask,
        "future_factual_fine_action_dt": future_dt,
        "future_factual_fine_sample_mask": future_sample_mask,
        "future_factual_coarse_action_values": torch.zeros(
            batch, cfg.K, groups, action_dim
        ),
        "future_factual_coarse_action_mask": torch.zeros(
            batch, cfg.K, groups, action_dim, dtype=torch.bool
        ),
        "action_group_ids": torch.tensor([[1, 2], [1, 0]]),
        "action_group_mask": group_mask,
        "action_semantic_ids": action_semantic_ids,
        "current_state_values": torch.randn(batch, groups, state_dim),
        "current_state_mask": current_mask,
        "state_semantic_ids": state_semantic_ids,
        "embodiment_ids": torch.tensor([2, 1]),
        "policy_query_dt": query_dt,
        "policy_query_mask": query_mask,
        "action_normalization_offset": torch.zeros(batch, groups, action_dim),
        "action_normalization_scale": torch.ones(batch, groups, action_dim),
    }


def test_one_core_handles_bimanual_nonuniform_state_and_action_times() -> None:
    cfg = _tiny_config()
    model = NativeWorldModel(cfg).eval()
    output = model(**_batch(cfg))

    assert output["pred_tokens"].shape == (2, cfg.K, cfg.P, cfg.token_dim)
    assert output["depth"].shape == (2, cfg.K, cfg.num_views, cfg.P)
    assert output["point"].shape == (2, cfg.K, cfg.num_views, cfg.P, 3)
    assert output["rgb"].shape == (
        2, cfg.K, cfg.num_views, 3, cfg.rgb_size, cfg.rgb_size
    )
    assert output["policy_action"].shape == (
        2,
        cfg.max_action_groups,
        cfg.max_policy_queries,
        cfg.max_action_dim,
    )
    assert output["policy_action_mask"][0].all()
    assert not output["policy_action_mask"][1, 1].any()
    assert torch.all((output["policy_action"][output["policy_gripper_mask"]] >= 0))
    assert torch.all((output["policy_action"][output["policy_gripper_mask"]] <= 1))


def test_rgb_decoder_uses_native_tokens_and_skips_unsupervised_views() -> None:
    cfg = _tiny_config()
    model = NativeWorldModel(cfg).train()
    batch = _batch(cfg)
    view_mask = torch.zeros(2, cfg.K, cfg.num_views, dtype=torch.bool)
    view_mask[:, :, 0] = True
    output = model(**batch, rgb_view_mask=view_mask)

    assert output["rgb"][:, :, 0].abs().sum() > 0
    assert output["rgb"][:, :, 1].count_nonzero() == 0
    output["rgb"][:, :, 0].square().mean().backward()
    assert model.token_output.weight.grad is not None
    assert torch.isfinite(model.token_output.weight.grad).all()
    assert model.token_output.weight.grad.abs().sum() > 0


def test_future_factual_action_changes_world_but_cannot_change_policy() -> None:
    cfg = _tiny_config()
    torch.manual_seed(11)
    model = NativeWorldModel(cfg).eval()
    batch = _batch(cfg)
    baseline = model(**batch)
    changed = dict(batch)
    changed["future_factual_fine_action_values"] = (
        batch["future_factual_fine_action_values"] + 100.0
    )
    counterfactual = model(**changed)

    torch.testing.assert_close(
        baseline["action_free_pred_tokens"],
        counterfactual["action_free_pred_tokens"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        baseline["policy_action_raw"],
        counterfactual["policy_action_raw"],
        rtol=0,
        atol=0,
    )
    assert not torch.allclose(baseline["pred_tokens"], counterfactual["pred_tokens"])


def test_swapping_commands_between_real_substep_times_changes_dynamics() -> None:
    """The encoder must retain command/timestamp pairing inside an interval."""

    cfg = _tiny_config()
    torch.manual_seed(12)
    model = NativeWorldModel(cfg).eval()
    batch = _batch(cfg)
    baseline = model(**batch)

    changed = dict(batch)
    values = batch["future_factual_fine_action_values"].clone()
    first = values[..., 0, :].clone()
    values[..., 0, :] = values[..., 1, :]
    values[..., 1, :] = first
    changed["future_factual_fine_action_values"] = values
    reordered = model(**changed)

    # Nothing about the policy input changed, while the physical trajectory
    # did.  A simple additive-then-mean encoder fails this regression.
    torch.testing.assert_close(
        baseline["policy_action_raw"], reordered["policy_action_raw"], rtol=0, atol=0
    )
    assert not torch.allclose(baseline["pred_tokens"], reordered["pred_tokens"])


def test_current_state_changes_policy_but_not_action_free_world_prior() -> None:
    cfg = _tiny_config()
    torch.manual_seed(13)
    model = NativeWorldModel(cfg).eval()
    batch = _batch(cfg)
    baseline = model(**batch)
    changed = dict(batch)
    changed["current_state_values"] = batch["current_state_values"] + 10.0
    counterfactual = model(**changed)

    torch.testing.assert_close(
        baseline["action_free_pred_tokens"],
        counterfactual["action_free_pred_tokens"],
        rtol=0,
        atol=0,
    )
    assert not torch.allclose(
        baseline["policy_action_raw"], counterfactual["policy_action_raw"]
    )


def test_policy_and_world_losses_reach_current_state_action_and_native_modules() -> None:
    cfg = _tiny_config()
    model = NativeWorldModel(cfg).train()
    output = model(**_batch(cfg))
    loss = (
        output["policy_action_raw"].square().mean()
        + output["pred_tokens"].square().mean()
        + output["rgb"].square().mean()
        + output["depth"].mean()
        + output["point"].square().mean()
    )
    loss.backward()

    named = dict(model.named_parameters())
    required_prefixes = (
        "current_state.value",
        "action_blocks.0",
        "state_blocks.0",
        "dynamics_blocks.0",
        "action_head.output",
        "token_output",
        "geometry_head",
    )
    for prefix in required_prefixes:
        gradients = [
            parameter.grad
            for name, parameter in named.items()
            if name.startswith(prefix) and parameter.requires_grad
        ]
        assert gradients, prefix
        assert any(
            gradient is not None
            and torch.isfinite(gradient).all()
            and gradient.abs().sum().item() > 0
            for gradient in gradients
        ), prefix

    audit = audit_gradient_ownership(model)
    assert audit["passed"] is True
    assert {
        "native_state_trunk",
        "factual_dynamics",
        "policy_action_trunk",
        "state_action_bridges",
        "current_state_proprio",
        "unified_action_head",
        "rgb_decoder",
        "geometry_decoder",
    } <= set(audit["owners"])
    assert {
        "native_state_inputs",
        "policy_action_inputs",
        "auxiliary_inputs",
    } <= set(audit["owners"])


def test_gradient_ownership_rejects_an_untrained_action_owner() -> None:
    cfg = _tiny_config()
    model = NativeWorldModel(cfg).train()
    output = model(**_batch(cfg))
    output["pred_tokens"].square().mean().backward()
    with pytest.raises(GradientOwnershipError, match="unified_action_head"):
        audit_gradient_ownership(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast is required")
def test_activation_checkpoint_recompute_preserves_bf16_autocast_metadata() -> None:
    cfg = replace(_tiny_config(), activation_checkpointing=True)
    model = NativeWorldModel(cfg).cuda().train()
    batch = {
        name: value.cuda() if isinstance(value, torch.Tensor) else value
        for name, value in _batch(cfg).items()
    }
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(**batch)
        loss = (
            output["pred_tokens"].square().mean()
            + output["policy_action_normalized"].square().mean()
            + output["rgb"].square().mean()
        )
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_activation_checkpoint_units_are_structural_and_state_dict_transparent() -> None:
    enabled = NativeWorldModel(replace(_tiny_config(), activation_checkpointing=True))
    disabled = NativeWorldModel(replace(_tiny_config(), activation_checkpointing=False))

    checkpoint_units = tuple(enabled.iter_activation_checkpoint_units())
    assert checkpoint_units
    assert all(isinstance(unit, CheckpointWrapper) for unit in checkpoint_units)
    assert {id(unit) for unit in checkpoint_units} < {
        id(unit) for unit in enabled.iter_fsdp_units()
    }
    assert not tuple(disabled.iter_activation_checkpoint_units())
    assert not any(
        isinstance(unit, CheckpointWrapper) for unit in disabled.iter_fsdp_units()
    )
    # CheckpointWrapper deliberately strips its private prefix from state
    # dictionaries.  DCP/exact-resume therefore sees one stable key contract.
    assert tuple(enabled.state_dict()) == tuple(disabled.state_dict())


def test_one_supervised_batch_trains_both_arms_through_the_same_policy_owner() -> None:
    cfg = _tiny_config()
    model = NativeWorldModel(cfg).train()
    batch = _batch(cfg)
    output = model(**batch)
    target = torch.randn_like(output["policy_action_raw"])
    # Only the genuinely bimanual sample participates in this regression.
    mask = output["policy_action_mask"].clone()
    mask[1] = False
    per_dim = torch.nn.functional.smooth_l1_loss(
        output["policy_action_raw"], target, reduction="none"
    )
    loss = (per_dim * mask).sum() / mask.sum()
    loss.backward()

    assert mask[0, 0].any() and mask[0, 1].any()
    query_grad = model.policy_query_seed.grad
    assert query_grad is not None and torch.isfinite(query_grad).all()
    assert query_grad.abs().sum().item() > 0
    for encoder in (model.history_action, model.current_state):
        group_grad = encoder.group.weight.grad
        assert group_grad is not None and torch.isfinite(group_grad).all()
        assert group_grad[1].abs().sum().item() > 0
        assert group_grad[2].abs().sum().item() > 0


def test_policy_query_times_are_not_derived_from_a_fixed_rate() -> None:
    cfg = _tiny_config()
    model = NativeWorldModel(cfg).eval()
    batch = _batch(cfg)
    batch["policy_query_dt"][0, 0] = torch.tensor([0.0, 0.173, 0.619])
    output = model(**batch)
    torch.testing.assert_close(output["policy_query_dt"], batch["policy_query_dt"])


def test_nonmonotonic_policy_query_times_fail_closed() -> None:
    cfg = _tiny_config()
    model = NativeWorldModel(cfg).eval()
    batch = _batch(cfg)
    batch["policy_query_dt"][0, 0] = torch.tensor([0.0, 0.2, 0.1])
    with pytest.raises(ValueError, match="strictly increasing"):
        model(**batch)


def test_policy_query_length_is_dynamic_below_the_profile_capacity() -> None:
    cfg = _tiny_config()
    model = NativeWorldModel(cfg).eval()
    batch = _batch(cfg)
    batch["policy_query_dt"] = batch["policy_query_dt"][..., :2]
    batch["policy_query_mask"] = batch["policy_query_mask"][..., :2]
    output = model(**batch)
    assert output["policy_action"].shape == (
        2,
        cfg.max_action_groups,
        2,
        cfg.max_action_dim,
    )


def test_policy_query_length_over_capacity_fails_closed() -> None:
    cfg = _tiny_config()
    model = NativeWorldModel(cfg).eval()
    batch = _batch(cfg)
    too_many = cfg.max_policy_queries + 1
    batch["policy_query_dt"] = torch.arange(too_many).float().view(1, 1, -1).expand(
        2, cfg.max_action_groups, -1
    )
    batch["policy_query_mask"] = torch.ones_like(
        batch["policy_query_dt"], dtype=torch.bool
    )
    with pytest.raises(ValueError, match="exceeds capacity"):
        model(**batch)


def test_query_capacity_does_not_change_parameter_count() -> None:
    small = _tiny_config()
    large = NativeWorldModelConfig(
        **{
            **small.__dict__,
            "max_policy_queries": small.max_policy_queries * 8,
        }
    )
    with torch.device("meta"):
        small_model = NativeWorldModel(small)
        large_model = NativeWorldModel(large)
    assert sum(p.numel() for p in small_model.parameters()) == sum(
        p.numel() for p in large_model.parameters()
    )


def test_1b_and_5b_profiles_use_one_model_class_and_sealed_parameter_counts() -> None:
    root = Path(__file__).resolve().parents[1]
    observed = {}
    for name in ("native_1b.yaml", "native_5b.yaml"):
        profile = yaml.safe_load((root / "configs/model" / name).read_text())
        with torch.device("meta"):
            model = build_world_model(profile)
        assert isinstance(model, NativeWorldModel)
        observed[name] = sum(parameter.numel() for parameter in model.parameters())
        assert observed[name] == profile["expected_parameter_count"]
    assert observed["native_5b.yaml"] > 4 * observed["native_1b.yaml"]


def test_model_data_gate_rejects_bimanual_capacity_truncation() -> None:
    profile = {
        "schema": "wm3d_v8_model_profile_v1",
        "name": "tiny",
        "architecture": "native_world_model",
        "sampling": {
            "mode": "observed_monotonic_subsequence",
            "history_action_leading_boundary": "observed_previous_state",
            "context_horizon_seconds": 0.2,
            "future_horizon_seconds": 0.2,
            "minimum_horizon_coverage": 0.9,
            "minimum_anchor_separation_seconds": 0.1,
            "policy_target_horizon_seconds": 0.2,
            "policy_training_times": "observed_action_timestamps",
            "interpolation": "forbidden",
        },
        "model": _tiny_config().__dict__ | {"max_action_groups": 1},
    }
    group = ActionGroupSpec(
        name="arm",
        group_id=1,
        action_semantics=("joint_position_rad",),
        state_semantics=("joint_position_rad",),
        action_frame="joint",
        state_frame="joint",
        composition_operators=("last",),
    )
    data_profile = type(
        "DataProfileFixture",
        (),
        {
            "cache_representation": {
                "token_grid": 2,
                "spatial_tokens": 4,
                "token_dim": 16,
                "num_views": 2,
                "rgb_size": 16,
            },
            "embodiments": {
                "dual": EmbodimentSpec(
                    name="dual",
                    embodiment_id=1,
                    groups=(group, ActionGroupSpec(**{**group.__dict__, "name": "right", "group_id": 2})),
                )
            },
        },
    )()
    with pytest.raises(ValueError, match="groups, model capacity"):
        validate_model_data_compatibility(profile, data_profile)


def test_model_profile_validation_reaches_native_architecture_fields() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = yaml.safe_load((root / "configs/model/native_1b.yaml").read_text())
    profile["model"]["state_heads"] = 7
    with pytest.raises(ValueError, match="divisible"):
        build_world_model(profile)
def _tiny_dual_path_config() -> NativeWorldModelConfig:
    return replace(
        _tiny_config(),
        appearance_enabled=True,
        appearance_P=16,
        appearance_context_frames=2,
        appearance_hidden=16,
        appearance_layers=1,
        appearance_heads=4,
        appearance_ff_mult=2.0,
    )


def _dual_path_batch(cfg: NativeWorldModelConfig) -> dict[str, torch.Tensor]:
    batch = _batch(cfg)
    batch_size = batch["world_tokens"].shape[0]
    torch.manual_seed(29)
    batch["appearance_context_tokens"] = torch.randn(
        batch_size,
        cfg.appearance_context_frames,
        cfg.num_views,
        cfg.appearance_P,
        cfg.token_dim,
    )
    batch["appearance_context_mask"] = torch.ones(
        batch_size,
        cfg.appearance_context_frames,
        cfg.num_views,
        cfg.appearance_P,
        dtype=torch.bool,
    )
    batch["target_appearance_tokens"] = torch.randn(
        batch_size,
        cfg.K,
        cfg.num_views,
        cfg.appearance_P,
        cfg.token_dim,
    )
    batch["target_appearance_mask"] = torch.ones(
        batch_size,
        cfg.K,
        cfg.num_views,
        cfg.appearance_P,
        dtype=torch.bool,
    )
    return batch


def test_dual_path_preserves_view_latents_and_conditions_rgb_on_geometry() -> None:
    cfg = _tiny_dual_path_config()
    torch.manual_seed(31)
    model = NativeWorldModel(cfg).train()
    batch = _dual_path_batch(cfg)

    predicted = model(**batch, appearance_teacher_ratio=0.0)
    teacher = model(**batch, appearance_teacher_ratio=1.0)
    assert predicted["appearance_pred_tokens"].shape == (
        2,
        cfg.K,
        cfg.num_views,
        cfg.appearance_P,
        cfg.token_dim,
    )
    assert predicted["appearance_pred_mask"].all()
    assert predicted["appearance_teacher_ratio"].item() == 0.0
    assert teacher["appearance_teacher_ratio"].item() == 1.0
    assert not torch.allclose(predicted["rgb"], teacher["rgb"])

    changed = dict(batch)
    changed_target = batch["target_appearance_tokens"].clone()
    changed_target[:, :, 1].add_(3.0)
    changed["target_appearance_tokens"] = changed_target
    changed_teacher = model(**changed, appearance_teacher_ratio=1.0)
    torch.testing.assert_close(
        teacher["rgb"][:, :, 0], changed_teacher["rgb"][:, :, 0], rtol=0, atol=0
    )
    assert not torch.allclose(teacher["rgb"][:, :, 1], changed_teacher["rgb"][:, :, 1])

    loss = (
        predicted["appearance_pred_tokens"].square().mean()
        + predicted["rgb"].square().mean()
    )
    loss.backward()
    appearance_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("appearance_dynamics")
    ]
    assert appearance_gradients
    assert any(
        gradient is not None
        and torch.isfinite(gradient).all()
        and gradient.abs().sum() > 0
        for gradient in appearance_gradients
    )
    geometry_conditioning = model.rgb_head.image_decoder.geometry_stem
    assert geometry_conditioning is not None
    assert geometry_conditioning.weight.grad is not None
    assert geometry_conditioning.weight.grad.abs().sum() > 0


def test_dual_path_inference_uses_predicted_appearance_without_future_targets() -> None:
    cfg = _tiny_dual_path_config()
    model = NativeWorldModel(cfg).eval()
    batch = _dual_path_batch(cfg)
    batch["appearance_context_mask"][:, :, 1] = False
    batch.pop("target_appearance_tokens")
    batch.pop("target_appearance_mask")

    output = model(**batch, appearance_teacher_ratio=0.0)
    assert output["appearance_pred_tokens"][:, :, 1].count_nonzero() == 0
    assert not output["appearance_pred_mask"][:, :, 1].any()
    with pytest.raises(ValueError, match="teacher forcing"):
        model(**batch, appearance_teacher_ratio=0.5)


def test_dual_path_1b_and_5b_profiles_are_materializable() -> None:
    root = Path(__file__).resolve().parents[1]
    counts = {}
    for name in ("native_1b_dual_path.yaml", "native_5b_dual_path.yaml"):
        profile = yaml.safe_load((root / "configs/model" / name).read_text())
        with torch.device("meta"):
            model = build_world_model(profile)
        counts[name] = sum(parameter.numel() for parameter in model.parameters())
        assert model.cfg.appearance_enabled is True
        assert model.cfg.appearance_P == 256
        assert counts[name] == profile["expected_parameter_count"]
    assert counts["native_5b_dual_path.yaml"] > 4 * counts["native_1b_dual_path.yaml"]
