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
    ActionBlock,
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
    action_semantic_ids = (
        torch.tensor(
            [
                ACTION_SEMANTIC_IDS["delta_position_m"],
                ACTION_SEMANTIC_IDS["delta_position_m"],
                ACTION_SEMANTIC_IDS["delta_position_m"],
                ACTION_SEMANTIC_IDS["delta_rotation_axis_angle_rad"],
                ACTION_SEMANTIC_IDS["delta_rotation_axis_angle_rad"],
                ACTION_SEMANTIC_IDS["delta_rotation_axis_angle_rad"],
                ACTION_SEMANTIC_IDS["absolute_gripper_open01"],
            ]
        )
        .view(1, 1, action_dim)
        .expand(batch, groups, -1)
        .clone()
    )
    state_semantic_ids = (
        torch.tensor(
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
        )
        .view(1, 1, state_dim)
        .expand(batch, groups, -1)
        .clone()
    )
    query_dt = torch.tensor([0.01, 0.073, 0.231]).view(1, 1, -1)
    query_dt = query_dt.expand(batch, groups, -1).clone()
    query_mask = group_mask[..., None].expand(-1, -1, cfg.max_policy_queries).clone()
    current_mask = group_mask[..., None].expand(-1, -1, state_dim).clone()
    return {
        "world_tokens": torch.randn(batch, cfg.T, cfg.num_views, cfg.P, cfg.token_dim),
        "view_mask": torch.ones(batch, cfg.T, cfg.num_views, dtype=torch.bool),
        "world_times_s": torch.tensor([[0.0, 0.13, 0.37, 0.82], [1.0, 1.08, 1.4, 2.1]]),
        "task_embedding": torch.randn(batch, cfg.task_dim),
        "history_fine_action_values": context_fine,
        "history_fine_action_mask": context_mask,
        "history_fine_action_dt": context_dt,
        "history_fine_sample_mask": sample_mask,
        "history_coarse_action_values": torch.zeros(batch, cfg.T, groups, action_dim),
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
        2,
        cfg.K,
        cfg.num_views,
        3,
        cfg.rgb_size,
        cfg.rgb_size,
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


def test_rgb_decoder_uses_rank_invariant_chunk_calls_for_sparse_views() -> None:
    cfg = _tiny_config()
    model = NativeWorldModel(cfg).eval()
    batch = _batch(cfg)
    calls: list[int] = []
    handle = model.rgb_head.image_decoder.register_forward_hook(
        lambda _module, inputs, _output: calls.append(int(inputs[0].shape[0]))
    )
    try:
        sparse = torch.zeros(2, cfg.K, cfg.num_views, dtype=torch.bool)
        sparse[:, :, 0] = True
        sparse_output = model(**batch, rgb_view_mask=sparse)
        sparse_calls = tuple(calls)
        calls.clear()

        dense = torch.ones_like(sparse)
        model(**batch, rgb_view_mask=dense)
        dense_calls = tuple(calls)
    finally:
        handle.remove()

    expected_slots = 2 * cfg.K * cfg.num_views
    assert sparse_calls == dense_calls == (1,) * expected_slots
    assert sparse_output["rgb"][:, :, 1].count_nonzero() == 0


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


def test_factual_action_residual_reaches_world_before_dynamics_only() -> None:
    cfg = replace(_tiny_config(), factual_action_residual_scale=0.25)
    torch.manual_seed(111)
    model = NativeWorldModel(cfg).eval()
    batch = _batch(cfg)
    changed = dict(batch)
    changed["future_factual_fine_action_values"] = (
        batch["future_factual_fine_action_values"] + 3.0
    )
    dynamics_inputs: list[torch.Tensor] = []
    handle = model.dynamics_blocks[0].register_forward_pre_hook(
        lambda _module, inputs: dynamics_inputs.append(inputs[0].detach().clone())
    )
    try:
        baseline = model(**batch)
        counterfactual = model(**changed)
    finally:
        handle.remove()

    assert len(dynamics_inputs) == 2
    assert not torch.allclose(dynamics_inputs[0], dynamics_inputs[1])
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


@pytest.mark.parametrize("value", (-0.1, float("nan"), float("inf")))
def test_factual_action_residual_scale_fails_closed(value: float) -> None:
    with pytest.raises(
        ValueError,
        match="factual_action_residual_scale must be finite and non-negative",
    ):
        NativeWorldModel(replace(_tiny_config(), factual_action_residual_scale=value))


@pytest.mark.parametrize("value", (-0.1, float("nan"), float("inf")))
def test_appearance_action_residual_scale_fails_closed(value: float) -> None:
    with pytest.raises(
        ValueError,
        match="appearance_action_residual_scale must be finite and non-negative",
    ):
        NativeWorldModel(
            replace(_tiny_config(), appearance_action_residual_scale=value)
        )


@pytest.mark.parametrize("value", (-0.1, float("nan"), float("inf")))
def test_rgb_context_action_scale_fails_closed(value: float) -> None:
    with pytest.raises(
        ValueError,
        match="rgb_context_action_scale must be finite and non-negative",
    ):
        NativeWorldModel(replace(_tiny_config(), rgb_context_action_scale=value))


def test_rgb_context_action_requires_context_renderer() -> None:
    with pytest.raises(
        ValueError,
        match="rgb_context_action_scale requires rgb_context_enabled",
    ):
        NativeWorldModel(replace(_tiny_config(), rgb_context_action_scale=1.0))


@pytest.mark.parametrize("value", (-0.1, float("nan"), float("inf")))
def test_rgb_context_appearance_delta_scale_fails_closed(value: float) -> None:
    with pytest.raises(
        ValueError,
        match=("rgb_context_appearance_delta_scale must be finite and non-negative"),
    ):
        NativeWorldModel(
            replace(_tiny_config(), rgb_context_appearance_delta_scale=value)
        )


def test_rgb_context_appearance_delta_requires_context_and_appearance() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "rgb_context_appearance_delta_scale requires context RGB and appearance"
        ),
    ):
        NativeWorldModel(
            replace(_tiny_config(), rgb_context_appearance_delta_scale=1.0)
        )


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


def test_policy_task_modulation_is_exact_identity_at_initialization() -> None:
    old_cfg = _tiny_config()
    new_cfg = replace(old_cfg, policy_task_modulation=True)
    torch.manual_seed(1301)
    old = NativeWorldModel(old_cfg).eval()
    torch.manual_seed(1302)
    new = NativeWorldModel(new_cfg).eval()
    incompatible = new.load_state_dict(old.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all("task_modulation" in name for name in incompatible.missing_keys)

    batch = _batch(old_cfg)
    baseline = old(**batch)
    conditioned = new(**batch)
    for name in (
        "policy_action_raw",
        "policy_latent",
        "action_free_pred_tokens",
        "pred_tokens",
        "rgb",
    ):
        torch.testing.assert_close(baseline[name], conditioned[name], rtol=0, atol=0)


def test_policy_task_modulation_changes_queries_without_relabeling_history() -> None:
    cfg = replace(_tiny_config(), policy_task_modulation=True)
    block = ActionBlock(cfg).eval()
    batch, steps, groups, dim = 2, 4, cfg.max_action_groups, cfg.action_hidden
    value = torch.randn(batch, steps, groups, dim)
    times = (
        torch.arange(steps, dtype=torch.float32)
        .view(1, -1, 1)
        .expand(batch, -1, groups)
    )
    valid = torch.ones(batch, steps, groups, dtype=torch.bool)
    query_mask = torch.zeros_like(valid)
    query_mask[:, -2:] = True
    first_task = torch.randn(batch, dim)
    second_task = torch.randn(batch, dim)

    baseline = block(value, times, valid, first_task, query_mask)
    torch.testing.assert_close(
        baseline,
        block(value, times, valid, second_task, query_mask),
        rtol=0,
        atol=0,
    )
    assert block.attn_task_modulation is not None
    assert block.ff_task_modulation is not None
    with torch.no_grad():
        block.attn_task_modulation.shift.fill_(0.1)
        block.ff_task_modulation.shift.fill_(0.1)
    changed = block(value, times, valid, second_task, query_mask)
    torch.testing.assert_close(baseline[:, :-2], changed[:, :-2], rtol=0, atol=0)
    assert not torch.allclose(baseline[:, -2:], changed[:, -2:])


def test_learned_policy_task_modulation_stays_out_of_world_and_rgb() -> None:
    cfg = replace(_tiny_config(), policy_task_modulation=True)
    model = NativeWorldModel(cfg).eval()
    batch = _batch(cfg)
    baseline = model(**batch)

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "task_modulation" in name and name.endswith(("scale", "shift")):
                parameter.fill_(0.1)

    changed = model(**batch)
    for name in ("action_free_pred_tokens", "pred_tokens", "rgb"):
        torch.testing.assert_close(baseline[name], changed[name], rtol=0, atol=0)
    assert not torch.allclose(
        baseline["policy_action_raw"], changed["policy_action_raw"]
    )


def test_policy_task_modulation_gates_receive_action_supervision() -> None:
    cfg = replace(_tiny_config(), policy_task_modulation=True)
    model = NativeWorldModel(cfg).train()
    output = model(**_batch(cfg))
    output["policy_action_raw"].square().mean().backward()
    gates = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "task_modulation" in name
        and (name.endswith("scale") or name.endswith("shift"))
    ]
    assert gates
    assert all(
        gradient is not None
        and torch.isfinite(gradient).all()
        and gradient.abs().sum() > 0
        for gradient in gates
    )


def test_policy_and_world_losses_reach_current_state_action_and_native_modules() -> (
    None
):
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


def test_activation_checkpoint_units_are_structural_and_state_dict_transparent() -> (
    None
):
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
    batch["policy_query_dt"] = (
        torch.arange(too_many)
        .float()
        .view(1, 1, -1)
        .expand(2, cfg.max_action_groups, -1)
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
                    groups=(
                        group,
                        ActionGroupSpec(
                            **{**group.__dict__, "name": "right", "group_id": 2}
                        ),
                    ),
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


def test_context_rgb_renderer_preserves_static_reference_and_masks_missing_views() -> (
    None
):
    cfg = replace(_tiny_dual_path_config(), rgb_context_enabled=True)
    torch.manual_seed(37)
    model = NativeWorldModel(cfg).eval()
    batch = _dual_path_batch(cfg)
    batch["context_rgb"] = torch.rand(2, cfg.num_views, 3, cfg.rgb_size, cfg.rgb_size)
    batch["context_rgb_mask"] = torch.tensor(
        [[True, False], [True, True]], dtype=torch.bool
    )
    decoder = model.rgb_head.image_decoder
    with torch.no_grad():
        decoder.head.weight.zero_()
        decoder.head.bias.zero_()
        decoder.head.bias[6] = -20.0
        decoder.motion_head.bias.fill_(-20.0)

    output = model(**batch, appearance_teacher_ratio=0.0)

    expected = batch["context_rgb"][:, None].expand(-1, cfg.K, -1, -1, -1, -1)
    torch.testing.assert_close(
        output["rgb"][0, :, 0],
        expected[0, :, 0],
        rtol=0,
        atol=1.0e-6,
    )
    assert output["rgb"][0, :, 1].count_nonzero() == 0
    assert output["rgb_motion_logit"].shape == (
        2,
        cfg.K,
        cfg.num_views,
        1,
        cfg.rgb_size,
        cfg.rgb_size,
    )
    assert output["rgb_blend"][0, :, 1].count_nonzero() == 0


def test_context_renderer_rgb_loss_reaches_per_view_p256_appearance_lane() -> None:
    cfg = replace(_tiny_dual_path_config(), rgb_context_enabled=True)
    torch.manual_seed(101)
    model = NativeWorldModel(cfg).train()
    batch = _dual_path_batch(cfg)
    batch["context_rgb"] = torch.rand(2, cfg.num_views, 3, cfg.rgb_size, cfg.rgb_size)
    batch["context_rgb_mask"] = torch.ones(2, cfg.num_views, dtype=torch.bool)

    teacher = model(**batch, appearance_teacher_ratio=1.0)
    changed = dict(batch)
    changed_target = batch["target_appearance_tokens"].clone()
    changed_target[:, :, 1] = changed_target[:, :, 1].roll(1, dims=0)
    changed["target_appearance_tokens"] = changed_target
    changed_teacher = model(**changed, appearance_teacher_ratio=1.0)
    torch.testing.assert_close(
        teacher["rgb"][:, :, 0], changed_teacher["rgb"][:, :, 0], rtol=0, atol=0
    )
    assert not torch.allclose(teacher["rgb"][:, :, 1], changed_teacher["rgb"][:, :, 1])

    model.zero_grad(set_to_none=True)
    predicted = model(**batch, appearance_teacher_ratio=0.0)
    predicted["appearance_pred_tokens"].retain_grad()
    predicted["rgb"].square().mean().backward()
    appearance_gradient = predicted["appearance_pred_tokens"].grad
    assert appearance_gradient is not None
    assert torch.isfinite(appearance_gradient).all()
    assert appearance_gradient.abs().sum() > 0
    token_stem = model.rgb_head.image_decoder.token_stem[0]
    assert token_stem.weight.grad is not None
    assert torch.isfinite(token_stem.weight.grad).all()
    assert token_stem.weight.grad.abs().sum() > 0


def test_context_renderer_p256_delta_has_persistent_multiscale_ownership() -> None:
    cfg = replace(
        _tiny_dual_path_config(),
        rgb_context_enabled=True,
        rgb_context_appearance_delta_scale=1.0,
    )
    torch.manual_seed(103)
    decoder = NativeWorldModel(cfg).rgb_head.train()
    batch = 2
    future_tokens = torch.randn(batch, cfg.K, cfg.P, cfg.token_dim)
    appearance_context = torch.randn(
        batch, cfg.num_views, cfg.appearance_P, cfg.token_dim
    )
    appearance = appearance_context[:, None].expand(-1, cfg.K, -1, -1, -1)
    appearance = appearance.clone()
    appearance[:, 1].add_(0.25)
    geometry = torch.randn(batch, cfg.K, cfg.P, cfg.state_hidden)
    context = torch.rand(batch, cfg.num_views, 3, cfg.rgb_size, cfg.rgb_size)
    context_mask = torch.ones(batch, cfg.num_views, dtype=torch.bool)
    task = torch.randn(batch, cfg.task_dim)
    observed_delta: list[torch.Tensor] = []

    delta_stem = decoder.image_decoder.appearance_delta_stem
    assert delta_stem is not None

    def record_delta(_module, inputs) -> None:
        observed_delta.append(inputs[0].detach())

    handle = delta_stem.register_forward_pre_hook(record_delta)
    try:
        rgb = decoder(
            future_tokens,
            None,
            appearance_tokens=appearance,
            appearance_context_tokens=appearance_context,
            geometry_state=geometry,
            task_embedding=task,
            context_rgb=context,
            context_rgb_mask=context_mask,
        )[0]
    finally:
        handle.remove()

    assert observed_delta
    assert any(value.count_nonzero() == 0 for value in observed_delta)
    assert any(value.count_nonzero() > 0 for value in observed_delta)
    rgb.square().mean().backward()
    delta_gradients = [
        parameter.grad
        for name, parameter in decoder.named_parameters()
        if "appearance_delta" in name
    ]
    assert delta_gradients
    assert all(gradient is not None for gradient in delta_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in delta_gradients)
    assert all(gradient.abs().sum() > 0 for gradient in delta_gradients)


def test_context_renderer_directly_uses_factual_action_summary() -> None:
    cfg = replace(
        _tiny_dual_path_config(),
        rgb_context_enabled=True,
        rgb_context_action_scale=1.0,
    )
    torch.manual_seed(127)
    decoder = NativeWorldModel(cfg).rgb_head.train()
    batch = 2
    future_tokens = torch.randn(batch, cfg.K, cfg.P, cfg.token_dim)
    appearance = torch.randn(
        batch, cfg.K, cfg.num_views, cfg.appearance_P, cfg.token_dim
    )
    geometry = torch.randn(batch, cfg.K, cfg.P, cfg.state_hidden)
    context = torch.rand(batch, cfg.num_views, 3, cfg.rgb_size, cfg.rgb_size)
    context_mask = torch.ones(batch, cfg.num_views, dtype=torch.bool)
    task = torch.randn(batch, cfg.task_dim)
    zero = torch.zeros(batch, cfg.K, cfg.state_hidden)
    factual = torch.randn_like(zero)

    zero_rgb = decoder(
        future_tokens,
        None,
        appearance_tokens=appearance,
        geometry_state=geometry,
        factual_action_summary=zero,
        task_embedding=task,
        context_rgb=context,
        context_rgb_mask=context_mask,
    )[0]
    factual_rgb = decoder(
        future_tokens,
        None,
        appearance_tokens=appearance,
        geometry_state=geometry,
        factual_action_summary=factual,
        task_embedding=task,
        context_rgb=context,
        context_rgb_mask=context_mask,
    )[0]

    assert not torch.allclose(factual_rgb, zero_rgb)
    factual_rgb.square().mean().backward()
    action_gradients = [
        parameter.grad
        for name, parameter in decoder.named_parameters()
        if "action_proj" in name
    ]
    assert action_gradients
    assert all(gradient is not None for gradient in action_gradients)
    assert any(gradient.abs().sum() > 0 for gradient in action_gradients)


def test_context_renderer_directly_uses_task_embedding() -> None:
    cfg = replace(_tiny_dual_path_config(), rgb_context_enabled=True)
    torch.manual_seed(131)
    decoder = NativeWorldModel(cfg).rgb_head.train()
    batch = 2
    future_tokens = torch.randn(batch, cfg.K, cfg.P, cfg.token_dim)
    appearance = torch.randn(
        batch, cfg.K, cfg.num_views, cfg.appearance_P, cfg.token_dim
    )
    geometry = torch.randn(batch, cfg.K, cfg.P, cfg.state_hidden)
    context = torch.rand(batch, cfg.num_views, 3, cfg.rgb_size, cfg.rgb_size)
    context_mask = torch.ones(batch, cfg.num_views, dtype=torch.bool)
    task = torch.randn(batch, cfg.task_dim)

    baseline = decoder(
        future_tokens,
        None,
        appearance_tokens=appearance,
        geometry_state=geometry,
        task_embedding=task,
        context_rgb=context,
        context_rgb_mask=context_mask,
    )[0]
    changed = decoder(
        future_tokens,
        None,
        appearance_tokens=appearance,
        geometry_state=geometry,
        task_embedding=task.roll(1, dims=0),
        context_rgb=context,
        context_rgb_mask=context_mask,
    )[0]

    assert not torch.allclose(baseline, changed)
    baseline.square().mean().backward()
    task_gradients = [
        parameter.grad
        for name, parameter in decoder.named_parameters()
        if "task_proj" in name
    ]
    assert task_gradients
    assert all(gradient is not None for gradient in task_gradients)
    assert any(gradient.abs().sum() > 0 for gradient in task_gradients)


def test_renderer_action_route_stays_out_of_action_free_policy_trunk() -> None:
    cfg = replace(
        _tiny_dual_path_config(),
        rgb_context_enabled=True,
        rgb_context_action_scale=1.0,
    )
    torch.manual_seed(129)
    model = NativeWorldModel(cfg).eval()
    batch = _dual_path_batch(cfg)
    batch["context_rgb"] = torch.rand(2, cfg.num_views, 3, cfg.rgb_size, cfg.rgb_size)
    batch["context_rgb_mask"] = torch.ones(2, cfg.num_views, dtype=torch.bool)
    zero_action = dict(batch)
    zero_action["future_factual_fine_action_values"] = torch.zeros_like(
        batch["future_factual_fine_action_values"]
    )
    zero_action["future_factual_coarse_action_values"] = torch.zeros_like(
        batch["future_factual_coarse_action_values"]
    )

    factual = model(**batch, appearance_teacher_ratio=1.0)
    zero = model(**zero_action, appearance_teacher_ratio=1.0)
    torch.testing.assert_close(
        factual["action_free_pred_tokens"],
        zero["action_free_pred_tokens"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        factual["policy_action_raw"], zero["policy_action_raw"], rtol=0, atol=0
    )
    assert not torch.allclose(factual["rgb"], zero["rgb"])


def test_renderer_direct_action_is_centered_on_same_mask_zero_command() -> None:
    cfg = replace(
        _tiny_dual_path_config(),
        rgb_context_enabled=True,
        rgb_context_action_scale=1.0,
    )
    torch.manual_seed(130)
    model = NativeWorldModel(cfg).eval()
    batch = _dual_path_batch(cfg)
    batch["context_rgb"] = torch.rand(2, cfg.num_views, 3, cfg.rgb_size, cfg.rgb_size)
    batch["context_rgb_mask"] = torch.ones(2, cfg.num_views, dtype=torch.bool)
    zero_action = dict(batch)
    zero_action["future_factual_fine_action_values"] = torch.zeros_like(
        batch["future_factual_fine_action_values"]
    )
    zero_action["future_factual_coarse_action_values"] = torch.zeros_like(
        batch["future_factual_coarse_action_values"]
    )
    observed: list[torch.Tensor] = []

    def record(_module, inputs) -> None:
        observed.append(inputs[0].detach())

    handle = model.rgb_head.image_decoder.action_proj.register_forward_pre_hook(record)
    try:
        model(**zero_action, appearance_teacher_ratio=1.0)
        zero_count = len(observed)
        assert zero_count > 0
        assert torch.cat(observed).count_nonzero() == 0
        model(**batch, appearance_teacher_ratio=1.0)
    finally:
        handle.remove()

    assert len(observed) > zero_count
    assert torch.cat(observed[zero_count:]).count_nonzero() > 0


def test_appearance_action_residual_changes_rgb_without_policy_leakage() -> None:
    cfg = replace(
        _tiny_dual_path_config(),
        appearance_action_residual_scale=0.25,
    )
    torch.manual_seed(131)
    model = NativeWorldModel(cfg).eval()
    batch = _dual_path_batch(cfg)
    zero_action = dict(batch)
    zero_action["future_factual_fine_action_values"] = torch.zeros_like(
        batch["future_factual_fine_action_values"]
    )
    zero_action["future_factual_coarse_action_values"] = torch.zeros_like(
        batch["future_factual_coarse_action_values"]
    )

    factual = model(**batch, appearance_teacher_ratio=0.0)
    zero = model(**zero_action, appearance_teacher_ratio=0.0)
    torch.testing.assert_close(
        factual["action_free_pred_tokens"],
        zero["action_free_pred_tokens"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        factual["policy_action_raw"],
        zero["policy_action_raw"],
        rtol=0,
        atol=0,
    )
    assert not torch.allclose(
        factual["appearance_pred_tokens"], zero["appearance_pred_tokens"]
    )
    assert not torch.allclose(factual["rgb"], zero["rgb"])


def test_appearance_action_conditioning_is_centered_and_spatially_resolved() -> None:
    cfg = replace(
        _tiny_dual_path_config(),
        appearance_action_residual_scale=0.3,
    )
    torch.manual_seed(132)
    model = NativeWorldModel(cfg).eval()
    batch = _dual_path_batch(cfg)
    zero_action = dict(batch)
    zero_action["future_factual_fine_action_values"] = torch.zeros_like(
        batch["future_factual_fine_action_values"]
    )
    zero_action["future_factual_coarse_action_values"] = torch.zeros_like(
        batch["future_factual_coarse_action_values"]
    )
    observed: list[torch.Tensor] = []
    updates: list[torch.Tensor] = []

    def record_input(_module, inputs) -> None:
        centered = inputs[1]
        centered.retain_grad()
        observed.append(centered)

    def record_update(_module, inputs, output) -> None:
        updates.append((output - inputs[0]).detach())

    assert model.appearance_dynamics is not None
    conditioner = model.appearance_dynamics.action_conditioner
    assert conditioner is not None
    input_handle = conditioner.register_forward_pre_hook(record_input)
    output_handle = conditioner.register_forward_hook(record_update)
    try:
        model(**zero_action, appearance_teacher_ratio=0.0)
        zero_call_count = len(observed)
        assert zero_call_count == 1
        assert observed[0].count_nonzero() == 0
        assert updates[0].count_nonzero() == 0
        factual = model(**batch, appearance_teacher_ratio=0.0)
    finally:
        input_handle.remove()
        output_handle.remove()

    assert len(observed) == zero_call_count + 1
    centered = observed[zero_call_count]
    assert centered.count_nonzero() > 0
    spatial_update = updates[zero_call_count]
    assert spatial_update.count_nonzero() > 0
    flattened = spatial_update.flatten(2, 3)
    assert bool(flattened.var(dim=2).gt(0).any())
    factual["appearance_pred_tokens"].float().square().mean().backward()
    assert centered.grad is not None
    assert torch.isfinite(centered.grad).all()
    assert centered.grad.count_nonzero() > 0
    assert conditioner.cross.key_value.weight.grad is not None
    assert torch.isfinite(conditioner.cross.key_value.weight.grad).all()
    assert conditioner.cross.key_value.weight.grad.count_nonzero() > 0
    for parameter in conditioner.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.count_nonzero() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast is required")
def test_appearance_action_conditioner_checkpoint_preserves_bf16_gradients() -> None:
    cfg = replace(
        _tiny_dual_path_config(),
        appearance_action_residual_scale=0.3,
        activation_checkpointing=True,
    )
    model = NativeWorldModel(cfg).cuda().train()
    batch = {
        name: value.cuda() if isinstance(value, torch.Tensor) else value
        for name, value in _dual_path_batch(cfg).items()
    }
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(**batch, appearance_teacher_ratio=0.0)
        loss = output["appearance_pred_tokens"].float().square().mean()
    loss.backward()

    assert model.appearance_dynamics is not None
    conditioner = model.appearance_dynamics.action_conditioner
    assert conditioner is not None
    for parameter in conditioner.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.count_nonzero() > 0


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
    assert 1_000_000_000 < counts["native_1b_dual_path.yaml"] < 2_000_000_000
    assert counts["native_5b_dual_path.yaml"] > 5_000_000_000


def test_factual_dynamics_repeats_do_not_touch_policy_branch() -> None:
    cfg = replace(_tiny_config(), factual_dynamics_repeats=3)
    torch.manual_seed(41)
    model = NativeWorldModel(cfg).eval()
    batch = _batch(cfg)
    calls = 0

    def record(_module, _inputs) -> None:
        nonlocal calls
        calls += 1

    handle = model.dynamics_blocks[0].register_forward_pre_hook(record)
    try:
        baseline = model(**batch)
        changed = dict(batch)
        changed["future_factual_fine_action_values"] = (
            batch["future_factual_fine_action_values"] + 2.0
        )
        counterfactual = model(**changed)
    finally:
        handle.remove()

    assert calls == 6
    torch.testing.assert_close(
        baseline["policy_action_raw"],
        counterfactual["policy_action_raw"],
        rtol=0,
        atol=0,
    )
    assert not torch.allclose(baseline["pred_tokens"], counterfactual["pred_tokens"])


def test_zero_action_control_reuses_the_exact_factual_path() -> None:
    cfg = replace(_tiny_config(), factual_dynamics_repeats=2, dropout=0.0)
    torch.manual_seed(43)
    model = NativeWorldModel(cfg).eval()
    batch = _batch(cfg)

    output = model(**batch, compute_zero_action_control=True)
    zero_batch = dict(batch)
    zero_batch["future_factual_fine_action_values"] = torch.zeros_like(
        batch["future_factual_fine_action_values"]
    )
    zero_batch["future_factual_coarse_action_values"] = torch.zeros_like(
        batch["future_factual_coarse_action_values"]
    )
    explicit = model(**zero_batch)

    assert output["zero_action_pred_tokens"].requires_grad
    torch.testing.assert_close(
        output["zero_action_pred_tokens"],
        explicit["pred_tokens"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        output["policy_action_raw"],
        explicit["policy_action_raw"],
        rtol=0,
        atol=0,
    )


def test_render_refinement_is_stable_while_token_refinement_stays_causal() -> None:
    reference_cfg = replace(
        _tiny_config(),
        factual_dynamics_repeats=1,
        factual_action_residual_scale=0.0,
        dropout=0.0,
    )
    split_cfg = replace(
        reference_cfg,
        factual_dynamics_repeats=2,
        factual_action_residual_scale=0.3,
        render_factual_dynamics_repeats=1,
        render_factual_action_residual_scale=0.0,
    )
    torch.manual_seed(47)
    reference = NativeWorldModel(reference_cfg).eval()
    split = NativeWorldModel(split_cfg).eval()
    split.load_state_dict(reference.state_dict(), strict=True)
    batch = _batch(reference_cfg)

    reference_output = reference(**batch)
    split_output = split(**batch)

    torch.testing.assert_close(split_output["rgb"], reference_output["rgb"])
    torch.testing.assert_close(split_output["depth"], reference_output["depth"])
    torch.testing.assert_close(split_output["point"], reference_output["point"])
    torch.testing.assert_close(
        split_output["camera_pose"], reference_output["camera_pose"]
    )
    torch.testing.assert_close(
        split_output["policy_action_raw"], reference_output["policy_action_raw"]
    )
    assert not torch.allclose(
        split_output["pred_tokens"], reference_output["pred_tokens"]
    )
