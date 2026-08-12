from __future__ import annotations

from copy import deepcopy

import pytest

from tests.test_native_world_model import _batch, _tiny_config
from wm3d_v3.models.native_world_model import NativeWorldModel
from wm3d_v3.training.gradient_ownership import (
    GRADIENT_OWNERSHIP_SCHEMA,
    GradientOwnershipError,
    _owner_parameters,
    audit_gradient_ownership,
    required_gradient_owner_names,
    validate_gradient_ownership_receipt,
)
from wm3d_v3.training.pretrain import PretrainError, _restore_gradient_ownership


def _real_receipt() -> tuple[NativeWorldModel, dict[str, object]]:
    model = NativeWorldModel(_tiny_config()).train()
    output = model(**_batch(model.cfg))
    (
        output["policy_action_raw"].square().mean()
        + output["pred_tokens"].square().mean()
        + output["rgb"].square().mean()
        + output["depth"].mean()
        + output["point"].square().mean()
    ).backward()
    return model, audit_gradient_ownership(model)


def test_owner_table_covers_each_trainable_parameter_exactly_once() -> None:
    model = NativeWorldModel(_tiny_config())
    owners = _owner_parameters(model)
    identities = [id(parameter) for parameters in owners.values() for parameter in parameters]
    expected = [
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    ]
    assert len(identities) == len(set(identities))
    assert set(identities) == set(expected)
    assert "native_state_inputs" in owners
    assert "policy_action_inputs" in owners
    assert "auxiliary_inputs" in owners
    assert "auxiliary_inputs" not in required_gradient_owner_names(model)


def test_real_receipt_has_complete_finite_owner_evidence() -> None:
    model, receipt = _real_receipt()
    assert receipt["schema"] == GRADIENT_OWNERSHIP_SCHEMA
    validate_gradient_ownership_receipt(receipt, model)
    assert set(receipt["owners"]) == set(_owner_parameters(model))
    for name, parameters in _owner_parameters(model).items():
        assert receipt["owners"][name]["parameter_elements"] == sum(
            parameter.numel() for parameter in parameters
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda receipt: receipt.update(schema="old"),
        lambda receipt: receipt["owners"].pop("unified_action_head"),
        lambda receipt: receipt["owners"]["unified_action_head"].update(
            l2_norm=0.0
        ),
        lambda receipt: receipt["owners"]["unified_action_head"].update(
            nonfinite_elements=1
        ),
        lambda receipt: receipt["owners"]["unified_action_head"].update(
            parameter_elements=1
        ),
        lambda receipt: receipt["owners"]["unified_action_head"].update(
            gradient_elements="199"
        ),
        lambda receipt: receipt["owners"]["unified_action_head"].update(
            l2_norm=True
        ),
        lambda receipt: receipt["owners"]["auxiliary_inputs"].update(
            required=True
        ),
    ],
)
def test_checkpoint_receipt_validation_is_fail_closed(mutation) -> None:
    model, receipt = _real_receipt()
    damaged = deepcopy(receipt)
    mutation(damaged)
    with pytest.raises(GradientOwnershipError):
        validate_gradient_ownership_receipt(damaged, model)


def test_parameter_without_owner_is_rejected(monkeypatch) -> None:
    model = NativeWorldModel(_tiny_config())
    import wm3d_v3.training.gradient_ownership as module

    original = module._required_owner_parameters

    def incomplete(value):
        result = dict(original(value))
        result["policy_action_inputs"] = tuple(
            parameter
            for parameter in result["policy_action_inputs"]
            if parameter is not value.policy_query_seed
        )
        return result

    monkeypatch.setattr(module, "_required_owner_parameters", incomplete)
    with pytest.raises(GradientOwnershipError, match="cover every trainable"):
        _owner_parameters(model)


def test_resume_metadata_requires_the_complete_current_owner_abi() -> None:
    model, receipt = _real_receipt()
    assert _restore_gradient_ownership({"gradient_ownership": receipt}, model) == receipt
    with pytest.raises(PretrainError, match="lacks a passed"):
        _restore_gradient_ownership({}, model)
    damaged = deepcopy(receipt)
    damaged["owners"].pop("policy_action_inputs")
    with pytest.raises(PretrainError, match="is invalid"):
        _restore_gradient_ownership({"gradient_ownership": damaged}, model)
