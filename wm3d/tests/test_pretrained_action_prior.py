"""Regression for the actual 24-to-42 pretrained action boundary expansion."""
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from safetensors.torch import save_file
from wm3d.models.foundation_world_model import (
    FoundationModelError, FoundationWorldModelConfig,
    _retain_expanded_wsa_action_prior,
)

PREFIX = "model.mot.mixtures.action."

def fixture(tmp_path, width=42):
    torch.manual_seed(7)
    old_encoder = nn.Linear(24, 1024)
    old_head = nn.Linear(1024, 24)
    source = {
        PREFIX+"action_encoder.weight": old_encoder.weight.detach().clone(),
        PREFIX+"action_encoder.bias": old_encoder.bias.detach().clone(),
        PREFIX+"head.weight": old_head.weight.detach().clone(),
        PREFIX+"head.bias": old_head.bias.detach().clone(),
    }
    save_file(source, str(tmp_path/"model.safetensors"))
    action = nn.Module()
    action.action_encoder = nn.Linear(width, 1024)
    action.head = nn.Linear(1024, width)
    proprio = nn.Linear(30,4096)
    policy = SimpleNamespace(model=SimpleNamespace(action_expert=action,proprio_encoder=proprio))
    return policy, source, old_encoder, old_head

def test_retains_released_coordinates_and_only_zeros_new_action_capacity(tmp_path):
    policy, source, old_encoder, old_head = fixture(tmp_path)
    proprio_before = policy.model.proprio_encoder.weight.detach().clone()
    receipt = _retain_expanded_wsa_action_prior(policy,tmp_path)
    action = policy.model.action_expert
    assert len(receipt)==4
    assert torch.equal(action.action_encoder.weight[:,:24],old_encoder.weight)
    assert torch.equal(action.action_encoder.bias,old_encoder.bias)
    assert torch.equal(action.head.weight[:24],old_head.weight)
    assert torch.equal(action.head.bias[:24],old_head.bias)
    assert torch.count_nonzero(action.action_encoder.weight[:,24:])==0
    assert torch.count_nonzero(action.head.weight[24:])==0
    assert torch.count_nonzero(action.head.bias[24:])==0
    assert torch.equal(policy.model.proprio_encoder.weight,proprio_before)
    # New storage coordinates cannot perturb the original action projection.
    x=torch.randn(2,32,42)
    encoded=action.action_encoder(x)
    torch.testing.assert_close(encoded,old_encoder(x[...,:24]),atol=1e-6,rtol=1e-5)
    output=action.head(encoded)
    torch.testing.assert_close(output[...,:24],old_head(encoded),atol=1e-6,rtol=1e-5)
    assert torch.count_nonzero(output[...,24:])==0
    # Newly valid coordinates are not frozen by their zero initialization.
    (output-torch.randn_like(output)).square().mean().backward()
    for grad in (action.action_encoder.weight.grad[:,24:],action.head.weight.grad[24:]):
        assert torch.isfinite(grad).all() and torch.count_nonzero(grad)>0

def test_rejects_truncation_before_mutating_any_parameter(tmp_path):
    policy, *_ = fixture(tmp_path,width=7)
    before=policy.model.action_expert.action_encoder.weight.detach().clone()
    with pytest.raises(FoundationModelError,match="expansion"):
        _retain_expanded_wsa_action_prior(policy,tmp_path)
    assert torch.equal(policy.model.action_expert.action_encoder.weight,before)

def test_old_profiles_keep_legacy_loading_and_new_mode_is_explicit():
    assert FoundationWorldModelConfig().pretrained_interface_init=="legacy_skip"
    FoundationWorldModelConfig(pretrained_interface_init="pretrained_action_prefix").validate()
    with pytest.raises(FoundationModelError,match="initialization"):
        FoundationWorldModelConfig(pretrained_interface_init="silently_truncate").validate()
