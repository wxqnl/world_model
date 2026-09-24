"""Real AdamW grouping/schedule regression; not a model-quality simulation."""
from pathlib import Path
import pytest
import torch
from torch import nn
import yaml
from wm3d.training.pretrain import PretrainError, _optimizer_parameter_groups, _set_learning_rate
from wm3d.training.runtime_contract import RuntimeContractError, validate_runtime_profile

class OwnedParameters(nn.Module):
    def __init__(self):
        super().__init__()
        self.conditioning=nn.Linear(2,2,bias=False)
        self.interface=nn.Linear(2,2,bias=False)
        self.frozen=nn.Linear(2,2,bias=False).requires_grad_(False)
    def gradient_owner_modules(self):
        return {"foundation_robot_conditioning":(self.conditioning,)}

def test_default_preserves_the_exact_old_optimizer_layout():
    model=OwnedParameters()
    parameters=[p for p in model.parameters() if p.requires_grad]
    assert _optimizer_parameter_groups(model,parameters,{"peak_lr":1e-4}) is parameters

def test_owner_partition_scaled_update_and_optimizer_state_restore():
    model=OwnedParameters()
    parameters=[p for p in model.parameters() if p.requires_grad]
    config={"peak_lr":1e-3,"robot_conditioning_lr_scale":.1}
    groups=_optimizer_parameter_groups(model,parameters,config)
    ids=[id(p) for g in groups for p in g["params"]]
    assert len(ids)==len(set(ids))==len(parameters)
    assert set(ids)=={id(p) for p in parameters}
    assert {id(p) for p in groups[1]["params"]}=={id(model.conditioning.weight)}
    optimizer=torch.optim.AdamW(groups,weight_decay=0.,eps=0.,foreach=False)
    _set_learning_rate(optimizer,1e-4)
    assert [g["lr"] for g in optimizer.param_groups]==pytest.approx([1e-4,1e-5])
    before={id(p):p.detach().clone() for p in parameters}
    sum(p.sum() for p in parameters).backward()
    assert all(torch.equal(p.grad,torch.ones_like(p)) for p in parameters)
    optimizer.step()
    torch.testing.assert_close(before[id(model.interface.weight)]-model.interface.weight,torch.full_like(model.interface.weight,1e-4),atol=3e-8,rtol=1e-4)
    torch.testing.assert_close(before[id(model.conditioning.weight)]-model.conditioning.weight,torch.full_like(model.conditioning.weight,1e-5),atol=3e-8,rtol=1e-4)
    restored=torch.optim.AdamW(_optimizer_parameter_groups(model,parameters,config),foreach=False)
    restored.load_state_dict(optimizer.state_dict())
    _set_learning_rate(restored,6e-5)
    assert [g["lr"] for g in restored.param_groups]==pytest.approx([6e-5,6e-6])
    assert all(p.grad is None for p in model.frozen.parameters())

def test_cannot_silently_apply_conditioning_scale_to_an_unowned_model():
    model=nn.Linear(2,2)
    with pytest.raises(PretrainError,match="owner"):
        _optimizer_parameter_groups(model,list(model.parameters()),{"peak_lr":1e-4,"robot_conditioning_lr_scale":.1})

@pytest.mark.parametrize("scale",[0.,-1.,1.1,float("nan"),float("inf")])
def test_invalid_scale_fails_in_runtime_and_optimizer(scale):
    root=Path(__file__).resolve().parents[1]
    runtime=yaml.safe_load((root/"configs/runtime/h100_8_fsdp2.yaml").read_text())
    runtime["optimizer"]["robot_conditioning_lr_scale"]=scale
    with pytest.raises(RuntimeContractError,match="robot_conditioning_lr_scale"):
        validate_runtime_profile(runtime)
    model=OwnedParameters()
    with pytest.raises(PretrainError,match="robot_conditioning_lr_scale"):
        _optimizer_parameter_groups(model,list(model.parameters()),runtime["optimizer"])

@pytest.mark.parametrize("implementation", ["single", "fused"])
def test_explicit_adamw_implementation(implementation):
    root = Path(__file__).resolve().parents[1]
    runtime = yaml.safe_load((root / "configs/runtime/h100_8_fsdp2.yaml").read_text())
    runtime["optimizer"]["implementation"] = implementation
    validate_runtime_profile(runtime)


def test_unknown_adamw_implementation_rejected():
    root = Path(__file__).resolve().parents[1]
    runtime = yaml.safe_load((root / "configs/runtime/h100_8_fsdp2.yaml").read_text())
    runtime["optimizer"]["implementation"] = "silently_fall_back"
    with pytest.raises(RuntimeContractError, match="optimizer.implementation"):
        validate_runtime_profile(runtime)
