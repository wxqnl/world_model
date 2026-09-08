"""Real BF16 interface math and AdamW updates; not synthetic task-quality evidence."""
from copy import deepcopy
import os
import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
DEVICE = torch.device(os.environ.get("WM3D_PRECISION_TEST_DEVICE", "cpu"))
from wm3d.models.foundation_precision import FP32TrainableInterface
from wm3d.models.foundation_world_model import FoundationWorldModelConfig, FoundationModelError


def test_precision_is_explicit_and_never_promotes_the_whole_base():
    assert FoundationWorldModelConfig().trainable_interface_dtype == "model"
    FoundationWorldModelConfig(finetune_mode="adapters",trainable_interface_dtype="fp32").validate()
    with pytest.raises(FoundationModelError,match="precision"):
        FoundationWorldModelConfig(trainable_interface_dtype="fp16").validate()
    with pytest.raises(FoundationModelError,match="adapters"):
        FoundationWorldModelConfig(trainable_interface_dtype="fp32").validate()


@pytest.mark.parametrize("kind", ["linear", "norm"])
def test_base_output_and_existing_hooks_are_preserved(kind):
    layer=(nn.Linear(1024,42) if kind=="linear" else nn.LayerNorm(1024)).bfloat16().to(DEVICE)
    x=torch.randn(2,4,1024,dtype=torch.bfloat16,device=DEVICE,requires_grad=True)
    calls=[]
    layer.register_forward_hook(lambda m,a,o: calls.append(o.detach().clone()))
    expected=layer(x).detach().clone()
    rng=torch.random.get_rng_state()
    wrapped=FP32TrainableInterface(layer)
    assert torch.equal(rng,torch.random.get_rng_state())
    actual=wrapped(x)
    assert torch.equal(expected,actual)
    assert len(calls)==2 and wrapped.layer is layer
    with torch.autocast(DEVICE.type,dtype=torch.bfloat16):
        training=checkpoint(wrapped,x,use_reentrant=False)
    assert torch.equal(training,actual)
    training.float().square().mean().backward()
    assert wrapped.weight.dtype==torch.float32
    assert wrapped.weight.grad.dtype==torch.float32
    assert torch.isfinite(wrapped.weight.grad).all()
    restored=FP32TrainableInterface(deepcopy(layer))
    restored.load_state_dict(wrapped.state_dict(),strict=True)
    assert torch.equal(restored(x),actual)


def test_sub_ulp_updates_accumulate_and_adam_resume_keeps_fp32_moments():
    source=nn.Linear(30,4096,bias=False).bfloat16()
    with torch.no_grad():source.weight.fill_(.1)
    legacy=deepcopy(source)
    fixed=FP32TrainableInterface(source)
    old=torch.optim.AdamW(legacy.parameters(),lr=1e-5,weight_decay=0.)
    new=torch.optim.AdamW(fixed.parameters(),lr=1e-5,weight_decay=0.)
    before=legacy.weight.detach().clone()
    for _ in range(100):
        old.zero_grad(set_to_none=True);new.zero_grad(set_to_none=True)
        legacy.weight.sum().backward();fixed.weight.sum().backward()
        old.step();new.step()
    assert torch.equal(legacy.weight,before)
    assert not torch.equal(fixed.weight.bfloat16(),before)
    state=new.state[fixed.weight]
    assert state["exp_avg"].dtype==state["exp_avg_sq"].dtype==torch.float32
    restored=torch.optim.AdamW(fixed.parameters(),lr=1e-5)
    restored.load_state_dict(new.state_dict())
    assert restored.state[fixed.weight]["exp_avg"].dtype==torch.float32

def test_full_native_readout_has_identical_training_and_serving_outputs():
    original=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,2048)).bfloat16().to(DEVICE)
    wrapped=nn.Sequential(*(FP32TrainableInterface(deepcopy(layer)) for layer in original))
    x=torch.randn(2,4,1024,dtype=torch.bfloat16,device=DEVICE)
    with torch.autocast(DEVICE.type,dtype=torch.bfloat16):
        expected_train=original(x)
        actual_train=wrapped(x)
    expected_serve=original(x)
    actual_serve=wrapped(x)
    assert torch.equal(expected_train,expected_serve)
    assert torch.equal(expected_train,actual_train)
    assert torch.equal(expected_serve,actual_serve)
