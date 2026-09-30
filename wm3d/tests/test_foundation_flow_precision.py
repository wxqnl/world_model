"""Regressions against the actual WSA scheduler, without constructing a model."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path

import pytest
import torch

from wm3d.models.foundation_flow_precision import FP32TrainingFlowScheduler


@pytest.fixture
def official():
    source = Path(os.environ.get(
        "WM3D_WSA_SOURCE_ROOT", "/data/Minko/third_party/WSA_native_20260923",
    ))
    path = source / "src/lerobot/policies/WSA_Large/core/models/wan22/schedulers/scheduler_continuous.py"
    if not path.is_file():
        pytest.skip("official WSA scheduler source is not installed")
    spec = importlib.util.spec_from_file_location("wsa_flow_precision_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WanContinuousFlowMatchScheduler()


def test_adapter_can_be_copied_without_recursive_attribute_lookup(official):
    original = FP32TrainingFlowScheduler(official)
    copied = deepcopy(original)
    assert copied._scheduler is not official
    assert copied.shift == original.shift


def test_same_random_draws_and_shift_without_bf16_zero_weight_interval(official):
    fixed = FP32TrainingFlowScheduler(official)
    torch.manual_seed(7340)
    reference = official.sample_training_t(100000, torch.device("cpu"), torch.float32)
    reference_rng = torch.random.get_rng_state()
    torch.manual_seed(7340)
    actual = fixed.sample_training_t(100000, torch.device("cpu"), torch.bfloat16)
    assert torch.equal(actual, reference)
    assert torch.equal(torch.random.get_rng_state(), reference_rng)
    assert actual.dtype == torch.float32
    old = reference.bfloat16()
    assert int((official.training_weight(old) == 0).sum()) > 0
    assert not bool((fixed.training_weight(actual) == 0).any())
    assert fixed.shift == official.shift
    assert fixed.num_train_timesteps == official.num_train_timesteps


@pytest.mark.parametrize("shape", [(2, 16, 42), (2, 48, 3, 8, 8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_noise_interpolation_matches_fp32_reference_and_restores_compute_dtype(
    official, shape, dtype,
):
    fixed = FP32TrainingFlowScheduler(official)
    torch.manual_seed(7340)
    clean = torch.randn(shape).to(dtype)
    noise = torch.randn_like(clean)
    t = torch.tensor([999.25, 500.25], dtype=torch.float32)
    expected = official.add_noise(clean.float(), noise.float(), t).to(dtype)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = fixed.add_noise(clean, noise, t)
    assert actual.dtype == dtype
    assert torch.equal(actual, expected)
    if dtype == torch.bfloat16:
        assert not torch.equal(actual, official.add_noise(clean, noise, t))


def test_clean_contribution_survives_near_upper_endpoint(official):
    fixed = FP32TrainingFlowScheduler(official)
    clean = torch.ones(1, 2, 7, dtype=torch.bfloat16)
    noise = torch.zeros_like(clean)
    t = torch.tensor([999.25])
    assert not bool(official.add_noise(clean, noise, t.bfloat16()).any())
    assert bool(fixed.add_noise(clean, noise, t).gt(0).all())
    assert bool(fixed.training_weight(t).gt(0))


def test_loss_target_and_inference_schedule_are_unchanged(official):
    fixed = FP32TrainingFlowScheduler(official)
    clean = torch.tensor([[-1., 0., 1.]], dtype=torch.bfloat16)
    noise = torch.tensor([[.5, -.5, .75]], dtype=torch.bfloat16)
    t = torch.tensor([525.])
    assert torch.equal(
        fixed.training_target(clean, noise, t),
        official.training_target(clean, noise, t),
    )
    original = official.build_inference_schedule(20, torch.device("cpu"), torch.bfloat16)
    delegated = fixed.build_inference_schedule(20, torch.device("cpu"), torch.bfloat16)
    assert all(torch.equal(a, b) for a, b in zip(original, delegated, strict=True))
    assert torch.equal(
        fixed.step(noise, delegated[1][0], clean),
        official.step(noise, original[1][0], clean),
    )
