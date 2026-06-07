import math

import pytest


def test_wsd_schedule_warms_up_holds_peak_then_decays_to_min_lr():
    from wm3d_v3.training.lr_schedule import build_lr_lambda

    cfg = {
        "train": {"lr": 2.0e-5, "warmup_steps": 3},
        "lr_schedule": {
            "type": "wsd",
            "peak_lr": 1.0e-5,
            "min_lr": 1.0e-6,
            "warmup_steps": 10,
            "decay_frac": 0.20,
        },
    }

    lr_lambda = build_lr_lambda(cfg, total_steps=100)

    assert lr_lambda(0) == pytest.approx(0.1)
    assert lr_lambda(9) == pytest.approx(1.0)
    assert lr_lambda(10) == pytest.approx(1.0)
    assert lr_lambda(81) == pytest.approx(1.0)
    assert lr_lambda(82) == pytest.approx(1.0)
    assert 0.1 < lr_lambda(90) < 1.0
    assert lr_lambda(99) == pytest.approx(0.1)
    assert lr_lambda(120) == pytest.approx(0.1)


def test_legacy_cosine_schedule_is_default_when_lr_schedule_is_absent():
    from wm3d_v3.training.lr_schedule import build_lr_lambda

    cfg = {"train": {"lr": 2.0e-5, "warmup_steps": 10}}
    lr_lambda = build_lr_lambda(cfg, total_steps=100)

    assert lr_lambda(0) == pytest.approx(0.1)
    assert lr_lambda(9) == pytest.approx(1.0)
    assert lr_lambda(10) == pytest.approx(1.0)
    step = 55
    prog = (step - 10) / max(1, 100 - 10)
    expected = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))
    assert lr_lambda(step) == pytest.approx(expected)


def test_optimizer_settings_prefer_top_level_1b_pretraining_fields():
    from wm3d_v3.training.lr_schedule import resolve_optimizer_settings

    cfg = {
        "train": {"lr": 2.0e-5, "weight_decay": 0.01},
        "optimizer": {"type": "adamw", "betas": [0.9, 0.95], "weight_decay": 0.02},
        "lr_schedule": {"type": "wsd", "peak_lr": 1.0e-5, "min_lr": 1.0e-6},
    }

    settings = resolve_optimizer_settings(cfg)

    assert settings.type == "adamw"
