import datetime

import pytest
import torch

from wm3d_v3.training import train


def test_eager_transport_is_noop_for_single_process():
    report = train.eager_initialize_distributed_transport(
        rank=0,
        world=1,
        device=torch.device("cpu"),
    )
    assert report == {
        "enabled": False,
        "backend": "single_process",
        "world_size": 1,
    }


def test_eager_transport_proves_rank_sum_before_data_io(monkeypatch):
    calls = []
    monkeypatch.setattr(train.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(train.dist, "get_backend", lambda: "nccl")

    def fake_all_reduce(tensor, op):
        calls.append(("all_reduce", tensor.dtype, tensor.device.type, op))
        tensor.fill_(6)

    monkeypatch.setattr(train.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(train.dist, "barrier", lambda: calls.append(("barrier",)))
    report = train.eager_initialize_distributed_transport(
        rank=1,
        world=3,
        device=torch.device("cpu"),
    )
    assert report["enabled"] is True
    assert report["rank_sum"] == 6
    assert report["expected_rank_sum"] == 6
    assert calls[0][:3] == ("all_reduce", torch.int64, "cpu")
    assert calls[1] == ("barrier",)


def test_eager_transport_fails_closed_on_wrong_rank_sum(monkeypatch):
    monkeypatch.setattr(train.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(train.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(train.dist, "all_reduce", lambda tensor, op: tensor.fill_(5))
    monkeypatch.setattr(train.dist, "barrier", lambda: None)
    with pytest.raises(RuntimeError, match="sum mismatch"):
        train.eager_initialize_distributed_transport(
            rank=0,
            world=3,
            device=torch.device("cpu"),
        )


def test_setup_ddp_honors_explicit_timeout(monkeypatch):
    observed = {}
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WM3D_DDP_BACKEND", "gloo")
    monkeypatch.setenv("WM3D_DDP_TIMEOUT_MINUTES", "7")
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)

    def fake_init_process_group(*, backend, timeout):
        observed["backend"] = backend
        observed["timeout"] = timeout

    monkeypatch.setattr(train.dist, "init_process_group", fake_init_process_group)
    monkeypatch.setattr(train.dist, "get_rank", lambda: 2)
    monkeypatch.setattr(train.dist, "get_world_size", lambda: 3)
    assert train.setup_ddp() == (2, 3, 0)
    assert observed == {
        "backend": "gloo",
        "timeout": datetime.timedelta(minutes=7),
    }
