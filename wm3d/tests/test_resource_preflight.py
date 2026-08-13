from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import wm3d.training.resource_preflight as resource_preflight
import wm3d.training.pretrain as pretrain
from wm3d.training.resource_preflight import (
    RESOURCE_PREFLIGHT_SCHEMA,
    ResourcePreflightError,
    _canonical_sha256,
    _parse_ib_rate_gbps,
    _parse_nvlink_topology,
    _visible_gpu_identifier,
    run_resource_preflight,
    validate_current_rank_identities,
    validate_resource_receipt,
)


def test_failed_resource_probe_returns_a_persistable_failed_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = {
        "rank": 0,
        "local_rank": 0,
        "hostname": "node0",
        "gpu": {"uuid": "GPU-0"},
        "errors": ["GPU name mismatch"],
    }
    monkeypatch.setattr(resource_preflight, "_local_report", lambda **_: report)

    def _all_gather_object(gathered: list, value: object) -> None:
        gathered[0] = value

    monkeypatch.setattr(
        resource_preflight.dist, "all_gather_object", _all_gather_object
    )
    monkeypatch.setattr(
        resource_preflight.dist,
        "broadcast_object_list",
        lambda values, src: None,
    )
    context = SimpleNamespace(
        rank=0,
        local_rank=0,
        local_world_size=1,
        world_size=1,
        device=torch.device("cpu"),
    )
    resources = _resources()
    receipt = run_resource_preflight(
        resources=resources,
        context=context,
        runtime_config_sha256="a" * 64,
        cache_root=tmp_path,
        output_root=tmp_path / "output",
    )
    assert receipt["passed"] is False
    assert receipt["errors"] == ["rank 0: GPU name mismatch"]
    with pytest.raises(ResourcePreflightError, match="not a clean pass"):
        validate_resource_receipt(
            receipt,
            resources=resources,
            runtime_config_sha256="a" * 64,
            world_size=1,
        )


def test_pretrain_persists_failed_resource_receipt_before_refusing_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resources = _resources()
    receipt = _receipt(resources, created_ns=123456789)
    receipt["reports"][0]["errors"] = ["GPU name mismatch"]
    receipt["passed"] = False
    receipt["errors"] = ["rank 0: GPU name mismatch"]
    monkeypatch.setattr(pretrain, "run_resource_preflight", lambda **_: receipt)
    monkeypatch.setattr(
        pretrain.dist, "broadcast_object_list", lambda values, src: None
    )
    config = {
        "runtime_profile": {"resources": resources},
        "run": {"output_root": str(tmp_path / "run")},
        "data_closure": {"cache_root": str(tmp_path)},
    }
    context = SimpleNamespace(is_rank0=True)
    with pytest.raises(pretrain.PretrainError, match="GPU name mismatch"):
        pretrain._resource_preflight(config, "a" * 64, context)
    paths = list((tmp_path / "run").glob("resource_preflight_*.json"))
    assert len(paths) == 1
    assert paths[0].read_text(encoding="utf-8")
    persisted = json.loads(paths[0].read_text(encoding="utf-8"))
    assert persisted == receipt


def test_parse_ib_rate_is_strict() -> None:
    assert _parse_ib_rate_gbps("400 Gb/sec (4X HDR)") == pytest.approx(400.0)
    with pytest.raises(ResourcePreflightError, match="unrecognized"):
        _parse_ib_rate_gbps("unknown")


def test_parse_nvlink_requires_exact_local_clique() -> None:
    output = """
            GPU0 GPU1 CPU Affinity
    GPU0     X    NV4  0-31
    GPU1     NV4  X    0-31
    """
    value = _parse_nvlink_topology(output, local_world_size=2)
    assert set(value["matrix"]) == {"GPU0", "GPU1"}
    with pytest.raises(ResourcePreflightError, match="incomplete"):
        _parse_nvlink_topology(output.replace("NV4", "PIX", 1), local_world_size=2)


def test_nvlink_parser_uses_selected_physical_gpus_not_local_ordinals() -> None:
    output = """
            GPU0 GPU1 GPU2 GPU3 CPU Affinity
    GPU0     X    NV4  NV4  NV4  0-31
    GPU1     NV4  X    NV4  NV4  0-31
    GPU2     NV4  NV4  X    NV4  0-31
    GPU3     NV4  NV4  NV4  X    0-31
    """
    value = _parse_nvlink_topology(output, physical_gpu_indices=[2, 3])
    assert value["matrix"] == {"GPU2": ["X", "NV4"], "GPU3": ["NV4", "X"]}
    styled = output.replace("GPU0 GPU1 GPU2 GPU3", "\x1b[4mGPU0 GPU1 GPU2 GPU3\x1b[0m")
    assert _parse_nvlink_topology(
        styled, physical_gpu_indices=[2, 3]
    ) == value


def test_visible_gpu_identifier_uses_pytorch_cuda_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resource_preflight.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        resource_preflight.torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(uuid=("physical-two" if index == 0 else "physical-three")),
    )
    assert _visible_gpu_identifier(0) == "GPU-physical-two"
    assert _visible_gpu_identifier(1) == "GPU-physical-three"


def test_resource_module_does_not_write_or_delete_paths() -> None:
    source = Path(__file__).resolve().parents[1] / "wm3d/training/resource_preflight.py"
    text = source.read_text(encoding="utf-8")
    assert ".unlink(" not in text
    assert ".write_" not in text


def _resources() -> dict:
    return {
        "gpu_name_substring": "H200",
        "minimum_gpu_memory_mib": 135000,
        "require_zero_uncorrected_ecc": True,
        "require_idle_gpu": True,
        "require_full_local_nvlink_clique": True,
        "minimum_ib_rate_gbps": 200.0,
        "forbid_nccl_ib_disable": True,
        "minimum_memlock_bytes": 1,
        "minimum_nofile": 1,
        "minimum_shm_bytes": 1,
        "minimum_data_free_bytes": 1,
        "minimum_output_free_bytes": 1,
        "minimum_allreduce_gbps": 1.0,
        "maximum_preflight_age_seconds": 60,
    }


def _receipt(resources: dict, *, created_ns: int) -> dict:
    reports = [
        {
            "rank": rank,
            "local_rank": rank,
            "hostname": "node0",
            "gpu": {
                "physical_index": rank,
                "name": "NVIDIA H200",
                "uuid": f"GPU-{rank}",
                "memory_total_mib": 141000,
                "uncorrected_volatile": 0,
                "uncorrected_aggregate": 0,
                "driver_version": "fixture",
                "compute_pids": [],
            },
            "nvlink": {
                "matrix": {"GPU0": ["X", "NV4"], "GPU1": ["NV4", "X"]},
                "matrix_sha256": _canonical_sha256(
                    {"GPU0": ["X", "NV4"], "GPU1": ["NV4", "X"]}
                ),
            },
            "infiniband": [
                {
                    "device": "mlx5_0",
                    "port": "1",
                    "state": "4: ACTIVE",
                    "rate": "400 Gb/sec (4X HDR)",
                }
            ],
            "memlock": -1,
            "nofile": 1024,
            "shm_free_bytes": 1024,
            "data_free_bytes": 1024,
            "output_free_bytes": 1024,
            "allreduce_gbps": 10.0,
            "errors": [],
        }
        for rank in range(2)
    ]
    return {
        "schema": RESOURCE_PREFLIGHT_SCHEMA,
        "created_unix_ns": created_ns,
        "runtime_config_sha256": "a" * 64,
        "world_size": 2,
        "local_world_size": 2,
        "hostnames": ["node0"],
        "resource_contract_sha256": _canonical_sha256(resources),
        "reports": reports,
        "passed": True,
        "errors": [],
    }


def test_persisted_resource_receipt_is_exact_and_fresh() -> None:
    resources = _resources()
    now_ns = 10_000_000_000
    receipt = _receipt(resources, created_ns=9_000_000_000)
    assert validate_resource_receipt(
        receipt,
        resources=resources,
        runtime_config_sha256="a" * 64,
        world_size=2,
        now_unix_ns=now_ns,
    ) == 9_000_000_000
    tampered = dict(receipt)
    tampered["hostnames"] = ["another"]
    with pytest.raises(ResourcePreflightError, match="rank/host closure"):
        validate_resource_receipt(
            tampered,
            resources=resources,
            runtime_config_sha256="a" * 64,
            world_size=2,
            now_unix_ns=now_ns,
        )
    with pytest.raises(ResourcePreflightError, match="stale"):
        validate_resource_receipt(
            receipt,
            resources=resources,
            runtime_config_sha256="a" * 64,
            world_size=2,
            now_unix_ns=100_000_000_000,
        )


def test_launch_rank_and_gpu_identities_must_match_preflight() -> None:
    receipt = _receipt(_resources(), created_ns=9_000_000_000)
    identities = [
        {"hostname": "node0", "local_rank": rank, "gpu_uuid": f"GPU-{rank}"}
        for rank in range(2)
    ]
    validate_current_rank_identities(receipt, identities)
    identities[1] = dict(identities[1], gpu_uuid="GPU-other")
    with pytest.raises(ResourcePreflightError, match="differs from preflight"):
        validate_current_rank_identities(receipt, identities)


def test_resource_receipt_rejects_duplicate_gpu_identity() -> None:
    resources = _resources()
    receipt = _receipt(resources, created_ns=9_000_000_000)
    receipt["reports"][1]["gpu"]["uuid"] = "GPU-0"
    with pytest.raises(ResourcePreflightError, match="duplicated"):
        validate_resource_receipt(
            receipt,
            resources=resources,
            runtime_config_sha256="a" * 64,
            world_size=2,
            now_unix_ns=10_000_000_000,
        )


def test_resource_receipt_rejects_rehashed_malformed_nvlink_matrix() -> None:
    resources = _resources()
    receipt = _receipt(resources, created_ns=9_000_000_000)
    matrix = {"GPU0": ["NV4", "X"], "GPU1": ["X", "NV4"]}
    receipt["reports"][0]["nvlink"] = {
        "matrix": matrix,
        "matrix_sha256": _canonical_sha256(matrix),
    }
    with pytest.raises(ResourcePreflightError, match="NVLink clique"):
        validate_resource_receipt(
            receipt,
            resources=resources,
            runtime_config_sha256="a" * 64,
            world_size=2,
            now_unix_ns=10_000_000_000,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("allreduce_gbps", 0.5, "all-reduce"),
        ("shm_free_bytes", 0, "shm_free_bytes"),
        ("nofile", "1024", "nofile"),
    ],
)
def test_resource_receipt_rechecks_persisted_thresholds_and_types(
    field: str, value: object, message: str
) -> None:
    resources = _resources()
    receipt = _receipt(resources, created_ns=9_000_000_000)
    receipt["reports"][0][field] = value
    with pytest.raises(ResourcePreflightError, match=message):
        validate_resource_receipt(
            receipt,
            resources=resources,
            runtime_config_sha256="a" * 64,
            world_size=2,
            now_unix_ns=10_000_000_000,
        )
