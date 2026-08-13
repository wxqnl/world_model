from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import time

import pytest

from wm3d.training.distributed_checkpoint import canonical_sha256, sha256_file
from wm3d.training.launch_qualification import (
    LaunchQualificationError,
    build_launch_qualification,
    publish_launch_qualification,
    validate_launch_qualification,
    verify_clean_runtime_checkout,
)
from wm3d.training.pretrain import _run_contract
from wm3d.training.pretrain import (
    PretrainError,
    _atomic_json_no_clobber,
    _require_stable_run_contract,
)
from wm3d.training.resource_preflight import RESOURCE_PREFLIGHT_SCHEMA


def _resources() -> dict[str, object]:
    return {
        "gpu_name_substring": "H200",
        "minimum_gpu_memory_mib": 1,
        "require_zero_uncorrected_ecc": True,
        "require_idle_gpu": True,
        "require_full_local_nvlink_clique": False,
        "minimum_ib_rate_gbps": 1.0,
        "forbid_nccl_ib_disable": True,
        "minimum_memlock_bytes": 1,
        "minimum_nofile": 1,
        "minimum_shm_bytes": 1,
        "minimum_data_free_bytes": 1,
        "minimum_output_free_bytes": 1,
        "minimum_allreduce_gbps": 1.0,
        "maximum_preflight_age_seconds": 60,
    }


def _report() -> dict[str, object]:
    return {
        "rank": 0,
        "local_rank": 0,
        "hostname": "node0",
        "gpu": {
            "physical_index": 0,
            "name": "NVIDIA H200",
            "uuid": "GPU-0",
            "memory_total_mib": 141000,
            "uncorrected_volatile": 0,
            "uncorrected_aggregate": 0,
            "driver_version": "fixture",
            "compute_pids": [],
        },
        "nvlink": {},
        "infiniband": [
            {"device": "mlx5_0", "port": "1", "state": "ACTIVE", "rate": "400 Gb/sec"}
        ],
        "memlock": -1,
        "nofile": 1024,
        "shm_free_bytes": 1024,
        "data_free_bytes": 1024,
        "output_free_bytes": 1024,
        "allreduce_gbps": 10.0,
        "errors": [],
    }


def _receipt(
    path: Path,
    resources: dict[str, object],
    created: int,
    *,
    runtime_config_sha256: str = "a" * 64,
) -> dict[str, object]:
    value = {
        "schema": RESOURCE_PREFLIGHT_SCHEMA,
        "created_unix_ns": created,
        "runtime_config_sha256": runtime_config_sha256,
        "world_size": 1,
        "local_world_size": 1,
        "hostnames": ["node0"],
        "resource_contract_sha256": canonical_sha256(resources),
        "reports": [_report()],
        "passed": True,
        "errors": [],
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": sha256_file(path), "created_unix_ns": created}


def _contract(resources: dict[str, object] | None) -> dict[str, object]:
    return {
        "schema": "wm3d_v8_run_contract_v3",
        "name": "run",
        "lineage": "lineage",
        "data_closure_sha256": "b" * 64,
        "model_contract_sha256": "c" * 64,
        "code_commit": "1" * 40,
        "environment_lock_sha256": "d" * 64,
        "topology_contract_sha256": "e" * 64,
        "resource_contract_sha256": canonical_sha256(resources),
        "parameter_counts": {"total": 1},
        "required_gradient_owners": ["owner"],
    }


def _identity() -> list[dict[str, object]]:
    return [{"rank": 0, "hostname": "node0", "local_rank": 0, "gpu_uuid": "GPU-0"}]


def _qualification(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object], int]:
    resources = _resources()
    now = time.time_ns()
    evidence = _receipt(tmp_path / "resource.json", resources, now)
    contract = _contract(resources)
    value = build_launch_qualification(
        launch_kind="fresh",
        runtime_config_sha256="a" * 64,
        run_contract=contract,
        resources=resources,
        resource_preflight=evidence,
        rank_identities=_identity(),
        world_size=1,
        local_world_size=1,
        distributed_strategy="fsdp2",
        shard_degree=1,
        source_checkpoint=None,
        created_unix_ns=now + 1,
    )
    return value, resources, contract, now


def test_new_resource_receipt_does_not_change_stable_run_contract() -> None:
    class Model:
        pass

    config = {
        "run": {
            "name": "run",
            "lineage": "lineage",
            "code_commit": "1" * 40,
            "environment_lock_sha256": "d" * 64,
        },
        "bindings": {
            "data_closure_sha256": "b" * 64,
            "model_contract_sha256": "c" * 64,
            "objective_profile_sha256": "f" * 64,
        },
        "runtime_profile": {
            "resources": _resources(),
            "optimizer": {},
            "schedule": {},
            "train": {"global_batch_size": 1, "total_steps": 2, "seed": 1, "gradient_clip": 1.0},
            "distributed": {"strategy": "fsdp2", "shard_degree": 1, "param_dtype": "bf16", "reduce_dtype": "fp32", "output_dtype": "bf16"},
        },
    }
    # required owner discovery is unrelated to receipt identity.
    import wm3d.training.pretrain as pretrain
    original = pretrain.required_gradient_owner_names
    pretrain.required_gradient_owner_names = lambda _: ("owner",)
    try:
        first = _run_contract(config, {"total": 1}, Model())
        second = _run_contract(deepcopy(config), {"total": 1}, Model())
    finally:
        pretrain.required_gradient_owner_names = original
    assert first == second
    assert "resource_preflight_sha256" not in first


def test_legacy_v2_run_contract_is_not_promoted_to_v3(tmp_path: Path) -> None:
    path = tmp_path / "run_contract.json"
    legacy = _contract(_resources())
    legacy["schema"] = "wm3d_v8_run_contract_v2"
    legacy["resource_preflight_sha256"] = "a" * 64
    path.write_text(json.dumps(legacy), encoding="utf-8")
    current = deepcopy(legacy)
    current["schema"] = "wm3d_v8_run_contract_v3"
    current.pop("resource_preflight_sha256")
    with pytest.raises(PretrainError, match="differs"):
        _atomic_json_no_clobber(path, current)


def test_launch_qualification_is_strict_fresh_and_no_clobber(tmp_path: Path) -> None:
    value, resources, contract, now = _qualification(tmp_path)
    validate_launch_qualification(
        value,
        launch_kind="fresh",
        runtime_config_sha256="a" * 64,
        run_contract=contract,
        resources=resources,
        rank_identities=_identity(),
        world_size=1,
        local_world_size=1,
        distributed_strategy="fsdp2",
        shard_degree=1,
        source_checkpoint=None,
        now_unix_ns=now + 10,
    )
    path, _ = publish_launch_qualification(tmp_path / "run", value)
    assert path.is_file()
    with pytest.raises(LaunchQualificationError, match="overwrite"):
        publish_launch_qualification(tmp_path / "run", value)

    tampered = deepcopy(value)
    tampered["runtime_config_sha256"] = "f" * 64
    with pytest.raises(LaunchQualificationError, match="runtime_config_sha256"):
        validate_launch_qualification(
            tampered,
            launch_kind="fresh",
            runtime_config_sha256="a" * 64,
            run_contract=contract,
            resources=resources,
            rank_identities=_identity(),
            world_size=1,
            local_world_size=1,
            distributed_strategy="fsdp2",
            shard_degree=1,
            source_checkpoint=None,
            now_unix_ns=now + 10,
        )
    with pytest.raises(LaunchQualificationError, match="stale"):
        validate_launch_qualification(
            value,
            launch_kind="fresh",
            runtime_config_sha256="a" * 64,
            run_contract=contract,
            resources=resources,
            rank_identities=_identity(),
            world_size=1,
            local_world_size=1,
            distributed_strategy="fsdp2",
            shard_degree=1,
            source_checkpoint=None,
            now_unix_ns=now + 61_000_000_000,
        )


def test_qualification_can_bind_runtime_and_sealed_resource_runtime_separately(
    tmp_path: Path,
) -> None:
    resources = _resources()
    now = time.time_ns()
    resource_runtime_sha = "a" * 64
    launch_runtime_sha = "b" * 64
    evidence = _receipt(
        tmp_path / "resource.json",
        resources,
        now,
        runtime_config_sha256=resource_runtime_sha,
    )
    contract = _contract(resources)
    value = build_launch_qualification(
        launch_kind="fresh",
        runtime_config_sha256=launch_runtime_sha,
        run_contract=contract,
        resources=resources,
        resource_preflight=evidence,
        rank_identities=_identity(),
        world_size=1,
        local_world_size=1,
        distributed_strategy="ddp",
        shard_degree=1,
        source_checkpoint=None,
        created_unix_ns=now + 1,
    )
    with pytest.raises(LaunchQualificationError, match="resource receipt is invalid"):
        validate_launch_qualification(
            value,
            launch_kind="fresh",
            runtime_config_sha256=launch_runtime_sha,
            run_contract=contract,
            resources=resources,
            rank_identities=_identity(),
            world_size=1,
            local_world_size=1,
            distributed_strategy="ddp",
            shard_degree=1,
            source_checkpoint=None,
            now_unix_ns=now + 10,
        )
    validate_launch_qualification(
        value,
        launch_kind="fresh",
        runtime_config_sha256=launch_runtime_sha,
        run_contract=contract,
        resources=resources,
        rank_identities=_identity(),
        world_size=1,
        local_world_size=1,
        distributed_strategy="ddp",
        shard_degree=1,
        source_checkpoint=None,
        resource_runtime_config_sha256=resource_runtime_sha,
        now_unix_ns=now + 10,
    )


def test_run_contract_requires_regular_stable_no_clobber_file(tmp_path: Path) -> None:
    expected = _contract(_resources())
    path = tmp_path / "run_contract.json"
    _atomic_json_no_clobber(path, expected)
    assert _require_stable_run_contract(path, expected) == expected

    link = tmp_path / "run_contract_link.json"
    link.symlink_to(path)
    with pytest.raises(PretrainError, match="symlink"):
        _require_stable_run_contract(link, expected)
    with pytest.raises(PretrainError, match="symlink"):
        _atomic_json_no_clobber(link, expected)


@pytest.mark.parametrize("launch_kind", ("exact_resume", "eval"))
def test_resume_and_eval_qualification_bind_committed_source(
    tmp_path: Path, launch_kind: str
) -> None:
    checkpoint = tmp_path / "step_00000001"
    checkpoint.mkdir()
    commit = checkpoint / "COMMITTED.json"
    commit.write_text("{}", encoding="utf-8")
    source = {
        "path": str(checkpoint),
        "step": 1,
        "committed_sha256": sha256_file(commit),
        "saved_world_size": 1,
        "saved_shard_degree": 1,
        "resume_mode": "exact",
    }
    contract = _contract({})
    value = build_launch_qualification(
        launch_kind=launch_kind,
        runtime_config_sha256="a" * 64,
        run_contract=contract,
        resources=None,
        resource_preflight=None,
        rank_identities=_identity(),
        world_size=1,
        local_world_size=1,
        distributed_strategy="fsdp2",
        shard_degree=1,
        source_checkpoint=source,
    )
    validate_launch_qualification(
        value,
        launch_kind=launch_kind,
        runtime_config_sha256="a" * 64,
        run_contract=contract,
        resources=None,
        rank_identities=_identity(),
        world_size=1,
        local_world_size=1,
        distributed_strategy="fsdp2",
        shard_degree=1,
        source_checkpoint=source,
    )
    commit.write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(LaunchQualificationError, match="COMMITTED SHA"):
        validate_launch_qualification(
            value,
            launch_kind=launch_kind,
            runtime_config_sha256="a" * 64,
            run_contract=contract,
            resources=None,
            rank_identities=_identity(),
            world_size=1,
            local_world_size=1,
            distributed_strategy="fsdp2",
            shard_degree=1,
            source_checkpoint=source,
        )


def test_runtime_checkout_must_be_clean_and_exact(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    (tmp_path / "code.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "code.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "base"], check=True)
    head = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    assert verify_clean_runtime_checkout(tmp_path, head) == head
    with pytest.raises(LaunchQualificationError, match="commit mismatch"):
        verify_clean_runtime_checkout(tmp_path, "0" * 40)
    (tmp_path / "untracked.py").write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(LaunchQualificationError, match="dirty"):
        verify_clean_runtime_checkout(tmp_path, head)
