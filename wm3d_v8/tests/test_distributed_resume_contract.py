from __future__ import annotations

from copy import deepcopy

import pytest

from wm3d_v3.training.distributed_checkpoint import (
    CheckpointIntegrityError,
    ResumeExpectations,
    _validate_metadata,
)
from wm3d_v3.training.pretrain import _topology_contract_sha256


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _expected(
    *,
    world_size: int = 8,
    shard_degree: int = 8,
    allow: bool = False,
    runtime_sha: str = SHA_A,
) -> ResumeExpectations:
    return ResumeExpectations(
        step=1000,
        run_lineage="wm3d-v8-native",
        runtime_config_sha256=runtime_sha,
        data_closure_sha256=SHA_B,
        model_contract_sha256=SHA_C,
        world_size=world_size,
        shard_degree=shard_degree,
        distributed_strategy="fsdp2",
        global_batch_size=128,
        topology_contract_sha256=SHA_D,
        allow_topology_reshard=allow,
    )


def _metadata(*, world_size: int = 8, shard_degree: int = 8) -> dict[str, object]:
    return {
        "step": 1000,
        "run_lineage": "wm3d-v8-native",
        "runtime_config_sha256": SHA_A,
        "data_closure_sha256": SHA_B,
        "model_contract_sha256": SHA_C,
        "world_size": world_size,
        "shard_degree": shard_degree,
        "distributed_strategy": "fsdp2",
        "global_batch_size": 128,
        "topology_contract_sha256": SHA_D,
        "sampler_progress": {"next_optimizer_step": 1000},
    }


def test_same_topology_is_always_exact_and_runtime_bound() -> None:
    assert _validate_metadata(_metadata(), _expected(allow=True)) == "exact"
    with pytest.raises(CheckpointIntegrityError, match="runtime_config_sha256"):
        _validate_metadata(_metadata(), _expected(allow=True, runtime_sha="e" * 64))


def test_world_size_change_is_fail_closed_by_default() -> None:
    with pytest.raises(CheckpointIntegrityError, match="allow_topology_reshard is false"):
        _validate_metadata(_metadata(world_size=8), _expected(world_size=16))


def test_fsdp2_replica_degree_can_change_with_same_shard_mesh() -> None:
    assert (
        _validate_metadata(
            _metadata(world_size=8, shard_degree=8),
            _expected(world_size=16, shard_degree=8, allow=True),
        )
        == "topology_reshard"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"distributed_strategy": "ddp"}, "only for FSDP2"),
        ({"shard_degree": 4}, "unchanged shard mesh degree"),
        ({"global_batch_size": 64}, "global_batch_size"),
        ({"topology_contract_sha256": "e" * 64}, "topology_contract_sha256"),
        ({"sampler_progress": {}}, "sampler progress"),
    ],
)
def test_topology_reshard_rejects_incompatible_contracts(
    mutation: dict[str, object], message: str
) -> None:
    metadata = _metadata(world_size=8)
    metadata.update(mutation)
    with pytest.raises(CheckpointIntegrityError, match=message):
        _validate_metadata(metadata, _expected(world_size=16, allow=True))


def _runtime_config() -> dict[str, object]:
    return {
        "run": {
            "lineage": "wm3d-v8-native",
            "code_commit": "1" * 40,
            "environment_lock_sha256": SHA_A,
        },
        "bindings": {
            "model_contract_sha256": SHA_B,
            "data_closure_sha256": SHA_C,
            "objective_profile_sha256": SHA_D,
        },
        "runtime_profile": {
            "optimizer": {"name": "adamw", "peak_lr": 1e-4},
            "schedule": {"name": "warmup_stable_cosine", "warmup_steps": 10},
            "train": {
                "global_batch_size": 128,
                "total_steps": 100_000,
                "seed": 3407,
                "gradient_clip": 1.0,
                "micro_batch_size": 1,
                "gradient_accumulation": 1,
                "num_workers": 8,
                "checkpoint_interval": 1000,
            },
            "distributed": {
                "strategy": "fsdp2",
                "shard_degree": 8,
                "param_dtype": "bf16",
                "reduce_dtype": "fp32",
                "output_dtype": "bf16",
            },
        },
    }


def test_topology_hash_excludes_launch_shape_but_not_training_semantics() -> None:
    first = _runtime_config()
    second = deepcopy(first)
    second["runtime_profile"]["train"].update(  # type: ignore[index]
        micro_batch_size=2,
        gradient_accumulation=4,
        num_workers=16,
        checkpoint_interval=500,
    )
    assert _topology_contract_sha256(first) == _topology_contract_sha256(second)

    second["runtime_profile"]["train"]["global_batch_size"] = 256  # type: ignore[index]
    assert _topology_contract_sha256(first) != _topology_contract_sha256(second)

