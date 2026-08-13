from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import shutil

import pytest
import torch
import torch.distributed as dist

import wm3d.training.distributed_checkpoint as checkpoint_module
from wm3d.training.distributed_checkpoint import (
    CheckpointIntegrityError,
    DistributedCheckpointManager,
    ResumeExpectations,
    _validate_metadata,
)
from wm3d.training.pretrain import _topology_contract_sha256


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
    extra_immutable_metadata: dict[str, object] | None = None,
) -> ResumeExpectations:
    return ResumeExpectations(
        step=1000,
        run_lineage="wm3d-native",
        runtime_config_sha256=runtime_sha,
        data_closure_sha256=SHA_B,
        model_contract_sha256=SHA_C,
        world_size=world_size,
        shard_degree=shard_degree,
        distributed_strategy="fsdp2",
        global_batch_size=128,
        topology_contract_sha256=SHA_D,
        allow_topology_reshard=allow,
        extra_immutable_metadata=extra_immutable_metadata,
    )


def _metadata(*, world_size: int = 8, shard_degree: int = 8) -> dict[str, object]:
    return {
        "step": 1000,
        "run_lineage": "wm3d-native",
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


def test_extra_immutable_metadata_is_optional_and_exact() -> None:
    metadata = _metadata()
    assert _validate_metadata(metadata, _expected()) == "exact"

    metadata["rollout_audit_sha256"] = SHA_D
    expected = _expected(
        extra_immutable_metadata={"rollout_audit_sha256": SHA_D},
    )
    assert _validate_metadata(metadata, expected) == "exact"

    missing = dict(metadata)
    del missing["rollout_audit_sha256"]
    with pytest.raises(CheckpointIntegrityError, match="rollout_audit_sha256"):
        _validate_metadata(missing, expected)

    wrong = dict(metadata, rollout_audit_sha256="e" * 64)
    with pytest.raises(CheckpointIntegrityError, match="rollout_audit_sha256"):
        _validate_metadata(wrong, expected)


def test_extra_immutable_metadata_rejects_invalid_expectation_keys() -> None:
    expected = _expected(extra_immutable_metadata={"": SHA_D})
    with pytest.raises(CheckpointIntegrityError, match="non-empty strings"):
        _validate_metadata(_metadata(), expected)


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
            "lineage": "wm3d-native",
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

    second = deepcopy(first)
    second["runtime_profile"]["resources"] = {"minimum_shm_bytes": 64_000_000_000}  # type: ignore[index]
    assert _topology_contract_sha256(first) != _topology_contract_sha256(second)


def _small_dcp_metadata(*, progress: int) -> dict[str, object]:
    return {
        "run_lineage": "private-snapshot-regression",
        "runtime_config_sha256": SHA_A,
        "data_closure_sha256": SHA_B,
        "model_contract_sha256": SHA_C,
        "shard_degree": 1,
        "distributed_strategy": "ddp",
        "global_batch_size": 1,
        "topology_contract_sha256": SHA_D,
        "sampler_progress": {"next_optimizer_step": progress},
        "initial_seed": 17,
    }


def _small_dcp_expected() -> ResumeExpectations:
    return ResumeExpectations(
        step=1,
        run_lineage="private-snapshot-regression",
        runtime_config_sha256=SHA_A,
        data_closure_sha256=SHA_B,
        model_contract_sha256=SHA_C,
        world_size=1,
        shard_degree=1,
        distributed_strategy="ddp",
        global_batch_size=1,
        topology_contract_sha256=SHA_D,
    )


def _replace_regular_file(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".attacker")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


@pytest.mark.parametrize("evaluation", [False, True])
def test_private_snapshot_is_the_only_dcp_load_source_after_original_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evaluation: bool,
) -> None:
    """A verified snapshot must survive replacement of the original checkpoint."""

    rendezvous = tmp_path / "process-group"
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1
    )
    try:
        manager = DistributedCheckpointManager(tmp_path / "checkpoints")
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with torch.no_grad():
            model.weight.fill_(1.0)
        trusted = manager.save(
            step=1,
            model=model,
            optimizer=optimizer,
            metadata=_small_dcp_metadata(progress=1),
            rank_state={"source": "trusted"},
        )
        with torch.no_grad():
            model.weight.fill_(2.0)
        attacker = manager.save(
            step=2,
            model=model,
            optimizer=optimizer,
            metadata=_small_dcp_metadata(progress=2),
            rank_state={"source": "attacker"},
        )
        with torch.no_grad():
            model.weight.zero_()

        original_verify = checkpoint_module._distributed_verify_payload
        swapped = False

        def verify_snapshot_then_swap_original(
            root: Path, files: dict[str, dict[str, object]]
        ) -> None:
            nonlocal swapped
            original_verify(root, files)
            if swapped:
                return
            swapped = True
            for source in sorted((attacker / "distcp").iterdir()):
                _replace_regular_file(source, trusted / "distcp" / source.name)
            _replace_regular_file(
                attacker / "rank_state/rank_00000.pt",
                trusted / "rank_state/rank_00000.pt",
            )

        monkeypatch.setattr(
            checkpoint_module,
            "_distributed_verify_payload",
            verify_snapshot_then_swap_original,
        )
        if evaluation:
            metadata = manager.load_model_for_evaluation(
                path=trusted,
                model=model,
                expected=_small_dcp_expected(),
            )
            assert metadata["step"] == 1
        else:
            metadata, progress = manager.load(
                path=trusted,
                model=model,
                optimizer=optimizer,
                expected=_small_dcp_expected(),
            )
            assert metadata["step"] == 1
            assert progress == {"source": "trusted"}
        assert swapped is True
        assert model.weight.detach().item() == 1.0
        snapshot_parent = tmp_path / ".wm3d_checkpoint_load_snapshots"
        assert snapshot_parent.is_dir()
        assert list(snapshot_parent.iterdir()) == []
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_private_snapshot_is_removed_when_payload_verification_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendezvous = tmp_path / "process-group"
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1
    )
    try:
        manager = DistributedCheckpointManager(tmp_path / "checkpoints")
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        checkpoint = manager.save(
            step=1,
            model=model,
            optimizer=optimizer,
            metadata=_small_dcp_metadata(progress=1),
            rank_state={"source": "trusted"},
        )

        def fail_verification(*_args: object, **_kwargs: object) -> None:
            raise CheckpointIntegrityError("injected payload verification failure")

        monkeypatch.setattr(
            checkpoint_module, "_distributed_verify_payload", fail_verification
        )
        with pytest.raises(
            CheckpointIntegrityError, match="injected payload verification failure"
        ):
            manager.load(
                path=checkpoint,
                model=model,
                optimizer=optimizer,
                expected=_small_dcp_expected(),
            )
        assert list((tmp_path / ".wm3d_checkpoint_load_snapshots").iterdir()) == []
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("evaluation", [False, True])
def test_private_snapshot_is_removed_when_dcp_load_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evaluation: bool,
) -> None:
    rendezvous = tmp_path / "process-group"
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1
    )
    try:
        manager = DistributedCheckpointManager(tmp_path / "checkpoints")
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        checkpoint = manager.save(
            step=1,
            model=model,
            optimizer=optimizer,
            metadata=_small_dcp_metadata(progress=1),
            rank_state={"source": "trusted"},
        )

        def fail_dcp_load(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("injected DCP load failure")

        monkeypatch.setattr(checkpoint_module.dcp, "load", fail_dcp_load)
        with pytest.raises(CheckpointIntegrityError, match="injected DCP load failure"):
            if evaluation:
                manager.load_model_for_evaluation(
                    path=checkpoint,
                    model=model,
                    expected=_small_dcp_expected(),
                )
            else:
                manager.load(
                    path=checkpoint,
                    model=model,
                    optimizer=optimizer,
                    expected=_small_dcp_expected(),
                )
        assert list((tmp_path / ".wm3d_checkpoint_load_snapshots").iterdir()) == []
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
