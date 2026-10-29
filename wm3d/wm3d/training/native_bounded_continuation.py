"""The explicit, single bounded transition from completed native step 783.

This changes the destination budget and LR, not the frozen source runtime.
It is complete-state continuation, never same-schedule exact trajectory resume.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping

CONTINUATION_SCHEMA = "wm3d_native_bounded_continuation_v1"
TRANSITION_SCHEMA = "wm3d_native_bounded_continuation_transition_v1"


def validate_native_bounded_declaration(value: object) -> None:
    from wm3d.training.runtime_contract import RuntimeContractError

    required = {"schema", "source_runtime_path", "source_checkpoint_path", "source_step"}
    if not isinstance(value, dict) or set(value) != required:
        raise RuntimeContractError("native bounded continuation fields mismatch")
    if value["schema"] != CONTINUATION_SCHEMA:
        raise RuntimeContractError("native bounded continuation schema mismatch")
    if type(value["source_step"]) is not int or value["source_step"] != 783:
        raise RuntimeContractError("native bounded continuation requires source step 783")
    for field in ("source_runtime_path", "source_checkpoint_path"):
        path = Path(str(value[field]))
        if not path.is_absolute() or path.is_symlink():
            raise RuntimeContractError(f"native bounded continuation {field} must be absolute and not a symlink")
    if Path(value["source_checkpoint_path"]).name != "step_00000783":
        raise RuntimeContractError("native bounded continuation checkpoint must explicitly name step 783")


def _differences(source: object, destination: object, prefix: str = "") -> dict[str, dict[str, Any]]:
    if isinstance(source, Mapping) and isinstance(destination, Mapping):
        changes: dict[str, dict[str, Any]] = {}
        for key in sorted(set(source) | set(destination)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in source or key not in destination:
                changes[path] = {"source": source.get(key), "destination": destination.get(key)}
            else:
                changes.update(_differences(source[key], destination[key], path))
        return changes
    return {} if source == destination else {prefix: {"source": source, "destination": destination}}


def validate_native_bounded_transition(
    source_config: Mapping[str, Any],
    destination_config: Mapping[str, Any],
    *,
    source_runtime_config_sha256: str,
    source_budget_stop: Mapping[str, Any],
    source_committed_sha256: str,
) -> dict[str, Any]:
    """Validate the full whitelist; inputs are officially validated runtimes."""
    from wm3d.training.runtime_contract import RuntimeContractError, canonical_sha256

    source = deepcopy(dict(source_config))
    destination = deepcopy(dict(destination_config))
    declaration = destination["runtime_profile"]["train"].get("native_bounded_continuation")
    validate_native_bounded_declaration(declaration)
    src_runtime, dst_runtime = source["runtime_profile"], destination["runtime_profile"]
    src_train, dst_train = src_runtime["train"], dst_runtime["train"]
    if src_train.get("native_bounded_continuation") is not None:
        raise RuntimeContractError("native bounded continuation cannot chain another transition")
    if any(train.get(field) is not None for train in (src_train, dst_train)
           for field in ("model_warmstart_checkpoint", "model_warmstart_new_parameter_prefixes")):
        raise RuntimeContractError("native bounded continuation forbids model warmstart")
    if any(runtime["checkpoint"].get("allow_topology_reshard") for runtime in (src_runtime, dst_runtime)):
        raise RuntimeContractError("native bounded continuation forbids topology reshard")
    if (source["model_profile"].get("architecture") != "foundation_world_model"
            or source["model_profile"]["model"].get("initialization") != "wan_video"):
        raise RuntimeContractError("native bounded continuation requires the native Wan source")
    source_root = Path(source["run"]["output_root"])
    if (Path(declaration["source_runtime_path"]) != source_root.parent / "runtime.json"
            or Path(declaration["source_checkpoint_path"]) != source_root / "checkpoints" / "step_00000783"):
        raise RuntimeContractError("native bounded continuation source paths disagree with original runtime")
    if (src_runtime["expected_world_size"] != 24
            or src_runtime["distributed"]["strategy"] != "fsdp2"
            or src_runtime["distributed"]["shard_degree"] != 8
            or src_train["micro_batch_size"] != 4
            or src_train["global_batch_size"] != 96
            or src_train["gradient_accumulation"] != 1
            or src_train["seed"] != 3407
            or src_train.get("sampling_mode") != "source_count_epochs_v1"
            or src_train.get("sampling_epochs") != 27
            or src_train["total_steps"] != 783):
        raise RuntimeContractError("native bounded continuation source topology/sampling contract differs")
    selection = src_train.get("small_data_selection", {})
    if (selection.get("source") != "oxe_berkeley_autolab_ur5"
            or selection.get("train_episodes") != 128 or selection.get("val_episodes") != 8
            or selection.get("window_count_by_split") != {"train": 2689, "val": 164}):
        raise RuntimeContractError("native bounded continuation source episode selection differs")
    if (source_budget_stop.get("status") != "bounded_training_completed"
            or type(source_budget_stop.get("step")) is not int
            or source_budget_stop["step"] != 783
            or source_budget_stop.get("completion_reason") != "configured_steps_completed"
            or source_budget_stop.get("source_checkpoint") is not None):
        raise RuntimeContractError("native bounded continuation source is not completed final step 783")
    for field in ("name", "lineage", "output_root", "code_commit"):
        if destination["run"][field] == source["run"][field]:
            raise RuntimeContractError(f"native bounded continuation requires new run.{field}")
    if (dst_train["total_steps"] != 1566 or dst_train.get("sampling_epochs") != 54
            or dst_train["checkpoint_interval"] != 783 or dst_train["checkpoint_steps"] != []
            or dst_runtime["optimizer"]["peak_lr"] != 1e-5
            or dst_runtime["optimizer"]["min_lr"] != 1e-5
            or dst_runtime["schedule"]["warmup_steps"] != 0
            or dst_runtime["optimizer"].get("video_backbone_lr_scale") != 0.5):
        raise RuntimeContractError("native bounded continuation destination budget/LR/only-final save differs")
    deadline = dst_train.get("small_data_deadline_unix")
    if (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline) or deadline <= 0 or deadline == src_train.get("small_data_deadline_unix")):
        raise RuntimeContractError("native bounded continuation requires a new explicit deadline")
    allowed = {
        "run.name", "run.lineage", "run.output_root", "run.code_commit", "runtime_profile.name",
        "runtime_profile.train.total_steps", "runtime_profile.train.sampling_epochs",
        "runtime_profile.train.small_data_deadline_unix",
        "runtime_profile.optimizer.peak_lr", "runtime_profile.optimizer.min_lr",
        "runtime_profile.schedule.warmup_steps",
    }
    dst_train.pop("native_bounded_continuation")
    # Derived binding changes only because its validated profile changes.
    source["bindings"].pop("runtime_profile_sha256")
    destination["bindings"].pop("runtime_profile_sha256")
    changes = _differences(source, destination)
    unexpected = sorted(set(changes) - allowed)
    if unexpected:
        raise RuntimeContractError(f"native bounded continuation unexpected changes: {unexpected}")
    return {
        "schema": TRANSITION_SCHEMA,
        "restore_mode": "complete_model_optimizer_same_rank_rng_progress",
        "source_resume_mode": "exact",
        "same_schedule_exact_trajectory": False,
        "source": {
            "runtime_path": declaration["source_runtime_path"],
            "runtime_config_sha256": source_runtime_config_sha256,
            "checkpoint_path": declaration["source_checkpoint_path"],
            "committed_sha256": source_committed_sha256,
            "step": 783,
            "run_lineage": source_config["run"]["lineage"],
            "budget_stop": dict(source_budget_stop),
        },
        "destination": {
            "runtime_config_sha256": canonical_sha256(destination_config),
            "run_lineage": destination_config["run"]["lineage"],
            "output_root": destination_config["run"]["output_root"],
            "start_step": 783, "first_optimizer_update": 784,
            "total_steps": 1566, "maximum_additional_optimizer_steps": 783,
            "sampling_epochs": 54, "additional_epochs": 27,
            "hard_seconds": 7200, "learning_rate": 1e-5, "video_learning_rate": 5e-6,
        },
        "allowed_change_paths": sorted(allowed),
        "changes": changes,
        "unchanged_model_data_objective_environment_topology_optimizer_non_lr": True,
    }


def resolve_native_bounded_continuation(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]] | None:
    """Load the ORIGINAL runtime with the official loader, never rewrite it."""
    from wm3d.training.runtime_contract import (
        RuntimeContractError, load_materialized_runtime, load_yaml, sha256_file,
    )
    from wm3d.training.distributed_checkpoint import DistributedCheckpointManager
    from wm3d.training.pretrain import _resume_expectations

    declaration = config["runtime_profile"]["train"].get("native_bounded_continuation")
    if declaration is None:
        return None
    validate_native_bounded_declaration(declaration)
    source_path = Path(declaration["source_runtime_path"])
    # Reject chains before entering the official loader (including self-cycles).
    original = load_yaml(source_path)
    if original.get("runtime_profile", {}).get("train", {}).get("native_bounded_continuation") is not None:
        raise RuntimeContractError("native bounded continuation source must be an original runtime")
    source_config, source_sha = load_materialized_runtime(source_path)
    source_root = Path(source_config["run"]["output_root"])
    budget_path = source_root / "BUDGET_STOP.json"
    committed_path = Path(declaration["source_checkpoint_path"]) / "COMMITTED.json"
    for path in (budget_path, committed_path):
        if path.is_symlink() or not path.is_file():
            raise RuntimeContractError(f"native bounded continuation source control is not regular: {path}")
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    audit = validate_native_bounded_transition(
        source_config, config, source_runtime_config_sha256=source_sha,
        source_budget_stop=budget, source_committed_sha256=sha256_file(committed_path),
    )
    # Control inspection is CPU-only and precedes both distributed init and
    # official resource/data preflight. Payload integrity remains exclusively
    # in the original manager.load path at the actual authorized restore.
    source_manager = DistributedCheckpointManager(source_root / "checkpoints")
    inspection = source_manager.inspect_committed(
        path=Path(declaration["source_checkpoint_path"]),
        expected=_resume_expectations(
            source_config, source_sha, step=783, world_size=24,
        ),
    )
    if inspection["resume_mode"] != "exact":
        raise RuntimeContractError("native bounded continuation requires exact source topology")
    audit["source"]["checkpoint_inspection"] = inspection
    return source_config, source_sha, audit


def native_bounded_rank_restore_receipt(
    *, source_checkpoint_path: Path, rank: int, model: Any, optimizer: Any,
    metadata: Mapping[str, Any], progress: Mapping[str, Any],
    source_step: int = 783, expected_progress: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure actual restored state after the unmodified complete DCP load.

    Model and moment values are restored/verified by the official manager.
    This small receipt checks their in-memory coverage, actual Adam step
    counters and the saved same-rank RNG, without rehashing any payload.
    """
    import numpy as np
    import random
    import torch
    from wm3d.training.runtime_contract import RuntimeContractError

    expected_progress = ({"next_optimizer_step": source_step}
                         if expected_progress is None else dict(expected_progress))
    if (metadata.get("step") != source_step
            or metadata.get("sampler_progress") != expected_progress
            or dict(progress) != expected_progress):
        raise RuntimeContractError("native bounded continuation restored progress differs")
    rank_path = Path(source_checkpoint_path) / "rank_state" / f"rank_{rank:05d}.pt"
    if rank_path.is_symlink() or not rank_path.is_file():
        raise RuntimeContractError("native bounded continuation saved rank state is not regular")
    saved = torch.load(rank_path, map_location="cpu", weights_only=False)
    if saved.get("progress") != dict(progress) or set(saved["rng"]) != {
        "python", "numpy", "torch_cpu", "torch_cuda",
    }:
        raise RuntimeContractError("native bounded continuation saved rank state differs")
    rng = saved["rng"]
    current_numpy = np.random.get_state()
    saved_numpy = rng["numpy"]
    rng_equal = {
        "python": random.getstate() == rng["python"],
        "numpy": (current_numpy[0] == saved_numpy[0]
                  and np.array_equal(current_numpy[1], saved_numpy[1])
                  and current_numpy[2:] == saved_numpy[2:]),
        "torch_cpu": torch.equal(torch.get_rng_state(), rng["torch_cpu"]),
        "torch_cuda": (rng["torch_cuda"] is not None
                       and torch.cuda.is_initialized()
                       and torch.equal(torch.cuda.get_rng_state(), rng["torch_cuda"])),
    }
    if not all(rng_equal.values()):
        raise RuntimeContractError("native bounded continuation same-rank RNG restore differs")

    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    histogram: dict[str, int] = {}
    moment_elements = 0
    for parameter in parameters:
        state = optimizer.state.get(parameter)
        if not isinstance(state, Mapping) or not {"step", "exp_avg", "exp_avg_sq"} <= set(state):
            raise RuntimeContractError("native bounded continuation Adam state is incomplete")
        for name in ("exp_avg", "exp_avg_sq"):
            moment = state[name]
            if moment.shape != parameter.shape or moment.dtype != torch.float32:
                raise RuntimeContractError("native bounded continuation Adam moment contract differs")
            local = moment.to_local() if hasattr(moment, "to_local") else moment
            moment_elements += local.numel()
        step = state["step"]
        if hasattr(step, "to_local"):
            step = step.to_local()
        step = float(step.item() if isinstance(step, torch.Tensor) else step)
        if not (0 <= step <= source_step and step.is_integer()):
            raise RuntimeContractError("native bounded continuation Adam step counter differs")
        key = str(int(step))
        histogram[key] = histogram.get(key, 0) + 1
    if not parameters or len(optimizer.state) != len(parameters) or str(source_step) not in histogram:
        raise RuntimeContractError("native bounded continuation Adam state coverage differs")
    model_parameters = list(model.parameters())
    return {
        "rank": int(rank), "source_step": source_step, "next_optimizer_step": source_step,
        "first_optimizer_update": source_step + 1,
        "official_complete_dcp_load_returned": True,
        "model_parameter_tensors": len(model_parameters),
        "model_local_parameter_elements": sum(
            (p.to_local() if hasattr(p, "to_local") else p).numel() for p in model_parameters
        ),
        "adam_state_tensors": len(parameters), "adam_moment_tensors": 2 * len(parameters),
        "adam_local_moment_elements": moment_elements,
        "adam_step_histogram": histogram,
        "saved_rank_state_path": str(rank_path),
        "same_rank_rng_equal": {name: bool(equal) for name, equal in rng_equal.items()},
        "sampler_progress_equal": True, "passed": True,
    }


def require_native_continuation_storage(
    *, source_checkpoint_path: Path, output_root: Path, final_save: bool,
) -> None:
    """Check the original snapshot cleanup and the bounded final-save reserve."""
    import shutil
    import stat
    from wm3d.training.runtime_contract import RuntimeContractError

    snapshot = Path(source_checkpoint_path).parent.parent / ".wm3d_checkpoint_load_snapshots"
    info = snapshot.lstat()
    if (not stat.S_ISDIR(info.st_mode) or snapshot.is_symlink()
            or info.st_uid != 0 or info.st_gid != 0
            or stat.S_IMODE(info.st_mode) != 0o700 or any(snapshot.iterdir())):
        raise RuntimeContractError("native bounded continuation source snapshot cleanup differs")
    if final_save:
        # Rank 0 is node43, the destination's actual backing filesystem.
        required_free = 82_000_000_000 + 10 * 2**30 + 100 * 2**30
        if shutil.disk_usage(output_root).free < required_free:
            raise RuntimeContractError("native bounded continuation final-save disk reserve exhausted")
