"""One explicit same-run F30000 sampling migration, never a fresh restart.

The parent runtime and run contract stay immutable. Model/AdamW/RNG loading
continues through the original complete DCP manager and 3+1 storage lease.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from wm3d.data.reweighted_continuation import MULTIPLIERS, TRANSITION_STEP, TOTAL_STEPS

F = Path("/data/Minko/wm3d_native_fresh_pretraining_20260930/formal_27source_2ep_20261005")
D = F / "resampling_resume_30000_20261008"
MODE = "formal30000_reweighted_continuation_v1"
SCHEMA = "wm3d_formal30000_sampling_transition_v1"
CONTRACT_PATH = F / "training/sampling_transition_30000/run_contract.json"
RUNTIME_PATH = D / "runtime.json"


def read(path):
    from wm3d.training.checkpoint_stores import read_json
    return read_json(Path(path))


def validate_declaration(value: Mapping[str, Any]) -> None:
    from wm3d.training.runtime_contract import RuntimeContractError
    required = {"schema", "transition_step", "total_steps", "parent_runtime_path",
                "parent_runtime_binding", "parent_run_contract", "sampler_migration"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise RuntimeContractError("formal sampling transition fields differ")
    if (value["schema"] != SCHEMA or type(value["transition_step"]) is not int
            or value["transition_step"] != TRANSITION_STEP
            or type(value["total_steps"]) is not int or value["total_steps"] != TOTAL_STEPS
            or value["parent_runtime_path"] != str(F / "runtime.json")):
        raise RuntimeContractError("only the authorized own F30000 transition is admitted")
    qualification = read(D / "REAL_SAMPLING_CPU_20261008_1452.json")
    if (qualification.get("passed") is not True
            or qualification.get("GPU_context") is not False
            or qualification.get("all_remaining_source_decisions_checked") != 137446
            or value["sampler_migration"] != qualification["migration"]
            or value["parent_runtime_binding"] != qualification["source_runtime_binding"]
            or value["parent_run_contract"] != read(F / "training/run_contract.json")):
        raise RuntimeContractError("transition differs from the real qualified full27 plan/parent")
    migration = value["sampler_migration"]
    if (migration["source_weight_multipliers"] != MULTIPLIERS
            or migration["global_optimizer_budget"] != TOTAL_STEPS
            or migration["transition_step"] != TRANSITION_STEP
            or migration["source_window_cursor_preserved"] is not True
            or migration["all_sources_complete_two_epochs_claimed"] is not False):
        raise RuntimeContractError("unauthorized mixture or reset of source cursors")


def resolve_transition(config):
    """Strictly allow only code identity, explicit sampler and runtime-path delta."""
    from wm3d.training import formal_omega_transition as quality
    corrected = quality.resolve_transition(config)
    if corrected is not None:
        return resolve_transition(corrected[0])
    declaration = config["runtime_profile"]["train"].get("formal_sampling_transition")
    if declaration is None:
        return None
    from wm3d.training.runtime_contract import load_materialized_runtime, RuntimeContractError
    validate_declaration(declaration)
    parent, parent_binding = load_materialized_runtime(F / "runtime.json")
    if parent["runtime_profile"]["train"].get("formal_sampling_transition") is not None:
        raise RuntimeContractError("sampling transitions cannot be chained")
    if parent_binding != declaration["parent_runtime_binding"]:
        raise RuntimeContractError("original formal runtime changed")
    normalized = copy.deepcopy(config)
    target_train = normalized["runtime_profile"]["train"]
    if (target_train.pop("formal_sampling_transition") != declaration
            or target_train.pop("sampling_mode") != MODE
            or "sampling_epochs" in target_train):
        raise RuntimeContractError("new distribution must not claim complete epochs")
    target_train["sampling_mode"] = "source_count_epochs_v1"
    target_train["sampling_epochs"] = 2
    plan = normalized["runtime_profile"]["checkpoint"]["storage_plan"]
    if plan["formal_runtime_path"] != str(RUNTIME_PATH):
        raise RuntimeContractError("transition runtime escaped its independent directory")
    plan["formal_runtime_path"] = str(F / "runtime.json")
    normalized["run"]["code_commit"] = parent["run"]["code_commit"]
    normalized["bindings"]["runtime_profile_sha256"] = parent["bindings"]["runtime_profile_sha256"]
    if normalized != parent:
        raise RuntimeContractError("transition changes more than the approved sampler/code/runtime path")
    if (config["run"]["output_root"] != str(F / "training")
            or config["run"]["lineage"] != "native_fresh_wan_full27source_2ep_20261005"):
        raise RuntimeContractError("transition is not the original formal lineage")
    return parent, parent_binding, dict(declaration)


def contract_path(config):
    from wm3d.training import formal_omega_transition as quality
    if quality.declaration(config) is not None:
        return quality.CONTRACT_PATH
    return (CONTRACT_PATH if config["runtime_profile"]["train"].get("formal_sampling_transition")
            is not None else Path(config["run"]["output_root"]) / "run_contract.json")


def progress_for(config, step):
    result = {"next_optimizer_step": int(step)}
    declaration = config["runtime_profile"]["train"].get("formal_sampling_transition")
    if declaration is not None:
        if not TRANSITION_STEP <= step <= TOTAL_STEPS:
            raise ValueError("sampler progress outside authorized continuation budget")
        result["formal_sampling_transition"] = declaration
    from wm3d.training import formal_omega_transition as quality
    correction = quality.declaration(config)
    if correction is not None:
        quality.require_resume_step(int(step))
        result[quality.FIELD] = correction
    return result


def require_resume_step(step):
    if type(step) is not int or not TRANSITION_STEP <= step <= TOTAL_STEPS:
        raise ValueError("only own F30000 or a later own migrated checkpoint can resume")


def require_initial_boundary(router, path):
    """Actual original stop/readback/rotation evidence before any GPU allocation."""
    path = Path(path)
    rows = router.records()
    if ([(r["run_lineage"], r["step"]) for r in rows] !=
            [(router.formal_lineage, s) for s in (10000, 20000, 30000)]):
        raise ValueError("F30000 full-model readback and original T rotation must precede restore")
    latest = router.latest_formal_committed()
    if path != latest or path.name != "step_00030000":
        raise ValueError("initial migration must consume own latest complete F30000")
    status = read(F / "checkpoint_pause_30000_20261008/STATUS.json")
    if status.get("phase") != "paused_after_own_F30000_committed":
        raise ValueError("original bound F30000 pause controller has not completed")
    row = next(r for r in rows if r["path"] == str(path))
    matches = []
    for attempt in sorted((F / "evaluation/step_00030000").glob("attempt_*")):
        full_path, strict_path = attempt / "FULL_MODEL_READBACK.json", attempt / "STRICT_MODEL_LOAD.json"
        if not full_path.exists() or not strict_path.exists():
            continue
        full, strict = read(full_path), read(strict_path)
        identity = {"run_lineage": router.formal_lineage, "source_step": 30000,
                    "source_checkpoint": str(path),
                    "original_COMMITTED": row["records"]["COMMITTED.json"],
                    "commit_file_identity": row["files"]["COMMITTED.json"]}
        if any(any(receipt.get(k) != v for k, v in identity.items())
               for receipt in (full, strict)):
            continue
        if (full.get("passed") is True
                and full.get("source_file_signatures") == {k: v[:3] for k, v in row["files"].items()}
                and strict.get("passed") is True
                and strict.get("all_model_tensors_exactly_equal") is True
                and strict.get("tensors") == 2354 and strict.get("elements") == 7251084294):
            matches.append(str(full_path))
    if not matches:
        raise ValueError("own F30000 actual CPU/GPU full-model readback is absent")
    return {"pause": status, "model_readback": matches[-1], "source_files": row["files"]}



def settled_storage_after_restore(router, *, timeout_seconds=60.0):
    """Wait only for freed snapshot blocks; re-run every original storage gate."""
    import os
    import time
    started = time.monotonic()
    observations = []
    while True:
        try:
            result = router.preflight()
            return {**result, "settle_wait_seconds": time.monotonic() - started,
                    "free_space_observations": observations}
        except RuntimeError as exc:
            if str(exc) != "no empty fourth slot satisfying actual floor+reserve+CP budget":
                raise
            observations.append({
                "elapsed_seconds": time.monotonic() - started,
                "stores": [{"root": store["root"],
                            "free_bytes": (lambda v: v.f_bavail * v.f_frsize)(
                                os.statvfs(store["root"]))}
                           for store in router.stores],
            })
            if time.monotonic() - started >= timeout_seconds:
                raise RuntimeError(
                    f"post-restore storage did not settle under original budget: {observations}"
                ) from exc
            time.sleep(1.0)


def publish_restore(*, config, context, model, optimizer, metadata, progress,
                    source_checkpoint, router, launch_qualification_path):
    """Audit actual restored same-rank RNG/Adam state before sampler construction."""
    import time
    import torch.distributed as dist
    from wm3d.training.native_bounded_continuation import native_bounded_rank_restore_receipt
    from wm3d.training.pretrain import _atomic_json_no_clobber, _learning_rate
    step = int(metadata["step"])
    require_resume_step(step)
    from wm3d.training import formal_omega_transition as quality
    correction = quality.resolve_transition(config)
    if correction is not None:
        quality.require_resume_step(step)
    source_config = correction[0] if correction is not None and step == quality.STEP else config
    expected_progress = ({"next_optimizer_step": step} if step == TRANSITION_STEP
                         else progress_for(source_config, step))
    local = None
    error = None
    try:
        local = native_bounded_rank_restore_receipt(
            source_checkpoint_path=Path(source_checkpoint["path"]), rank=context.rank,
            model=model, optimizer=optimizer, metadata=metadata, progress=progress,
            source_step=step, expected_progress=expected_progress)
        # The original loop saves the LR used for the last completed update.
        expected_lr = _learning_rate(step - 1, config["runtime_profile"])
        rates = []
        for group in optimizer.param_groups:
            factor = float(group.get("lr_scale", 1.0))
            if float(group["lr"]) != expected_lr * factor:
                raise ValueError("restored AdamW LR is not the last completed global-step LR")
            rates.append({"lr": float(group["lr"]), "lr_scale": factor})
        local["optimizer_group_lrs"] = rates
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    gathered = [None] * context.world_size
    dist.all_gather_object(gathered, {"receipt": local, "error": error})
    if any(item["error"] for item in gathered):
        raise ValueError(f"formal full-state restore audit failed: {gathered}")
    ranks = [item["receipt"] for item in gathered]
    if context.world_size != 24 or [r["rank"] for r in ranks] != list(range(24)):
        raise ValueError("full-state restore lacks the same 24 ranks")
    path = contract_path(config).parent / "restores" / (Path(launch_qualification_path).stem + ".json")
    verdict = [None]
    if context.is_rank0:
        try:
            # Original router rechecks all snapshots empty and real fourth-slot reserve.
            storage = settled_storage_after_restore(router)
            receipt = {"schema": "wm3d_formal_sampling_complete_state_restore_v1",
                       "passed": True, "created_unix_ns": time.time_ns(),
                       "source_checkpoint": dict(source_checkpoint), "restored_step": step,
                       "first_optimizer_update": step + 1,
                       "formal_sampling_transition": config["runtime_profile"]["train"]["formal_sampling_transition"],
                       "launch_qualification_path": launch_qualification_path,
                       "original_sampler_progress": dict(progress),
                       "next_sampler_progress": progress_for(config, step),
                       "next_global_step_lr": _learning_rate(step, config["runtime_profile"]),
                       "model_optimizer_same_rank_rng_restored": True,
                       "optimizer_reset": False, "warmup_reset": False,
                       **({quality.FIELD: correction[2],
                           "quality_contract_changed": step == quality.STEP,
                           "parameter_schema_changed": False,
                           "training_objective_unchanged_claimed": step != quality.STEP}
                          if correction is not None else {}),
                       "sampling_distribution_changed": step == TRANSITION_STEP,
                       "same_sampling_trajectory_claimed": step != TRANSITION_STEP,
                       "storage_after_snapshot_cleanup": storage, "ranks": ranks}
            _atomic_json_no_clobber(path, receipt)
            verdict[0] = {"ok": True, "path": str(path), "receipt": receipt}
        except Exception as exc:
            verdict[0] = {"ok": False, "error": str(exc)}
    dist.broadcast_object_list(verdict, src=0)
    if verdict[0]["ok"] is not True:
        raise ValueError(f"formal restore receipt publication failed: {verdict[0]}")
    return verdict[0]["path"], verdict[0]["receipt"]
