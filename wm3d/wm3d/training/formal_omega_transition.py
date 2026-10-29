"""Explicit F40000 rollback into the qualified BC-Z/view-loss fixes and Omega.

The source is the original D checkpoint at 40000, not unsaved corrected updates.
Only the authorized frozen feature teacher is added to the already qualified
BC-Z/view-loss corrections; trainable model/AdamW/sampler/LR schemas stay literal.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

F = Path("/data/Minko/wm3d_native_fresh_pretraining_20260930/formal_27source_2ep_20261005")
D = F / "resampling_resume_30000_20261008"
BASE_Q = F / "quality_correction_40000_20261009"
Q = F / "omega_transition_20261010"
STEP = 40000
TOTAL = 167446
SCHEMA = "wm3d_formal40000_omega_transition_v1"
CONTRACT_PATH = F / "training/omega_transition_40000/run_contract.json"
RUNTIME_PATH = Q / "runtime.json"
FIELD = "formal_omega_transition"


def read(path):
    from wm3d.training.checkpoint_stores import read_json
    return read_json(Path(path))


def declaration(config):
    return config["runtime_profile"]["train"].get(FIELD)


def validate_declaration(value: Mapping[str, Any]) -> None:
    from wm3d.training.runtime_contract import RuntimeContractError
    fields = {"schema", "transition_step", "total_steps", "parent_runtime_path",
              "parent_runtime_binding", "parent_run_contract", "correction_specification"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RuntimeContractError("formal quality transition fields differ")
    if (value["schema"] != SCHEMA or type(value["transition_step"]) is not int
            or value["transition_step"] != STEP or type(value["total_steps"]) is not int
            or value["total_steps"] != TOTAL
            or value["parent_runtime_path"] != str(D / "runtime.json")
            or value["parent_run_contract"] != read(F / "training/sampling_transition_30000/run_contract.json")
            or value["correction_specification"] != read(Q / "CORRECTION_SPECIFICATION.json")):
        raise RuntimeContractError("only the declared own-F40000 correction is admitted")
    spec = value["correction_specification"]
    required = {"schema", "state_transform", "repaired_source", "changed_normalization_dimensions",
                "video_loss_reduction", "sampling_plan_changed", "action_transform_changed",
                "parameter_schema_changed", "optimizer_reset", "warmup_reset",
                "CPU_data_qualification", "GPU_objective_qualification",
                "frozen_geometry_teacher", "Omega_GPU_qualification", "qualified_base_runtime"}
    if (set(spec) != required or spec["schema"] != "wm3d_quality40000_correction_spec_v1"
            or spec["state_transform"] != "pos_axis_angle_8d"
            or spec["repaired_source"] != "oxe_bc_z"
            or spec["changed_normalization_dimensions"] != [3, 4, 5, 6, 7, 8]
            or spec["video_loss_reduction"] != "valid_view_mean_v1"
            or any(spec[k] is not False for k in (
                "sampling_plan_changed", "action_transform_changed", "parameter_schema_changed",
                "optimizer_reset", "warmup_reset"))
            or spec["CPU_data_qualification"] != str(BASE_Q / "data_rebuild/COMPLETE.json")
            or spec["GPU_objective_qualification"] != str(BASE_Q / "OBJECTIVE_GPU_QUALIFICATION.json")):
        raise RuntimeContractError("unauthorized correction or reset")
    from wm3d.encoders.native_omega import validate_teacher
    validate_teacher(spec["frozen_geometry_teacher"])
    if (spec["frozen_geometry_teacher"] != read(Q / "TEACHER_SPECIFICATION_PRODUCTION.json")
            or spec["Omega_GPU_qualification"] != str(Q / "OMEGA_GPU_QUALIFICATION.json")
            or spec["qualified_base_runtime"] != str(BASE_Q / "runtime.json")):
        raise RuntimeContractError("Omega teacher/qualified base identity differs")
    omega = read(spec["Omega_GPU_qualification"])
    if (omega.get("passed") is not True or omega.get("full_model_parameter_backward") is not True
            or omega.get("optimizer_updates") != 0 or omega.get("optimizer_created") is not False
            or omega.get("full_teacher_GPU_readback") is not True
            or omega.get("Omega_current_and_future_targets") is not True
            or omega.get("all_master_parameters_exactly_unchanged") is not True):
        raise RuntimeContractError("actual complete-model Omega GPU qualification is required")
    tested_teacher = dict(omega["teacher_specification"])
    candidate_teacher = dict(spec["frozen_geometry_teacher"])
    for key in ("source_root", "checkpoint_path"):
        tested_teacher.pop(key)
        candidate_teacher.pop(key)
    if tested_teacher != candidate_teacher:
        raise RuntimeContractError("GPU-qualified Omega semantics differ from production teacher")
    data, gpu = read(spec["CPU_data_qualification"]), read(spec["GPU_objective_qualification"])
    if (data.get("passed") is not True or data.get("official_train_episode_count") != 14361
            or data.get("other_26_source_normalization_rows_exactly_equal") is not True
            or data.get("unchanged_BCZ_11_rows_exactly_preserved") is not True
            or gpu.get("passed") is not True
            or gpu.get("full_model_parameter_backward") is not True
            or gpu.get("optimizer_updates") != 0):
        raise RuntimeContractError("real data/whole-model backward qualification is incomplete")


def resolve_transition(config):
    value = declaration(config)
    if value is None:
        return None
    from wm3d.training.runtime_contract import load_materialized_runtime, RuntimeContractError
    validate_declaration(value)
    parent, binding = load_materialized_runtime(D / "runtime.json")
    if declaration(parent) is not None or binding != value["parent_runtime_binding"]:
        raise RuntimeContractError("quality transition parent changed or corrections were chained")
    if config["data_closure"] != read(BASE_Q / "corrected_runtime_data/data_closure.json"):
        raise RuntimeContractError("quality transition data is not the qualified BC-Z-only repair")
    normalized = copy.deepcopy(config)
    train = normalized["runtime_profile"]["train"]
    if train.pop("frozen_geometry_teacher") != value["correction_specification"]["frozen_geometry_teacher"]:
        raise RuntimeContractError("Omega actual teacher differs from authorized migration")
    if train.pop(FIELD) != value:
        raise RuntimeContractError("quality declaration changed")
    if normalized["model_profile"]["model"].pop("video_loss_reduction") != "valid_view_mean_v1":
        raise RuntimeContractError("unexpected corrected video objective")
    # Only this loss-only option is new. No architecture/action-conditioning flag.
    if normalized["model_profile"] != parent["model_profile"]:
        raise RuntimeContractError("quality correction changed parameter/model architecture")
    plan = normalized["runtime_profile"]["checkpoint"]["storage_plan"]
    if plan["formal_runtime_path"] != str(RUNTIME_PATH):
        raise RuntimeContractError("quality runtime escaped its independent directory")
    plan["formal_runtime_path"] = str(D / "runtime.json")
    normalized["data_closure"] = copy.deepcopy(parent["data_closure"])
    normalized["run"]["code_commit"] = parent["run"]["code_commit"]
    # Original validation already recomputes these small-runtime ABI identities.
    normalized["bindings"] = copy.deepcopy(parent["bindings"])
    if normalized != parent:
        raise RuntimeContractError("quality correction changed more than its explicit data/loss/code delta")
    return parent, binding, dict(value)


def require_resume_step(step):
    if type(step) is not int or not STEP <= step <= TOTAL:
        raise ValueError("quality correction can resume only own F40000 or later corrected checkpoint")


def require_initial_boundary(router, path):
    path = Path(path)
    rows = router.records()
    if [(r["run_lineage"], r["step"]) for r in rows] != [
            (router.formal_lineage, s) for s in (20000, 30000, 40000)]:
        raise ValueError("F40000 full readback and approved F10000 retention must precede restore")
    if path != router.latest_formal_committed() or path.name != "step_00040000":
        raise ValueError("initial correction must consume own latest complete F40000")
    authorization = read(Q / "USER_AUTHORIZATION.json")
    stopped = read(Q / "STOPPED_CORRECTED_PARENT.json")
    if (authorization.get("rollback_to_step") != STEP
            or authorization.get("discard_post40000_updates_explicitly_accepted") is not True
            or stopped.get("passed") is not True or stopped.get("resume_saved_step") != STEP
            or stopped.get("all_parent_units_inactive_and_empty") is not True):
        raise ValueError("accepted rollback and actual stopped corrected parents are required")
    pause = read(F / "checkpoint_pause_40000_20261009/STATUS.json")
    if pause.get("phase") != "paused_after_own_F40000_committed":
        raise ValueError("bound F40000 pause has not completed")
    row = next(r for r in rows if r["path"] == str(path))
    matches = []
    for attempt in sorted((F / "evaluation/step_00040000").glob("attempt_*")):
        full_path, strict_path = attempt / "FULL_MODEL_READBACK.json", attempt / "STRICT_MODEL_LOAD.json"
        if not full_path.exists() or not strict_path.exists():
            continue
        full, strict = read(full_path), read(strict_path)
        identity = {"run_lineage": router.formal_lineage, "source_step": STEP,
                    "source_checkpoint": str(path),
                    "original_COMMITTED": row["records"]["COMMITTED.json"],
                    "commit_file_identity": row["files"]["COMMITTED.json"]}
        if any(any(receipt.get(k) != v for k, v in identity.items()) for receipt in (full, strict)):
            continue
        if (full.get("passed") is True
                and full.get("source_file_signatures") == {k: v[:3] for k, v in row["files"].items()}
                and strict.get("passed") is True
                and strict.get("all_model_tensors_exactly_equal") is True
                and strict.get("tensors") == 2354 and strict.get("elements") == 7251084294):
            matches.append(str(full_path))
    if not matches:
        raise ValueError("own F40000 actual complete CPU/GPU model readback is absent")
    return {"pause": pause, "model_readback": matches[-1], "source_files": row["files"]}


def validate_checkpoint_provenance(cfg, contract, metadata, step, expected_progress):
    """Keep old and corrected checkpoints separately and literally bound."""
    value = declaration(cfg)
    fields = (FIELD, "formal_omega_restore", "initial_formal_omega_restore")
    if value is None:
        if any(k in metadata for k in fields):
            raise RuntimeError("parent checkpoint cannot silently contain quality-corrected semantics")
        return
    validate_declaration(value)
    initial = metadata.get("initial_formal_omega_restore", {})
    restored = metadata.get("formal_omega_restore", {})
    if (step <= STEP or contract.get(FIELD) != value or metadata.get(FIELD) != value
            or initial.get("passed") is not True or initial.get("restored_step") != STEP
            or initial.get(FIELD) != value
            or initial.get("source_checkpoint", {}).get("step") != STEP
            or initial.get("model_optimizer_same_rank_rng_restored") is not True
            or restored.get("passed") is not True or restored.get(FIELD) != value):
        raise RuntimeError("quality correction/restoration provenance differs")
    expected_progress[FIELD] = value
