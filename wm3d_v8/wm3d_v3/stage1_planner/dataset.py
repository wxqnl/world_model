"""Fail-closed loader for real unified WM3D V8 Stage1 branches."""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from wm3d_v3.data.manifest_contract import SHA256_RE
from wm3d_v3.stage1_planner.rollout_audit import read_regular_bytes


BRANCH_SCHEMA = "wm3d_v8_unified_stage1_branch_v3"
BRANCH_INDEX_SCHEMA = "wm3d_v8_unified_stage1_branch_index_v3"
BRANCH_SEAL_SCHEMA = "wm3d_v8_unified_stage1_branch_seal_v3"
GENERATOR_RECEIPT_SCHEMA = "wm3d_v8_unified_stage1_candidate_generator_receipt_v2"
DATASET_SCHEMA = "wm3d_v8_unified_stage1_dataset_v3"
GENERATOR_RECEIPT_FIELDS = {
    "schema", "sample_index", "sample_id", "source", "split", "embodiment",
    "payload_sha256", "runtime_config_sha256", "data_profile_sha256",
    "model_profile_sha256", "window_index_sha256",
    "grouped_normalization_sha256", "task_bank_index_sha256",
    "encoder_contract_sha256", "task_encoder_contract_sha256",
    "representation_contract_sha256", "stage0_checkpoint_commit_sha256",
    "rollout_audit_sha256", "source_manifest_sha256",
    "adapter_contract_sha256", "simulator_revision", "simulator_seed",
    "real_simulator_outcomes", "future_observation_leakage",
    "candidate_action_abi", "candidate_actions_from_adapter",
    "candidate_actions_grouped_normalized",
    "native_evidence_from_frozen_encoder",
}


class Stage1BranchError(ValueError):
    pass


def validate_rollout_audit_binding(
    row: dict[str, Any], receipt: dict[str, Any]
) -> str:
    if "rollout_audit_sha256" not in row or "rollout_audit_sha256" not in receipt:
        raise Stage1BranchError("rollout-audit binding is missing")
    observed = _sha(row["rollout_audit_sha256"], "rollout audit SHA")
    if receipt["rollout_audit_sha256"] != observed:
        raise Stage1BranchError("rollout-audit binding mismatch")
    return observed


def _sha(value: object, label: str) -> str:
    text = str(value)
    if SHA256_RE.fullmatch(text) is None:
        raise Stage1BranchError(f"{label} must be lowercase SHA256")
    return text


def _regular_sha(path: Path, expected: str, label: str) -> tuple[Path, bytes]:
    try:
        path, payload, observed = read_regular_bytes(path, label)
    except (OSError, ValueError) as error:
        raise Stage1BranchError(str(error)) from error
    if observed != _sha(expected, f"{label} SHA"):
        raise Stage1BranchError(f"{label} SHA mismatch: {observed} != {expected}")
    return path, payload


@dataclass(frozen=True)
class Stage1BranchDatasetConfig:
    branch_index: Path
    branch_index_sha256: str
    branch_seal: Path
    branch_seal_sha256: str
    runtime_config_sha256: str
    data_profile_sha256: str
    model_profile_sha256: str
    window_index_sha256: str
    grouped_normalization_sha256: str
    task_bank_index_sha256: str
    encoder_contract_sha256: str
    task_encoder_contract_sha256: str
    representation_contract_sha256: str
    stage0_checkpoint_commit_sha256: str
    rollout_audit_sha256: str
    split: str


_BRANCH_KEYS = {
    "candidate_fine_action_values", "candidate_fine_action_mask",
    "candidate_fine_action_dt", "candidate_fine_sample_mask",
    "candidate_coarse_action_values", "candidate_coarse_action_mask",
    "branch_future_tokens", "branch_future_dt_s", "branch_token_mask",
    "branch_depth", "branch_depth_mask", "branch_point", "branch_point_mask",
    "branch_camera_pose", "branch_camera_pose_mask",
    "branch_geometry_confidence", "branch_view_mask",
    "branch_rewards", "branch_dones", "branch_success", "branch_valid",
}


def _validate_candidate_action_shapes(
    payload: dict[str, torch.Tensor],
    *,
    candidates: int,
    K: int,
    model: dict[str, Any],
) -> None:
    fine = payload["candidate_fine_action_values"]
    if fine.ndim != 5 or fine.shape[:2] != (candidates, K):
        raise Stage1BranchError("candidate fine actions do not cover sealed K")
    if tuple(fine.shape[2:]) != (
        int(model["max_action_groups"]),
        int(model["max_action_substeps"]),
        int(model["max_action_dim"]),
    ):
        raise Stage1BranchError("candidate fine action capacities differ from Stage0")
    if payload["candidate_fine_action_mask"].shape != fine.shape:
        raise Stage1BranchError("candidate fine action mask mismatch")
    expected_sample_shape = fine.shape[:-1]
    if (
        payload["candidate_fine_action_dt"].shape != expected_sample_shape
        or payload["candidate_fine_sample_mask"].shape != expected_sample_shape
    ):
        raise Stage1BranchError("candidate fine action timestamps/mask mismatch")
    coarse = payload["candidate_coarse_action_values"]
    expected_coarse_shape = (
        candidates,
        K,
        int(model["max_action_groups"]),
        int(model["max_action_dim"]),
    )
    if tuple(coarse.shape) != expected_coarse_shape:
        raise Stage1BranchError("candidate coarse action capacities differ from Stage0")
    if payload["candidate_coarse_action_mask"].shape != coarse.shape:
        raise Stage1BranchError("candidate coarse action mask mismatch")


class Stage1BranchDataset(Dataset[dict[str, Any]]):
    """Join one sealed Stage0 window with real candidate branch evidence."""

    def __init__(self, cfg: Stage1BranchDatasetConfig, stage0_dataset: Dataset):
        self.cfg = cfg
        if cfg.split not in {"train", "val", "test"}:
            raise Stage1BranchError("Stage1 split is invalid")
        index_path, index_payload = _regular_sha(
            cfg.branch_index, cfg.branch_index_sha256, "branch index"
        )
        _seal_path, seal_payload = _regular_sha(
            cfg.branch_seal, cfg.branch_seal_sha256, "branch seal"
        )
        seal = json.loads(seal_payload)
        required_seal = {
            "schema", "branch_index_path", "branch_index_sha256", "row_count",
            "row_count_by_split", "candidate_count", "horizon",
            "runtime_config_sha256", "data_profile_sha256",
            "model_profile_sha256", "window_index_sha256",
            "grouped_normalization_sha256", "task_bank_index_sha256",
            "encoder_contract_sha256", "task_encoder_contract_sha256",
            "representation_contract_sha256", "stage0_checkpoint_commit_sha256",
            "rollout_audit_sha256",
        }
        if not isinstance(seal, dict) or set(seal) != required_seal:
            raise Stage1BranchError("branch seal fields mismatch")
        if seal["schema"] != BRANCH_SEAL_SCHEMA:
            raise Stage1BranchError("branch seal schema mismatch")
        bindings = {
            "branch_index_sha256": cfg.branch_index_sha256,
            "runtime_config_sha256": cfg.runtime_config_sha256,
            "data_profile_sha256": cfg.data_profile_sha256,
            "model_profile_sha256": cfg.model_profile_sha256,
            "window_index_sha256": cfg.window_index_sha256,
            "grouped_normalization_sha256": cfg.grouped_normalization_sha256,
            "task_bank_index_sha256": cfg.task_bank_index_sha256,
            "encoder_contract_sha256": cfg.encoder_contract_sha256,
            "task_encoder_contract_sha256": cfg.task_encoder_contract_sha256,
            "representation_contract_sha256": cfg.representation_contract_sha256,
            "stage0_checkpoint_commit_sha256": cfg.stage0_checkpoint_commit_sha256,
            "rollout_audit_sha256": cfg.rollout_audit_sha256,
        }
        for name, expected in bindings.items():
            if seal.get(name) != _sha(expected, name):
                raise Stage1BranchError(f"branch seal {name} mismatch")
        counts = seal["row_count_by_split"]
        if (
            not isinstance(counts, dict)
            or set(counts) != {"train", "val", "test"}
            or any(int(value) < 0 for value in counts.values())
            or sum(int(value) for value in counts.values()) != int(seal["row_count"])
        ):
            raise Stage1BranchError("branch seal split counts are invalid")
        if int(seal["candidate_count"]) < 2 or int(seal["horizon"]) <= 0:
            raise Stage1BranchError("branch seal candidate/horizon contract is invalid")

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        required_row = {
            "schema", "branch_id", "sample_index", "sample_id", "source", "split",
            "embodiment", "path", "payload_sha256", "candidates", "horizon",
            "K", "P", "token_dim", "num_views", "runtime_config_sha256",
            "data_profile_sha256", "model_profile_sha256", "window_index_sha256",
            "grouped_normalization_sha256", "task_bank_index_sha256",
            "encoder_contract_sha256", "task_encoder_contract_sha256",
            "representation_contract_sha256", "stage0_checkpoint_commit_sha256",
            "rollout_audit_sha256",
            "source_manifest_sha256", "adapter_contract_sha256",
            "generator_receipt_path", "generator_receipt_sha256",
            "real_simulator_outcomes", "future_observation_leakage",
            "candidate_action_abi",
        }
        for line_number, line in enumerate(index_payload.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != required_row or row.get("schema") != BRANCH_INDEX_SCHEMA:
                raise Stage1BranchError(f"branch index row {line_number} fields/schema mismatch")
            branch_id = _sha(row["branch_id"], "branch_id")
            if branch_id in seen:
                raise Stage1BranchError("duplicate branch_id")
            seen.add(branch_id)
            if row["split"] != cfg.split:
                continue
            if row["real_simulator_outcomes"] is not True or row["future_observation_leakage"] is not False:
                raise Stage1BranchError("Stage1 requires causal real-simulator outcomes")
            if row["candidate_action_abi"] != "wm3d_v8_grouped_robot_v1":
                raise Stage1BranchError("candidate action ABI is not unified grouped robot")
            for name, expected in bindings.items():
                if name != "branch_index_sha256" and row.get(name) != expected:
                    raise Stage1BranchError(f"branch row {name} mismatch")
            sample_index = int(row["sample_index"])
            if not 0 <= sample_index < len(stage0_dataset):
                raise Stage1BranchError("branch sample_index lies outside Stage0 dataset")
            stage0_entry = stage0_dataset.entries[sample_index]
            identity = {
                "sample_id": stage0_entry.sample_id,
                "source": stage0_entry.source,
                "split": stage0_entry.split,
                "embodiment": stage0_entry.embodiment,
            }
            if any(str(row[name]) != str(value) for name, value in identity.items()):
                raise Stage1BranchError("branch row does not identify its Stage0 window")
            source_spec = next(
                source for source in stage0_dataset.data_profile.sources
                if source.name == stage0_entry.source
            )
            if row["source_manifest_sha256"] != source_spec.manifest_sha256:
                raise Stage1BranchError("branch source-manifest binding mismatch")
            if row["adapter_contract_sha256"] != source_spec.adapter_contract_sha256:
                raise Stage1BranchError("branch adapter binding mismatch")
            receipt, receipt_payload = _regular_sha(
                Path(row["generator_receipt_path"]),
                row["generator_receipt_sha256"],
                "candidate generator receipt",
            )
            receipt_value = json.loads(receipt_payload)
            if (
                not isinstance(receipt_value, dict)
                or set(receipt_value) != GENERATOR_RECEIPT_FIELDS
                or receipt_value.get("schema") != GENERATOR_RECEIPT_SCHEMA
            ):
                raise Stage1BranchError("candidate generator receipt schema mismatch")
            validate_rollout_audit_binding(row, receipt_value)
            receipt_bindings = {
                **{name: row[name] for name in (
                    "sample_index", "sample_id", "source", "split", "embodiment",
                    "payload_sha256", "source_manifest_sha256", "adapter_contract_sha256",
                )},
                **{name: expected for name, expected in bindings.items() if name != "branch_index_sha256"},
            }
            if any(receipt_value.get(name) != value for name, value in receipt_bindings.items()):
                raise Stage1BranchError("candidate generator receipt identity/lineage mismatch")
            receipt_gates = (
                receipt_value.get("real_simulator_outcomes"),
                receipt_value.get("candidate_actions_from_adapter"),
                receipt_value.get("candidate_actions_grouped_normalized"),
                receipt_value.get("native_evidence_from_frozen_encoder"),
            )
            if any(value is not True for value in receipt_gates) or receipt_value.get("future_observation_leakage") is not False:
                raise Stage1BranchError("candidate generator receipt gates did not pass")
            payload_path, payload = _regular_sha(
                Path(row["path"]), row["payload_sha256"], "branch payload"
            )
            row["path"] = str(payload_path)
            rows.append(row)
        if len(rows) != int(counts[cfg.split]):
            raise Stage1BranchError("branch seal row count differs from selected split")
        if not rows:
            raise Stage1BranchError(f"Stage1 split {cfg.split!r} is empty")
        if any(
            int(row["candidates"]) != int(seal["candidate_count"])
            or int(row["horizon"]) != int(seal["horizon"])
            for row in rows
        ):
            raise Stage1BranchError("branch rows differ from sealed candidate/horizon contract")
        self.rows = tuple(rows)
        self.stage0_dataset = stage0_dataset

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sample = dict(self.stage0_dataset[int(row["sample_index"])])
        payload_path, payload_bytes = _regular_sha(
            Path(row["path"]), row["payload_sha256"], "branch payload"
        )
        if str(payload_path) != row["path"]:
            raise Stage1BranchError("branch payload resolved path changed")
        payload = torch.load(
            io.BytesIO(payload_bytes), map_location="cpu", weights_only=True
        )
        if not isinstance(payload, dict) or set(payload) != _BRANCH_KEYS:
            raise Stage1BranchError("branch payload tensor fields mismatch")
        if any(not isinstance(value, torch.Tensor) for value in payload.values()):
            raise Stage1BranchError("branch payload contains non-tensor values")
        candidates, horizon = payload["branch_rewards"].shape
        expected = (int(row["candidates"]), int(row["horizon"]))
        if (candidates, horizon) != expected or not 0 < horizon <= int(row["K"]):
            raise Stage1BranchError("branch candidate/horizon contract mismatch")
        token_shape = (candidates, horizon, int(row["P"]), int(row["token_dim"]))
        if payload["branch_future_tokens"].shape != token_shape:
            raise Stage1BranchError("branch tokens do not match unified model profile")
        evidence_float = (
            "branch_future_tokens", "branch_future_dt_s", "branch_depth",
            "branch_point", "branch_camera_pose", "branch_geometry_confidence",
            "candidate_fine_action_values", "candidate_fine_action_dt",
            "candidate_coarse_action_values", "branch_rewards",
        )
        if any(not payload[name].is_floating_point() for name in evidence_float):
            raise Stage1BranchError("branch evidence/action/reward tensors must be floating point")
        if payload["branch_future_dt_s"].shape != (candidates, horizon):
            raise Stage1BranchError("branch future timestamps do not match H")
        if (
            not bool(torch.isfinite(payload["branch_future_dt_s"]).all())
            or bool((payload["branch_future_dt_s"] <= 0).any())
            or (horizon > 1 and not bool(torch.diff(payload["branch_future_dt_s"], dim=-1).gt(0).all()))
        ):
            raise Stage1BranchError("branch future timestamps are not finite/increasing")
        stage0_future_dt = (
            sample["world_times_s"][self.stage0_dataset.T : self.stage0_dataset.T + horizon]
            - sample["world_times_s"][self.stage0_dataset.T - 1]
        )
        if not torch.equal(
            payload["branch_future_dt_s"].to(dtype=stage0_future_dt.dtype),
            stage0_future_dt[None].expand(candidates, -1),
        ):
            raise Stage1BranchError("branch future timestamps differ from Stage0 window")
        token_mask_shape = token_shape[:-1]
        if payload["branch_token_mask"].shape != token_mask_shape or payload["branch_token_mask"].dtype != torch.bool:
            raise Stage1BranchError("branch native token mask mismatch")
        if not bool(payload["branch_token_mask"].any(dim=-1).all()):
            raise Stage1BranchError("branch contains a future frame without token evidence")
        view_prefix = (candidates, horizon, int(row["num_views"]), int(row["P"]))
        if payload["branch_depth"].shape != view_prefix or payload["branch_point"].shape != (*view_prefix, 3):
            raise Stage1BranchError("branch geometry does not match unified VGGT layout")
        if payload["branch_depth_mask"].shape != view_prefix or payload["branch_depth_mask"].dtype != torch.bool:
            raise Stage1BranchError("branch depth mask mismatch")
        if payload["branch_point_mask"].shape != view_prefix or payload["branch_point_mask"].dtype != torch.bool:
            raise Stage1BranchError("branch point mask mismatch")
        if payload["branch_camera_pose"].shape != (*view_prefix[:3], 9):
            raise Stage1BranchError("branch camera pose shape mismatch")
        if payload["branch_camera_pose_mask"].shape != view_prefix[:3] or payload["branch_camera_pose_mask"].dtype != torch.bool:
            raise Stage1BranchError("branch camera pose mask mismatch")
        if payload["branch_geometry_confidence"].shape != view_prefix:
            raise Stage1BranchError("branch confidence shape mismatch")
        if payload["branch_view_mask"].shape != view_prefix[:3] or payload["branch_view_mask"].dtype != torch.bool:
            raise Stage1BranchError("branch view mask mismatch")
        visible = payload["branch_view_mask"][..., None]
        if bool((payload["branch_depth_mask"] & ~visible).any()) or bool((payload["branch_point_mask"] & ~visible).any()):
            raise Stage1BranchError("branch geometry masks exist outside measured views")
        if bool((payload["branch_camera_pose_mask"] & ~payload["branch_view_mask"]).any()):
            raise Stage1BranchError("branch camera poses exist outside measured views")
        if bool((payload["branch_geometry_confidence"] < 0).any()):
            raise Stage1BranchError("branch geometry confidence must be non-negative")
        model = self.stage0_dataset.model
        _validate_candidate_action_shapes(
            payload,
            candidates=candidates,
            K=int(row["K"]),
            model=model,
        )
        if payload["candidate_fine_action_mask"].dtype != torch.bool or payload["candidate_fine_sample_mask"].dtype != torch.bool:
            raise Stage1BranchError("candidate fine action masks must be boolean")
        if bool((payload["candidate_fine_action_mask"].any(dim=-1) & ~payload["candidate_fine_sample_mask"]).any()):
            raise Stage1BranchError("candidate fine dimensions exist outside real samples")
        valid_dt = payload["candidate_fine_action_dt"][payload["candidate_fine_sample_mask"]]
        if not bool(torch.isfinite(valid_dt).all()) or bool((valid_dt < 0).any()):
            raise Stage1BranchError("candidate fine action timestamps are invalid")
        future_boundaries = sample["future_world_boundaries_dt"]
        if future_boundaries.shape != (int(row["K"]) + 1,):
            raise Stage1BranchError("Stage0 future boundary clock is invalid")
        interval_dt = torch.diff(future_boundaries)
        upper = interval_dt[None, :, None, None].expand_as(
            payload["candidate_fine_action_dt"]
        )
        if bool(
            (
                payload["candidate_fine_sample_mask"]
                & (payload["candidate_fine_action_dt"] >= upper)
            ).any()
        ):
            raise Stage1BranchError("candidate fine command lies outside its world interval")
        if (
            payload["candidate_coarse_action_mask"].dtype != torch.bool
        ):
            raise Stage1BranchError("candidate coarse action ABI mismatch")
        if not bool(
            payload["candidate_fine_sample_mask"].any()
            or payload["candidate_coarse_action_mask"].any()
        ):
            raise Stage1BranchError("candidate branches contain no measured action")
        if payload["branch_valid"].shape != (candidates,) or not bool(payload["branch_valid"].all()):
            raise Stage1BranchError("branch validity is incomplete")
        outcomes = (payload["branch_rewards"], payload["branch_dones"], payload["branch_success"])
        if any(value.shape != (candidates, horizon) for value in outcomes):
            raise Stage1BranchError("branch outcome trajectories are not aligned")
        if any(value.dtype != torch.bool for value in (
            payload["branch_dones"], payload["branch_success"], payload["branch_valid"]
        )):
            raise Stage1BranchError("branch done/success/valid tensors must be boolean")
        if not bool(torch.isfinite(payload["branch_rewards"]).all()):
            raise Stage1BranchError("branch rewards contain NaN/Inf")
        utility = payload["branch_success"].any(dim=-1).float() + payload["branch_rewards"].amax(dim=-1)
        if bool(torch.allclose(utility, utility[:1].expand_as(utility))):
            raise Stage1BranchError("branch candidates contain no outcome supervision signal")
        sample.update(payload)
        sample["branch_id"] = row["branch_id"]
        sample["sample_id"] = row["sample_id"]
        return sample
