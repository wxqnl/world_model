#!/usr/bin/env python3
"""Seal real simulator branches into the unified Stage1 evidence contract."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import uuid

import torch

from wm3d_v3.data.manifest_contract import (
    SHA256_RE,
    canonical_sha256,
    load_data_profile,
    load_cache_index,
    sha256_file,
)
from wm3d_v3.stage1_planner.dataset import (
    BRANCH_INDEX_SCHEMA,
    BRANCH_SCHEMA,
    BRANCH_SEAL_SCHEMA,
    GENERATOR_RECEIPT_FIELDS,
    GENERATOR_RECEIPT_SCHEMA,
    validate_rollout_audit_binding,
)
from wm3d_v3.stage1_planner.train import _verify_runtime_checkout
from wm3d_v3.training.runtime_contract import load_materialized_runtime


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict:
    path = path.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return value


def _commit_sha(path: Path) -> str:
    commit = path.resolve(strict=True) / "COMMITTED.json"
    if commit.is_symlink() or not commit.is_file():
        raise ValueError("Stage0 source must be a committed DCP checkpoint directory")
    return sha256_file(commit)


def _validate_candidate_payload(value: dict, *, model: dict, horizon: int) -> None:
    required = {
        "candidate_fine_action_values", "candidate_fine_action_mask",
        "candidate_fine_action_dt", "candidate_fine_sample_mask",
        "candidate_coarse_action_values", "candidate_coarse_action_mask",
        "branch_future_tokens", "branch_future_dt_s", "branch_token_mask",
        "branch_depth", "branch_depth_mask", "branch_point", "branch_point_mask",
        "branch_camera_pose", "branch_camera_pose_mask",
        "branch_geometry_confidence", "branch_view_mask",
        "branch_rewards", "branch_dones", "branch_success", "branch_valid",
    }
    if set(value) != required or any(not isinstance(item, torch.Tensor) for item in value.values()):
        raise ValueError("candidate payload must contain exactly the unified tensor fields")
    candidates, observed_h = value["branch_rewards"].shape
    if candidates < 2 or observed_h != horizon:
        raise ValueError("candidate outcomes must be [C,H] with C>=2")
    K, P, D, V = (int(model[name]) for name in ("K", "P", "token_dim", "num_views"))
    if not 0 < horizon <= K:
        raise ValueError("Stage1 horizon must lie inside Stage0 K")
    if value["branch_future_tokens"].shape != (candidates, horizon, P, D):
        raise ValueError("candidate branch tokens are not unified VGGT evidence")
    evidence_float = (
        "branch_future_tokens", "branch_future_dt_s", "branch_depth",
        "branch_point", "branch_camera_pose", "branch_geometry_confidence",
        "candidate_fine_action_values", "candidate_fine_action_dt",
        "candidate_coarse_action_values", "branch_rewards",
    )
    if any(not value[name].is_floating_point() for name in evidence_float):
        raise ValueError("candidate evidence/action/reward tensors must be floating point")
    if value["branch_future_dt_s"].shape != (candidates, horizon):
        raise ValueError("candidate future timestamps must be [C,H]")
    if (
        not bool(torch.isfinite(value["branch_future_dt_s"]).all())
        or bool((value["branch_future_dt_s"] <= 0).any())
        or (horizon > 1 and not bool(torch.diff(value["branch_future_dt_s"], dim=-1).gt(0).all()))
    ):
        raise ValueError("candidate future timestamps must be finite/positive/increasing")
    if value["branch_token_mask"].shape != (candidates, horizon, P) or value["branch_token_mask"].dtype != torch.bool:
        raise ValueError("candidate native token mask mismatch")
    if not bool(value["branch_token_mask"].any(dim=-1).all()):
        raise ValueError("every future frame requires native token evidence")
    geometry = (candidates, horizon, V, P)
    if value["branch_depth"].shape != geometry or value["branch_point"].shape != (*geometry, 3):
        raise ValueError("candidate native geometry shape mismatch")
    for name in ("branch_depth_mask", "branch_point_mask"):
        if value[name].shape != geometry or value[name].dtype != torch.bool:
            raise ValueError(f"candidate {name} mismatch")
    if value["branch_camera_pose"].shape != (candidates, horizon, V, 9):
        raise ValueError("candidate camera pose shape mismatch")
    if value["branch_camera_pose_mask"].shape != (candidates, horizon, V) or value["branch_camera_pose_mask"].dtype != torch.bool:
        raise ValueError("candidate camera pose mask mismatch")
    if value["branch_geometry_confidence"].shape != geometry:
        raise ValueError("candidate geometry confidence shape mismatch")
    if value["branch_view_mask"].shape != (candidates, horizon, V) or value["branch_view_mask"].dtype != torch.bool:
        raise ValueError("candidate view mask mismatch")
    visible = value["branch_view_mask"][..., None]
    if bool((value["branch_depth_mask"] & ~visible).any()) or bool((value["branch_point_mask"] & ~visible).any()):
        raise ValueError("candidate geometry masks exist outside measured views")
    if bool((value["branch_camera_pose_mask"] & ~value["branch_view_mask"]).any()):
        raise ValueError("candidate camera poses exist outside measured views")
    if bool((value["branch_geometry_confidence"] < 0).any()):
        raise ValueError("candidate geometry confidence must be non-negative")
    fine = value["candidate_fine_action_values"]
    if fine.ndim != 5 or fine.shape[:2] != (candidates, K):
        raise ValueError("candidate fine actions must cover the same sealed K")
    if fine.shape[2:] != (
        int(model["max_action_groups"]), int(model["max_action_substeps"]),
        int(model["max_action_dim"]),
    ):
        raise ValueError("candidate fine action capacities differ from model profile")
    if value["candidate_fine_action_mask"].shape != fine.shape:
        raise ValueError("candidate fine action mask mismatch")
    if value["candidate_fine_action_dt"].shape != fine.shape[:-1] or value["candidate_fine_sample_mask"].shape != fine.shape[:-1]:
        raise ValueError("candidate fine action timestamp/mask mismatch")
    if value["candidate_fine_action_mask"].dtype != torch.bool or value["candidate_fine_sample_mask"].dtype != torch.bool:
        raise ValueError("candidate fine action masks must be boolean")
    if bool((value["candidate_fine_action_mask"].any(dim=-1) & ~value["candidate_fine_sample_mask"]).any()):
        raise ValueError("candidate fine dimensions exist outside real samples")
    valid_dt = value["candidate_fine_action_dt"][value["candidate_fine_sample_mask"]]
    if not bool(torch.isfinite(valid_dt).all()) or bool((valid_dt < 0).any()):
        raise ValueError("candidate action timestamps are invalid")
    coarse = value["candidate_coarse_action_values"]
    if (
        coarse.shape != (candidates, K, int(model["max_action_groups"]), int(model["max_action_dim"]))
        or value["candidate_coarse_action_mask"].shape != coarse.shape
        or value["candidate_coarse_action_mask"].dtype != torch.bool
    ):
        raise ValueError("candidate coarse action shape mismatch")
    if not bool(
        value["candidate_fine_sample_mask"].any()
        or value["candidate_coarse_action_mask"].any()
    ):
        raise ValueError("candidate branches contain no measured action")
    if value["branch_valid"].shape != (candidates,) or value["branch_valid"].dtype != torch.bool or not bool(value["branch_valid"].all()):
        raise ValueError("all sealed candidate branches must be valid")
    if any(value[name].shape != (candidates, horizon) for name in ("branch_rewards", "branch_dones", "branch_success")):
        raise ValueError("candidate reward/done/success trajectories differ")
    if value["branch_dones"].dtype != torch.bool or value["branch_success"].dtype != torch.bool:
        raise ValueError("candidate done/success trajectories must be boolean")
    for name, tensor in value.items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"candidate payload {name} contains NaN/Inf")
    utility = value["branch_success"].any(dim=-1).float() + value["branch_rewards"].amax(dim=-1)
    if bool(torch.allclose(utility, utility[:1].expand_as(utility))):
        raise ValueError("candidate branches contain no label signal inside H")


def _validate_stage0_window_clock(
    value: dict, *, sample: dict, context: int, K: int, horizon: int
) -> None:
    world_times = sample["world_times_s"]
    if world_times.shape != (context + K,):
        raise ValueError("Stage0 world clock differs from sealed T/K")
    expected_future = world_times[context : context + horizon] - world_times[context - 1]
    candidates = int(value["branch_future_dt_s"].shape[0])
    if not torch.equal(
        value["branch_future_dt_s"].to(dtype=expected_future.dtype),
        expected_future[None].expand(candidates, -1),
    ):
        raise ValueError("candidate future timestamps differ from Stage0 window")
    boundaries = sample["future_world_boundaries_dt"]
    if boundaries.shape != (K + 1,) or not bool(torch.diff(boundaries).gt(0).all()):
        raise ValueError("Stage0 future boundary clock is invalid")
    upper = torch.diff(boundaries)[None, :, None, None].expand_as(
        value["candidate_fine_action_dt"]
    )
    if bool(
        (
            value["candidate_fine_sample_mask"]
            & (value["candidate_fine_action_dt"] >= upper)
        ).any()
    ):
        raise ValueError("candidate fine command lies outside its world interval")


def _validate_generator_receipt(
    path: Path,
    expected_sha: str,
    *,
    raw: dict,
    expected: dict,
    source_manifest_sha: str,
    adapter_sha: str,
) -> Path:
    if SHA256_RE.fullmatch(expected_sha) is None:
        raise ValueError("candidate generator receipt SHA is invalid")
    path = path.resolve(strict=True)
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha:
        raise ValueError("candidate generator receipt path/SHA mismatch")
    receipt = _load_json(path)
    if (
        set(receipt) != GENERATOR_RECEIPT_FIELDS
        or receipt.get("schema") != GENERATOR_RECEIPT_SCHEMA
    ):
        raise ValueError("candidate generator receipt fields/schema mismatch")
    identity = {name: raw[name] for name in ("sample_index", "sample_id", "source", "split", "embodiment", "payload_sha256")}
    if any(receipt.get(name) != value for name, value in {**identity, **expected}.items()):
        raise ValueError("candidate generator receipt lineage/identity mismatch")
    validate_rollout_audit_binding(raw, receipt)
    if receipt["source_manifest_sha256"] != source_manifest_sha or receipt["adapter_contract_sha256"] != adapter_sha:
        raise ValueError("candidate generator receipt source/adapter mismatch")
    if not isinstance(receipt["simulator_revision"], str) or not receipt["simulator_revision"].strip():
        raise ValueError("candidate generator receipt lacks simulator revision")
    if not isinstance(receipt["simulator_seed"], int):
        raise ValueError("candidate generator receipt lacks integer simulator seed")
    gates = (
        receipt["real_simulator_outcomes"],
        receipt["candidate_actions_from_adapter"],
        receipt["candidate_actions_grouped_normalized"],
        receipt["native_evidence_from_frozen_encoder"],
    )
    if any(value is not True for value in gates) or receipt["future_observation_leakage"] is not False:
        raise ValueError("candidate generator receipt did not pass causal unified gates")
    if receipt["candidate_action_abi"] != "wm3d_v8_grouped_robot_v1":
        raise ValueError("candidate generator receipt action ABI mismatch")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--stage0-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--output-seal", type=Path, required=True)
    args = parser.parse_args()
    runtime, runtime_sha = load_materialized_runtime(args.runtime)
    repo = Path(__file__).resolve().parents[1]
    _verify_runtime_checkout(runtime, repo)
    closure = runtime["data_closure"]
    profile = load_data_profile(Path(closure["data_profile_path"]), verify_source_manifests=False)
    source_by_name = {source.name: source for source in profile.sources}
    source_order = {name: index for index, name in enumerate(profile.source_order)}
    all_windows = load_cache_index(
        Path(closure["cache_index_path"]),
        expected_sha256=closure["cache_index_sha256"],
    )
    windows_by_split = {
        split: tuple(sorted(
            (entry for entry in all_windows if entry.split == split),
            key=lambda entry: (source_order[entry.source], entry.sample_id),
        ))
        for split in ("train", "val", "test")
    }
    # Reuse the exact Stage0 runtime loader for window-clock validation instead
    # of reconstructing timestamps from manifest metadata.
    from wm3d_v3.stage1_planner.train import _stage0_dataset

    stage0_by_split: dict[str, object] = {}
    window_seal = _load_json(Path(closure["cache_seal_path"]))
    lineage = {
        "runtime_config_sha256": runtime_sha,
        "data_profile_sha256": closure["data_profile_sha256"],
        "model_profile_sha256": runtime["bindings"]["model_profile_sha256"],
        "window_index_sha256": closure["cache_index_sha256"],
        "grouped_normalization_sha256": closure["grouped_normalization_sha256"],
        "task_bank_index_sha256": window_seal["task_bank_index_sha256"],
        "encoder_contract_sha256": window_seal["encoder_contract_sha256"],
        "task_encoder_contract_sha256": window_seal["task_encoder_contract_sha256"],
        "representation_contract_sha256": window_seal["representation_contract_sha256"],
        "stage0_checkpoint_commit_sha256": _commit_sha(args.stage0_checkpoint),
    }
    if any(SHA256_RE.fullmatch(str(value)) is None for value in lineage.values()):
        raise ValueError("Stage0 runtime/checkpoint lineage contains invalid SHA values")
    rows = []
    output_root = args.output_root.absolute()
    required_manifest = {
        "schema", "sample_index", "sample_id", "source", "split", "embodiment",
        "payload", "payload_sha256", "generator_receipt",
        "generator_receipt_sha256",
        "rollout_audit_sha256",
    } | set(lineage)
    rollout_audit_shas: set[str] = set()
    for line_number, line in enumerate(args.candidate_manifest.resolve(strict=True).read_text().splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict) or set(raw) != required_manifest or raw["schema"] != BRANCH_SCHEMA:
            raise ValueError(f"candidate manifest row {line_number} fields/schema mismatch")
        if any(raw.get(name) != value for name, value in lineage.items()):
            raise ValueError(f"candidate manifest row {line_number} lineage mismatch")
        if SHA256_RE.fullmatch(str(raw["rollout_audit_sha256"])) is None:
            raise ValueError("candidate manifest rollout-audit SHA is invalid")
        rollout_audit_shas.add(str(raw["rollout_audit_sha256"]))
        source_spec = source_by_name.get(str(raw["source"]))
        if source_spec is None or source_spec.embodiment != str(raw["embodiment"]):
            raise ValueError("candidate manifest source/embodiment is outside data profile")
        split_windows = windows_by_split.get(str(raw["split"]), ())
        sample_index = int(raw["sample_index"])
        window = split_windows[sample_index] if 0 <= sample_index < len(split_windows) else None
        if window is None or any(
            str(raw[name]) != str(getattr(window, name))
            for name in ("sample_id", "source", "split", "embodiment")
        ):
            raise ValueError("candidate manifest does not identify its Stage0 window")
        source = Path(raw["payload"])
        if source.is_symlink() or not source.is_file() or sha256_file(source) != raw["payload_sha256"]:
            raise ValueError("candidate payload SHA/path mismatch")
        receipt = _validate_generator_receipt(
            Path(raw["generator_receipt"]), str(raw["generator_receipt_sha256"]),
            raw=raw, expected=lineage,
            source_manifest_sha=source_spec.manifest_sha256,
            adapter_sha=source_spec.adapter_contract_sha256,
        )
        value = torch.load(source, map_location="cpu", weights_only=True)
        model = runtime["model_profile"]["model"]
        horizon = int(value["branch_rewards"].shape[1])
        _validate_candidate_payload(value, model=model, horizon=horizon)
        split = str(raw["split"])
        if split not in stage0_by_split:
            stage0_by_split[split] = _stage0_dataset(runtime, split)
        stage0_sample = stage0_by_split[split][sample_index]
        _validate_stage0_window_clock(
            value,
            sample=stage0_sample,
            context=int(model["T"]),
            K=int(model["K"]),
            horizon=horizon,
        )
        branch_id = canonical_sha256({
            **{name: raw[name] for name in sorted(raw) if name not in {"payload", "generator_receipt"}},
            "source_manifest_sha256": source_spec.manifest_sha256,
            "adapter_contract_sha256": source_spec.adapter_contract_sha256,
        })
        target = output_root / str(raw["split"]) / f"{branch_id}.pt"
        _publish(target, source.read_bytes())
        rows.append({
            "schema": BRANCH_INDEX_SCHEMA, "branch_id": branch_id,
            "sample_index": int(raw["sample_index"]), "sample_id": str(raw["sample_id"]),
            "source": str(raw["source"]), "split": str(raw["split"]),
            "embodiment": str(raw["embodiment"]), "path": str(target.absolute()),
            "payload_sha256": sha256_file(target), "candidates": int(value["branch_rewards"].shape[0]),
            "horizon": horizon, "K": int(model["K"]), "P": int(model["P"]),
            "token_dim": int(model["token_dim"]), "num_views": int(model["num_views"]),
            **lineage, "source_manifest_sha256": source_spec.manifest_sha256,
            "adapter_contract_sha256": source_spec.adapter_contract_sha256,
            "generator_receipt_path": str(receipt),
            "generator_receipt_sha256": sha256_file(receipt),
            "rollout_audit_sha256": str(raw["rollout_audit_sha256"]),
            "real_simulator_outcomes": True, "future_observation_leakage": False,
            "candidate_action_abi": "wm3d_v8_grouped_robot_v1",
        })
    if not rows:
        raise ValueError("candidate manifest is empty")
    if len(rollout_audit_shas) != 1:
        raise ValueError("candidate manifest mixes rollout audits")
    candidates = {int(row["candidates"]) for row in rows}
    horizons = {int(row["horizon"]) for row in rows}
    if len(candidates) != 1 or len(horizons) != 1:
        raise ValueError("sealed Stage1 closure requires one candidate count and horizon")
    rows.sort(key=lambda item: (item["split"], item["source"], item["sample_id"]))
    index_payload = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode()
    _publish(args.output_index.absolute(), index_payload)
    split_counts = Counter(row["split"] for row in rows)
    if any(split_counts[name] <= 0 for name in ("train", "val", "test")):
        raise ValueError("Stage1 release closure requires non-empty train/val/test branches")
    seal = {
        "schema": BRANCH_SEAL_SCHEMA, "branch_index_path": str(args.output_index.absolute()),
        "branch_index_sha256": sha256_file(args.output_index.absolute()), "row_count": len(rows),
        "row_count_by_split": {name: int(split_counts[name]) for name in ("train", "val", "test")},
        "candidate_count": next(iter(candidates)), "horizon": next(iter(horizons)),
        "rollout_audit_sha256": next(iter(rollout_audit_shas)),
        **lineage,
    }
    _publish(args.output_seal.absolute(), (json.dumps(seal, sort_keys=True, indent=2) + "\n").encode())
    print(json.dumps(seal, sort_keys=True))


if __name__ == "__main__":
    main()
