#!/usr/bin/env python3
"""Re-encode audited real RoboCasa rollouts into unified V8 Stage1 payloads."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import torch

from scripts.data.run_cache_worker import _encode, _strict_encoder, _view_batch
from wm3d_v3.data.grouped_normalization import (
    GroupedRobotNormalizer,
    normalize_grouped_masked,
)
from wm3d_v3.data.grouped_robot import GroupedRobotLimits, pack_grouped_robot_window
from wm3d_v3.data.manifest_contract import (
    load_cache_episode_index,
    load_data_profile,
    sha256_file,
)
from wm3d_v3.data.episode_robot import _exact_current_states
from wm3d_v3.data.source_adapters import adapt_action_series, load_adapter_contract
from wm3d_v3.data.unified_cache_dataset import _fuse_target_tokens, _pool_masked
from wm3d_v3.encoders.native_vggt import NativeVGGTEncoder
from wm3d_v3.stage1_planner.dataset import BRANCH_SCHEMA, GENERATOR_RECEIPT_SCHEMA
from wm3d_v3.stage1_planner.train import _stage0_dataset, _verify_runtime_checkout
from wm3d_v3.training.runtime_contract import load_materialized_runtime


AUDIT_SCHEMA = "wm3d_v8_robocasa_real_rollout_audit_v2"


class _Arrays:
    def __init__(self, values: dict[str, np.ndarray]):
        self.values = values

    def array(self, key: str) -> np.ndarray:
        if key not in self.values:
            raise RuntimeError(f"candidate adapter requested unknown array {key!r}")
        return self.values[key]


def _validate_rollout_audit_authority(
    audit: dict, expected_code_commit: str
) -> None:
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("passed") is not True:
        raise RuntimeError("rollout audit did not pass")
    if audit.get("code_commit") != expected_code_commit:
        raise RuntimeError("rollout audit code commit differs from Stage0 runtime")


def _publish(path: Path, payload: bytes) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite non-identical output: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _torch_bytes(value: dict[str, torch.Tensor], path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.encode.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            torch.save({key: tensor.detach().cpu().contiguous() for key, tensor in value.items()}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary.read_bytes()
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_commit_sha(path: Path) -> str:
    commit = path.resolve(strict=True) / "COMMITTED.json"
    if commit.is_symlink() or not commit.is_file():
        raise RuntimeError("Stage0 checkpoint must be a committed DCP directory")
    return sha256_file(commit)


def _runtime_lineage(runtime: dict, runtime_sha: str, checkpoint: Path) -> dict[str, str]:
    closure = runtime["data_closure"]
    window_seal = json.loads(Path(closure["cache_seal_path"]).read_text())
    return {
        "runtime_config_sha256": runtime_sha,
        "data_profile_sha256": closure["data_profile_sha256"],
        "model_profile_sha256": runtime["bindings"]["model_profile_sha256"],
        "window_index_sha256": closure["cache_index_sha256"],
        "grouped_normalization_sha256": closure["grouped_normalization_sha256"],
        "task_bank_index_sha256": window_seal["task_bank_index_sha256"],
        "encoder_contract_sha256": window_seal["encoder_contract_sha256"],
        "task_encoder_contract_sha256": window_seal["task_encoder_contract_sha256"],
        "representation_contract_sha256": window_seal["representation_contract_sha256"],
        "stage0_checkpoint_commit_sha256": _checkpoint_commit_sha(checkpoint),
    }


def _find_sample(
    dataset, *, source: str, episode_id: int, t0: int, offsets: list[int],
    expected_feature_shard: str, expected_robot_shard: str,
) -> tuple[int, dict]:
    expected_episode = f"{source}:{episode_id:09d}"
    episode_rows: dict[tuple[str, str], tuple[int, object]] = {}
    for index, entry in enumerate(dataset.entries):
        if (
            entry.feature_shard != expected_feature_shard
            or entry.robot_shard != expected_robot_shard
        ):
            continue
        frame = dataset.shards.read_many(
            entry.feature_shard, ("source_observation_row",),
            (entry.context_feature_rows[-1],) + entry.future_feature_rows,
        )["source_observation_row"].to(torch.int64)
        if int(frame[0]) == t0 and frame[1:].tolist() == [t0 + item for item in offsets]:
            episode_rows.setdefault((entry.robot_shard, entry.feature_shard), (index, entry))
    if len(episode_rows) != 1:
        raise RuntimeError(
            f"expected one exact Stage0 window for episode={expected_episode} t0={t0}, "
            f"found {len(episode_rows)}"
        )
    index, entry = next(iter(episode_rows.values()))
    return index, dataset[index]


def _candidate_actions(
    *, simulator: np.ndarray, adapter, embodiment, boundaries: np.ndarray,
    limits: GroupedRobotLimits, normalizer: GroupedRobotNormalizer,
    source: str, stage0_sample: dict, current_state,
    candidate_timestamps_s: np.ndarray,
) -> dict[str, torch.Tensor]:
    candidates = simulator.shape[0]
    values, masks, dt, sample_masks, coarse_values, coarse_masks = [], [], [], [], [], []
    normalization = normalizer.tensors_for(
        source=source,
        embodiment_id=int(stage0_sample["embodiment_ids"]),
        group_ids=stage0_sample["action_group_ids"],
        action_semantic_ids=stage0_sample["action_semantic_ids"],
        state_semantic_ids=stage0_sample["state_semantic_ids"],
    )
    for candidate in range(candidates):
        raw = np.concatenate(
            (simulator[candidate, :, 7:11], simulator[candidate, :, 11:12], simulator[candidate, :, 0:7]),
            axis=-1,
        ).astype(np.float32, copy=False)
        timestamp = np.asarray(candidate_timestamps_s, dtype=np.float64)
        if timestamp.ndim != 1 or not 0 < timestamp.size <= raw.shape[0]:
            raise RuntimeError("Stage0 source clock cannot bind candidate simulator commands")
        raw = raw[: timestamp.size]
        accessor = _Arrays({"action": raw, "timestamp": timestamp})
        series = adapt_action_series(accessor=accessor, contract=adapter, embodiment=embodiment)
        packed = pack_grouped_robot_window(
            embodiment=embodiment,
            limits=limits,
            world_boundaries_s=boundaries,
            action_series=series,
            current_state=current_state,
            policy_chunk_start_s=float(boundaries[0]),
        )
        fine_value = torch.from_numpy(packed.fine_action_values)
        fine_mask = torch.from_numpy(packed.fine_action_mask)
        fine_value = normalize_grouped_masked(
            fine_value, fine_mask,
            offset=normalization.fine_action_offset,
            scale=normalization.fine_action_scale,
            group_axis=1,
        )
        coarse_value = torch.from_numpy(packed.coarse_action_values)
        coarse_mask = torch.from_numpy(packed.coarse_action_mask)
        coarse_value = normalize_grouped_masked(
            coarse_value, coarse_mask,
            offset=normalization.coarse_action_offset,
            scale=normalization.coarse_action_scale,
            group_axis=1,
        )
        values.append(fine_value)
        masks.append(fine_mask)
        dt.append(torch.from_numpy(packed.fine_action_dt))
        sample_masks.append(torch.from_numpy(packed.fine_sample_mask))
        coarse_values.append(coarse_value)
        coarse_masks.append(coarse_mask)
    return {
        "candidate_fine_action_values": torch.stack(values),
        "candidate_fine_action_mask": torch.stack(masks),
        "candidate_fine_action_dt": torch.stack(dt),
        "candidate_fine_sample_mask": torch.stack(sample_masks),
        "candidate_coarse_action_values": torch.stack(coarse_values),
        "candidate_coarse_action_mask": torch.stack(coarse_masks),
    }


def _evidence(
    *, rgb: np.ndarray, rgb_indices: list[int], encoder: NativeVGGTEncoder,
    encoder_config, device: torch.device, batch_frames: int, model_grid: int,
) -> dict[str, torch.Tensor]:
    selected = np.asarray(rgb[:, rgb_indices], dtype=np.uint8)
    candidates, horizon, views = selected.shape[:3]
    decoded = {
        name: SimpleNamespace(frames=selected[:, :, slot].reshape(-1, *selected.shape[3:]))
        for slot, name in enumerate(("agentview_left", "agentview_right", "eye_in_hand"))
    }
    images, view_mask = _view_batch(
        decoded=decoded,
        slots=("agentview_left", "agentview_right", "eye_in_hand"),
        input_size=encoder_config.input_rgb_size,
    )
    encoded = _encode(
        encoder=encoder, images=images, view_mask=view_mask, device=device,
        batch_frames=batch_frames,
    )
    source_grid = int(encoder_config.token_grid)
    confidence = encoded["geometry_confidence"].float()
    real = encoded["view_mask"][..., None]
    geometry_valid = confidence > 0
    world_mask = (real & geometry_valid).any(dim=1)
    token, token_mask = _fuse_target_tokens(
        encoded["view_tokens"], confidence, encoded["view_mask"].bool(), world_mask,
        source_grid=source_grid, target_grid=model_grid,
    )
    depth_mask = real & geometry_valid & torch.isfinite(encoded["depth"]) & (encoded["depth"] > 0)
    point_mask = real & geometry_valid & torch.isfinite(encoded["point"]).all(dim=-1)
    depth, depth_valid = _pool_masked(
        encoded["depth"], depth_mask, source_grid=source_grid, target_grid=model_grid
    )
    point, point_valid = _pool_masked(
        encoded["point"], point_mask, source_grid=source_grid, target_grid=model_grid
    )
    confidence_pooled, _ = _pool_masked(
        confidence, real.expand_as(confidence), source_grid=source_grid, target_grid=model_grid
    )
    camera_mask = encoded["view_mask"] & torch.isfinite(encoded["camera_pose"]).all(dim=-1)
    def shape(value: torch.Tensor, *tail: int) -> torch.Tensor:
        return value.reshape(candidates, horizon, *tail)
    patches = model_grid * model_grid
    return {
        "branch_future_tokens": shape(token, patches, token.shape[-1]),
        "branch_token_mask": shape(token_mask, patches),
        "branch_depth": shape(depth, views, patches),
        "branch_depth_mask": shape(depth_valid, views, patches),
        "branch_point": shape(point, views, patches, 3),
        "branch_point_mask": shape(point_valid, views, patches),
        "branch_camera_pose": shape(encoded["camera_pose"], views, 9),
        "branch_camera_pose_mask": shape(camera_mask, views),
        "branch_geometry_confidence": shape(confidence_pooled, views, patches),
        "branch_view_mask": shape(encoded["view_mask"], views),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--stage0-checkpoint", type=Path, required=True)
    parser.add_argument("--rollout-audit", type=Path, required=True)
    parser.add_argument("--encoder-contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-frames", type=int, default=2)
    args = parser.parse_args()
    if args.batch_frames <= 0:
        raise RuntimeError("batch size must be positive")
    runtime, runtime_sha = load_materialized_runtime(args.runtime)
    repo = Path(__file__).resolve().parents[1]
    _verify_runtime_checkout(runtime, repo)
    profile = load_data_profile(Path(runtime["data_closure"]["data_profile_path"]), verify_source_manifests=False)
    sources = {source.name: source for source in profile.sources}
    if not sources or any(not name.startswith("robocasa_stage1_real") for name in sources):
        raise RuntimeError("real Stage1 producer requires sealed RoboCasa-only sources")
    audit_path = args.rollout_audit.resolve(strict=True)
    audit = json.loads(audit_path.read_text())
    _validate_rollout_audit_authority(
        audit, str(runtime["run"]["code_commit"])
    )
    rollout_audit_sha = sha256_file(audit_path)
    audited_roots = {
        name: Path(path).resolve(strict=True)
        for name, path in audit.get("source_roots", {}).items()
    }
    if audited_roots != {name: source.raw_root for name, source in sources.items()}:
        raise RuntimeError("rollout audit/source profile roots mismatch")
    offsets_seconds = [float(item) for item in runtime["model_profile"]["sampling"]["future_offsets_seconds"]]
    action_period_s = float(audit["simulator_action_period_seconds"])
    if not np.isfinite(action_period_s) or action_period_s <= 0:
        raise RuntimeError("rollout audit simulator cadence is invalid")
    expected_offsets = [int(round(item / action_period_s)) for item in offsets_seconds]
    if any(
        not np.isclose(offset * action_period_s, seconds, rtol=0.0, atol=1.0e-8)
        for offset, seconds in zip(expected_offsets, offsets_seconds)
    ):
        raise RuntimeError("model future offsets are not real sealed simulator ticks")
    lineage = _runtime_lineage(runtime, runtime_sha, args.stage0_checkpoint)
    normalizer = GroupedRobotNormalizer.load(
        Path(runtime["data_closure"]["grouped_normalization_path"]),
        expected_sha256=lineage["grouped_normalization_sha256"],
        expected_data_profile_sha256=lineage["data_profile_sha256"],
        expected_model_profile_sha256=lineage["model_profile_sha256"],
        expected_window_index_sha256=lineage["window_index_sha256"], data_profile=profile,
    )
    encoder_config = _strict_encoder(args.encoder_contract)
    if sha256_file(args.encoder_contract.resolve(strict=True)) != lineage["encoder_contract_sha256"]:
        raise RuntimeError("encoder contract differs from Stage0 cache closure")
    device = torch.device(args.device)
    encoder = NativeVGGTEncoder(encoder_config, device=str(device), local_files_only=True).eval().to(device)
    model = runtime["model_profile"]["model"]
    limits = GroupedRobotLimits(
        max_groups=int(model["max_action_groups"]), max_substeps=int(model["max_action_substeps"]),
        max_action_dim=int(model["max_action_dim"]), max_state_dim=int(model["max_state_dim"]),
    )
    model_grid = int(round(int(model["P"]) ** 0.5))
    if model_grid * model_grid != int(model["P"]):
        raise RuntimeError("Stage0 P is not a square grid")
    rows = []
    output_root = args.output_root.absolute()
    simulator_revision = json.dumps(audit["simulator_revision"], sort_keys=True, separators=(",", ":"))
    episode_entries = load_cache_episode_index(
        Path(runtime["data_closure"]["episode_cache_index_path"]),
        expected_sha256=runtime["data_closure"]["episode_cache_index_sha256"],
    )
    episode_by_id = {entry.episode_id: entry for entry in episode_entries}
    for audited in audit["rows"]:
        split = str(audited["split"])
        source_name = str(audited["source"])
        source = sources.get(source_name)
        if source is None:
            raise RuntimeError(f"rollout audit names unknown source {source_name!r}")
        adapter = load_adapter_contract(
            source.adapter_config_path,
            expected_sha256=source.adapter_contract_sha256,
        )
        embodiment = profile.embodiments[source.embodiment]
        if audited["source_future_row_offsets"] != expected_offsets:
            raise RuntimeError("rollout audit future rows differ from model sampling profile")
        dataset = _stage0_dataset(runtime, split)
        episode_key = f"{source_name}:{int(audited['episode_id']):09d}"
        cache_episode = episode_by_id.get(episode_key)
        if cache_episode is None or cache_episode.split != split:
            raise RuntimeError(f"Stage0 episode cache lacks audited {split} root {episode_key}")
        sample_index, stage0 = _find_sample(
            dataset, source=source_name, episode_id=int(audited["episode_id"]),
            t0=int(audited["t0"]), offsets=expected_offsets,
            expected_feature_shard=cache_episode.feature_shard,
            expected_robot_shard=cache_episode.robot_shard,
        )
        entry = dataset.entries[sample_index]
        with np.load(Path(audited["runtime_payload_path"]), allow_pickle=False) as npz:
            simulator = np.asarray(npz["simulator_actions"], dtype=np.float32)
            rewards = torch.from_numpy(np.asarray(npz["branch_rewards"][:, audited["outcome_indices"]], dtype=np.float32))
            dones = torch.from_numpy(np.asarray(npz["branch_dones"][:, audited["outcome_indices"]], dtype=np.bool_))
            success = torch.from_numpy(np.asarray(npz["branch_success"][:, audited["outcome_indices"]], dtype=np.bool_))
            native = _evidence(
                rgb=np.asarray(npz["branch_rgb"]), rgb_indices=audited["branch_rgb_indices"],
                encoder=encoder, encoder_config=encoder_config, device=device,
                batch_frames=args.batch_frames, model_grid=model_grid,
            )
        boundaries_dt = stage0["future_world_boundaries_dt"].double().numpy()
        anchor = float(stage0["world_times_s"][int(model["T"])-1])
        boundaries = anchor + boundaries_dt
        if not np.allclose(boundaries_dt[1:], np.asarray(offsets_seconds), rtol=0.0, atol=1.0e-6):
            raise RuntimeError("Stage0 window clock differs from audited real rollout offsets")
        _robot_values, prepared_robot = dataset.robot.read(
            entry.robot_shard, embodiment=embodiment
        )
        current_state = _exact_current_states(
            prepared_robot.state_series, timestamp_s=anchor
        )
        action_clocks = []
        for series in prepared_robot.action_series:
            if series.timestamps_s is None:
                raise RuntimeError("real simulator candidates require timestamped fine-command groups")
            clock = np.asarray(series.timestamps_s, dtype=np.float64)
            clock = clock[(clock >= np.float64(boundaries[0])) & (clock < np.float64(boundaries[-1]))]
            action_clocks.append(clock)
        candidate_timestamps = action_clocks[0]
        if not all(np.array_equal(clock, candidate_timestamps) for clock in action_clocks[1:]):
            raise RuntimeError("RoboCasa grouped action clocks differ; candidate replay cannot be rebound")
        if candidate_timestamps.size > simulator.shape[1]:
            raise RuntimeError("real candidate trajectory is shorter than the sealed Stage0 horizon")
        observed_dt = np.diff(candidate_timestamps)
        if observed_dt.size and not np.allclose(
            observed_dt, action_period_s, rtol=0.0, atol=1.0e-6
        ):
            raise RuntimeError("Stage0 source action clock differs from audited simulator cadence")
        actions = _candidate_actions(
            simulator=simulator, adapter=adapter, embodiment=embodiment,
            boundaries=boundaries, limits=limits, normalizer=normalizer,
            source=source_name, stage0_sample=stage0,
            current_state=current_state, candidate_timestamps_s=candidate_timestamps,
        )
        candidates, horizon = rewards.shape
        exact_future_dt = (
            stage0["world_times_s"][int(model["T"]):int(model["T"])+horizon]
            - stage0["world_times_s"][int(model["T"])-1]
        )
        payload = {
            **actions, **native,
            "branch_future_dt_s": exact_future_dt[None].expand(candidates, -1).clone(),
            "branch_rewards": rewards, "branch_dones": dones, "branch_success": success,
            "branch_valid": torch.ones(candidates, dtype=torch.bool),
        }
        payload_path = output_root / split / f"{entry.sample_id}.pt"
        _publish(payload_path, _torch_bytes(payload, payload_path))
        payload_sha = sha256_file(payload_path)
        receipt = {
            "schema": GENERATOR_RECEIPT_SCHEMA,
            "sample_index": sample_index, "sample_id": entry.sample_id,
            "source": entry.source, "split": entry.split, "embodiment": entry.embodiment,
            "payload_sha256": payload_sha, **lineage,
            "rollout_audit_sha256": rollout_audit_sha,
            "source_manifest_sha256": source.manifest_sha256,
            "adapter_contract_sha256": source.adapter_contract_sha256,
            "simulator_revision": simulator_revision,
            "simulator_seed": int(audited["candidate_seed"]),
            "real_simulator_outcomes": True, "future_observation_leakage": False,
            "candidate_action_abi": "wm3d_v8_grouped_robot_v1",
            "candidate_actions_from_adapter": True,
            "candidate_actions_grouped_normalized": True,
            "native_evidence_from_frozen_encoder": True,
        }
        receipt_path = output_root / "receipts" / f"{entry.sample_id}.json"
        _publish(receipt_path, (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode())
        rows.append({
            "schema": BRANCH_SCHEMA,
            "sample_index": sample_index, "sample_id": entry.sample_id,
            "source": entry.source, "split": entry.split, "embodiment": entry.embodiment,
            "payload": str(payload_path), "payload_sha256": payload_sha,
            "generator_receipt": str(receipt_path), "generator_receipt_sha256": sha256_file(receipt_path),
            "rollout_audit_sha256": rollout_audit_sha,
            **lineage,
        })
    if {row["split"] for row in rows} != {"train", "val", "test"}:
        raise RuntimeError("candidate generation did not close train/val/test")
    rows.sort(key=lambda row: row["split"])
    manifest = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode()
    _publish(args.output_manifest, manifest)
    print(json.dumps({
        "candidate_manifest": str(args.output_manifest.absolute()),
        "candidate_manifest_sha256": sha256_file(args.output_manifest.absolute()),
        "rows": len(rows), "splits": sorted(row["split"] for row in rows),
        "rollout_audit_sha256": rollout_audit_sha, **lineage,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
