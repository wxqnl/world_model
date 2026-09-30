#!/usr/bin/env python3
"""Admit the frozen RoboTwin release as observed-state targets, without hashes.

This does not infer unrecorded controls or re-decode video. It reuses the full
read receipt, then runs the production field/clock inventory on real Parquet.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from wm3d.data.file_binding import bind_stat_asset, verify_stat_asset
from wm3d.data.source_adapters import load_adapter_contract, adapt_action_series, adapt_state_series
from wm3d.data.grouped_robot import EmbodimentSpec, ActionGroupSpec, GroupedRobotLimits
from wm3d.data.episode_robot import PreparedEpisodeRobot, assemble_robot_window_from_prepared_episode
from wm3d.data.episode_io import VerifiedAssetStore, open_admitted_episode_accessor, select_episode_cache_rows
from wm3d.data.source_inventory import _episode_metadata, scan_lerobot_source
from wm3d.data.foundation_bridge import FoundationBridgeConfig, pack_grouped_effect_sequence
from wm3d.data.window_selection import select_observed_world_window, WindowSelectionError


def publish(path, value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(path)
        return
    with path.open("x") as f:
        f.write(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--full-read", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--admission-id", required=True)
    p.add_argument("--inventory", action="store_true")
    a = p.parse_args()
    root = a.raw_root.resolve(strict=True)
    a.output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    adapter_path = repo / "configs/adapters/robotwin_unified_recorded_state.yaml"
    template_path = repo / "configs/data/robotwin_unified_state.admission.yaml"
    bind = lambda p: bind_stat_asset(p.parent, p.name, admission_id=a.admission_id)
    ab, tb, rb = bind(adapter_path), bind(template_path), bind(a.full_read)
    full = json.loads(a.full_read.read_text())
    if not full.get("passed"):
        raise RuntimeError("RoboTwin full-read receipt has not passed")
    template = yaml.safe_load(template_path.read_text())
    e = template["embodiments"][0]
    emb = EmbodimentSpec(e["name"], e["embodiment_id"], tuple(ActionGroupSpec(
        **{k: tuple(v) if isinstance(v,list) else v for k,v in g.items()}
    ) for g in e["groups"]))
    adapter = load_adapter_contract(adapter_path, expected_binding=ab)
    metadata = sorted(_episode_metadata(root), key=lambda x: int(x["episode_index"]))
    ids = [int(x["episode_index"]) for x in metadata]
    if ids != list(range(27500)) or sum(int(x["length"]) for x in metadata) != 6075103:
        raise RuntimeError("Frozen release membership changed")
    # Local IDs are the recorded IDs of this one release, not fabricated global
    # identities or a claim of equality with another release.
    order = np.random.default_rng(3407).permutation(ids)
    val = set(int(x) for x in order[:1375])
    catalog = {i: {"episode_id": f"lerobot/robotwin_unified:frozen20260927:episode{i:06d}",
                   "split": "val" if i in val else "train"} for i in ids}
    catpath = a.output / "episode_catalog.json"
    publish(catpath, dict(schema="wm3d_admitted_episode_catalog_v1",
        admission_id=a.admission_id, source="robotwin_unified_state", episodes=catalog))
    audit = dict(schema="wm3d_admitted_source_adapter_audit_receipt_v1",
        source="robotwin_unified_state", admission_id=a.admission_id,
        adapter_contract_binding=ab, data_template_binding=tb,
        episode_catalog_binding=bind(catpath),
        structural_checks="pass", semantic_review="operator_confirmed_fail_closed",
        user_decision="保留数据，明确区分状态目标与控制指令",
        supervision="coarse_effect", physical_units="source-native motor coordinates",
        calibrated_metric_or_gripper_polarity_claim=False,
        mapping="All 14 columns unchanged; action[t] equals measured state[t+1]; drop final label without successor.",
        semantic_embedding_id=16, command_prediction_claim=False,
        original_episode_identity="Recorded episode_index in the single frozen lerobot/robotwin_unified release; no stacked conversion or second release.",
        cross_release_global_identity_claim=False,
        split_policy=dict(seed=3407, train_episodes=26125, val_episodes=1375, unit="whole original episode"),
        full_read_receipt_path=str(a.full_read), full_read_receipt_binding=rb)
    publish(a.output / "adapter_audit.json", audit)
    publish(a.output / "adapter.binding.json", ab)
    if not a.inventory:
        print(json.dumps(dict(status="adapter_and_catalog_published", episodes=len(ids))), flush=True)
        return
    rows, receipt = scan_lerobot_source(root=root, source="robotwin_unified_state",
        embodiment=emb, adapter=adapter, split_seed=3407, train_fraction=.95,
        validation_fraction=.05, default_task="", admission_id=a.admission_id,
        episode_catalog=catalog)
    manifest = a.output / "manifest.jsonl"
    with manifest.open("x") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",",":"))+"\n")
    # One real sufficiently long episode through the same grouped composition
    # and bridge used by training. No synthesized RGB, latent, or task vector.
    model = yaml.safe_load((repo.parent / ".." / "wm3d_native_fresh_pretraining_20260930/candidates_three_node_24/model.candidate.yaml").resolve().read_text())
    replay = None
    for row in rows:
        if row["observation_samples"] < 180:
            continue
        access = open_admitted_episode_accessor(source_root=root, payload=row["payload"],
            payload_binding=row["payload_binding"], row_start=row["payload_row_start"],
            row_stop=row["payload_row_stop"], adapter=adapter, asset_verifier=VerifiedAssetStore())
        clock = np.asarray(access.array("timestamp"), dtype=np.float64)
        acts = tuple(adapt_action_series(accessor=access, contract=adapter, embodiment=emb))
        states = tuple(adapt_state_series(accessor=access, contract=adapter, embodiment=emb))
        # Text embeddings are not used by grouped robot alignment.
        prepared = PreparedEpisodeRobot(clock, acts, states, None)
        selected = select_episode_cache_rows(clock, minimum_separation_s=.2)
        s = model["sampling"]; m = model["model"]
        for anchor in range(len(selected)):
            try:
                w = select_observed_world_window(clock[selected], anchor_index=anchor,
                    context_samples=m["T"], future_samples=m["K"],
                    context_horizon_s=s["context_horizon_seconds"],
                    future_horizon_s=s["future_horizon_seconds"],
                    minimum_horizon_coverage=s["minimum_horizon_coverage"])
            except WindowSelectionError:
                continue
            bounds = selected[np.r_[w.leading_boundary_index,w.context_indices,w.future_indices]]
            sample = assemble_robot_window_from_prepared_episode(prepared=prepared,
                embodiment=emb, selected_source_boundary_indices=bounds,
                limits=GroupedRobotLimits(max_groups=3,max_action_dim=14,max_state_dim=10),
                context_samples=16, max_policy_queries=256, policy_target_horizon_s=1.6)
            batch = {k:v.unsqueeze(0) for k,v in sample.items()}
            result = pack_grouped_effect_sequence(batch, FoundationBridgeConfig())
            history = pack_grouped_effect_sequence(batch, FoundationBridgeConfig(), history=True)
            truth = np.asarray(access.array("observation.state"),dtype=np.float32)[bounds[17:]]
            measured = torch.cat([result["action"][0,:,:7],result["action"][0,:,14:21]],-1)
            if not torch.equal(measured,torch.from_numpy(truth)):
                raise RuntimeError("RoboTwin coarse bridge differs from actual endpoint states")
            assert not sample["future_factual_fine_action_mask"].any()
            assert (sample["action_semantic_ids"][:2,:7] == 16).all()
            replay = dict(episode_id=row["episode_id"], source_boundary_rows=bounds.tolist(),
                actual_endpoint_values_exact=True, all_14_columns_preserved=True,
                future_targets=8, history_targets=16, command_lane_empty=True,
                endpoint_times_s=result["action_time_s"][0,:,0].tolist(),
                state_target_semantic_id=16, no_video_decode_or_SHA=True)
            break
        if replay is not None:
            break
    if replay is None:
        raise RuntimeError("No full-model-horizon RoboTwin window found")
    verify_stat_asset(adapter_path.parent,adapter_path.name,ab)
    verify_stat_asset(template_path.parent,template_path.name,tb)
    receipt.update(dict(manifest_path=str(manifest), manifest_binding=bind(manifest),
        adapter_audit_receipt_path=str(a.output/"adapter_audit.json"),
        adapter_audit_receipt_binding=bind(a.output/"adapter_audit.json"),
        data_template_path=str(template_path), data_template_binding=tb,
        adapter_contract_path=str(adapter_path), adapter_contract_binding=ab,
        real_foundation_effect_bridge_replay=replay,
        whole_episode_split=True, started_formal_training=False,
        checked_at=datetime.now().astimezone().isoformat()))
    publish(a.output / "receipt.json", receipt)
    print(json.dumps(dict(status="robotwin_state_inventory_complete", episodes=len(rows),
        replay=replay, manifest_bytes=manifest.stat().st_size)), flush=True)


if __name__ == "__main__":
    main()
