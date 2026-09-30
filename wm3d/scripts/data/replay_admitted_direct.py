"""Replay actual admitted RGB and grouped targets through the production reader."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import yaml
from wm3d.data.manifest_contract import DataProfile,SourceSpec
from wm3d.data.direct_raw import _DirectWindowSource,ADMITTED_DIRECT_RAW_DATA_CLOSURE_SCHEMA
from wm3d.data.admitted_direct_tasks import ADMITTED_DIRECT_TASK_SCHEMA
from wm3d.data.admitted_window_index import AdmittedWindowEntry
from wm3d.data.grouped_robot import GroupedRobotLimits
from wm3d.data.episode_robot import assemble_robot_window_from_prepared_episode
from wm3d.data.window_selection import select_observed_world_window,WindowSelectionError
from wm3d.data.foundation_bridge import FoundationBridgeConfig,pack_grouped_effect_sequence,pack_grouped_future_action
from scripts.data.admit_agibot_native import bind,publish,embodiment

def main():
 p=argparse.ArgumentParser();p.add_argument("--receipt",action="append",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True);p.add_argument("--model",type=Path,required=True);a=p.parse_args()
 model=yaml.safe_load(a.model.read_text());m=model["model"];sampling=model["sampling"]
 legacy=yaml.safe_load(Path("/data/Minko/wm3d_full_robot_data_20260904/data_profile.yaml").read_text())
 admission="wm3d_fresh_new_data_20260930";results=[]
 for ordinal,rp in enumerate(a.receipt):
  r=json.loads(rp.read_text());template=yaml.safe_load(Path(r["data_template_path"]).read_text());emb=embodiment(template)
  row=None
  with Path(r["manifest_path"]).open() as f:
   for line in f:
    candidate=json.loads(line)
    if candidate["observation_clock"]["end_s"]-candidate["observation_clock"]["start_s"]>=6 and candidate["split"]=="train":
     row=candidate;break
  if row is None:raise ValueError("no actual full-horizon episode")
  keys=("source","episode_id","payload","payload_binding","payload_row_start","payload_row_stop",
        "assets","views","task_text","embodiment","split","observation_samples","observation_clock","robot_groups","admission_id")
  task={k:row[k] for k in keys};task.update(schema=ADMITTED_DIRECT_TASK_SCHEMA,task_id=f"task{ordinal:09d}",
      origin="admitted",inherited_record=None)
  directory=a.output/task["source"];directory.mkdir(parents=True,exist_ok=True)
  tp=directory/"task.jsonl"
  with tp.open("x") as f:f.write(json.dumps(task)+"\n")
  source=SourceSpec(name=row["source"],adapter="lerobot",raw_root=Path(r["raw_root"]),
    adapter_config_path=Path(r["adapter_contract_path"]),adapter_contract_sha256="",
    manifest_path=Path(r["manifest_path"]),manifest_sha256="",embodiment=emb.name,weight=1,
    nominal_hours=None,license_id="upstream_preserved",manifest_binding=r["manifest_binding"],
    adapter_contract_binding=r["adapter_contract_binding"],admission_id=admission,origin="admitted")
  profile=DataProfile(path=rp,profile_sha256="",name="actual_admitted_window_replay",sources=(source,),
    embodiments={emb.name:emb},cache_representation=legacy["cache_representation"],cache=legacy["cache"],
    admission_id=admission)
  closure=dict(schema=ADMITTED_DIRECT_RAW_DATA_CLOSURE_SCHEMA,foundation_text_only=True,
    task_manifest_path=str(tp),task_manifest_binding=bind(tp,admission),
    initial_raw_asset_repairs=None,direct_input_rgb_size=518,direct_decode_workers=3,direct_robot_cache_episodes=2)
  reader=_DirectWindowSource(closure=closure,profile=profile);t=reader._load_task(task["task_id"])
  episode=reader._load_episode(t);clock=episode.prepared_robot.observation_times_s[episode.selected_rows]
  limits=GroupedRobotLimits(max_groups=m["max_action_groups"],max_substeps=m["max_action_substeps"],
    max_action_dim=m["max_action_dim"],max_state_dim=m["max_state_dim"])
  for anchor in range(len(clock)):
   try:
    w=select_observed_world_window(clock,anchor_index=anchor,context_samples=m["T"],future_samples=m["K"],
      context_horizon_s=sampling["context_horizon_seconds"],future_horizon_s=sampling["future_horizon_seconds"],
      minimum_horizon_coverage=sampling["minimum_horizon_coverage"],future_offsets_s=sampling.get("future_offsets_seconds"))
    boundary=episode.selected_rows[[w.leading_boundary_index,*w.context_indices,*w.future_indices]]
    fields=assemble_robot_window_from_prepared_episode(prepared=episode.prepared_robot,embodiment=emb,
      selected_source_boundary_indices=boundary.tolist(),limits=limits,context_samples=m["T"],
      max_policy_queries=m["max_policy_queries"],policy_target_horizon_s=sampling["policy_target_horizon_seconds"])
    break
   except (WindowSelectionError,ValueError,RuntimeError):continue
  else:raise RuntimeError("actual episode has no valid window")
  entry=AdmittedWindowEntry(sample_id="actual-replay-"+task["task_id"],source=source.name,split=t.split,
    embodiment=emb.name,task_id=t.task_id,leading_feature_row=int(w.leading_boundary_index),
    context_feature_rows=tuple(map(int,w.context_indices)),future_feature_rows=tuple(map(int,w.future_indices)))
  images,mask,framekeys,actualboundary,prepared=reader.decode_window(entry=entry,task_id=t.task_id)
  if not torch.equal(actualboundary,torch.from_numpy(boundary)):raise RuntimeError("decode/robot row mismatch")
  if images.shape[:3]!=(24,3,3) or not bool(mask.any()):raise RuntimeError("actual RGB window incomplete")
  batch={k:v.unsqueeze(0) for k,v in fields.items() if torch.is_tensor(v)}
  lane=prepared.action_series[0].supervision;cfg=FoundationBridgeConfig()
  packed=pack_grouped_effect_sequence(batch,cfg) if lane=="coarse_effect" else pack_grouped_future_action(batch,cfg)
  if not bool(torch.isfinite(packed["action"]).all()):raise RuntimeError("nonfinite packed targets")
  result=dict(source=source.name,episode=t.episode_id,source_boundary_rows=boundary.tolist(),RGB_shape=list(images.shape),
    view_valid_counts=mask.sum(0).tolist(),action_supervision_kind=lane,target_shape=list(packed["action"].shape),
    command_lane_empty=not bool(fields["future_factual_fine_action_mask"].any()),
    actual_RGB_decoded=True,raw_targets_not_normalized=True,SHA_scans=False,GPU_used=False)
  publish(directory/"receipt.json",result);results.append(result);print(json.dumps(result),flush=True)
 publish(a.output/"COMPLETE.json",dict(passed=True,results=results,scope="actual direct RGB/target bridge, not GPU qualification"))
if __name__=="__main__":main()
