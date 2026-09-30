"""Materialize new-source window metadata and train-only statistics, never RGB caches."""
import argparse,json,os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import numpy as np
import yaml
from wm3d.data.file_binding import verify_stat_asset
from wm3d.data.manifest_contract import load_data_profile,ADMITTED_CACHE_INDEX_SCHEMA,timestamp_evidence
from wm3d.data.admitted_direct_tasks import ADMITTED_DIRECT_TASK_SCHEMA,admitted_task_from_mapping
from wm3d.data.episode_io import VerifiedAssetStore,open_admitted_episode_accessor,select_episode_cache_rows
from wm3d.data.episode_robot import PreparedEpisodeRobot,assemble_robot_window_from_prepared_episode
from wm3d.data.source_adapters import load_adapter_contract,adapt_action_series,adapt_state_series
from wm3d.data.grouped_robot import GroupedRobotLimits
from wm3d.data.window_selection import select_observed_world_window,WindowSelectionError
from wm3d.data.grouped_normalization import (_RunningMoments,accumulate_window_moments,normalization_rows,
 ADMITTED_NORMALIZATION_SCHEMA,GROUPED_NORMALIZATION_ESTIMATOR,GROUPED_NORMALIZATION_SPLIT)
from scripts.data.admit_agibot_native import publish,bind

PROFILE=None;MODEL=None;ADAPTERS=None;STORE=None

def initialize(profile_path,profile_binding,model):
 global PROFILE,MODEL,ADAPTERS,STORE
 PROFILE=load_data_profile(Path(profile_path),expected_binding=profile_binding,verify_source_manifests=False)
 MODEL=model
 ADAPTERS={s.name:load_adapter_contract(s.adapter_config_path,expected_binding=s.adapter_contract_binding)
           for s in PROFILE.sources if s.origin=="admitted"}
 STORE=VerifiedAssetStore()

def process(job):
 number,rows,root=job;root=Path(root);out=root/"parts"/f"{number:05d}";out.mkdir(parents=True,exist_ok=True)
 model=MODEL["model"];sampling=MODEL["sampling"];admission=PROFILE.admission_id
 sources={s.name:s for s in PROFILE.sources}
 limits=GroupedRobotLimits(max_groups=model["max_action_groups"],max_substeps=model["max_action_substeps"],
    max_action_dim=model["max_action_dim"],max_state_dim=model["max_state_dim"])
 counts={s:Counter() for s in ("train","val","test")};moments={};lanes={};rejected=Counter();used=Counter()
 with (out/"tasks.jsonl").open("x") as tf,(out/"windows.jsonl").open("x") as wf:
  for ordinal,row in rows:
   source=sources[row["source"]];adapter=ADAPTERS[source.name];emb=PROFILE.embodiments[source.embodiment]
   for asset in row["assets"]:STORE.verify_admitted(source.raw_root,asset["path"],asset["binding"])
   accessor=open_admitted_episode_accessor(source_root=source.raw_root,payload=row["payload"],
     payload_binding=row["payload_binding"],row_start=row["payload_row_start"],row_stop=row["payload_row_stop"],
     adapter=adapter,asset_verifier=STORE)
   clock=np.asarray(accessor.array(adapter.observation_time_key),dtype=np.float64).reshape(-1)
   if timestamp_evidence(key=adapter.observation_time_key,values=clock,admitted_payload=True)!=row["observation_clock"]:
    raise RuntimeError("admitted numeric clock changed")
   prepared=PreparedEpisodeRobot(observation_times_s=clock,action_series=tuple(adapt_action_series(
     accessor=accessor,contract=adapter,embodiment=emb)),state_series=tuple(adapt_state_series(
     accessor=accessor,contract=adapter,embodiment=emb)),task_embedding=None)
   selected=select_episode_cache_rows(clock,minimum_separation_s=PROFILE.cache_representation["state_frame_selection"]["minimum_separation_seconds"])
   selected=np.asarray(selected,dtype=np.int64);sample_clock=clock[selected];taskid=f"task{ordinal:09d}"
   task={k:row[k] for k in ("source","episode_id","payload","payload_binding","payload_row_start",
      "payload_row_stop","assets","views","task_text","embodiment","split","observation_samples","observation_clock","robot_groups","admission_id")}
   task.update(schema=ADMITTED_DIRECT_TASK_SCHEMA,task_id=taskid,origin="admitted",inherited_record=None)
   admitted_task_from_mapping(task);tf.write(json.dumps(task,separators=(",",":"),ensure_ascii=False)+"\n")
   last=-np.inf;n=0
   for anchor in range(len(sample_clock)):
    if sample_clock[anchor]-last+1.e-12<sampling["minimum_anchor_separation_seconds"]:continue
    try:
     w=select_observed_world_window(sample_clock,anchor_index=anchor,context_samples=model["T"],future_samples=model["K"],
       context_horizon_s=sampling["context_horizon_seconds"],future_horizon_s=sampling["future_horizon_seconds"],
       minimum_horizon_coverage=sampling["minimum_horizon_coverage"],future_offsets_s=sampling.get("future_offsets_seconds"))
    except WindowSelectionError:
     rejected["visual_boundary_coverage"]+=1;continue
    boundaries=selected[[w.leading_boundary_index,*w.context_indices,*w.future_indices]]
    assembled=assemble_robot_window_from_prepared_episode(prepared=prepared,embodiment=emb,
      selected_source_boundary_indices=boundaries.tolist(),limits=limits,context_samples=model["T"],
      max_policy_queries=model["max_policy_queries"],policy_target_horizon_s=sampling["policy_target_horizon_seconds"])
    entry=dict(schema=ADMITTED_CACHE_INDEX_SCHEMA,sample_id="admitted:"+taskid+":"+str(anchor),
      source=source.name,split=row["split"],embodiment=emb.name,task_id=taskid,
      leading_feature_row=int(w.leading_boundary_index),context_feature_rows=w.context_indices.tolist(),
      future_feature_rows=w.future_indices.tolist())
    wf.write(json.dumps(entry,separators=(",",":"))+"\n");counts[row["split"]][source.name]+=1;n+=1
    if row["split"]=="train":
     accumulate_window_moments(assembled=assembled,prepared=prepared,embodiment=emb,
       source_name=source.name,moments=moments,lane_by_group=lanes)
    last=sample_clock[anchor]
   used["episodes_with_windows" if n else "episodes_shorter_than_required_window"]+=1
   for asset in row["assets"]:STORE.verify_admitted(source.raw_root,asset["path"],asset["binding"])
 result=dict(passed=True,part=number,tasks=len(rows),counts={k:dict(v) for k,v in counts.items()},
    moments=[dict(key=list(k),stat=asdict(v)) for k,v in moments.items()],
    lanes=[dict(key=list(k),value=v) for k,v in lanes.items()],rejected=dict(rejected),episode_usage=dict(used),
    files={name:dict(path=str(out/name),binding=bind(out/name,admission)) for name in ("tasks.jsonl","windows.jsonl") if (out/name).stat().st_size},
    empty_window_part=(out/"windows.jsonl").stat().st_size==0,
    SHA_scans=False,RGB_decode=False)
 publish(out/"receipt.json",result)
 return str(out/"receipt.json")

def merge_moment(target,other):
 n=other["count"]
 if not n:return
 if not target.count:
  target.count=n;target.mean=other["mean"];target.m2=other["m2"]
  target.minimum=other["minimum"];target.maximum=other["maximum"];return
 total=target.count+n;delta=other["mean"]-target.mean
 target.m2+=other["m2"]+delta*delta*target.count*n/total
 target.mean+=delta*n/total;target.count=total
 target.minimum=min(target.minimum,other["minimum"]);target.maximum=max(target.maximum,other["maximum"])

def main():
 p=argparse.ArgumentParser();p.add_argument("--profile",type=Path,required=True);p.add_argument("--model",type=Path,required=True)
 p.add_argument("--legacy",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--workers",type=int,default=32);a=p.parse_args();admission="wm3d_fresh_new_data_20260930"
 pb=bind(a.profile,admission);profile=load_data_profile(a.profile,expected_binding=pb,verify_source_manifests=False)
 model=yaml.safe_load(a.model.read_text());legacy=json.loads((a.legacy/"COMPLETE.json").read_text())
 if legacy.get("passed") is not True:raise RuntimeError("old corpus reuse incomplete")
 a.output.mkdir(parents=True,exist_ok=True);jobs=[];batch=[];ordinal=legacy["next_task_ordinal"];first=ordinal
 source_counts=Counter();texts=set()
 for source in profile.sources:
  if source.origin!="admitted":continue
  verify_stat_asset(source.manifest_path.parent,source.manifest_path.name,source.manifest_binding)
  with source.manifest_path.open() as f:
   for line in f:
    row=json.loads(line);source_counts[source.name]+=1;texts.add(row["task_text"])
    batch.append((ordinal,row));ordinal+=1
    if len(batch)==128:jobs.append((len(jobs),batch,str(a.output)));batch=[]
  verify_stat_asset(source.manifest_path.parent,source.manifest_path.name,source.manifest_binding)
 if batch:jobs.append((len(jobs),batch,str(a.output)))
 with (a.output/"new_task_texts.jsonl").open("x") as f:
  for text in sorted(texts):f.write(json.dumps(dict(text=text),ensure_ascii=False)+"\n")
 publish(a.output/"PLAN.json",dict(admission_id=admission,new_episodes=sum(source_counts.values()),
   source_episode_counts=dict(source_counts),parts=len(jobs),first_task_ordinal=first,next_task_ordinal=ordinal,
   profile=dict(path=str(a.profile),binding=pb),model_profile=model,RGB_decode=False,SHA_scans=False))
 receipts=[]
 with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(str(a.profile),pb,model)) as pool:
  for future in as_completed([pool.submit(process,j) for j in jobs]):
   receipts.append(future.result())
   if len(receipts)%20==0 or len(receipts)==len(jobs):
    print(json.dumps(dict(stage="new_windows_and_train_statistics",parts=len(receipts),total=len(jobs))),flush=True)
 moments={};lanes={};counts={k:Counter() for k in ("train","val","test")};rejected=Counter();usage=Counter()
 for rp in sorted(receipts):
  r=json.loads(Path(rp).read_text())
  for split,items in r["counts"].items():counts[split].update(items)
  for item in r["moments"]:merge_moment(moments.setdefault(tuple(item["key"]),_RunningMoments()),item["stat"])
  for item in r["lanes"]:
   key=tuple(item["key"]);previous=lanes.setdefault(key,item["value"])
   if previous!=item["value"]:raise RuntimeError("source/group supervision changes")
  rejected.update(r["rejected"]);usage.update(r["episode_usage"])
 rows=normalization_rows(data_profile=profile,moments=moments,lane_by_group=lanes,
   source_names=[s.name for s in profile.sources if s.origin=="admitted"])
 publish(a.output/"new_normalization_rows.json",dict(rows=rows,counts={k:dict(v) for k,v in counts.items()},
    estimator=GROUPED_NORMALIZATION_ESTIMATOR,split=GROUPED_NORMALIZATION_SPLIT))
 publish(a.output/"COMPLETE.json",dict(passed=True,completed_at=datetime.now().astimezone().isoformat(),
   parts=sorted(receipts),window_count_by_split_source={k:dict(v) for k,v in counts.items()},
   task_count_by_source=dict(source_counts),next_task_ordinal=ordinal,rejected_boundaries=dict(rejected),
   episode_usage=dict(usage),profile=dict(path=str(a.profile),binding=pb),model_profile=model,
   new_normalization_rows=dict(path=str(a.output/"new_normalization_rows.json"),
    binding=bind(a.output/"new_normalization_rows.json",admission)),RGB_decode=False,SHA_scans=False))
 print(json.dumps(dict(stage="new_window_materialization_complete",windows=sum(sum(x.values()) for x in counts.values()))),flush=True)
if __name__=="__main__":main()
