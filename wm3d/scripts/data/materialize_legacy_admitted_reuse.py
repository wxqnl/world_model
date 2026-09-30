"""Reuse the selected original corpus/index/normalizer; no RGB or index hashes."""
import argparse,json,threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime
from pathlib import Path
import yaml
from wm3d.data.file_binding import verify_stat_asset
from wm3d.data.admitted_direct_tasks import ADMITTED_DIRECT_TASK_SCHEMA,APPROVED_REPAIR_STAT_KIND,admitted_task_from_mapping
from wm3d.data.manifest_contract import ADMITTED_CACHE_INDEX_SCHEMA
from scripts.data.admit_agibot_native import bind,publish

def main():
 p=argparse.ArgumentParser();p.add_argument("--inheritance",type=Path,required=True)
 p.add_argument("--repair",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--workers",type=int,default=32);a=p.parse_args()
 evidence=json.loads(a.inheritance.read_text());admission=evidence["admission_id"];inherit=evidence["inherited"]
 for name in ("profile","metadata_seal","selection"):
  path=Path(inherit[name+"_path"]);verify_stat_asset(path.parent,path.name,inherit[name+"_binding"])
 old=yaml.safe_load(Path(inherit["profile_path"]).read_text());seal=json.loads(Path(inherit["metadata_seal_path"]).read_text())
 selection=json.loads(Path(inherit["selection_path"]).read_text())
 if selection["schema"]!="v7_training_episode_selection_v1":raise ValueError("wrong saved selection")
 allowed={s:set(v) for s,v in selection["episodes"].items()}
 roots={s["name"]:Path(s["raw_root"]) for s in evidence["sources"]}
 repairs=json.loads(a.repair.read_text());repair_index={(x["source_root"],x["relative_path"]):x for x in repairs["assets"]}
 for s in evidence["sources"]:
  if (seal["source_manifest_sha256_by_name"][s["name"]]!=s["manifest_sha256"]
      or seal["adapter_contract_sha256_by_name"][s["name"]]!=s["adapter_contract_sha256"]):
   raise ValueError("legacy source lineage differs")
 a.output.mkdir(parents=True,exist_ok=True)
 files={};lock=threading.Lock()
 def once(path):
  key=str(path)
  with lock:cached=files.get(key)
  if cached is not None:return cached
  value=bind(path,admission)
  with lock:
   existing=files.setdefault(key,value)
   if existing!=value:raise RuntimeError("legacy file changed during admission")
  return value
 def convert(pair):
  ordinal,r=pair;root=roots[r["source"]];assets=[]
  for asset in r["assets"]:
   path=root/asset["path"];b=once(path);repair=repair_index.get((str(root),asset["path"]))
   if repair:
    if repair["sealed_digest"]!=asset["sha256"]:raise ValueError("legacy repair original differs")
    b=dict(kind=APPROVED_REPAIR_STAT_KIND,admission_id=admission,original=b,
      replacement=once(root/repair["replacement_relative_path"]),repair_asset=repair)
   assets.append(dict(role=asset["role"],path=asset["path"],binding=b))
  keys=("source","episode_id","payload","payload_row_start","payload_row_stop","views","task_text",
        "embodiment","split","observation_samples","observation_clock","robot_groups")
  result={k:r[k] for k in keys}
  result.update(schema=ADMITTED_DIRECT_TASK_SCHEMA,task_id=f"task{ordinal:09d}",admission_id=admission,
      payload_binding=once(root/r["payload"]),assets=assets,origin="inherited_legacy",inherited_record=r)
  admitted_task_from_mapping(result)
  return result
 manifest=Path(seal["task_manifest_path"]);mb=bind(manifest,admission)
 oldid_to_new={};identity={};seen={s:set() for s in roots};taskcounts=Counter();ordinal=0
 tp=a.output/"tasks.jsonl"
 with ThreadPoolExecutor(max_workers=a.workers) as pool,tp.open("x") as out,manifest.open() as inp:
  batch=[]
  def emit(items):
   for r in pool.map(convert,items):
    oid=r["inherited_record"]["task_id"];oldid_to_new[oid]=r["task_id"]
    identity[(r["source"],r["episode_id"])]=r["task_id"];taskcounts[r["source"]]+=1
    out.write(json.dumps(r,separators=(",",":"),ensure_ascii=False)+"\n")
  for line in inp:
   r=json.loads(line)
   if r["split"]=="train" and r["episode_id"] not in allowed[r["source"]]:continue
   if r["split"]=="train":seen[r["source"]].add(r["episode_id"])
   batch.append((ordinal,r));ordinal+=1
   if len(batch)==512:
    emit(batch);batch=[]
    if ordinal%51200==0:print(json.dumps(dict(stage="legacy_selected_tasks",tasks=ordinal,unique_assets=len(files))),flush=True)
  if batch:emit(batch)
 verify_stat_asset(manifest.parent,manifest.name,mb)
 if seen!=allowed:raise RuntimeError("legacy selected episode set changed")
 episode=Path(seal["episode_index_path"]);eb=bind(episode,admission);feature={}
 with episode.open() as f:
  for line in f:
   r=json.loads(line);key=(r["source"],r["episode_id"])
   if key in identity:
    tid=identity[key];oldid=r["feature_shard"].split("/")[2]
    if oldid_to_new.get(oldid)!=tid:raise RuntimeError("legacy feature/task identity changed")
    feature[r["feature_shard"]]=(tid,r["source"],r["split"],r["embodiment"])
 verify_stat_asset(episode.parent,episode.name,eb)
 wp=Path(seal["window_index_path"]);wb=bind(wp,admission)
 counts={k:Counter() for k in ("train","val","test")};skipped=Counter();n=0
 with wp.open() as inp,(a.output/"windows.jsonl").open("x") as out:
  for line in inp:
   r=json.loads(line);item=feature.get(r["feature_shard"])
   if item is None:
    if r["split"]!="train":raise RuntimeError("legacy validation/test was dropped")
    skipped[r["source"]]+=1;continue
   tid,source,split,emb=item
   if (r["source"],r["split"],r["embodiment"])!=(source,split,emb):raise RuntimeError("legacy window metadata changed")
   result={k:r[k] for k in ("sample_id","source","split","embodiment","leading_feature_row","context_feature_rows","future_feature_rows")}
   result.update(schema=ADMITTED_CACHE_INDEX_SCHEMA,task_id=tid)
   out.write(json.dumps(result,separators=(",",":"))+"\n");counts[split][source]+=1;n+=1
   if n%500000==0:print(json.dumps(dict(stage="legacy_selected_windows",windows=n)),flush=True)
 verify_stat_asset(wp.parent,wp.name,wb)
 if dict(counts["train"])!=selection["expected_windows"] or sum(counts["train"].values())!=4922451:
  raise RuntimeError("original train window count changed")
 if sum(counts["val"].values())!=127991 or any(n for s,n in skipped.items() if s.startswith("oxe_")):
  raise RuntimeError("original validation/OXE selection changed")
 def verify_item(item):
  path,binding=item;path=Path(path);verify_stat_asset(path.parent,path.name,binding)
 with ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(verify_item,files.items()))
 publish(a.output/"COMPLETE.json",dict(passed=True,completed_at=datetime.now().astimezone().isoformat(),
  inherited=evidence,source_metadata_bindings={str(manifest):mb,str(episode):eb,str(wp):wb},
  task_count_by_source=dict(taskcounts),task_count=ordinal,window_count_by_split_source={k:dict(v) for k,v in counts.items()},
  task_manifest=dict(path=str(tp),binding=bind(tp,admission)),
  window_index=dict(path=str(a.output/"windows.jsonl"),binding=bind(a.output/"windows.jsonl",admission)),
  next_task_ordinal=ordinal,unique_asset_count=len(files),original_window_rows_unchanged=True,
  original_normalization_unchanged=True,raw_video_read=False,SHA_scans=False))
 print(json.dumps(dict(stage="legacy_reuse_complete",tasks=ordinal,windows=n)),flush=True)
if __name__=="__main__":main()
