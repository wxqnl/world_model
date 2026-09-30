"""Publish the actual combined corpus, inherited normalization, and two-epoch run."""
import argparse,json,os,shutil
from collections import Counter
from pathlib import Path
import yaml
from scripts.data.admit_agibot_native import bind,publish
from wm3d.data.file_binding import verify_stat_asset
from wm3d.data.manifest_contract import load_data_profile,canonical_sha256
from wm3d.data.grouped_normalization import ADMITTED_NORMALIZATION_SCHEMA,GroupedRobotNormalizer
from wm3d.data.admitted_closure import SCHEMA,SEAL_SCHEMA,seal_contract,validate_admitted_closure
from wm3d.training.runtime_contract import RUNTIME_CONFIG_SCHEMA,validate_materialized_runtime

def append_file(target,item,admission):
 p=Path(item["path"]);verify_stat_asset(p.parent,p.name,item["binding"])
 if item["binding"]["admission_id"]!=admission:raise ValueError("part admission changed")
 with p.open("rb") as f:shutil.copyfileobj(f,target,8<<20)
 verify_stat_asset(p.parent,p.name,item["binding"])

def main():
 p=argparse.ArgumentParser();p.add_argument("--new-root",type=Path,required=True)
 p.add_argument("--windows-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--code-commit",required=True);a=p.parse_args()
 n=a.new_root;admission="wm3d_fresh_new_data_20260930";combined=n/"combined_profile_20261001"
 profilepath=combined/"data_profile.yaml";pb=bind(profilepath,admission)
 profile=load_data_profile(profilepath,expected_binding=pb,verify_source_manifests=False)
 model=json.loads((combined/"model.json").read_text())
 old=json.loads((n/"legacy_reuse_cpu_r2_20261001/result/COMPLETE.json").read_text())
 new=json.loads((a.windows_root/"COMPLETE.json").read_text())
 if old.get("passed") is not True or new.get("passed") is not True:raise ValueError("window materialization incomplete")
 if new["profile"]!={"path":str(profilepath),"binding":pb} or new["model_profile"]!=model:
  raise ValueError("new windows profile/model changed")
 # Reuse the old tensor store; add the freshly encoded disjoint instruction set.
 textplan=json.loads((n/"text_preparation_20261001/PLAN.json").read_text())
 textroot=n/"text_encode_r2_20261001/new_context"
 newtext=json.loads((textroot/"index.json").read_text())
 with Path(textplan["new_text_file"]).open() as f:expected=sorted({json.loads(x)["text"] for x in f})
 if newtext.get("schema")!="wm3d_foundation_context_store_v1" or newtext["tasks"]!=expected:
  raise ValueError("encoded instruction inventory differs")
 for filename,key in (("index.json","index_binding"),("context.safetensors","tensor_binding")):
  verify_stat_asset(Path(textplan["base"]["root"]),filename,textplan["base"][key])
 from safetensors import safe_open
 with safe_open(str(textroot/"context.safetensors"),framework="pt",device="cpu") as st:
  if (set(st.keys())!={"context","mask"} or st.get_slice("context").get_shape()!=[len(expected),128,4096]
      or st.get_slice("mask").get_shape()!=[len(expected),128]):raise ValueError("actual text encoding dimensions differ")
 publish(n/"foundation_context/index.json",dict(schema="wm3d_admitted_foundation_context_union_v1",
  admission_id=admission,parts=[textplan["base"],dict(root=str(textroot),
   index_binding=bind(textroot/"index.json",admission),tensor_binding=bind(textroot/"context.safetensors",admission))]))
 a.output.mkdir(parents=True,exist_ok=True)
 tasks=a.output/"tasks.jsonl";windows=a.output/"windows.jsonl";seen=set();newcounts={k:Counter() for k in ("train","val","test")}
 with tasks.open("xb") as tf,windows.open("xb") as wf:
  append_file(tf,old["task_manifest"],admission);append_file(wf,old["window_index"],admission)
  task_total=old["task_count"]
  for rp in new["parts"]:
   if rp in seen:raise ValueError("repeated new part")
   seen.add(rp);r=json.loads(Path(rp).read_text())
   if r.get("passed") is not True:raise ValueError("failed new part")
   append_file(tf,r["files"]["tasks.jsonl"],admission);task_total+=r["tasks"]
   if not r["empty_window_part"]:append_file(wf,r["files"]["windows.jsonl"],admission)
   elif any(sum(x.values()) for x in r["counts"].values()):raise ValueError("empty part has nonzero counts")
   for split,x in r["counts"].items():newcounts[split].update(x)
 if {k:dict(v) for k,v in newcounts.items()}!=new["window_count_by_split_source"]:
  raise ValueError("new part counts differ from completion")
 counts={split:{name:old["window_count_by_split_source"][split].get(name,0)+newcounts[split].get(name,0)
                for name in profile.source_order} for split in ("train","val","test")}
 taskcounts={name:old["task_count_by_source"].get(name,0)+new["task_count_by_source"].get(name,0) for name in profile.source_order}
 if task_total!=sum(taskcounts.values()) or sum(old["window_count_by_split_source"]["train"].values())!=4922451:
  raise ValueError("original selection or tasks changed")
 wb=bind(windows,admission);tb=bind(tasks,admission)
 normalpath=Path("/data/Minko/wm3d_full_robot_data_20260904/grouped_normalization.json")
 oldnormal=json.loads(normalpath.read_text());np=Path(new["new_normalization_rows"]["path"])
 verify_stat_asset(np.parent,np.name,new["new_normalization_rows"]["binding"]);newnormal=json.loads(np.read_text())
 if newnormal["counts"]!=new["window_count_by_split_source"]:raise ValueError("normalization counts differ")
 normal={k:v for k,v in oldnormal.items() if k not in {"schema","data_profile_sha256","model_profile_sha256","window_index_sha256","rows_sha256"}}
 normal.update(schema=ADMITTED_NORMALIZATION_SCHEMA,admission_id=admission,data_profile_path=str(profilepath),
  data_profile_binding=pb,model_profile=model,window_index_path=str(windows),window_index_binding=wb,
  train_window_count_by_source=counts["train"],rows=oldnormal["rows"]+newnormal["rows"],
  inherited_normalization=dict(path=str(normalpath),binding=bind(normalpath,admission),
   sources=[s.name for s in profile.sources if s.origin=="inherited_legacy"]))
 normalfile=a.output/"grouped_normalization.json";publish(normalfile,normal)
 oldruntime=yaml.safe_load(Path("/data/Minko/wm3d_native_1000h_formal_2node16_micro6_step50000_20260924/maintenance/migration24_step35000_20260928/runtime24.yaml").read_text())
 oldc=oldruntime["data_closure"]
 encoder=a.output/"vggt_encoder.yaml";os.link(oldc["encoder_contract_path"],encoder)
 env=a.output/"environment_receipt.json";os.link(oldruntime["run"]["environment_lock_path"],env)
 runtime=yaml.safe_load((n/"candidates_three_node_24/runtime.candidate.yaml").read_text())
 train=runtime["train"];train.pop("train_episode_allowlist")
 train["admitted_selection"]=wb;train["total_steps"]=2*sum((x+95)//96 for x in counts["train"].values())
 if runtime["expected_world_size"]!=24 or train["global_batch_size"]!=96 or train["sampling_epochs"]!=2:
  raise ValueError("authorized topology or epoch budget changed")
 closure=dict(schema=SCHEMA,name="base1000_and_frozen_B1_B2_B3_stat_direct",
  admission_id=admission,metadata_root=str(a.output),foundation_text_only=True,
  initial_raw_asset_repairs=train["initial_raw_asset_repairs"],
  source_manifest_binding_by_name={s.name:s.manifest_binding for s in profile.sources},
  adapter_contract_binding_by_name={s.name:s.adapter_contract_binding for s in profile.sources},
  source_action_supervision={})
 for source in profile.sources:
  lanes={r["lane"] for r in normal["rows"] if r["source"]==source.name and r["kind"]=="action"}
  if len(lanes)!=1:raise ValueError("mixed or missing action supervision")
  closure["source_action_supervision"][source.name]=next(iter(lanes))
 for key in ("direct_input_rgb_size","direct_decode_workers","direct_robot_cache_episodes","direct_prefetch_windows",
             "direct_video_index_cache_assets","direct_encode_chunk_rows","direct_minimum_chunk_rows",
             "appearance_token_grid","appearance_feature_layer","direct_ignored_action_dimensions"):
  closure[key]=oldc[key]
 for key,path in (("data_profile",profilepath),("cache_index",windows),("task_manifest",tasks),
                  ("grouped_normalization",normalfile),("encoder_contract",encoder)):
  closure[key+"_path"]=str(path);closure[key+"_binding"]=bind(path,admission)
 seal=a.output/"metadata_seal.json"
 publish(seal,dict(schema=SEAL_SCHEMA,admission_id=admission,contract=seal_contract(closure),model_profile=model,
  window_count_by_split_source=counts,task_count_by_source=taskcounts,
  selection=dict(mode="frozen_base_selection_plus_admitted_episodes",new_episodes_split_disjoint=True,
   inherited=profile.inherited,published_release_identity_not_global_recording_certification=True)))
 closure.update(metadata_seal_path=str(seal),metadata_seal_binding=bind(seal,admission))
 validate_admitted_closure(closure,model)
 obj=yaml.safe_load((n/"candidates_three_node_24/objective.candidate.yaml").read_text())
 output=n/"training";output.mkdir(exist_ok=True)
 cfg=dict(schema=RUNTIME_CONFIG_SCHEMA,
  run=dict(name="native_fresh_base1000_plus_B1_B2_B3_2epochs",lineage="native_fresh_20261001",
   output_root=str(output),code_commit=a.code_commit,environment_lock_path=str(env),environment_lock_binding=bind(env,admission)),
  model_profile=model,data_closure=closure,runtime_profile=runtime,objective_profile=obj)
 cfg["bindings"]={k+"_sha256":canonical_sha256(v) for k,v in
  (("model_profile",model),("data_closure",closure),("runtime_profile",runtime),("objective_profile",obj),
   ("model_contract",{"architecture":model["architecture"],"model":model["model"]}))}
 validate_materialized_runtime(cfg);publish(a.output/"runtime.json",cfg)
 publish(a.output/"COMPLETE.json",dict(passed=True,formal_training_started=False,source_checkpoint=None,
  epochs=2,total_steps=train["total_steps"],global_batch_size=96,world_size=24,task_count=task_total,
  window_count_by_split_source=counts,tail_repetitions_per_epoch={k:((v+95)//96)*96-v for k,v in counts["train"].items()},
  raw_video_redecoded=False,SHA_file_scans=False,old_normalization_rows_exact=True,
  runtime_path=str(a.output/"runtime.json"),new_window_receipt=str(a.windows_root/"COMPLETE.json")))
 print(json.dumps(dict(stage="fresh_runtime_complete",steps=train["total_steps"],train_windows=sum(counts["train"].values()))),flush=True)
if __name__=="__main__":main()
