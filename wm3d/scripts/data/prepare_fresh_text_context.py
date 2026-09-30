"""Extract only previously unseen exact instructions from admitted source manifests."""
import argparse,json
from pathlib import Path
from wm3d.data.file_binding import verify_stat_asset
from wm3d.data.manifest_contract import load_data_profile
from scripts.data.admit_agibot_native import bind,publish
def main():
 p=argparse.ArgumentParser();p.add_argument("--profile",type=Path,required=True)
 p.add_argument("--base",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 admission="wm3d_fresh_new_data_20260930";pb=bind(a.profile,admission)
 profile=load_data_profile(a.profile,expected_binding=pb,verify_source_manifests=False)
 ib=bind(a.base/"index.json",admission);old=json.loads((a.base/"index.json").read_text())
 verify_stat_asset(a.base,"index.json",ib)
 if old.get("schema")!="wm3d_foundation_context_store_v1":raise ValueError("wrong base context")
 existing=set(old["tasks"]);texts=set()
 for source in profile.sources:
  if source.origin!="admitted":continue
  verify_stat_asset(source.manifest_path.parent,source.manifest_path.name,source.manifest_binding)
  with source.manifest_path.open() as f:
   for line in f:
    text=json.loads(line)["task_text"].strip()
    if not text:raise ValueError("empty actual task")
    texts.add(text)
  verify_stat_asset(source.manifest_path.parent,source.manifest_path.name,source.manifest_binding)
 a.output.mkdir(parents=True,exist_ok=True)
 with (a.output/"new_texts.jsonl").open("x") as f:
  for t in sorted(texts-existing):f.write(json.dumps(dict(text=t),ensure_ascii=False)+"\n")
 publish(a.output/"PLAN.json",dict(admission_id=admission,new_unique=len(texts-existing),
  reused_unique=len(texts&existing),base_unique=len(existing),source_texts=len(texts),SHA_scans=False,
  base=dict(root=str(a.base),index_binding=ib,tensor_binding=bind(a.base/"context.safetensors",admission)),
  new_text_file=str(a.output/"new_texts.jsonl"),profile=dict(path=str(a.profile),binding=pb)))
if __name__=="__main__":main()
