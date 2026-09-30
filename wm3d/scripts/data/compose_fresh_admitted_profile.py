"""Compose the original 18-source profile with completed frozen admissions."""
import argparse,json,os
from pathlib import Path
from types import SimpleNamespace
import yaml
from wm3d.data.file_binding import verify_stat_asset
from wm3d.data.manifest_contract import validate_source_manifest
from scripts.data.admit_agibot_native import bind,publish,embodiment
from scripts.data.materialize_data_profile import _materialize_admitted

def main():
 p=argparse.ArgumentParser();p.add_argument("--new-root",type=Path,required=True)
 p.add_argument("--intern-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 base=json.loads((a.new_root/"LEGACY_STAT_BINDINGS_EXACT_NS_20261001.json").read_text())
 admission=base["admission_id"];old=yaml.safe_load(Path(base["inherited"]["profile_path"]).read_text())
 agi=a.new_root/"admitted/agibot_native_20261001"
 for cp in (agi/"COMPLETE.json",a.intern_root/"COMPLETE.json"):
  if json.loads(cp.read_text()).get("passed") is not True:raise RuntimeError("admission incomplete")
 rob=a.new_root/"admitted/robotwin_state_20261001";r=json.loads((rob/"receipt.json").read_text())
 audit=json.loads(Path(r["adapter_audit_receipt_path"]).read_text())
 cp=rob/"episode_catalog.json"
 if bind(cp,admission)!=audit["episode_catalog_binding"]:raise RuntimeError("RoboTwin catalog changed")
 template=yaml.safe_load(Path(r["data_template_path"]).read_text())
 validation=validate_source_manifest(Path(r["manifest_path"]),expected_source=r["source"],
     expected_embodiment=embodiment(template),admission_id=admission)
 r.update(episode_catalog_path=str(cp),episode_catalog_binding=bind(cp,admission),manifest_validation=validation,
          original_receipt_path=str(rob/"receipt.json"),original_receipt_binding=bind(rob/"receipt.json",admission))
 a.output.mkdir(parents=True,exist_ok=True)
 for field in ("data_template","adapter_contract"):
  source=Path(r[field+"_path"]);target=a.output/("robotwin_"+source.name)
  if not target.exists():os.link(source,target)
  verify_stat_asset(target.parent,target.name,r[field+"_binding"])
  r[field+"_path"]=str(target)
 repaired=a.output/"robotwin_receipt_complete.json";publish(repaired,r)
 receipts=[repaired,*sorted(agi.glob("agibot*/receipt.json")),*sorted(a.intern_root.glob("interndata*/receipt.json"))]
 if len(receipts)!=9:raise RuntimeError("frozen admission variants differ")
 value=dict(old);value["name"]="fresh_base1000_plus_frozen_B1_B2_B3_2epochs"
 value["sources"]=[dict(x) for x in base["sources"]]
 value["inherited"]=base["inherited"];value["embodiments"]=list(old["embodiments"])
 names={s["name"] for s in value["sources"]};embs={e["name"] for e in value["embodiments"]};args=[]
 for rp in receipts:
  r=json.loads(rp.read_text());path=Path(r["data_template_path"])
  verify_stat_asset(path.parent,path.name,r["data_template_binding"])
  t=yaml.safe_load(path.read_text());s=dict(t["sources"][0]);name=s["name"]
  if name in names:raise RuntimeError("duplicate new source")
  names.add(name)
  for e in t["embodiments"]:
   if e["name"] in embs:raise RuntimeError("new embodiment reused legacy name")
   embs.add(e["name"]);value["embodiments"].append(e)
  s.update(origin="admitted",adapter="lerobot",weight=1,
    license_id="CC-BY-NC-SA-4.0" if not name.startswith("robotwin") else "upstream_frozen_release_license")
  value["sources"].append(s);args.append(name+"="+str(rp))
 template_path=a.output/"combined.template.json";publish(template_path,value)
 _materialize_admitted(SimpleNamespace(admission_id=admission,inventory=args,
     output=a.output/"data_profile.yaml",receipt=a.output/"data_profile_receipt.json"),
     template_path,value,bind(template_path,admission))
 model=yaml.safe_load((a.new_root/"candidates_three_node_24/model.candidate.yaml").read_text())
 model["model"]["wan_context_store_path"]=str(a.new_root/"foundation_context")
 publish(a.output/"model.json",model)
 publish(a.output/"COMPLETE.json",dict(passed=True,sources=len(names),embodiments=len(embs),
   new_sources=9,legacy_sources=18,source_receipts=args,SHA_scans=False,formal_training_started=False))
if __name__=="__main__":main()
