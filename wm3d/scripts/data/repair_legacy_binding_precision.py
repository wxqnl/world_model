"""Correct JSON-number precision lost in a previously transported metadata receipt."""
import argparse,json
from pathlib import Path
from datetime import datetime
from scripts.data.admit_agibot_native import bind,publish

def main():
 p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 r=json.loads(a.input.read_text());admission=r["admission_id"];changes=[]
 def replace(path,binding):
  actual=bind(Path(path),admission)
  if (actual["kind"]!=binding["kind"] or actual["admission_id"]!=binding["admission_id"]
      or actual["size_bytes"]!=binding["size_bytes"] or actual["inode"]!=binding["inode"]
      or abs(actual["mtime_ns"]-binding["mtime_ns"])>256
      or float(actual["mtime_ns"])!=float(binding["mtime_ns"])):
   raise RuntimeError("difference is not solely the established JSON float precision loss")
  changes.append(dict(path=str(path),recorded_mtime_ns=binding["mtime_ns"],actual_mtime_ns=actual["mtime_ns"]))
  return actual
 for name in ("profile","metadata_seal","selection"):
  r["inherited"][name+"_binding"]=replace(r["inherited"][name+"_path"],r["inherited"][name+"_binding"])
 for s in r["sources"]:
  for name,binding in (("manifest","manifest_binding"),("adapter_config","adapter_contract_binding")):
   s[binding]=replace(s[name],s[binding])
 r["checked_at"]=datetime.now().astimezone().isoformat()
 publish(a.output,r)
 publish(a.output.with_suffix(".precision_receipt.json"),dict(passed=True,original_receipt=str(a.input),
   corrected_receipt=str(a.output),changes=changes,original_files_unchanged=True,
   validation_not_relaxed=True,SHA_scans=False,
   cause="Nanosecond integers were transported through IEEE754 JSON numbers. Correct receipt is emitted directly by Python integers."))
 print(json.dumps(dict(passed=True,corrected_metadata_bindings=len(changes),old_receipt_preserved=True)),flush=True)
if __name__=="__main__":main()
