#!/usr/bin/env python3
"""Build the full existing-source closure through the production data tools.

No downloads, RGB/VGGT feature caching, source reweighting or raw-data edits.
Requires the all-usable-episodes selection to have completed first.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import shutil
import sys
import yaml
from wm3d.data.source_inventory import (
    _episode_metadata, _task_lookup, _task_text, SourceInventoryError,
)

from wm3d.data.manifest_contract import sha256_file

def validate_selected_tasks(root, source):
    """Keep every selected episode with a real task; preserve rejected evidence."""
    path = root / "episode_indices" / (source["name"] + ".txt")
    selected = tuple(map(int, path.read_text().split()))
    wanted = set(selected)
    raw = Path(source["raw_root"])
    lookup = _task_lookup(raw)
    missing = set()
    found = set()
    for row in _episode_metadata(raw):
        index = int(row["episode_index"])
        if index not in wanted:
            continue
        found.add(index)
        try:
            _task_text(row, lookup, "")
        except SourceInventoryError:
            missing.add(index)
    if found != wanted:
        raise RuntimeError(f"{source['name']}: selected episode metadata is missing")
    if missing:
        keep = tuple(index for index in selected if index not in missing)
        if not keep:
            raise RuntimeError(f"{source['name']}: no real task annotations")
        backup = path.with_suffix(".before_task_filter.txt")
        if not backup.exists():
            shutil.copyfile(path, backup)
        temporary = path.with_suffix(".validated.tmp")
        temporary.write_text("".join(f"{index}\n" for index in keep))
        temporary.replace(path)
        evidence = root / "task_coverage"
        evidence.mkdir(exist_ok=True)
        (evidence / (source["name"] + ".json")).write_text(json.dumps({
            "selected_before": len(selected), "selected_after": len(keep),
            "excluded_missing_real_task": sorted(missing),
            "raw_data_modified": False, "source_weight_changed": False,
        }, indent=2) + "\n")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--selection-root",type=Path,required=True)
    parser.add_argument("--reference-root",type=Path,required=True)
    parser.add_argument("--model-profile",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--metadata-workers",type=int,default=16)
    parser.add_argument("--metadata-processes",type=int,default=1)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    code=Path(__file__).resolve().parents[2]
    root=args.selection_root.resolve(strict=True)
    template=root/"data_template.yaml"
    value=yaml.safe_load(template.read_text())
    reference=yaml.safe_load((args.reference_root/"data_template.yaml").read_text())
    if value["notes"].get("selection_mode")!="all_usable_episodes":
        raise RuntimeError("Formal closure cannot use minimum-window/canary selection")
    old_sources={s["name"]:s for s in reference["sources"]}
    old_emb={e["name"]:e for e in reference["embodiments"]}
    new_emb={e["name"]:e for e in value["embodiments"]}
    logs=root/"logs";logs.mkdir(exist_ok=True)
    def run(script,arguments,label):
        with (logs/(label+".log")).open("ab") as log:
            subprocess.run([sys.executable,str(code/"scripts"/script),*map(str,arguments)],
                           cwd=code,stdout=log,stderr=subprocess.STDOUT,check=True)
    def inventory(source):
        name=source["name"];old=old_sources.get(name)
        if old is None or old["weight"]!=source["weight"]:
            raise RuntimeError(f"{name}: source set/weight differs from existing contract")
        adapter=Path(source["adapter_config"])
        previous=Path(old["adapter_config"])
        if yaml.safe_load(adapter.read_text())!=yaml.safe_load(previous.read_text()):
            raise RuntimeError(f"{name}: action/state/camera adapter changed")
        if new_emb[source["embodiment"]]!=old_emb[old["embodiment"]]:
            raise RuntimeError(f"{name}: physical embodiment contract changed")
        old_audit=json.loads((args.reference_root/"audits"/(name+".adapter.json")).read_text())
        audit=root/"audits"/(name+".adapter.json")
        receipt=root/"inventory"/(name+".receipt.json")
        manifest=root/"manifests"/(name+".jsonl")
        if not audit.exists():
            run("data/audit_adapter_contract.py",[
                "--schema-audit",old_audit["schema_audit_path"],
                "--adapter-candidate",old_audit["adapter_candidate_path"],
                "--adapter-contract",adapter,"--adapter-contract-sha256",sha256_file(adapter),
                "--data-template",template,"--source",name,
                "--operator","codex_existing_unchanged_physical_contract",
                "--confirm",old_audit["explicit_confirmation"],"--output",audit],
                "full_audit."+name)
        if not receipt.exists():
            validate_selected_tasks(root, source)
            arguments=["--data-template",template,"--source",name,
                "--raw-root",source["raw_root"],"--adapter-contract",adapter,
                "--adapter-contract-sha256",sha256_file(adapter),
                "--adapter-audit-receipt",audit,"--output-manifest",manifest,
                "--output-receipt",receipt,"--episode-index-file",root/"episode_indices"/(name+".txt")]
            ranges=root/"episode_ranges"/(name+".nonidle.json")
            if ranges.exists(): arguments+=["--episode-range-file",ranges]
            run("data/materialize_source_inventory.py",arguments,"full_inventory."+name)
        info=json.loads(receipt.read_text())
        print(json.dumps({"event":"inventory_complete","source":name,
              "selection":"all_usable_episodes","manifest":str(manifest)}),flush=True)
        return name,receipt
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(inventory,s) for s in value["sources"]]
        receipts=dict(f.result() for f in as_completed(futures))
    profile=root/"data_profile.yaml"
    if not profile.exists():
        arguments=["--template",template,"--output",profile,"--receipt",root/"data_profile.receipt.json"]
        for name,receipt in sorted(receipts.items()): arguments+=["--inventory",f"{name}={receipt}"]
        run("data/materialize_data_profile.py",arguments,"full_profile")
    encoder=code/"configs/encoder/vggt_native_p64.yaml"
    task_encoder=code/"configs/encoder/task_qwen3_vl_embedding_2b.yaml"
    task_bank=root/"task_bank"
    if not (task_bank/"receipt.json").exists():
        run("data/build_task_embeddings.py",["--data-profile",profile,
            "--encoder-contract",task_encoder,"--output-root",task_bank,"--device",args.device],"full_task_bank")
    tasks=root/"tasks.jsonl"
    if not tasks.exists():
        run("data/plan_cache_tasks.py",["--data-profile",profile,
            "--encoder-contract",encoder,"--task-encoder-contract",task_encoder,
            "--task-bank-index",task_bank/"index.jsonl","--output",tasks],"full_task_plan")
    seal=root/"streaming_metadata_seal.json"
    if not seal.exists():
        run("data/materialize_streaming_metadata.py",[
            "--task-manifest",tasks,"--data-profile",profile,"--model-profile",args.model_profile,
            "--encoder-contract",encoder,"--task-bank-root",task_bank,
            "--task-bank-index-sha256",sha256_file(task_bank/"index.jsonl"),
            "--output-root",root/"metadata","--episode-index",root/"episode_index.jsonl",
            "--window-index",root/"window_index.jsonl","--grouped-normalization",root/"grouped_normalization.json",
            "--output-seal",seal,"--workers",args.metadata_workers,
            "--processes",args.metadata_processes],"full_streaming_metadata")
    result=json.loads(seal.read_text())
    print(json.dumps({"event":"full_closure_ready","data_profile":str(profile),"seal":str(seal),
        "sources":len(value["sources"]),"episodes":result["episode_count"],"windows":result["window_count"],
        "excluded_sources":value["notes"].get("excluded_sources",[])}),flush=True)
if __name__=="__main__": main()
