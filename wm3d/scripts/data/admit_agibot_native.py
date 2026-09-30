#!/usr/bin/env python3
"""Admit frozen AgiBot2026 recorded controls through the production reader."""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import re
import numpy as np
from wm3d.data.agibot_native_contract import agibot_native_contract
from wm3d.data.file_binding import bind_stat_asset, verify_stat_asset
from wm3d.data.grouped_robot import EmbodimentSpec, ActionGroupSpec
from wm3d.data.source_adapters import load_adapter_contract, adapt_action_series, adapt_state_series
from wm3d.data.source_inventory import scan_lerobot_source, _episode_metadata
from wm3d.data.episode_io import VerifiedAssetStore, open_admitted_episode_accessor
from wm3d.data.lossless_action_layout import reconstruct_native_actions
from wm3d.data.manifest_contract import validate_source_manifest


def publish(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n"
    if path.exists():
        if path.read_text()!=payload: raise FileExistsError(path)
    else:
        with path.open("x") as f:f.write(payload)


def bind(path,admission):
    return bind_stat_asset(path.parent,path.name,admission_id=admission)


def embodiment(template):
    e=template["embodiments"][0]
    return EmbodimentSpec(e["name"],e["embodiment_id"],tuple(
        ActionGroupSpec(**{k:tuple(v) if isinstance(v,list) else v for k,v in g.items()})
        for g in e["groups"]))


def scan(job):
    root=Path(job["root"]); output=Path(job["output"]); admission=job["admission"]
    ap=Path(job["adapter"]); ab=bind(ap,admission)
    adapter=load_adapter_contract(ap,expected_binding=ab)
    emb=embodiment(job["template"])
    rows,receipt=scan_lerobot_source(root=root,source=adapter.name,embodiment=emb,
        adapter=adapter,split_seed=3407,train_fraction=.95,validation_fraction=.05,
        default_task="",admission_id=admission,episode_catalog=job["catalog"])
    first=rows[0]
    access=open_admitted_episode_accessor(source_root=root,payload=first["payload"],
        payload_binding=first["payload_binding"],row_start=first["payload_row_start"],
        row_stop=first["payload_row_stop"],adapter=adapter,asset_verifier=VerifiedAssetStore())
    actions=adapt_action_series(accessor=access,contract=adapter,embodiment=emb)
    states=adapt_state_series(accessor=access,contract=adapter,embodiment=emb)
    native=np.asarray(access.array("action"))
    restored=reconstruct_native_actions({x.group:x.values for x in actions},
        layout=adapter.raw_action_layout,groups=adapter.groups)
    if not np.array_equal(restored,native):
        raise RuntimeError("native action inversion is not exact")
    if any(x.values.shape[1]>14 for x in actions) or any(x.values.shape[1]>10 for x in states):
        raise RuntimeError("mapped controls/state exceed production model capacity")
    prefix=root.relative_to(Path(job["common_root"])).as_posix()
    transformed=[]
    for raw in rows:
        row=dict(raw)
        row["payload"]=prefix+"/"+row["payload"]
        row["assets"]=[dict(asset,path=prefix+"/"+asset["path"]) for asset in row["assets"]]
        transformed.append(row)
    output.mkdir(parents=True,exist_ok=True)
    manifest=output/"manifest.jsonl"
    with manifest.open("x") as f:
        for row in transformed:f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
    receipt.update(dict(manifest_path=str(manifest),manifest_binding=bind(manifest,admission),
        original_numeric_receipt=job["numeric_receipt"],
        original_numeric_binding=job["numeric_binding"],
        actual_native_action_replay=dict(episode_id=first["episode_id"],rows=len(native),
            width=native.shape[1],exact_round_trip=True,all_controls_preserved=True),
        metadata_root_prefix=prefix,common_raw_root=job["common_root"]))
    publish(output/"receipt.json",receipt)
    return dict(source=adapter.name,manifest=str(manifest),receipt=str(output/"receipt.json"),
        episodes=len(rows),replay=receipt["actual_native_action_replay"])


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--admission-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--admission-id",required=True)
    p.add_argument("--workers",type=int,default=8)
    a=p.parse_args(); ad=a.admission_root.resolve(strict=True)
    rem="/data/Minko/wm3d_native_expansion_admission_20260930"
    local=lambda v:Path(str(v).replace(rem,str(ad),1))
    stages={"b1":"selected_rgb_b1_r2_20260930","b2":"selected_rgb_b2_20260930","b3":"selected_rgb_b3_20260930"}
    jobs=[]; variants={}; ids=set(); stage_bindings={}
    for batch,stage in stages.items():
        rgb=ad/stage/"result/COMPLETE.json"; rb=bind(rgb,a.admission_id)
        complete=json.loads(rgb.read_text())
        if not complete["passed"] or complete["selected_rgb_files"]!=complete["selected_rgb_decoded"] or complete["expected_rgb_frames"]!=complete["decoded_frames"]:
            raise RuntimeError("selected RGB has not completed")
        stage_bindings[batch]=dict(path=str(rgb),binding=rb)
        numeric=ad/f"numeric_controls_{batch}_20260930/result/COMPLETE.json"
        c=json.loads(numeric.read_text())
        if not c["passed"]:raise RuntimeError("numeric controls not complete")
        for rp in c["receipt_paths"]:
            rp=local(rp); r=json.loads(rp.read_text())
            if r["input"]["family"]!="agibot_world":continue
            if not r["passed"] or not r["all_numeric_fields_finite"]:
                raise RuntimeError("unpassed numeric receipt")
            oldroot=local(r["input"]["root"])
            tail=Path("batch_"+str(("b1","b2","b3").index(batch)).zfill(2))
            parts=oldroot.parts; pos=parts.index(str(tail))
            root=ad/stage/"production_payloads"/Path(*parts[pos:])
            info=json.loads((root/"meta/info.json").read_text())
            oldinfo=local(r["input"]["info"])
            if (root/"meta/info.json").stat().st_ino != oldinfo.stat().st_ino:
                raise RuntimeError("materialized metadata no longer aliases staged metadata")
            adapter,template,review=agibot_native_contract(info); name=adapter["name"]
            # Constant declarations are justified by the completed full numeric read.
            bounds=r["numeric_ranges"]["action"]
            for col,val in adapter["raw_action_layout"]["constant_columns"].items():
                if bounds["min"][int(col)]!=val or bounds["max"][int(col)]!=val:
                    raise RuntimeError("constant action declaration lacks full-read evidence")
            spec=dict(adapter=adapter,template=template,review=review)
            if name in variants and variants[name]!=spec:raise RuntimeError("same variant drift")
            variants[name]=spec
            catalog={}
            for item in _episode_metadata(root):
                i=int(item["episode_index"]); h5=info["h5_path"][str(i)]
                m=re.search(r"/proprio_stats/(\d+)/(\d+)/proprio_stats\.h5$",h5)
                if m is None:m=re.search(r"/GENIE/(\d+)/[^/]+/[^/]+/(\d+)/aligned_joints\.h5$",h5)
                if m is None:raise RuntimeError("unrecognized original trajectory identity")
                eid="agibot2026:"+m.group(1)+":"+m.group(2)
                if eid in ids:raise RuntimeError("original trajectory repeated")
                ids.add(eid);catalog[i]=dict(episode_id=eid,split="train")
            index=len(jobs)
            jobs.append(dict(root=str(root),common_root=str(ad),output=str(a.output/"parts"/f"{index:03d}"),
                admission=a.admission_id,template=template,source=name,catalog=catalog,
                adapter=str(a.output/name/"adapter.yaml"),
                numeric_receipt=str(rp),numeric_binding=bind(rp,a.admission_id)))
    if len(jobs)!=43 or len(ids)!=2767:raise RuntimeError("frozen B1/B2/B3 AgiBot membership changed")
    for name,v in variants.items():
        names=sorted(rec["episode_id"] for job in jobs if job["source"]==name for rec in job["catalog"].values())
        perm=np.random.default_rng(3407).permutation(len(names))
        val={names[int(i)] for i in perm[:max(1,round(len(names)*.05))]}
        for job in jobs:
            if job["source"]==name:
                for rec in job["catalog"].values():rec["split"]="val" if rec["episode_id"] in val else "train"
        publish(a.output/name/"adapter.yaml",v["adapter"])
        publish(a.output/name/"source.template.json",v["template"])
    completed=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(scan,job) for job in jobs]):
            item=future.result();completed.append(item)
            print(json.dumps(dict(status="agibot_inventory_progress",archives=len(completed),total=43,latest=item)),flush=True)
    for name,v in variants.items():
        out=a.output/name; manifest=out/"manifest.jsonl"
        selected=sorted((x for x in completed if x["source"]==name),key=lambda x:x["manifest"])
        rows=[]
        for x in selected:
            with Path(x["manifest"]).open() as f:rows.extend(json.loads(l) for l in f)
        rows.sort(key=lambda r:r["episode_id"])
        with manifest.open("x") as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n")
        catalog={i:dict(episode_id=r["episode_id"],split=r["split"]) for i,r in enumerate(rows)}
        cp=out/"episode_catalog.json"
        publish(cp,dict(schema="wm3d_admitted_episode_catalog_v1",admission_id=a.admission_id,source=name,episodes=catalog))
        validation=validate_source_manifest(manifest,expected_source=name,
            expected_embodiment=embodiment(v["template"]),admission_id=a.admission_id)
        audit=dict(schema="wm3d_admitted_source_adapter_audit_receipt_v1",source=name,
            admission_id=a.admission_id,structural_checks="pass",
            semantic_review="operator_confirmed_fail_closed",
            adapter_contract_binding=bind(out/"adapter.yaml",a.admission_id),
            data_template_binding=bind(out/"source.template.json",a.admission_id),
            episode_catalog_binding=bind(cp,a.admission_id),review=v["review"],
            basis="Published per-field indices; native channel values and timestamp retained. No G1/Alpha/X2 unit convention is borrowed.",
            upstream_documentation="https://www.modelscope.cn/datasets/agibot_world/AgiBotWorld2026/summary",
            original_identity="Original task/trajectory from h5_path, unique across all three frozen batches.",
            full_numeric_and_rgb_receipts_reused=True,raw_media_redecoded=False)
        publish(out/"adapter_audit.json",audit)
        receipt=dict(schema="wm3d_admitted_source_inventory_receipt_v1",
            source=name,raw_root=str(ad),admission_id=a.admission_id,
            adapter_contract_path=str(out/"adapter.yaml"),adapter_contract_binding=audit["adapter_contract_binding"],
            data_template_path=str(out/"source.template.json"),data_template_binding=audit["data_template_binding"],
            episode_catalog_path=str(cp),episode_catalog_binding=audit["episode_catalog_binding"],
            adapter_audit_receipt_path=str(out/"adapter_audit.json"),
            adapter_audit_receipt_binding=bind(out/"adapter_audit.json",a.admission_id),
            manifest_path=str(manifest),manifest_binding=bind(manifest,a.admission_id),
            manifest_validation=validation,episode_count=len(rows),
            split_count={s:sum(r["split"]==s for r in rows) for s in ("train","val","test")},
            component_receipts=selected,stage_completions=stage_bindings,
            binding_kind="admitted_file_stat_v1",cryptographic_content_binding=False,
            global_deduplication_performed_by_inventory=False,SHA_scans=False)
        publish(out/"receipt.json",receipt)
    publish(a.output/"COMPLETE.json",dict(passed=True,completed_at=datetime.now().astimezone().isoformat(),
        archives=43,original_episodes=2767,variants=sorted(variants),
        exact_native_action_roundtrips=len(completed),all_controls_preserved=True,
        state_conditioning_selection_explicit=True,SHA_scans=False,
        full_new_training_ready=False,formal_training_started=False))


if __name__=="__main__":main()
