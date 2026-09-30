#!/usr/bin/env python3
"""Admit the frozen InternData release with explicit observed-state supervision.

Original recording IDs were not published. Release membership is explicit;
exact numeric candidate groups receive exact RGB comparisons, not hashes.
Only proven exact copies are excluded. All raw archives remain untouched.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from wm3d.data.file_binding import verify_stat_asset
from wm3d.data.source_inventory import scan_lerobot_source, _episode_metadata, _path_values
from wm3d.data.source_adapters import load_adapter_contract, adapt_action_series
from wm3d.data.episode_io import VerifiedAssetStore, open_admitted_episode_accessor
from wm3d.data.manifest_contract import validate_source_manifest
from scripts.data.admit_agibot_native import publish, bind, embodiment

FAMILIES = {"franka": (244, 7, ("",)), "split_aloha": (245, 6, ("left_", "right_")),
            "lift2": (246, 6, ("left_", "right_")), "genie1": (247, 7, ("left_", "right_"))}


def contract(family, info):
    eid, joints, arms = FAMILIES[family]
    features = info["features"]
    name = "interndata_" + family + "_observed_state"
    mapping = []; groups = []
    term = lambda key, n: dict(key=key, columns=list(range(n)), scale=[1]*n, offset=[0]*n)
    for slot, arm in enumerate(arms):
        joint = arm + "joint.position"; grip = arm + "gripper.position"; dim = joints+1
        for prefix in ("actions.", "states."):
            if features[prefix+joint]["shape"] != [joints] or features[prefix+grip]["shape"] != [1]:
                raise RuntimeError("official robot family and actual feature layout disagree")
        group = "arm" if not arm else arm + "arm"
        mapping.append(dict(group=group, supervision="coarse_effect",
            action=[term("actions."+joint,joints),term("actions."+grip,1)],
            state=[term("states."+joint,joints),term("states."+grip,1)],
            action_time_key=None, state_time_key="timestamp", world_interval_index_key="frame_index",
            action_transform="identity", state_transform="identity",
            action_value_mask=[True]*dim, state_value_mask=[True]*dim,
            action_alignment="next_observation_drop_terminal"))
        frame = family + "_recorded_joint_and_gripper_coordinates_native_units"
        groups.append(dict(name=group,group_id=20+slot,action_semantics=["observed_state_native"]*dim,
            state_semantics=["controller_state"]*dim,action_frame=frame,state_frame=frame,
            composition_operators=["last"]*dim))
    camera = [("head","images.rgb.head")]
    camera += ([("left_wrist","images.rgb.hand")] if family=="franka" else
               [("left_wrist","images.rgb.hand_left"),("right_wrist","images.rgb.hand_right")])
    if any(features[k]["dtype"]!="video" for _,k in camera):raise RuntimeError("missing recorded camera")
    adapter = dict(schema="wm3d_source_adapter_v6",name=name,raw_format="lerobot_parquet_video",
        observation_time_key="timestamp",views=[dict(name=n,key=k,color_order="rgb") for n,k in camera],
        groups=mapping)
    template = dict(schema="wm3d_admitted_source_template_v1",
        sources=[dict(name=name,embodiment=name)],
        embodiments=[dict(name=name,embodiment_id=eid,groups=groups)])
    return adapter,template


def materialized(path, ad):
    p=Path(path);parts=p.parts;pos=parts.index("batch_00")
    return ad/"selected_rgb_b1_r2_20260930/production_payloads"/Path(*parts[pos:])


@lru_cache(maxsize=1024)
def info_metadata(root):
    return json.loads((root/"meta/info.json").read_text()), {
        int(x["episode_index"]):x for x in _episode_metadata(root)}


def cameras(member, ad, admission):
    path = materialized(member["path"],ad)
    st=path.stat(); expected=member["identity"]
    if (st.st_size,st.st_mtime_ns,st.st_ino)!=(expected["bytes"],expected["mtime_ns"],expected["inode"]):
        raise RuntimeError("numeric duplicate evidence changed")
    root=path.parents[2]
    info,metadata=info_metadata(root);i=int(member["episode_index"])
    values=_path_values(metadata[i],i)
    result=[]
    for key,f in sorted(info["features"].items()):
        if f["dtype"]!="video" or not key.startswith("images.rgb."):continue
        relative=info.get("video_path","videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4").format(**dict(values,video_key=key))
        p=root/relative
        result.append((key,p,bind(p,admission)))
    if not result:raise RuntimeError("exact candidate has no selected RGB")
    return result


def equal_cameras(a,b):
    if [x[0] for x in a]!=[x[0] for x in b]:return False
    equal=True
    for (_,p,pb),(_,q,qb) in zip(a,b):
        if pb["size_bytes"]!=qb["size_bytes"]:equal=False;break
        with p.open("rb") as pf,q.open("rb") as qf:
            while True:
                x=pf.read(1<<20); y=qf.read(1<<20)
                if x!=y:equal=False;break
                if not x:break
        if not equal:break
    for _,p,binding in (*a,*b):verify_stat_asset(p.parent,p.name,binding)
    return equal


def review_bucket(bucket,ad,admission):
    d=json.loads(bucket.read_text());aliases={};clusters={};groups=[];compared=0
    for gi,members in enumerate(d["exact_numeric_groups"]):
        if len(members)<2:continue
        cluster=f"{bucket.name}:{gi}";representatives=defaultdict(list);all_views=[]
        for member in sorted(members,key=lambda m:m["path"]):
            path=str(materialized(member["path"],ad));clusters[path]=cluster
            views=cameras(member,ad,admission);all_views.extend(views)
            # Different lengths cannot contain identical RGB byte streams.
            signature=tuple((key,binding["size_bytes"]) for key,_,binding in views)
            match=None
            for prev,pviews in representatives[signature]:
                compared+=1
                if equal_cameras(views,pviews):match=prev;break
            if match is None:representatives[signature].append((path,views))
            else:aliases[path]=match
        # Also bind size-excluded members across the complete comparison interval.
        for _,path,binding in all_views:verify_stat_asset(path.parent,path.name,binding)
        groups.append(dict(numeric_cluster=cluster,members=len(members),
            distinct_rgb_records=sum(len(v) for v in representatives.values())))
    return dict(bucket=bucket.name,aliases=aliases,split_clusters=clusters,groups=groups,
        exact_RGB_pair_comparisons=compared,SHA_scans=False)


def reuse_or_review(bucket,ad,admission,reuse):
    prior=None if reuse is None else reuse/bucket.name
    if prior is None or not prior.is_file():return review_bucket(bucket,ad,admission)
    binding=bind(prior,admission);result=json.loads(prior.read_text())
    verify_stat_asset(prior.parent,prior.name,binding)
    groups=json.loads(bucket.read_text())["exact_numeric_groups"]
    expected={str(materialized(m["path"],ad)):f"{bucket.name}:{i}"
              for i,g in enumerate(groups) if len(g)>1 for m in g}
    if (result.get("bucket")!=bucket.name or result.get("split_clusters")!=expected
            or result.get("SHA_scans") is not False
            or any(k not in expected or v not in expected for k,v in result["aliases"].items())):
        raise RuntimeError("completed duplicate review membership changed")
    # Completed comparisons came from the immediately preceding immutable code snapshot.
    # Reconfirm its numerical source identity without repeating RGB comparisons.
    for g in groups:
        if len(g)<2:continue
        for m in g:
            st=materialized(m["path"],ad).stat();identity=m["identity"]
            if (st.st_size,st.st_mtime_ns,st.st_ino)!=(identity["bytes"],identity["mtime_ns"],identity["inode"]):
                raise RuntimeError("completed duplicate review source changed")
    return result


def scan(job):
    root=Path(job["root"]);out=Path(job["output"]);admission=job["admission"]
    ap=Path(job["adapter"]);adapter=load_adapter_contract(ap,expected_binding=bind(ap,admission))
    emb=embodiment(job["template"])
    rows,receipt=scan_lerobot_source(root=root,source=adapter.name,embodiment=emb,adapter=adapter,
        split_seed=3407,train_fraction=.95,validation_fraction=.05,default_task="",
        episode_indices=sorted(job["catalog"]),admission_id=admission,episode_catalog=job["catalog"])
    first=rows[0]
    accessor=open_admitted_episode_accessor(source_root=root,payload=first["payload"],
        payload_binding=first["payload_binding"],row_start=first["payload_row_start"],
        row_stop=first["payload_row_stop"],adapter=adapter,asset_verifier=VerifiedAssetStore())
    actions=adapt_action_series(accessor=accessor,contract=adapter,embodiment=emb)
    for group,series in zip(adapter.groups,actions):
        truth=np.concatenate([np.asarray(accessor.array(t.key.replace("actions.","states."))).reshape(len(series.values)+1,-1)[:,t.columns]
                              for t in group.action],axis=1)[1:]
        if not np.array_equal(series.values,truth.astype(np.float32)):
            raise RuntimeError("actual next-state alignment changed")
    prefix=root.relative_to(Path(job["common_root"])).as_posix()
    out.mkdir(parents=True,exist_ok=True);manifest=out/"manifest.jsonl"
    with manifest.open("x") as f:
        for raw in rows:
            row=dict(raw);row["payload"]=prefix+"/"+row["payload"]
            row["assets"]=[dict(x,path=prefix+"/"+x["path"]) for x in row["assets"]]
            f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
    receipt.update(manifest_path=str(manifest),manifest_binding=bind(manifest,admission),
        numeric_receipt=job["numeric_receipt"],numeric_binding=job["numeric_binding"],
        recorded_release=job["recorded_release"],actual_next_state_replay=True)
    publish(out/"receipt.json",receipt)
    return dict(source=adapter.name,manifest=str(manifest),receipt=str(out/"receipt.json"),episodes=len(rows))


def main():
    p=argparse.ArgumentParser();p.add_argument("--admission-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--admission-id",required=True)
    p.add_argument("--workers",type=int,default=24)
    p.add_argument("--resume-reviewed-buckets",type=Path)
    a=p.parse_args()
    ad=a.admission_root.resolve(strict=True);rem="/data/Minko/wm3d_native_expansion_admission_20260930"
    local=lambda p:Path(str(p).replace(rem,str(ad),1))
    full=ad/"selected_rgb_b1_r2_20260930/result/COMPLETE.json"
    f=json.loads(full.read_text())
    if not f["passed"] or f["expected_rgb_frames"]!=f["decoded_frames"]:raise RuntimeError("RGB incomplete")
    numeric=ad/"numeric_controls_b1_20260930/result/COMPLETE.json";c=json.loads(numeric.read_text())
    if not c["passed"]:raise RuntimeError("numeric incomplete")
    duplicates={};split_clusters={};compared=0;exact_groups=[]
    buckets=sorted((ad/"numeric_dedup_b1_r2_20260930/result").glob("bucket_*.json"))
    with ProcessPoolExecutor(max_workers=min(8,a.workers)) as pool:
        futures=[pool.submit(reuse_or_review,bucket,ad,a.admission_id,a.resume_reviewed_buckets) for bucket in buckets]
        for future in as_completed(futures):
            item=future.result()
            duplicates.update(item["aliases"]);split_clusters.update(item["split_clusters"])
            compared+=item["exact_RGB_pair_comparisons"];exact_groups.extend(item["groups"])
            publish(a.output/"duplicate_buckets"/item["bucket"],item)
            print(json.dumps(dict(status="exact_duplicate_review",bucket=item["bucket"],
                aliases=len(item["aliases"]),pair_comparisons=item["exact_RGB_pair_comparisons"])),flush=True)
    publish(a.output/"duplicate_review.json",dict(passed=True,exact_numeric_and_RGB_aliases=duplicates,
        exact_numeric_groups=exact_groups,exact_RGB_pair_comparisons=compared,
        criterion="Previously exact numeric trajectory plus byte-identical selected RGB. No action-only deletion.",
        split_clusters=split_clusters,all_raw_preserved=True,SHA_scans=False,
        source_global_recording_ID_published=False,cross_release_global_ID_certified=False))
    jobs=[];variants={};stages={};allkeys=set()
    for rp in c["receipt_paths"]:
        rp=local(rp);r=json.loads(rp.read_text())
        if r["input"]["family"]!="interndata_a1":continue
        sp=local(r["input"]["stage_receipt"])
        if str(sp) not in stages:stages[str(sp)]=json.loads(sp.read_text())["input"]["item"]
        item=stages[str(sp)]
        if not item["path"].startswith("sim_updated/") or "lerobotv30" in item["path"]:raise RuntimeError("wrong release")
        family=item["path"].split("/")[2]
        root=materialized(r["input"]["root"],ad);info,metadata=info_metadata(root)
        adapter,template=contract(family,info);name=adapter["name"]
        spec=dict(adapter=adapter,template=template)
        if name in variants and variants[name]!=spec:raise RuntimeError("robot-specific contract drift")
        variants[name]=spec;catalog={}
        for group in adapter["groups"]:
            for term in group["action"]:
                pair=r["action_state_pairs"][term["key"]]
                if pair["next_max_abs"]!=0 or pair["next_squared_error_sum"]!=0:
                    raise RuntimeError("full numeric receipt does not prove next-state relation")
        for i,m in sorted(metadata.items()):
            values=_path_values(m,i);relative=info.get("data_path","data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet").format(**values)
            payload=str(root/relative)
            if payload in duplicates:continue
            release_key=f'{item["revision"]}:{item["path"]}:{root.relative_to(ad).as_posix().split("/payloads/",1)[1]}:{i}'
            if release_key in allkeys:raise RuntimeError("repeated frozen release member")
            allkeys.add(release_key)
            # This ID identifies a published member, NOT an unpublished global trajectory.
            eid="interndata_a1:published_member:"+release_key
            catalog[i]=dict(episode_id=eid,split="train")
        if not catalog:continue
        jobs.append(dict(root=str(root),common_root=str(ad),source=name,template=template,
            adapter=str(a.output/name/"adapter.yaml"),admission=a.admission_id,catalog=catalog,
            output=str(a.output/"parts"/f"{len(jobs):04d}"),
            numeric_receipt=str(rp),numeric_binding=bind(rp,a.admission_id),
            recorded_release=item,metadata=metadata,info=info))
    if len(allkeys)+len(duplicates)!=170674:raise RuntimeError("frozen Intern membership drift")
    for name,v in variants.items():
        keys={}
        for job in jobs:
            if job["source"]!=name:continue
            for i,rec in job["catalog"].items():
                values=_path_values(job["metadata"][i],i)
                path=str(Path(job["root"])/job["info"].get("data_path","data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet").format(**values))
                key=split_clusters.get(path,rec["episode_id"]);keys.setdefault(key,[]).append(rec)
        order=sorted(keys);perm=np.random.default_rng(3407).permutation(len(order))
        val={order[int(i)] for i in perm[:max(1,round(len(order)*.05))]}
        for key,records in keys.items():
            for rec in records:rec["split"]="val" if key in val else "train"
        publish(a.output/name/"adapter.yaml",v["adapter"])
        publish(a.output/name/"source.template.json",v["template"])
    completed=[]
    for job in jobs:job.pop("metadata");job.pop("info")
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(scan,j) for j in jobs]):
            item=future.result();completed.append(item)
            if len(completed)%20==0 or len(completed)==len(jobs):
                print(json.dumps(dict(status="intern_admission_progress",datasets=len(completed),
                    total=len(jobs),episodes=sum(x["episodes"] for x in completed))),flush=True)
    for name,v in variants.items():
        out=a.output/name;manifest=out/"manifest.jsonl"
        parts=sorted([x for x in completed if x["source"]==name],key=lambda x:x["manifest"])
        catalog={};counts=defaultdict(int);num=0
        with manifest.open("x") as target:
            for part in parts:
                with Path(part["manifest"]).open() as inp:
                    for line in inp:
                        row=json.loads(line);catalog[num]=dict(episode_id=row["episode_id"],split=row["split"])
                        counts[row["split"]]+=1;num+=1;target.write(line)
        cp=out/"episode_catalog.json"
        publish(cp,dict(schema="wm3d_admitted_episode_catalog_v1",admission_id=a.admission_id,source=name,episodes=catalog))
        validation=validate_source_manifest(manifest,expected_source=name,
            expected_embodiment=embodiment(v["template"]),admission_id=a.admission_id)
        audit=dict(schema="wm3d_admitted_source_adapter_audit_receipt_v1",source=name,
            admission_id=a.admission_id,structural_checks="pass",semantic_review="operator_confirmed_fail_closed",
            adapter_contract_binding=bind(out/"adapter.yaml",a.admission_id),
            data_template_binding=bind(out/"source.template.json",a.admission_id),
            episode_catalog_binding=bind(cp,a.admission_id),supervision="coarse_effect",
            review=dict(family_from_fixed_official_path=True,not_robot_type_piper=True,
                selected_target="Recorded next joint positions and gripper positions, source-native unchanged.",
                alignment="action[t] == observed_state[t+1]; drop unpaired terminal row",
                fine_control_command_claim=False,gripper_clipping=False,
                excluded_from_model_targets="Redundant Cartesian pose variants and master_actions retained in raw files; not relabelled as fine commands.",
                global_recording_identity_published=False,
                identity="One frozen sim_updated release; published-member identity, exact numeric+RGB copies excluded.",
                split="Whole episodes; exact numeric candidate clusters stay on one side, regardless of RGB differences."),
            duplicate_review_path=str(a.output/"duplicate_review.json"),
            duplicate_review_binding=bind(a.output/"duplicate_review.json",a.admission_id))
        publish(out/"adapter_audit.json",audit)
        receipt=dict(schema="wm3d_admitted_source_inventory_receipt_v1",source=name,
            raw_root=str(ad),admission_id=a.admission_id,
            adapter_contract_path=str(out/"adapter.yaml"),adapter_contract_binding=audit["adapter_contract_binding"],
            data_template_path=str(out/"source.template.json"),data_template_binding=audit["data_template_binding"],
            episode_catalog_path=str(cp),episode_catalog_binding=audit["episode_catalog_binding"],
            adapter_audit_receipt_path=str(out/"adapter_audit.json"),
            adapter_audit_receipt_binding=bind(out/"adapter_audit.json",a.admission_id),
            manifest_path=str(manifest),manifest_binding=bind(manifest,a.admission_id),
            manifest_validation=validation,episode_count=num,split_count=dict(counts),
            component_receipts=parts,RGB_complete=dict(path=str(full),binding=bind(full,a.admission_id)),
            global_recording_identity_certified=False,SHA_scans=False)
        publish(out/"receipt.json",receipt)
    publish(a.output/"COMPLETE.json",dict(passed=True,completed_at=datetime.now().astimezone().isoformat(),
        original_published_members=170674,admitted_members=len(allkeys),exact_copy_aliases=len(duplicates),
        variants=sorted(variants),supervision="coarse_effect",global_recording_identity_certified=False,
        SHA_scans=False,formal_training_started=False))
    print(json.dumps(dict(status="intern_admission_complete",episodes=len(allkeys),aliases=len(duplicates))),flush=True)


if __name__=="__main__":main()
