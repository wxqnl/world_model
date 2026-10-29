"""Build real benchmark-native episode catalogs without video conversion or hashes."""
from __future__ import annotations
import argparse, collections, json, math, os, re, time
from pathlib import Path
import h5py
import numpy as np

def binding(path):
    p=Path(path);s=p.lstat()
    if p.is_symlink() or not p.is_file():raise ValueError("not a regular immutable asset: "+str(p))
    return dict(device=s.st_dev,inode=s.st_ino,size_bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)

def plain(value):
    if isinstance(value,bytes):return value.decode()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value

def read_rows(path):
    with Path(path).open() as h:
        for line in h:
            if line.strip():yield json.loads(line)

class Writer:
    def __init__(self,root):
        self.root=root;self.handles={};self.counts=collections.Counter()
    def row(self,split,value):
        if split not in self.handles:self.handles[split]=(self.root/(split+".jsonl")).open("x")
        self.handles[split].write(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n")
        self.counts[split]+=1
    def close(self):
        for h in self.handles.values():h.flush();os.fsync(h.fileno());h.close()

def publish(path,value):
    with path.open("x") as h:
        json.dump(value,h,indent=2,ensure_ascii=False,allow_nan=False);h.write("\n");h.flush();os.fsync(h.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)

def libero(root,out):
    catalog_path=root/"data_catalogs.json";source=binding(catalog_path)
    data=json.loads(catalog_path.read_text())["libero"];result={}
    for suite,meta in data.items():
        folder=out/suite;folder.mkdir();w=Writer(folder);windows=0;frames=0;controllers=[]
        for item in meta["files"]:
            path=Path(item["file"]);before=binding(path)
            with h5py.File(path,"r") as h:
                env=json.loads(h["data"].attrs["env_args"])["env_kwargs"]
                ctrl=env["controller_configs"]
                assert ctrl["type"]=="OSC_POSE" and env["control_freq"]==20 and ctrl["control_delta"] is True
                assert np.allclose(ctrl["output_max"],[.05,.05,.05,.5,.5,.5])
                assert np.allclose(ctrl["output_min"],[-.05,-.05,-.05,-.5,-.5,-.5])
                instruction=json.loads(h["data"].attrs["problem_info"])["language_instruction"]
                names=sorted((k for k in h["data"] if re.fullmatch(r"demo_[0-9]+",k)),key=lambda s:int(s[5:]))
                assert len(names)==50
                for name in names:
                    g=h["data"][name];L=int(g["actions"].shape[0])
                    assert g["actions"].shape==(L,7)
                    assert g["obs/agentview_rgb"].shape==(L,128,128,3)
                    assert g["obs/eye_in_hand_rgb"].shape==(L,128,128,3)
                    assert g["obs/ee_pos"].shape==(L,3) and g["obs/ee_ori"].shape==(L,3)
                    assert g["obs/gripper_states"].shape==(L,2)
                    count=max(0,L-33);assert count>0
                    w.row("train",dict(id=suite+"/"+path.stem+"/"+name,suite=suite,
                        file=str(path),file_binding=before,hdf5_group="data/"+name,split="train",
                        instruction=instruction,raw_action_rows=L,full_future_window_starts=count,
                        first_anchor=1,last_anchor=L-33,action_rate_hz=20,world_frame_stride=4,
                        future_action_steps=32,future_world_frames=8,
                        camera_keys=["obs/agentview_rgb","obs/eye_in_hand_rgb"],
                        stored_image_convention=plain(h["data"].attrs.get("macros_image_convention","unspecified")),
                        controller_type="OSC_POSE",controller_input_min=plain(ctrl["input_min"]),
                        controller_input_max=plain(ctrl["input_max"]),
                        controller_output_min=plain(ctrl["output_min"]),controller_output_max=plain(ctrl["output_max"]),
                        action_semantic_transform="controller-scaled delta position/axis-angle + signed close01",
                        state_source=["obs/ee_pos","obs/ee_ori","obs/gripper_states"],
                        clocks_from_recorded_env_control_freq=True,pixels_decoded=False))
                    windows+=count;frames+=L
                controllers.append(dict(file=str(path),control_freq=20,controller="OSC_POSE"))
            assert binding(path)==before
        w.close();assert w.counts["train"]==500
        value=dict(benchmark="LIBERO",suite=suite,training_demos=500,valid_full_future_windows=windows,
                   window_passes=40,global_batch=128,optimizer_updates=math.ceil(windows*40/128),
                   raw_action_rows=frames,controller_contract_files=controllers,source_catalog_binding=source,
                   numeric_normalization_fitted=False,raw_image_payload_converted=False,
                   native_training_batch_adapter_tested=False,metadata_complete=True)
        publish(folder/"CATALOG.json",value);result[suite]={k:v for k,v in value.items() if k!="controller_contract_files"}
    assert binding(catalog_path)==source
    return result

def worldarena(root,out):
    job=json.loads((root/"worldarena/job.json").read_text());path=Path(job["source_index"]);before=binding(path)
    w=Writer(out);task_counts=collections.Counter();eligible=0
    for row in read_rows(path):
        p=Path(row["hdf5_file"]);b=binding(p)
        with h5py.File(p,"r") as h:
            L=int(h["endpose/left_endpose"].shape[0])
            assert h["endpose/left_endpose"].shape==(L,7) and h["endpose/right_endpose"].shape==(L,7)
            assert h["joint_action/vector"].shape==(L,14)
            cameras=["front_camera","left_camera","right_camera"]
            assert all(h["observation/"+c+"/rgb"].shape==(L,) for c in cameras)
            clock=[k for k in ["timestamp","timestamps","time","observation/timestamp","observation/time"] if k in h]
            clock_meta={k:list(h[k].shape) for k in clock}
            assert L>=10
        assert binding(p)==b
        instruction=Path(row["instruction_file"]);ib=binding(instruction)
        # Preserve the actual instruction structure for the downstream protocol selector.
        language=json.loads(instruction.read_text());assert binding(instruction)==ib
        w.row(row["split"],dict(id=row["id"],task=row["task"],episode=row["episode"],split=row["split"],
            file=str(p),file_binding=b,video_file=row["video_file"],video_binding=binding(row["video_file"]),
            instruction_file=str(instruction),instruction_binding=ib,instruction_payload=language,
            frame_rows=L,full_native_K8_windows_with_prior=max(0,L-9),camera_names=cameras,
            actual_joint_action_key="joint_action/vector",raw_command_dimensions=14,
            pose_keys=["endpose/left_endpose","endpose/right_endpose"],
            timestamp_dataset_shapes=clock_meta,
            physical_cadence_resolved=bool(clock),clock_resolution_required=not bool(clock),
            no_cadence_inferred_from_video_playback=True,pixels_decoded=False))
        task_counts[(row["split"],row["task"])]+=1
        if row["split"]=="train":eligible+=max(0,L-9)
    w.close();assert dict(w.counts)=={"train":2000,"test":500}
    assert len(task_counts)==100 and set(v for (s,t),v in task_counts.items() if s=="train")=={40}
    assert binding(path)==before
    result=dict(benchmark="WorldArena",split_counts=dict(w.counts),source_index_binding=before,
        actual_native_K8_windows_with_prior=eligible,training_segments_per_video=1280,
        optimizer_updates=20000,global_batch=128,metadata_complete=True,
        raw_joint_commands_preserved_as_source_fields=True,
        native_training_batch_adapter_tested=False,numeric_normalization_fitted=False,
        physical_clock_policy_requires_source_resolution=True)
    publish(out/"CATALOG.json",result);return result

def robonet(root,out):
    job=json.loads((root/"robonet/job.json").read_text())
    source=Path(job["trainval_index"]);test=Path(job["test_index"])
    source_sig=binding(source);test_sig=binding(test)
    w=Writer(out);robots=collections.Counter();dims=collections.Counter();primitives=collections.Counter()
    eligible=collections.Counter();reasons=collections.Counter();seen={}
    for path in [source,test]:
        for row in read_rows(path):
            split=row.get("split","test")
            if split not in ["train","val","test"]:raise ValueError("unknown split")
            if path==source and split=="test":continue
            identifier=row["id"]
            if identifier in seen:raise ValueError("duplicate/cross-split episode: "+identifier)
            seen[identifier]=split
            p=Path(row["file"]);b=binding(p)
            with h5py.File(p,"r") as h:
                a=h["policy/actions"];state=h["env/state"];env=h["env"]
                assert len(a.shape)==len(state.shape)==2 and a.shape[1]>0 and state.shape[1]>0
                metadata={k:plain(v) for k,v in h["metadata"].attrs.items()}
                camera=h["env/cam0_video"];encoding=plain(env.attrs["cam_encoding"])
                if encoding=="mp4":
                    raw=camera["frames"];T=int(raw.attrs["T"])
                    image_format=plain(raw.attrs["image_format"])
                    image_shape=plain(raw.attrs["shape"])
                elif encoding=="jpg":
                    T=len([k for k in camera if re.fullmatch(r"frame[0-9]+",k)])
                    image_format="BGR_encoded_JPEG";image_shape=None
                else:raise ValueError("unsupported raw RoboNet image encoding: "+str(encoding))
                length=min(T,int(state.shape[0]),int(a.shape[0])+1)
                primitive=str(metadata.get("primitives",""));adim=int(a.shape[1])
                reason="shorter_than2_context_plus10_future" if length<12 else None
                robot=str(metadata.get("robot","unknown"));assert robot!="unknown"
                transforms=[]
                if adim==4 and primitive=="autograsp":
                    transforms.append("official iVideoGPT autograsp fifth channel reconstructed from next measured gripper state and recorded bounds; preserve explicit provenance")
                elif adim<5:transforms.append("missing command dimensions stay masked; do not unmask synthetic zeros")
                elif adim>5:transforms.append("raw action layout wider than benchmark5D; resolve named native layout explicitly before training, no silent truncation")
                record=dict(id=identifier,split=split,file=str(p),file_binding=b,
                    video_encoding=encoding,camera=0,declared_video_frames=T,aligned_frames=length,
                    action_rows=int(a.shape[0]),action_dimensions=adim,state_rows=int(state.shape[0]),
                    state_dimensions=int(state.shape[1]),image_format=image_format,image_shape=image_shape,
                    robot=robot,primitives=primitive,action_space=metadata.get("action_space"),
                    gripper=metadata.get("gripper"),low_bound=metadata.get("low_bound"),high_bound=metadata.get("high_bound"),
                    eligible=reason is None,ineligible_reason=reason,valid12_frame_starts=max(0,length-11),
                    context_frames=2,future_frames=10,raw_command_contract_notes=transforms,
                    actual_MP4_decode_frame_count_verified=False,physical_clock_resolved=False,
                    no_clock_inferred_from_MP4_playback=True)
                if reason is None:eligible[split]+=1
                else:reasons[(split,reason)]+=1
                robots[(split,robot)]+=1;dims[(split,str(adim),str(state.shape[1]))]+=1
                primitives[(split,primitive)]+=1
            assert binding(p)==b
            w.row(split,record)
            if sum(w.counts.values())%10000==0:
                print(json.dumps({"benchmark":"RoboNet","metadata_records":sum(w.counts.values()),
                                  "eligible":dict(eligible)},separators=(",",":")),flush=True)
    w.close()
    assert binding(source)==source_sig and binding(test)==test_sig
    assert w.counts["train"]==161649 and w.counts["val"]==512 and w.counts["test"]==256
    result=dict(benchmark="RoboNet",catalog_split_counts=dict(w.counts),eligible_split_counts=dict(eligible),
        rejected_counts=[dict(split=s,reason=t,count=n) for (s,t),n in reasons.items()],
        robot_counts=[dict(split=s,robot=t,count=n) for (s,t),n in robots.items()],
        dimensions=[dict(split=s,action_dim=int(a),state_dim=int(t),count=n) for (s,a,t),n in dims.items()],
        primitives=[dict(split=s,primitives=t,count=n) for (s,t),n in primitives.items()],
        source_index_binding=source_sig,test_index_binding=test_sig,no_train_test_episode_id_overlap=True,
        trajectory_passes=80,global_batch=128,optimizer_updates=math.ceil(eligible["train"]*80/128),
        actual_requested_training_segment_exposures=eligible["train"]*80,
        metadata_complete=True,numeric_normalization_fitted=False,native_training_batch_adapter_tested=False,
        real_video_decoder_alignment_tested=False,physical_clock_policy_requires_source_resolution=True)
    publish(out/"CATALOG.json",result)
    assert eligible["val"]==512 and eligible["test"]==256,"fixed val/test contains short episodes; preserve catalog and resolve protocol explicitly"
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    assert Path.cwd()==Path("/data/Minko") and os.environ.get("CUDA_VISIBLE_DEVICES")==""
    assert not a.output.exists(),"one new metadata identity per materialization"
    a.output.mkdir(parents=True,mode=0o700)
    start=time.time();result={}
    for name,worker in [("libero",libero),("worldarena",worldarena),("robonet",robonet)]:
        folder=a.output/name;folder.mkdir()
        result[name]=worker(a.root,folder)
    publish(a.output/"COMPLETE.json",dict(schema="wm3d_benchmark_native_episode_catalogs",
        created_unix=time.time(),seconds=time.time()-start,benchmarks=result,
        source_payload_hashes_computed=False,RGB_video_payload_copies_created=False,
        metadata_only=True,native_adapter_or_training_completion_claimed=False))
    print(json.dumps({"metadata_complete":True,"output":str(a.output),"seconds":time.time()-start}),flush=True)
if __name__=="__main__":main()
