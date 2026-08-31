from __future__ import annotations
from collections.abc import Iterator, Mapping as MappingABC
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from io import BytesIO
import fcntl, hashlib, inspect, itertools, json, math, os, platform, re, stat, zipfile
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping
import numpy as np
import torch
from wm3d_v3.data.manifest import OXEClipRecord
from wm3d_v3.stage1.action_cache import ActionCacheResolutionError,validate_formal_droid_cache_index
from wm3d_v3.stage1.action_contract import action_contract_key, canonical_dataset_name
from wm3d_v3.stage1.action_contract_evidence import FORMAL_DROID_CONTRACT_KEY, FORMAL_OXE_CONTRACT_KEYS
from wm3d_v3.stage1.action_contract_split import frozen_contract_split_from_mapping
from wm3d_v3.stage1.action_evidence_sources import bind_temporal_window, safe_clip_id
from wm3d_v3.stage1.droid_interval_action import DROID_INTERVAL_ACTION_CACHE_SUBDIR
from wm3d_v3.stage1.immutable_artifact import ImmutableArtifactConflict, publish_immutable_bytes

SCHEMA_VERSION="wm3d_v6_stage1_robot_mask_cache_v3"
SPLIT_SCHEMA="wm3d_v6_action_contract_split_v1"
SPLIT_DERIVATION="sha256(seed|contract_key|independence_group_id)"
PROMPT="robot arm. robot gripper."
WINDOW=17
EXPECTED_WINDOWS=576
EXPECTED_CONTRACTS=frozenset((*FORMAL_OXE_CONTRACT_KEYS,FORMAL_DROID_CONTRACT_KEY))
HEX40=re.compile(r"^[0-9a-f]{40}$")
FORMAL_GROUNDING_MODEL="IDEA-Research/grounding-dino-tiny"
FORMAL_GROUNDING_REVISION="a2bb814dd30d776dcf7e30523b00659f4f141c71"
FORMAL_SAM2_MODEL="facebook/sam2.1-hiera-small"
FORMAL_SAM2_REVISION="ee5bba1d82bb8749febdf90f45e84b687142ba03"

class RobotMaskCacheError(RuntimeError): pass

MaskIdentity=tuple[str,str,str,int]

@dataclass(frozen=True)
class ValidatedRobotMaskIndex(MappingABC[MaskIdentity,Mapping[str,Any]]):
    index_path:str
    index_sha256:str
    metadata:Mapping[str,Any]
    entries:Mapping[MaskIdentity,Mapping[str,Any]]

    def __getitem__(self,key:MaskIdentity)->Mapping[str,Any]:
        return self.entries[key]
    def __iter__(self)->Iterator[MaskIdentity]:
        return iter(self.entries)
    def __len__(self)->int:
        return len(self.entries)


@dataclass(frozen=True,order=True)
class MaskCandidate:
    contract_key:str; role:str; clip_id:str; start:int; group_id:str
    legal_starts_sha256:str; selection_sha256:str
    action_path:str=""
    action_sha256:str=""
    action_shape:tuple[int,...]=()
    action_dtype:str=""
    droid_cache_index_path:str|None=None
    droid_cache_index_sha256:str|None=None
    droid_cache_root:str|None=None
    @property
    def identity(self): return (self.contract_key,self.role,self.clip_id,self.start)

@dataclass(frozen=True)
class RobotMaskConfig:
    prompt:str=PROMPT; anchor_indices:tuple[int,...]=(0,8,16)
    anchor_search_radius:int=2
    max_anchor_sequence_attempts:int=8
    box_threshold:float=.25; text_threshold:float=.25
    min_box_area_ratio:float=.0005; max_box_area_ratio:float=.38
    max_fine_grained_box_area_ratio:float=.75
    min_mask_area_ratio:float=.0005; max_mask_area_ratio:float=.40
    max_components:int=8; max_disconnected_fraction:float=.02; min_largest_component_fraction:float=.70
    min_direction_iou:float=.35; min_temporal_iou:float=.10
    min_anchor_box_iou:float=.03; max_centroid_jump:float=.30
    max_bbox_jump:float=.40; min_object_score_logit:float=-2.
    def validate(self):
        if self.prompt!=PROMPT or self.anchor_indices!=(0,8,16): raise RobotMaskCacheError("fixed prompt/anchors mismatch")
        if self.anchor_search_radius!=2:
            raise RobotMaskCacheError("anchor_search_radius must be exactly 2")
        numeric=("box_threshold","text_threshold","min_box_area_ratio","max_box_area_ratio",
                 "max_fine_grained_box_area_ratio",
                 "min_mask_area_ratio","max_mask_area_ratio","max_disconnected_fraction",
                 "min_largest_component_fraction","min_direction_iou","min_temporal_iou",
                 "min_anchor_box_iou","max_centroid_jump","max_bbox_jump","min_object_score_logit")
        if any(isinstance(getattr(self,name),bool) or not math.isfinite(float(getattr(self,name))) for name in numeric):
            raise RobotMaskCacheError("all numeric thresholds must be finite real values")
        if not 0<=self.box_threshold<=1 or not 0<=self.text_threshold<=1:
            raise RobotMaskCacheError("detector thresholds must be in [0,1]")
        if self.box_threshold!=.25 or self.text_threshold!=.25: raise RobotMaskCacheError("registered thresholds must be 0.25")
        for kind in ("box","mask"):
            if not 0<getattr(self,f"min_{kind}_area_ratio")<getattr(self,f"max_{kind}_area_ratio")<1:
                raise RobotMaskCacheError(f"invalid {kind} area limits")
        if not self.max_box_area_ratio<self.max_fine_grained_box_area_ratio<1:
            raise RobotMaskCacheError("invalid fine-grained proposal area limit")
        if self.max_anchor_sequence_attempts!=8:
            raise RobotMaskCacheError("max_anchor_sequence_attempts must be exactly 8")
        if not isinstance(self.max_components,int) or isinstance(self.max_components,bool) or self.max_components<1:
            raise RobotMaskCacheError("max_components must be a positive integer")
        if not 0<=self.max_disconnected_fraction<1 or not 0<self.min_largest_component_fraction<=1:
            raise RobotMaskCacheError("invalid connectivity thresholds")
        if self.max_disconnected_fraction>1-self.min_largest_component_fraction:
            raise RobotMaskCacheError("disconnected fraction conflicts with largest-component threshold")
        for name in ("min_direction_iou","min_temporal_iou","min_anchor_box_iou"):
            if not 0<=getattr(self,name)<=1: raise RobotMaskCacheError(f"{name} must be in [0,1]")
        if not 0<self.max_centroid_jump<=math.sqrt(2) or not 0<self.max_bbox_jump<=1:
            raise RobotMaskCacheError("invalid centroid/bbox jump limits")

def _canonical(v,pretty=False): return (json.dumps(v,sort_keys=True,indent=2 if pretty else None,separators=None if pretty else (",",":"),allow_nan=False)+"\n").encode()
def _ph(v): return hashlib.sha256(_canonical(v)).hexdigest()
def sha256_file(path):
    h=hashlib.sha256()
    with _open_regular(path,"hashed file") as stream:
        before=os.fstat(stream.fileno())
        for b in iter(lambda:stream.read(4<<20),b""): h.update(b)
        after=os.fstat(stream.fileno())
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(
                after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise RobotMaskCacheError(f"hashed file changed while reading: {path}")
    return h.hexdigest()
def _input_fingerprint(path,label):
    p=Path(path).absolute(); digest=sha256_file(p); metadata=os.stat(p,follow_symlinks=False)
    return {"path":str(p),"sha256":digest,"device":int(metadata.st_dev),"inode":int(metadata.st_ino),
            "size":int(metadata.st_size),"mtime_ns":int(metadata.st_mtime_ns),"ctime_ns":int(metadata.st_ctime_ns)}
def _assert_inputs_unchanged(fingerprints):
    for label,expected in fingerprints.items():
        if _input_fingerprint(expected["path"],label)!=expected:
            raise RobotMaskCacheError(f"{label} changed during robot-mask build")
def _ah(a):
    a=np.ascontiguousarray(a); h=hashlib.sha256(f"{a.dtype}|{a.shape}|".encode()); h.update(memoryview(a).cast("B")); return h.hexdigest()

def _no_symlinks(path,label,allow_missing=False):
    p=Path(path).absolute(); cur=Path(p.anchor)
    for i,part in enumerate(p.parts[1:]):
        cur/=part
        try: info=os.lstat(cur)
        except FileNotFoundError:
            if allow_missing: continue
            raise RobotMaskCacheError(f"missing {label}: {p}")
        if stat.S_ISLNK(info.st_mode): raise RobotMaskCacheError(f"{label} path contains symlink: {cur}")
def _regular(path,label):
    _no_symlinks(path,label)
    if not stat.S_ISREG(os.lstat(path).st_mode): raise RobotMaskCacheError(f"{label} is not regular: {path}")
def _directory(path,label):
    _no_symlinks(path,label)
    if not Path(path).is_dir(): raise RobotMaskCacheError(f"{label} is not directory: {path}")
def _json(path,label):
    try:
        with _open_regular(path,label) as stream: v=json.load(stream)
    except RobotMaskCacheError: raise
    except Exception as e: raise RobotMaskCacheError(f"invalid {label}: {path}") from e
    if not isinstance(v,dict): raise RobotMaskCacheError(f"invalid {label}")
    return v

@contextmanager
def _open_regular(path,label):
    _regular(path,label)
    flags=os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)
    try: fd=os.open(path,flags)
    except OSError as exc: raise RobotMaskCacheError(f"cannot open {label} without following links: {path}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode): raise RobotMaskCacheError(f"{label} changed type: {path}")
        with os.fdopen(fd,"rb",closefd=False) as stream: yield stream
    finally: os.close(fd)

def _read_manifest_regular(path,label):
    try:
        with _open_regular(path,label) as stream:
            lines=stream.read().decode("utf-8").splitlines()
        known=set(OXEClipRecord.__dataclass_fields__)
        return [OXEClipRecord(**{key:value for key,value in json.loads(line).items() if key in known})
                for raw in lines if (line:=raw.strip())]
    except RobotMaskCacheError: raise
    except (UnicodeDecodeError,json.JSONDecodeError,TypeError) as exc:
        raise RobotMaskCacheError(f"invalid {label}: {path}") from exc

def _npy_meta(path,label):
    try:
        with _open_regular(path,label) as stream:
            version=np.lib.format.read_magic(stream)
            shape,fortran,dtype=np.lib.format._read_array_header(stream,version)
        return tuple(shape),np.dtype(dtype),bool(fortran)
    except RobotMaskCacheError: raise
    except Exception as exc: raise RobotMaskCacheError(f"invalid {label} NPY: {path}") from exc

def _load_npy(path,label):
    try:
        with _open_regular(path,label) as stream: return np.load(stream,allow_pickle=False)
    except RobotMaskCacheError: raise
    except Exception as exc: raise RobotMaskCacheError(f"invalid {label} NPY: {path}") from exc

def load_mask_candidates(split_path,manifest_path,cache_root,*,droid_cache_index=None):
    sp,mp,root=map(lambda x:Path(x).absolute(),(split_path,manifest_path,cache_root))
    split=_json(sp,"split"); _regular(mp,"manifest"); _directory(root,"cache root")
    if split.get("schema_version")!=SPLIT_SCHEMA or split.get("immutable") is not True or split.get("seed")!=1729 or split.get("derivation")!=SPLIT_DERIVATION:
        raise RobotMaskCacheError("split schema/seed/derivation mismatch")
    src=split.get("source_manifest")
    if not isinstance(src,dict) or set(src)!={"path","sha256"} or Path(src["path"]).absolute()!=mp or src["sha256"]!=sha256_file(mp):
        raise RobotMaskCacheError("split/source manifest binding mismatch")
    groups=split.get("groups")
    if not isinstance(groups,dict) or set(groups)!=EXPECTED_CONTRACTS: raise RobotMaskCacheError("split must contain exact six contracts")
    records={}; safe_seen={}
    for r in _read_manifest_regular(mp,"manifest"):
        safe=safe_clip_id(r.clip_id)
        if r.clip_id in records or safe in safe_seen: raise RobotMaskCacheError("duplicate manifest identity/safe-id")
        records[r.clip_id]=r; safe_seen[safe]=r.clip_id
    droid_split=frozen_contract_split_from_mapping(groups[FORMAL_DROID_CONTRACT_KEY])
    droid_clip_ids=tuple(
        cid for role in ("calibration","qualification","confirmation")
        for cid in getattr(droid_split,f"{role}_clip_ids")
    )
    droid_records=[records[cid] for cid in droid_clip_ids]
    if droid_cache_index is None:
        raise RobotMaskCacheError("formal six-domain mask cache requires droid_cache_index")
    droid_index_path=Path(droid_cache_index).absolute()
    droid_index=_json(droid_index_path,"DROID finalized cache index")
    droid_root=Path(str(droid_index.get("cache_root",""))).absolute()
    _directory(droid_root,"DROID finalized cache root")
    try:
        droid_sources=validate_formal_droid_cache_index(
            droid_records,cache_root=droid_root,index_path=droid_index_path,
        )
    except ActionCacheResolutionError as exc:
        raise RobotMaskCacheError(str(exc)) from exc
    droid_index_sha=sha256_file(droid_index_path)
    out=[]
    for key in sorted(groups):
        frozen=frozen_contract_split_from_mapping(groups[key])
        if frozen.contract_key!=key: raise RobotMaskCacheError("contract key mismatch")
        for role in ("calibration","qualification","confirmation"):
            for cid in getattr(frozen,f"{role}_clip_ids"):
                r=records.get(cid)
                if r is None or action_contract_key(r)!=key: raise RobotMaskCacheError(f"manifest contract mismatch: {cid}")
                safe=safe_clip_id(cid); rgb=root/"rgb_256"/f"{safe}.npy"
                if canonical_dataset_name(r.dataset)=="droid":
                    source=droid_sources[cid]["actions"]
                    action=Path(str(source["path"])).absolute()
                    index_path,index_sha,index_root=str(droid_index_path),droid_index_sha,str(droid_root)
                else:
                    action=root/"actions"/f"{safe}.npy"
                    index_path=index_sha=index_root=None
                rgb_shape,rgb_dtype,_=_npy_meta(rgb,"RGB")
                action_shape,action_dtype,_=_npy_meta(action,"action")
                if len(rgb_shape)!=4 or rgb_shape[-1]!=3 or rgb_dtype!=np.uint8 or not action_shape:
                    raise RobotMaskCacheError(f"invalid RGB/action cache shape: {cid}")
                action_sha=sha256_file(action)
                if canonical_dataset_name(r.dataset)=="droid" and action_sha!=str(source["sha256"]):
                    raise RobotMaskCacheError(f"DROID action hash mismatch: {cid}")
                b=bind_temporal_window(r,group_id=frozen.clip_to_group_id[cid],seed=1729,usable_frames=min(r.n_frames,int(rgb_shape[0])),n_action_frames=int(action_shape[0]),target_length=16)
                out.append(MaskCandidate(
                    key,role,cid,b.start,b.group_id,b.legal_starts_sha256,b.selection_sha256,
                    str(action),action_sha,tuple(int(v) for v in action_shape),str(action_dtype),
                    index_path,index_sha,index_root,
                ))
    ids=[x.identity for x in out]; windows=[(x.clip_id,x.start) for x in out]
    if len(out)!=576 or len(set(ids))!=576 or len(set(windows))!=576: raise RobotMaskCacheError("expected exact 576 unique identities")
    return tuple(out)

def _npz(arrays):
    result=BytesIO()
    with zipfile.ZipFile(result,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for k in sorted(arrays):
            s=BytesIO(); np.lib.format.write_array(s,np.asarray(arrays[k]),allow_pickle=False)
            i=zipfile.ZipInfo(f"{k}.npy",(1980,1,1,0,0,0)); i.compress_type=zipfile.ZIP_DEFLATED; i.external_attr=0o600<<16
            z.writestr(i,s.getvalue(),compresslevel=6)
    return result.getvalue()
def _name(x): return f"robot_mask_{hashlib.sha256('|'.join(map(str,x.identity)).encode()).hexdigest()}__s{x.start:08d}.npz"

IDENTITY_KEYS={"model_id","requested_revision","resolved_commit","snapshot_path","snapshot_tree_sha256",
               "config_sha256","processor_sha256","tokenizer_sha256","image_processor_sha256",
               "processor_files_sha256","tokenizer_files_sha256","image_processor_files_sha256"}
def _snapshot_tree_hash(root):
    root=Path(root); rows=[]
    for path in sorted(root.rglob("*"),key=lambda value:value.relative_to(root).as_posix()):
        if path.is_symlink() and path.resolve(strict=True).is_dir():
            raise RobotMaskCacheError(f"snapshot entry is a directory symlink: {path}")
        if path.is_dir(): continue
        target=path.resolve(strict=True)
        if not target.is_file(): raise RobotMaskCacheError(f"snapshot entry is not a regular file: {path}")
        rows.append({"path":path.relative_to(root).as_posix(),"sha256":sha256_file(target),"size":target.stat().st_size})
    return _ph(rows)

def _model_identity(detector,propagator):
    adapters={}; digest=hashlib.sha256(); manifest=[]; count=0
    registered={
        "detector":(FORMAL_GROUNDING_MODEL,FORMAL_GROUNDING_REVISION),
        "propagator":(FORMAL_SAM2_MODEL,FORMAL_SAM2_REVISION),
    }
    for owner,a in (("detector",detector),("propagator",propagator)):
        ident=dict(getattr(a,"identity_components",{}))
        expected_model,expected_revision=registered[owner]
        if (set(ident)!=IDENTITY_KEYS or ident.get("model_id")!=expected_model
                or ident.get("requested_revision")!=expected_revision
                or ident.get("resolved_commit")!=expected_revision):
            raise RobotMaskCacheError(f"invalid formal {owner} identity")
        snapshot=Path(ident["snapshot_path"]); _directory(snapshot,f"{owner} snapshot")
        if snapshot.name!=ident["resolved_commit"]:
            raise RobotMaskCacheError(f"invalid {owner} snapshot commit path")
        if ident["snapshot_tree_sha256"]!=_snapshot_tree_hash(snapshot): raise RobotMaskCacheError(f"{owner} snapshot tree hash mismatch")
        for key in ("config_sha256","processor_sha256","tokenizer_sha256","image_processor_sha256",
                    "processor_files_sha256","tokenizer_files_sha256","image_processor_files_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}",str(ident[key])): raise RobotMaskCacheError(f"invalid {owner} {key}")
        adapters[owner]=ident
        models=getattr(a,"models",None)
        if not isinstance(models,Mapping) or not models: raise RobotMaskCacheError(f"missing {owner} models")
        for mn,m in sorted(models.items()):
            for kind,getter in (("parameter",m.named_parameters),("buffer",m.named_buffers)):
                for n,t in sorted(getter(),key=lambda p:p[0]):
                    t=t.detach().contiguous(); row=[owner,mn,kind,n,str(t.dtype),list(t.shape)]
                    manifest.append(row); digest.update(_canonical(row)); raw=t.view(torch.uint8).reshape(-1)
                    for start in range(0,raw.numel(),16<<20): digest.update(memoryview(raw[start:start+(16<<20)].cpu().numpy()).cast("B"))
                    count+=t.numel()
    return {"adapters":adapters,"tensor_manifest_sha256":_ph(manifest),"parameter_and_buffer_content_sha256":digest.hexdigest(),"parameter_and_buffer_count":int(count)}
def _code_identity(detector,propagator):
    import transformers
    root=Path(__file__).resolve().parents[2]
    names=("wm3d_v3/stage1/robot_mask_cache.py","scripts/build_stage1_robot_mask_cache.py",
           "wm3d_v3/stage1/action_contract.py","wm3d_v3/stage1/action_contract_split.py",
           "wm3d_v3/stage1/action_contract_evidence.py","wm3d_v3/stage1/action_evidence_sources.py",
           "wm3d_v3/stage1/action_cache.py","wm3d_v3/stage1/droid_interval_action.py",
           "wm3d_v3/stage1/immutable_artifact.py","wm3d_v3/data/manifest.py")
    repo_files=[{"path":name,"sha256":sha256_file(root/name)} for name in names]
    implementations={}
    for adapter in (detector,propagator):
        implementation_files=tuple(getattr(adapter,"implementation_files",()))
        if not implementation_files:
            raise RobotMaskCacheError("formal mask adapter lacks implementation provenance")
        for raw in implementation_files:
            path=Path(raw).resolve(strict=True); _regular(path,"called implementation")
            implementations[str(path)]={"sha256":sha256_file(path),"size":path.stat().st_size}
    environment={"python_implementation":platform.python_implementation(),
                 "python_version":platform.python_version(),"numpy_version":np.__version__,
                 "transformers_version":transformers.__version__,"torch_version":torch.__version__,
                 "torch_cuda_version":torch.version.cuda}
    payload={"repo_files":repo_files,"implementation_files":[{"path":path,**value} for path,value in sorted(implementations.items())],
             "environment":environment}
    return {**payload,"tree_sha256":_ph(payload)}
def _source(item,root,hashes):
    p=root/"rgb_256"/f"{safe_clip_id(item.clip_id)}.npy"; rgb=_load_npy(p,"source RGB")
    if rgb.ndim!=4 or rgb.shape[-1]!=3 or rgb.dtype!=np.uint8 or item.start+17>rgb.shape[0]: raise RobotMaskCacheError("invalid/missing source RGB")
    w=np.array(rgb[item.start:item.start+17],copy=True,order="C")
    return w,{"path":str(p),"sha256":hashes.setdefault(p,sha256_file(p)),"window_sha256":_ah(w),"shape":list(rgb.shape),"dtype":str(rgb.dtype)}

def _fine_grained_tokens(label):
    tokens=tuple(re.findall(r"[a-z0-9]+",str(label).lower()))
    allowed=frozenset(("robot","arm","gripper"))
    if not tokens or any(token not in allowed for token in tokens):
        return frozenset()
    return frozenset(token for token in tokens if token in ("arm","gripper"))


def _is_robot_proposal_label(label):
    tokens=tuple(re.findall(r"[a-z0-9]+",str(label).lower()))
    allowed=frozenset(("robot","arm","gripper"))
    return (
        bool(tokens)
        and all(token in allowed for token in tokens)
        and ("robot" in tokens or "gripper" in tokens)
    )


def _proposal_area_cap(label,cfg):
    return (
        cfg.max_fine_grained_box_area_ratio
        if _fine_grained_tokens(label)
        else cfg.max_box_area_ratio
    )


def _accepted_detections(result,shape,cfg,anchor):
    boxes=np.asarray(result.get("boxes",[]),np.float32); scores=np.asarray(result.get("scores",[]),np.float32).reshape(-1)
    labels=[str(x).lower().strip() for x in result.get("text_labels",[])]
    boxes=boxes.reshape(-1,4) if boxes.size else np.empty((0,4),np.float32)
    if len(boxes)!=len(scores) or len(boxes)!=len(labels): raise RobotMaskCacheError("detector result shape mismatch")
    h,w=shape; accepted=[]
    for box,score,label in zip(boxes,scores,labels):
        x1,y1,x2,y2=box.tolist(); box=np.array([max(x1,0),max(y1,0),min(x2,w),min(y2,h)],np.float32)
        area=max(float(box[2]-box[0]),0)*max(float(box[3]-box[1]),0); ratio=area/(h*w)
        if np.isfinite(box).all() and np.isfinite(score) and cfg.box_threshold<=score<=1 and cfg.min_box_area_ratio<=ratio<=_proposal_area_cap(label,cfg) and _is_robot_proposal_label(label):
            accepted.append((float(score)*np.sqrt(area),float(score),label,tuple(box.tolist())))
    if not accepted: raise RobotMaskCacheError(f"no reliable robot/gripper detection at anchor {anchor}")
    return [
        {"box":np.array(b,np.float32),"score":s,"label":l,"utility":u}
        for u,s,l,b in sorted(accepted,key=lambda x:(-x[0],-x[1],x[2],x[3]))
    ]


def _anchor_search_frames(nominal,cfg):
    offsets=(0,-1,1,-2,2)
    return tuple(
        nominal+offset for offset in offsets
        if abs(offset)<=cfg.anchor_search_radius and 0<=nominal+offset<17
    )


def _detect_anchor_candidate_lists(frames,detector,cfg):
    shape=tuple(frames.shape[1:3]); cached={}; actual=[]; candidate_lists=[]
    for nominal in cfg.anchor_indices:
        selected=None
        for frame_index in _anchor_search_frames(nominal,cfg):
            if frame_index not in cached:
                cached[frame_index]=detector.detect(
                    frames[frame_index],cfg.prompt,
                    box_threshold=cfg.box_threshold,
                    text_threshold=cfg.text_threshold,
                )
            try:
                selected=_accepted_detections(
                    cached[frame_index],shape,cfg,frame_index
                )
            except RobotMaskCacheError as exc:
                if not str(exc).startswith("no reliable robot/gripper detection"):
                    raise
                continue
            actual.append(frame_index); candidate_lists.append(selected); break
        if selected is None:
            raise RobotMaskCacheError(
                f"no reliable robot/gripper detection near anchor {nominal}"
            )
    if len(actual)!=3 or not actual[0]<actual[1]<actual[2]:
        raise RobotMaskCacheError("invalid resolved anchor ordering")
    return tuple(actual),candidate_lists


def _select(result,shape,cfg,anchor):
    return _accepted_detections(result,shape,cfg,anchor)[0]


def _rank_consistent_anchor_sequences(results,shape,cfg):
    candidates=[
        _accepted_detections(result,shape,cfg,anchor)
        for anchor,result in zip(cfg.anchor_indices,results)
    ]
    return _rank_consistent_anchor_candidate_lists(candidates)


def _rank_consistent_anchor_candidate_lists(candidates):
    if len(candidates)!=3 or any(not values for values in candidates):
        raise RobotMaskCacheError("invalid anchor candidate inventory")
    ranked=[]
    for sequence in itertools.product(*candidates):
        common=set(_fine_grained_tokens(sequence[0]["label"]))
        for item in sequence[1:]: common.intersection_update(_fine_grained_tokens(item["label"]))
        areas=[]
        for item in sequence:
            x1,y1,x2,y2=(float(value) for value in item["box"])
            areas.append(max(x2-x1,1e-6)*max(y2-y1,1e-6))
        area_span=max(math.log(area) for area in areas)-min(math.log(area) for area in areas)
        key=(
            int(bool(common)),
            -area_span,
            int("gripper" in common),
            sum(float(item["score"]) for item in sequence),
            sum(float(item["utility"]) for item in sequence),
            tuple((item["label"],tuple(float(value) for value in item["box"])) for item in sequence),
        )
        ranked.append((key,sequence))
    if not ranked: raise RobotMaskCacheError("no consistent anchor detection sequence")
    ranked.sort(key=lambda item:item[0],reverse=True)
    return [list(sequence) for _,sequence in ranked]


def _select_consistent_anchor_sequence(results,shape,cfg):
    return _rank_consistent_anchor_sequences(results,shape,cfg)[0]
def _iou(a,b):
    union=np.logical_or(a,b).sum(); return float(np.logical_and(a,b).sum()/union) if union else 0.
def _components(mask):
    seen=np.zeros_like(mask,bool); count=largest=0
    for y,x in np.argwhere(mask):
        if seen[y,x]: continue
        count+=1; size=0; stack=[(int(y),int(x))]; seen[y,x]=1
        while stack:
            cy,cx=stack.pop(); size+=1
            for ny,nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                if 0<=ny<mask.shape[0] and 0<=nx<mask.shape[1] and mask[ny,nx] and not seen[ny,nx]: seen[ny,nx]=1; stack.append((ny,nx))
        largest=max(largest,size)
    return count,largest


def _temporal_geometry_valid(tiou,centroid_jump,bbox_jump,cfg):
    low_iou=np.asarray(tiou)<cfg.min_temporal_iou
    sustained_low_iou=(low_iou[:-1]&low_iou[1:]).any() if len(low_iou)>1 else False
    return not (
        sustained_low_iou
        or np.any(np.asarray(centroid_jump)>cfg.max_centroid_jump)
        or np.any(np.asarray(bbox_jump)>cfg.max_bbox_jump)
    )

PROP_KEYS={"forward_masks","backward_masks","forward_anchor_masks","backward_anchor_masks",
           "forward_object_logits","backward_object_logits",
           "forward_anchor_object_logits","backward_anchor_object_logits"}
RAW_KEYS=("forward_masks","backward_masks","forward_anchor_masks","backward_anchor_masks",
          "forward_object_logits","backward_object_logits",
          "forward_anchor_object_logits","backward_anchor_object_logits")
DERIVED_KEYS=("mask","object_score_logits","direction_iou","temporal_iou",
              "anchor_box_iou","centroids","bboxes")


def _validate_propagation_structure(raw):
    if not isinstance(raw,Mapping) or set(raw)!=PROP_KEYS:
        raise RobotMaskCacheError("invalid propagation result")
    f=np.asarray(raw["forward_masks"],np.float32)
    b=np.asarray(raw["backward_masks"],np.float32)
    fa=np.asarray(raw["forward_anchor_masks"],np.float32)
    ba=np.asarray(raw["backward_anchor_masks"],np.float32)
    fl=np.asarray(raw["forward_object_logits"],np.float32)
    bl=np.asarray(raw["backward_object_logits"],np.float32)
    fal=np.asarray(raw["forward_anchor_object_logits"],np.float32)
    bal=np.asarray(raw["backward_anchor_object_logits"],np.float32)
    if f.ndim!=3 or f.shape[0]!=17 or b.shape!=f.shape:
        raise RobotMaskCacheError("both directions must cover every frame")
    shape=f.shape[1:]
    if (fa.shape!=(3,*shape) or ba.shape!=fa.shape or fl.shape!=(17,)
            or bl.shape!=(17,) or fal.shape!=(3,) or bal.shape!=(3,)):
        raise RobotMaskCacheError("propagation evidence shape mismatch")
    if not all(np.isfinite(value).all() for value in (f,b,fa,ba,fl,bl,fal,bal)):
        raise RobotMaskCacheError("non-finite propagation evidence")


def _quality(raw,anchor_boxes,cfg,anchor_indices=None):
    anchor_indices=tuple(cfg.anchor_indices if anchor_indices is None else anchor_indices)
    if (len(anchor_indices)!=3 or not anchor_indices[0]<anchor_indices[1]<anchor_indices[2]
            or any(not 0<=index<17 for index in anchor_indices)):
        raise RobotMaskCacheError("anchor index evidence invalid")
    f=np.asarray(raw["forward_masks"],np.float32); b=np.asarray(raw["backward_masks"],np.float32)
    fa=np.asarray(raw["forward_anchor_masks"],np.float32); ba=np.asarray(raw["backward_anchor_masks"],np.float32)
    fl=np.asarray(raw["forward_object_logits"],np.float32); bl=np.asarray(raw["backward_object_logits"],np.float32)
    fal=np.asarray(raw["forward_anchor_object_logits"],np.float32); bal=np.asarray(raw["backward_anchor_object_logits"],np.float32)
    if f.ndim!=3 or f.shape[0]!=17 or b.shape!=f.shape: raise RobotMaskCacheError("both directions must cover every frame")
    shape=f.shape[1:]
    if fa.shape!=(3,*shape) or ba.shape!=fa.shape or fl.shape!=(17,) or bl.shape!=(17,) or fal.shape!=(3,) or bal.shape!=(3,):
        raise RobotMaskCacheError("propagation evidence shape mismatch")
    if not all(np.isfinite(x).all() for x in (f,b,fa,ba,fl,bl,fal,bal)): raise RobotMaskCacheError("non-finite propagation evidence")
    fb,bb,fab,bab=f>0,b>0,fa>0,ba>0
    diou=np.asarray([_iou(fb[i],bb[i]) for i in range(17)],np.float32)
    if np.any(diou<cfg.min_direction_iou): raise RobotMaskCacheError("direction consistency failure")
    votes=fb.astype(np.uint8)+bb.astype(np.uint8); counts=np.full(17,2,np.uint8)
    for i,a in enumerate(anchor_indices):
        votes[a]+=fab[i].astype(np.uint8)+bab[i].astype(np.uint8)
        counts[a]=4
    masks=votes>(counts[:,None,None]//2)
    logits=np.minimum(fl,bl)
    anchor_min=np.minimum(fal,bal)
    logits[list(anchor_indices)]=np.minimum(logits[list(anchor_indices)],anchor_min)
    if np.any(logits<cfg.min_object_score_logit): raise RobotMaskCacheError("object logits failure")
    h,w=shape; cent=[]; boxes=[]; tiou=[]
    for i,m in enumerate(masks):
        area=int(m.sum()); comp,largest=_components(m)
        if (not area or not cfg.min_mask_area_ratio<=area/(h*w)<=cfg.max_mask_area_ratio
                or largest/area<cfg.min_largest_component_fraction
                or (comp>cfg.max_components and 1-largest/area>cfg.max_disconnected_fraction)):
            raise RobotMaskCacheError(f"mask area/connectivity failure {i}")
        pts=np.argwhere(m); cent.append([pts[:,1].mean()/w,pts[:,0].mean()/h])
        boxes.append([pts[:,1].min()/w,pts[:,0].min()/h,(pts[:,1].max()+1)/w,(pts[:,0].max()+1)/h])
        if i: tiou.append(_iou(masks[i-1],m))
    cent=np.asarray(cent,np.float32); boxes=np.asarray(boxes,np.float32); tiou=np.asarray(tiou,np.float32)
    centroid_jump=np.linalg.norm(np.diff(cent,axis=0),axis=1); bbox_jump=np.max(np.abs(np.diff(boxes,axis=0)),axis=1)
    if not _temporal_geometry_valid(tiou,centroid_jump,bbox_jump,cfg):
        raise RobotMaskCacheError("temporal geometry failure")
    anchor_boxes=np.asarray(anchor_boxes,np.float32)
    if anchor_boxes.shape!=(3,4) or not np.isfinite(anchor_boxes).all(): raise RobotMaskCacheError("anchor box evidence invalid")
    abi=[]
    for i,a in enumerate(anchor_indices):
        bm=np.zeros(shape,bool); x1,y1,x2,y2=np.rint(anchor_boxes[i]).astype(int)
        bm[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]=1; abi.append(_iou(masks[a],bm))
    abi=np.asarray(abi,np.float32)
    if np.any(abi<cfg.min_anchor_box_iou): raise RobotMaskCacheError("anchor mask-box IoU failure")
    normalized={key:np.asarray(raw[key],np.float32) for key in RAW_KEYS}
    return {**normalized,"mask":masks.astype(np.uint8),"object_score_logits":logits.astype(np.float32),
            "direction_iou":diou,"temporal_iou":tiou,"anchor_box_iou":abi,
            "centroids":cent,"bboxes":boxes}


QUALITY_FAILURE_PREFIXES=(
    ("direction consistency failure","direction_consistency"),
    ("object logits failure","object_logits"),
    ("mask area/connectivity failure","mask_area_connectivity"),
    ("temporal geometry failure","temporal_geometry"),
    ("anchor box evidence invalid","anchor_box_evidence"),
    ("anchor mask-box IoU failure","anchor_mask_box_iou"),
)
QUALITY_FAILURE_CODES=frozenset(code for _,code in QUALITY_FAILURE_PREFIXES)


def _quality_failure_code(error):
    message=str(error)
    for prefix,code in QUALITY_FAILURE_PREFIXES:
        if message.startswith(prefix):
            return code
    return None


def _anchor_proposal_ledger(candidate_lists):
    flat=[]; offsets=[0]
    for candidates in candidate_lists:
        flat.extend(candidates); offsets.append(len(flat))
    return {
        "anchor_proposal_offsets":np.asarray(offsets,np.int64),
        "anchor_proposal_boxes_xyxy":np.stack(
            [item["box"] for item in flat]
        ).astype(np.float32),
        "anchor_proposal_scores":np.asarray(
            [item["score"] for item in flat],np.float32
        ),
        "anchor_proposal_utilities":np.asarray(
            [item["utility"] for item in flat],np.float32
        ),
        "anchor_proposal_labels":np.asarray([item["label"] for item in flat]),
    }


def _infer(frames,detector,propagator,cfg):
    anchor_indices,candidate_lists=_detect_anchor_candidate_lists(
        frames,detector,cfg
    )
    sequences=_rank_consistent_anchor_candidate_lists(candidate_lists)
    proposal_ledger=_anchor_proposal_ledger(candidate_lists)
    failures=[]
    for attempt_index,selections in enumerate(
            sequences[:cfg.max_anchor_sequence_attempts],start=1):
        boxes=np.stack([item["box"] for item in selections]).astype(np.float32)
        raw=propagator.propagate(
            frames,
            {anchor:item["box"] for anchor,item in zip(anchor_indices,selections)},
        )
        _validate_propagation_structure(raw)
        try:
            quality=_quality(raw,boxes,cfg,anchor_indices)
        except RobotMaskCacheError as exc:
            failure_code=_quality_failure_code(exc)
            if failure_code is None:
                raise
            failures.append(failure_code)
            continue
        return {**quality,**proposal_ledger,
                "anchor_indices":np.asarray(anchor_indices,np.int64),
                "anchor_boxes_xyxy":boxes,
                "anchor_scores":np.asarray([item["score"] for item in selections],np.float32),
                "anchor_utilities":np.asarray([item["utility"] for item in selections],np.float32),
                "anchor_labels":np.asarray([item["label"] for item in selections]),
                "anchor_sequence_rank":np.asarray(attempt_index,np.int64),
                "anchor_sequence_attempts":np.asarray(attempt_index,np.int64),
                "anchor_sequence_candidate_count":np.asarray(len(sequences),np.int64),
                "anchor_sequence_failure_codes":np.asarray(failures,dtype=np.str_)}
    raise RobotMaskCacheError(
        "no ranked anchor proposal sequence passed quality: "+"; ".join(
            code.replace("_"," ") for code in failures
        )
    )

NPZ_KEYS=set(RAW_KEYS)|set(DERIVED_KEYS)|{"anchor_indices","anchor_boxes_xyxy","anchor_scores",
          "anchor_utilities","anchor_labels","anchor_proposal_offsets",
          "anchor_proposal_boxes_xyxy","anchor_proposal_scores",
          "anchor_proposal_utilities","anchor_proposal_labels","anchor_sequence_rank",
          "anchor_sequence_attempts","anchor_sequence_candidate_count",
          "anchor_sequence_failure_codes","start","frame_indices","source_rgb_sha256",
          "source_rgb_window_sha256","source_action_sha256","droid_cache_index_sha256",
          "split_sha256","manifest_sha256","model_identity_sha256","config_sha256",
          "code_sha256","mask_sha256"}


def _validate_anchor_proposal_ledger(arrays,h,w,cfg):
    offsets=arrays["anchor_proposal_offsets"]
    boxes=arrays["anchor_proposal_boxes_xyxy"]
    scores=arrays["anchor_proposal_scores"]
    utilities=arrays["anchor_proposal_utilities"]
    labels=arrays["anchor_proposal_labels"]
    count=len(scores) if scores.ndim==1 else -1
    if (offsets.shape!=(4,) or offsets.dtype!=np.int64
            or offsets[0]!=0 or offsets[-1]!=count
            or np.any(np.diff(offsets)<=0)
            or boxes.shape!=(count,4) or boxes.dtype!=np.float32
            or scores.shape!=(count,) or scores.dtype!=np.float32
            or utilities.shape!=(count,) or utilities.dtype!=np.float32
            or labels.shape!=(count,) or labels.dtype.kind not in ("U","S")
            or not all(np.isfinite(value).all() for value in (boxes,scores,utilities))):
        raise RobotMaskCacheError("NPZ anchor sequence audit conflict")
    candidate_lists=[]
    for start,end in zip(offsets[:-1],offsets[1:]):
        candidates=[]
        for index in range(int(start),int(end)):
            box=boxes[index]; score=float(scores[index]); utility=float(utilities[index])
            label=str(labels[index]); x1,y1,x2,y2=(float(value) for value in box)
            area=(x2-x1)*(y2-y1); ratio=area/(h*w)
            if not (0<=x1<x2<=w and 0<=y1<y2<=h
                    and cfg.box_threshold<=score<=1 and utility>=0
                    and _is_robot_proposal_label(label)
                    and cfg.min_box_area_ratio<=ratio<=_proposal_area_cap(label,cfg)
                    and np.isclose(utility,score*math.sqrt(area),rtol=1e-5,atol=1e-5)):
                raise RobotMaskCacheError("NPZ anchor sequence audit conflict")
            candidates.append({"box":box.copy(),"score":score,"label":label,"utility":utility})
        order=sorted(
            range(len(candidates)),
            key=lambda index:(
                -candidates[index]["utility"],-candidates[index]["score"],
                candidates[index]["label"],
                tuple(float(value) for value in candidates[index]["box"]),
            ),
        )
        if order!=list(range(len(candidates))):
            raise RobotMaskCacheError("NPZ anchor sequence audit conflict")
        candidate_lists.append(candidates)
    return candidate_lists


def _validate_npz(path,entry,cfg):
    _regular(path,"NPZ")
    if sha256_file(path)!=entry["output"]["sha256"]: raise RobotMaskCacheError("output hash conflict")
    try:
        with _open_regular(path,"NPZ") as stream:
            with np.load(stream,allow_pickle=False) as data:
                if set(data.files)!=NPZ_KEYS: raise RobotMaskCacheError("NPZ key conflict")
                arrays={key:np.asarray(data[key]).copy() for key in data.files}
    except RobotMaskCacheError: raise
    except Exception as exc: raise RobotMaskCacheError(f"invalid NPZ: {path}") from exc
    mask=arrays["mask"]
    if mask.dtype!=np.uint8 or mask.ndim!=3 or mask.shape[0]!=17 or min(mask.shape[1:])<1 or not np.isin(mask,[0,1]).all():
        raise RobotMaskCacheError("NPZ mask shape/dtype/binary conflict")
    h,w=mask.shape[1:]
    output=entry.get("output",{})
    if output.get("shape")!=list(mask.shape) or output.get("dtype")!="uint8":
        raise RobotMaskCacheError("NPZ output metadata conflict")
    shapes={"forward_masks":(17,h,w),"backward_masks":(17,h,w),
            "forward_anchor_masks":(3,h,w),"backward_anchor_masks":(3,h,w),
            "forward_object_logits":(17,),"backward_object_logits":(17,),
            "forward_anchor_object_logits":(3,),"backward_anchor_object_logits":(3,),
            "anchor_boxes_xyxy":(3,4),"anchor_scores":(3,),"anchor_utilities":(3,),
            "object_score_logits":(17,),"direction_iou":(17,),"temporal_iou":(16,),
            "anchor_box_iou":(3,),"centroids":(17,2),"bboxes":(17,4)}
    for key,shape in shapes.items():
        value=arrays[key]
        if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():
            raise RobotMaskCacheError(f"NPZ {key} shape/dtype/finite conflict")
    labels=arrays["anchor_labels"]
    if labels.shape!=(3,) or labels.dtype.kind not in ("U","S") or any(not _is_robot_proposal_label(x) for x in labels):
        raise RobotMaskCacheError("NPZ anchor labels conflict")
    anchor_indices=arrays["anchor_indices"]
    if (anchor_indices.shape!=(3,) or anchor_indices.dtype!=np.int64
            or not int(anchor_indices[0])<int(anchor_indices[1])<int(anchor_indices[2])
            or any(
                int(actual) not in _anchor_search_frames(nominal,cfg)
                for actual,nominal in zip(anchor_indices,cfg.anchor_indices)
            )):
        raise RobotMaskCacheError("NPZ anchor indices conflict")
    if arrays["frame_indices"].dtype!=np.int64 or arrays["frame_indices"].tolist()!=entry["frame_indices"]:
        raise RobotMaskCacheError("NPZ frame indices conflict")
    if (arrays["start"].shape!=() or arrays["start"].dtype!=np.int64
            or int(arrays["start"])!=entry["start"]
            or entry["frame_indices"]!=list(range(entry["start"],entry["start"]+17))):
        raise RobotMaskCacheError("NPZ start dtype conflict")
    boxes=arrays["anchor_boxes_xyxy"]; scores=arrays["anchor_scores"]; utilities=arrays["anchor_utilities"]
    for i,(box,score,label,utility) in enumerate(zip(boxes,scores,labels,utilities)):
        x1,y1,x2,y2=box.tolist(); area=(x2-x1)*(y2-y1); ratio=area/(h*w)
        if not (0<=x1<x2<=w and 0<=y1<y2<=h and cfg.box_threshold<=score<=1 and utility>=0
                and cfg.min_box_area_ratio<=ratio<=_proposal_area_cap(label,cfg)):
            raise RobotMaskCacheError(f"NPZ selected anchor gate failure {i}")
        if not np.isclose(utility,float(score)*math.sqrt(area),rtol=1e-5,atol=1e-5):
            raise RobotMaskCacheError("NPZ anchor utility conflict")
    rank=arrays["anchor_sequence_rank"]
    attempts=arrays["anchor_sequence_attempts"]
    candidate_count=arrays["anchor_sequence_candidate_count"]
    failure_codes=arrays["anchor_sequence_failure_codes"]
    candidate_lists=_validate_anchor_proposal_ledger(arrays,h,w,cfg)
    ranked=_rank_consistent_anchor_candidate_lists(candidate_lists)
    if (rank.shape!=() or rank.dtype!=np.int64
            or attempts.shape!=() or attempts.dtype!=np.int64
            or candidate_count.shape!=() or candidate_count.dtype!=np.int64
            or not 1<=int(rank)==int(attempts)<=cfg.max_anchor_sequence_attempts
            or int(candidate_count)!=len(ranked)
            or failure_codes.shape!=(int(attempts)-1,)
            or failure_codes.dtype.kind not in ("U","S")
            or any(str(code) not in QUALITY_FAILURE_CODES for code in failure_codes)):
        raise RobotMaskCacheError("NPZ anchor sequence audit conflict")
    selected=ranked[int(rank)-1]
    if (not np.allclose(boxes,np.stack([item["box"] for item in selected]),rtol=0,atol=0)
            or not np.allclose(scores,np.asarray([item["score"] for item in selected]),rtol=1e-6,atol=1e-6)
            or not np.allclose(utilities,np.asarray([item["utility"] for item in selected]),rtol=1e-6,atol=1e-6)
            or labels.tolist()!=[item["label"] for item in selected]):
        raise RobotMaskCacheError("NPZ anchor sequence audit conflict")
    recomputed=_quality(
        {key:arrays[key] for key in RAW_KEYS},boxes,cfg,
        tuple(int(value) for value in anchor_indices),
    )
    for key in DERIVED_KEYS:
        if arrays[key].shape!=recomputed[key].shape or not np.allclose(arrays[key],recomputed[key],rtol=1e-6,atol=1e-6):
            raise RobotMaskCacheError(f"NPZ recomputed quality conflict: {key}")
    checks={"start":entry["start"],"source_rgb_sha256":entry["source_rgb"]["sha256"],
            "source_rgb_window_sha256":entry["source_rgb"]["window_sha256"],
            "source_action_sha256":entry["source_action"]["sha256"],
            "droid_cache_index_sha256":entry["droid_cache_index_sha256"] or "",
            "split_sha256":entry["split_sha256"],"manifest_sha256":entry["manifest_sha256"],
            "model_identity_sha256":entry["model_identity_sha256"],"config_sha256":entry["config_sha256"],
            "code_sha256":entry["code_sha256"],"mask_sha256":entry["output"]["mask_sha256"]}
    for key,value in checks.items():
        if arrays[key].shape!=() or str(arrays[key].item())!=str(value):
            raise RobotMaskCacheError(f"NPZ scalar identity conflict: {key}")
    for key in ("source_rgb_sha256","source_rgb_window_sha256","source_action_sha256",
                "split_sha256","manifest_sha256","model_identity_sha256","config_sha256",
                "code_sha256","mask_sha256"):
        if arrays[key].dtype.kind not in ("U","S") or not re.fullmatch(r"[0-9a-f]{64}",str(arrays[key].item())):
            raise RobotMaskCacheError(f"NPZ scalar hash conflict: {key}")
    droid_hash=str(arrays["droid_cache_index_sha256"].item())
    if arrays["droid_cache_index_sha256"].dtype.kind not in ("U","S") or (
            droid_hash!="" and not re.fullmatch(r"[0-9a-f]{64}",droid_hash)):
        raise RobotMaskCacheError("NPZ DROID index hash conflict")
    if _ah(mask)!=entry["output"]["mask_sha256"]: raise RobotMaskCacheError("NPZ mask hash conflict")

@contextmanager
def _lock(root):
    _no_symlinks(root.parent,"output parent"); root.mkdir(exist_ok=True); _directory(root,"output root")
    p=root/".robot_mask_cache.lock"
    if p.is_symlink(): raise RobotMaskCacheError("lock symlink")
    fd=os.open(p,os.O_CREAT|os.O_RDWR|getattr(os,"O_NOFOLLOW",0),0o600)
    try: fcntl.flock(fd,fcntl.LOCK_EX); yield
    finally: fcntl.flock(fd,fcntl.LOCK_UN); os.close(fd)

def _base(item,source,split_sha,manifest_sha,model_sha,config_sha,code_sha):
    source_action={"path":item.action_path,"sha256":item.action_sha256,
                   "shape":list(item.action_shape),"dtype":item.action_dtype}
    return {"identity":list(item.identity),"group_id":item.group_id,
            "legal_starts_sha256":item.legal_starts_sha256,
            "selection_sha256":item.selection_sha256,"start":item.start,
            "frame_indices":list(range(item.start,item.start+17)),
            "source_rgb":source,"source_action":source_action,
            "droid_cache_index_sha256":item.droid_cache_index_sha256,
            "split_sha256":split_sha,"manifest_sha256":manifest_sha,
            "model_identity_sha256":model_sha,"config_sha256":config_sha,
            "code_sha256":code_sha}
def _validate_index(path,expected,root,cfg):
    a=_json(path,"index")
    for k in ("schema_version","immutable","split_artifact","source_manifest","droid_cache_index","config","config_sha256","code","code_sha256","model_identity","model_identity_sha256","expected_identity_sha256"):
        if a.get(k)!=expected.get(k): raise RobotMaskCacheError(f"index conflict {k}")
    entries=a.get("entries"); expected_map={tuple(x["identity"]):x for x in expected["entries"]}
    if not isinstance(entries,list) or len(entries)!=576 or {tuple(x.get("identity",())) for x in entries}!=set(expected_map): raise RobotMaskCacheError("index identity set conflict")
    names=set(); descriptors=[]
    for e in entries:
        base=expected_map[tuple(e["identity"])]
        if any(e.get(k)!=v for k,v in base.items()): raise RobotMaskCacheError("index binding conflict")
        x=MaskCandidate(*e["identity"],e["group_id"],e["legal_starts_sha256"],e["selection_sha256"]); n=_name(x)
        if "output" in e:
            if set(e)!={*base,"output"} or e["output"].get("path")!=n or n in names:
                raise RobotMaskCacheError("fixed output name conflict")
            names.add(n); _validate_npz(root/n,e,cfg); descriptors.append(e["output"])
        elif "non_informative" in e:
            if set(e)!={*base,"non_informative"}:
                raise RobotMaskCacheError("non-informative entry schema conflict")
            _validate_non_informative(e["non_informative"],cfg); descriptors.append(e["non_informative"])
        else:
            raise RobotMaskCacheError("entry result missing")
    if {p.name for p in root.glob("*.npz")}!=names or a.get("output_hashes_sha256")!=_ph(descriptors): raise RobotMaskCacheError("output inventory/hash conflict")
    return {"status":"skipped","window_count":576,"index":str(path),"index_sha256":sha256_file(path)}

def _freeze_consumer(value):
    if isinstance(value,dict):
        return MappingProxyType({key:_freeze_consumer(item) for key,item in value.items()})
    if isinstance(value,list):
        return tuple(_freeze_consumer(item) for item in value)
    return value

def _non_informative(error,cfg):
    message=str(error)
    prefix="no ranked anchor proposal sequence passed quality: "
    if not message.startswith(prefix):
        raise error
    raw=[value.strip().replace(" ","_") for value in message[len(prefix):].split(";") if value.strip()]
    if not raw or any(value not in QUALITY_FAILURE_CODES for value in raw):
        raise RobotMaskCacheError("non-informative failure audit conflict")
    payload={"status":"non_informative","reason":"quality_exhausted",
             "failure_codes":raw,"attempts":cfg.max_anchor_sequence_attempts}
    return {**payload,"binding_sha256":_ph(payload)}

def _validate_non_informative(value,cfg):
    if not isinstance(value,dict) or set(value)!={
            "status","reason","failure_codes","attempts","binding_sha256"}:
        raise RobotMaskCacheError("non-informative entry schema conflict")
    payload={key:value[key] for key in ("status","reason","failure_codes","attempts")}
    if (value["status"]!="non_informative" or value["reason"]!="quality_exhausted"
            or value["attempts"]!=cfg.max_anchor_sequence_attempts
            or not isinstance(value["failure_codes"],list) or not value["failure_codes"]
            or any(code not in QUALITY_FAILURE_CODES for code in value["failure_codes"])
            or value["binding_sha256"]!=_ph(payload)):
        raise RobotMaskCacheError("non-informative entry audit conflict")

def _consumer_file(path,label,directory=False):
    raw=Path(path)
    if ".." in raw.parts: raise RobotMaskCacheError(f"{label} contains path traversal: {raw}")
    absolute=raw.absolute()
    if directory: _directory(absolute,label)
    else: _regular(absolute,label)
    return absolute

def _hex64(value):
    return isinstance(value,str) and re.fullmatch(r"[0-9a-f]{64}",value) is not None

def _bound_consumer_path(value,expected):
    if not isinstance(value,str): return False
    raw=Path(value)
    return raw.is_absolute() and ".." not in raw.parts and raw==expected

def _validate_mask_provenance(payload,formal_config):
    if payload.get("config")!=formal_config or payload.get("config_sha256")!=_ph(formal_config):
        raise RobotMaskCacheError("index formal config/hash conflict")
    code=payload.get("code")
    if not isinstance(code,dict) or set(code)!={"repo_files","implementation_files","environment","tree_sha256"}:
        raise RobotMaskCacheError("index code identity schema conflict")
    code_payload={key:value for key,value in code.items() if key!="tree_sha256"}
    if code["tree_sha256"]!=_ph(code_payload) or payload.get("code_sha256")!=code["tree_sha256"]:
        raise RobotMaskCacheError("index code hash conflict")
    repo_files=code["repo_files"]
    if not isinstance(repo_files,list) or not repo_files:
        raise RobotMaskCacheError("index repo file inventory conflict")
    registered_files={
        "wm3d_v3/stage1/robot_mask_cache.py",
        "scripts/build_stage1_robot_mask_cache.py",
        "wm3d_v3/stage1/action_contract.py",
        "wm3d_v3/stage1/action_contract_split.py",
        "wm3d_v3/stage1/action_contract_evidence.py",
        "wm3d_v3/stage1/action_evidence_sources.py",
        "wm3d_v3/stage1/action_cache.py",
        "wm3d_v3/stage1/droid_interval_action.py",
        "wm3d_v3/stage1/immutable_artifact.py",
        "wm3d_v3/data/manifest.py",
    }
    if {row.get("path") for row in repo_files if isinstance(row,dict)}!=registered_files:
        raise RobotMaskCacheError("index registered code inventory conflict")
    seen=set()
    for row in repo_files:
        if (not isinstance(row,dict) or set(row)!={"path","sha256"}
                or not isinstance(row["path"],str) or Path(row["path"]).is_absolute()
                or ".." in Path(row["path"]).parts or row["path"] in seen
                or not _hex64(row["sha256"])):
            raise RobotMaskCacheError("index repo file identity conflict")
        seen.add(row["path"])
    implementations=code["implementation_files"]
    if not isinstance(implementations,list) or not implementations:
        raise RobotMaskCacheError("index implementation inventory conflict")
    implementation_seen=set()
    for row in implementations:
        path=Path(str(row.get("path",""))) if isinstance(row,dict) else Path("")
        if (not isinstance(row,dict) or set(row)!={"path","sha256","size"}
                or not path.is_absolute() or ".." in path.parts
                or str(path) in implementation_seen or not _hex64(row["sha256"])
                or not isinstance(row["size"],int) or isinstance(row["size"],bool) or row["size"]<0):
            raise RobotMaskCacheError("index implementation identity conflict")
        implementation_seen.add(str(path))
    environment=code["environment"]
    if (not isinstance(environment,dict) or set(environment)!={
            "python_implementation","python_version","numpy_version",
            "transformers_version","torch_version","torch_cuda_version"}):
        raise RobotMaskCacheError("index code environment conflict")

    model=payload.get("model_identity")
    if not isinstance(model,dict) or set(model)!={
            "adapters","tensor_manifest_sha256","parameter_and_buffer_content_sha256",
            "parameter_and_buffer_count"}:
        raise RobotMaskCacheError("index model identity schema conflict")
    adapters=model["adapters"]
    registered={
        "detector":(FORMAL_GROUNDING_MODEL,FORMAL_GROUNDING_REVISION),
        "propagator":(FORMAL_SAM2_MODEL,FORMAL_SAM2_REVISION),
    }
    if not isinstance(adapters,dict) or set(adapters)!=set(registered):
        raise RobotMaskCacheError("index registered model adapters conflict")
    for owner,(model_id,revision) in registered.items():
        identity=adapters[owner]
        if not isinstance(identity,dict) or set(identity)!=IDENTITY_KEYS:
            raise RobotMaskCacheError(f"index {owner} identity schema conflict")
        snapshot=Path(str(identity.get("snapshot_path","")))
        if (identity.get("model_id")!=model_id
                or identity.get("requested_revision")!=revision
                or identity.get("resolved_commit")!=revision
                or not snapshot.is_absolute() or ".." in snapshot.parts or snapshot.name!=revision):
            raise RobotMaskCacheError(f"index registered {owner} identity conflict")
        _directory(snapshot,f"{owner} snapshot")
        if identity.get("snapshot_tree_sha256")!=_snapshot_tree_hash(snapshot):
            raise RobotMaskCacheError(f"index {owner} snapshot tree hash conflict")
        for key in IDENTITY_KEYS-{"model_id","requested_revision","resolved_commit","snapshot_path"}:
            if not _hex64(identity.get(key)):
                raise RobotMaskCacheError(f"index {owner} hash conflict: {key}")
    if (not _hex64(model["tensor_manifest_sha256"])
            or not _hex64(model["parameter_and_buffer_content_sha256"])
            or not isinstance(model["parameter_and_buffer_count"],int)
            or isinstance(model["parameter_and_buffer_count"],bool)
            or model["parameter_and_buffer_count"]<0):
        raise RobotMaskCacheError("index model content identity conflict")
    if payload.get("model_identity_sha256")!=_ph(model):
        raise RobotMaskCacheError("index model identity hash conflict")

def load_validated_robot_mask_index(
    index_path,*,split_path,manifest_path,cache_root,droid_cache_index,
):
    index=_consumer_file(index_path,"robot-mask index")
    split=_consumer_file(split_path,"split artifact")
    manifest=_consumer_file(manifest_path,"source manifest")
    root=_consumer_file(cache_root,"common cache root",directory=True)
    droid=_consumer_file(droid_cache_index,"DROID finalized cache index")
    output_root=_consumer_file(index.parent,"robot-mask output root",directory=True)
    fingerprints={
        "index":_input_fingerprint(index,"robot-mask index"),
        "split":_input_fingerprint(split,"split artifact"),
        "manifest":_input_fingerprint(manifest,"source manifest"),
        "droid":_input_fingerprint(droid,"DROID finalized cache index"),
    }
    candidates=load_mask_candidates(split,manifest,root,droid_cache_index=droid)
    payload=_json(index,"robot-mask index")
    required_top={"schema_version","immutable","split_artifact","source_manifest",
                  "droid_cache_index","config","config_sha256","code","code_sha256",
                  "model_identity","model_identity_sha256","expected_identity_sha256",
                  "entries","output_hashes_sha256"}
    if set(payload)!=required_top or payload.get("schema_version")!=SCHEMA_VERSION:
        raise RobotMaskCacheError("index schema conflict")
    if payload.get("immutable") is not True:
        raise RobotMaskCacheError("index must be immutable")
    split_binding=payload.get("split_artifact"); manifest_binding=payload.get("source_manifest")
    droid_binding=payload.get("droid_cache_index")
    candidate_droid={(item.droid_cache_index_path,item.droid_cache_index_sha256,item.droid_cache_root)
                     for item in candidates if item.droid_cache_index_path is not None}
    if len(candidate_droid)!=1: raise RobotMaskCacheError("DROID binding is not unique")
    droid_path,droid_sha,droid_root=next(iter(candidate_droid))
    if (not isinstance(split_binding,dict) or set(split_binding)!={"path","sha256"}
            or not _bound_consumer_path(split_binding.get("path"),split)
            or split_binding.get("sha256")!=fingerprints["split"]["sha256"]):
        raise RobotMaskCacheError("index split path/hash conflict")
    if (not isinstance(manifest_binding,dict) or set(manifest_binding)!={"path","sha256"}
            or not _bound_consumer_path(manifest_binding.get("path"),manifest)
            or manifest_binding.get("sha256")!=fingerprints["manifest"]["sha256"]):
        raise RobotMaskCacheError("index manifest path/hash conflict")
    if (not isinstance(droid_binding,dict) or set(droid_binding)!={"path","sha256","cache_root"}
            or not _bound_consumer_path(droid_binding.get("path"),droid)
            or droid_binding.get("sha256")!=fingerprints["droid"]["sha256"]
            or droid_binding!={"path":droid_path,"sha256":droid_sha,"cache_root":droid_root}):
        raise RobotMaskCacheError("index DROID path/hash conflict")
    config=RobotMaskConfig(); config.validate()
    formal_config=asdict(config); formal_config["anchor_indices"]=list(config.anchor_indices)
    _validate_mask_provenance(payload,formal_config)
    expected_ids=[item.identity for item in candidates]
    if payload.get("expected_identity_sha256")!=_ph([list(identity) for identity in expected_ids]):
        raise RobotMaskCacheError("index expected identity hash conflict")
    entries=payload.get("entries")
    if not isinstance(entries,list) or len(entries)!=EXPECTED_WINDOWS:
        raise RobotMaskCacheError("index entry count conflict")
    actual_ids=[]
    for entry in entries:
        raw=entry.get("identity") if isinstance(entry,dict) else None
        actual_ids.append(tuple(raw) if isinstance(raw,list) and len(raw)==4 else ())
    if len(set(actual_ids))!=EXPECTED_WINDOWS or set(actual_ids)!=set(expected_ids):
        raise RobotMaskCacheError("index identity set conflict")
    by_id={identity:entry for identity,entry in zip(actual_ids,entries,strict=True)}
    hashes={}; names=set(); validated={}; descriptors=[]
    split_sha=fingerprints["split"]["sha256"]; manifest_sha=fingerprints["manifest"]["sha256"]
    for item in candidates:
        entry=by_id[item.identity]
        _,source=_source(item,root,hashes)
        expected=_base(item,source,split_sha,manifest_sha,payload["model_identity_sha256"],
                       payload["config_sha256"],payload["code_sha256"])
        if any(entry.get(key)!=value for key,value in expected.items()):
            raise RobotMaskCacheError(f"index entry binding conflict: {item.identity}")
        name=_name(item)
        if "output" in entry:
            if set(entry)!={*expected,"output"}:
                raise RobotMaskCacheError(f"index output schema conflict: {item.identity}")
            output=entry["output"]
            if (not isinstance(output,dict) or set(output)!={"path","sha256","shape","dtype","mask_sha256"}
                    or output.get("path")!=name or name in names):
                raise RobotMaskCacheError(f"index output identity conflict: {item.identity}")
            relative=Path(name)
            if relative.is_absolute() or len(relative.parts)!=1 or ".." in relative.parts:
                raise RobotMaskCacheError(f"unsafe indexed output path: {name}")
            names.add(name); _validate_npz(output_root/relative,entry,config); descriptors.append(output)
        elif "non_informative" in entry:
            if set(entry)!={*expected,"non_informative"}:
                raise RobotMaskCacheError(f"index non-informative schema conflict: {item.identity}")
            _validate_non_informative(entry["non_informative"],config)
            descriptors.append(entry["non_informative"])
        else:
            raise RobotMaskCacheError(f"index result missing: {item.identity}")
        validated[item.identity]=_freeze_consumer(entry)
    actual_names={path.name for path in output_root.iterdir() if path.name.endswith(".npz")}
    if actual_names!=names: raise RobotMaskCacheError("output inventory conflict")
    if payload.get("output_hashes_sha256")!=_ph(descriptors):
        raise RobotMaskCacheError("output hash inventory conflict")
    _assert_inputs_unchanged({
        "robot-mask index":fingerprints["index"],"split artifact":fingerprints["split"],
        "source manifest":fingerprints["manifest"],"DROID finalized cache index":fingerprints["droid"],
    })
    return ValidatedRobotMaskIndex(
        index_path=str(index),index_sha256=fingerprints["index"]["sha256"],
        metadata=_freeze_consumer({key:value for key,value in payload.items() if key!="entries"}),
        entries=MappingProxyType(validated),
    )

def dry_run_bridge(*,split_path,manifest_path,cache_root,detector_factory,propagator_factory,config=RobotMaskConfig()):
    config.validate(); sp,mp,root=Path(split_path).absolute(),Path(manifest_path).absolute(),Path(cache_root).absolute()
    split=_json(sp,"split"); _regular(mp,"manifest"); _directory(root,"cache root")
    if split.get("schema_version")!=SPLIT_SCHEMA or split.get("immutable") is not True or split.get("seed")!=1729 or split.get("derivation")!=SPLIT_DERIVATION:
        raise RobotMaskCacheError("split schema/seed/derivation mismatch")
    src_binding=split.get("source_manifest"); groups=split.get("groups")
    if not isinstance(src_binding,dict) or set(src_binding)!={"path","sha256"} or Path(src_binding["path"]).absolute()!=mp or src_binding["sha256"]!=sha256_file(mp):
        raise RobotMaskCacheError("split/source manifest binding mismatch")
    if not isinstance(groups,dict) or set(groups)!=EXPECTED_CONTRACTS: raise RobotMaskCacheError("split must contain exact six contracts")
    frozen_groups={key:frozen_contract_split_from_mapping(groups[key]) for key in sorted(groups)}
    all_ids=[cid for group in frozen_groups.values() for role in ("calibration","qualification","confirmation") for cid in getattr(group,f"{role}_clip_ids")]
    if len(all_ids)!=576 or len(set(all_ids))!=576: raise RobotMaskCacheError("split identity inventory mismatch")
    records={r.clip_id:r for r in _read_manifest_regular(mp,"manifest")}; key=next(key for key in sorted(groups) if key.startswith("bridge|")); frozen=frozen_groups[key]
    cid=frozen.calibration_clip_ids[0]; record=records[cid]; safe=safe_clip_id(cid)
    rgb=root/"rgb_256"/f"{safe}.npy"; action=root/"actions"/f"{safe}.npy"
    rgb_shape,rgb_dtype,_=_npy_meta(rgb,"Bridge RGB"); action_shape,_,_=_npy_meta(action,"Bridge action")
    if len(rgb_shape)!=4 or rgb_shape[-1]!=3 or rgb_dtype!=np.uint8 or not action_shape: raise RobotMaskCacheError("invalid Bridge RGB/action cache")
    binding=bind_temporal_window(record,group_id=frozen.clip_to_group_id[cid],seed=1729,usable_frames=min(record.n_frames,int(rgb_shape[0])),n_action_frames=int(action_shape[0]),target_length=16)
    item=MaskCandidate(key,"calibration",cid,binding.start,binding.group_id,binding.legal_starts_sha256,binding.selection_sha256)
    d,p=detector_factory(),propagator_factory(); model=_model_identity(d,p); frames,src=_source(item,root,{}); out=_infer(frames,d,p,config)
    return {"status":"dry-run-ok","identity":list(item.identity),"source_rgb_window_sha256":src["window_sha256"],"model_identity_sha256":_ph(model),"mask_sha256":_ah(out["mask"]),"min_direction_iou":float(out["direction_iou"].min()),"min_temporal_iou":float(out["temporal_iou"].min()),"min_anchor_box_iou":float(out["anchor_box_iou"].min()),"min_object_score_logit":float(out["object_score_logits"].min())}

def _resume_output(path,name):
    _regular(path,"resumable NPZ")
    try:
        with _open_regular(path,"resumable NPZ") as stream:
            with np.load(stream,allow_pickle=False) as data:
                shape=list(data["mask"].shape); mask_sha=str(data["mask_sha256"].item())
    except RobotMaskCacheError: raise
    except Exception as exc: raise RobotMaskCacheError(f"invalid resumable NPZ: {path}") from exc
    return {"path":name,"sha256":sha256_file(path),"shape":shape,"dtype":"uint8","mask_sha256":mask_sha}

def _expected_payload(sp,mp,config,candidates,model,code,bases):
    cfg=asdict(config); cfg["anchor_indices"]=list(config.anchor_indices)
    split_sha,manifest_sha=sha256_file(sp),sha256_file(mp)
    model_sha,code_sha,cfg_sha=_ph(model),code["tree_sha256"],_ph(cfg)
    droid_bindings={
        (item.droid_cache_index_path,item.droid_cache_index_sha256,item.droid_cache_root)
        for item in candidates if item.droid_cache_index_path is not None
    }
    if len(droid_bindings)!=1:
        raise RobotMaskCacheError("DROID finalized cache binding must be unique")
    droid_path,droid_sha,droid_root=next(iter(droid_bindings))
    return {"schema_version":SCHEMA_VERSION,"immutable":True,
            "split_artifact":{"path":str(sp),"sha256":split_sha},
            "source_manifest":{"path":str(mp),"sha256":manifest_sha},
            "droid_cache_index":{"path":droid_path,"sha256":droid_sha,
                                 "cache_root":droid_root},
            "config":cfg,"config_sha256":cfg_sha,"code":code,"code_sha256":code_sha,
            "model_identity":model,"model_identity_sha256":model_sha,
            "expected_identity_sha256":_ph([list(item.identity) for item in candidates]),
            "entries":bases}

def build_robot_mask_cache(*,split_path,manifest_path,cache_root,output_root,
                           droid_cache_index,detector_factory,propagator_factory,
                           config=RobotMaskConfig(),after_publish=None):
    config.validate()
    sp,mp,cr=map(lambda value:Path(value).absolute(),(split_path,manifest_path,cache_root))
    root=Path(output_root).absolute(); _no_symlinks(root,"output root",True)
    # The process lock deliberately precedes all input scans, factories, model loads, and CUDA work.
    with _lock(root):
        if droid_cache_index is None: raise RobotMaskCacheError("droid_cache_index is required")
        input_fingerprints={
            "split artifact":_input_fingerprint(sp,"split artifact"),
            "source manifest":_input_fingerprint(mp,"source manifest"),
            "DROID finalized cache index":_input_fingerprint(droid_cache_index,"DROID finalized cache index"),
        }
        candidates=load_mask_candidates(sp,mp,cr,droid_cache_index=droid_cache_index)
        _assert_inputs_unchanged(input_fingerprints)
        detector=detector_factory(); propagator=propagator_factory()
        model=_model_identity(detector,propagator); code=_code_identity(detector,propagator)
        _assert_inputs_unchanged(input_fingerprints)
        cfg=asdict(config); cfg["anchor_indices"]=list(config.anchor_indices)
        split_sha,manifest_sha=sha256_file(sp),sha256_file(mp)
        model_sha,code_sha,cfg_sha=_ph(model),code["tree_sha256"],_ph(cfg)
        index=root/"index.json"; hashes={}
        if index.exists() or index.is_symlink():
            bases=[]
            for item in candidates:
                frames,source=_source(item,cr,hashes)
                bases.append(_base(item,source,split_sha,manifest_sha,model_sha,cfg_sha,code_sha))
                del frames
            expected=_expected_payload(sp,mp,config,candidates,model,code,bases)
            _assert_inputs_unchanged(input_fingerprints)
            return _validate_index(index,expected,root,config)
        allowed={_name(item) for item in candidates}
        unknown=sorted(path.name for path in root.glob("*.npz") if path.name not in allowed)
        if unknown: raise RobotMaskCacheError(f"unexpected output: {unknown[0]}")
        complete=[]; bases=[]
        for ordinal,item in enumerate(candidates):
            # Exactly one RGB clip/window is resident at a time.
            frames,source=_source(item,cr,hashes)
            base=_base(item,source,split_sha,manifest_sha,model_sha,cfg_sha,code_sha)
            bases.append(base); name,path=_name(item),root/_name(item)
            if path.exists() or path.is_symlink():
                entry={**base,"output":_resume_output(path,name)}
                _validate_npz(path,entry,config); complete.append(entry); del frames; continue
            try:
                inferred=_infer(frames,detector,propagator,config)
            except RobotMaskCacheError as exc:
                del frames
                complete.append({**base,"non_informative":_non_informative(exc,config)})
                continue
            del frames
            mask_sha=_ah(inferred["mask"])
            arrays={**inferred,"start":np.asarray(item.start,np.int64),
                    "frame_indices":np.arange(item.start,item.start+17,dtype=np.int64),
                    "source_rgb_sha256":np.asarray(source["sha256"]),
                    "source_rgb_window_sha256":np.asarray(source["window_sha256"]),
                    "source_action_sha256":np.asarray(item.action_sha256),
                    "droid_cache_index_sha256":np.asarray(item.droid_cache_index_sha256 or ""),
                    "split_sha256":np.asarray(split_sha),"manifest_sha256":np.asarray(manifest_sha),
                    "model_identity_sha256":np.asarray(model_sha),"config_sha256":np.asarray(cfg_sha),
                    "code_sha256":np.asarray(code_sha),"mask_sha256":np.asarray(mask_sha)}
            try: result=publish_immutable_bytes(path,_npz(arrays))
            except ImmutableArtifactConflict as exc: raise RobotMaskCacheError(str(exc)) from exc
            entry={**base,"output":{"path":name,"sha256":result.sha256,
                   "shape":list(inferred["mask"].shape),"dtype":"uint8","mask_sha256":mask_sha}}
            _validate_npz(path,entry,config); complete.append(entry)
            if after_publish: after_publish(ordinal+1,path)
        expected=_expected_payload(sp,mp,config,candidates,model,code,bases)
        descriptors=[entry.get("output",entry.get("non_informative")) for entry in complete]
        payload={**{key:value for key,value in expected.items() if key!="entries"},
                 "entries":complete,"output_hashes_sha256":_ph(descriptors)}
        _assert_inputs_unchanged(input_fingerprints)
        try: result=publish_immutable_bytes(index,_canonical(payload,True))
        except ImmutableArtifactConflict as exc: raise RobotMaskCacheError(str(exc)) from exc
        return {"status":"built","window_count":576,"index":str(index),"index_sha256":result.sha256}
