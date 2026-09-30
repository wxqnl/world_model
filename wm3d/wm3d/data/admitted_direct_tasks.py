"""Stat-bound direct-raw task ABI for an explicitly admitted fresh dataset.

This ABI never masquerades as the legacy SHA task ABI. Original legacy
records remain provenance, while reads are tied to the admitted file stat.
Approved RGB replacements require the full original repair item unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping
from .file_binding import require_stat_binding, verify_stat_asset
from .manifest_contract import safe_relative_path
from .episode_io import VerifiedAssetStore, EpisodeIOError

ADMITTED_DIRECT_TASK_SCHEMA = "wm3d_admitted_direct_task_v1"
APPROVED_REPAIR_STAT_KIND = "approved_legacy_rgb_repair_stat_v1"


def require_asset_binding(value: Mapping[str, Any]):
    if not isinstance(value,dict):
        raise ValueError("direct task asset must have explicit stat binding")
    if value.get("kind") != APPROVED_REPAIR_STAT_KIND:
        return require_stat_binding(value)
    if set(value)!={"kind","admission_id","original","replacement","repair_asset"}:
        raise ValueError("repair stat binding fields differ")
    for name in ("original","replacement"):
        if require_stat_binding(value[name])["admission_id"]!=value["admission_id"]:
            raise ValueError("repair stat admission mismatch")
    repair=value["repair_asset"]
    expected={"source_root","relative_path","sealed_digest","replacement_relative_path",
              "replacement_digest","decoded_frames","time_base","timestamps_unchanged"}
    if not isinstance(repair,dict) or set(repair)!=expected:
        raise ValueError("repair lineage item is incomplete")
    return value


@dataclass(frozen=True)
class AdmittedDirectTask:
    task_id: str
    source: str
    episode_id: str
    payload: str
    payload_binding: Mapping[str,Any]
    payload_row_start: int
    payload_row_stop: int
    assets: tuple
    views: tuple
    task_text: str
    embodiment: str
    split: str
    observation_samples: int
    observation_clock: Mapping[str,Any]
    robot_groups: Mapping[str,Any]
    admission_id: str
    origin: str
    inherited_record: Mapping[str,Any] | None
    timestamp_binding_mode: str = "admitted_payload_stat"


def admitted_task_from_mapping(value: Mapping[str,Any]) -> AdmittedDirectTask:
    fields={"schema","task_id","source","episode_id","payload","payload_binding",
        "payload_row_start","payload_row_stop","assets","views","task_text","embodiment",
        "split","observation_samples","observation_clock","robot_groups",
        "admission_id","origin","inherited_record"}
    if not isinstance(value,dict) or set(value)!=fields or value["schema"]!=ADMITTED_DIRECT_TASK_SCHEMA:
        raise ValueError("admitted direct task fields/schema mismatch")
    if not re.fullmatch(r"task[0-9]{9}",value["task_id"]):
        raise ValueError("direct task ordinal must be explicit")
    admission=value["admission_id"]
    if not isinstance(admission,str) or not admission.strip():
        raise ValueError("direct task needs admission identity")
    for name in ("source","episode_id","embodiment","task_text"):
        if not isinstance(value[name],str) or not value[name].strip():
            raise ValueError(f"direct task {name} must be nonempty")
    if value["origin"] not in {"admitted","inherited_legacy"}:
        raise ValueError("direct task origin is unknown")
    inherited=value["inherited_record"]
    if value["origin"]=="inherited_legacy":
        if not isinstance(inherited,dict) or inherited.get("schema")!="wm3d_v8_episode_cache_task_v4":
            raise ValueError("legacy task must retain its original sealed record")
        for name in ("source","episode_id","payload","payload_row_start","payload_row_stop",
                     "views","task_text","embodiment","split","observation_samples",
                     "observation_clock","robot_groups"):
            if value[name]!=inherited[name]:
                raise ValueError(f"legacy task content changed: {name}")
    elif inherited is not None:
        raise ValueError("newly admitted task cannot claim legacy lineage")
    primary=require_stat_binding(value["payload_binding"])
    if primary["admission_id"]!=admission:raise ValueError("payload admission differs")
    assets=[]
    for raw in value["assets"]:
        if not isinstance(raw,dict) or set(raw)!={"role","path","binding"}:
            raise ValueError("direct asset fields differ")
        b=require_asset_binding(raw["binding"])
        if b["admission_id"]!=admission:raise ValueError("asset admission differs")
        assets.append((raw["role"],safe_relative_path(raw["path"]),b))
    if not assets or len({x[0] for x in assets})!=len(assets):
        raise ValueError("direct asset roles duplicated or empty")
    if [(p,b) for r,p,b in assets if r=="primary_payload"]!=[(value["payload"],primary)]:
        raise ValueError("direct primary payload mismatch")
    if inherited is not None:
        old={x["role"]:x for x in inherited["assets"]}
        if set(old)!={x[0] for x in assets} or any(old[r]["path"]!=p for r,p,b in assets):
            raise ValueError("legacy asset membership changed")
        for r,p,b in assets:
            if b["kind"]==APPROVED_REPAIR_STAT_KIND and b["repair_asset"]["sealed_digest"]!=old[r]["sha256"]:
                raise ValueError("repair is not bound to original sealed digest")
    views=[]
    for raw in value["views"]:
        if not isinstance(raw,dict) or set(raw)!={"name","asset_role","segment_kind","start_s","stop_s"}:
            raise ValueError("direct view fields differ")
        if raw["asset_role"] not in {x[0] for x in assets}:raise ValueError("view asset missing")
        if raw["segment_kind"] not in {"entire_file","recorded_pts_range"}:
            raise ValueError("view segment kind invalid")
        if raw["segment_kind"]=="entire_file" and (raw["start_s"] is not None or raw["stop_s"] is not None):
            raise ValueError("whole video has partial bounds")
        if raw["segment_kind"]=="recorded_pts_range" and not (
            isinstance(raw["start_s"],(int,float)) and isinstance(raw["stop_s"],(int,float))
            and 0<=raw["start_s"]<raw["stop_s"]):
            raise ValueError("video range invalid")
        views.append(tuple(raw[k] for k in ("name","asset_role","segment_kind","start_s","stop_s")))
    if not views or len({x[0] for x in views})!=len(views):raise ValueError("duplicate/empty views")
    if value["split"] not in {"train","val","test"}:raise ValueError("invalid split")
    start,stop,n=(value[k] for k in ("payload_row_start","payload_row_stop","observation_samples"))
    if any(type(x)is not int for x in (start,stop,n)) or not 0<=start<stop or n!=stop-start:
        raise ValueError("direct payload bounds/cardinality differs")
    clock=value["observation_clock"]
    if not isinstance(clock,dict) or clock.get("sample_count")!=n:
        raise ValueError("direct clock cardinality differs")
    if value["origin"]=="admitted" and clock.get("timestamp_binding")!="recorded_in_admitted_payload":
        raise ValueError("new task needs actual admitted clock evidence")
    return AdmittedDirectTask(**{k:value[k] for k in fields-{"schema","assets","views"}},
        assets=tuple(assets),views=tuple(views))


class AdmittedAssetStore(VerifiedAssetStore):
    """Explicit stat reads, including only originally approved RGB repairs."""
    def __init__(self,*,admission_id:str,initial_raw_asset_repairs):
        super().__init__()
        if self._repair!=initial_raw_asset_repairs:
            raise EpisodeIOError("admitted asset store repair contract differs")
        self.admission_id=admission_id

    def verify(self,root:Path,relative:str,binding:Mapping[str,Any])->Path:
        with self._lock:
            b=require_asset_binding(binding)
            if b["admission_id"]!=self.admission_id:
                raise EpisodeIOError("asset belongs to another admission")
            item=self._repair_by_asset.get((Path(root).resolve(strict=True),relative))
            if item is None:
                if b.get("kind")==APPROVED_REPAIR_STAT_KIND:
                    raise EpisodeIOError("unapproved RGB repair stat binding")
                return verify_stat_asset(root,relative,b)
            if b.get("kind")!=APPROVED_REPAIR_STAT_KIND or b["repair_asset"]!=item:
                raise EpisodeIOError("approved RGB repair lineage changed or was omitted")
            verify_stat_asset(root,relative,b["original"])
            return verify_stat_asset(root,item["replacement_relative_path"],b["replacement"])

    def verify_admitted(self,root,relative,binding):
        return self.verify(root,relative,binding)

    def content_digest(self,path,original_digest):
        raise EpisodeIOError("stat admission must not request a fabricated content digest")
