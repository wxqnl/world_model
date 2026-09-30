"""Production closure for admitted direct RGB reads without content scans."""
from pathlib import Path
import json
from .file_binding import require_stat_binding, verify_stat_asset
from .manifest_contract import load_data_profile
from .grouped_normalization import GroupedRobotNormalizer

SCHEMA = "wm3d_admitted_direct_raw_data_closure_v1"
SEAL_SCHEMA = "wm3d_admitted_direct_metadata_seal_v1"
FILES = ("data_profile", "cache_index", "task_manifest", "grouped_normalization", "encoder_contract")
POSITIVE = ("direct_input_rgb_size", "direct_decode_workers", "direct_robot_cache_episodes",
    "direct_prefetch_windows", "direct_video_index_cache_assets", "direct_encode_chunk_rows",
    "direct_minimum_chunk_rows", "appearance_token_grid")


def verify_file(path, binding, admission):
    path=Path(path)
    if not path.is_absolute() or require_stat_binding(binding)["admission_id"]!=admission:
        raise ValueError("closure file path/admission differs")
    return verify_stat_asset(path.parent,path.name,binding)


def seal_contract(closure):
    return {k:v for k,v in closure.items() if k not in
            {"name","metadata_seal_path","metadata_seal_binding"}}


def validate_admitted_closure(closure, model):
    fields={"schema","name","admission_id","metadata_root","metadata_seal_path","metadata_seal_binding",
        "foundation_text_only","initial_raw_asset_repairs","source_manifest_binding_by_name",
        "adapter_contract_binding_by_name","source_action_supervision","appearance_feature_layer","direct_ignored_action_dimensions",
        *POSITIVE}
    fields|={base+suffix for base in FILES for suffix in ("_path","_binding")}
    if not isinstance(closure,dict) or set(closure)!=fields or closure["schema"]!=SCHEMA:
        raise ValueError("admitted direct closure fields/schema mismatch")
    admission=closure["admission_id"]
    if not isinstance(admission,str) or not admission or closure["foundation_text_only"] is not True:
        raise ValueError("admitted foundation identity/mode mismatch")
    if model.get("architecture")!="foundation_world_model":
        raise ValueError("admitted direct path requires the foundation model")
    for name in POSITIVE:
        if type(closure[name]) is not int or closure[name]<=0:
            raise ValueError("invalid admitted direct resource field: "+name)
    if (closure["direct_input_rgb_size"]%14
            or closure["direct_minimum_chunk_rows"]>closure["direct_encode_chunk_rows"]
            or closure["appearance_feature_layer"] not in (4,11,17,23)):
        raise ValueError("invalid direct RGB/feature configuration")
    root=Path(closure["metadata_root"])
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("metadata root unavailable")
    for base in FILES:
        verify_file(closure[base+"_path"],closure[base+"_binding"],admission)
    seal_path=verify_file(closure["metadata_seal_path"],closure["metadata_seal_binding"],admission)
    seal=json.loads(seal_path.read_text())
    verify_file(seal_path,closure["metadata_seal_binding"],admission)
    if (set(seal)!={"schema","contract","model_profile","window_count_by_split_source",
                   "task_count_by_source","selection","admission_id"}
            or seal["schema"]!=SEAL_SCHEMA or seal["admission_id"]!=admission
            or seal["contract"]!=seal_contract(closure) or seal["model_profile"]!=model):
        raise ValueError("metadata seal and runtime contract differ")
    profile=load_data_profile(Path(closure["data_profile_path"]),verify_source_manifests=False,
        expected_binding=closure["data_profile_binding"])
    if (profile.admission_id!=admission
            or closure["source_manifest_binding_by_name"]!={s.name:s.manifest_binding for s in profile.sources}
            or closure["adapter_contract_binding_by_name"]!={s.name:s.adapter_contract_binding for s in profile.sources}):
        raise ValueError("admitted closure/profile source bindings differ")
    labels=closure["source_action_supervision"]
    if set(labels)!=set(profile.source_order) or any(x not in {"fine_command","coarse_effect"} for x in labels.values()):
        raise ValueError("source target semantics are incomplete")
    counts=seal["window_count_by_split_source"];tasks=seal["task_count_by_source"]
    if (set(counts)!={"train","val","test"} or set(tasks)!=set(profile.source_order)
            or any(type(x)is not int or x<=0 for x in tasks.values())):
        raise ValueError("admitted task counts invalid")
    for split,source_counts in counts.items():
        if (set(source_counts)!=set(profile.source_order)
                or any(type(x)is not int or x<0 for x in source_counts.values())):
            raise ValueError("admitted window counts invalid")
    if any(x<=0 for x in counts["train"].values()):
        raise ValueError("each admitted source must have train windows")
    selection=seal["selection"]
    if (not isinstance(selection,dict) or selection.get("mode")!="frozen_base_selection_plus_admitted_episodes"
            or selection.get("new_episodes_split_disjoint") is not True):
        raise ValueError("admitted episode selection is not frozen")
    from .direct_raw import _ignored_action_slots
    _ignored_action_slots(closure["direct_ignored_action_dimensions"],data_profile=profile)
    # Excluding new controls silently is forbidden; old exclusions are inherited explicitly.
    old={s.name for s in profile.sources if s.origin=="inherited_legacy"}
    if any(x["source"] not in old for x in closure["direct_ignored_action_dimensions"]):
        raise ValueError("new admitted controls cannot be ignored")
    GroupedRobotNormalizer.load_admitted(Path(closure["grouped_normalization_path"]),
        expected_binding=closure["grouped_normalization_binding"],
        expected_window_index_binding=closure["cache_index_binding"],
        model_profile=model,data_profile=profile)
    return profile,seal
