"""Bounded two-store routing around the unchanged native checkpoint manager.

Discovery is a metadata/stat audit, not a payload integrity or full-state readback.
The original manager remains responsible for save commits and full-state restores.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping


SNAPSHOTS = ".wm3d_checkpoint_load_snapshots"
NATIVE_SCHEMA = "wm3d_v8_distributed_checkpoint_v2"
COMMIT_SCHEMA = "wm3d_v8_distributed_checkpoint_commit_v2"


def real_directory(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"storage path must be an absolute fixed path: {path}")
    current = Path("/")
    for part in path.parts[1:]:
        current /= part
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"non-real storage directory: {current}")
    return path


def read_json(path: Path) -> Any:
    real_directory(path.parent)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > 16 * (1 << 20):
        raise RuntimeError(f"invalid or oversized JSON control: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
        value = json.load(stream)
        after = os.fstat(stream.fileno())
    if (info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
        after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
    ):
        raise RuntimeError(f"JSON control changed while reading: {path}")
    return value


def native_record(path: Path, member: Mapping[str, Any]) -> dict[str, Any]:
    """Audit real native52 files against that run's original literal contracts."""
    real_directory(path)
    root = real_directory(Path(member["root"]))
    if path.parent != root or re.fullmatch(r"step_[0-9]{8}", path.name) is None:
        raise RuntimeError("checkpoint outside the explicit native member root")
    # The same physical store may contain both sides of the one explicit
    # transition. Old commits retain their literal original runtime/contract.
    version = member.get("after_step_30000")
    if version is not None and int(path.name[5:]) > 30000:
        member = {**member, **version}
    corrected_version = member.get("after_step_40000")
    if corrected_version is not None and int(path.name[5:]) > 40000:
        member = {**member, **corrected_version}
    cfg = read_json(Path(member["runtime_path"]))
    contract = read_json(Path(member["contract_path"]))
    records = {name: read_json(path / name) for name in
               ("metadata.json", "MANIFEST.json", "COMMITTED.json")}
    meta, manifest, commit = (records[name] for name in
                             ("metadata.json", "MANIFEST.json", "COMMITTED.json"))
    step = int(path.name[5:])
    if (meta["schema"] != NATIVE_SCHEMA or manifest["schema"] != NATIVE_SCHEMA
            or commit["schema"] != COMMIT_SCHEMA
            or {meta["step"], manifest["step"], commit["step"]} != {step}):
        raise RuntimeError("native checkpoint format or step identity mismatch")
    declaration = cfg["runtime_profile"]["train"].get("formal_sampling_transition")
    expected_progress = {"next_optimizer_step": step}
    if declaration is not None:
        from wm3d.training.formal_sampling_transition import validate_declaration
        validate_declaration(declaration)
        expected_progress["formal_sampling_transition"] = declaration
        initial = meta.get("initial_formal_sampling_restore", {})
        restored = meta.get("formal_sampling_restore", {})
        if (step <= 30000 or contract.get("formal_sampling_transition") != declaration
                or meta.get("formal_sampling_transition") != declaration
                or initial.get("passed") is not True or initial.get("restored_step") != 30000
                or initial.get("formal_sampling_transition") != declaration
                or initial.get("source_checkpoint", {}).get("step") != 30000
                or restored.get("passed") is not True
                or restored.get("formal_sampling_transition") != declaration):
            raise RuntimeError("formal sampling transition/restoration provenance differs")
    elif any(key in meta for key in ("formal_sampling_transition", "formal_sampling_restore",
                                     "initial_formal_sampling_restore")):
        raise RuntimeError("old formal checkpoint cannot silently contain a new sampling contract")
    from wm3d.training.formal_omega_transition import validate_checkpoint_provenance
    validate_checkpoint_provenance(cfg, contract, meta, step, expected_progress)
    comparisons = {
        "run_lineage": (cfg["run"]["lineage"], contract["lineage"], commit["run_lineage"]),
        "run_name": (cfg["run"]["name"], contract["name"]),
        "runtime_config_sha256": (member["runtime_binding"],),
        "model_contract_sha256": (cfg["bindings"]["model_contract_sha256"], contract["model_contract_sha256"]),
        "data_closure_sha256": (cfg["bindings"]["data_closure_sha256"], contract["data_closure_sha256"]),
        "topology_contract_sha256": (contract["topology_contract_sha256"],),
        "initial_seed": (cfg["runtime_profile"]["train"]["seed"],),
        "global_batch_size": (cfg["runtime_profile"]["train"]["global_batch_size"], 96),
        "world_size": (24,),
        "shard_degree": (8,),
        "distributed_strategy": ("fsdp2",),
        "sampler_progress": (expected_progress,),
        "raw_asset_repairs": (cfg["data_closure"]["initial_raw_asset_repairs"],),
    }
    for key, expected in comparisons.items():
        if any(meta.get(key) != item for item in expected):
            raise RuntimeError(f"native run contract mismatch: {key}")
    if meta.get("gradient_ownership", {}).get("passed") is not True:
        raise RuntimeError("native checkpoint lacks the original gradient-owner proof")
    if "native_bounded_continuation" in contract and (
        meta.get("native_bounded_continuation") != contract["native_bounded_continuation"]
        or meta.get("complete_state_restore", {}).get("passed") is not True
    ):
        raise RuntimeError("bounded-continuation provenance mismatch")
    files = {}
    allocated = 0
    for directory, dirs, names in os.walk(path, followlinks=False):
        base = Path(directory)
        for name in dirs:
            real_directory(base / name)
        for name in names:
            item = base / name
            info = item.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
                raise RuntimeError(f"non-regular/empty native payload: {item}")
            with item.open("rb") as stream:
                if not stream.read(1):
                    raise RuntimeError(f"unreadable native payload: {item}")
            files[str(item.relative_to(path))] = [
                info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
            allocated += info.st_blocks * 512
    expected = {"metadata.json", "MANIFEST.json", "COMMITTED.json", "distcp/.metadata"}
    expected |= {f"distcp/__{rank}_0.distcp" for rank in range(24)}
    expected |= {f"rank_state/rank_{rank:05d}.pt" for rank in range(24)}
    if set(files) != expected or set(manifest["files"]) != expected - {"MANIFEST.json", "COMMITTED.json"}:
        raise RuntimeError("native checkpoint must retain the exact original 52-file family")
    for relative, entry in manifest["files"].items():
        if files[relative][1] != entry["size"]:
            raise RuntimeError(f"native payload size mismatch: {relative}")
    return {"path": str(path), "step": step, "physical_node": member["physical_node"],
            "run_lineage": meta["run_lineage"], "files": files, "records": records,
            "commit_mtime_ns": files["COMMITTED.json"][2],
            "allocated_bytes": allocated, "metadata_stat_audit_only": True,
            "payload_hashes_computed": False, "full_state_readback": False}


def validate_storage_plan(plan: Mapping[str, Any]) -> None:
    if not isinstance(plan, Mapping) or plan.get("schema") != "wm3d_native_two_store_plan_v1":
        raise RuntimeError("unsupported native storage plan")
    stores = plan.get("stores")
    if (not isinstance(stores, list) or len(stores) != 2
            or [item.get("physical_node") for item in stores] != [43, 41]
            or [item.get("slots") for item in stores] != [3, 1]):
        raise RuntimeError("formal native storage requires exactly node43:3/node41:1 slots")
    for key, minimum in (("checkpoint_envelope_bytes", 82000000000),
                         ("floor_bytes", 107374182400), ("reserve_bytes", 10737418240)):
        if type(plan.get(key)) is not int or plan[key] < minimum:
            raise RuntimeError(f"invalid native storage budget: {key}")
    paths = [plan["formal_runtime_path"], plan["operation_lock"], plan["namespace_receipt_path"]]
    for item in stores:
        paths.extend([item["root"], item["snapshot_parent"]])
        if Path(item["snapshot_parent"]).name != SNAPSHOTS:
            raise RuntimeError("snapshot namespace must preserve the original private directory name")
    if len({item["root"] for item in stores}) != 2:
        raise RuntimeError("native roots must be independent")
    if len(plan.get("legacy_members", [])) != 3:
        raise RuntimeError("formal handoff requires the explicit I/L/T native family")
    for item in plan["legacy_members"]:
        if item.get("physical_node") not in (43, 41) or not isinstance(item.get("runtime_binding"), str):
            raise RuntimeError("invalid native legacy member contract")
        paths.extend([item["root"], item["runtime_path"], item["contract_path"]])
    for item in paths:
        path = Path(item)
        if not path.is_absolute() or ".." in path.parts:
            raise RuntimeError("native storage declarations must use fixed absolute paths")


def committed_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema": "wm3d_native_committed_index_v1",
            "entries": [{"path": item["path"], "step": item["step"],
                "physical_node": item["physical_node"], "run_lineage": item["run_lineage"],
                "commit_mtime_ns": item["commit_mtime_ns"],
                "commit_file_identity": item["files"]["COMMITTED.json"],
                "original_COMMITTED": item["records"]["COMMITTED.json"]} for item in rows]}


class CheckpointStores:
    """Exactly two physical native stores, with shared save/restore exclusion."""

    def __init__(self, plan: Mapping[str, Any], *, config: Mapping[str, Any], config_sha: str):
        validate_storage_plan(plan)
        self.plan = dict(plan)
        self.formal_lineage = config["run"]["lineage"]
        self.stores = list(plan["stores"])
        if [item["physical_node"] for item in self.stores] != [43, 41]:
            raise RuntimeError("formal native storage is limited to 43 then 41")
        if [item["slots"] for item in self.stores] != [3, 1]:
            raise RuntimeError("formal native storage requires exactly 3+1 slots")
        self.members = [dict(item) for item in plan["legacy_members"]]
        from wm3d.training.formal_sampling_transition import resolve_transition, CONTRACT_PATH
        transition = resolve_transition(config)
        from wm3d.training import formal_omega_transition as quality
        correction = quality.resolve_transition(config)
        for store in self.stores:
            root = Path(store["root"])
            member = {
                "root": str(root), "runtime_path": plan["formal_runtime_path"],
                "contract_path": str(Path(config["run"]["output_root"]) / "run_contract.json"),
                "runtime_binding": config_sha, "physical_node": store["physical_node"],
            }
            if transition is not None:
                parent, parent_binding, declaration = transition
                member.update(runtime_path=declaration["parent_runtime_path"],
                              runtime_binding=parent_binding)
                member["after_step_30000"] = {
                    "runtime_path": plan["formal_runtime_path"],
                    "runtime_binding": config_sha, "contract_path": str(CONTRACT_PATH)}
                if correction is not None:
                    parent_config, parent_binding, _ = correction
                    member["after_step_30000"] = {
                        "runtime_path": str(quality.D / "runtime.json"),
                        "runtime_binding": parent_binding, "contract_path": str(CONTRACT_PATH)}
                    member["after_step_40000"] = {
                        "runtime_path": str(quality.RUNTIME_PATH),
                        "runtime_binding": config_sha, "contract_path": str(quality.CONTRACT_PATH)}
            self.members.append(member)
        roots = [item["root"] for item in self.members]
        if len(roots) != len(set(roots)):
            raise RuntimeError("duplicate native member roots")
        self.lock_path = Path(plan["operation_lock"])
        self.index_path = Path(config["run"]["output_root"]) / "checkpoint_index.json"
        self._active = False
        receipt = read_json(Path(plan["namespace_receipt_path"]))
        expected = [{"physical_node": item["physical_node"], "root": item["root"],
                     "snapshot_parent": item["snapshot_parent"]} for item in self.stores]
        if (receipt.get("schema") != "wm3d_native_two_store_namespace_receipt_v1"
                or receipt.get("passed") is not True
                or receipt.get("three_host_private_namespace_equal") is not True
                or receipt.get("stores") != expected):
            raise RuntimeError("actual reviewed three-host native namespaces are not ready")

    def records(self) -> list[dict[str, Any]]:
        records = []
        for member in self.members:
            root = real_directory(Path(member["root"]))
            for path in root.iterdir():
                if path.name.startswith(".") and ".incomplete." in path.name:
                    raise RuntimeError(f"incomplete checkpoint occupies fourth slot: {path}")
                if re.fullmatch(r"step_[0-9]{8}", path.name):
                    records.append(native_record(path, member))
        times = [item["commit_mtime_ns"] for item in records]
        if len(times) != len(set(times)):
            raise RuntimeError("ambiguous cross-run COMMITTED order")
        return sorted(records, key=lambda item: item["commit_mtime_ns"])

    def _empty_store(self) -> Mapping[str, Any]:
        records = self.records()
        if len(records) != 3:
            raise RuntimeError("settle exactly3 approved native commits before any save/snapshot")
        for store in self.stores:
            root = real_directory(Path(store["root"]))
            parent = Path(store["snapshot_parent"])
            if parent.name != SNAPSHOTS:
                raise RuntimeError("invalid private snapshot namespace")
            real_directory(parent)
            info = parent.lstat()
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise RuntimeError("snapshot parent must remain root-owned mode700")
            if any(parent.iterdir()):
                raise RuntimeError("snapshot residue must be collectively removed before save/restore")
        for store in self.stores:
            count = sum(item["physical_node"] == store["physical_node"] for item in records)
            available = os.statvfs(store["root"])
            free = available.f_bavail * available.f_frsize
            required = (int(self.plan["checkpoint_envelope_bytes"])
                        + int(self.plan["floor_bytes"]) + int(self.plan["reserve_bytes"]))
            if count < store["slots"] and free >= required:
                return store
        raise RuntimeError("no empty fourth slot satisfying actual floor+reserve+CP budget")

    @contextmanager
    def operation(self, kind: str, *, source: Path | None = None):
        """Hold rank0's shared lease throughout the existing collective operation."""
        import torch.distributed as dist
        from wm3d.training.distributed_checkpoint import DistributedCheckpointManager
        if kind not in {"save", "restore"} or (kind == "restore") != (source is not None):
            raise RuntimeError("invalid checkpoint operation")
        if not dist.is_initialized():
            raise RuntimeError("checkpoint storage selection requires the original process group")
        descriptor = None
        verdict = [None]
        if dist.get_rank() == 0:
            try:
                real_directory(self.lock_path.parent)
                descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise RuntimeError("invalid shared operation lock")
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                store = self._empty_store()
                root = Path(store["root"])
                if source is not None:
                    matches = [item for item in self.members if Path(item["root"]) == source.parent]
                    if len(matches) != 1:
                        raise RuntimeError("restore source outside explicit original roots")
                    native_record(source, matches[0])
                    root = source.parent
                verdict[0] = {"ok": True, "root": str(root), "snapshot_parent": store["snapshot_parent"]}
            except Exception as exc:
                verdict[0] = {"ok": False, "error": str(exc)}
        dist.broadcast_object_list(verdict, src=0)
        try:
            if verdict[0]["ok"] is not True:
                raise RuntimeError(f"native storage selection failed: {verdict[0]}")
            self._active = True
            yield DistributedCheckpointManager(
                Path(verdict[0]["root"]), snapshot_parent=Path(verdict[0]["snapshot_parent"]))
            dist.barrier()
        finally:
            self._active = False
            if descriptor is not None:
                os.close(descriptor)

    def publish_index(self) -> None:
        """Point to actual unchanged commits, never substitute model-only exports."""
        import torch.distributed as dist
        from wm3d.training.distributed_checkpoint import _atomic_json, _collective_error
        error = None
        if dist.get_rank() == 0:
            try:
                rows = self.records()
                _atomic_json(self.index_path, committed_index(rows), exclusive=False)
            except Exception as exc:
                error = exc
        _collective_error("cross-store committed index publication", error)

    def preflight(self) -> dict[str, Any]:
        store = self._empty_store()
        return {"passed": True, "empty_fourth_slot": store["root"],
                "native_commits": 3, "snapshot_or_incomplete_residue": False}

    def latest_formal_committed(self) -> Path:
        index = read_json(self.index_path)
        if index.get("schema") != "wm3d_native_committed_index_v1":
            raise RuntimeError("invalid global native index")
        entries = [item for item in index["entries"] if item["run_lineage"] == self.formal_lineage]
        if not entries:
            raise RuntimeError("global index has no committed checkpoint for this formal run")
        entry = max(entries, key=lambda item: item["commit_mtime_ns"])
        path = Path(entry["path"])
        members = [item for item in self.members if Path(item["root"]) == path.parent]
        if len(members) != 1 or path.parent not in {Path(item["root"]) for item in self.stores}:
            raise RuntimeError("index escaped current formal roots")
        record = native_record(path, members[0])
        if (record["files"]["COMMITTED.json"] != entry["commit_file_identity"]
                or record["records"]["COMMITTED.json"] != entry["original_COMMITTED"]):
            raise RuntimeError("global index no longer points to the actual original commit")
        return path

    def inspect_committed(self, *, path: Path, expected):
        from wm3d.training.distributed_checkpoint import DistributedCheckpointManager
        path = Path(path)
        matches = [item for item in self.members if Path(item["root"]) == path.parent]
        if len(matches) != 1 or path.parent not in {Path(item["root"]) for item in self.stores}:
            raise RuntimeError("formal resume must use this formal run, never legacy short-run weights")
        native_record(path, matches[0])
        return DistributedCheckpointManager(path.parent).inspect_committed(
            path=path, expected=expected)

    def save(self, **kwargs):
        with self.operation("save") as manager:
            path = manager.save(**kwargs)
            self.publish_index()
            return path

    def load(self, *, path: Path, **kwargs):
        with self.operation("restore", source=Path(path)) as manager:
            return manager.load(path=path, **kwargs)
