"""Serialize three benchmark saves on the existing shared volume.

This composes the original transactional DCP manager and preserves its format.
It only retires this SFT run's oldest committed checkpoint after a new commit,
keeping two latest full states. Pretraining checkpoints are never candidates.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import time

import torch.distributed as dist

from .distributed_checkpoint import DistributedCheckpointManager, _atomic_json, _fsync_directory
from .checkpoint_stores import real_directory

BASE = Path("/data/Minko/wm3d_native_fresh_pretraining_20260930")
RUNS = BASE/"downstream_runs_20261028"
MIN_FREE = 200111600640


def _sig(path):
    s = Path(path).lstat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError("non-regular SFT checkpoint member")
    return [s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def _tree(path):
    result = {}
    for directory, dirs, names in os.walk(path, followlinks=False):
        for name in dirs:
            p=Path(directory)/name
            if p.is_symlink():
                raise ValueError("symlink in SFT checkpoint")
        for name in names:
            p=Path(directory)/name
            result[str(p.relative_to(path))]=_sig(p)
    return result


def _actual_node():
    ips=subprocess.check_output(["hostname","-I"],text=True).split()
    matches=[node for node,ip in [(43,"172.27.0.6"),(44,"172.27.0.7"),(41,"172.27.0.4")] if ip in ips]
    if len(matches)!=1:raise ValueError("unknown benchmark host identity")
    return matches[0]


def _host_path(relative):
    rel=Path(relative)
    if rel.is_absolute() or ".." in rel.parts or len(rel.parts)!=4:
        raise ValueError("reader target must be downstream_runs/run/checkpoints/numbered_step")
    if rel.parts[0]!="downstream_runs_20261028" or rel.parts[2]!="checkpoints" or not re.fullmatch(r"step_[0-9]{8}",rel.parts[3]):
        raise ValueError("checkpoint reader review cannot address pretraining or other files")
    node=_actual_node()
    root=BASE if node==43 else Path(f"/data/Minko/wm3d_native_fresh_node{node}_20260930/mounts/run")
    return root/rel


def scan_local(relative):
    target=_host_path(relative);real_directory(target)
    members=_tree(target)
    targets={(os.lstat(target/name).st_dev,info[0]) for name,info in members.items()}
    identities={(os.major(dev),os.minor(dev),inode) for dev,inode in targets}
    hits=[];errors=[]
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():
            continue
        pid=int(proc.name)
        try:
            cmd=(proc/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
            observer=("wm3d.training.benchmark_checkpoint" in cmd or "downstream-checkpoint-readers" in cmd)
            # Observer command lines necessarily contain the reviewed target;
            # their actual FDs and mmap objects are still checked.
            if relative in cmd and not observer:
                hits.append(dict(pid=pid,kind="direct_command_reference",command=cmd[:500]))
            for fd in (proc/"fd").iterdir():
                try:
                    st=fd.stat()
                    if (st.st_dev,st.st_ino) in targets:
                        hits.append(dict(pid=pid,kind="fd",fd=fd.name))
                except FileNotFoundError:
                    pass
                except PermissionError as exc:
                    errors.append(dict(pid=pid,kind="fd_permission",error=str(exc)))
            for line in (proc/"maps").read_text().splitlines():
                fields=line.split(None,5)
                if len(fields)<5 or int(fields[4])==0:continue
                major,minor=(int(v,16) for v in fields[3].split(":"))
                if (major,minor,int(fields[4])) in identities:
                    hits.append(dict(pid=pid,kind="mmap",mapping=line[:700]))
        except (FileNotFoundError,ProcessLookupError):
            pass
        except PermissionError as exc:
            errors.append(dict(pid=pid,kind="proc_permission",error=str(exc)))
        except OSError as exc:
            errors.append(dict(pid=pid,kind="proc_read_error",error=str(exc)))
    if _tree(target)!=members:raise ValueError("checkpoint changed during reader review")
    return dict(at_unix=time.time(),node=_actual_node(),target=str(target),
                relative=relative,file_count=len(members),hits=hits,errors=errors,
                passed=not hits and not errors,observer_pid=os.getpid())


def review_all(relative):
    if _actual_node()!=43:raise ValueError("multi-host review starts on43")
    rows=[scan_local(relative)]
    root="/data/Minko/wm3d_downstream_20261028/code/wm3d"
    for node,ip in [(44,"172.27.0.7"),(41,"172.27.0.4")]:
        python="/data/Minko/.venvs/wm3d/bin/python" if node==41 else "/data/Minko/.venvs/wm3d-foundation-v10-final/bin/python"
        import shlex
        command="cd /data/Minko && CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="+root+" "+python+" -m wm3d.training.benchmark_checkpoint scan --relative "+shlex.quote(relative)
        cp=subprocess.run(["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=15","-S",str(BASE/"control_20261001"/f"ssh{node}.sock"),"root@"+ip,command],capture_output=True,text=True,timeout=180)
        if cp.returncode:
            rows.append(dict(node=node,passed=False,error=cp.stderr[-3000:],returncode=cp.returncode))
        else:
            rows.append(json.loads(cp.stdout))
    return dict(at_unix=time.time(),relative=relative,nodes=rows,
                passed=len(rows)==3 and all(r["passed"] for r in rows))


def remote_review(target):
    relative=str(target.relative_to(BASE))
    if _actual_node()==43:return review_all(relative)
    import shlex
    root="/data/Minko/wm3d_downstream_20261028/code/wm3d"
    command="cd /data/Minko && CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHON_BIN=/data/Minko/.venvs/wm3d-foundation-v10-final/bin/python bash "+root+"/run_wm3d.sh downstream-checkpoint-readers review --relative "+shlex.quote(relative)
    cp=subprocess.run(["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=15","root@172.27.0.6",command],text=True,capture_output=True,timeout=600)
    if cp.returncode:raise RuntimeError("SFT reader review failed: "+cp.stderr[-3000:])
    return json.loads(cp.stdout)


class BenchmarkCheckpointManager:
    def __init__(self, root, *, lineage):
        self.root=Path(root)
        if self.root.parent.parent!=RUNS or self.root.name!="checkpoints":
            raise ValueError("SFT checkpoint root must be an explicit new benchmark run")
        self.lineage=lineage
        self.inner=DistributedCheckpointManager(self.root)
        self.snapshot_parent=None

    def __getattr__(self,name):
        return getattr(self.inner,name)

    def _commits(self):
        rows=[]
        for path in self.root.iterdir():
            if path.name.startswith(".step_") and ".incomplete." in path.name:
                raise RuntimeError("existing incomplete SFT save must be preserved and investigated")
            if not re.fullmatch(r"step_[0-9]{8}",path.name):continue
            metadata,_=self.inner._verify_controls(path)
            if metadata["run_lineage"]!=self.lineage or metadata["world_size"]!=8:
                raise ValueError("foreign checkpoint in own SFT run")
            if metadata["sampler_progress"]["next_optimizer_step"]!=metadata["step"]:
                raise ValueError("SFT sampler checkpoint progress differs")
            rows.append((metadata["step"],path))
        return sorted(rows)

    def save(self,**kwargs):
        rank=dist.get_rank();fd=None
        gate=[None]
        if rank==0:
            try:
                RUNS.mkdir(exist_ok=True,parents=True);real_directory(RUNS)
                self.root.mkdir(exist_ok=True,parents=True);real_directory(self.root)
                fd=os.open(RUNS/".checkpoint_operation.lock",os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
                fcntl.flock(fd,fcntl.LOCK_EX)
                free=os.statvfs(RUNS);available=free.f_bavail*free.f_frsize
                if available<MIN_FREE:raise RuntimeError(f"SFT actual free below save floor: {available} < {MIN_FREE}")
                if kwargs["metadata"]["run_lineage"]!=self.lineage:raise ValueError("SFT save lineage differs")
                self._commits()
                gate[0]=dict(passed=True,actual_free_before=available)
            except Exception as exc:
                gate[0]=dict(passed=False,error=repr(exc))
        dist.broadcast_object_list(gate,src=0)
        if not gate[0]["passed"]:
            if fd is not None:os.close(fd)
            raise RuntimeError("SFT checkpoint save gate: "+str(gate[0]))
        try:
            result=self.inner.save(**kwargs)
            maintenance=[None]
            if rank==0:
                try:
                    rows=self._commits()
                    retired=[]
                    for step,target in rows[:-2]:
                        before=_tree(target)
                        review=remote_review(target)
                        if not review["passed"] or time.time()-review["at_unix"]>=600:
                            raise RuntimeError("reader review blocks SFT retention: "+json.dumps(review))
                        if _tree(target)!=before:raise RuntimeError("old SFT checkpoint changed after reader review")
                        local=scan_local(str(target.relative_to(BASE)))
                        if not local["passed"]:raise RuntimeError("old SFT checkpoint acquired a local reader")
                        self.inner._verify_controls(target)
                        _atomic_json(self.root.parent/f"RETIREMENT_REVIEW_{step:08d}_{time.time_ns()}.json",
                            dict(passed=True,target=str(target),files=before,review=review,
                                 newest_step=kwargs["step"],lineage=self.lineage))
                        shutil.rmtree(target);_fsync_directory(self.root)
                        if target.exists():raise RuntimeError("SFT target still exists after retirement")
                        _atomic_json(self.root.parent/f"RETIREMENT_COMPLETED_{step:08d}_{time.time_ns()}.json",
                            dict(passed=True,deletion_completed=True,removed_step=step,newest_step=kwargs["step"],lineage=self.lineage))
                        retired.append(step)
                    remaining=self._commits()
                    _atomic_json(self.root.parent/"checkpoint_index.json",
                        dict(schema="wm3d_benchmark_committed_checkpoint_index_v1",
                             lineage=self.lineage,entries=[dict(step=s,path=str(p)) for s,p in remaining],
                             actual_free=os.statvfs(RUNS).f_bavail*os.statvfs(RUNS).f_frsize),
                        exclusive=False)
                    maintenance[0]=dict(passed=True,retired_steps=retired,retained_steps=[s for s,p in remaining])
                except Exception as exc:
                    # Preserve the successful new commit and continue healthy
                    # updates. Unknown readers never become "no readers".
                    maintenance[0]=dict(passed=False,error=repr(exc),new_commit_preserved=str(result))
                    _atomic_json(self.root.parent/f"RETENTION_DEFERRED_{time.time_ns()}.json",maintenance[0])
                finally:
                    fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd);fd=None
            dist.broadcast_object_list(maintenance,src=0)
            if rank==0:print(json.dumps({"benchmark_checkpoint":str(result),"storage_gate":gate[0],"retention":maintenance[0]}),flush=True)
            return result
        finally:
            if rank==0 and fd is not None:
                fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)

    def load(self,**kwargs):
        # Own exact resume uses the unchanged native complete-state loader.
        # Save and retirement share the same storage-wide lock.
        rank=dist.get_rank();fd=None;gate=[None]
        if rank==0:
            try:
                fd=os.open(RUNS/".checkpoint_operation.lock",os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
                fcntl.flock(fd,fcntl.LOCK_EX)
                rows=self._commits()
                if not rows or Path(kwargs["path"])!=rows[-1][1]:
                    raise ValueError("SFT resume may not roll back the latest own complete checkpoint")
                v=os.statvfs(RUNS)
                if v.f_bavail*v.f_frsize<MIN_FREE:raise RuntimeError("SFT restore actual free below original floor")
                gate[0]=dict(passed=True)
            except Exception as exc:gate[0]=dict(passed=False,error=repr(exc))
        dist.broadcast_object_list(gate,src=0)
        if not gate[0]["passed"]:
            if fd is not None:os.close(fd)
            raise RuntimeError("SFT exact restore gate: "+str(gate[0]))
        try:return self.inner.load(**kwargs)
        finally:
            if rank==0:
                fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)


def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["scan","review"])
    p.add_argument("--relative",required=True);a=p.parse_args()
    if Path.cwd()!=Path("/data/Minko"):raise ValueError("checkpoint safety commands start in /data/Minko")
    result=scan_local(a.relative) if a.mode=="scan" else review_all(a.relative)
    print(json.dumps(result,separators=(",",":")))
if __name__=="__main__":main()
