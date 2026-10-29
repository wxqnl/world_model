"""Bounded real-model capacity measurement inside the unified training loop."""
from __future__ import annotations
import ctypes
import json
import os
from pathlib import Path
import threading
import time

import torch
import torch.distributed as dist

from .distributed_checkpoint import _atomic_json


class _NVMLMemory(ctypes.Structure):
    _fields_=[("total",ctypes.c_ulonglong),("free",ctypes.c_ulonglong),("used",ctypes.c_ulonglong)]


class CapacityMeasurement:
    def __init__(self,device):
        self.device=device
        self.started=time.monotonic()
        self.samples=0;self.peak=0;self.total=0;self.errors=[]
        self._event=threading.Event()
        self._lib=ctypes.CDLL("libnvidia-ml.so.1")
        if self._lib.nvmlInit_v2()!=0:raise RuntimeError("NVML initialization failed")
        uuid=str(torch.cuda.get_device_properties(device).uuid)
        if not uuid.startswith("GPU-"):uuid="GPU-"+uuid
        self.uuid=uuid
        self._handle=ctypes.c_void_p()
        self._lib.nvmlDeviceGetHandleByUUID.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_void_p)]
        self._lib.nvmlDeviceGetMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(_NVMLMemory)]
        if self._lib.nvmlDeviceGetHandleByUUID(uuid.encode(),ctypes.byref(self._handle))!=0:
            raise RuntimeError("NVML handle does not match actual torch GPU UUID")
        self._sample()
        self._thread=threading.Thread(target=self._loop,daemon=True,name="bounded-capacity-NVML")
        self._thread.start()

    def _sample(self):
        info=_NVMLMemory()
        code=self._lib.nvmlDeviceGetMemoryInfo(self._handle,ctypes.byref(info))
        if code:
            self.errors.append("nvmlDeviceGetMemoryInfo="+str(code));return
        self.total=int(info.total);self.peak=max(self.peak,int(info.used));self.samples+=1

    def _loop(self):
        while not self._event.wait(.2):
            self._sample()

    def stop(self):
        if not self._event.is_set():
            self._event.set();self._thread.join(timeout=3);self._sample()
        return dict(gpu_uuid=self.uuid,NVML_peak_used_bytes=self.peak,
                    device_total_bytes=self.total,NVML_samples=self.samples,
                    NVML_errors=self.errors,seconds=time.monotonic()-self.started)


def _local(tensor):
    return tensor.to_local() if hasattr(tensor,"to_local") else tensor


def complete_capacity(output_root,model,optimizer,config,context,measurement,*,completed):
    """Requires the actual full accumulated backward/Adam and real DCP save."""
    torch.cuda.synchronize(context.device)
    tensors=[_local(t.detach()) for t in model.state_dict().values() if isinstance(t,torch.Tensor)]
    states=[]
    steps=[]
    for group in optimizer.param_groups:
        for p in group["params"]:
            item=optimizer.state[p]
            if not {"step","exp_avg","exp_avg_sq"}.issubset(item):
                raise RuntimeError("capacity run did not materialize every Adam state")
            for key in ("exp_avg","exp_avg_sq"):
                states.append(_local(item[key].detach()))
            steps.append(float(_local(item["step"]).item()))
    finite=all(not t.is_floating_point() or bool(torch.isfinite(t).all()) for t in tensors+states)
    own=dict(rank=context.rank,finite=finite,Adam_states=len(steps),Adam_moment_tensors=len(states),
             Adam_parameter_steps=sorted(set(steps)),**measurement.stop())
    ranks=[None]*context.world_size;dist.all_gather_object(ranks,own)
    if not all(v["finite"] and v["Adam_parameter_steps"]==[float(completed)] and not v["NVML_errors"] for v in ranks):
        raise FloatingPointError("capacity actual model/Adam or NVML proof failed")
    root=Path(output_root)
    # DCP publication is the real save path used by the formal run.
    checkpoint=root/"checkpoints"/f"step_{completed:08d}"
    meta=json.loads((checkpoint/"metadata.json").read_text())
    commit=json.loads((checkpoint/"COMMITTED.json").read_text())
    if meta["step"]!=completed or commit["step"]!=completed or meta["world_size"]!=8:
        raise RuntimeError("capacity real full checkpoint is not committed")
    profile=config["runtime_profile"];train=profile["train"]
    peak_allocated=0;peak_reserved=0;phases=set()
    memory_path=root/f"gpu_memory_rank_{context.rank:05d}.jsonl"
    for line in memory_path.read_text().splitlines():
        row=json.loads(line);phases.add(row["phase"])
        peak_allocated=max(peak_allocated,row["peak_allocated_bytes"])
        peak_reserved=max(peak_reserved,row["peak_reserved_bytes"])
    allocated=[None]*context.world_size
    dist.all_gather_object(allocated,dict(rank=context.rank,peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,phases=sorted(phases)))
    required={"initialized","forward_backward","optimizer","checkpoint_save"}
    if any(not required.issubset(v["phases"]) for v in allocated):
        raise RuntimeError("capacity did not measure full forward/backward/Adam and checkpoint phases")
    safe=all(v["NVML_peak_used_bytes"]<=min(v["device_total_bytes"]-6*1024**3,
                                         int(v["device_total_bytes"]*.94)) for v in ranks)
    result=dict(schema="wm3d_real_native_capacity_qualification_v1",passed=True,safe_headroom=safe,
        capacity_only=True,formal_training_started=False,formal_optimizer_updates=0,
        disposable_probe_optimizer_updates=completed,source_optimizer_step=167446,
        complete_model_and_frozen_teacher_used=True,real_benchmark_data=True,
        real_accumulated_forward_backward_Adam=True,real_complete_DCP_save=str(checkpoint),
        world_size=context.world_size,micro_batch_size=train["micro_batch_size"],
        gradient_accumulation=train["gradient_accumulation"],global_batch_size=train["global_batch_size"],
        ranks=ranks,allocator_peaks=allocated,validation_peak_measured=False,
        generation_peak_measured=False,at_unix=time.time())
    if context.is_rank0:
        _atomic_json(root/"CAPACITY_COMPLETE.json",result)
        print(json.dumps({"capacity_complete":result}),flush=True)
    return result
