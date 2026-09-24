"""Replay AdamW on ALL trainable tensors from a real production DCP snapshot.

This measures only the optimizer, not RGB/action quality or full training speed.
The saved first moment supplies an identical nonzero replay gradient to both
implementations; no checkpoint or training run is modified.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import empty, Replicate, Shard
from wm3d.training.distributed_runtime import initialize_adamw_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 16:
        raise ValueError("This production replay requires the original 2 x 8 mesh")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    try:
        mesh = init_device_mesh("cuda", (2, 8), mesh_dim_names=("replicate", "shard"))
        checkpoint = args.checkpoint / "distcp"
        metadata = dcp.FileSystemReader(checkpoint).read_metadata().state_dict_metadata
        prefix, suffix = "optimizer.state.", ".exp_avg"
        names = sorted(k[len(prefix):-len(suffix)] for k in metadata
                       if k.startswith(prefix) and k.endswith(suffix))
        def allocate(name):
            spec = metadata["model." + name]
            assert spec.properties.dtype == torch.float32
            return empty(spec.size, device_mesh=mesh,
                         placements=(Replicate(), Shard(0)), dtype=torch.float32)
        state = {"model": {name: allocate(name) for name in names},
                 "optimizer": {"state": {
                     name: {"exp_avg": allocate(name), "exp_avg_sq": allocate(name),
                            "step": torch.zeros((), dtype=torch.float32)}
                     for name in names}}}
        print(json.dumps({"rank": rank, "phase": "load_real_optimizer",
                          "tensors": len(names),
                          "global_parameters": sum(metadata["model."+n].size.numel()
                                                   for n in names)}), flush=True)
        dcp.load(state, checkpoint_id=checkpoint)
        base = [torch.nn.Parameter(state["model"][n]) for n in names]
        fused = [torch.nn.Parameter(p.detach().clone()) for p in base]
        def groups(params):
            video = [p for n,p in zip(names,params) if ".mixtures.video." in n]
            other = [p for n,p in zip(names,params) if ".mixtures.video." not in n]
            return [{"params": video, "lr": 1e-5}, {"params": other, "lr": 1e-4}]
        optimizers = [
            torch.optim.AdamW(groups(params), betas=(.9,.95), eps=1e-8,
                              weight_decay=.01, foreach=False, fused=flag)
            for params,flag in ((base,False),(fused,True))
        ]
        for optimizer, params in zip(optimizers, (base,fused)):
            initialize_adamw_state(optimizer)
            for index,(name,param) in enumerate(zip(names,params)):
                saved = state["optimizer"]["state"][name]
                optimizer.state[param]["exp_avg"].copy_(saved["exp_avg"])
                optimizer.state[param]["exp_avg_sq"].copy_(saved["exp_avg_sq"])
                optimizer.state[param]["step"].copy_(saved["step"])
                # Preserve a non-updated group as well as active tensors.
                param.grad = None if index % 31 == 0 else saved["exp_avg"].clone()
        def timed_step(optimizer):
            torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = torch.tensor(time.perf_counter()-start, device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            return elapsed.item()
        times = {"single": [], "fused": []}
        for step in range(8):
            # Alternate execution order to reduce warm-clock ordering bias.
            order = (0,1) if step % 2 == 0 else (1,0)
            for index in order:
                elapsed = timed_step(optimizers[index])
                if step >= 2:
                    times[("single","fused")[index]].append(elapsed)
        errors = []
        for a,b in zip(base,fused):
            pairs = [(a,b)]
            for key in ("exp_avg","exp_avg_sq","step"):
                pairs.append((optimizers[0].state[a][key],
                              optimizers[1].state[b][key]))
            for x,y in pairs:
                x = x.to_local() if hasattr(x,"to_local") else x
                y = y.to_local() if hasattr(y,"to_local") else y
                x,y = x.to(device), y.to(device)
                if x.numel():
                    torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-7)
                    errors.append((x-y).abs().max())
        max_error = torch.stack(errors).max()
        dist.all_reduce(max_error,op=dist.ReduceOp.MAX)
        result = {"rank":rank, "passed":True, "scope":"optimizer replay only",
                  "tensor_count":len(names), "seconds":times,
                  "max_abs_error":max_error.item(),
                  "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30}
        # Exercise DCP's actual fused device-step save/load layout on genuine
        # tensors from each expert, without writing another 76 GiB checkpoint.
        picked = [next(n for n in names if ".mixtures."+expert+"." in n
                       and n.endswith(".bias")) for expert in ("video","action","future_3d")]
        sample = {n: optimizers[1].state[fused[names.index(n)]] for n in picked}
        roundtrip = args.output / "fused_state_roundtrip"
        args.output.mkdir(parents=True,exist_ok=True)
        dcp.save({"optimizer":sample}, checkpoint_id=roundtrip)
        restored = {n:{k:torch.zeros_like(v) for k,v in sample[n].items()} for n in picked}
        dcp.load({"optimizer":restored}, checkpoint_id=roundtrip)
        for n in picked:
            for k,v in sample[n].items():
                a = v.to_local() if hasattr(v,"to_local") else v
                b = restored[n][k]
                b = b.to_local() if hasattr(b,"to_local") else b
                torch.testing.assert_close(a,b,rtol=0,atol=0)
        result["device_step_dcp_roundtrip"] = True
        (args.output/f"rank_{rank:05d}.json").write_text(json.dumps(result,indent=2)+"\n")
        if rank==0:
            print(json.dumps(result),flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
