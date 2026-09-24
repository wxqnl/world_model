"""Compare official gradient clipping kernels on the complete production mesh.

Uses all real trainable tensor shapes and saved first moments as replay gradients.
This is an operator benchmark, not a model update or action/RGB quality test.
"""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != 16:
        raise ValueError("Requires the production 2 x 8 mesh")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    try:
        mesh = init_device_mesh("cuda", (2, 8), mesh_dim_names=("replicate", "shard"))
        metadata = dcp.FileSystemReader(args.checkpoint / "distcp").read_metadata().state_dict_metadata
        prefix, suffix = "optimizer.state.", ".exp_avg"
        names = sorted(k[len(prefix):-len(suffix)] for k in metadata
                       if k.startswith(prefix) and k.endswith(suffix))
        saved = {n: {"exp_avg": empty(metadata["model." + n].size, device_mesh=mesh,
                                     placements=(Replicate(), Shard(0)), dtype=torch.float32)}
                 for n in names}
        dcp.load({"optimizer": {"state": saved}}, checkpoint_id=args.checkpoint / "distcp")
        parameters = [[torch.nn.Parameter(torch.empty_like(saved[n]["exp_avg"]))
                       for n in names] for _ in range(2)]
        results = []
        for iteration in range(6):
            for params in parameters:
                for i, (n, p) in enumerate(zip(names, params)):
                    p.grad = None if i % 31 == 0 else saved[n]["exp_avg"].clone()
            # Cover clipped and unclipped cases, alternating kernel ordering.
            max_norm = .01 if iteration % 2 == 0 else 1e9
            norms, timings = {}, {}
            for index in ((0, 1) if iteration % 2 == 0 else (1, 0)):
                dist.barrier()
                torch.cuda.synchronize()
                start = time.perf_counter()
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters[index], max_norm, foreach=(None if index == 0 else True))
                norm = norm.full_tensor() if hasattr(norm, "full_tensor") else norm
                torch.cuda.synchronize()
                seconds = torch.tensor(time.perf_counter() - start, device=device)
                dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
                norms[index], timings[index] = norm, seconds.item()
            torch.testing.assert_close(norms[0], norms[1], rtol=2e-6, atol=1e-7)
            errors = []
            for a, b in zip(*parameters):
                if a.grad is not None:
                    x, y = a.grad.to_local(), b.grad.to_local()
                    torch.testing.assert_close(x, y, rtol=2e-6, atol=1e-7)
                    if x.numel():
                        errors.append((x - y).abs().max())
            error = torch.stack(errors).max()
            dist.all_reduce(error, op=dist.ReduceOp.MAX)
            record = {"iteration": iteration, "max_norm": max_norm,
                      "default_seconds": timings[0], "foreach_seconds": timings[1],
                      "norm": norms[0].item(), "max_abs_error": error.item()}
            results.append(record)
            if rank == 0:
                print(json.dumps(record), flush=True)
        receipt = {"passed": True, "rank": rank, "tensor_count": len(names),
                   "scope": "Official clipping operator replay; not training or generation quality",
                   "results": results}
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / f"rank_{rank:05d}.json").open("x") as handle:
            json.dump(receipt, handle, indent=2)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
