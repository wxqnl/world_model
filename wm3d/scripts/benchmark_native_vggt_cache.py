"""Measure the production frozen VGGT cache on exact sampler-addressed raw data.

Preparation is CPU-only. Evaluation requires an otherwise idle GPU, not a
training GPU shared with another process. No model/optimizer/checkpoint changes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch

from wm3d.training.runtime_contract import load_materialized_runtime


def prepare(args, config):
    from wm3d.training.pretrain import _build_mixed_dataset, _make_loader

    args.output.mkdir(parents=True, exist_ok=False)
    dataset, profile = _build_mixed_dataset(config, split="train", device=torch.device("cpu"))
    runtime = config["runtime_profile"]
    model = config["model_profile"]["model"]
    first = int(model["T"]) - 1
    length = int(model["K"]) + 1
    manifest = {"runtime": str(args.runtime), "start_step": args.start_step,
                "steps": args.steps, "world_size": 16, "ranks": args.ranks,
                "sampling_changed": False, "cases": []}
    for rank in args.ranks:
        loader = _make_loader(
            dataset, profile, runtime, rank=rank, world_size=16,
            start_step=args.start_step, num_steps=args.steps,
            seed=int(runtime["train"]["seed"]), gradient_accumulation=1,
        )
        for offset, batch in enumerate(loader):
            name = f"rank{rank:02d}_step{args.start_step + offset + 1:08d}.pt"
            # Only the production VGGT input slice; do not write RGB/action targets
            # as model context or resample windows to increase cache hits.
            case = {
                "images": batch["direct_rgb_uint8"][:, first:first + length].contiguous(),
                "view_mask": batch["direct_view_mask"][:, first:first + length].contiguous(),
                "frame_keys": batch["direct_frame_keys"][:, first:first + length].contiguous(),
                "source_id": batch["source_id"], "sample_index": batch["sample_index"],
            }
            with (args.output / name).open("xb") as handle:
                torch.save(case, handle)
            entry = {"file": name, "rank": rank, "step": args.start_step + offset + 1,
                     "source_id": case["source_id"].tolist(),
                     "sample_index": case["sample_index"].tolist(),
                     "shape": list(case["images"].shape)}
            manifest["cases"].append(entry)
            print(json.dumps(entry), flush=True)
    dataset._executor.shutdown(wait=True)
    with (args.output / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)


def evaluate(args, config):
    from wm3d.data.foundation_input import build_foundation_input_adapter

    output = args.output / f"receipt_rank{args.rank:02d}.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((args.output / "manifest.json").read_text())
    cases = [item for item in manifest["cases"] if item["rank"] == args.rank]
    if not cases:
        raise ValueError("No prepared cases for this sampler rank")
    adapter = build_foundation_input_adapter(config, device="cuda:0")
    adapter.vggt_cache_limit = args.cache_gib * 1024**3
    adapter.eval()
    results = []
    # Warm only kernels/weights; discard cache before the timed natural sequence.
    warm = torch.load(args.output / cases[0]["file"], weights_only=True)
    adapter._encode_native_rows(warm["images"][:1, :2], warm["view_mask"][:1, :2])
    torch.cuda.synchronize()
    for entry in cases:
        case = torch.load(args.output / entry["file"], weights_only=True)
        images, mask, keys = (case[k] for k in ("images", "view_mask", "frame_keys"))
        rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
        start = time.perf_counter()
        reference = adapter._encode_native_rows(images, mask)
        torch.cuda.synchronize()
        uncached_seconds = time.perf_counter() - start
        before = dict(adapter.metrics)
        start = time.perf_counter()
        reused = adapter._cached_native_rows(images, mask, keys)
        torch.cuda.synchronize()
        cached_seconds = time.perf_counter() - start
        equal = torch.equal(reference, reused)
        max_error = float((reference.float() - reused.float()).abs().max())
        # Cache must not consume model noise RNG or yield inference tensors.
        rng_equal = torch.equal(rng_cpu, torch.get_rng_state()) and torch.equal(
            rng_cuda, torch.cuda.get_rng_state())
        finite = bool(torch.isfinite(reused).all())
        normal_tensor = not torch.is_inference(reused)
        record = {**entry, "uncached_seconds": uncached_seconds,
                  "cached_seconds": cached_seconds, "bitwise_equal": equal,
                  "max_abs_error": max_error, "rng_unchanged": rng_equal,
                  "finite": finite, "ordinary_tensor": normal_tensor,
                  "cache_delta": {k: adapter.metrics[k] - before[k] for k in (
                      "vggt_cache_hits", "vggt_cache_misses",
                      "vggt_coalesced_rows", "vggt_encoded_rows")},
                  "cache_bytes": adapter.vggt_cache_bytes}
        results.append(record)
        print(json.dumps(record), flush=True)
        if not (equal and rng_equal and finite and normal_tensor):
            break
    passed = len(results) == len(cases) and all(
        r["bitwise_equal"] and r["rng_unchanged"] and r["finite"] and r["ordinary_tensor"]
        for r in results)
    receipt = {"passed": passed, "scope": "Real frozen VGGT only; not full-step or generation quality",
               "sampler_rank": args.rank, "cache_gib": args.cache_gib, "results": results,
               "uncached_mean_seconds": statistics.mean(r["uncached_seconds"] for r in results),
               "cached_mean_seconds": statistics.mean(r["cached_seconds"] for r in results)}
    with output.open("x") as handle:
        json.dump(receipt, handle, indent=2)
    print(json.dumps({k: v for k, v in receipt.items() if k != "results"}), flush=True)
    if not passed:
        raise RuntimeError("Frozen feature/RNG equivalence failed; do not enable cache")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "evaluate"))
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-step", type=int, default=480)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 8])
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--cache-gib", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config, _ = load_materialized_runtime(args.runtime)
    if args.mode == "prepare":
        if torch.cuda.is_available():
            raise RuntimeError("Prepare with CUDA_VISIBLE_DEVICES=''")
        prepare(args, config)
    else:
        evaluate(args, config)


if __name__ == "__main__":
    main()
