"""Strict final-model-only initialization for the three authorized SFT jobs.

Uses the complete current FoundationWorldModel before FSDP/Adam construction.
Only rank0 reads the consolidated file, paced at 128 MiB/s, then broadcasts
the actual parameters and buffers. It never restores the pretraining optimizer.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import stat
import time
from typing import Mapping

import torch
import torch.distributed as dist

SCHEMA = "wm3d_benchmark_final_model_initialization_v1"
BENCHMARKS = {"LIBERO", "WorldArena", "RoboNet"}
AUTHORITY = Path("/data/Minko/wm3d_downstream_20261028/USER_START_AFTER_PRETRAINING.json")
HANDOFF = Path("/data/Minko/wm3d_native_fresh_pretraining_20260930/downstream_handoff_20261028")
FINAL = HANDOFF.parent/"formal_27source_2ep_20261005"
RATE = 128 * 1024 * 1024


def identity(path):
    p = Path(path)
    s = p.lstat()
    if not stat.S_ISREG(s.st_mode) or p.is_symlink():
        raise ValueError("benchmark initialization requires a regular immutable source")
    # Stable over the existing NFS client mount; no payload hash or new scan.
    return [s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def _read(path):
    return json.loads(Path(path).read_text())


def validate_declaration(value):
    if set(value) != {"schema", "benchmark", "source_step", "authority_path",
                      "handoff_manifest_path", "source_model_config_path"}:
        raise ValueError("benchmark initialization declaration fields differ")
    if value["schema"] != SCHEMA or value["benchmark"] not in BENCHMARKS or value["source_step"] != 167446:
        raise ValueError("only the requested three jobs and exact final167446 are authorized")
    if Path(value["authority_path"]) != AUTHORITY:
        raise ValueError("benchmark initialization authority path differs")
    if Path(value["handoff_manifest_path"]) != HANDOFF/"MODEL_ONLY_READY.json":
        raise ValueError("benchmark initialization must use the final shared model-only handoff")
    if Path(value["source_model_config_path"]) != HANDOFF/"SOURCE_MODEL_CONFIG.json":
        raise ValueError("benchmark source model declaration differs")
    auth = _read(AUTHORITY)
    if not auth["automatic_start_authorized"] or value["benchmark"] not in auth["only_benchmarks"]:
        raise ValueError("benchmark has no direct execution authority")
    source = _read(value["handoff_manifest_path"])
    completion = _read(FINAL/"INSPECTION_FINAL_PRETRAINING_20261028_2012.json")
    if not (source["passed"] and source["source_step"] == 167446 and source["model_only"]
            and source["source_unchanged"] and source["all_model_tensors_read_finite"]
            and completion["passed"] and completion["pretraining_complete"]
            and completion["optimizer_step"] == 167446 and completion["all_tensors_finite"]
            and completion["all_24_saved_sampler_and_RNG_deserialized"]):
        raise ValueError("actual final CPU integrity/storage completion is missing")
    cp = Path(source["source_checkpoint"])
    if cp != FINAL/"training/checkpoints/step_00167446":
        raise ValueError("cannot fall back to an older model")
    if identity(cp/"COMMITTED.json") != source["source_commit_file_identity"]:
        raise ValueError("final checkpoint commit identity changed")
    if identity(cp/"COMMITTED.json") != completion["final_commit_file_identity"]:
        raise ValueError("completion proof belongs to another final checkpoint")
    retirement = _read(completion["normal_retention_completed"])
    if not (retirement["passed"] and retirement["physical_release"]["deleted"]
            and retirement["retained_steps"] == [150000, 160000, 167446]
            and retirement["preflight"]["passed"]):
        raise ValueError("normal final checkpoint retention has not completed")
    model_file = Path(source["model_file"])
    if model_file != HANDOFF/"final_model.safetensors":
        raise ValueError("unexpected final model-only payload")
    sig = source["model_file_identity"]
    expected = [sig[k] for k in ("inode", "size_bytes", "mtime_ns", "ctime_ns")]
    if identity(model_file) != expected:
        raise ValueError("final model-only identity changed")
    proof = _read(source["source_own_full_CPU_proof"])
    if not (proof["passed"] and proof["all_tensors_finite"] and proof["source_unchanged"]
            and proof["adam_states"] == 2157 and proof["adam_moment_tensors"] == 4314
            and proof["adam_step_histogram"] == {"167446": 2157}):
        raise ValueError("own final full-state proof is not the verified source")
    return source


def validate_model_inheritance(declaration, model_profile, train):
    source = validate_declaration(declaration)
    original = _read(declaration["source_model_config_path"])
    if train.get("frozen_geometry_teacher") != original["frozen_geometry_teacher"]:
        raise ValueError("downstream must preserve the source frozen Omega teacher")
    if model_profile["architecture"] != original["model_profile"]["architecture"]:
        raise ValueError("downstream architecture differs from final pretraining")
    mutable = {"action_loss_weight", "video_loss_weight", "native_3d_loss_weight",
               "wan_context_store_path"}
    actual = model_profile["model"]
    expected = original["model_profile"]["model"]
    if set(actual) != set(expected) or any(actual[k] != expected[k] for k in expected if k not in mutable):
        raise ValueError("downstream changed the inherited native model architecture")
    if any(train.get(k) is not None for k in (
        "model_warmstart_checkpoint", "formal_sampling_transition",
        "formal_omega_transition", "native_bounded_continuation")):
        raise ValueError("new benchmark SFT cannot restore or rearm old training transitions")
    return source


@torch.no_grad()
def initialize_final_model(model, declaration):
    """Every original parameter and buffer is loaded; no missing-key tolerance."""
    from safetensors import safe_open
    source = validate_declaration(declaration)
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    state = model.state_dict()
    if len(state) != source["model_tensors"] or sum(t.numel() for t in state.values()) != source["model_elements"]:
        raise ValueError("actual FoundationWorldModel does not have the final model layout")
    layout = [(k, list(t.shape), str(t.dtype)) for k,t in sorted(state.items())]
    if distributed:
        layouts = [None] * dist.get_world_size()
        dist.all_gather_object(layouts, layout)
        if any(v != layout for v in layouts):
            raise ValueError("benchmark rank model layouts disagree before initialization")
    result = [None]
    before = identity(source["model_file"])
    if rank == 0:
        try:
            started = time.monotonic()
            read_bytes = 0
            with safe_open(source["model_file"], framework="pt", device="cpu") as reader:
                if set(reader.keys()) != set(state):
                    raise ValueError("final handoff keys do not exactly match the current model")
                for name, target in state.items():
                    original = reader.get_tensor(name)
                    if original.shape != target.shape or original.dtype != target.dtype:
                        raise ValueError("final handoff shape/dtype differs: " + name)
                    target.copy_(original, non_blocking=False)
                    read_bytes += original.numel() * original.element_size()
                    delay = read_bytes / RATE - (time.monotonic() - started)
                    if delay > 0:
                        time.sleep(delay)
                    del original
            if identity(source["model_file"]) != before:
                raise ValueError("final model changed during strict initialization")
            result[0] = dict(passed=True, source_step=167446, model_tensors=len(state),
                model_elements=source["model_elements"], strict_all_parameters_and_buffers=True,
                source_model_file=source["model_file"], source_model_identity=before,
                rank0_actual_read_bytes=read_bytes, rank0_rate_bytes_per_second=RATE,
                seconds=time.monotonic()-started, optimizer_created_or_restored=False,
                sampler_or_RNG_restored=False)
        except Exception as exc:
            result[0] = dict(passed=False, error=repr(exc))
    if distributed:
        dist.broadcast_object_list(result, src=0)
    if not result[0]["passed"]:
        raise RuntimeError("strict final model initialization failed: "+str(result[0]))
    if distributed:
        for tensor in state.values():
            dist.broadcast(tensor, src=0)
    finite = all(not t.is_floating_point() or bool(torch.isfinite(t).all()) for t in state.values())
    flag = torch.tensor(int(finite), dtype=torch.int32, device=next(model.parameters()).device)
    if distributed:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise FloatingPointError("strictly initialized model is not finite on every rank")
    if identity(source["model_file"]) != before:
        raise ValueError("final model-only source changed during rank broadcast")
    return dict(result[0], all_ranks_finite=True, rank=rank)
