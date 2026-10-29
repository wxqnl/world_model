"""Bound benchmark-native closures for the unified trainer (no legacy model path)."""
from __future__ import annotations
import json
import math
from pathlib import Path

from .benchmark_catalog import binding, read_rows

SCHEMA = "wm3d_benchmark_native_data_closure_v1"


def verify(path, expected):
    actual = binding(path)
    # Existing NFS mounts have a different st_dev but preserve source inode.
    fields = ("inode", "size_bytes", "mtime_ns", "ctime_ns")
    if any(actual[k] != expected[k] for k in fields):
        raise ValueError("benchmark immutable source identity changed: "+str(path))


def validate_closure(closure, model_profile, runtime_profile):
    required = {"schema", "admission_id", "benchmark", "suite",
                "catalog_path", "catalog_binding", "normalization_path",
                "normalization_binding", "raw_file_bindings", "encoder_contract_path",
                "encoder_contract_binding", "CPU_materialization_proof_path",
                "CPU_materialization_proof_binding"}
    if set(closure) != required or closure["schema"] != SCHEMA:
        raise ValueError("benchmark native closure fields/schema differ")
    if closure["benchmark"] != "LIBERO":
        raise ValueError("this native reader currently implements LIBERO only")
    from .benchmark_libero import SUITES, SCHEMA as NORMALIZATION_SCHEMA
    if closure["suite"] not in SUITES:
        raise ValueError("unknown LIBERO suite")
    for path_key, binding_key in (
        ("catalog_path", "catalog_binding"),
        ("normalization_path", "normalization_binding"),
        ("encoder_contract_path", "encoder_contract_binding"),
        ("CPU_materialization_proof_path", "CPU_materialization_proof_binding"),
    ):
        p = Path(closure[path_key])
        if not p.is_absolute():
            raise ValueError("benchmark source paths must be absolute")
        verify(p, closure[binding_key])
    rows = list(read_rows(closure["catalog_path"]))
    if len(rows) != 500 or any(r["split"] != "train" or r["suite"] != closure["suite"] for r in rows):
        raise ValueError("LIBERO training must use all 500 original suite demonstrations")
    normal = json.loads(Path(closure["normalization_path"]).read_text())
    if normal["schema"] != NORMALIZATION_SCHEMA or normal["suite"] != closure["suite"]:
        raise ValueError("LIBERO normalization belongs to another source")
    if normal["fitting_split"] != "train" or normal["train_demos"] != 500:
        raise ValueError("normalization may not include validation or test")
    files = closure["raw_file_bindings"]
    if set(files) != {r["file"] for r in rows} or set(files) != set(normal["files"]):
        raise ValueError("LIBERO asset closure does not cover the 10 original task HDF5 files")
    for path, sig in files.items():
        verify(path, sig)
        if sig["size_bytes"] != normal["files"][path]["size_bytes"]:
            raise ValueError("deployed LIBERO asset size differs from the normalized source")
    proof = json.loads(Path(closure["CPU_materialization_proof_path"]).read_text())
    suite_proof = proof["suites"][closure["suite"]]
    windows = sum(r["full_future_window_starts"] for r in rows)
    if not (proof["normalization_train_only"] and suite_proof["foundation_bridge_passed"]
            and suite_proof["actual_current_and_future_RGB_decoded"]
            and suite_proof["valid_windows"] == windows):
        raise ValueError("real native CPU data materialization has not completed")
    model = model_profile["model"]
    for k,v in dict(T=16,K=8,P=64,num_views=3,max_action_groups=3,max_action_dim=14,max_state_dim=10).items():
        if model[k] != v:
            raise ValueError("LIBERO native architecture capacity differs from the final source")
    from .foundation_input import FoundationContextStore
    bank = FoundationContextStore.from_directory(model["wan_context_store_path"])
    bank.lookup(sorted({r["instruction"] for r in rows}))
    if bank.prompt_format != model["wan_prompt_format"]:
        raise ValueError("LIBERO instruction feature format differs from the native model")
    train = runtime_profile["train"]
    if (train["global_batch_size"] != 128
            or train["total_steps"] != math.ceil(windows*40/128)
            or train["validate_every"] != 0):
        raise ValueError("LIBERO fixed full-suite exposure/final-checkpoint protocol changed")
    declaration = train.get("downstream_initialization")
    if not declaration or declaration["benchmark"] != "LIBERO":
        raise ValueError("benchmark closure requires explicit final model-only SFT initialization")
    return dict(native_windows=windows, train_demos=500, train_only_normalization=True)


def build_dataset(closure, *, split):
    if split != "train":
        raise ValueError("LIBERO simulator evaluation is separate from its full demonstration training split")
    from .benchmark_libero import LiberoNativeDataset
    dataset = LiberoNativeDataset(closure["catalog_path"], closure["normalization_path"],
        file_bindings=closure["raw_file_bindings"], catalog_binding=closure["catalog_binding"])
    return dataset, dataset.profile
