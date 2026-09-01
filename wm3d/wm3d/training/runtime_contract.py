"""Strict materialized runtime contract for the unified WM3D trainer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from wm3d.data.cache_tasks import (
    CACHE_EPISODE_SEAL_SCHEMA,
    CACHE_WINDOW_SEAL_SCHEMA,
)
from wm3d.data.manifest_contract import SHA256_RE, load_data_profile, sha256_file
from wm3d.data.direct_raw import DIRECT_RAW_DATA_CLOSURE_SCHEMA
from wm3d.data.streaming_raw import (
    STREAMING_DATA_CLOSURE_SCHEMA,
    load_streaming_metadata_seal,
)
from wm3d.data.formal_cache_adapter import (
    FORMAL_CACHE_CLOSURE_SCHEMA,
    FormalCacheError,
    validate_formal_cache_closure,
)
from wm3d.models.model_factory import (
    validate_model_data_compatibility,
    validate_model_profile,
)
from wm3d.training.distributed_runtime import strategy_from_mapping
from wm3d.training.native_objective import objective_config_from_mapping
from wm3d.training.rgb_flow_runtime import raft_config_from_mapping


RUNTIME_PROFILE_SCHEMA = "wm3d_v8_runtime_profile_v1"
RUNTIME_CONFIG_SCHEMA = "wm3d_v8_materialized_runtime_v2"
MODEL_PROFILE_SCHEMA = "wm3d_v8_model_profile_v1"
OBJECTIVE_PROFILE_SCHEMA = "wm3d_v8_objective_profile_v1"
DATA_CLOSURE_SCHEMA = "wm3d_v8_dataset_closure_v2"
_PLACEHOLDER = re.compile(r"PENDING|MATERIALIZE_REQUIRED|__")


_STREAMING_MODEL_DATA_NON_BINDING_FIELDS = {
    "appearance_enabled",
    "appearance_P",
    "appearance_context_frames",
    "appearance_hidden",
    "appearance_layers",
    "appearance_heads",
    "appearance_ff_mult",
    "appearance_autoregressive_steps",
    "appearance_action_residual_scale",
    "appearance_flow_aligned_detail",
    "appearance_state_detail",
    "appearance_detail_dim",
    "policy_task_modulation",
    "policy_calibration_conditioning",
    "factual_dynamics_repeats",
    "factual_action_residual_scale",
    "factual_v7_early_action_conditioning",
    "factual_v7_early_action_scale",
    "factual_v7_bridge_layers_state",
    "render_factual_dynamics_repeats",
    "render_factual_action_residual_scale",
    "rgb_context_enabled",
    "rgb_original_v7_context",
    "rgb_v7_high_frequency_refiner",
    "rgb_v7_high_frequency_channels",
    "rgb_v7_high_frequency_scale",
    "rgb_context_alignment_enabled",
    "rgb_render_action_free_prior",
    "rgb_context_residual_scale",
    "rgb_context_motion_blend_gain",
    "rgb_context_action_scale",
    "rgb_context_appearance_delta_scale",
    "rgb_detail_residual_scale",
    # Decoder capacity, batching, and the subset of raw-video horizons decoded
    # for RGB supervision do not alter the sealed window/token metadata.
    "rgb_hidden",
    "rgb_res_blocks",
    "rgb_decode_chunk_size",
    "rgb_decode_indices",
}


class RuntimeContractError(ValueError):
    pass


def _streaming_model_data_core(model_profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return model fields that constrain an existing streaming-data seal."""
    return {
        name: item
        for name, item in model_profile["model"].items()
        if name not in _STREAMING_MODEL_DATA_NON_BINDING_FIELDS
    }


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeContractError(f"YAML root must be a mapping: {path}")
    return value


def _require_sha(value: object, field: str) -> str:
    text = str(value)
    if SHA256_RE.fullmatch(text) is None:
        raise RuntimeContractError(f"{field} must be lowercase SHA256")
    return text


def _strict_profile(
    value: Mapping[str, Any], *, schema: str, allowed: set[str], label: str
) -> None:
    if value.get("schema") != schema:
        raise RuntimeContractError(f"{label} schema must be {schema}")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise RuntimeContractError(f"unknown {label} fields: {unknown}")
    if not str(value.get("name", "")):
        raise RuntimeContractError(f"{label} name must be non-empty")


def validate_runtime_profile(value: Mapping[str, Any]) -> None:
    _strict_profile(
        value,
        schema=RUNTIME_PROFILE_SCHEMA,
        allowed={
            "schema",
            "name",
            "expected_world_size",
            "distributed",
            "resources",
            "train",
            "optimizer",
            "schedule",
            "checkpoint",
        },
        label="runtime profile",
    )
    expected_world_size = int(value["expected_world_size"])
    if expected_world_size <= 0:
        raise RuntimeContractError("expected_world_size must be positive")
    distributed = value.get("distributed")
    if not isinstance(distributed, dict):
        raise RuntimeContractError("distributed must be a mapping")
    strategy_from_mapping(distributed).validate(world_size=expected_world_size)
    resources = value.get("resources")
    if resources is not None:
        if not isinstance(resources, dict):
            raise RuntimeContractError("resources must be a mapping")
        required_resources = {
            "gpu_name_substring",
            "minimum_gpu_memory_mib",
            "require_zero_uncorrected_ecc",
            "require_idle_gpu",
            "require_full_local_nvlink_clique",
            "minimum_ib_rate_gbps",
            "forbid_nccl_ib_disable",
            "minimum_memlock_bytes",
            "minimum_nofile",
            "minimum_shm_bytes",
            "minimum_data_free_bytes",
            "minimum_output_free_bytes",
            "minimum_allreduce_gbps",
            "maximum_preflight_age_seconds",
        }
        if set(resources) != required_resources:
            raise RuntimeContractError(
                "resource fields mismatch: "
                f"missing={sorted(required_resources-set(resources))} "
                f"unknown={sorted(set(resources)-required_resources)}"
            )
        if not str(resources["gpu_name_substring"]):
            raise RuntimeContractError("resources.gpu_name_substring must be non-empty")
        for field in (
            "minimum_gpu_memory_mib",
            "minimum_memlock_bytes",
            "minimum_nofile",
            "minimum_shm_bytes",
            "minimum_data_free_bytes",
            "minimum_output_free_bytes",
            "maximum_preflight_age_seconds",
        ):
            if isinstance(resources[field], bool) or int(resources[field]) <= 0:
                raise RuntimeContractError(f"resources.{field} must be a positive integer")
        for field in ("minimum_ib_rate_gbps", "minimum_allreduce_gbps"):
            if isinstance(resources[field], bool) or float(resources[field]) <= 0:
                raise RuntimeContractError(f"resources.{field} must be positive")
        for field in (
            "require_zero_uncorrected_ecc",
            "require_idle_gpu",
            "require_full_local_nvlink_clique",
            "forbid_nccl_ib_disable",
        ):
            if not isinstance(resources[field], bool):
                raise RuntimeContractError(f"resources.{field} must be boolean")
    train = value.get("train")
    optimizer = value.get("optimizer")
    schedule = value.get("schedule")
    checkpoint = value.get("checkpoint")
    if not all(isinstance(item, dict) for item in (train, optimizer, schedule, checkpoint)):
        raise RuntimeContractError(
            "train/optimizer/schedule/checkpoint must all be mappings"
        )
    required_train = {
        "micro_batch_size",
        "gradient_accumulation",
        "global_batch_size",
        "total_steps",
        "seed",
        "validation_seed",
        "num_workers",
        "prefetch_factor",
        "persistent_workers",
        "gradient_clip",
        "log_every",
        "validate_every",
        "validation_steps",
        "checkpoint_interval",
        "checkpoint_steps",
    }
    optional_train = {
        "validation_micro_batch_size",
        "activation_checkpointing",
        "cudnn_benchmark",
        "rgb_decode_chunk_size",
        "rgb_perceptual_chunk_size",
        "rgb_flow_teacher",
        "appearance_teacher_start_ratio",
        "appearance_teacher_end_ratio",
        "appearance_teacher_decay_steps",
        "appearance_teacher0_every_steps",
        "appearance_validation_three_way",
        "model_warmstart_checkpoint",
        "model_warmstart_new_parameter_prefixes",
    }
    if not required_train.issubset(train) or set(train) - required_train - optional_train:
        raise RuntimeContractError(
            f"train fields mismatch: missing={sorted(required_train-set(train))} "
            f"unknown={sorted(set(train)-required_train-optional_train)}"
        )
    derived_global = (
        expected_world_size
        * int(train["micro_batch_size"])
        * int(train["gradient_accumulation"])
    )
    if derived_global != int(train["global_batch_size"]):
        raise RuntimeContractError(
            f"global batch {train['global_batch_size']} != derived {derived_global}"
        )
    for field in (
        "micro_batch_size",
        "gradient_accumulation",
        "global_batch_size",
        "total_steps",
        "log_every",
        "validation_steps",
        "checkpoint_interval",
    ):
        if int(train[field]) <= 0:
            raise RuntimeContractError(f"train.{field} must be positive")
    if (
        "validation_micro_batch_size" in train
        and int(train["validation_micro_batch_size"]) <= 0
    ):
        raise RuntimeContractError(
            "train.validation_micro_batch_size must be positive"
        )
    if (
        "activation_checkpointing" in train
        and not isinstance(train["activation_checkpointing"], bool)
    ):
        raise RuntimeContractError("train.activation_checkpointing must be boolean")
    if (
        "cudnn_benchmark" in train
        and not isinstance(train["cudnn_benchmark"], bool)
    ):
        raise RuntimeContractError("train.cudnn_benchmark must be boolean")
    if (
        "rgb_decode_chunk_size" in train
        and (
            not isinstance(train["rgb_decode_chunk_size"], int)
            or isinstance(train["rgb_decode_chunk_size"], bool)
            or train["rgb_decode_chunk_size"] <= 0
        )
    ):
        raise RuntimeContractError("train.rgb_decode_chunk_size must be a positive integer")
    if (
        "rgb_perceptual_chunk_size" in train
        and (
            not isinstance(train["rgb_perceptual_chunk_size"], int)
            or isinstance(train["rgb_perceptual_chunk_size"], bool)
            or train["rgb_perceptual_chunk_size"] <= 0
        )
    ):
        raise RuntimeContractError(
            "train.rgb_perceptual_chunk_size must be a positive integer"
        )
    rgb_flow_teacher = train.get("rgb_flow_teacher")
    if rgb_flow_teacher is not None:
        if not isinstance(rgb_flow_teacher, dict):
            raise RuntimeContractError("train.rgb_flow_teacher must be a mapping")
        try:
            flow_config = raft_config_from_mapping(rgb_flow_teacher)
        except (TypeError, ValueError) as exc:
            raise RuntimeContractError(
                f"invalid train.rgb_flow_teacher: {exc}"
            ) from exc
        for field in ("source_root", "checkpoint"):
            path = Path(str(getattr(flow_config, field)))
            if not path.is_absolute():
                raise RuntimeContractError(
                    f"train.rgb_flow_teacher.{field} must be absolute"
                )
        if flow_config.input_size <= 0 or flow_config.input_size % 8:
            raise RuntimeContractError(
                "train.rgb_flow_teacher.input_size must be positive and divisible by 8"
            )
        if (
            flow_config.iterations <= 0
            or flow_config.output_grid <= 0
            or flow_config.batch_chunk <= 0
            or flow_config.flow_max_pixels <= 0.0
        ):
            raise RuntimeContractError(
                "train.rgb_flow_teacher numeric fields must be positive"
            )
        if (
            flow_config.consistency_relative < 0.0
            or flow_config.consistency_absolute < 0.0
        ):
            raise RuntimeContractError(
                "train.rgb_flow_teacher consistency fields cannot be negative"
            )
    if int(train["num_workers"]) < 0 or int(train["prefetch_factor"]) <= 0:
        raise RuntimeContractError("dataloader worker/prefetch values are invalid")
    if float(train["gradient_clip"]) <= 0:
        raise RuntimeContractError("train.gradient_clip must be positive")
    appearance_schedule = {
        "appearance_teacher_start_ratio",
        "appearance_teacher_end_ratio",
        "appearance_teacher_decay_steps",
    }
    present_appearance_schedule = appearance_schedule & set(train)
    if present_appearance_schedule and present_appearance_schedule != appearance_schedule:
        raise RuntimeContractError(
            "appearance teacher schedule fields must be provided together"
        )
    if present_appearance_schedule:
        start_ratio = float(train["appearance_teacher_start_ratio"])
        end_ratio = float(train["appearance_teacher_end_ratio"])
        if not 0.0 <= end_ratio <= start_ratio <= 1.0:
            raise RuntimeContractError(
                "appearance teacher ratios must satisfy 0 <= end <= start <= 1"
            )
        decay_steps = train["appearance_teacher_decay_steps"]
        if (
            isinstance(decay_steps, bool)
            or int(decay_steps) <= 0
            or int(decay_steps) > int(train["total_steps"])
        ):
            raise RuntimeContractError(
                "train.appearance_teacher_decay_steps must lie within total_steps"
            )
    teacher0_every = train.get("appearance_teacher0_every_steps")
    if teacher0_every is not None:
        if (
            isinstance(teacher0_every, bool)
            or not isinstance(teacher0_every, int)
            or teacher0_every < 2
        ):
            raise RuntimeContractError(
                "train.appearance_teacher0_every_steps must be an integer >= 2"
            )
        if not present_appearance_schedule:
            raise RuntimeContractError(
                "periodic teacher0 training requires an appearance teacher schedule"
            )
    appearance_validation_three_way = train.get(
        "appearance_validation_three_way", False
    )
    if not isinstance(appearance_validation_three_way, bool):
        raise RuntimeContractError(
            "train.appearance_validation_three_way must be boolean"
        )
    if appearance_validation_three_way and not present_appearance_schedule:
        raise RuntimeContractError(
            "three-way appearance validation requires an appearance teacher schedule"
        )
    warmstart_path = train.get("model_warmstart_checkpoint")
    warmstart_prefixes = train.get("model_warmstart_new_parameter_prefixes")
    if (warmstart_path is None) != (warmstart_prefixes is None):
        raise RuntimeContractError(
            "model warmstart checkpoint and new parameter prefixes are required together"
        )
    if warmstart_path is not None:
        checkpoint_path = Path(str(warmstart_path))
        if not checkpoint_path.is_absolute() or re.fullmatch(
            r"step_[0-9]{8}", checkpoint_path.name
        ) is None:
            raise RuntimeContractError(
                "model warmstart must be an absolute numbered checkpoint"
            )
        if (
            not isinstance(warmstart_prefixes, list)
            or not warmstart_prefixes
            or any(
                not isinstance(prefix, str)
                or not prefix
                or prefix.startswith(".")
                for prefix in warmstart_prefixes
            )
        ):
            raise RuntimeContractError(
                "model warmstart new parameter prefixes must be explicit names"
            )
        if len(set(warmstart_prefixes)) != len(warmstart_prefixes):
            raise RuntimeContractError(
                "model warmstart new parameter prefixes contain duplicates"
            )
    if optimizer.get("name") != "adamw":
        raise RuntimeContractError("optimizer.name must be adamw")
    start_lr = float(optimizer.get("start_lr", 0.0))
    peak_lr = float(optimizer["peak_lr"])
    min_lr = float(optimizer["min_lr"])
    if not 0.0 <= start_lr <= peak_lr or not 0.0 < min_lr <= peak_lr:
        raise RuntimeContractError(
            "optimizer learning rates must satisfy 0 <= start <= peak and 0 < min <= peak"
        )
    if schedule.get("name") != "warmup_stable_cosine":
        raise RuntimeContractError("schedule.name must be warmup_stable_cosine")
    total_steps = int(train["total_steps"])
    if not 0 <= int(schedule["warmup_steps"]) < total_steps:
        raise RuntimeContractError("schedule warmup lies outside total steps")
    if not 0.0 < float(schedule["stable_fraction"]) < 1.0:
        raise RuntimeContractError("schedule.stable_fraction must be in (0,1)")


def validate_data_closure(value: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "name",
        "data_profile_path",
        "data_profile_sha256",
        "cache_root",
        "episode_cache_index_path",
        "episode_cache_index_sha256",
        "episode_cache_seal_path",
        "episode_cache_seal_sha256",
        "cache_index_path",
        "cache_index_sha256",
        "cache_seal_path",
        "cache_seal_sha256",
        "grouped_normalization_path",
        "grouped_normalization_sha256",
        "source_manifest_sha256_by_name",
        "adapter_contract_sha256_by_name",
    }
    if set(value) != required or value.get("schema") != DATA_CLOSURE_SCHEMA:
        raise RuntimeContractError("dataset closure fields/schema mismatch")
    for field in (
        "data_profile_sha256",
        "episode_cache_index_sha256",
        "episode_cache_seal_sha256",
        "cache_index_sha256",
        "cache_seal_sha256",
        "grouped_normalization_sha256",
    ):
        _require_sha(value[field], field)
    source_digests = value["source_manifest_sha256_by_name"]
    if not isinstance(source_digests, dict) or not source_digests:
        raise RuntimeContractError("dataset closure has no source manifest digests")
    for name, digest in source_digests.items():
        _require_sha(digest, f"source manifest {name}")
    adapter_digests = value["adapter_contract_sha256_by_name"]
    if not isinstance(adapter_digests, dict) or set(adapter_digests) != set(source_digests):
        raise RuntimeContractError(
            "adapter contract digests must exactly match closure sources"
        )
    for name, digest in adapter_digests.items():
        _require_sha(digest, f"adapter contract {name}")
    root = Path(str(value["cache_root"]))
    if not root.is_absolute():
        raise RuntimeContractError("dataset closure cache_root must be absolute")
    for path_field, digest_field in (
        ("data_profile_path", "data_profile_sha256"),
        ("episode_cache_index_path", "episode_cache_index_sha256"),
        ("episode_cache_seal_path", "episode_cache_seal_sha256"),
        ("cache_index_path", "cache_index_sha256"),
        ("cache_seal_path", "cache_seal_sha256"),
        ("grouped_normalization_path", "grouped_normalization_sha256"),
    ):
        path = Path(str(value[path_field]))
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise RuntimeContractError(f"closure {path_field} is not a regular absolute file")
        if sha256_file(path) != value[digest_field]:
            raise RuntimeContractError(f"closure {path_field} digest mismatch")
    # The profile itself is already SHA-bound.  At launch we compare its
    # declared source/adapter identities without rescanning all raw manifests;
    # the cache seal and per-shard lazy verification carry that provenance.
    profile = load_data_profile(
        Path(str(value["data_profile_path"])), verify_source_manifests=False
    )
    exact_source = {
        source.name: source.manifest_sha256 for source in profile.sources
    }
    exact_adapter = {
        source.name: source.adapter_contract_sha256 for source in profile.sources
    }
    if source_digests != exact_source:
        raise RuntimeContractError("closure source manifest digests disagree with profile")
    if adapter_digests != exact_adapter:
        raise RuntimeContractError("closure adapter digests disagree with profile")
    seal_path = Path(str(value["cache_seal_path"]))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    required_seal = {
        "schema",
        "episode_seal_path",
        "episode_seal_sha256",
        "episode_index_sha256",
        "data_profile_sha256",
        "model_profile_path",
        "model_profile_sha256",
        "encoder_contract_sha256",
        "task_encoder_contract_sha256",
        "task_bank_index_sha256",
        "representation_contract_sha256",
        "window_count",
        "window_index_path",
        "window_index_sha256",
    }
    if not isinstance(seal, dict) or set(seal) != required_seal:
        raise RuntimeContractError("cache seal fields mismatch")
    if seal.get("schema") != CACHE_WINDOW_SEAL_SCHEMA:
        raise RuntimeContractError("cache seal schema mismatch")
    if int(seal.get("window_count", 0)) <= 0:
        raise RuntimeContractError("cache seal counts must be positive")
    if Path(str(seal["window_index_path"])).resolve(strict=True) != Path(
        str(value["cache_index_path"])
    ).resolve(strict=True):
        raise RuntimeContractError("cache seal points at a different cache index")
    if seal["window_index_sha256"] != value["cache_index_sha256"]:
        raise RuntimeContractError("cache seal/index SHA mismatch")
    if seal["data_profile_sha256"] != value["data_profile_sha256"]:
        raise RuntimeContractError("window seal/data profile SHA mismatch")
    if seal["episode_index_sha256"] != value["episode_cache_index_sha256"]:
        raise RuntimeContractError("window seal/episode index SHA mismatch")
    episode_seal_path = Path(str(value["episode_cache_seal_path"]))
    if Path(str(seal["episode_seal_path"])).resolve(strict=True) != episode_seal_path.resolve(strict=True):
        raise RuntimeContractError("window seal points at another episode seal")
    if seal["episode_seal_sha256"] != value["episode_cache_seal_sha256"]:
        raise RuntimeContractError("window/episode seal SHA mismatch")
    episode_seal = json.loads(episode_seal_path.read_text(encoding="utf-8"))
    required_episode = {
        "schema", "task_manifest_path", "task_manifest_sha256", "task_count",
        "episode_count", "cache_root", "source_manifest_sha256_by_name",
        "adapter_contract_sha256_by_name", "encoder_contract_sha256",
        "task_encoder_contract_sha256",
        "task_bank_index_sha256",
        "representation_contract_sha256", "receipt_sha256_by_task", "episode_index_path",
        "episode_index_sha256", "payload_verification",
    }
    if not isinstance(episode_seal, dict) or set(episode_seal) != required_episode:
        raise RuntimeContractError("episode cache seal fields mismatch")
    if episode_seal.get("schema") != CACHE_EPISODE_SEAL_SCHEMA:
        raise RuntimeContractError("episode cache seal schema mismatch")
    if int(episode_seal.get("task_count", 0)) <= 0 or int(episode_seal.get("episode_count", 0)) <= 0:
        raise RuntimeContractError("episode cache seal counts must be positive")
    if episode_seal["task_count"] != episode_seal["episode_count"]:
        raise RuntimeContractError("episode cache requires exactly one episode per task")
    if Path(str(episode_seal["cache_root"])).resolve(strict=True) != root.resolve(strict=True):
        raise RuntimeContractError("episode seal points at another cache root")
    if episode_seal["source_manifest_sha256_by_name"] != source_digests:
        raise RuntimeContractError("episode seal/source manifest digests mismatch")
    if episode_seal["adapter_contract_sha256_by_name"] != adapter_digests:
        raise RuntimeContractError("episode seal/adapter contract digests mismatch")
    for field in (
        "encoder_contract_sha256",
        "task_encoder_contract_sha256",
        "task_bank_index_sha256",
        "representation_contract_sha256",
    ):
        _require_sha(episode_seal[field], f"episode seal {field}")
        if seal[field] != episode_seal[field]:
            raise RuntimeContractError(f"window/episode seal {field} mismatch")
    expected_representation_sha = canonical_sha256(profile.cache_representation)
    if episode_seal["representation_contract_sha256"] != expected_representation_sha:
        raise RuntimeContractError("episode seal/data representation digest mismatch")
    if episode_seal["payload_verification"] != (
        "generation_sha256_plus_seal_size_plus_lazy_open_sha256"
    ):
        raise RuntimeContractError("episode cache payload verification mode mismatch")
    receipts = episode_seal["receipt_sha256_by_task"]
    if (
        not isinstance(receipts, dict)
        or len(receipts) != int(episode_seal["task_count"])
        or any(SHA256_RE.fullmatch(str(item)) is None for item in receipts.values())
    ):
        raise RuntimeContractError("episode seal receipt digest set is invalid")
    if episode_seal["episode_index_sha256"] != value["episode_cache_index_sha256"]:
        raise RuntimeContractError("episode seal/index SHA mismatch")
    if Path(str(episode_seal["episode_index_path"])).resolve(strict=True) != Path(
        str(value["episode_cache_index_path"])
    ).resolve(strict=True):
        raise RuntimeContractError("episode seal points at another episode index")
    task_manifest_path = Path(str(episode_seal["task_manifest_path"])).resolve(
        strict=True
    )
    if sha256_file(task_manifest_path) != episode_seal["task_manifest_sha256"]:
        raise RuntimeContractError("episode seal/task manifest SHA mismatch")
    from wm3d.data.grouped_normalization import GroupedRobotNormalizer

    GroupedRobotNormalizer.load(
        Path(str(value["grouped_normalization_path"])),
        expected_sha256=value["grouped_normalization_sha256"],
        expected_data_profile_sha256=value["data_profile_sha256"],
        expected_model_profile_sha256=seal["model_profile_sha256"],
        expected_window_index_sha256=value["cache_index_sha256"],
        data_profile=profile,
    )


def validate_streaming_data_closure(value: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "name",
        "data_profile_path",
        "data_profile_sha256",
        "metadata_seal_path",
        "metadata_seal_sha256",
        "metadata_root",
        "episode_index_path",
        "episode_index_sha256",
        "cache_index_path",
        "cache_index_sha256",
        "grouped_normalization_path",
        "grouped_normalization_sha256",
        "task_manifest_path",
        "task_manifest_sha256",
        "encoder_contract_path",
        "encoder_contract_sha256",
        "task_bank_root",
        "task_bank_index_sha256",
        "source_manifest_sha256_by_name",
        "adapter_contract_sha256_by_name",
        "lru_root",
        "lru_max_bytes_per_rank",
        "encode_batch_frames",
        "decode_workers",
    }
    optional = {"appearance_token_grid", "appearance_feature_layer"}
    if not isinstance(value, dict) or not required.issubset(value) or set(value) - required - optional:
        raise RuntimeContractError("streaming_raw closure fields mismatch")
    if value.get("schema") != STREAMING_DATA_CLOSURE_SCHEMA:
        raise RuntimeContractError("streaming_raw closure schema mismatch")
    for field in (
        "data_profile_sha256",
        "metadata_seal_sha256",
        "episode_index_sha256",
        "cache_index_sha256",
        "grouped_normalization_sha256",
        "task_manifest_sha256",
        "encoder_contract_sha256",
        "task_bank_index_sha256",
    ):
        _require_sha(value[field], field)
    for field in (
        "lru_max_bytes_per_rank",
        "encode_batch_frames",
        "decode_workers",
    ):
        if isinstance(value[field], bool) or int(value[field]) <= 0:
            raise RuntimeContractError(f"streaming_raw {field} must be positive")
    if "appearance_token_grid" in value and (
        isinstance(value["appearance_token_grid"], bool)
        or int(value["appearance_token_grid"]) <= 0
    ):
        raise RuntimeContractError(
            "streaming_raw appearance_token_grid must be positive"
        )
    if "appearance_feature_layer" in value and (
        "appearance_token_grid" not in value
        or isinstance(value["appearance_feature_layer"], bool)
        or int(value["appearance_feature_layer"]) < 0
    ):
        raise RuntimeContractError(
            "streaming_raw appearance_feature_layer requires appearance tokens "
            "and must be non-negative"
        )
    for field in ("metadata_root", "task_bank_root"):
        root = Path(str(value[field]))
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise RuntimeContractError(f"streaming_raw {field} is invalid")
    lru_root = Path(str(value["lru_root"]))
    if not lru_root.is_absolute() or lru_root.is_symlink():
        raise RuntimeContractError("streaming_raw lru_root must be absolute/non-symlink")
    path_fields = {
        "data_profile_path": "data_profile_sha256",
        "metadata_seal_path": "metadata_seal_sha256",
        "episode_index_path": "episode_index_sha256",
        "cache_index_path": "cache_index_sha256",
        "grouped_normalization_path": "grouped_normalization_sha256",
        "task_manifest_path": "task_manifest_sha256",
        "encoder_contract_path": "encoder_contract_sha256",
    }
    for path_name, sha_name in path_fields.items():
        path = Path(str(value[path_name]))
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != value[sha_name]
        ):
            raise RuntimeContractError(f"streaming_raw {path_name} is invalid")
    try:
        seal = load_streaming_metadata_seal(
            Path(str(value["metadata_seal_path"])),
            expected_sha256=str(value["metadata_seal_sha256"]),
        )
    except Exception as exc:
        raise RuntimeContractError("streaming_raw metadata seal is invalid") from exc
    closure_to_seal = {
        "data_profile_path": "data_profile_path",
        "data_profile_sha256": "data_profile_sha256",
        "metadata_root": "metadata_root",
        "episode_index_path": "episode_index_path",
        "episode_index_sha256": "episode_index_sha256",
        "cache_index_path": "window_index_path",
        "cache_index_sha256": "window_index_sha256",
        "grouped_normalization_path": "grouped_normalization_path",
        "grouped_normalization_sha256": "grouped_normalization_sha256",
        "task_manifest_path": "task_manifest_path",
        "task_manifest_sha256": "task_manifest_sha256",
        "encoder_contract_path": "encoder_contract_path",
        "encoder_contract_sha256": "encoder_contract_sha256",
        "task_bank_root": "task_bank_root",
        "task_bank_index_sha256": "task_bank_index_sha256",
        "source_manifest_sha256_by_name": "source_manifest_sha256_by_name",
        "adapter_contract_sha256_by_name": "adapter_contract_sha256_by_name",
    }
    for closure_name, seal_name in closure_to_seal.items():
        if value[closure_name] != seal[seal_name]:
            raise RuntimeContractError(
                f"streaming_raw closure differs from metadata seal: {closure_name}"
            )
    profile = load_data_profile(
        Path(str(value["data_profile_path"])), verify_source_manifests=False
    )
    base_grid = int(profile.cache_representation["token_grid"])
    selected_appearance_grid = int(
        value.get("appearance_token_grid", base_grid)
    )
    profile_appearance_grid = int(
        profile.cache_representation.get("appearance_token_grid", base_grid)
    )
    if selected_appearance_grid not in {base_grid, profile_appearance_grid}:
        raise RuntimeContractError(
            "streaming_raw appearance_token_grid is neither disabled nor sealed"
        )
    # Selecting the base geometry grid explicitly disables the optional
    # appearance feature tap.  The layer remains sealed provenance for direct
    # raw closures, but NativeVGGTEncoder will not materialize it.
    if selected_appearance_grid != base_grid:
        profile_appearance_layer = profile.cache_representation.get(
            "appearance_feature_layer"
        )
        if profile_appearance_layer is not None and int(
            value.get("appearance_feature_layer", -1)
        ) != int(profile_appearance_layer):
            raise RuntimeContractError(
                "streaming_raw appearance_feature_layer differs from data profile"
            )
    if value["source_manifest_sha256_by_name"] != {
        source.name: source.manifest_sha256 for source in profile.sources
    }:
        raise RuntimeContractError("streaming_raw source manifest closure mismatch")
    if value["adapter_contract_sha256_by_name"] != {
        source.name: source.adapter_contract_sha256 for source in profile.sources
    }:
        raise RuntimeContractError("streaming_raw adapter closure mismatch")


def validate_direct_raw_data_closure(value: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "name",
        "data_profile_path",
        "data_profile_sha256",
        "metadata_seal_path",
        "metadata_seal_sha256",
        "metadata_root",
        "episode_index_path",
        "episode_index_sha256",
        "cache_index_path",
        "cache_index_sha256",
        "grouped_normalization_path",
        "grouped_normalization_sha256",
        "task_manifest_path",
        "task_manifest_sha256",
        "encoder_contract_path",
        "encoder_contract_sha256",
        "task_bank_root",
        "task_bank_index_sha256",
        "source_manifest_sha256_by_name",
        "adapter_contract_sha256_by_name",
        "appearance_token_grid",
        "appearance_feature_layer",
        "direct_input_rgb_size",
        "direct_decode_workers",
        "direct_robot_cache_episodes",
        "direct_prefetch_windows",
        "direct_video_index_cache_assets",
        "direct_encode_chunk_rows",
        "direct_minimum_chunk_rows",
    }
    optional = {"direct_ignored_action_dimensions"}
    fields = frozenset(value) if isinstance(value, dict) else frozenset()
    if fields not in {frozenset(required), frozenset(required | optional)}:
        raise RuntimeContractError("direct_raw closure fields mismatch")
    if value.get("schema") != DIRECT_RAW_DATA_CLOSURE_SCHEMA:
        raise RuntimeContractError("direct_raw closure schema mismatch")
    positive = (
        "direct_input_rgb_size",
        "direct_decode_workers",
        "direct_robot_cache_episodes",
        "direct_prefetch_windows",
        "direct_video_index_cache_assets",
        "direct_encode_chunk_rows",
        "direct_minimum_chunk_rows",
        "appearance_token_grid",
    )
    for field in positive:
        if isinstance(value[field], bool) or int(value[field]) <= 0:
            raise RuntimeContractError(f"direct_raw {field} must be positive")
    if int(value["direct_input_rgb_size"]) % 14:
        raise RuntimeContractError(
            "direct_raw input RGB size must be divisible by VGGT patch size 14"
        )
    if int(value["direct_minimum_chunk_rows"]) > int(
        value["direct_encode_chunk_rows"]
    ):
        raise RuntimeContractError(
            "direct_raw minimum chunk rows exceed the initial chunk"
        )
    if (
        isinstance(value["appearance_feature_layer"], bool)
        or int(value["appearance_feature_layer"]) not in (4, 11, 17, 23)
    ):
        raise RuntimeContractError(
            "direct_raw appearance layer must be a cached VGGT feature layer"
        )

    ignored = value.get("direct_ignored_action_dimensions", [])
    if not isinstance(ignored, list):
        raise RuntimeContractError(
            "direct_raw ignored action dimensions must be a list"
        )
    known_sources = set(value["source_manifest_sha256_by_name"])
    seen_ignored: set[tuple[str, str]] = set()
    for item in ignored:
        if (
            not isinstance(item, dict)
            or set(item) != {"source", "group", "dimensions"}
        ):
            raise RuntimeContractError(
                "direct_raw ignored action dimension entry is invalid"
            )
        source = str(item["source"])
        group = str(item["group"])
        dimensions = item["dimensions"]
        identity = (source, group)
        if (
            source not in known_sources
            or not group
            or identity in seen_ignored
            or not isinstance(dimensions, list)
            or not dimensions
            or any(isinstance(value, bool) for value in dimensions)
        ):
            raise RuntimeContractError(
                "direct_raw ignored action dimension identity is invalid"
            )
        normalized = [int(value) for value in dimensions]
        if (
            normalized != sorted(set(normalized))
            or any(value < 0 for value in normalized)
        ):
            raise RuntimeContractError(
                "direct_raw ignored action dimensions must be sorted unique non-negative integers"
            )
        seen_ignored.add(identity)

    direct_only = {
        "direct_ignored_action_dimensions",
        "direct_input_rgb_size",
        "direct_decode_workers",
        "direct_robot_cache_episodes",
        "direct_prefetch_windows",
        "direct_video_index_cache_assets",
        "direct_encode_chunk_rows",
        "direct_minimum_chunk_rows",
    }
    compatible = {
        name: item for name, item in value.items() if name not in direct_only
    }
    compatible.update(
        {
            "schema": STREAMING_DATA_CLOSURE_SCHEMA,
            "lru_root": value["metadata_root"],
            "lru_max_bytes_per_rank": 1,
            "encode_batch_frames": value["direct_encode_chunk_rows"],
            "decode_workers": value["direct_decode_workers"],
        }
    )
    validate_streaming_data_closure(compatible)


def validate_materialized_runtime(value: Mapping[str, Any]) -> None:
    allowed = {
        "schema",
        "run",
        "model_profile",
        "data_closure",
        "runtime_profile",
        "objective_profile",
        "bindings",
    }
    if set(value) != allowed or value.get("schema") != RUNTIME_CONFIG_SCHEMA:
        raise RuntimeContractError("materialized runtime fields/schema mismatch")
    serialized = json.dumps(value, sort_keys=True)
    if _PLACEHOLDER.search(serialized):
        raise RuntimeContractError("materialized runtime contains a placeholder")
    run = value["run"]
    required_run = {
        "name",
        "lineage",
        "output_root",
        "code_commit",
        "environment_lock_path",
        "environment_lock_sha256",
    }
    if not isinstance(run, dict) or set(run) != required_run:
        raise RuntimeContractError("runtime run fields mismatch")
    if not all(str(run[field]) for field in ("name", "lineage", "output_root", "code_commit")):
        raise RuntimeContractError("runtime run identity cannot be empty")
    if not Path(str(run["output_root"])).is_absolute():
        raise RuntimeContractError("run.output_root must be absolute")
    environment = Path(str(run["environment_lock_path"]))
    if not environment.is_absolute() or environment.is_symlink() or not environment.is_file():
        raise RuntimeContractError("environment lock must be an absolute regular file")
    if sha256_file(environment) != _require_sha(
        run["environment_lock_sha256"], "environment_lock_sha256"
    ):
        raise RuntimeContractError("environment lock SHA mismatch")

    model = value["model_profile"]
    validate_model_profile(model)
    runtime = value["runtime_profile"]
    validate_runtime_profile(runtime)
    objective = value["objective_profile"]
    _strict_profile(
        objective,
        schema=OBJECTIVE_PROFILE_SCHEMA,
        allowed={"schema", "name", "objective"},
        label="objective profile",
    )
    objective_config_from_mapping(objective["objective"])
    closure = value["data_closure"]
    if closure.get("schema") == FORMAL_CACHE_CLOSURE_SCHEMA:
        try:
            receipt = validate_formal_cache_closure(closure)
        except FormalCacheError as exc:
            raise RuntimeContractError(str(exc)) from exc
        representation = receipt["cache_representation"]
        required = {
            "T": 16,
            "P": int(representation["spatial_tokens"]),
            "K": 8,
            "token_dim": int(representation["token_dim"]),
            "task_dim": 2048,
            "num_views": int(representation["num_views"]),
            "rgb_size": int(representation["rgb_size"]),
        }
        for name, expected in required.items():
            if int(model["model"][name]) != expected:
                raise RuntimeContractError(
                    f"formal cache/model {name} mismatch: {model['model'][name]} != {expected}"
                )
    elif closure.get("schema") in {
        STREAMING_DATA_CLOSURE_SCHEMA,
        DIRECT_RAW_DATA_CLOSURE_SCHEMA,
    }:
        if closure.get("schema") == DIRECT_RAW_DATA_CLOSURE_SCHEMA:
            validate_direct_raw_data_closure(closure)
        else:
            validate_streaming_data_closure(closure)
        data_profile = load_data_profile(
            Path(str(closure["data_profile_path"])), verify_source_manifests=False
        )
        validate_model_data_compatibility(
            model,
            data_profile,
            appearance_cache_grid=int(
                closure.get(
                    "appearance_token_grid",
                    data_profile.cache_representation["token_grid"],
                )
            ),
        )
        metadata_seal = load_streaming_metadata_seal(
            Path(str(closure["metadata_seal_path"])),
            expected_sha256=str(closure["metadata_seal_sha256"]),
        )
        sealed_model = load_yaml(Path(str(metadata_seal["model_profile_path"])))
        sealed_core = _streaming_model_data_core(sealed_model)
        runtime_core = _streaming_model_data_core(model)
        if (
            sealed_model.get("schema") != model.get("schema")
            or sealed_model.get("architecture") != model.get("architecture")
            or sealed_model.get("sampling") != model.get("sampling")
            or sealed_core != runtime_core
        ):
            raise RuntimeContractError(
                "streaming metadata model data contract differs from runtime"
            )
    else:
        validate_data_closure(closure)
        data_profile = load_data_profile(
            Path(str(closure["data_profile_path"])), verify_source_manifests=False
        )
        validate_model_data_compatibility(model, data_profile)
        window_seal = json.loads(
            Path(str(closure["cache_seal_path"])).read_text(encoding="utf-8")
        )
        if window_seal["model_profile_sha256"] != canonical_sha256(model):
            raise RuntimeContractError("window seal belongs to a different model profile")
        sealed_model_path = Path(str(window_seal["model_profile_path"])).resolve(
            strict=True
        )
        sealed_model = load_yaml(sealed_model_path)
        if canonical_sha256(sealed_model) != window_seal["model_profile_sha256"]:
            raise RuntimeContractError("window seal model profile file/content mismatch")
        if sealed_model != model:
            raise RuntimeContractError("materialized runtime model differs from window seal")
    bindings = value["bindings"]
    required_bindings = {
        "model_profile_sha256",
        "data_closure_sha256",
        "runtime_profile_sha256",
        "objective_profile_sha256",
        "model_contract_sha256",
    }
    if not isinstance(bindings, dict) or set(bindings) != required_bindings:
        raise RuntimeContractError("runtime bindings mismatch")
    exact = {
        "model_profile_sha256": canonical_sha256(model),
        "data_closure_sha256": canonical_sha256(closure),
        "runtime_profile_sha256": canonical_sha256(runtime),
        "objective_profile_sha256": canonical_sha256(objective),
        "model_contract_sha256": canonical_sha256(
            {"architecture": model["architecture"], "model": model["model"]}
        ),
    }
    for name, expected in exact.items():
        observed = _require_sha(bindings.get(name), name)
        if observed != expected:
            raise RuntimeContractError(f"runtime binding {name} mismatch")


def load_materialized_runtime(path: Path) -> tuple[dict[str, Any], str]:
    path = Path(path).resolve(strict=True)
    value = load_yaml(path)
    validate_materialized_runtime(value)
    return value, sha256_file(path)
