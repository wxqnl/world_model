"""Strict Stage1 parent-load, optimizer, and recipe contracts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NATIVE_KEY_ALLOWLIST = (
    PROJECT_ROOT / "configs" / "wm3d_v6_stage1_native_key_allowlist.json"
)
STAGE0_PARENT_CHECKPOINT = Path(
    "/0604-10T-test/wm3d_v6/results/"
    "wm3d_v6_stage0_core_rgb_native3d_1b_rgbfirst_16gpu_"
    "node43_node44_v3_ib_b3/ckpt/best.pt"
)
STAGE0_PARENT_SHA256 = (
    "9f7bb2d357135ecaf67c19819e927f0f6f14646968df0ec75dd1b543fe4058c4"
)
STAGE1_SEMANTIC_RECIPE = "stage1_rgb_world_refine"
STAGE1_SEMANTIC_TRAINER = "stage1_rgb_world_refine"
STAGE1_FSDP_ROOTS = frozenset(
    {"wm", "wan_transformer", "wan_control_adapter"}
)
STAGE1_RESUME_UPDATES = (512, 1024, 2560, 5120, 9984)

_NATIVE_LR = 1.0e-7
_CONTEXT_PIXEL_LR = 4.0e-7
_WAN_CONTROL_LR = 1.0e-5
_WAN_TRANSFORMER_LR = 1.0e-7
_ALLOWED_WAN_BLOCKS = (28, 29)
_FORBIDDEN_WM_MODULES = (
    "action_policy",
    "action_proposer",
    "progress_head",
    "world_prior",
)


class Stage1ContractError(ValueError):
    """Raised when a strict Stage1 contract is not exact."""


@dataclass(frozen=True)
class LoadReport:
    checkpoint: str
    step: int
    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...] = ()
    owner: str = "model"

    @property
    def loaded_count(self) -> int:
        return len(self.loaded_keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "step": self.step,
            "owner": self.owner,
            "loaded_count": self.loaded_count,
            "loaded_keys": list(self.loaded_keys),
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
        }


@dataclass(frozen=True)
class CanonicalParameter:
    name: str
    module: str
    shape: tuple[int, ...]
    numel: int
    group: str
    lr: float
    parameter: nn.Parameter = field(repr=False, compare=False)

    @property
    def parameter_id(self) -> int:
        return id(self.parameter)


@dataclass(frozen=True)
class _CapturedParameter:
    name: str
    shape: tuple[int, ...]
    numel: int
    parameter: nn.Parameter = field(repr=False, compare=False)


@dataclass(frozen=True)
class CanonicalParamRegistry:
    """Pre-FSDP parameter identities and their exact optimizer allowlist."""

    all_parameters: tuple[_CapturedParameter, ...]
    allowed: tuple[CanonicalParameter, ...]

    @property
    def canonical_names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.allowed)


@dataclass(frozen=True)
class OptimizerParameterManifest:
    name: str
    module: str
    shape: tuple[int, ...]
    numel: int
    lr: float
    group: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "module": self.module,
            "shape": list(self.shape),
            "numel": self.numel,
            "lr": self.lr,
            "group": self.group,
        }


@dataclass(frozen=True)
class OptimizerManifest:
    parameters: tuple[OptimizerParameterManifest, ...]
    schema_version: int = 1

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        groups: dict[str, dict[str, Any]] = {}
        for entry in self.parameters:
            group = groups.setdefault(
                entry.group,
                {"lr": entry.lr, "tensors": 0, "numel": 0},
            )
            if float(group["lr"]) != entry.lr:
                raise Stage1ContractError(
                    f"manifest group {entry.group} has inconsistent LR"
                )
            group["tensors"] = int(group["tensors"]) + 1
            group["numel"] = int(group["numel"]) + entry.numel
        return {
            "schema_version": self.schema_version,
            "optimizer": "AdamW",
            "groups": groups,
            "parameters": [entry.to_dict() for entry in self.parameters],
        }

    def write_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage1ContractError(f"{label} must be a mapping")
    return value


def _nested(mapping: Mapping[str, Any], *parts: str) -> Any:
    value: Any = mapping
    for part in parts:
        if not isinstance(value, Mapping) or part not in value:
            dotted = ".".join(parts)
            raise Stage1ContractError(f"checkpoint cfg missing {dotted}")
        value = value[part]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_missing_allowlist(
    path: str | Path,
    *,
    checkpoint: Path,
) -> tuple[str, ...]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage1ContractError(
            f"invalid Stage1 native-key allowlist {source}: {exc}"
        ) from exc
    data = _mapping(payload, "native-key allowlist")
    if data.get("schema_version") != 1:
        raise Stage1ContractError(
            "native-key allowlist schema_version must be 1"
        )
    approved_path = Path(str(data.get("checkpoint", "")))
    if approved_path != checkpoint:
        raise Stage1ContractError(
            "native-key allowlist approved parent path mismatch"
        )
    approved_sha256 = str(data.get("checkpoint_sha256", ""))
    actual_sha256 = _sha256_file(checkpoint)
    if approved_sha256 != actual_sha256:
        raise Stage1ContractError(
            "native-key allowlist approved parent SHA256 mismatch"
        )
    if data.get("owner") != "model":
        raise Stage1ContractError(
            "native-key allowlist owner must be model"
        )
    if source.resolve() == DEFAULT_NATIVE_KEY_ALLOWLIST.resolve():
        if approved_path != STAGE0_PARENT_CHECKPOINT:
            raise Stage1ContractError(
                "checked-in allowlist approved parent path was tampered"
            )
        if approved_sha256 != STAGE0_PARENT_SHA256:
            raise Stage1ContractError(
                "checked-in allowlist approved parent SHA256 was tampered"
            )
    raw_keys = data.get("allowed_missing_keys")
    if not isinstance(raw_keys, list) or not all(
        isinstance(key, str) and key for key in raw_keys
    ):
        raise Stage1ContractError(
            "native-key allowlist allowed_missing_keys must be strings"
        )
    if len(raw_keys) != len(set(raw_keys)):
        raise Stage1ContractError(
            "native-key allowlist contains duplicate missing keys"
        )
    return tuple(sorted(raw_keys))


def _validate_parent_metadata(payload: Mapping[str, Any]) -> None:
    if payload.get("step") != 40000:
        raise Stage1ContractError(
            f"Stage0 parent must have step=40000, got {payload.get('step')!r}"
        )
    cfg = _mapping(payload.get("cfg"), "checkpoint cfg")
    expected = {
        ("data", "T"): 16,
        ("data", "k"): 8,
        ("data", "stride"): 4,
        ("model", "state", "T"): 16,
        ("model", "state", "k"): 8,
        ("model", "action", "T"): 16,
        ("model", "action", "k"): 8,
    }
    errors = []
    for path, wanted in expected.items():
        actual = _nested(cfg, *path)
        if actual != wanted:
            errors.append(
                f"{'.'.join(path)} must be {wanted}, got {actual!r}"
            )
    if errors:
        raise Stage1ContractError(
            "Stage0 parent temporal contract mismatch: " + "; ".join(errors)
        )


def load_stage0_model_mmap(
    checkpoint_path: str | Path,
    *,
    model: nn.Module,
    allowlist_path: str | Path = DEFAULT_NATIVE_KEY_ALLOWLIST,
    evaluation: bool = False,
) -> LoadReport:
    """Load only the trusted Stage0 model owner after exact validation."""

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Stage0 parent checkpoint not found: {checkpoint}")
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    data = _mapping(payload, "Stage0 checkpoint")
    _validate_parent_metadata(data)
    if "model" not in data:
        raise Stage1ContractError(
            "Stage0 checkpoint must expose native state under model owner"
        )
    state = _mapping(data["model"], "Stage0 checkpoint model")
    if not all(isinstance(key, str) for key in state):
        raise Stage1ContractError("Stage0 model state keys must be strings")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise Stage1ContractError("Stage0 model state values must be tensors")

    checkpoint_keys = set(state)
    owner_keys = sorted(
        key
        for key in checkpoint_keys
        if key.startswith(("module.", "model.", "wm_model."))
    )
    if owner_keys:
        raise Stage1ContractError(
            "Stage0 model key owner is not canonical: "
            + ", ".join(owner_keys[:8])
        )
    stage1_keys = sorted(
        key
        for key in checkpoint_keys
        if key.startswith(
            (
                "wan.",
                "wan_",
                "transformer.",
                "wan_transformer.",
                "wan_control_adapter.",
            )
        )
    )
    if stage1_keys:
        raise Stage1ContractError(
            "Stage0 parent unexpectedly owns Stage1 Wan state: "
            + ", ".join(stage1_keys[:8])
        )

    target_state = model.state_dict()
    target_keys = set(target_state)
    unexpected = tuple(sorted(checkpoint_keys - target_keys))
    if unexpected:
        raise Stage1ContractError(
            "Stage0 model has unexpected native keys: "
            + ", ".join(unexpected[:12])
        )

    shape_mismatches = []
    for key in sorted(checkpoint_keys & target_keys):
        source_shape = tuple(int(size) for size in state[key].shape)
        target_shape = tuple(int(size) for size in target_state[key].shape)
        if source_shape != target_shape:
            shape_mismatches.append(
                f"{key}: checkpoint={source_shape} target={target_shape}"
            )
    if shape_mismatches:
        raise Stage1ContractError(
            "Stage0 model shape mismatch: " + "; ".join(shape_mismatches[:12])
        )

    missing = tuple(sorted(target_keys - checkpoint_keys))
    allowed_missing = _read_missing_allowlist(
        allowlist_path,
        checkpoint=checkpoint,
    )
    unknown_allowed = tuple(sorted(set(allowed_missing) - target_keys))
    if unknown_allowed:
        raise Stage1ContractError(
            "native-key allowlist names keys absent from target model: "
            + ", ".join(unknown_allowed[:12])
        )
    if evaluation and missing:
        raise Stage1ContractError(
            "evaluation loading requires zero missing keys, got "
            + ", ".join(missing[:12])
        )
    if not evaluation and missing != allowed_missing:
        raise Stage1ContractError(
            "Stage0 model missing keys do not exactly match checked-in "
            f"allowlist: missing={list(missing)} "
            f"allowlist={list(allowed_missing)}"
        )

    incompatible = model.load_state_dict(dict(state), strict=False)
    actual_missing = tuple(sorted(incompatible.missing_keys))
    actual_unexpected = tuple(sorted(incompatible.unexpected_keys))
    if actual_missing != missing or actual_unexpected:
        raise Stage1ContractError(
            "model loader report diverged from validated key contract: "
            f"missing={actual_missing} unexpected={actual_unexpected}"
        )
    loaded = tuple(sorted(checkpoint_keys))
    report = LoadReport(
        checkpoint=str(checkpoint),
        step=40000,
        loaded_keys=loaded,
        missing_keys=missing,
    )
    del payload
    return report


def _named_parameters_with_duplicates(
    module: nn.Module,
) -> list[tuple[str, nn.Parameter]]:
    try:
        return list(module.named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover
        return list(module.named_parameters())


def _module_parameters(
    module: Any,
    canonical_module: str,
) -> list[tuple[str, nn.Parameter]]:
    if not isinstance(module, nn.Module):
        raise Stage1ContractError(
            f"required optimizer module {canonical_module} is missing"
        )
    parameters = _named_parameters_with_duplicates(module)
    if not parameters:
        raise Stage1ContractError(
            f"required optimizer module {canonical_module} must be nonempty"
        )
    return parameters


def _canonical_entries(
    module: nn.Module,
    *,
    canonical_module: str,
    group: str,
    lr: float,
) -> list[CanonicalParameter]:
    entries = []
    for relative_name, parameter in _module_parameters(
        module, canonical_module
    ):
        name = (
            f"{canonical_module}.{relative_name}"
            if relative_name
            else canonical_module
        )
        entries.append(
            CanonicalParameter(
                name=name,
                module=canonical_module,
                shape=tuple(int(size) for size in parameter.shape),
                numel=int(parameter.numel()),
                group=group,
                lr=lr,
                parameter=parameter,
            )
        )
    return entries


def _all_root_parameters(
    roots: Sequence[tuple[str, nn.Module]],
) -> tuple[_CapturedParameter, ...]:
    entries: list[_CapturedParameter] = []
    names_by_id: dict[int, str] = {}
    for root_name, root in roots:
        if not isinstance(root, nn.Module):
            raise Stage1ContractError(f"{root_name} root must be an nn.Module")
        for relative_name, parameter in _named_parameters_with_duplicates(root):
            name = (
                f"{root_name}.{relative_name}"
                if relative_name
                else root_name
            )
            existing = names_by_id.get(id(parameter))
            if existing is not None:
                raise Stage1ContractError(
                    "duplicate parameter object has multiple canonical names: "
                    f"{existing}, {name}"
                )
            names_by_id[id(parameter)] = name
            entries.append(
                _CapturedParameter(
                    name=name,
                    shape=tuple(int(size) for size in parameter.shape),
                    numel=int(parameter.numel()),
                    parameter=parameter,
                )
            )
    return tuple(entries)


def capture_canonical_param_registry(
    wm_model: nn.Module,
    wan_control_adapter: nn.Module,
    wan_transformer: nn.Module,
) -> CanonicalParamRegistry:
    """Capture the exact pre-FSDP parameter-object allowlist."""

    for name in _FORBIDDEN_WM_MODULES:
        if isinstance(getattr(wm_model, name, None), nn.Module):
            raise Stage1ContractError(
                f"forbidden Stage1 module wm.{name} is instantiated"
            )

    blocks = getattr(wan_transformer, "blocks", None)
    if not isinstance(blocks, (nn.ModuleList, nn.Sequential, list, tuple)):
        raise Stage1ContractError(
            "wan_transformer.blocks must be an indexed module collection"
        )
    if len(blocks) <= max(_ALLOWED_WAN_BLOCKS):
        raise Stage1ContractError(
            "wan_transformer does not contain required blocks 28 and 29"
        )

    allowed: list[CanonicalParameter] = []
    for name in ("dual", "action_proj", "geom"):
        allowed.extend(
            _canonical_entries(
                getattr(wm_model, name, None),
                canonical_module=f"wm.{name}",
                group="native",
                lr=_NATIVE_LR,
            )
        )
    allowed.extend(
        _canonical_entries(
            getattr(wm_model, "context_pixel", None),
            canonical_module="wm.context_pixel",
            group="context_pixel",
            lr=_CONTEXT_PIXEL_LR,
        )
    )
    allowed.extend(
        _canonical_entries(
            wan_control_adapter,
            canonical_module="wan_control_adapter",
            group="wan_control_adapter",
            lr=_WAN_CONTROL_LR,
        )
    )
    for index in _ALLOWED_WAN_BLOCKS:
        allowed.extend(
            _canonical_entries(
                blocks[index],
                canonical_module=f"wan_transformer.blocks.{index}",
                group="wan_transformer",
                lr=_WAN_TRANSFORMER_LR,
            )
        )
    allowed.extend(
        _canonical_entries(
            getattr(wan_transformer, "head", None),
            canonical_module="wan_transformer.head",
            group="wan_transformer",
            lr=_WAN_TRANSFORMER_LR,
        )
    )

    roots = (
        ("wm", wm_model),
        ("wan_control_adapter", wan_control_adapter),
        ("wan_transformer", wan_transformer),
    )
    all_parameters = _all_root_parameters(roots)
    all_by_id = {id(entry.parameter): entry for entry in all_parameters}
    allowed_by_id: dict[int, CanonicalParameter] = {}
    for entry in allowed:
        if entry.parameter_id in allowed_by_id:
            other = allowed_by_id[entry.parameter_id]
            raise Stage1ContractError(
                "duplicate parameter object in optimizer allowlist: "
                f"{other.name}, {entry.name}"
            )
        if entry.parameter_id not in all_by_id:
            raise Stage1ContractError(
                f"optimizer parameter {entry.name} is absent from its root"
            )
        if not entry.parameter.requires_grad:
            raise Stage1ContractError(
                f"allowed optimizer parameter {entry.name} is frozen"
            )
        allowed_by_id[entry.parameter_id] = entry

    forbidden_trainable = [
        entry.name
        for entry in all_parameters
        if entry.parameter.requires_grad
        and id(entry.parameter) not in allowed_by_id
    ]
    if forbidden_trainable:
        raise Stage1ContractError(
            "forbidden trainable parameter object(s): "
            + ", ".join(forbidden_trainable[:16])
        )
    return CanonicalParamRegistry(
        all_parameters=all_parameters,
        allowed=tuple(sorted(allowed, key=lambda entry: entry.name)),
    )


def _contains_fsdp(module: nn.Module) -> bool:
    return any(
        child.__class__.__name__ == "FullyShardedDataParallel"
        for child in module.modules()
    )


def _verify_post_wrap_objects(
    registry: CanonicalParamRegistry,
    roots: Sequence[tuple[str, nn.Module]],
) -> None:
    current: dict[int, nn.Parameter] = {}
    duplicate_roots: list[int] = []
    for _, root in roots:
        for parameter in root.parameters():
            parameter_id = id(parameter)
            if parameter_id in current:
                duplicate_roots.append(parameter_id)
            current[parameter_id] = parameter
    if duplicate_roots:
        raise Stage1ContractError(
            "post-wrap roots contain duplicate parameter object ownership"
        )

    captured = {
        id(entry.parameter): entry for entry in registry.all_parameters
    }
    missing_ids = set(captured) - set(current)
    extra_ids = set(current) - set(captured)
    if missing_ids or extra_ids:
        missing_names = [captured[item].name for item in sorted(missing_ids)]
        raise Stage1ContractError(
            "FSDP post-wrap parameter object identity mismatch: "
            f"missing={missing_names[:12]} extra_count={len(extra_ids)}"
        )

    fsdp_present = any(_contains_fsdp(root) for _, root in roots)
    if not fsdp_present:
        shape_errors = []
        for parameter_id, entry in captured.items():
            current_shape = tuple(
                int(size) for size in current[parameter_id].shape
            )
            if current_shape != entry.shape:
                shape_errors.append(
                    f"{entry.name}: captured={entry.shape} "
                    f"current={current_shape}"
                )
        if shape_errors:
            raise Stage1ContractError(
                "post-wrap parameter shape mismatch: "
                + "; ".join(shape_errors[:12])
            )


def build_stage1_optimizer(
    registry: CanonicalParamRegistry,
    *,
    wm_model: nn.Module,
    wan_control_adapter: nn.Module,
    wan_transformer: nn.Module,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.95),
    manifest_path: str | Path | None = None,
) -> tuple[torch.optim.AdamW, OptimizerManifest]:
    """Verify post-FSDP identities and build the exact AdamW groups."""

    roots = (
        ("wm", wm_model),
        ("wan_control_adapter", wan_control_adapter),
        ("wan_transformer", wan_transformer),
    )
    _verify_post_wrap_objects(registry, roots)

    allowed_ids = {entry.parameter_id for entry in registry.allowed}
    current_trainable = {
        id(parameter)
        for _, root in roots
        for parameter in root.parameters()
        if parameter.requires_grad
    }
    if current_trainable != allowed_ids:
        missing = [
            entry.name
            for entry in registry.allowed
            if entry.parameter_id not in current_trainable
        ]
        extra = len(current_trainable - allowed_ids)
        raise Stage1ContractError(
            "post-wrap trainable object allowlist mismatch: "
            f"missing={missing[:12]} extra_count={extra}"
        )

    group_order = (
        "native",
        "context_pixel",
        "wan_control_adapter",
        "wan_transformer",
    )
    groups = []
    optimizer_ids: list[int] = []
    for group_name in group_order:
        entries = [
            entry
            for entry in registry.allowed
            if entry.group == group_name
        ]
        if not entries:
            raise Stage1ContractError(
                f"optimizer group {group_name} must be nonempty"
            )
        lrs = {entry.lr for entry in entries}
        if len(lrs) != 1:
            raise Stage1ContractError(
                f"optimizer group {group_name} has inconsistent LR"
            )
        parameters = [entry.parameter for entry in entries]
        optimizer_ids.extend(id(parameter) for parameter in parameters)
        groups.append(
            {
                "name": group_name,
                "params": parameters,
                "lr": lrs.pop(),
                "weight_decay": float(weight_decay),
            }
        )
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise Stage1ContractError(
            "optimizer contains duplicate parameter object"
        )
    if set(optimizer_ids) != allowed_ids:
        raise Stage1ContractError(
            "optimizer parameters are not the exact final allowlist"
        )

    optimizer = torch.optim.AdamW(groups, betas=tuple(betas))
    actual_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != allowed_ids:
        raise Stage1ContractError(
            "constructed optimizer changed the exact parameter allowlist"
        )

    manifest = OptimizerManifest(
        parameters=tuple(
            OptimizerParameterManifest(
                name=entry.name,
                module=entry.module,
                shape=entry.shape,
                numel=entry.numel,
                lr=entry.lr,
                group=entry.group,
            )
            for entry in registry.allowed
        )
    )
    if manifest_path is not None:
        manifest.write_json(manifest_path)
    return optimizer, manifest


def is_stage1_launch(
    stage: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> bool:
    stage_flow = cfg.get("stage_flow", {})
    out = cfg.get("out", {})
    markers = (
        int(stage.get("index", -1)) == 1,
        cfg.get("semantic_recipe") == STAGE1_SEMANTIC_RECIPE,
        isinstance(stage_flow, Mapping)
        and int(stage_flow.get("stage", -1)) == 1,
        "stage1_rgb_world_refine"
        in str(stage.get("name", "")).lower(),
        "stage1_rgb_world_refine"
        in str(stage.get("run_root", "")).lower(),
        isinstance(out, Mapping)
        and "stage1_rgb_world_refine"
        in str(out.get("root", "")).lower(),
    )
    return any(markers)


def validate_stage1_launch(
    stage: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> None:
    if not is_stage1_launch(stage, cfg):
        return
    stage_flow = _mapping(cfg.get("stage_flow"), "Stage1 stage_flow")
    errors = []
    expected = STAGE1_SEMANTIC_RECIPE
    if int(stage.get("index", -1)) != 1:
        errors.append("Stage1 launch index must be 1")
    if cfg.get("semantic_recipe") != expected:
        errors.append("Stage1 launch requires the approved semantic recipe")
    if stage.get("recipe") != expected or stage_flow.get("recipe") != expected:
        errors.append("Stage1 launch requires the dedicated recipe")
    if (
        stage.get("trainer") != STAGE1_SEMANTIC_TRAINER
        or stage_flow.get("trainer") != STAGE1_SEMANTIC_TRAINER
    ):
        errors.append("Stage1 launch requires the dedicated trainer")
    if int(stage_flow.get("stage", -1)) != 1:
        errors.append("Stage1 config stage_flow.stage must be 1")
    if errors:
        raise Stage1ContractError(
            "Stage1 launch contract failed: " + "; ".join(errors)
        )


def validate_stage1_fsdp_wrap_report(
    report: Mapping[str, Any],
) -> None:
    data = _mapping(report, "Stage1 FSDP wrap report")
    errors = []
    modules = set(data.get("modules") or ())
    wrapped = set(data.get("wrapped_roots") or ())
    coverage = data.get("root_coverage")
    if data.get("enabled") is not True:
        errors.append("FSDP must be enabled")
    if data.get("use_orig_params") is not True:
        errors.append("FSDP use_orig_params must be true")
    if modules != STAGE1_FSDP_ROOTS:
        errors.append("FSDP modules are not the exact roots")
    if wrapped != STAGE1_FSDP_ROOTS:
        errors.append("FSDP wrapped roots are incomplete")
    if not isinstance(coverage, Mapping):
        errors.append("FSDP root coverage is missing")
    else:
        if set(coverage) != STAGE1_FSDP_ROOTS:
            errors.append("FSDP root coverage keys are not exact")
        for root in STAGE1_FSDP_ROOTS:
            entry = coverage.get(root)
            if not isinstance(entry, Mapping):
                errors.append(f"FSDP root {root} coverage is missing")
                continue
            total = int(entry.get("trainable_tensors", 0) or 0)
            covered = int(
                entry.get("covered_trainable_tensors", 0) or 0
            )
            if total <= 0 or covered != total:
                errors.append(f"FSDP root {root} is not fully covered")
    if errors:
        raise Stage1ContractError(
            "Stage1 FSDP wrap report failed: " + "; ".join(errors)
        )


def _is_nonzero(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    try:
        return abs(float(value)) > 1.0e-12
    except (TypeError, ValueError):
        return bool(value)


def _canonical_wan_pattern(pattern: Any) -> str | None:
    value = str(pattern).strip()
    plain = value.replace("\\.", ".")
    if plain.startswith("^"):
        plain = plain[1:]
    for suffix in (".*$", ".*", "$"):
        if plain.endswith(suffix):
            plain = plain[: -len(suffix)]
            break
    if plain in {"blocks.28.", "blocks.29.", "head."}:
        return plain
    return None


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_stage1_recipe(
    cfg: Mapping[str, Any],
    *,
    run_root: str | Path | None = None,
    output_paths: Sequence[str | Path] = (),
    topology: Mapping[str, Any] | None = None,
) -> None:
    """Raise on any forbidden or non-exact Stage1 recipe setting."""

    config = _mapping(cfg, "Stage1 recipe")
    data = _mapping(config.get("data"), "Stage1 data")
    model = _mapping(config.get("model"), "Stage1 model")
    train = _mapping(config.get("train"), "Stage1 train")
    out = _mapping(config.get("out"), "Stage1 out")
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    stage_flow = _mapping(config.get("stage_flow"), "Stage1 stage_flow")
    require(
        config.get("semantic_recipe") == STAGE1_SEMANTIC_RECIPE,
        "semantic recipe must be stage1_rgb_world_refine",
    )
    require(
        int(stage_flow.get("stage", -1)) == 1,
        "Stage1 stage_flow.stage must be 1",
    )
    require(
        stage_flow.get("recipe") == STAGE1_SEMANTIC_RECIPE,
        "Stage1 recipe must be the approved semantic recipe",
    )
    require(
        stage_flow.get("trainer") == STAGE1_SEMANTIC_TRAINER,
        "Stage1 trainer must be the dedicated trainer",
    )

    disable_sections = (config, train, config.get("runtime", {}))
    for section in disable_sections:
        if not isinstance(section, Mapping):
            continue
        if section.get("disable_stage1_validator") is True:
            errors.append("Stage1 validator cannot be disabled")
        if section.get("skip_stage1_validator") is True:
            errors.append("Stage1 validator cannot be disabled")
        if section.get("stage1_validator_enabled") is False:
            errors.append("Stage1 validator cannot be disabled")

    temporal_values = {
        "data.T": (data.get("T"), 16),
        "data.k": (data.get("k"), 8),
        "data.stride": (data.get("stride"), 4),
    }
    for branch in ("state", "action"):
        branch_cfg = model.get(branch)
        if not isinstance(branch_cfg, Mapping):
            errors.append(f"model.{branch} must be configured")
            continue
        temporal_values[f"model.{branch}.T"] = (branch_cfg.get("T"), 16)
        temporal_values[f"model.{branch}.k"] = (branch_cfg.get("k"), 8)
    for label, (actual, expected) in temporal_values.items():
        require(actual == expected, f"{label} must be {expected}, got {actual!r}")

    require(
        train.get("wan_frame_num") == 9,
        "train.wan_frame_num must be 9",
    )
    require(
        train.get("wan_condition_latent_frames") == 1,
        "train.wan_condition_latent_frames must be 1",
    )
    require(train.get("fsdp_enabled") is True, "Stage1 requires FSDP")
    require(
        train.get("fsdp_use_orig_params") is True,
        "train.fsdp_use_orig_params must be true",
    )
    require(
        set(train.get("fsdp_modules") or ()) == STAGE1_FSDP_ROOTS
        and len(train.get("fsdp_modules") or ()) == len(STAGE1_FSDP_ROOTS),
        "Stage1 FSDP roots must be exactly wm, wan_transformer, "
        "wan_control_adapter",
    )

    require(
        train.get("wan_wm_source_feed_predicted_to_dit") is False,
        "predicted-source feed is forbidden",
    )
    source_weight_keys = sorted(
        key
        for key in train
        if isinstance(key, str)
        and "source" in key
        and (key.endswith("_weight") or key == "wan_wm_source_scale")
    )
    for key in source_weight_keys:
        require(
            not _is_nonzero(train.get(key)),
            f"source loss/source-CF setting {key} must be zero",
        )
    require(
        not _is_nonzero(train.get("wan_action_cf_weight")),
        "full-path CF weight must be zero",
    )
    require(
        train.get("wan_action_cf_train_full_wm_path") is not True,
        "full-path CF is forbidden",
    )
    require(
        not _is_nonzero(train.get("wan_adapter_action_cf_weight")),
        "adapter-CF weight must be zero",
    )
    require(
        not _is_nonzero(train.get("wan_zero_action_hold_weight")),
        "zero-hold weight must be zero",
    )

    for name in _FORBIDDEN_WM_MODULES:
        require(
            model.get(f"enable_{name}") is False,
            f"model.enable_{name} must be false",
        )

    sampler_values = [
        data.get("sampler"),
        data.get("sampler_type"),
        train.get("sampler"),
        train.get("sampler_type"),
    ]
    require(
        not any(
            "weight" in str(value).lower()
            for value in sampler_values
            if value is not None
        ),
        "weighted sampler is forbidden; use hierarchical sampling",
    )
    for section in (data, train):
        for key in (
            "weighted_sampler",
            "use_weighted_sampler",
            "repeat_weight_sampling",
            "use_repeat_weight",
        ):
            require(
                not _is_nonzero(section.get(key)),
                f"weighted sampler setting {key} is forbidden",
            )
    for key in (
        "max_windows_per_episode",
        "max_train_windows",
        "max_val_windows",
    ):
        require(
            data.get(key) in (None, 0),
            f"data.{key} must be full/unset, got {data.get(key)!r}",
        )

    max_steps = train.get("max_steps")
    require(
        isinstance(max_steps, int) and 0 < max_steps <= 10000,
        "train.max_steps must be in [1, 10000]",
    )
    require(
        Path(str(train.get("init_from_core_rgb_ckpt", "")))
        == STAGE0_PARENT_CHECKPOINT,
        "Stage0 parent checkpoint is not the immutable selected parent",
    )
    require(
        not train.get("init_from_stage0_wan_ckpt"),
        "Stage1 must load the model-only Stage0 parent, not joint Wan state",
    )

    require(
        train.get("wan_dit_train_lora") is False,
        "Stage1 forbids Wan LoRA; only exact module objects are trainable",
    )
    raw_patterns = train.get("wan_dit_trainable_patterns")
    normalized: list[str | None] = []
    if isinstance(raw_patterns, Sequence) and not isinstance(
        raw_patterns, (str, bytes)
    ):
        normalized = [_canonical_wan_pattern(item) for item in raw_patterns]
    require(
        len(normalized) == 3
        and None not in normalized
        and len(set(normalized)) == 3
        and set(normalized)
        == {"blocks.28.", "blocks.29.", "head."},
        "wan_dit_trainable_patterns must select exactly blocks 28/29 and head",
    )
    require(
        not train.get("wan_dit_trainable_exclude"),
        "wan_dit_trainable_exclude must be empty",
    )

    expected_lrs = {
        "wm_lr": _NATIVE_LR,
        "context_pixel_lr": _CONTEXT_PIXEL_LR,
        "wan_control_lr": _WAN_CONTROL_LR,
        "wan_base_lr": _WAN_TRANSFORMER_LR,
    }
    for key, expected in expected_lrs.items():
        try:
            actual = float(train.get(key))
        except (TypeError, ValueError):
            actual = float("nan")
        require(
            actual == expected,
            f"train.{key} must be exactly {expected:.1e}",
        )

    runtime = _mapping(config.get("runtime"), "Stage1 runtime")
    evaluation = _mapping(config.get("evaluation"), "Stage1 evaluation")
    is_resume_smoke = int(train.get("max_steps", 0) or 0) == 104
    if is_resume_smoke:
        resume_test = _mapping(runtime.get("resume_test"), "Stage1 resume smoke")
        cadence_ok = (
            runtime.get("phase_length") == 32
            and runtime.get("rolling_start_update") == 32
            and tuple(runtime.get("expected_rolling_updates") or ()) == (32, 64, 96)
            and tuple(runtime.get("resume_updates") or ()) == (96,)
            and tuple(evaluation.get("gate_updates") or ()) == ()
            and resume_test.get("save_after_update") == 96
            and resume_test.get("restart_next_update") == 97
            and resume_test.get("complete_through_update") == 104
        )
    else:
        cadence_ok = (
            runtime.get("phase_length") == 32
            and runtime.get("rolling_start_update") == 512
            and runtime.get("expected_rolling_updates") == "formal_cadence"
            and tuple(runtime.get("resume_updates") or ()) == STAGE1_RESUME_UPDATES
            and tuple(evaluation.get("gate_updates") or ()) == STAGE1_RESUME_UPDATES
        )
    require(cadence_ok, "Stage1 resume/evaluation cadence binding is not exact")

    layout: Mapping[str, Any] | None = topology
    if layout is None:
        flow_cfg = config.get("flow")
        if isinstance(flow_cfg, Mapping):
            candidate = flow_cfg.get("node_layout")
            if isinstance(candidate, Mapping):
                layout = candidate
    if layout is not None:
        require(
            layout.get("node42_allowed") is not True,
            "node42 topology is forbidden",
        )
        nodes = layout.get("nodes", ())
        if isinstance(nodes, Sequence):
            for node in nodes:
                if not isinstance(node, Mapping):
                    continue
                identifiers = {
                    str(node.get(key, "")).strip().lower()
                    for key in ("name", "ip", "host", "hostname")
                }
                require(
                    "node42" not in identifiers
                    and "172.27.0.5" not in identifiers,
                    "node42 topology is forbidden",
                )

    configured_root = Path(str(out.get("root", ""))).resolve()
    effective_root = (
        Path(run_root).resolve()
        if run_root is not None
        else configured_root
    )
    require(
        bool(str(out.get("root", "")).strip()),
        "out.root must be configured",
    )
    require(
        _path_within(configured_root, effective_root),
        f"out.root is outside run root {effective_root}",
    )
    paths_to_check = list(output_paths)
    for key, value in out.items():
        if key == "root" or value in (None, ""):
            continue
        if not (
            key.endswith("_dir")
            or key.endswith("_path")
            or key.endswith("_root")
        ):
            continue
        path = Path(str(value))
        paths_to_check.append(
            path if path.is_absolute() else configured_root / path
        )
    for value in paths_to_check:
        path = Path(value).resolve()
        require(
            _path_within(path, effective_root),
            f"output path {path} is outside run root {effective_root}",
        )

    if errors:
        raise Stage1ContractError(
            "Stage1 recipe validation failed: " + "; ".join(errors)
        )
