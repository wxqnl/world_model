"""Read the sealed formal WM3D cache through the current Stage0 tensor ABI.

The 1.3 TB formal cache predates the episode-cache container used by the
current trainer.  Its observations, actions, timestamps and geometry are
already sealed, so rebuilding it would only duplicate storage.  This module is
the single compatibility boundary: it invokes the pinned, read-only cache
reader and maps its tensors into the current grouped-robot/native-3D ABI.  It
does not import or execute the former model, objective, sampler, or trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from wm3d.data.grouped_robot import (
    ACTION_SEMANTIC_IDS,
    COMPOSITION_OPERATOR_IDS,
    STATE_SEMANTIC_IDS,
)
from wm3d.data.manifest_contract import SHA256_RE, sha256_file


FORMAL_CACHE_CLOSURE_SCHEMA = "wm3d_formal_cache_closure_v1"
FORMAL_CACHE_RECEIPT_SCHEMA = "wm3d_formal_cache_receipt_v1"

_SOURCE_ORDER = (
    "oxe_droid_action",
    "oxe_bridge_action",
    "robocasa_atomic",
    "robocasa_composite",
    "robocasa_mg",
)
_SOURCE_WEIGHTS = {
    "oxe_droid_action": 35,
    "oxe_bridge_action": 15,
    "robocasa_atomic": 10,
    "robocasa_composite": 20,
    "robocasa_mg": 20,
}
_EMBODIMENT_IDS = {
    "oxe_droid_action": 1,
    "oxe_bridge_action": 2,
    "robocasa_atomic": 3,
    "robocasa_composite": 3,
    "robocasa_mg": 3,
}
_GIT_OBJECT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


class FormalCacheError(RuntimeError):
    pass


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _regular(path_value: object, *, field: str) -> Path:
    path = Path(str(path_value))
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise FormalCacheError(f"{field} must be an absolute regular non-symlink file")
    return path.resolve(strict=True)


def _sha(value: object, *, field: str) -> str:
    text = str(value)
    if SHA256_RE.fullmatch(text) is None:
        raise FormalCacheError(f"{field} must be a lowercase SHA256")
    return text


def _git_object_id(value: object, *, field: str) -> str:
    text = str(value)
    if _GIT_OBJECT_RE.fullmatch(text) is None:
        raise FormalCacheError(f"{field} must be a lowercase Git object ID")
    return text


def validate_formal_cache_closure(
    closure: Mapping[str, Any],
) -> dict[str, Any]:
    required = {"schema", "name", "cache_root", "receipt_path", "receipt_sha256"}
    if set(closure) != required or closure.get("schema") != FORMAL_CACHE_CLOSURE_SCHEMA:
        raise FormalCacheError("formal cache closure fields/schema mismatch")
    cache_root = Path(str(closure["cache_root"]))
    if not cache_root.is_absolute() or cache_root.is_symlink() or not cache_root.is_dir():
        raise FormalCacheError("formal cache root must be an absolute non-symlink directory")
    receipt_path = _regular(closure["receipt_path"], field="receipt_path")
    expected_receipt_sha = _sha(closure["receipt_sha256"], field="receipt_sha256")
    if sha256_file(receipt_path) != expected_receipt_sha:
        raise FormalCacheError("formal cache receipt SHA mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required_receipt = {
        "schema",
        "passed",
        "cache_root",
        "legacy_reader_root",
        "legacy_reader_commit",
        "legacy_reader_tree",
        "legacy_runtime_config_path",
        "legacy_runtime_config_sha256",
        "closure_report_path",
        "closure_report_sha256",
        "token_codec_path",
        "token_codec_sha256",
        "source_order",
        "source_weights",
        "source_lengths_by_split",
        "cache_representation",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != required_receipt
        or receipt.get("schema") != FORMAL_CACHE_RECEIPT_SCHEMA
        or receipt.get("passed") is not True
    ):
        raise FormalCacheError("formal cache receipt fields/schema/pass gate mismatch")
    if Path(str(receipt["cache_root"])).resolve(strict=True) != cache_root.resolve(strict=True):
        raise FormalCacheError("formal cache receipt root mismatch")
    reader = Path(str(receipt["legacy_reader_root"]))
    if not reader.is_absolute() or reader.is_symlink() or not reader.is_dir():
        raise FormalCacheError("legacy reader root must be an absolute clean checkout")
    commit = _git_object_id(
        receipt["legacy_reader_commit"], field="legacy_reader_commit"
    )
    tree = _git_object_id(receipt["legacy_reader_tree"], field="legacy_reader_tree")
    if _git(reader, "status", "--porcelain"):
        raise FormalCacheError("legacy cache reader checkout is dirty")
    if _git(reader, "rev-parse", "HEAD") != commit or _git(reader, "rev-parse", "HEAD^{tree}") != tree:
        raise FormalCacheError("legacy cache reader commit/tree mismatch")
    for path_field, sha_field in (
        ("legacy_runtime_config_path", "legacy_runtime_config_sha256"),
        ("closure_report_path", "closure_report_sha256"),
        ("token_codec_path", "token_codec_sha256"),
    ):
        path = _regular(receipt[path_field], field=path_field)
        if sha256_file(path) != _sha(receipt[sha_field], field=sha_field):
            raise FormalCacheError(f"{path_field} SHA mismatch")
    if tuple(receipt["source_order"]) != _SOURCE_ORDER:
        raise FormalCacheError("formal source order mismatch")
    if receipt["source_weights"] != _SOURCE_WEIGHTS:
        raise FormalCacheError("formal source weights mismatch")
    lengths = receipt["source_lengths_by_split"]
    if (
        not isinstance(lengths, dict)
        or set(lengths) != {"train", "val"}
        or any(set(lengths[split]) != set(_SOURCE_ORDER) for split in lengths)
        or any(
            isinstance(count, bool) or int(count) <= 0
            for split in lengths.values()
            for count in split.values()
        )
    ):
        raise FormalCacheError("formal source length closure is invalid")
    expected_representation = {
        "spatial_tokens": 64,
        "token_grid": 8,
        "stored_token_dim": 384,
        "token_dim": 2048,
        "num_views": 3,
        "rgb_size": 256,
    }
    if receipt["cache_representation"] != expected_representation:
        raise FormalCacheError("formal cache representation mismatch")
    report = json.loads(
        Path(str(receipt["closure_report_path"])).read_text(encoding="utf-8")
    )
    if report.get("pass") is not True:
        raise FormalCacheError("formal cache closure report did not pass")
    return receipt


@dataclass(frozen=True)
class FormalSource:
    name: str


@dataclass
class FormalCacheProfile:
    source_order: tuple[str, ...]
    source_weights: dict[str, int]
    cache_representation: dict[str, int]
    sources: tuple[FormalSource, ...]
    declared_eval_coverage_lanes: frozenset[str]
    _train: Any
    _val: Any


class FormalCacheDataset(Dataset[dict[str, torch.Tensor]]):
    """Map one sealed formal-cache row into the current native model ABI."""

    def __init__(
        self,
        legacy: Any,
        profile: FormalCacheProfile,
        model_profile: Mapping[str, Any],
        *,
        split: str,
        codec: Mapping[str, torch.Tensor],
    ) -> None:
        self.legacy = legacy
        self.profile = profile
        self.model = model_profile["model"]
        self.split = str(split)
        self.source_names = tuple(str(name) for name in legacy.source_names)
        self.source_spans = {
            str(name): (int(span[0]), int(span[1]))
            for name, span in legacy.source_spans.items()
        }
        if self.source_names != _SOURCE_ORDER or set(self.source_spans) != set(_SOURCE_ORDER):
            raise FormalCacheError("legacy reader source closure drifted")
        self.mean = codec["mean"].float().contiguous()
        self.components = codec["components"].float().contiguous()
        if tuple(self.mean.shape) != (2048,) or tuple(self.components.shape) != (384, 2048):
            raise FormalCacheError("formal PCA token codec tensor shapes mismatch")
        if bool(self.model.get("appearance_enabled", False)):
            raise FormalCacheError(
                "dual-path appearance training requires the raw per-view token cache"
            )
        required_model = {
            "T": 16,
            "P": 64,
            "K": 8,
            "token_dim": 2048,
            "task_dim": 2048,
            "num_views": 3,
            "max_action_groups": 8,
            "max_action_dim": 16,
            "max_state_dim": 32,
            "rgb_size": 256,
        }
        for field, expected in required_model.items():
            if int(self.model[field]) != expected:
                raise FormalCacheError(
                    f"formal cache requires model.{field}={expected}, got {self.model[field]}"
                )
        self.rgb_indices = tuple(
            int(value) for value in self.model["rgb_decode_indices"]
        )
        if (
            not self.rgb_indices
            or tuple(sorted(set(self.rgb_indices))) != self.rgb_indices
            or any(index < 0 or index >= required_model["K"] for index in self.rgb_indices)
        ):
            raise FormalCacheError(
                "formal cache RGB indices must be unique increasing K-frame indices"
            )

    def __len__(self) -> int:
        return int(len(self.legacy))

    def _decode(self, value: torch.Tensor) -> torch.Tensor:
        if tuple(value.shape[-2:]) != (64, 384):
            raise FormalCacheError("formal token tensor must end in [64,384]")
        return torch.addmm(
            self.mean,
            value.float().reshape(-1, 384),
            self.components,
        ).reshape(*value.shape[:-1], 2048).to(torch.bfloat16)

    def _source(self, index: int) -> str:
        for name in self.source_names:
            start, stop = self.source_spans[name]
            if start <= index < stop:
                return name
        raise IndexError(index)

    @staticmethod
    def _close01(value: torch.Tensor) -> torch.Tensor:
        return ((value.float() + 1.0) * 0.5).clamp(0.0, 1.0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        index = int(index)
        source = self._source(index)
        old = self.legacy[index]
        if old.get("v8_action_contract_version") != "wm3d_v8_dual_rate_action_v1":
            raise FormalCacheError("formal row action contract mismatch")
        if old.get("v8_action_history_schema") != "wm3d_v8_action_history_20hz_dt_valid_v1":
            raise FormalCacheError("formal row action-history contract mismatch")

        T, K, G, S, A, C, V, P = 16, 8, 8, 4, 16, 8, 3, 64
        fine = torch.zeros(T, G, S, A)
        fine_mask = torch.zeros(T, G, S, A, dtype=torch.bool)
        fine_dt = torch.zeros(T, G, S)
        fine_sample = torch.zeros(T, G, S, dtype=torch.bool)
        coarse = torch.zeros(T, G, A)
        coarse_mask = torch.zeros(T, G, A, dtype=torch.bool)
        future_fine = torch.zeros(K, G, S, A)
        future_fine_mask = torch.zeros(K, G, S, A, dtype=torch.bool)
        future_fine_dt = torch.zeros(K, G, S)
        future_fine_sample = torch.zeros(K, G, S, dtype=torch.bool)
        future_coarse = torch.zeros(K, G, A)
        future_coarse_mask = torch.zeros(K, G, A, dtype=torch.bool)
        target_fine = torch.zeros(G, C, A)
        target_fine_mask = torch.zeros(G, C, A, dtype=torch.bool)
        target_coarse = torch.zeros(K, G, A)
        target_coarse_norm = torch.zeros(K, G, A)
        target_coarse_mask = torch.zeros(K, G, A, dtype=torch.bool)
        policy_dt = torch.zeros(G, C)
        policy_mask = torch.zeros(G, C, dtype=torch.bool)

        dynamics = old["v8_dynamics_action_cond"].float()
        if tuple(dynamics.shape) != (K, 36):
            raise FormalCacheError("formal dynamics action tensor must be [8,36]")
        values = dynamics[:, :28].reshape(K, S, 7)
        valid = dynamics[:, 28:32] > 0.5
        recorded_dt = dynamics[:, 32:36]
        history = old["action_history"].float()
        if tuple(history.shape) != (T, 9):
            raise FormalCacheError("formal action history must be [16,9]")
        history_valid = history[:, 8] > 0.5
        fine_source = source.startswith("robocasa_")
        pose_mean = (
            old["policy_action_pose_mean"] if fine_source else old["policy_action_coarse_pose_mean"]
        ).float()
        pose_std = (
            old["policy_action_pose_std"] if fine_source else old["policy_action_coarse_pose_std"]
        ).float()
        if tuple(pose_mean.shape) != (6,) or tuple(pose_std.shape) != (6,) or bool((pose_std <= 0).any()):
            raise FormalCacheError("formal action normalization statistics are invalid")

        if fine_source:
            rows = torch.nonzero(history_valid, as_tuple=False).flatten()
            if rows.numel() != 16:
                raise FormalCacheError("RoboCasa history must contain 16 real 20Hz commands")
            hist = history.index_select(0, rows)
            for item in range(16):
                frame, substep = divmod(item, 4)
                frame += 12
                fine[frame, 0, substep, :6] = (hist[item, :6] - pose_mean) / pose_std
                fine[frame, 0, substep, 6] = hist[item, 6].clamp(0, 1)
                fine_mask[frame, 0, substep, :7] = True
                fine_dt[frame, 0, substep] = 0.05 * substep
                fine_sample[frame, 0, substep] = True
            future_fine[:, 0, :, :6] = values[..., :6]
            future_fine[:, 0, :, 6] = self._close01(values[..., 6])
            future_fine_mask[:, 0, :, :7] = valid[..., None]
            future_fine_dt[:, 0] = torch.arange(S).float() * 0.05
            future_fine_sample[:, 0] = valid
            if not bool(valid.all()) or not bool(torch.isfinite(recorded_dt).all()):
                raise FormalCacheError("RoboCasa future action clock/mask is incomplete")
            count = int(old["policy_action_valid_mask"].sum().item())
            if count != 8:
                raise FormalCacheError("RoboCasa policy target must contain 8 commands")
            target_fine[0, :count, :6] = old["policy_action_tgt_norm"][:count].float()
            target_fine[0, :count, 6] = old["policy_action_tgt"][:count, 6].float().clamp(0, 1)
            target_fine_mask[0, :count, :7] = True
            policy_dt[0, :count] = torch.arange(count).float() * 0.05
            policy_mask[0, :count] = True
        else:
            rows = torch.nonzero(history_valid, as_tuple=False).flatten()
            if rows.numel() != 4:
                raise FormalCacheError("OXE coarse history must contain 4 real commands")
            hist = history.index_select(0, rows)
            for item in range(4):
                frame = 12 + item
                coarse[frame, 0, :6] = (hist[item, :6] - pose_mean) / pose_std
                coarse[frame, 0, 6] = hist[item, 6].clamp(0, 1)
                coarse_mask[frame, 0, :7] = True
            if not bool(valid[:, 0].all()) or bool(valid[:, 1:].any()):
                raise FormalCacheError("OXE future action must have one real coarse lane per interval")
            future_coarse[:, 0, :6] = values[:, 0, :6]
            future_coarse[:, 0, 6] = self._close01(values[:, 0, 6])
            future_coarse_mask[:, 0, :7] = True
            count = int(old["policy_action_coarse_valid_mask"].sum().item())
            if count != 2:
                raise FormalCacheError("OXE policy target must contain 2 coarse effects")
            target_coarse[:count, 0, :7] = old["policy_action_coarse_tgt"][:count].float()
            target_coarse_norm[:count, 0, :6] = old["policy_action_coarse_tgt_norm"][:count].float()
            target_coarse_norm[:count, 0, 6] = target_coarse[:count, 0, 6].clamp(0, 1)
            target_coarse_mask[:count, 0, :7] = True
            policy_dt[0, :count] = torch.arange(count).float() * 0.2
            policy_mask[0, :count] = True

        group_ids = torch.zeros(G, dtype=torch.long)
        group_ids[0] = 1
        group_mask = torch.zeros(G, dtype=torch.bool)
        group_mask[0] = True
        action_semantics = torch.zeros(G, A, dtype=torch.long)
        action_semantics[0, :3] = ACTION_SEMANTIC_IDS["delta_position_m"]
        action_semantics[0, 3:6] = ACTION_SEMANTIC_IDS["delta_rotation_axis_angle_rad"]
        action_semantics[0, 6] = ACTION_SEMANTIC_IDS["absolute_gripper_close01"]
        composition = torch.zeros(G, A, dtype=torch.long)
        composition[0, :3] = COMPOSITION_OPERATOR_IDS["sum"]
        composition[0, 3:6] = COMPOSITION_OPERATOR_IDS["so3_axis_angle_body_right"]
        composition[0, 6] = COMPOSITION_OPERATOR_IDS["logical_last"]
        state_semantics = torch.zeros(G, 32, dtype=torch.long)
        state_semantics[0, :3] = STATE_SEMANTIC_IDS["eef_position_m"]
        state_semantics[0, 3:9] = STATE_SEMANTIC_IDS["eef_rotation_6d"]
        state_semantics[0, 9] = STATE_SEMANTIC_IDS["gripper_close01"]
        current_state = torch.zeros(G, 32)
        current_state[0, :10] = old["lowdim_state"].float()
        current_state_mask = torch.zeros(G, 32, dtype=torch.bool)
        current_state_mask[0, :10] = True
        norm_offset = torch.zeros(G, A)
        norm_scale = torch.ones(G, A)
        norm_offset[0, :6] = pose_mean
        norm_scale[0, :6] = pose_std

        context_main = self._decode(old["s_in"])
        context_wrist = self._decode(old["s_wrist"])
        world_tokens = torch.zeros(T, V, P, 2048, dtype=torch.bfloat16)
        world_tokens[:, 0] = context_main
        world_tokens[:, 1] = context_wrist
        view_mask = torch.zeros(T, V, dtype=torch.bool)
        view_mask[:, :2] = old["view_mask"].bool()
        target_tokens = self._decode(old["s_tgt_codec"])
        target_token_mask = torch.isfinite(target_tokens.float()).all(dim=-1)

        rgb_indices = torch.tensor(self.rgb_indices, dtype=torch.long)
        selected_rgb = old["rgb_tgt"].index_select(0, rgb_indices).float().permute(0, 3, 1, 2)
        if selected_rgb.max() > 1.5:
            selected_rgb = selected_rgb / 255.0
        target_rgb = torch.zeros(len(self.rgb_indices), V, 3, 256, 256)
        target_rgb[:, 0] = selected_rgb
        target_rgb_mask = torch.zeros(
            len(self.rgb_indices), V, 1, 1, 1, dtype=torch.bool
        )
        target_rgb_mask[:, 0] = True
        depth = old["depth_tgt"].float().reshape(K, P)
        depth_conf = old["depth_conf_tgt"].float().reshape(K, P)
        target_depth = torch.zeros(K, V, P)
        target_depth[:, 0] = depth
        target_depth_mask = torch.zeros(K, V, P, dtype=torch.bool)
        target_depth_mask[:, 0] = torch.isfinite(depth) & torch.isfinite(depth_conf) & (depth > 0) & (depth_conf > 0)
        point = old["point_tgt"].float().reshape(K, P, 3)
        point_conf = old["point_conf_tgt"].float().reshape(K, P)
        target_point = torch.zeros(K, V, P, 3)
        target_point[:, 0] = point
        target_point_mask = torch.zeros(K, V, P, dtype=torch.bool)
        target_point_mask[:, 0] = torch.isfinite(point).all(dim=-1) & torch.isfinite(point_conf) & (point_conf > 0)
        target_camera_pose = torch.zeros(K, V, 9)
        target_camera_pose[:, 0] = old["pose_geom_tgt"].float()
        target_camera_pose_mask = torch.zeros(K, V, dtype=torch.bool)
        target_camera_pose_mask[:, 0] = torch.isfinite(old["pose_geom_tgt"].float()).all(dim=-1)

        result = {
            "world_tokens": world_tokens,
            "view_mask": view_mask,
            "world_times_s": torch.arange(T + K).float() * 0.2,
            "task_embedding": old["c"].float(),
            "history_fine_action_values": fine,
            "history_fine_action_mask": fine_mask,
            "history_fine_action_dt": fine_dt,
            "history_fine_sample_mask": fine_sample,
            "history_coarse_action_values": coarse,
            "history_coarse_action_mask": coarse_mask,
            "future_factual_fine_action_values": future_fine,
            "future_factual_fine_action_mask": future_fine_mask,
            "future_factual_fine_action_dt": future_fine_dt,
            "future_factual_fine_sample_mask": future_fine_sample,
            "future_factual_coarse_action_values": future_coarse,
            "future_factual_coarse_action_mask": future_coarse_mask,
            "action_group_ids": group_ids,
            "action_group_mask": group_mask,
            "action_semantic_ids": action_semantics,
            "composition_operator_ids": composition,
            "current_state_values": current_state,
            "current_state_mask": current_state_mask,
            "state_semantic_ids": state_semantics,
            "embodiment_ids": torch.tensor(_EMBODIMENT_IDS[source], dtype=torch.long),
            "policy_query_dt": policy_dt,
            "policy_query_mask": policy_mask,
            "target_fine_action": target_fine,
            "target_fine_action_mask": target_fine_mask,
            "target_coarse_action": target_coarse,
            "target_coarse_action_mask": target_coarse_mask,
            "target_coarse_action_normalized": target_coarse_norm,
            "future_world_boundaries_dt": torch.arange(K + 1).float() * 0.2,
            "action_normalization_offset": norm_offset,
            "action_normalization_scale": norm_scale,
            "state_normalization_offset": torch.zeros(G, 32),
            "state_normalization_scale": torch.ones(G, 32),
            "aux_values": torch.zeros(T, 16, 256),
            "aux_mask": torch.zeros(T, 16, dtype=torch.bool),
            "aux_type_ids": torch.zeros(T, 16, dtype=torch.long),
            "target_tokens": target_tokens,
            "target_token_mask": target_token_mask,
            "target_depth": target_depth,
            "target_depth_mask": target_depth_mask,
            "target_point": target_point,
            "target_point_mask": target_point_mask,
            "target_camera_pose": target_camera_pose,
            "target_camera_pose_mask": target_camera_pose_mask,
            "rgb_frame_indices": rgb_indices,
            "target_rgb": target_rgb,
            "target_rgb_mask": target_rgb_mask,
            "source_id": torch.tensor(self.source_names.index(source), dtype=torch.long),
            "sample_index": torch.tensor(index, dtype=torch.long),
        }
        for name, value in result.items():
            if not isinstance(value, torch.Tensor):
                raise FormalCacheError(f"mapped field {name} is not a tensor")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise FormalCacheError(f"mapped field {name} contains NaN/Inf")
        return result


def build_formal_cache_dataset(
    runtime: Mapping[str, Any], *, split: str, profile: FormalCacheProfile | None = None
) -> tuple[FormalCacheDataset, FormalCacheProfile]:
    receipt = validate_formal_cache_closure(runtime["data_closure"])
    if profile is None:
        reader = Path(str(receipt["legacy_reader_root"]))
        reader_text = str(reader)
        if reader_text not in sys.path:
            sys.path.insert(0, reader_text)
        from wm3d_v3.training.train import build_datasets, load_train_config  # type: ignore[import-not-found]

        config = load_train_config(Path(str(receipt["legacy_runtime_config_path"])))
        config["data"]["view_dropout"] = 0.0
        train, val = build_datasets(config)
        observed_lengths = {
            "train": {name: stop - start for name, (start, stop) in train.source_spans.items()},
            "val": {name: stop - start for name, (start, stop) in val.source_spans.items()},
        }
        if observed_lengths != receipt["source_lengths_by_split"]:
            raise FormalCacheError("legacy reader lengths differ from sealed receipt")
        profile = FormalCacheProfile(
            source_order=_SOURCE_ORDER,
            source_weights=dict(_SOURCE_WEIGHTS),
            cache_representation=dict(receipt["cache_representation"]),
            sources=tuple(FormalSource(name) for name in _SOURCE_ORDER),
            declared_eval_coverage_lanes=frozenset(
                {
                    "native_token_supervised_elements",
                    "rgb_supervised_elements",
                    "depth_supervised_elements",
                    "point_supervised_elements",
                    "camera_pose_supervised_elements",
                    "current_state_supervised_dimensions",
                    "fine_supervised_dimensions",
                    "fine_continuous_supervised_dimensions",
                    "fine_binary_supervised_dimensions",
                    "coarse_supervised_dimensions",
                }
            ),
            _train=train,
            _val=val,
        )
    legacy = profile._train if split == "train" else profile._val
    codec = torch.load(
        Path(str(receipt["token_codec_path"])), map_location="cpu", weights_only=True
    )
    if codec.get("version") != "wm3d_v7_pca_int8_v2":
        raise FormalCacheError("formal token codec version mismatch")
    return FormalCacheDataset(
        legacy,
        profile,
        runtime["model_profile"],
        split=split,
        codec=codec,
    ), profile
