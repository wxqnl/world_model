"""Dataset reader for the episode-shared WM3D-v7 compact cache."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from .v8_causal_dual_view import (
    CAUSAL_DUAL_VIEW_REPRESENTATION,
    CAUSAL_DUAL_VIEW_SCHEMA,
    validate_causal_dual_view_archive,
)
from .v8_action_contract import (
    POLICY_HISTORY_DIM,
    POLICY_HISTORY_LEN,
    PoseStats,
    V8_ACTION_SIDECAR_INDEX_SCHEMA,
    V8_ACTION_SIDECAR_SCHEMA,
    V8_ACTION_STATS_SCHEMA,
    build_real_20hz_window_contract,
    require_v8_pinned_file,
    torchify_v8_action_fields,
)
from .v8_proprio_contract import V8_PROPRIO_ANCHOR, V8ProprioStore


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class V7CompactDatasetConfig:
    index_path: Path
    split: str
    T: int = 16
    k: int = 8
    stride: int = 2
    view_dropout: float = 0.0
    seed: int = 0
    require_task_emb: bool = True
    action_stats: Path | None = None
    action_stats_sha256: str | None = None
    require_action_stats: bool = True
    rgb_sidecar_indices: tuple[Path, ...] = ()
    require_rgb_sidecar: bool = False
    action_only: bool = False
    policy_action_history_len: int = 0
    policy_action_history_dim: int = 7
    causal_dual_view_required: bool = False
    causal_dual_view_representation: str | None = None
    trusted_index_fast_init: bool = False
    trusted_index_sha256: str | None = None
    v8_dual_rate_action_enabled: bool = False
    v8_action_sidecar_index: Path | None = None
    v8_action_sidecar_index_sha256: str | None = None
    v8_action_sidecar_stats: Path | None = None
    v8_action_sidecar_stats_sha256: str | None = None
    v8_proprio_enabled: bool = False
    v8_proprio_index: Path | None = None
    v8_proprio_index_sha256: str | None = None
    v8_proprio_stats: Path | None = None
    v8_proprio_stats_sha256: str | None = None


@dataclass
class V7SameRootBranchDatasetConfig:
    index_path: Path
    split: str
    T: int = 16
    k: int = 8
    require_task_emb: bool = True
    action_stats: Path | None = None
    require_action_stats: bool = True


class V7CompactWindowDataset(Dataset):
    """Return compressed tokens; the fixed codec is decoded on GPU by the model."""

    def __init__(self, cfg: V7CompactDatasetConfig):
        self.cfg = cfg
        if cfg.split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split: {cfg.split}")
        if min(cfg.T, cfg.k, cfg.stride) <= 0:
            raise ValueError("T, k, and stride must be positive")
        if cfg.policy_action_history_len < 0:
            raise ValueError("policy_action_history_len must be non-negative")
        if cfg.v8_dual_rate_action_enabled:
            if (
                cfg.policy_action_history_len != POLICY_HISTORY_LEN
                or cfg.policy_action_history_dim != POLICY_HISTORY_DIM
            ):
                raise ValueError(
                    "V8 dual-rate compact policy history must be exact [16,9]"
                )
        elif cfg.policy_action_history_dim != 7:
            raise ValueError("canonical V7 policy action history must be 7D")
        if cfg.causal_dual_view_required:
            if (
                cfg.causal_dual_view_representation
                != CAUSAL_DUAL_VIEW_REPRESENTATION
            ):
                raise ValueError("causal compact mode needs the exact V8 representation")
            if cfg.action_only:
                raise ValueError("causal compact mode requires native 3D targets")
        if cfg.trusted_index_fast_init:
            if not cfg.causal_dual_view_required:
                raise ValueError(
                    "trusted compact index fast init is V8 causal dual-view only"
                )
            if not cfg.trusted_index_sha256:
                raise ValueError(
                    "trusted compact index fast init needs a pinned index digest"
                )
            observed_index_sha256 = _sha256_file(Path(cfg.index_path))
            if observed_index_sha256 != str(cfg.trusted_index_sha256):
                raise RuntimeError(
                    "trusted compact index digest mismatch: "
                    f"observed={observed_index_sha256} "
                    f"expected={cfg.trusted_index_sha256}"
                )
        rows: list[dict] = []
        with Path(cfg.index_path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row["split"] == cfg.split:
                        if not row.get("action_valid") or row.get("pseudo_outcomes") is not False:
                            raise ValueError(f"invalid formal cache row: {row.get('clip_hash')}")
                        teacher = row.get("geometry_teacher") or {}
                        if not teacher.get("pseudo_teacher") or not teacher.get("confidence_stored"):
                            raise ValueError(f"missing geometry pseudo-teacher provenance: {row.get('clip_hash')}")
                        rows.append(row)
        self.records = rows
        self.v8_proprio_store: V8ProprioStore | None = None
        if cfg.v8_proprio_enabled:
            if not cfg.v8_dual_rate_action_enabled:
                raise ValueError(
                    "V8 compact proprio requires the V8 dual-rate action contract"
                )
            if (
                cfg.v8_proprio_index is None
                or not cfg.v8_proprio_index_sha256
                or cfg.v8_proprio_stats is None
                or not cfg.v8_proprio_stats_sha256
            ):
                raise ValueError("V8 compact proprio needs pinned index and stats")
            self.v8_proprio_store = V8ProprioStore(
                index_path=cfg.v8_proprio_index,
                index_sha256=cfg.v8_proprio_index_sha256,
                stats_path=cfg.v8_proprio_stats,
                stats_sha256=cfg.v8_proprio_stats_sha256,
                source="robocasa",
                split=cfg.split,
                expected_identities=(row["clip_hash"] for row in rows),
            )
        self.v8_action_records: dict[str, dict] = {}
        self._v8_action_array_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._v8_action_array_cache_capacity = 64
        self.v8_fine_action_stats: PoseStats | None = None
        self.v8_coarse_action_stats_key: str | None = None
        if cfg.v8_dual_rate_action_enabled:
            required_paths = (
                cfg.v8_action_sidecar_index,
                cfg.v8_action_sidecar_stats,
            )
            required_digests = (
                cfg.v8_action_sidecar_index_sha256,
                cfg.v8_action_sidecar_stats_sha256,
            )
            if any(path is None for path in required_paths) or any(
                not digest for digest in required_digests
            ):
                raise ValueError(
                    "V8 dual-rate compact mode requires pinned sidecar index and stats"
                )
            sidecar_index = Path(cfg.v8_action_sidecar_index).resolve()
            sidecar_stats = Path(cfg.v8_action_sidecar_stats).resolve()
            for path, expected, label in (
                (
                    sidecar_index,
                    str(cfg.v8_action_sidecar_index_sha256),
                    "V8 action sidecar index",
                ),
                (
                    sidecar_stats,
                    str(cfg.v8_action_sidecar_stats_sha256),
                    "V8 action sidecar stats",
                ),
            ):
                if path.is_symlink() or not path.is_file():
                    raise FileNotFoundError(f"{label} is missing/not regular: {path}")
                observed = _sha256_file(path)
                if observed != expected:
                    raise RuntimeError(
                        f"{label} digest mismatch: observed={observed} expected={expected}"
                    )
            with sidecar_index.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    action_row = json.loads(line)
                    if action_row.get("schema") != V8_ACTION_SIDECAR_INDEX_SCHEMA:
                        raise ValueError(
                            f"unexpected V8 action sidecar schema at line {line_number}"
                        )
                    clip_hash = str(action_row.get("clip_hash", ""))
                    if not clip_hash or clip_hash in self.v8_action_records:
                        raise ValueError(
                            f"blank/duplicate V8 action sidecar clip_hash {clip_hash!r}"
                        )
                    action_path = Path(action_row["path"])
                    if action_path.is_symlink() or not action_path.is_file():
                        raise FileNotFoundError(action_path)
                    self.v8_action_records[clip_hash] = action_row
            missing = [
                row["clip_hash"]
                for row in rows
                if row["clip_hash"] not in self.v8_action_records
            ]
            if missing:
                raise ValueError(
                    f"V8 action sidecars omit {len(missing)} compact clips; first={missing[:8]}"
                )
            for row in rows:
                action_row = self.v8_action_records[row["clip_hash"]]
                for key in ("split", "source", "v7_source", "action_audit_sha256"):
                    if str(action_row.get(key)) != str(row.get(key)):
                        raise ValueError(
                            f"V8 action sidecar {key} mismatch: {row['clip_hash']}"
                        )
            with np.load(sidecar_stats, allow_pickle=False) as stats:
                if str(np.asarray(stats["schema"]).item()) != V8_ACTION_STATS_SCHEMA:
                    raise ValueError("unexpected V8 action stats schema")
                if str(np.asarray(stats["split"]).item()) != "train":
                    raise ValueError("V8 action stats must be fit on train only")
                self.v8_fine_action_stats = PoseStats(
                    mean=np.asarray(stats["mean"], dtype=np.float32),
                    std=np.asarray(stats["std"], dtype=np.float32),
                    key=f"robocasa20:{cfg.v8_action_sidecar_stats_sha256}",
                )
            if cfg.action_stats is None:
                raise ValueError("V8 dual-rate compact mode requires action_stats")
            action_stats_path = require_v8_pinned_file(
                cfg.action_stats,
                str(cfg.action_stats_sha256 or ""),
                label="V8 RoboCasa coarse action stats",
            )
            self.v8_coarse_action_stats_key = (
                f"robocasa5:{cfg.action_stats_sha256}"
            )
        self.rgb_records: dict[str, dict] = {}
        for index_path in cfg.rgb_sidecar_indices:
            with Path(index_path).open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    rgb_row = json.loads(line)
                    if rgb_row.get("schema") != "wm3d_v7_rgb_sidecar_v1":
                        raise ValueError(
                            f"unexpected RGB sidecar schema {index_path}:{line_number}"
                        )
                    clip_hash = str(rgb_row.get("clip_hash", ""))
                    if not clip_hash or clip_hash in self.rgb_records:
                        raise ValueError(
                            f"blank or duplicate RGB sidecar clip_hash: {clip_hash}"
                        )
                    self.rgb_records[clip_hash] = rgb_row
        if cfg.require_rgb_sidecar and not cfg.rgb_sidecar_indices:
            raise ValueError("require_rgb_sidecar=true requires rgb_sidecar_indices")
        if cfg.require_rgb_sidecar:
            missing = [row["clip_hash"] for row in rows if row["clip_hash"] not in self.rgb_records]
            if missing:
                raise ValueError(
                    f"RGB sidecar is missing {len(missing)} compact clips; first={missing[:8]}"
                )
        for row in rows:
            rgb_row = self.rgb_records.get(row["clip_hash"])
            if rgb_row is None:
                continue
            if str(rgb_row.get("split")) != str(row["split"]):
                raise ValueError(f"RGB sidecar split mismatch: {row['clip_hash']}")
            if int(rgb_row.get("model_frames", -1)) != int(row["model_frames"]):
                raise ValueError(f"RGB sidecar frame-count mismatch: {row['clip_hash']}")
            if not Path(rgb_row["path"]).is_file():
                raise FileNotFoundError(rgb_row["path"])
        if cfg.action_stats is None:
            if cfg.require_action_stats:
                raise ValueError("formal compact training requires train-only action_stats")
            self.action_mean = np.zeros(6, dtype=np.float32)
            self.action_std = np.ones(6, dtype=np.float32)
        else:
            with np.load(cfg.action_stats, allow_pickle=False) as stats:
                if str(stats["split"].item()) != "train":
                    raise ValueError("action statistics must be fit on train split only")
                self.action_mean = np.asarray(stats["mean"], dtype=np.float32)
                self.action_std = np.asarray(stats["std"], dtype=np.float32)
            if self.action_mean.shape != (6,) or self.action_std.shape != (6,) or np.any(self.action_std <= 0):
                raise ValueError("invalid canonical action statistics")
        self.index: list[tuple[int, int]] = []
        required = cfg.T + cfg.k
        for record_index, row in enumerate(rows):
            if cfg.causal_dual_view_required:
                if row.get("paired_views") is not True:
                    raise ValueError(
                        f"causal RoboCasa cache requires paired views: "
                        f"{row.get('clip_hash')}"
                    )
                if row.get("schema") != CAUSAL_DUAL_VIEW_SCHEMA:
                    raise ValueError(f"unexpected causal dual-view schema: {row.get('clip_hash')}")
                if row.get("representation") != CAUSAL_DUAL_VIEW_REPRESENTATION:
                    raise ValueError(f"unexpected causal compact representation: {row.get('clip_hash')}")
                if row.get("context_future_leakage") is not False:
                    raise ValueError(f"context_future_leakage must be false: {row.get('clip_hash')}")
                if row.get("target_usage") != "supervision_only":
                    raise ValueError(f"target_usage must be supervision_only: {row.get('clip_hash')}")
                if row.get("geometry_coordinate_frame") != "first_observed_camera":
                    raise ValueError(f"unexpected geometry gauge: {row.get('clip_hash')}")
                row_starts = np.asarray(row.get("window_starts"), dtype=np.int64)
                if row_starts.ndim != 1:
                    raise ValueError(
                        f"compact window_starts must be 1D: {row['clip_hash']}"
                    )
                declared_windows = row.get("windows")
                if declared_windows is None:
                    if cfg.trusted_index_fast_init:
                        raise ValueError(
                            "trusted compact index row omits window count: "
                            f"{row['clip_hash']}"
                        )
                elif int(declared_windows) != len(row_starts):
                    raise ValueError(
                        f"compact window count mismatch: {row['clip_hash']}"
                    )
                if not cfg.trusted_index_fast_init:
                    with np.load(row["path"], allow_pickle=False) as archive:
                        summary = validate_causal_dual_view_archive(
                            archive,
                            T=cfg.T,
                            k=cfg.k,
                            paired_views=bool(row.get("paired_views", False)),
                        )
                        if not summary["compact"]:
                            raise ValueError(
                                "RoboCasa causal cache must have a W dimension"
                            )
                        for key, expected in (
                            ("clip_hash", row["clip_hash"]),
                            ("split", row["split"]),
                            ("source", row["source"]),
                            (
                                "action_adapter_version",
                                row["action_adapter_version"],
                            ),
                            ("action_audit_sha256", row["action_audit_sha256"]),
                        ):
                            if str(np.asarray(archive[key]).item()) != str(expected):
                                raise ValueError(
                                    "compact cache identity mismatch for "
                                    f"{key}: {row['clip_hash']}"
                                )
                        archive_starts = np.asarray(
                            archive["window_starts"], dtype=np.int64
                        )
                    if not np.array_equal(row_starts, archive_starts):
                        raise ValueError(
                            "compact window_starts identity mismatch: "
                            f"{row['clip_hash']}"
                        )
                if len(row_starts) == 0 or np.any(np.diff(row_starts) <= 0):
                    raise ValueError(f"compact window_starts must be sorted unique: {row['clip_hash']}")
                if any(
                    start < 0 or start + required > int(row["model_frames"])
                    for start in row_starts.tolist()
                ):
                    raise ValueError(f"compact window start outside clip: {row['clip_hash']}")
                row["_causal_window_lookup"] = {
                    int(start): window_index
                    for window_index, start in enumerate(row_starts.tolist())
                }
                self.index.extend(
                    (record_index, int(start)) for start in row_starts.tolist()
                )
                continue
            segments = row.get("geometry_segments")
            if not segments:
                raise ValueError(f"formal cache has no VGGT gauge segments: {row.get('clip_hash')}")
            for segment_start, segment_stop in segments:
                segment_start, segment_stop = int(segment_start), int(segment_stop)
                if not (0 <= segment_start < segment_stop <= int(row["model_frames"])):
                    raise ValueError(f"invalid geometry segment: {row.get('clip_hash')}")
                if cfg.action_only:
                    last_start = min(
                        segment_stop - cfg.T,
                        int(row["model_frames"]) - cfg.T - cfg.k + 1,
                    )
                    if last_start >= segment_start:
                        self.index.extend(
                            (record_index, start)
                            for start in range(
                                segment_start, last_start + 1, cfg.stride
                            )
                        )
                else:
                    self.index.extend(
                        (record_index, start)
                        for start in range(
                            segment_start,
                            max(segment_start, segment_stop - required + 1),
                            cfg.stride,
                        )
                    )
        if not self.index:
            raise RuntimeError(f"compact cache split {cfg.split} contains no valid windows")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _latent(archive, prefix: str) -> np.ndarray:
        codes = np.asarray(archive[f"{prefix}_codes"], dtype=np.int8)
        scale = np.asarray(archive[f"{prefix}_scale"], dtype=np.float32)
        return codes.astype(np.float32) * scale

    def _drop_wrist(self, sample_index: int) -> bool:
        if self.cfg.split != "train" or self.cfg.view_dropout <= 0:
            return False
        generator = np.random.default_rng(
            np.random.SeedSequence((int(self.cfg.seed), int(self.epoch), int(sample_index)))
        )
        return bool(generator.random() < self.cfg.view_dropout)

    def _load_v8_fine_actions(self, row: dict) -> np.ndarray:
        clip_hash = str(row["clip_hash"])
        cached = self._v8_action_array_cache.get(clip_hash)
        if cached is not None:
            self._v8_action_array_cache.move_to_end(clip_hash)
            return cached
        action_row = self.v8_action_records.get(clip_hash)
        if action_row is None:
            raise RuntimeError(f"V8 action sidecar is missing {clip_hash}")
        path = Path(action_row["path"])
        observed_sha256 = _sha256_file(path)
        expected_sha256 = str(action_row.get("artifact_sha256", ""))
        if not expected_sha256 or observed_sha256 != expected_sha256:
            raise RuntimeError(
                "V8 action sidecar digest mismatch "
                f"{clip_hash}: observed={observed_sha256} expected={expected_sha256}"
            )
        with np.load(path, allow_pickle=False) as archive:
            if str(np.asarray(archive["schema"]).item()) != V8_ACTION_SIDECAR_SCHEMA:
                raise ValueError(f"unexpected V8 action sidecar payload: {clip_hash}")
            for key in (
                "clip_hash",
                "split",
                "source",
                "v7_source",
                "action_audit_sha256",
            ):
                if str(np.asarray(archive[key]).item()) != str(action_row.get(key)):
                    raise ValueError(
                        f"V8 action sidecar payload identity mismatch {key}: {clip_hash}"
                    )
            actions = np.asarray(archive["fine_actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError(f"invalid V8 fine action sidecar: {clip_hash}")
        if int(action_row.get("fine_action_count", -1)) != len(actions):
            raise ValueError(f"V8 fine action count mismatch: {clip_hash}")
        self._v8_action_array_cache[clip_hash] = actions
        self._v8_action_array_cache.move_to_end(clip_hash)
        while len(self._v8_action_array_cache) > self._v8_action_array_cache_capacity:
            self._v8_action_array_cache.popitem(last=False)
        return actions

    def __getitem__(self, sample_index: int) -> dict:
        record_index, start = self.index[sample_index]
        row = self.records[record_index]
        T, k = self.cfg.T, self.cfg.k
        with np.load(row["path"], allow_pickle=False) as archive:
            if self.cfg.causal_dual_view_required:
                validate_causal_dual_view_archive(
                    archive, T=T, k=k, paired_views=True
                )
            elif str(archive["schema"].item()) != "wm3d_v7_compact_geom_v3":
                raise ValueError(f"unexpected compact cache schema: {row['path']}")
            for key, expected in (
                ("clip_hash", row["clip_hash"]),
                ("split", row["split"]),
                ("source", row["source"]),
                ("action_adapter_version", row["action_adapter_version"]),
                ("action_audit_sha256", row["action_audit_sha256"]),
            ):
                if str(archive[key].item()) != str(expected):
                    raise ValueError(f"compact cache identity mismatch for {key}: {row['clip_hash']}")
            paired_views = bool(row.get("paired_views", False))
            actions = np.asarray(archive["actions"], dtype=np.float32)
            action_valid_mask = np.asarray(archive["action_valid_mask"], dtype=np.bool_)
            task_text = (
                str(archive["task_text"].item())
                if "task_text" in archive.files
                else ""
            )
            task = np.asarray(archive["task_emb"], dtype=np.float32)
            if self.cfg.causal_dual_view_required:
                archive_starts = np.asarray(
                    archive["window_starts"], dtype=np.int64
                )
                row_starts = np.asarray(row["window_starts"], dtype=np.int64)
                if not np.array_equal(archive_starts, row_starts):
                    raise ValueError(
                        "compact window_starts identity mismatch: "
                        f"{row['clip_hash']}"
                    )
                window_index = row["_causal_window_lookup"].get(int(start))
                if window_index is None:
                    raise ValueError(f"compact cache omits indexed start {start}")
                anchor = np.asarray(archive["context_codes"][window_index], dtype=np.int8).astype(np.float32)
                anchor *= np.asarray(archive["context_scale"][window_index], dtype=np.float32)
                wrist = np.asarray(archive["wrist_context_codes"][window_index], dtype=np.int8).astype(np.float32)
                wrist *= np.asarray(archive["wrist_context_scale"][window_index], dtype=np.float32)
                future = np.asarray(archive["future_codes"][window_index], dtype=np.int8).astype(np.float32)
                future *= np.asarray(archive["future_scale"][window_index], dtype=np.float32)
                depth = np.asarray(archive["future_depth_patch"][window_index], dtype=np.float32)
                depth_conf = np.asarray(archive["future_depth_conf_patch"][window_index], dtype=np.float32)
                points = np.asarray(archive["future_point_patch"][window_index], dtype=np.float32)
                point_conf = np.asarray(archive["future_point_conf_patch"][window_index], dtype=np.float32)
                pose = np.asarray(archive["future_pose_enc"][window_index], dtype=np.float32)
                geometry_segment_id = None
            else:
                anchor = self._latent(archive, "anchor")
                wrist = self._latent(archive, "wrist") if paired_views else np.zeros_like(anchor)
                geometry_segment_id = np.asarray(archive["geometry_segment_id"], dtype=np.int16)
                if not self.cfg.action_only:
                    depth = np.asarray(archive["depth_patch"], dtype=np.float32)
                    depth_conf = np.asarray(archive["depth_conf_patch"], dtype=np.float32)
                    points = np.asarray(archive["point_patch"], dtype=np.float32)
                    point_conf = np.asarray(archive["point_conf_patch"], dtype=np.float32)
                    pose = np.asarray(archive["pose_enc"], dtype=np.float32)
        if self.cfg.require_task_emb and (task.shape != (2048,) or not np.any(task)):
            raise RuntimeError(f"missing real task embedding: {row['clip_hash']}")
        context_end = start + T
        future_end = context_end + k
        action_start = context_end - 1
        action_end = action_start + k
        context_short = (
            min(len(anchor), len(wrist)) < T
            if self.cfg.causal_dual_view_required
            else min(len(anchor), len(wrist)) < context_end
        )
        if context_short or len(actions) < action_end:
            raise RuntimeError(f"short compact cache record: {row['clip_hash']}")
        if (
            len(action_valid_mask) < action_end
            or not action_valid_mask[action_start:action_end].all()
        ):
            raise RuntimeError(f"invalid action interval: {row['clip_hash']}")
        if not self.cfg.causal_dual_view_required:
            geometry_end = context_end if self.cfg.action_only else future_end
            if len(geometry_segment_id) < geometry_end:
                raise RuntimeError(f"short geometry segment record: {row['clip_hash']}")
            if len(np.unique(geometry_segment_id[start:geometry_end])) != 1:
                raise RuntimeError(f"window crossed a VGGT geometry gauge boundary: {row['clip_hash']}")
        # The action at model index t drives the transition from frame t to t+1.
        action_window = actions[action_start : action_start + k]
        previous_grip = actions[action_start - 1 : action_start, 6]
        action_history = None
        if (
            self.cfg.policy_action_history_len > 0
            and not self.cfg.v8_dual_rate_action_enabled
        ):
            history_start = action_start - self.cfg.policy_action_history_len
            if history_start < 0:
                raise RuntimeError(f"short canonical action history: {row['clip_hash']}")
            action_history = np.array(
                actions[
                    history_start:action_start,
                    : self.cfg.policy_action_history_dim,
                ],
                dtype=np.float32,
                copy=True,
            )
            # Use one executable convention across every source: physical
            # canonical pose plus close01 gripper state.
            action_history[:, 6] = np.clip(
                (action_history[:, 6] + 1.0) * 0.5,
                0.0,
                1.0,
            )
        wrist_dropped = (not paired_views) or self._drop_wrist(sample_index)
        view_mask = np.ones((T, 2), dtype=np.bool_)
        if wrist_dropped:
            view_mask[:, 1] = False
        sample = {
            "s_in": torch.from_numpy(
                anchor.copy()
                if self.cfg.causal_dual_view_required
                else anchor[start : start + T].copy()
            ),
            "s_wrist": torch.from_numpy(
                wrist.copy()
                if self.cfg.causal_dual_view_required
                else wrist[start : start + T].copy()
            ),
            "view_mask": torch.from_numpy(view_mask),
            "action_tgt": torch.from_numpy(action_window.copy()),
            "action_tgt_norm": torch.from_numpy(
                ((action_window[:, :6] - self.action_mean) / self.action_std).copy()
            ),
            # Multi-source S1 must reconstruct physical actions with the
            # statistics of this sample, never ActionProjHead's single global
            # convenience buffer.
            "action_pose_mean": torch.from_numpy(self.action_mean.copy()),
            "action_pose_std": torch.from_numpy(self.action_std.copy()),
            "action_prev_grip": torch.from_numpy(previous_grip.copy()),
            "c": torch.from_numpy(task.copy()),
            "task_text": task_text,
            "clip_id": row["clip_hash"],
            "start": start,
            "dataset": row.get("v7_source", row["source"]),
            "action_frame_indices": torch.arange(action_start, action_start + k),
            "action_valid_count": len(actions),
            "action_contract_key": "robocasa365|5|wm3d_v7_base_delta_axisangle_gripclose_v1",
            "action_frame_offset": -1,
        }
        if self.v8_proprio_store is not None:
            proprio = self.v8_proprio_store.current(row["clip_hash"], action_start)
            sample.update(
                {
                    "lowdim_state": torch.from_numpy(proprio.normalized),
                    "policy_proprio_raw": torch.from_numpy(proprio.raw),
                    "embodiment_id": torch.tensor(
                        proprio.embodiment_id, dtype=torch.long
                    ),
                    "policy_proprio_stats_key": proprio.stats_key,
                    "policy_proprio_anchor": V8_PROPRIO_ANCHOR,
                    "policy_proprio_frame_index": proprio.anchor_frame_index,
                }
            )
        if action_history is not None:
            sample["action_history"] = torch.from_numpy(action_history)
        if self.cfg.v8_dual_rate_action_enabled:
            if self.v8_fine_action_stats is None:
                raise RuntimeError("V8 fine action statistics were not initialized")
            if self.v8_coarse_action_stats_key is None:
                raise RuntimeError("V8 coarse action statistics were not initialized")
            v8_fields = build_real_20hz_window_contract(
                fine_actions=self._load_v8_fine_actions(row),
                world_actions=actions,
                world_action_start=action_start,
                world_horizon=k,
                fine_stats=self.v8_fine_action_stats,
                coarse_stats=PoseStats(
                    mean=self.action_mean,
                    std=self.action_std,
                    key=self.v8_coarse_action_stats_key,
                ),
            )
            sample.update(torchify_v8_action_fields(v8_fields))
        if not self.cfg.action_only:
            if self.cfg.causal_dual_view_required:
                target_arrays = (future, depth, depth_conf, points, point_conf, pose)
                if min(len(value) for value in target_arrays) < k:
                    raise RuntimeError(f"short causal compact target: {row['clip_hash']}")
                target_slice = slice(0, k)
                state_target = future
            else:
                if min(
                    len(anchor), len(depth), len(depth_conf), len(points),
                    len(point_conf), len(pose),
                ) < future_end:
                    raise RuntimeError(f"short compact target record: {row['clip_hash']}")
                target_slice = slice(context_end, future_end)
                state_target = anchor
            sample.update(
                {
                    "s_tgt_codec": torch.from_numpy(state_target[target_slice].copy()),
                    "depth_tgt": torch.from_numpy(depth[target_slice].copy()),
                    "depth_conf_tgt": torch.from_numpy(depth_conf[target_slice].copy()),
                    "point_tgt": torch.from_numpy(points[target_slice].copy()),
                    "point_conf_tgt": torch.from_numpy(point_conf[target_slice].copy()),
                    "pose_geom_tgt": torch.from_numpy(pose[target_slice].copy()),
                }
            )
        rgb_row = self.rgb_records.get(row["clip_hash"])
        if rgb_row is not None:
            with np.load(rgb_row["path"], allow_pickle=False) as rgb_archive:
                if str(rgb_archive["schema"].item()) != "wm3d_v7_rgb_sidecar_v1":
                    raise ValueError(f"unexpected RGB payload schema: {row['clip_hash']}")
                for key, expected in (
                    ("clip_hash", row["clip_hash"]),
                    ("split", row["split"]),
                    ("source", row["source"]),
                ):
                    if str(rgb_archive[key].item()) != str(expected):
                        raise ValueError(
                            f"RGB sidecar identity mismatch for {key}: {row['clip_hash']}"
                        )
                rgb = np.asarray(rgb_archive["rgb_anchor"], dtype=np.uint8)
            if rgb.shape != (int(row["model_frames"]), 256, 256, 3):
                raise RuntimeError(
                    f"RGB sidecar shape mismatch for {row['clip_hash']}: {rgb.shape}"
                )
            sample["rgb_in"] = torch.from_numpy(rgb[start : start + T].copy()).float().div_(255.0)
            if not self.cfg.action_only:
                sample["rgb_tgt"] = torch.from_numpy(
                    rgb[context_end:future_end].copy()
                ).float().div_(255.0)
        return sample


class V7SameRootBranchDataset(Dataset):
    """One true same-root K-branch calibration sample per simulator root."""

    def __init__(self, cfg: V7SameRootBranchDatasetConfig):
        self.cfg = cfg
        if cfg.split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split: {cfg.split}")
        self.records = []
        with Path(cfg.index_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["split"] != cfg.split:
                    continue
                if row.get("schema") != "wm3d_v7_same_root_branch_compact_v1":
                    raise ValueError(f"unexpected same-root schema: {row.get('root_id')}")
                if not row.get("same_root_current_runtime_exact"):
                    raise ValueError(f"same-root exactness missing: {row.get('root_id')}")
                if row.get("historical_runtime_reconstruction_exact") is not False:
                    raise ValueError(f"historical runtime provenance missing: {row.get('root_id')}")
                if row.get("pseudo_outcomes") is not False:
                    raise ValueError(f"pseudo same-root outcome forbidden: {row.get('root_id')}")
                if int(row["context_frames"]) != cfg.T or int(row["future_frames"]) != cfg.k:
                    raise ValueError(f"same-root horizon mismatch: {row.get('root_id')}")
                self.records.append(row)
        if not self.records:
            raise RuntimeError(f"same-root compact split {cfg.split} is empty")
        if cfg.action_stats is None:
            if cfg.require_action_stats:
                raise ValueError("same-root training requires train-only action_stats")
            self.action_mean = np.zeros(6, dtype=np.float32)
            self.action_std = np.ones(6, dtype=np.float32)
        else:
            with np.load(cfg.action_stats, allow_pickle=False) as stats:
                if str(stats["split"].item()) != "train":
                    raise ValueError("action statistics must be fit on train split only")
                self.action_mean = np.asarray(stats["mean"], dtype=np.float32)
                self.action_std = np.asarray(stats["std"], dtype=np.float32)
            if self.action_mean.shape != (6,) or self.action_std.shape != (6,):
                raise ValueError("invalid same-root action statistics")
            if np.any(self.action_std <= 0):
                raise ValueError("same-root action std must be positive")

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _dequantize(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return np.asarray(codes, dtype=np.int8).astype(np.float32) * np.asarray(
            scale, dtype=np.float32
        )

    def __getitem__(self, sample_index: int) -> dict:
        row = self.records[sample_index]
        with np.load(row["path"], allow_pickle=False) as archive:
            if str(archive["schema"].item()) != "wm3d_v7_same_root_branch_compact_v1":
                raise ValueError(f"same-root payload schema mismatch: {row['root_id']}")
            if str(archive["root_id"].item()) != row["root_id"]:
                raise ValueError(f"same-root payload identity mismatch: {row['root_id']}")
            context = self._dequantize(archive["anchor_codes"], archive["anchor_scale"])
            wrist = self._dequantize(archive["wrist_codes"], archive["wrist_scale"])
            factual = self._dequantize(archive["factual_codes"], archive["factual_scale"])
            branches = self._dequantize(archive["branch_codes"], archive["branch_scales"])
            branch_actions = np.asarray(archive["branch_actions"], dtype=np.float32)
            branch_valid = np.asarray(archive["branch_valid"], dtype=np.bool_)
            task = np.asarray(archive["task_emb"], dtype=np.float32)
            depth = np.asarray(archive["depth_tgt"], dtype=np.float32)
            depth_conf = np.asarray(archive["depth_conf_tgt"], dtype=np.float32)
            points = np.asarray(archive["point_tgt"], dtype=np.float32)
            point_conf = np.asarray(archive["point_conf_tgt"], dtype=np.float32)
            pose = np.asarray(archive["pose_geom_tgt"], dtype=np.float32)
            branch_rewards = np.asarray(archive["branch_rewards"], dtype=np.float32)
            branch_dones = np.asarray(archive["branch_dones"], dtype=np.bool_)
            branch_success = np.asarray(archive["branch_success"], dtype=np.bool_)
        if context.shape[0] != self.cfg.T or wrist.shape[0] != self.cfg.T:
            raise RuntimeError(f"same-root context length mismatch: {row['root_id']}")
        if factual.shape[0] != self.cfg.k or branches.shape[1] != self.cfg.k:
            raise RuntimeError(f"same-root target length mismatch: {row['root_id']}")
        if branch_actions.shape[:2] != branches.shape[:2] or branch_actions.shape[-1] != 7:
            raise RuntimeError(f"same-root action/target mismatch: {row['root_id']}")
        outcome_shape = branch_actions.shape[:2]
        if any(
            outcome.shape != outcome_shape
            for outcome in (branch_rewards, branch_dones, branch_success)
        ):
            raise RuntimeError(f"same-root outcome/action mismatch: {row['root_id']}")
        if branch_valid.shape != (branches.shape[0],) or not branch_valid.all():
            raise RuntimeError(f"same-root branch validity mismatch: {row['root_id']}")
        if self.cfg.require_task_emb and (task.shape != (2048,) or not np.any(task)):
            raise RuntimeError(f"missing same-root task embedding: {row['root_id']}")
        action_tgt = branch_actions[0]
        branch_action_cond = branch_actions.copy()
        branch_action_cond[:, :, :6] = (
            branch_action_cond[:, :, :6] - self.action_mean[None, None]
        ) / self.action_std[None, None]
        branch_action_cond[:, :, 6] = (branch_actions[:, :, 6] > 0.5).astype(np.float32)
        return {
            "s_in": torch.from_numpy(context.copy()),
            "s_wrist": torch.from_numpy(wrist.copy()),
            "view_mask": torch.ones((self.cfg.T, 2), dtype=torch.bool),
            "s_tgt_codec": torch.from_numpy(factual.copy()),
            "depth_tgt": torch.from_numpy(depth.copy()),
            "depth_conf_tgt": torch.from_numpy(depth_conf.copy()),
            "point_tgt": torch.from_numpy(points.copy()),
            "point_conf_tgt": torch.from_numpy(point_conf.copy()),
            "pose_geom_tgt": torch.from_numpy(pose.copy()),
            "action_tgt": torch.from_numpy(action_tgt.copy()),
            "action_tgt_norm": torch.from_numpy(
                ((action_tgt[:, :6] - self.action_mean) / self.action_std).copy()
            ),
            "action_pose_mean": torch.from_numpy(self.action_mean.copy()),
            "action_pose_std": torch.from_numpy(self.action_std.copy()),
            # The current compact branch schema does not contain the true
            # root gripper state.  In particular, action_tgt[0, 6] is the
            # first *future command* and must never be presented as history.
            # Event supervision therefore ignores the first step for these
            # legacy roots and uses only within-horizon true transitions.
            "branch_actions": torch.from_numpy(branch_action_cond),
            "branch_s_tgt_codec": torch.from_numpy(branches.copy()),
            "branch_valid": torch.from_numpy(branch_valid.copy()),
            "branch_rewards": torch.from_numpy(branch_rewards.copy()),
            "branch_dones": torch.from_numpy(branch_dones.copy()),
            "branch_success": torch.from_numpy(branch_success.copy()),
            "c": torch.from_numpy(task.copy()),
            "clip_id": row["root_id"],
            "start": 0,
            "dataset": "robocasa_same_root_current_runtime",
            "action_frame_indices": torch.arange(self.cfg.k),
            "action_valid_count": self.cfg.k,
            "action_contract_key": "robocasa365|5|wm3d_v7_base_delta_axisangle_gripclose_v1",
            "action_frame_offset": -1,
        }
