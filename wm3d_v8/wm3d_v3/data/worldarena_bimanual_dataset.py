"""Leakage-locked WorldArena bimanual adaptation cache reader."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from wm3d_v3.data.action_condition import make_action_condition


CACHE_SCHEMA = "wm3d_v7_worldarena_bimanual_compact_v1"
INDEX_SCHEMA = "wm3d_v7_worldarena_bimanual_index_v1"
CONTEXT_FRAMES = 16
SEGMENT_FRAMES = 32


class WorldArenaBimanualDataset(Dataset):
    """Read fixed-gauge 32-frame records without exposing test episodes.

    A protocol-matched example may use the first cached frame as its only
    observation and all remaining 31 frames as recurrent future targets.  The
    previous 16-frame cap was an adaptation-policy choice, not a cache-format
    constraint, and prevented training the long causal path used at inference.
    """

    def __init__(
        self,
        index: Path,
        action_stats: Path,
        *,
        split: str,
        future_horizon: int = 8,
        protocol_match_first_frame: bool = False,
        start_offsets: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("WorldArena adaptation split must be train or val")
        if not 1 <= int(future_horizon) <= SEGMENT_FRAMES - 1:
            raise ValueError(
                f"future_horizon must be in 1..{SEGMENT_FRAMES - 1}"
            )
        self.split = split
        self.future_horizon = int(future_horizon)
        self.protocol_match_first_frame = bool(protocol_match_first_frame)
        if start_offsets is None:
            start_offsets = [0]
        self.start_offsets = [int(offset) for offset in start_offsets]
        if not self.start_offsets:
            raise ValueError("start_offsets must be non-empty")
        max_offset = SEGMENT_FRAMES - 1 - self.future_horizon
        if min(self.start_offsets) < 0 or max(self.start_offsets) > max_offset:
            raise ValueError(f"start_offsets must be in 0..{max_offset}")
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(Path(index).read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != INDEX_SCHEMA:
                raise RuntimeError(f"unexpected index schema at line {line_number}")
            episode = int(row.get("episode", -1))
            if episode >= 40:
                raise RuntimeError(f"WorldArena adaptation index contains test episode {episode}")
            if episode < 0:
                raise RuntimeError(f"invalid episode at index line {line_number}")
            expected_split = "train" if episode <= 35 else "val"
            if row.get("split") != expected_split:
                raise RuntimeError(
                    f"adaptation split mismatch for episode {episode}: {row.get('split')}"
                )
            if row.get("split") == split:
                rows.append(row)
        if not rows:
            raise RuntimeError(f"WorldArena adaptation split is empty: {split}")
        ids = [str(row.get("record_id", "")) for row in rows]
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise RuntimeError("WorldArena adaptation index has blank or duplicate record IDs")
        for row in rows:
            if not Path(row["path"]).is_file():
                raise FileNotFoundError(row["path"])
            for key in ("source_video_sha256", "source_hdf5_sha256"):
                value = str(row.get(key, ""))
                if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                    raise RuntimeError(f"invalid {key} for {row['record_id']}")
        self.rows = rows
        self.examples: list[tuple[int, int]] = []
        for row_index in range(len(rows)):
            for offset in self.start_offsets:
                self.examples.append((row_index, offset))

        with np.load(action_stats, allow_pickle=False) as stats:
            if str(stats["split"].item()) != "train":
                raise RuntimeError("WorldArena action statistics must be train-only")
            self.mean = np.asarray(stats["mean"], dtype=np.float32)
            self.std = np.asarray(stats["std"], dtype=np.float32)
        if self.mean.shape != (6,) or self.std.shape != (6,):
            raise RuntimeError("WorldArena pose action statistics must have shape (6,)")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise RuntimeError("WorldArena action statistics are nonfinite or nonpositive")

    def __len__(self) -> int:
        return len(self.examples)

    @staticmethod
    def _array(archive: Any, key: str, shape: tuple[int, ...]) -> np.ndarray:
        value = np.asarray(archive[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise RuntimeError(f"invalid cache field {key}: {value.shape} != {shape}")
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index, start_offset = self.examples[index]
        row = self.rows[row_index]
        horizon = self.future_horizon
        with np.load(row["path"], allow_pickle=False) as archive:
            for key, expected in (
                ("schema", CACHE_SCHEMA),
                ("record_id", row["record_id"]),
                ("task", row["task"]),
                ("episode", int(row["episode"])),
                ("split", row["split"]),
                ("segment_start", int(row["segment_start"])),
            ):
                observed = archive[key].item()
                if str(observed) != str(expected):
                    raise RuntimeError(
                        f"WorldArena cache identity mismatch for {key}: {observed} != {expected}"
                    )
            codes = np.asarray(archive["anchor_codes"], dtype=np.int8)
            scale = np.asarray(archive["anchor_scale"], dtype=np.float32)
            if codes.shape != (SEGMENT_FRAMES, 64, 384) or scale.shape != (SEGMENT_FRAMES, 1, 1):
                raise RuntimeError("invalid frozen-codec cache shapes")
            latent = codes.astype(np.float32) * scale
            rgb = np.asarray(archive["rgb"], dtype=np.uint8)
            if rgb.shape != (SEGMENT_FRAMES, 256, 256, 3):
                raise RuntimeError(f"invalid WorldArena RGB cache shape: {rgb.shape}")
            task_emb = self._array(archive, "task_emb", (2048,)).astype(np.float32)
            if not np.any(task_emb):
                raise RuntimeError(f"missing real task embedding: {row['record_id']}")
            depth = self._array(archive, "depth_patch", (SEGMENT_FRAMES, 8, 8)).astype(np.float32)
            depth_conf = self._array(
                archive, "depth_conf_patch", (SEGMENT_FRAMES, 8, 8)
            ).astype(np.float32)
            point = self._array(
                archive, "point_patch", (SEGMENT_FRAMES, 8, 8, 3)
            ).astype(np.float32)
            point_conf = self._array(
                archive, "point_conf_patch", (SEGMENT_FRAMES, 8, 8)
            ).astype(np.float32)
            pose = self._array(archive, "pose_enc", (SEGMENT_FRAMES, 9)).astype(np.float32)
            left = self._array(archive, "left_actions", (SEGMENT_FRAMES - 1, 7)).astype(np.float32)
            right = self._array(archive, "right_actions", (SEGMENT_FRAMES - 1, 7)).astype(np.float32)

        if np.any((left[:, 6] < 0) | (left[:, 6] > 1)) or np.any(
            (right[:, 6] < 0) | (right[:, 6] > 1)
        ):
            raise RuntimeError(f"WorldArena gripper is not close01: {row['record_id']}")
        if self.protocol_match_first_frame:
            anchor_index = start_offset
            target_start = start_offset + 1
            context_latent = np.repeat(latent[anchor_index : anchor_index + 1], CONTEXT_FRAMES, axis=0)
            context_rgb_np = rgb[anchor_index]
        else:
            anchor_index = CONTEXT_FRAMES - 1
            target_start = CONTEXT_FRAMES
            context_latent = latent[:CONTEXT_FRAMES]
            context_rgb_np = rgb[anchor_index]
        future_stop = target_start + horizon
        action_start = anchor_index
        left_window = left[action_start : action_start + horizon]
        right_window = right[action_start : action_start + horizon]
        if left_window.shape != (horizon, 7) or right_window.shape != (horizon, 7):
            raise RuntimeError(f"short bimanual action window: {row['record_id']}")
        record_id = str(row["record_id"])
        if self.protocol_match_first_frame:
            record_id = f"{record_id}@offset{start_offset}"
        return {
            "record_id": record_id,
            "task": str(row["task"]),
            "episode": int(row["episode"]),
            "segment_start": int(row["segment_start"]),
            "start_offset": start_offset,
            "protocol_match_first_frame": self.protocol_match_first_frame,
            "context_latent": torch.from_numpy(context_latent.copy()),
            "target_latent": torch.from_numpy(latent[target_start:future_stop].copy()),
            "context_rgb": torch.from_numpy(context_rgb_np.copy()).permute(2, 0, 1).float().div_(255.0),
            "target_rgb": torch.from_numpy(rgb[target_start:future_stop].copy()).permute(0, 3, 1, 2).float().div_(255.0),
            "depth_tgt": torch.from_numpy(depth[target_start:future_stop].copy()),
            "depth_conf_tgt": torch.from_numpy(depth_conf[target_start:future_stop].copy()),
            "point_tgt": torch.from_numpy(point[target_start:future_stop].copy()),
            "point_conf_tgt": torch.from_numpy(point_conf[target_start:future_stop].copy()),
            "pose_tgt": torch.from_numpy(pose[target_start:future_stop].copy()),
            "left_action": torch.from_numpy(left_window.copy()),
            "right_action": torch.from_numpy(right_window.copy()),
            "left_action_norm": torch.from_numpy(
                ((left_window[:, :6] - self.mean) / self.std).copy()
            ),
            "right_action_norm": torch.from_numpy(
                ((right_window[:, :6] - self.mean) / self.std).copy()
            ),
            "action_pose_mean": torch.from_numpy(self.mean.copy()),
            "action_pose_std": torch.from_numpy(self.std.copy()),
            "task_emb": torch.from_numpy(task_emb.copy()),
        }


class WorldArenaWanWindowDataset(Dataset):
    """Expose leakage-safe bimanual windows through the S2+Wan batch contract.

    The native WM receives the two arm streams separately (the trainer performs
    the three-branch left/right/physical-zero fusion).  Wan's direct action
    stream receives the dominant physical arm action, matching the public
    WorldArena renderer protocol instead of an artificial all-zero action.
    """

    def __init__(
        self,
        index: Path,
        action_stats: Path,
        *,
        split: str,
        start_offsets: list[int] | tuple[int, ...],
    ) -> None:
        self.base = WorldArenaBimanualDataset(
            index,
            action_stats,
            split=split,
            future_horizon=8,
            protocol_match_first_frame=True,
            start_offsets=start_offsets,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        left = item["left_action"]
        right = item["right_action"]
        left_norm = item["left_action_norm"]
        right_norm = item["right_action_norm"]
        if left.shape != (8, 7) or right.shape != (8, 7):
            raise RuntimeError("WorldArena Wan requires exact K8 bimanual actions")
        choose_left = torch.linalg.vector_norm(left[:, :6], dim=-1) >= torch.linalg.vector_norm(
            right[:, :6], dim=-1
        )
        dominant = torch.where(choose_left[:, None], left, right)
        dominant_norm = torch.where(choose_left[:, None], left_norm, right_norm)

        # Wan consumes both arms explicitly.  Keep the native dominant-arm
        # targets below byte-for-byte unchanged: they are the frozen WM's
        # existing 7D contract, while this renderer-only tensor is the new
        # direct bimanual boundary.
        left_condition = make_action_condition(left, left_norm)
        right_condition = make_action_condition(right, right_norm)
        renderer_action_cond = torch.cat(
            [left_condition, right_condition], dim=-1
        )
        zero_action = torch.zeros_like(left)
        zero_pose_norm = (
            -item["action_pose_mean"] / item["action_pose_std"]
        ).to(dtype=left.dtype)
        renderer_zero_action_cond = make_action_condition(
            zero_action,
            zero_pose_norm.unsqueeze(0).expand(8, -1),
        )
        if renderer_action_cond.shape != (8, 14):
            raise RuntimeError("WorldArena renderer action must be exact [K8,14]")
        if renderer_zero_action_cond.shape != (8, 7):
            raise RuntimeError("WorldArena renderer zero action must be exact [K8,7]")
        if not bool(torch.isfinite(renderer_action_cond).all()) or not bool(
            torch.isfinite(renderer_zero_action_cond).all()
        ):
            raise RuntimeError("WorldArena renderer action contract is non-finite")

        state = item["context_latent"]
        target = item["target_latent"]
        if state.shape[:2] != (16, 64) or target.shape[:2] != (8, 64):
            raise RuntimeError("WorldArena Wan must preserve native T16/K8 codec tokens")
        view_mask = torch.zeros((16, 2), dtype=torch.bool)
        view_mask[:, 0] = True
        return {
            "s_in": state,
            "s_wrist": torch.zeros_like(state),
            "view_mask": view_mask,
            "s_tgt_codec": target,
            "rgb_in": item["context_rgb"].permute(1, 2, 0).unsqueeze(0).contiguous(),
            "rgb_tgt": item["target_rgb"].permute(0, 2, 3, 1).contiguous(),
            "depth_tgt": item["depth_tgt"],
            "depth_conf_tgt": item["depth_conf_tgt"],
            "point_tgt": item["point_tgt"],
            "point_conf_tgt": item["point_conf_tgt"],
            "pose_geom_tgt": item["pose_tgt"],
            "left_action": left,
            "right_action": right,
            "action_tgt": dominant,
            "action_tgt_norm": dominant_norm,
            "action_grip_close01": (dominant[:, 6] > 0.5).float(),
            "renderer_action_cond": renderer_action_cond,
            "renderer_zero_action_cond": renderer_zero_action_cond,
            "action_pose_mean": item["action_pose_mean"],
            "action_pose_std": item["action_pose_std"],
            "c": item["task_emb"],
            "clip_id": item["record_id"],
            "start": int(item["start_offset"]),
            "dataset": "worldarena_train_only",
            "task_text": str(item["task"]).replace("_", " "),
        }


class WorldArenaNativeS0WindowDataset(WorldArenaWanWindowDataset):
    """Expose WorldArena through the native S0 action-dynamics contract.

    ``action_tgt`` intentionally remains the dominant physical 7D arm action:
    it is only used by legacy diagnostic/action-head targets.  The factual
    world transition receives both arms directly through
    ``native_action_cond``.  This avoids the old left/right/zero three-forward
    fusion while keeping every non-bimanual S0 code path byte-for-byte
    unchanged.
    """

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        native_action_cond = sample["renderer_action_cond"]
        if native_action_cond.shape != (8, 14):
            raise RuntimeError(
                "WorldArena native S0 action must be exact [K8,14]"
            )
        if not bool(torch.isfinite(native_action_cond).all()):
            raise RuntimeError("WorldArena native S0 action is non-finite")
        sample["native_action_cond"] = native_action_cond
        sample["dataset"] = "worldarena_native_s0_train_only"
        return sample


@dataclass(frozen=True)
class WorldArenaK8Pair:
    """Identity-locked pair of adjacent K8 windows from one cache record."""

    a_sample_index: int
    b_sample_index: int
    row_index: int
    start_a: int
    start_b: int
    record_id: str
    episode: int
    segment_start: int


class WorldArenaPairedK8Dataset(Dataset):
    """Expose ``offset`` and ``offset + 8`` from the same 32-frame record.

    WorldArena's first-frame rollout protocol does not have the window-cache
    gauge used by the generic Stage1 ``PairedK8Dataset``.  Pairing by the
    immutable row index is stricter for this cache: both samples necessarily
    share the same task, episode, source segment, and NPZ payload.  Validation
    and test episodes are rejected so rolling adaptation cannot leak them.
    """

    def __init__(
        self,
        base: WorldArenaWanWindowDataset,
        *,
        pair_offset: int = 8,
    ) -> None:
        if not isinstance(base, WorldArenaWanWindowDataset):
            raise TypeError("WorldArena paired-K8 requires WorldArenaWanWindowDataset")
        if int(pair_offset) != 8:
            raise ValueError("WorldArena paired-K8 requires exact offset 8")
        source = base.base
        if source.split != "train":
            raise ValueError("WorldArena paired-K8 is train-only")
        if source.future_horizon != 8 or not source.protocol_match_first_frame:
            raise ValueError(
                "WorldArena paired-K8 requires first-frame protocol with K=8"
            )

        lookup: dict[tuple[int, int], int] = {}
        for sample_index, (row_index, start) in enumerate(source.examples):
            key = (int(row_index), int(start))
            if key in lookup:
                raise ValueError(
                    "duplicate WorldArena rolling window topology: "
                    f"row={row_index} start={start}"
                )
            lookup[key] = int(sample_index)

        pairs: list[WorldArenaK8Pair] = []
        for (row_index, start_a), a_sample_index in sorted(lookup.items()):
            start_b = int(start_a) + int(pair_offset)
            b_sample_index = lookup.get((row_index, start_b))
            if b_sample_index is None:
                continue
            row = source.rows[row_index]
            episode = int(row["episode"])
            if row.get("split") != "train" or not 0 <= episode <= 35:
                raise RuntimeError(
                    "WorldArena rolling pair crossed the train-only boundary: "
                    f"record={row.get('record_id')} episode={episode}"
                )
            if not 0 <= start_a < start_b <= 15:
                raise RuntimeError(
                    "WorldArena rolling pair lies outside the 32-frame K8 gauge: "
                    f"start={start_a}/{start_b}"
                )
            pairs.append(
                WorldArenaK8Pair(
                    a_sample_index=a_sample_index,
                    b_sample_index=b_sample_index,
                    row_index=row_index,
                    start_a=start_a,
                    start_b=start_b,
                    record_id=str(row["record_id"]),
                    episode=episode,
                    segment_start=int(row["segment_start"]),
                )
            )
        if not pairs:
            raise RuntimeError(
                "WorldArena paired-K8 found no same-record offset/+8 pairs"
            )
        self.base = base
        self.pairs = tuple(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, pair_index: int) -> dict[str, Any]:
        pair = self.pairs[pair_index]
        sample_a = self.base[pair.a_sample_index]
        sample_b = self.base[pair.b_sample_index]
        if (
            int(sample_a["start"]) != pair.start_a
            or int(sample_b["start"]) != pair.start_b
            or str(sample_a["clip_id"]).split("@offset", 1)[0] != pair.record_id
            or str(sample_b["clip_id"]).split("@offset", 1)[0] != pair.record_id
        ):
            raise RuntimeError(
                "WorldArena paired-K8 runtime identity differs from frozen topology"
            )
        return {
            "a": sample_a,
            "b": sample_b,
            "pair_index": int(pair_index),
            "pair_start_a": pair.start_a,
            "pair_start_b": pair.start_b,
            "pair_record_id": pair.record_id,
            "pair_episode": pair.episode,
            "pair_segment_start": pair.segment_start,
        }


@dataclass(frozen=True)
class WorldArenaK8Triplet:
    """Identity-locked A/B/C windows at offsets ``x/x+8/x+16``."""

    a_sample_index: int
    b_sample_index: int
    c_sample_index: int
    row_index: int
    start_a: int
    start_b: int
    start_c: int
    record_id: str
    episode: int
    segment_start: int


class WorldArenaTripletK8Dataset(Dataset):
    """Expose three adjacent K8 windows from one train-only 32-frame record.

    The third window is the first training target whose recurrent native state
    can be composed entirely of predictions: ``pred(A) || pred(B)``.  This
    matches the serving boundary after two generated chunks without reading
    future RGB or native tokens during inference.
    """

    def __init__(
        self,
        base: WorldArenaWanWindowDataset,
        *,
        triplet_offset: int = 8,
    ) -> None:
        if not isinstance(base, WorldArenaWanWindowDataset):
            raise TypeError("WorldArena triplet-K8 requires WorldArenaWanWindowDataset")
        if int(triplet_offset) != 8:
            raise ValueError("WorldArena triplet-K8 requires exact offset 8")
        source = base.base
        if source.split != "train":
            raise ValueError("WorldArena triplet-K8 is train-only")
        if source.future_horizon != 8 or not source.protocol_match_first_frame:
            raise ValueError(
                "WorldArena triplet-K8 requires first-frame protocol with K=8"
            )

        lookup: dict[tuple[int, int], int] = {}
        for sample_index, (row_index, start) in enumerate(source.examples):
            key = (int(row_index), int(start))
            if key in lookup:
                raise ValueError(
                    "duplicate WorldArena triplet topology: "
                    f"row={row_index} start={start}"
                )
            lookup[key] = int(sample_index)

        triplets: list[WorldArenaK8Triplet] = []
        for (row_index, start_a), a_sample_index in sorted(lookup.items()):
            start_b = int(start_a) + int(triplet_offset)
            start_c = int(start_b) + int(triplet_offset)
            b_sample_index = lookup.get((row_index, start_b))
            c_sample_index = lookup.get((row_index, start_c))
            if b_sample_index is None or c_sample_index is None:
                continue
            row = source.rows[row_index]
            episode = int(row["episode"])
            if row.get("split") != "train" or not 0 <= episode <= 35:
                raise RuntimeError(
                    "WorldArena rolling triplet crossed the train-only boundary: "
                    f"record={row.get('record_id')} episode={episode}"
                )
            if not 0 <= start_a < start_b < start_c <= 23:
                raise RuntimeError(
                    "WorldArena rolling triplet lies outside the 32-frame K8 gauge: "
                    f"start={start_a}/{start_b}/{start_c}"
                )
            triplets.append(
                WorldArenaK8Triplet(
                    a_sample_index=a_sample_index,
                    b_sample_index=b_sample_index,
                    c_sample_index=c_sample_index,
                    row_index=row_index,
                    start_a=start_a,
                    start_b=start_b,
                    start_c=start_c,
                    record_id=str(row["record_id"]),
                    episode=episode,
                    segment_start=int(row["segment_start"]),
                )
            )
        if not triplets:
            raise RuntimeError(
                "WorldArena triplet-K8 found no same-record x/x+8/x+16 chains"
            )
        self.base = base
        self.triplets = tuple(triplets)

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, triplet_index: int) -> dict[str, Any]:
        triplet = self.triplets[triplet_index]
        sample_a = self.base[triplet.a_sample_index]
        sample_b = self.base[triplet.b_sample_index]
        sample_c = self.base[triplet.c_sample_index]
        samples = (sample_a, sample_b, sample_c)
        expected_starts = (
            triplet.start_a,
            triplet.start_b,
            triplet.start_c,
        )
        if any(
            int(sample["start"]) != expected_start
            or str(sample["clip_id"]).split("@offset", 1)[0] != triplet.record_id
            for sample, expected_start in zip(samples, expected_starts)
        ):
            raise RuntimeError(
                "WorldArena triplet-K8 runtime identity differs from frozen topology"
            )
        return {
            "a": sample_a,
            "b": sample_b,
            "c": sample_c,
            "triplet_index": int(triplet_index),
            "triplet_start_a": triplet.start_a,
            "triplet_start_b": triplet.start_b,
            "triplet_start_c": triplet.start_c,
            "triplet_record_id": triplet.record_id,
            "triplet_episode": triplet.episode,
            "triplet_segment_start": triplet.segment_start,
        }


def build_worldarena_first_frame_rolling_context(
    s_a: torch.Tensor,
    pred_a: torch.Tensor,
) -> torch.Tensor:
    """Advance native tokens exactly like the WorldArena serving rollout.

    The evaluator advances ``state = cat(state[:, 8:], pred_tokens)`` after
    every K8 chunk.  Training must use A's retained native context, never B's
    ground-truth repeated anchor, or the rolling loss would leak its target.
    """

    if (
        s_a.ndim < 3
        or pred_a.ndim != s_a.ndim
        or int(s_a.shape[1]) != 16
        or int(pred_a.shape[1]) != 8
        or int(s_a.shape[0]) != int(pred_a.shape[0])
        or tuple(s_a.shape[2:]) != tuple(pred_a.shape[2:])
    ):
        raise ValueError(
            "WorldArena rolling context requires Bx16x... A context and "
            "Bx8x... A prediction with matching tails; "
            f"got s_a={tuple(s_a.shape)} pred_a={tuple(pred_a.shape)}"
        )
    return torch.cat((s_a[:, 8:].detach(), pred_a.detach()), dim=1).contiguous()
