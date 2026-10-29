"""LIBERO HDF5 adapter for the current FoundationWorldModel grouped ABI.

The 20 Hz controller clock is recorded in each HDF5 env_args. World frames
are selected every four actual commands; no action resampling is performed.
Only current + K future images are consumed by the native model. Missing
visual history and the third camera remain masked, rather than repeated.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import OrderedDict
import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

from .benchmark_catalog import binding, publish, read_rows
from .source_adapters import _rotvec_to_matrix

SCHEMA = "wm3d_libero_native_normalization_v1"
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")


def canonical_robot(group):
    """Apply recorded OSC scaling and the same physical state ABI as serving."""
    raw = np.asarray(group["actions"], dtype=np.float64)
    position = np.asarray(group["obs/ee_pos"], dtype=np.float64)
    orientation = np.asarray(group["obs/ee_ori"], dtype=np.float64)
    gripper = np.asarray(group["obs/gripper_states"], dtype=np.float64)
    if raw.shape[1:] != (7,) or not all(
        np.isfinite(v).all() for v in (raw, position, orientation, gripper)
    ):
        raise ValueError("LIBERO robot arrays are not finite native measurements")
    if not np.isin(raw[:, 6], [-1.0, 1.0]).all():
        raise ValueError("unresolved non-binary signed LIBERO gripper command")
    action = raw.copy()
    action[:, :6] = np.clip(raw[:, :6], -1, 1) * [.05, .05, .05, .5, .5, .5]
    action[:, 6] = (raw[:, 6] + 1) / 2
    # LIBERO create_dataset.py records quat2axisangle, not Euler angles.
    rotation = _rotvec_to_matrix(orientation)
    rotation6d = np.swapaxes(rotation[:, :, :2], 1, 2).reshape(len(raw), 6)
    close = 1 - np.clip(np.abs(gripper[:, 0] - gripper[:, 1]) / .08, 0, 1)
    state = np.column_stack((position, rotation6d, close))
    if state.shape != (len(raw), 10):
        raise ValueError("LIBERO state cardinality changed")
    return action.astype(np.float32), state.astype(np.float32), int(
        np.count_nonzero(np.abs(raw[:, :6]) > 1)
    )


class Moments:
    def __init__(self, width):
        self.count = 0
        self.total = np.zeros(width, dtype=np.float64)
        self.square = np.zeros(width, dtype=np.float64)

    def add(self, value):
        value = value.astype(np.float64)
        self.count += len(value)
        self.total += value.sum(0)
        self.square += np.square(value).sum(0)

    def export(self):
        mean = self.total / self.count
        scale = np.sqrt(np.maximum(self.square / self.count - mean**2, 0))
        return dict(count=self.count, offset=mean.tolist(),
                    scale=np.maximum(scale, 1e-6).tolist())


def fit_suite(catalog, destination):
    rows = list(read_rows(catalog))
    if len(rows) != 500 or set(r["split"] for r in rows) != {"train"}:
        raise ValueError("normalization must use the 500 training demos only")
    action_moments, state_moments = Moments(7), Moments(10)
    files = {}
    clipped = 0
    for row in rows:
        path = row["file"]
        if binding(path) != row["file_binding"]:
            raise ValueError("LIBERO source changed before normalization")
        with h5py.File(path, "r") as h:
            action, state, n = canonical_robot(h[row["hdf5_group"]])
        if binding(path) != row["file_binding"]:
            raise ValueError("LIBERO source changed during normalization")
        action_moments.add(action)
        state_moments.add(state)
        clipped += n
        files[path] = row["file_binding"]
    action, state = action_moments.export(), state_moments.export()
    # Preserve binary gripper semantics, as in the pretraining normalizer.
    action["offset"][6], action["scale"][6] = 0.0, 1.0
    state["offset"][9], state["scale"][9] = 0.0, 1.0
    value = dict(schema=SCHEMA, suite=rows[0]["suite"], train_demos=500,
        fitting_split="train", action=action, state=state,
        catalog=str(catalog), catalog_binding=binding(catalog), files=files,
        input_commands_clipped_by_recorded_controller=clipped,
        action_coordinates="base-frame metres / axis-angle radians / close01",
        state_coordinates="xyz metres / rotation6d columns / close01",
        state_axis_angle_provenance="/data/Minko/benchmarks/LIBERO/scripts/create_dataset.py:207",
        controller_rate_hz=20, world_frame_stride=4,
        state_gripper_open_width_m=.08, state_gripper_closed_width_m=0.0,
        source_images="opengl as stored; no implicit rotation or flip",
        third_camera="masked", future_frames_are_targets_only=True)
    publish(destination, value)
    return value


class LiberoNativeDataset(Dataset):
    """Full-suite deterministic window indices; bounded HDF5/robot LRU per worker."""
    requires_main_process = False

    def __init__(self, catalog, normalization, *, file_bindings=None, catalog_binding=None):
        self.catalog = Path(catalog)
        self.rows = list(read_rows(self.catalog))
        self.stats = json.loads(Path(normalization).read_text())
        expected_catalog = self.stats["catalog_binding"] if catalog_binding is None else catalog_binding
        current_catalog = binding(self.catalog)
        if self.stats["schema"] != SCHEMA or any(current_catalog[k] != expected_catalog[k] for k in ("inode", "size_bytes", "mtime_ns", "ctime_ns")):
            raise ValueError("LIBERO normalization/catalog identity changed")
        self.suite = self.stats["suite"]
        if len(self.rows) != 500 or any(r["suite"] != self.suite for r in self.rows):
            raise ValueError("incomplete or mixed LIBERO suite")
        self.files = dict(self.stats["files"] if file_bindings is None else file_bindings)
        if set(self.files) != set(self.stats["files"]):
            raise ValueError("deployed asset binding must cover the same 10 task files")
        self.ends = []
        spans = []
        cursor = 0
        for row in self.rows:
            count = row["full_future_window_starts"]
            spans.append((cursor, cursor + count))
            cursor += count
            self.ends.append(cursor)
        self.source_names = (self.suite,)
        self.source_spans = {self.suite: (0, cursor)}
        self.source_episode_spans = {self.suite: tuple(spans)}
        self.source_sampling_weights = {self.suite: 1}
        self.profile = SimpleNamespace(source_weights=self.source_sampling_weights,
            source_order=self.source_names, cache_representation=dict(
                spatial_tokens=64, token_dim=2048, num_views=3, rgb_size=518))
        self._open = OrderedDict()
        self._robot = OrderedDict()
        self._owner_pid = os.getpid()
        self.action_offset = np.asarray(self.stats["action"]["offset"], np.float32)
        self.action_scale = np.asarray(self.stats["action"]["scale"], np.float32)
        self.state_offset = np.asarray(self.stats["state"]["offset"], np.float32)
        self.state_scale = np.asarray(self.stats["state"]["scale"], np.float32)

    def __len__(self):
        return self.ends[-1]

    def _group(self, row):
        if os.getpid() != self._owner_pid:
            self._open = OrderedDict()
            self._robot = OrderedDict()
            self._owner_pid = os.getpid()
        path = row["file"]
        if path not in self._open:
            if binding(path) != self.files[path]:
                raise ValueError("LIBERO raw asset identity changed")
            self._open[path] = h5py.File(path, "r")
        self._open.move_to_end(path)
        while len(self._open) > 2:
            old, handle = self._open.popitem(last=False)
            handle.close()
            if binding(old) != self.files[old]:
                raise ValueError("LIBERO raw asset changed during read")
        return self._open[path][row["hdf5_group"]]

    def __getitem__(self, index):
        from scripts.data.run_cache_worker import _view_batch
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect_right(self.ends, index)
        row = self.rows[episode]
        start = 0 if episode == 0 else self.ends[episode-1]
        anchor = 1 + index - start
        group = self._group(row)
        if episode not in self._robot:
            action, state, _ = canonical_robot(group)
            self._robot[episode] = ((action-self.action_offset)/self.action_scale,
                                    (state-self.state_offset)/self.state_scale)
        self._robot.move_to_end(episode)
        while len(self._robot) > 8:
            self._robot.popitem(last=False)
        action, state = self._robot[episode]
        selected = anchor + np.arange(9, dtype=np.int64)*4
        decoded = {name: SimpleNamespace(frames=np.asarray(group[key][selected]))
                   for name, key in zip(("front", "left"), row["camera_keys"])}
        images, masks = _view_batch(decoded=decoded, slots=("front", "left", "right"),
                                    input_size=518)
        rgb = torch.zeros(24, 3, 3, 518, 518, dtype=torch.uint8)
        view = torch.zeros(24, 3, dtype=torch.bool)
        rgb[15:] = images.mul(255).round().clamp_(0,255).to(torch.uint8)
        view[15:] = masks
        keys = torch.full((24,), -1, dtype=torch.long)
        keys[15:] = torch.from_numpy((episode << 32) + selected)
        result = dict(direct_rgb_uint8=rgb, direct_view_mask=view,
            direct_frame_keys=keys, source_id=torch.tensor(0, dtype=torch.long),
            sample_index=torch.tensor(index, dtype=torch.long), task_text=row["instruction"],
            embodiment_ids=torch.tensor(2, dtype=torch.long),
            action_group_ids=torch.tensor([12,0,0], dtype=torch.long),
            action_group_mask=torch.tensor([True,False,False]),
            action_semantic_ids=torch.zeros(3,14,dtype=torch.long),
            state_semantic_ids=torch.zeros(3,10,dtype=torch.long),
            action_normalization_offset=torch.zeros(3,14),
            action_normalization_scale=torch.ones(3,14),
            state_normalization_offset=torch.zeros(3,10),
            state_normalization_scale=torch.ones(3,10),
            current_state_values=torch.zeros(3,10),
            current_state_mask=torch.zeros(3,10,dtype=torch.bool),
            history_world_boundaries_dt=torch.arange(-16,1,dtype=torch.float32)/5,
            future_world_boundaries_dt=torch.arange(9,dtype=torch.float32)/5,
            policy_query_dt=torch.zeros(3,32),
            policy_query_mask=torch.zeros(3,32,dtype=torch.bool),
            target_fine_action=torch.zeros(3,32,14),
            target_fine_action_mask=torch.zeros(3,32,14,dtype=torch.bool))
        result["action_semantic_ids"][0,:7] = torch.tensor([1,1,1,3,3,3,5])
        result["state_semantic_ids"][0] = torch.tensor([1,1,1,2,2,2,2,2,2,4])
        for name, value in (("action_normalization_offset",self.action_offset),
                            ("action_normalization_scale",self.action_scale),
                            ("state_normalization_offset",self.state_offset),
                            ("state_normalization_scale",self.state_scale)):
            result[name][0,:len(value)] = torch.from_numpy(value)
        result["current_state_values"][0] = torch.from_numpy(state[anchor])
        result["current_state_mask"][0] = True
        for prefix, length, first in (("history",16,anchor-64), ("future_factual",8,anchor)):
            values = torch.zeros(length,3,4,14)
            mask = torch.zeros_like(values,dtype=torch.bool)
            sample_mask = torch.zeros(length,3,4,dtype=torch.bool)
            dt = torch.zeros(length,3,4)
            native_indices = first + np.arange(length*4).reshape(length,4)
            active = native_indices >= 0
            safe_indices = np.maximum(native_indices,0)
            values[:,0,:,:7] = torch.from_numpy(action[safe_indices]) * torch.from_numpy(active[...,None])
            mask[:,0,:,:7] = torch.from_numpy(np.broadcast_to(active[...,None],(length,4,7)).copy())
            sample_mask[:,0] = torch.from_numpy(active)
            dt[:,0] = torch.arange(4,dtype=torch.float32)[None,:]/20
            result[prefix+"_fine_action_values"] = values
            result[prefix+"_fine_action_mask"] = mask
            result[prefix+"_fine_action_dt"] = dt
            result[prefix+"_fine_sample_mask"] = sample_mask
        result["policy_query_dt"][0] = torch.arange(32,dtype=torch.float32)/20
        result["policy_query_mask"][0] = True
        result["target_fine_action"][0,:,:7] = torch.from_numpy(action[anchor:anchor+32])
        result["target_fine_action_mask"][0,:,:7] = True
        if binding(row["file"]) != self.files[row["file"]]:
            raise ValueError("LIBERO source changed during window read")
        return result

    def close(self):
        for path, handle in self._open.items():
            handle.close()
            if binding(path) != self.files[path]:
                raise ValueError("LIBERO raw source changed while open")
        self._open.clear()
        self._robot.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if Path.cwd() != Path("/data/Minko") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("numeric preparation runs on43 CPU from /data/Minko")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    from .foundation_bridge import FoundationBridgeConfig, build_foundation_sample
    from wm3d.training.pretrain import _collate_and_trim
    summary = {}
    for suite in SUITES:
        catalog = args.catalog_root/suite/"train.jsonl"
        normalization = args.output/(suite+"_normalization.json")
        stats = fit_suite(catalog, normalization)
        dataset = LiberoNativeDataset(catalog, normalization)
        # Real first/last and two interior windows exercise early-history masking
        # and the complete 32-command horizon, using actual decoded HDF5 images.
        indices = [0, len(dataset)//3, len(dataset)//2, len(dataset)-1]
        checked = []
        for i in indices:
            raw = _collate_and_trim([dataset[i]])
            batch = build_foundation_sample(raw, FoundationBridgeConfig())
            if batch["action"].shape != (1,32,42) or batch["video"].shape != (1,3,9,384,320):
                raise ValueError("actual foundation bridge shape changed")
            if not bool(torch.isfinite(batch["video"]).all()) or not bool(torch.isfinite(batch["action"]).all()):
                raise ValueError("non-finite real benchmark batch")
            if not bool((batch["history_action_time_s"][batch["history_action_element_mask"]] < 0).all()):
                raise ValueError("LIBERO history includes a future command")
            if batch["action_dim_is_pad"][0,:7].any() or not batch["action_dim_is_pad"][0,7:].all():
                raise ValueError("missing Panda action dimensions became valid")
            checked.append(dict(index=i, active_future_commands=int((~batch["action_is_pad"]).sum()),
                                active_history_elements=int(batch["history_action_element_mask"].sum())))
        dataset.close()
        summary[suite] = dict(training_demos=stats["train_demos"], train_action_rows=stats["action"]["count"],
            valid_windows=len(dataset), normalization=str(normalization), actual_windows_checked=checked,
            actual_current_and_future_RGB_decoded=True, foundation_bridge_passed=True,
            GPU_model_or_teacher_not_executed=True)
        print(json.dumps(dict(suite=suite, native_CPU_data_ready=True)),flush=True)
    publish(args.output/"COMPLETE.json",dict(schema="wm3d_libero_native_CPU_materialization_v1",
        suites=summary, actual_hdf5_robot_arrays_and_pixels_used=True,
        normalization_train_only=True, model_training_started=False,
        no_copied_raw_payloads=True, no_payload_hashes=True))
if __name__ == "__main__":
    main()
