from __future__ import annotations

import sys
import types
from collections import defaultdict

import numpy as np
import pytest
import torch

from wm3d_v3.data.manifest import OXEClipRecord
from wm3d_v3.data.window_dataset import OXEWindowDataset, WindowConfig
from wm3d_v3.losses import LossWeights, compute_losses
from wm3d_v3.training.train import (
    WeightedDistributedSampler,
    _forward_joint_model,
    action_policy_kwargs_from_targets,
    batch_to_device,
    build_dataset_sample_weights,
)


def _record(clip_id: str = "droid/clip0", dataset: str = "droid", n_frames: int = 5) -> OXEClipRecord:
    return OXEClipRecord(
        clip_id=clip_id,
        dataset=dataset,
        tar_path="unused.tar",
        pickle_member="unused.pkl",
        n_frames=n_frames,
        fps=5,
        robot="robot",
        task_text="task",
    )


def _write_cache(root, rec: OXEClipRecord, *, geom_extra: bool = True, real_progress: bool = False) -> None:
    cid = rec.clip_id.replace("/", "__")
    for sub in ("vggt_pooled", "vggt_geom", "actions", "qwen_taskemb"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    np.save(root / "vggt_pooled" / f"{cid}.npy", np.arange(rec.n_frames * 2 * 4, dtype=np.float16).reshape(rec.n_frames, 2, 4))
    np.save(root / "actions" / f"{cid}.npy", np.zeros((rec.n_frames, 7), dtype=np.float32))
    np.save(root / "qwen_taskemb" / f"{cid}.npy", np.ones(2048, dtype=np.float16))
    geom = {"depth": np.ones((rec.n_frames, 4, 4), dtype=np.float16)}
    if geom_extra:
        geom.update(
            {
                "world_points": np.ones((rec.n_frames, 4, 4, 3), dtype=np.float16) * 2,
                "world_points_conf": np.ones((rec.n_frames, 4, 4), dtype=np.float16) * 0.5,
                "pose_enc": np.ones((rec.n_frames, 9), dtype=np.float16) * 3,
                "depth_conf": np.ones((rec.n_frames, 4, 4), dtype=np.float16) * 0.75,
            }
        )
    if real_progress:
        geom.update(
            {
                "progress": np.linspace(0, 1, rec.n_frames, dtype=np.float32),
                "terminal_success": np.array(1.0, dtype=np.float32),
                "plausibility": np.array(0.25, dtype=np.float32),
            }
        )
    np.savez_compressed(root / "vggt_geom" / f"{cid}.npz", **geom)


def test_dataset_loads_future_vggt_geometry_targets(tmp_path):
    rec = _record(n_frames=5)
    _write_cache(tmp_path, rec, geom_extra=True)
    ds = OXEWindowDataset([rec], WindowConfig(T=2, k=2, stride=1, cache_root=tmp_path, load_rgb=False))

    sample = ds[0]

    assert sample["point_tgt"].shape == (2, 4, 4, 3)
    assert sample["point_conf_tgt"].shape == (2, 4, 4)
    assert sample["pose_geom_tgt"].shape == (2, 9)
    assert sample["depth_conf_tgt"].shape == (2, 4, 4)
    assert "progress_tgt" not in sample
    assert "terminal_success_tgt" not in sample
    assert "plausibility_tgt" not in sample


def test_dataset_can_require_geometry_extra(tmp_path):
    rec = _record(n_frames=5)
    _write_cache(tmp_path, rec, geom_extra=False)

    with pytest.raises(RuntimeError, match="require_geom_extra"):
        OXEWindowDataset(
            [rec],
            WindowConfig(T=2, k=2, stride=1, cache_root=tmp_path, load_rgb=False, require_geom_extra=True),
        )


def test_dataset_only_emits_real_progress_targets_unless_pseudo_allowed(tmp_path):
    rec = _record(n_frames=5)
    _write_cache(tmp_path, rec, geom_extra=False, real_progress=False)

    sample = OXEWindowDataset([rec], WindowConfig(T=2, k=2, stride=1, cache_root=tmp_path, load_rgb=False))[0]
    assert "progress_tgt" not in sample

    pseudo = OXEWindowDataset(
        [rec],
        WindowConfig(T=2, k=2, stride=1, cache_root=tmp_path, load_rgb=False, allow_pseudo_progress_targets=True),
    )[0]
    assert torch.allclose(pseudo["progress_tgt"], torch.tensor([0.5, 0.75]))


def test_geometry_losses_use_targets_and_confidence_masks():
    out = {
        "pred_tokens": torch.zeros(1, 2, 1, 1),
        "depth": torch.ones(1, 2, 2, 2),
        "point": torch.zeros(1, 2, 2, 2, 3),
        "pose_geom": torch.zeros(1, 2, 9),
        "pose": torch.zeros(1, 2, 6),
        "gripper_logit": torch.zeros(1, 2),
        "z_a": torch.zeros(1, 1),
    }
    tgt = {
        "s_tgt": torch.zeros(1, 2, 1, 1),
        "depth_tgt": torch.ones(1, 2, 2, 2),
        "point_tgt": torch.tensor(
            [[
                [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0], [1.0, 1.0, 0.5]]],
                [[[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]], [[0.0, 1.2, 0.0], [1.0, 1.0, 0.7]]],
            ]]
        ),
        "point_conf_tgt": torch.tensor([[[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]]]),
        "pose_geom_tgt": torch.ones(1, 2, 9),
        "action_tgt": torch.zeros(1, 2, 7),
    }

    losses = compute_losses(out, tgt, LossWeights(geom_point=1.0, geom_pose=1.0, action=0.0, idm_reg=0.0))

    assert losses["L_point"].item() > 0
    assert losses["L_pose_g"].item() > 0
    assert losses["geom_point_active"].item() == 1.0
    assert losses["geom_pose_active"].item() == 1.0


def test_batch_to_device_moves_geometry_and_rich_policy_state():
    batch = {
        "s_in": torch.zeros(1, 2, 1, 4),
        "c": torch.zeros(1, 2048),
        "action_tgt": torch.zeros(1, 2, 7),
        "action_tgt_norm": torch.zeros(1, 2, 6),
        "point_tgt": torch.zeros(1, 2, 4, 4, 3),
        "pose_geom_tgt": torch.zeros(1, 2, 9),
        "point_conf_tgt": torch.ones(1, 2, 4, 4),
        "lowdim_state": torch.ones(1, 5),
        "object_state": torch.ones(1, 3),
        "plan_state": torch.ones(1, 4),
        "action_history": torch.ones(1, 2, 7),
        "progress_tgt": torch.ones(1, 2),
    }

    _, _, _, _, tgt = batch_to_device(batch, torch.device("cpu"), k=2, direct_policy_only=True)

    for key in ("point_tgt", "pose_geom_tgt", "point_conf_tgt", "lowdim_state", "object_state", "plan_state", "action_history", "progress_tgt"):
        assert key in tgt
        assert tgt[key].device.type == "cpu"


def test_build_dataset_sample_weights_balances_by_dataset():
    records = [_record("a/0", "big"), _record("a/1", "big"), _record("b/0", "small")]

    class TinyDataset:
        def __init__(self):
            self.records = records
            self.index = [(0, 0), (0, 1), (1, 0), (2, 0)]

        def __len__(self):
            return len(self.index)

    weights = build_dataset_sample_weights(TinyDataset(), {"enabled": True, "dataset_weights": {"small": 4.0}})

    assert weights.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3, 4.0])



def test_geometry_loss_resizes_point_confidence_mask():
    out = {
        "pred_tokens": torch.zeros(1, 1, 1, 1),
        "depth": torch.ones(1, 1, 4, 4),
        "point": torch.zeros(1, 1, 4, 4, 3),
        "pose_geom": torch.zeros(1, 1, 9),
        "pose": torch.zeros(1, 1, 6),
        "gripper_logit": torch.zeros(1, 1),
        "z_a": torch.zeros(1, 1),
    }
    tgt = {
        "s_tgt": torch.zeros(1, 1, 1, 1),
        "depth_tgt": torch.ones(1, 1, 4, 4),
        "point_tgt": torch.tensor([[[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0], [1.0, 1.0, 0.5]]]]]),
        "point_conf_tgt": torch.ones(1, 1, 2, 2),
        "pose_geom_tgt": torch.ones(1, 1, 9),
        "action_tgt": torch.zeros(1, 1, 7),
    }

    losses = compute_losses(out, tgt, LossWeights(geom_point=1.0, geom_pose=1.0, action=0.0, idm_reg=0.0))

    assert losses["L_point"].item() > 0
    assert losses["geom_point_active"].item() == 1.0


def test_dataset_per_frame_terminal_and_plausibility_use_window_end(tmp_path):
    rec = _record(n_frames=5)
    _write_cache(tmp_path, rec, geom_extra=True, real_progress=False)
    cid = rec.clip_id.replace("/", "__")
    geom_path = tmp_path / "vggt_geom" / f"{cid}.npz"
    with np.load(geom_path) as d:
        payload = {k: np.array(d[k]) for k in d.files}
    payload["terminal_success"] = np.linspace(0.0, 0.4, rec.n_frames, dtype=np.float32)
    payload["plausibility"] = np.linspace(1.0, 0.6, rec.n_frames, dtype=np.float32)
    np.savez_compressed(geom_path, **payload)

    sample = OXEWindowDataset([rec], WindowConfig(T=2, k=2, stride=1, cache_root=tmp_path, load_rgb=False))[0]

    assert sample["terminal_success_tgt"].item() == pytest.approx(0.3)
    assert sample["plausibility_tgt"].item() == pytest.approx(0.7)



def test_dataset_loads_policy_state_and_synthesizes_action_history(tmp_path):
    rec = _record(n_frames=6)
    _write_cache(tmp_path, rec, geom_extra=True)
    cid = rec.clip_id.replace("/", "__")
    actions = np.arange(rec.n_frames * 7, dtype=np.float32).reshape(rec.n_frames, 7)
    np.save(tmp_path / "actions" / f"{cid}.npy", actions)
    geom_path = tmp_path / "vggt_geom" / f"{cid}.npz"
    with np.load(geom_path) as d:
        payload = {k: np.array(d[k]) for k in d.files}
    payload["lowdim_state"] = np.arange(rec.n_frames * 3, dtype=np.float32).reshape(rec.n_frames, 3)
    payload["object_state"] = np.array([1.0, 2.0], dtype=np.float32)
    payload["plan_state"] = np.arange(rec.n_frames * 2, dtype=np.float32).reshape(rec.n_frames, 2)
    np.savez_compressed(geom_path, **payload)

    ds = OXEWindowDataset(
        [rec],
        WindowConfig(
            T=3,
            k=1,
            stride=1,
            cache_root=tmp_path,
            load_rgb=False,
            load_policy_state=True,
            policy_lowdim_dim=4,
            policy_object_state_dim=3,
            policy_plan_state_dim=2,
            policy_action_history_len=2,
            policy_action_history_dim=7,
        ),
    )
    sample = ds[0]

    assert sample["lowdim_state"].tolist() == pytest.approx([6.0, 7.0, 8.0, 0.0])
    assert sample["object_state"].tolist() == pytest.approx([1.0, 2.0, 0.0])
    assert sample["plan_state"].tolist() == pytest.approx([4.0, 5.0])
    assert sample["action_history"].shape == (2, 7)
    assert np.allclose(sample["action_history"].numpy(), actions[1:3])


def test_dataset_require_policy_state_fails_fast_when_missing(tmp_path):
    rec = _record(n_frames=5)
    _write_cache(tmp_path, rec, geom_extra=True)

    with pytest.raises(RuntimeError, match="missing lowdim_state"):
        OXEWindowDataset(
            [rec],
            WindowConfig(
                T=2,
                k=2,
                stride=1,
                cache_root=tmp_path,
                load_rgb=False,
                load_policy_state=True,
                require_policy_state=True,
                policy_lowdim_dim=4,
            ),
        )



def test_vggt_encoder_emits_geometry_extras_with_fake_model(monkeypatch):
    class FakeVGGT:
        @classmethod
        def from_pretrained(cls, _model_name):
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

        def aggregator(self, images):
            b, t = images.shape[:2]
            base = torch.arange(b * t * 4 * 8, dtype=torch.float32, device=images.device).reshape(b, t, 4, 8)
            return [base], 0

        def depth_head(self, aggregated_tokens, images, patch_start_idx):
            b, t = images.shape[:2]
            depth = torch.ones(b, t, 2, 2, 1, device=images.device)
            conf = torch.ones(b, t, 2, 2, device=images.device) * 0.5
            return depth, conf

        def camera_head(self, aggregated_tokens):
            b, t = aggregated_tokens[-1].shape[:2]
            return [torch.ones(b, t, 9, device=aggregated_tokens[-1].device) * 3]

        def point_head(self, aggregated_tokens, images, patch_start_idx):
            b, t = images.shape[:2]
            points = torch.ones(b, t, 2, 2, 3, device=images.device) * 2
            conf = torch.ones(b, t, 2, 2, device=images.device) * 0.25
            return points, conf

    fake_vggt_pkg = types.ModuleType("vggt")
    fake_models_pkg = types.ModuleType("vggt.models")
    fake_model_mod = types.ModuleType("vggt.models.vggt")
    fake_model_mod.VGGT = FakeVGGT
    monkeypatch.setitem(sys.modules, "vggt", fake_vggt_pkg)
    monkeypatch.setitem(sys.modules, "vggt.models", fake_models_pkg)
    monkeypatch.setitem(sys.modules, "vggt.models.vggt", fake_model_mod)

    from wm3d_v3.encoders.vggt_encoder import VGGTEncoder

    enc = VGGTEncoder(device="cpu", token_grid=2, return_depth=True, return_geom_extra=True, dtype=torch.float32)
    out = enc(torch.zeros(1, 3, 3, 4, 4))

    assert out["pooled"].shape == (1, 3, 4, 8)
    assert out["depth"].shape == (1, 3, 2, 2)
    assert out["depth_conf"].shape == (1, 3, 2, 2)
    assert out["world_points"].shape == (1, 3, 2, 2, 3)
    assert out["world_points_conf"].shape == (1, 3, 2, 2)
    assert out["pose_enc"].shape == (1, 3, 9)
    assert "geom_extra_missing" not in out


def test_cache_oxe_geometry_completeness_and_payload_merge(tmp_path):
    from scripts.cache_oxe import GEOM_EXTRA_KEYS, _append_chunk, _existing_geom_payload, cache_complete, geom_extra_complete

    cid = "droid__clip0"
    for sub in ("vggt_pooled", "actions", "rgb_256", "qwen_taskemb", "vggt_geom"):
        (tmp_path / sub).mkdir()
    np.save(tmp_path / "vggt_pooled" / f"{cid}.npy", np.zeros((2, 64, 2048), dtype=np.float16))
    np.save(tmp_path / "actions" / f"{cid}.npy", np.zeros((2, 7), dtype=np.float32))
    np.save(tmp_path / "rgb_256" / f"{cid}.npy", np.zeros((2, 256, 256, 3), dtype=np.uint8))
    np.save(tmp_path / "qwen_taskemb" / f"{cid}.npy", np.zeros(2048, dtype=np.float16))
    geom_path = tmp_path / "vggt_geom" / f"{cid}.npz"
    np.savez_compressed(geom_path, depth=np.ones((2, 224, 224), dtype=np.float16))

    assert cache_complete(tmp_path, cid, need_qwen=True, need_rgb=True, need_geom=True, need_geom_extra=False)
    assert not geom_extra_complete(geom_path)
    assert not cache_complete(tmp_path, cid, need_qwen=True, need_rgb=True, need_geom=True, need_geom_extra=True)

    payload = _existing_geom_payload(geom_path)
    chunks = defaultdict(list)
    out = {key: torch.ones(1, 2, 224, 224, 3 if key == "world_points" else 1) for key in GEOM_EXTRA_KEYS}
    out["pose_enc"] = torch.ones(1, 2, 9)
    out["depth_conf"] = torch.ones(1, 2, 224, 224)
    out["world_points_conf"] = torch.ones(1, 2, 224, 224)
    for key in GEOM_EXTRA_KEYS:
        _append_chunk(chunks, out, key)
        payload[key] = np.concatenate(chunks[key], axis=0)
    np.savez_compressed(geom_path, **payload)

    assert geom_extra_complete(geom_path)
    assert cache_complete(tmp_path, cid, need_qwen=True, need_rgb=True, need_geom=True, need_geom_extra=True)


def test_dataset_skips_short_cache_windows_in_init(tmp_path):
    rec = _record(n_frames=5)
    _write_cache(tmp_path, rec, geom_extra=True)
    cid = rec.clip_id.replace("/", "__")
    np.save(tmp_path / "actions" / f"{cid}.npy", np.zeros((3, 7), dtype=np.float32))

    ds = OXEWindowDataset([rec], WindowConfig(T=2, k=2, stride=1, cache_root=tmp_path, load_rgb=False))

    assert len(ds) == 0


def test_dataset_require_progress_fails_fast_when_missing_or_invalid(tmp_path):
    rec = _record(n_frames=5)
    _write_cache(tmp_path, rec, geom_extra=True)

    with pytest.raises(RuntimeError, match="require_progress"):
        OXEWindowDataset([rec], WindowConfig(T=2, k=2, stride=1, cache_root=tmp_path, load_rgb=False, require_progress=True))

    cid = rec.clip_id.replace("/", "__")
    geom_path = tmp_path / "vggt_geom" / f"{cid}.npz"
    with np.load(geom_path) as d:
        payload = {k: np.array(d[k]) for k in d.files}
    payload["progress"] = np.array([0.0, 0.2, 1.2, 0.5, 1.0], dtype=np.float32)
    np.savez_compressed(geom_path, **payload)

    with pytest.raises(RuntimeError, match="outside"):
        OXEWindowDataset([rec], WindowConfig(T=2, k=2, stride=1, cache_root=tmp_path, load_rgb=False, require_progress=True))


def test_weighted_distributed_sampler_caps_without_replacement_evenly_across_ranks():
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.double)
    samplers = [
        WeightedDistributedSampler(
            weights,
            num_replicas=4,
            rank=rank,
            replacement=False,
            num_samples=4,
            seed=7,
        )
        for rank in range(4)
    ]

    sampled_by_rank = [list(iter(sampler)) for sampler in samplers]

    assert [len(sampled) for sampled in sampled_by_rank] == [1, 1, 1, 1]
    assert all(sampler.total_size == 4 for sampler in samplers)
    assert len({idx for sampled in sampled_by_rank for idx in sampled}) == 4


def test_weighted_distributed_sampler_without_replacement_rejects_too_few_positive_samples():
    with pytest.raises(ValueError, match="num_replicas positive"):
        WeightedDistributedSampler(
            torch.tensor([1.0, 2.0], dtype=torch.double),
            num_replicas=4,
            rank=0,
            replacement=False,
            seed=7,
        )


def test_full_model_forward_receives_rich_policy_state():
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kwargs = None

        def forward(self, s, c, **kwargs):
            self.kwargs = kwargs
            return {"pred_tokens": s}

    tgt = {
        "lowdim_state": torch.ones(1, 3),
        "object_state": torch.ones(1, 2),
        "plan_state": torch.ones(1, 4),
        "action_history": torch.ones(1, 2, 7),
        "progress_tgt": torch.tensor([[0.25, 0.5]]),
    }
    model = FakeModel()

    _forward_joint_model(
        model,
        torch.zeros(1, 2, 3, 4),
        torch.zeros(1, 16),
        action_cond=torch.zeros(1, 2, 7),
        context_rgb=None,
        pixel=False,
        bridging=False,
        policy_kwargs=action_policy_kwargs_from_targets(tgt),
    )

    for key in ("lowdim_state", "object_state", "plan_state", "action_history", "progress_state"):
        assert key in model.kwargs
    assert torch.allclose(model.kwargs["progress_state"], torch.tensor([[0.25]]))


def test_cache_helper_rejects_partial_shape_and_length_mismatch(tmp_path):
    from scripts.cache_geom_utils import atomic_savez_compressed, validate_geom_npz
    from scripts.cache_oxe import cache_complete

    cid = "droid__clip0"
    for sub in ("vggt_pooled", "actions", "rgb_256", "qwen_taskemb", "vggt_geom"):
        (tmp_path / sub).mkdir()
    np.save(tmp_path / "vggt_pooled" / f"{cid}.npy", np.zeros((2, 64, 2048), dtype=np.float16))
    np.save(tmp_path / "actions" / f"{cid}.npy", np.zeros((2, 7), dtype=np.float32))
    np.save(tmp_path / "rgb_256" / f"{cid}.npy", np.zeros((2, 256, 256, 3), dtype=np.uint8))
    np.save(tmp_path / "qwen_taskemb" / f"{cid}.npy", np.zeros(2048, dtype=np.float16))
    geom_path = tmp_path / "vggt_geom" / f"{cid}.npz"
    atomic_savez_compressed(
        geom_path,
        depth=np.ones((2, 224, 224), dtype=np.float16),
        world_points=np.ones((1, 224, 224, 3), dtype=np.float16),
        world_points_conf=np.ones((2, 224, 224), dtype=np.float16),
        pose_enc=np.ones((2, 9), dtype=np.float16),
        depth_conf=np.ones((2, 224, 224), dtype=np.float16),
    )

    assert not validate_geom_npz(geom_path, expected_frames=2, require_geom_extra=True)
    assert not cache_complete(tmp_path, cid, need_qwen=True, need_rgb=True, need_geom=True, need_geom_extra=True)
    assert not any(geom_path.parent.glob("*.tmp.*.npz"))

def test_cache_helper_rejects_malformed_same_length_arrays(tmp_path):
    from scripts.cache_oxe import cache_complete

    cid = "droid__clip0"
    for sub in ("vggt_pooled", "actions", "rgb_256", "qwen_taskemb", "vggt_geom"):
        (tmp_path / sub).mkdir()
    np.save(tmp_path / "vggt_pooled" / f"{cid}.npy", np.zeros((2, 64, 2048), dtype=np.float16))
    np.save(tmp_path / "actions" / f"{cid}.npy", np.zeros((2, 1), dtype=np.float32))
    np.save(tmp_path / "rgb_256" / f"{cid}.npy", np.zeros((2, 1, 1, 3), dtype=np.uint8))
    np.save(tmp_path / "qwen_taskemb" / f"{cid}.npy", np.zeros(1, dtype=np.float16))
    np.savez_compressed(tmp_path / "vggt_geom" / f"{cid}.npz", depth=np.ones((2, 224, 224), dtype=np.float16))

    assert not cache_complete(tmp_path, cid, need_qwen=True, need_rgb=True, need_geom=True, need_geom_extra=False)

    np.save(tmp_path / "actions" / f"{cid}.npy", np.zeros((2, 7), dtype=np.float32))
    assert not cache_complete(tmp_path, cid, need_qwen=True, need_rgb=True, need_geom=True, need_geom_extra=False)

    np.save(tmp_path / "rgb_256" / f"{cid}.npy", np.zeros((2, 256, 256, 3), dtype=np.uint8))
    assert not cache_complete(tmp_path, cid, need_qwen=True, need_rgb=True, need_geom=True, need_geom_extra=False)

    np.save(tmp_path / "qwen_taskemb" / f"{cid}.npy", np.zeros(2048, dtype=np.float16))
    assert cache_complete(tmp_path, cid, need_qwen=True, need_rgb=True, need_geom=True, need_geom_extra=False)

    np.save(tmp_path / "vggt_pooled" / f"{cid}.npy", np.zeros((2, 4, 8), dtype=np.float16))
    assert not cache_complete(tmp_path, cid, need_qwen=True, need_rgb=True, need_geom=True, need_geom_extra=False)

    np.save(tmp_path / "vggt_pooled" / f"{cid}.npy", np.zeros((2, 64, 2048), dtype=np.float16))
    np.savez_compressed(tmp_path / "vggt_geom" / f"{cid}.npz", depth=np.ones((2, 2, 2), dtype=np.float16))
    assert not cache_complete(tmp_path, cid, need_qwen=True, need_rgb=True, need_geom=True, need_geom_extra=False)


def test_dataset_uses_geom_for_policy_state_without_emitting_geom_targets_when_load_geom_false(tmp_path):
    rec = _record(n_frames=6)
    _write_cache(tmp_path, rec, geom_extra=True)
    cid = rec.clip_id.replace("/", "__")
    geom_path = tmp_path / "vggt_geom" / f"{cid}.npz"
    with np.load(geom_path) as d:
        payload = {k: np.array(d[k]) for k in d.files}
    payload["lowdim_state"] = np.arange(rec.n_frames * 3, dtype=np.float32).reshape(rec.n_frames, 3)
    np.savez_compressed(geom_path, **payload)

    sample = OXEWindowDataset(
        [rec],
        WindowConfig(
            T=3,
            k=1,
            stride=1,
            cache_root=tmp_path,
            load_rgb=False,
            load_geom=False,
            load_policy_state=True,
            policy_lowdim_dim=3,
        ),
    )[0]

    assert sample["lowdim_state"].tolist() == pytest.approx([6.0, 7.0, 8.0])
    assert "depth_tgt" not in sample
    assert "point_tgt" not in sample
    assert "pose_geom_tgt" not in sample

def test_manifest_builders_reject_malformed_existing_cache(tmp_path):
    from scripts.build_oxe_trainable_manifest import has_required_cache
    from scripts.build_stage1_oxe_droid_manifest import cache_ready

    cid = "droid__clip0"
    clip_id = "droid/clip0"
    for sub in ("vggt_pooled", "actions", "rgb_256", "qwen_taskemb", "vggt_geom"):
        (tmp_path / sub).mkdir()
    np.save(tmp_path / "vggt_pooled" / f"{cid}.npy", np.zeros((2, 64, 2048), dtype=np.float16))
    np.save(tmp_path / "actions" / f"{cid}.npy", np.zeros((2, 1), dtype=np.float32))
    np.save(tmp_path / "rgb_256" / f"{cid}.npy", np.zeros((2, 1, 1, 3), dtype=np.uint8))
    np.save(tmp_path / "qwen_taskemb" / f"{cid}.npy", np.zeros(1, dtype=np.float16))
    np.savez_compressed(tmp_path / "vggt_geom" / f"{cid}.npz", depth=np.ones((2, 224, 224), dtype=np.float16))

    assert not has_required_cache(
        tmp_path,
        clip_id,
        require_policy=True,
        require_qwen=True,
        require_rgb=True,
        require_geom=True,
        expected_frames=2,
    )
    assert not cache_ready({"clip_id": clip_id, "n_frames": 2}, tmp_path, require_task_emb=True, min_frames=2)

    np.save(tmp_path / "actions" / f"{cid}.npy", np.zeros((2, 7), dtype=np.float32))
    np.save(tmp_path / "rgb_256" / f"{cid}.npy", np.zeros((2, 256, 256, 3), dtype=np.uint8))
    np.save(tmp_path / "qwen_taskemb" / f"{cid}.npy", np.zeros(2048, dtype=np.float16))

    assert has_required_cache(
        tmp_path,
        clip_id,
        require_policy=True,
        require_qwen=True,
        require_rgb=True,
        require_geom=True,
        expected_frames=2,
    )
    assert cache_ready({"clip_id": clip_id, "n_frames": 2}, tmp_path, require_task_emb=True, min_frames=2)

    np.save(tmp_path / "vggt_pooled" / f"{cid}.npy", np.zeros((2, 4, 8), dtype=np.float16))
    assert not has_required_cache(
        tmp_path,
        clip_id,
        require_policy=True,
        require_qwen=True,
        require_rgb=True,
        require_geom=True,
        expected_frames=2,
    )
    assert not cache_ready({"clip_id": clip_id, "n_frames": 2}, tmp_path, require_task_emb=True, min_frames=2)

    np.save(tmp_path / "vggt_pooled" / f"{cid}.npy", np.zeros((2, 64, 2048), dtype=np.float16))
    np.savez_compressed(tmp_path / "vggt_geom" / f"{cid}.npz", depth=np.ones((2, 2, 2), dtype=np.float16))
    assert not has_required_cache(
        tmp_path,
        clip_id,
        require_policy=True,
        require_qwen=True,
        require_rgb=True,
        require_geom=True,
        expected_frames=2,
    )
    assert not cache_ready({"clip_id": clip_id, "n_frames": 2}, tmp_path, require_task_emb=True, min_frames=2)
