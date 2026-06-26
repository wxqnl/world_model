from __future__ import annotations

import tarfile

import numpy as np
import pytest
import torch

from wm3d_v3.data.manifest import OXEClipRecord
from wm3d_v3.data.window_dataset import OXEWindowDataset, WindowConfig
from wm3d_v3.losses import LossWeights, compute_losses


def _record(n_frames: int = 5) -> OXEClipRecord:
    return OXEClipRecord(
        clip_id="droid/native3d0",
        dataset="droid",
        tar_path="",
        pickle_member="",
        n_frames=n_frames,
        fps=5,
        robot="robot",
        task_text="pick up object",
    )


def _safe(cid: str) -> str:
    return cid.replace("/", "__")


def _write_base_cache(root, rec: OXEClipRecord) -> None:
    cid = _safe(rec.clip_id)
    for sub in ("vggt_geom", "actions", "qwen_taskemb"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    np.save(root / "actions" / f"{cid}.npy", np.zeros((rec.n_frames, 7), dtype=np.float32))
    np.save(root / "qwen_taskemb" / f"{cid}.npy", np.ones(2048, dtype=np.float16))
    np.savez_compressed(root / "vggt_geom" / f"{cid}.npz", depth=np.ones((rec.n_frames, 4, 4), dtype=np.float16))


def _write_window_cache(root, rec: OXEClipRecord, *, complete: bool = True) -> None:
    cid = _safe(rec.clip_id)
    out_dir = root / "vggt_window_geom_p64_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "pooled": np.arange(4 * 2 * 3, dtype=np.float16).reshape(4, 2, 3),
        "depth": np.ones((2, 4, 4), dtype=np.float16) * 2,
        "point": np.ones((2, 4, 4, 3), dtype=np.float16),
        "point_conf": np.ones((2, 4, 4), dtype=np.float16),
        "depth_conf": np.ones((2, 4, 4), dtype=np.float16),
    }
    if complete:
        payload["pose"] = np.ones((2, 9), dtype=np.float16)
        payload["pose_conf"] = np.ones((2,), dtype=np.float16)
    np.savez_compressed(out_dir / f"{cid}__start_000000.npz", **payload)


def _pack_window_cache_as_tar_shard(root, rec: OXEClipRecord) -> tuple[object, object]:
    cid = _safe(rec.clip_id)
    name = f"{cid}__start_000000.npz"
    src = root / "vggt_window_geom_p64_test" / name
    shard_root = root / "window_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    shard = shard_root / "shard_000.tar"
    with tarfile.open(shard, "w") as tf:
        tf.add(src, arcname=name)
    index = shard_root / "index.tsv"
    with tarfile.open(shard, "r") as tf:
        member = tf.getmember(name)
        index.write_text(f"{name}\t{shard.name}\t{member.offset_data}\t{member.size}\n")
    src.unlink()
    return index, shard_root


def test_window_tokens_and_geometry_are_loaded_from_same_window_cache(tmp_path):
    rec = _record()
    _write_base_cache(tmp_path, rec)
    _write_window_cache(tmp_path, rec)

    ds = OXEWindowDataset(
        [rec],
        WindowConfig(
            T=2,
            k=2,
            stride=1,
            cache_root=tmp_path,
            load_rgb=False,
            load_geom=True,
            load_geom_extra=True,
            require_geom_extra=True,
            use_window_tokens=True,
            window_geom_subdir="vggt_window_geom_p64_test",
        ),
    )
    sample = ds[0]

    assert len(ds) == 1
    assert sample["s_in"].shape == (2, 2, 3)
    assert sample["s_tgt"].shape == (2, 2, 3)
    assert sample["s_in"][0, 0, 0].item() == 0
    assert sample["s_tgt"][0, 0, 0].item() == 12
    assert sample["point_tgt"].shape == (2, 4, 4, 3)
    assert sample["pose_geom_tgt"].shape == (2, 9)
    assert sample["pose_geom_conf_tgt"].shape == (2,)
    assert sample["depth_tgt"].shape == (2, 4, 4)
    assert sample["depth_tgt"].mean().item() == pytest.approx(2.0)


def test_window_tokens_and_geometry_load_from_tar_shard(tmp_path):
    rec = _record()
    _write_base_cache(tmp_path, rec)
    _write_window_cache(tmp_path, rec)
    index, shard_root = _pack_window_cache_as_tar_shard(tmp_path, rec)

    ds = OXEWindowDataset(
        [rec],
        WindowConfig(
            T=2,
            k=2,
            stride=1,
            cache_root=tmp_path,
            load_rgb=False,
            load_geom=True,
            load_geom_extra=True,
            require_geom_extra=True,
            use_window_tokens=True,
            window_geom_subdir="vggt_window_geom_p64_test",
            window_geom_shard_index=index,
            window_geom_shard_root=shard_root,
        ),
    )
    sample = ds[0]

    assert len(ds) == 1
    assert sample["s_in"][0, 0, 0].item() == 0
    assert sample["s_tgt"][0, 0, 0].item() == 12
    assert sample["point_tgt"].shape == (2, 4, 4, 3)
    assert sample["pose_geom_tgt"].shape == (2, 9)


def test_require_window_geometry_rejects_incomplete_npz(tmp_path):
    rec = _record()
    _write_base_cache(tmp_path, rec)
    _write_window_cache(tmp_path, rec, complete=False)

    with pytest.raises(RuntimeError, match="no usable window geom cache"):
        OXEWindowDataset(
            [rec],
            WindowConfig(
                T=2,
                k=2,
                stride=1,
                cache_root=tmp_path,
                load_rgb=False,
                load_geom=True,
                load_geom_extra=True,
                require_geom_extra=True,
                use_window_tokens=True,
                window_geom_subdir="vggt_window_geom_p64_test",
            ),
        )


def _minimal_out_tgt():
    out = {
        "pred_tokens": torch.zeros(1, 2, 1, 1),
        "depth": torch.ones(1, 2, 2, 2),
        "pose": torch.zeros(1, 2, 6),
        "gripper_logit": torch.zeros(1, 2),
        "z_a": torch.zeros(1, 1),
    }
    tgt = {
        "s_tgt": torch.zeros(1, 2, 1, 1),
        "depth_tgt": torch.ones(1, 2, 2, 2),
        "action_tgt": torch.zeros(1, 2, 7),
    }
    return out, tgt


def test_point_loss_fails_fast_when_weight_enabled_without_targets():
    out, tgt = _minimal_out_tgt()

    with pytest.raises(ValueError, match="geom_point/point_temporal"):
        compute_losses(out, tgt, LossWeights(geom_depth=0.0, geom_point=1.0, action=0.0, idm_reg=0.0))


def test_point_loss_is_translation_invariant_after_relative_normalization():
    out, tgt = _minimal_out_tgt()
    base = torch.tensor(
        [[
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0], [1.0, 1.0, 0.5]]],
            [[[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]], [[0.0, 1.2, 0.0], [1.0, 1.0, 0.7]]],
        ]]
    )
    shift = torch.tensor([100.0, -50.0, 25.0])
    out["point"] = base + shift
    tgt["point_tgt"] = base
    tgt["point_conf_tgt"] = torch.ones(1, 2, 2, 2)

    losses = compute_losses(
        out,
        tgt,
        LossWeights(geom_depth=0.0, geom_point=1.0, point_temporal=0.0, action=0.0, idm_reg=0.0),
    )

    assert losses["L_point"].item() < 1e-6


def test_point_loss_is_scale_and_translation_invariant_after_relative_normalization():
    out, tgt = _minimal_out_tgt()
    base = torch.tensor(
        [[
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0], [1.0, 1.0, 0.5]]],
            [[[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]], [[0.0, 1.2, 0.0], [1.0, 1.0, 0.7]]],
        ]]
    )
    shift = torch.tensor([-12.0, 8.0, 3.0])
    out["point"] = base * 3.5 + shift
    tgt["point_tgt"] = base
    tgt["point_conf_tgt"] = torch.ones(1, 2, 2, 2)

    losses = compute_losses(
        out,
        tgt,
        LossWeights(geom_depth=0.0, geom_point=1.0, point_temporal=0.0, action=0.0, idm_reg=0.0),
    )

    assert losses["L_point"].item() < 1e-6


def test_depth_temporal_static_mask_ignores_rgb_motion_region():
    out, tgt = _minimal_out_tgt()
    out["depth"] = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], [[2.0, 1.0], [1.0, 1.0]]]])
    tgt["depth_tgt"] = torch.ones(1, 2, 2, 2)
    rgb = torch.zeros(1, 2, 3, 2, 2)
    rgb[:, 1, :, 0, 0] = 1.0
    tgt["rgb_tgt_p"] = rgb

    losses = compute_losses(
        out,
        tgt,
        LossWeights(
            geom_depth=0.0,
            depth_temporal=1.0,
            depth_temporal_static_only=True,
            action=0.0,
            idm_reg=0.0,
        ),
    )

    assert losses["L_depth_temporal"].item() == pytest.approx(0.0)
