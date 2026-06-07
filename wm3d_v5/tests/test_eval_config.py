from pathlib import Path
from types import SimpleNamespace


def test_eval_and_demo_window_config_use_tokens_subdir_from_cfg():
    from wm3d_v3.eval.make_demo_gif import window_config_from_cfg as demo_window_config
    from wm3d_v3.eval.run_eval import window_config_from_cfg as eval_window_config

    cfg = {
        "data": {
            "T": 16,
            "k": 8,
            "stride": 4,
            "cache_root": "/tmp/wm3d_cache",
            "tokens_subdir": "vggt_p256",
        }
    }

    for build_window_config in (eval_window_config, demo_window_config):
        wcfg = build_window_config(cfg)
        assert wcfg.T == 16
        assert wcfg.k == 8
        assert wcfg.stride == 4
        assert wcfg.cache_root == Path("/tmp/wm3d_cache")
        assert wcfg.tokens_subdir == "vggt_p256"


def test_long_rollout_uses_tokens_subdir_from_cfg():
    from wm3d_v3.eval.make_long_rollout_gif import tokens_subdir_from_cfg

    assert tokens_subdir_from_cfg({"data": {"tokens_subdir": "vggt_p256"}}) == "vggt_p256"
    assert tokens_subdir_from_cfg({"data": {}}) == "vggt_pooled"


def test_eval_dataset_for_split_uses_episode_records(monkeypatch):
    from wm3d_v3.eval import run_eval

    class FakeWindowDataset:
        def __init__(self, records, cfg):
            self.records = records
            self.cfg = cfg

        def __len__(self):
            return len(self.records)

    monkeypatch.setattr(run_eval, "OXEWindowDataset", FakeWindowDataset)
    records = [
        SimpleNamespace(clip_id="c1", dataset="bridge"),
        SimpleNamespace(clip_id="c2", dataset="bridge"),
        SimpleNamespace(clip_id="c3", dataset="bridge"),
    ]
    cfg = {
        "data": {
            "T": 2,
            "k": 1,
            "stride": 1,
            "cache_root": "/tmp/wm3d_cache",
            "split": {"mode": "episode", "val_clip_ids": ["c2"]},
        }
    }

    val_ds = run_eval.build_dataset_for_split(records, cfg, split="val")
    train_ds = run_eval.build_dataset_for_split(records, cfg, split="train")

    assert [r.clip_id for r in val_ds.records] == ["c2"]
    assert {r.clip_id for r in train_ds.records} == {"c1", "c3"}


def _tiny_eval_cfg():
    return {
        "data": {
            "T": 2,
            "k": 2,
            "stride": 1,
            "cache_root": "/tmp/wm3d_cache",
            "require_task_emb": True,
        },
        "train": {"batch_size_per_gpu": 1, "num_workers": 0},
        "model": {
            "state": {
                "T": 2,
                "P": 4,
                "D": 16,
                "hidden": 32,
                "n_layers": 1,
                "n_heads": 4,
                "k": 2,
                "cond_dim": 16,
                "action_cond_dim": 7,
            },
            "action": {
                "T": 2,
                "P": 4,
                "D": 16,
                "hidden": 32,
                "n_layers": 1,
                "n_heads": 4,
                "k": 2,
                "z_dim": 8,
                "cond_dim": 16,
                "action_cond_dim": 7,
            },
            "xattn_layers_state": [],
            "xattn_n_heads": 4,
            "action_proj_hidden": 32,
            "action_proj_layers": 2,
            "geom_hidden": 16,
            "enable_geom_extra": False,
            "pixel_hidden": 16,
            "pixel_n_res": 1,
            "enable_pixel": False,
            "enable_context_pixel": False,
            "enable_bridging": False,
            "enable_world_prior": True,
            "world_prior_hidden": 32,
            "world_prior_layers": 1,
            "world_prior_heads": 4,
            "world_prior_task_dim": 16,
            "world_prior_action_dim": 7,
            "world_prior_use_context": True,
            "world_prior_use_action": True,
            "world_prior_predict_initial": True,
        },
    }


def test_eval_build_model_includes_world_prior_fields():
    from wm3d_v3.eval.run_eval import build_model

    model = build_model(_tiny_eval_cfg())

    assert model.world_prior is not None
    assert model.cfg.enable_world_prior is True
    assert model.cfg.world_prior_task_dim == 16


def test_world_prior_eval_refuses_configs_without_task_embeddings(tmp_path, monkeypatch):
    import pytest
    import yaml
    from wm3d_v3.eval import world_prior_eval

    cfg = _tiny_eval_cfg()
    cfg["data"]["require_task_emb"] = False
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setattr(
        "sys.argv",
        [
            "world_prior_eval",
            "--cfg", str(cfg_path),
            "--ckpt", str(tmp_path / "missing.pt"),
            "--out", str(tmp_path / "out.json"),
        ],
    )

    with pytest.raises(RuntimeError, match="require_task_emb"):
        world_prior_eval.main()


def test_eval_window_config_inherits_v5_data_flags_and_policy_progress():
    from wm3d_v3.eval.run_eval import window_config_from_cfg

    cfg = {
        "data": {
            "T": 4,
            "k": 2,
            "stride": 1,
            "cache_root": "/tmp/wm3d_cache",
            "tokens_subdir": "vggt_pooled",
            "load_rgb": False,
            "load_geom": True,
            "load_state_tgt": False,
            "require_geom_extra": True,
            "load_policy_state": True,
            "require_policy_state": True,
            "policy_lowdim_dim": 5,
        },
        "model": {
            "policy_use_progress": True,
            "policy_object_state_dim": 3,
            "policy_plan_state_dim": 4,
            "policy_action_history_len": 2,
            "policy_action_history_dim": 7,
        },
    }

    wcfg = window_config_from_cfg(cfg)

    assert wcfg.load_rgb is False
    assert wcfg.load_geom is True
    assert wcfg.load_state_tgt is False
    assert wcfg.require_geom_extra is True
    assert wcfg.load_policy_state is True
    assert wcfg.require_policy_state is True
    assert wcfg.require_progress is True
    assert wcfg.policy_lowdim_dim == 5
    assert wcfg.policy_object_state_dim == 3
    assert wcfg.policy_plan_state_dim == 4
    assert wcfg.policy_action_history_len == 2

def test_run_eval_fails_fast_when_required_modalities_disabled():
    import pytest
    from wm3d_v3.eval.run_eval import validate_eval_data_flags

    cfg = _tiny_eval_cfg()
    cfg["data"]["load_state_tgt"] = False

    with pytest.raises(ValueError, match="data.load_state_tgt"):
        validate_eval_data_flags(cfg, rgb_metrics=False)

    cfg = _tiny_eval_cfg()
    cfg["data"]["load_geom"] = False

    with pytest.raises(ValueError, match="data.load_geom"):
        validate_eval_data_flags(cfg, rgb_metrics=False)

    cfg = _tiny_eval_cfg()
    cfg["data"]["load_rgb"] = False
    cfg["model"]["enable_context_pixel"] = True

    with pytest.raises(ValueError, match="data.load_rgb"):
        validate_eval_data_flags(cfg, rgb_metrics=False)

def test_run_eval_allows_rgb_disabled_when_rgb_is_not_used():
    from wm3d_v3.eval.run_eval import validate_eval_data_flags

    cfg = _tiny_eval_cfg()
    cfg["data"]["load_rgb"] = False
    cfg["model"]["enable_context_pixel"] = False

    validate_eval_data_flags(cfg, rgb_metrics=False)


def test_world3d_eval_requires_rgb_for_motion_masks():
    import pytest
    from wm3d_v3.eval.run_eval import validate_eval_data_flags

    cfg = _tiny_eval_cfg()
    cfg["data"]["load_rgb"] = False

    with pytest.raises(ValueError, match="data.load_rgb"):
        validate_eval_data_flags(cfg, rgb_metrics=True)

