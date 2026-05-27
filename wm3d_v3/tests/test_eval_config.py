from pathlib import Path


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
