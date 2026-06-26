from pathlib import Path


ROOT = Path("/data/Minko/world_model/wm3d_v5")
SCRIPT = ROOT / "scripts" / "train_stage0_hunyuan_dit_joint.py"
CONFIG = ROOT / "configs" / "v5_p64_1b_stage0_native3d_hunyuan_dit_jointpt_fromscratch_2node_v1.yaml"


def test_joint_stage0_script_is_not_adapter_only():
    text = SCRIPT.read_text()
    assert "ap.add_argument(\"--cfg\"" in text
    assert "--wm_ckpt" not in text
    assert "freeze_module(wm_model)" not in text
    assert "wm_model = build_model(cfg).to(device)" in text
    assert "compute_losses(" in text
    assert "controlled_dit_forward(" in text
    assert "hunyuan_dit_velocity" in text
    assert "hunyuan_control_adapter" in text
    assert "hunyuan_trainable" in text


def test_joint_stage0_config_removes_pixel_decoder_and_uses_true_dit():
    text = CONFIG.read_text()
    assert "enable_pixel: false" in text
    assert "enable_hunyuan_dit_loss: true" in text
    assert "enable_hunyuan_latent_loss: false" in text
    assert "enable_pixel_loss: false" in text
    assert "hunyuan_dit_train_lora: false" in text
    assert "max_steps: 0" in text


def test_joint_stage0_uses_project_wsd_scheduler():
    script = SCRIPT.read_text()
    config = CONFIG.read_text()
    assert "from wm3d_v3.training.lr_schedule import build_lr_scheduler" in script
    assert "sched = build_lr_scheduler(opt, cfg, total_steps)" in script
    assert "def build_scheduler(" not in script
    assert "type: wsd" in config
    assert "stable_frac:" in config or "decay_frac:" in config


def test_joint_stage0_does_not_empty_cuda_cache_every_step():
    script = SCRIPT.read_text()
    config = CONFIG.read_text()
    assert "empty_cache_every_steps" in script
    assert "torch.cuda.empty_cache()" in script
    assert "empty_cache_every > 0" in script
    assert "empty_cache_every_steps: 0" in config
