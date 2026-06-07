from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_generation_scripts_do_not_cd_to_v3_root():
    script_paths = [
        ROOT / "scripts" / "run_generation_canary_v1.sh",
        ROOT / "scripts" / "watch_generation_canary_v1.sh",
        ROOT / "scripts" / "run_v5_generation_stage_hunyuan_adapter_v1.sh",
        ROOT / "scripts" / "run_v5_generation_stage_hunyuan_dit_control_v1.sh",
    ]

    for path in script_paths:
        text = path.read_text()
        assert "/data/Minko/world_model/wm3d_v3" not in text
        assert "wm3d_v5" in text or "V5_ROOT" in text
    assert "split(\".\")" in (ROOT / "scripts" / "run_generation_canary_v1.sh").read_text()



def test_generation_canary_hunyuan_dit_generation_requires_all_switches(tmp_path):
    fake_py = tmp_path / "fake_py.py"
    fake_py.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ[\"FAKE_PY_LOG\"]).open(\"a\").write(json.dumps(sys.argv[1:]) + \"\\n\")\n"
        "if len(sys.argv) > 1 and sys.argv[1] == \"-\":\n"
        "    sys.stdin.read()\n"
        "    print(\"false\")\n"
        "sys.exit(0)\n"
    )
    fake_py.chmod(0o755)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model:\n  enable_world_prior: false\n  enable_pixel: false\ndata:\n  load_rgb: false\n")
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"not-a-real-checkpoint")
    control_ckpt = tmp_path / "dit_control.pt"
    control_ckpt.write_bytes(b"not-a-real-control-checkpoint")
    log = tmp_path / "calls.jsonl"

    base_env = os.environ.copy()
    base_env.update(
        {
            "PY": str(fake_py),
            "FAKE_PY_LOG": str(log),
            "V5_ROOT": str(ROOT),
            "CFG": str(cfg),
            "CKPT": str(ckpt),
            "OUT_DIR": str(tmp_path / "out_skip"),
            "RUN_HUNYUAN_DIT_GENERATION": "1",
            "ALLOW_REAL_HUNYUAN_GPU_GENERATION": "0",
            "HUNYUAN_DIT_CONTROL_CKPT": str(control_ckpt),
        }
    )
    subprocess.run(
        ["bash", "scripts/run_generation_canary_v1.sh"],
        cwd=ROOT,
        env=base_env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    module_calls = [call for call in calls if len(call) >= 3 and call[0] == "-m"]
    assert not any(call[1] == "wm3d_v3.eval.make_hunyuan_dit_control_demo" for call in module_calls)

    log.write_text("")
    run_env = base_env.copy()
    run_env.update(
        {
            "OUT_DIR": str(tmp_path / "out_run"),
            "ALLOW_REAL_HUNYUAN_GPU_GENERATION": "1",
            "HUNYUAN_DIT_HEIGHT": "320",
            "HUNYUAN_DIT_WIDTH": "512",
            "HUNYUAN_DIT_FRAMES": "9",
            "HUNYUAN_DIT_STEPS": "4",
            "HUNYUAN_DIT_SEED": "123",
        }
    )
    subprocess.run(
        ["bash", "scripts/run_generation_canary_v1.sh"],
        cwd=ROOT,
        env=run_env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    module_calls = [call for call in calls if len(call) >= 3 and call[0] == "-m"]
    dit_calls = [call for call in module_calls if call[1] == "wm3d_v3.eval.make_hunyuan_dit_control_demo"]
    assert len(dit_calls) == 1
    dit_call = dit_calls[0]
    assert "--wm_ckpt" in dit_call and str(ckpt) in dit_call
    assert "--control_ckpt" in dit_call and str(control_ckpt) in dit_call
    assert "--height" in dit_call and "320" in dit_call
    assert "--width" in dit_call and "512" in dit_call
    assert "--frames" in dit_call and "9" in dit_call
    assert "--steps" in dit_call and "4" in dit_call
    assert "--seed" in dit_call and "123" in dit_call

def test_generation_canary_defaults_to_core_eval_only(tmp_path):
    fake_py = tmp_path / "fake_py.py"
    fake_py.write_text(
        """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
Path(os.environ[\"FAKE_PY_LOG\"]).open(\"a\").write(json.dumps(sys.argv[1:]) + "\\n")
if len(sys.argv) > 1 and sys.argv[1] == \"-\":
    sys.stdin.read()
    print(\"false\")
sys.exit(0)
"""
    )
    fake_py.chmod(0o755)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model:\n  enable_world_prior: false\n  enable_pixel: false\ndata:\n  load_rgb: false\n")
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"not-a-real-checkpoint")
    log = tmp_path / "calls.jsonl"
    out_dir = tmp_path / "out"

    env = os.environ.copy()
    env.update(
        {
            "PY": str(fake_py),
            "FAKE_PY_LOG": str(log),
            "V5_ROOT": str(ROOT),
            "CFG": str(cfg),
            "CKPT": str(ckpt),
            "OUT_DIR": str(out_dir),
        }
    )

    subprocess.run(
        ["bash", "scripts/run_generation_canary_v1.sh"],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    calls = [json.loads(line) for line in log.read_text().splitlines()]
    module_calls = [call for call in calls if len(call) >= 3 and call[0] == "-m"]

    assert any(call[1] == "wm3d_v3.eval.run_eval" and "--skip_rgb_metrics" in call for call in module_calls)
    assert not any(call[1] == "wm3d_v3.eval.world_prior_eval" for call in module_calls)
    assert not any(call[1] == "wm3d_v3.eval.make_demo_gif" for call in module_calls)
    assert not any(call[1] == "wm3d_v3.eval.make_hunyuan_latent_demo" for call in module_calls)
    assert not any(call[1] == "wm3d_v3.eval.make_hunyuan_dit_control_demo" for call in module_calls)


def test_generation_canary_ignores_flow_ckpt_for_latent_demo(tmp_path):
    fake_py = tmp_path / "fake_py.py"
    fake_py.write_text(
        """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
Path(os.environ[\"FAKE_PY_LOG\"]).open(\"a\").write(json.dumps(sys.argv[1:]) + "\\n")
if len(sys.argv) > 1 and sys.argv[1] == \"-\":
    sys.stdin.read()
    print(\"false\")
sys.exit(0)
"""
    )
    fake_py.chmod(0o755)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model:\n  enable_world_prior: false\n  enable_pixel: false\ndata:\n  load_rgb: false\n")
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"not-a-real-checkpoint")
    flow_ckpt = tmp_path / "flow.pt"
    flow_ckpt.write_bytes(b"not-a-real-flow-checkpoint")
    log = tmp_path / "calls.jsonl"

    env = os.environ.copy()
    env.update(
        {
            "PY": str(fake_py),
            "FAKE_PY_LOG": str(log),
            "V5_ROOT": str(ROOT),
            "CFG": str(cfg),
            "CKPT": str(ckpt),
            "HUNYUAN_FLOW_CKPT": str(flow_ckpt),
            "OUT_DIR": str(tmp_path / "out"),
        }
    )

    completed = subprocess.run(
        ["bash", "scripts/run_generation_canary_v1.sh"],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    calls = [json.loads(line) for line in log.read_text().splitlines()]
    module_calls = [call for call in calls if len(call) >= 3 and call[0] == "-m"]

    assert not any(call[1] == "wm3d_v3.eval.make_hunyuan_latent_demo" for call in module_calls)
    assert "flow_demo_not_supported_by_latent_adapter_demo" in completed.stdout


def test_hunyuan_demo_resolves_embedded_adapter_checkpoint():
    from wm3d_v3.eval import make_hunyuan_latent_demo as demo

    joint = {
        "model": {"wm.weight": object()},
        "hunyuan_adapter": {"adapter.weight": object()},
        "hunyuan_adapter_cfg": {"hidden": 64, "n_blocks": 2},
    }

    payload = demo.resolve_hunyuan_demo_checkpoint(
        joint,
        ckpt_path=Path("joint.pt"),
        wm_ckpt_path=None,
        load_checkpoint=lambda path: pytest.fail(f"unexpected load {path}"),
    )

    assert payload.source_format == "trainer_embedded_adapter"
    assert payload.world_model_state is joint["model"]
    assert payload.adapter_state is joint["hunyuan_adapter"]
    assert payload.adapter_cfg.hidden == 64
    assert payload.adapter_cfg.n_blocks == 2


def test_hunyuan_demo_resolves_standalone_adapter_checkpoint_with_wm_ckpt():
    from wm3d_v3.eval import make_hunyuan_latent_demo as demo

    standalone = {
        "model": {"adapter.weight": object()},
        "cfg": {"hidden": 128, "n_blocks": 3},
    }
    wm = {"model": {"wm.weight": object()}}
    loaded = []

    def load_checkpoint(path):
        loaded.append(path)
        return wm

    payload = demo.resolve_hunyuan_demo_checkpoint(
        standalone,
        ckpt_path=Path("adapter.pt"),
        wm_ckpt_path=Path("wm.pt"),
        load_checkpoint=load_checkpoint,
    )

    assert loaded == [Path("wm.pt")]
    assert payload.source_format == "standalone_adapter"
    assert payload.world_model_state is wm["model"]
    assert payload.adapter_state is standalone["model"]
    assert payload.adapter_cfg.hidden == 128
    assert payload.adapter_cfg.n_blocks == 3


def test_hunyuan_demo_standalone_adapter_requires_wm_ckpt():
    from wm3d_v3.eval import make_hunyuan_latent_demo as demo

    standalone = {"model": {}, "cfg": {"hidden": 128, "n_blocks": 3}}

    with pytest.raises(RuntimeError, match="--wm_ckpt"):
        demo.resolve_hunyuan_demo_checkpoint(
            standalone,
            ckpt_path=Path("adapter.pt"),
            wm_ckpt_path=None,
            load_checkpoint=lambda path: pytest.fail(f"unexpected load {path}"),
        )



def test_world_prior_no_pixel_does_not_require_rgb_in():
    from wm3d_v3.eval import world_prior_eval

    batch = {}
    cfg = {"model": {"enable_context_pixel": False}}
    context_pixel_cfg = {"model": {"enable_context_pixel": True}}

    assert world_prior_eval.context_rgb_for_world_prior(batch, cfg, device="cpu", pixel=False) is None
    assert world_prior_eval.context_rgb_for_world_prior(batch, context_pixel_cfg, device="cpu", pixel=False) is None

    with pytest.raises(KeyError, match="rgb_in"):
        world_prior_eval.context_rgb_for_world_prior(batch, cfg, device="cpu", pixel=True)


def test_hunyuan_demo_rejects_flow_denoiser_checkpoint_format():
    from wm3d_v3.eval import make_hunyuan_latent_demo as demo

    flow_like = {
        "model": {"denoiser.weight": object()},
        "cfg": {"hidden": 128, "n_blocks": 3, "use_rough_latents": True},
    }
    wm = {"model": {"wm.weight": object()}}

    with pytest.raises(RuntimeError, match="HunyuanLatentAdapterConfig"):
        demo.resolve_hunyuan_demo_checkpoint(
            flow_like,
            ckpt_path=Path("flow.pt"),
            wm_ckpt_path=Path("wm.pt"),
            load_checkpoint=lambda path: wm,
        )


def test_eval_report_mode_marks_rgb_as_not_hunyuan_generation(tmp_path, monkeypatch):
    import json
    import torch
    from types import SimpleNamespace
    from wm3d_v3.eval import run_eval

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, state):
            return None

        def __call__(self, *args, **kwargs):
            raise AssertionError("max_batches=0 should not run model forward")

    class FakeLPIPS:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

        def parameters(self):
            return []

    monkeypatch.setitem(sys.modules, "lpips", SimpleNamespace(LPIPS=FakeLPIPS))
    monkeypatch.setattr(run_eval, "read_manifest", lambda path: [])
    monkeypatch.setattr(run_eval, "build_dataset_for_split", lambda records, cfg, split: [])
    monkeypatch.setattr(run_eval, "build_model", lambda cfg: FakeModel())

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "data:\n"
        "  manifest: /tmp/empty.jsonl\n"
        "  T: 2\n"
        "  k: 1\n"
        "  stride: 1\n"
        "  cache_root: /tmp/cache\n"
        "  load_rgb: true\n"
        "  load_geom: true\n"
        "  load_state_tgt: true\n"
        "train:\n"
        "  batch_size_per_gpu: 1\n"
        "  num_workers: 0\n"
        "model:\n"
        "  enable_pixel: true\n"
        "  enable_context_pixel: true\n"
    )
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model": {}, "epoch": 0, "val_total": 0.0}, ckpt)
    out = tmp_path / "eval.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval",
            "--cfg", str(cfg),
            "--ckpt", str(ckpt),
            "--out", str(out),
            "--max_batches", "0",
            "--batch_size", "1",
        ],
    )

    run_eval.main()
    report = json.loads(out.read_text())

    assert report["mode"]["rgb_metrics"] is True
    assert report["mode"]["rgb_metrics_active"] is True
    assert report["mode"]["video_generation_active"] is False
    assert report["mode"]["hunyuan_generation_active"] is False
