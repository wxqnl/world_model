from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

import yaml


ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ("ImitationLearning", "RichInteraction", "ReinforcementLearning")


def _model_bundle(root: Path) -> dict[str, Path]:
    vggt_revision = yaml.safe_load(
        (ROOT / "configs/encoder/vggt_native_p144.yaml").read_text()
    )["model_revision"]
    task_revision = yaml.safe_load(
        (ROOT / "configs/encoder/task_qwen3_vl_embedding_2b.yaml").read_text()
    )["model_revision"]
    source = root / "vision/vggt-source"
    (source / "vggt/models").mkdir(parents=True)
    (source / "vggt/models/vggt.py").write_text("class VGGT: pass\n")
    (source / "vggt/models/aggregator.py").write_text("class Aggregator: pass\n")
    vggt = root / "weights/VGGT-1B" / vggt_revision
    vggt.mkdir(parents=True)
    (vggt / "config.json").write_text("{}\n")
    (vggt / "model.safetensors").write_bytes(b"fixture")
    task = root / "weights/Qwen3-VL-Embedding-2B" / task_revision
    task.mkdir(parents=True)
    (task / "modules.json").write_text("[]\n")
    (task / "config.json").write_text("{}\n")
    (task / "model.safetensors").write_bytes(b"fixture")
    return {"source": source, "vggt": vggt, "task": task}


def _archive(root: Path, prefix: str) -> None:
    payload = root / "payload" / prefix / "lerobot"
    (payload / "meta").mkdir(parents=True)
    (payload / "data/chunk-000").mkdir(parents=True)
    (payload / "videos/camera").mkdir(parents=True)
    (payload / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "fps": 30,
                "features": {
                    "action": {"dtype": "float32", "shape": [14]},
                    "observation.state": {"dtype": "float32", "shape": [14]},
                },
            }
        )
    )
    (payload / "data/chunk-000/episode.parquet").write_bytes(b"fixture")
    (payload / "videos/camera/episode.mp4").write_bytes(b"fixture")
    destination = root / "modelscope/cache/AgiBotWorld2026" / prefix / "part.tar"
    destination.parent.mkdir(parents=True)
    with tarfile.open(destination, "w") as handle:
        handle.add(payload, arcname="lerobot")


def _run(
    model_root: Path,
    data_root: Path,
    work_root: Path,
    site: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/cluster/configure_5b_inputs.py"),
            "--model-root",
            str(model_root),
            "--data-root",
            str(data_root),
            "--work-root",
            str(work_root),
            "--site-output",
            str(site),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_5b_configure_accepts_nested_modelscope_archives_and_writes_site(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    paths = _model_bundle(models)
    data = tmp_path / "downloaded-data"
    for prefix in PREFIXES:
        _archive(data, prefix)
    site = tmp_path / "work/control/5b_canary1k.env"
    result = _run(models, data, tmp_path / "work", site)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["input_check"] == "PASS"
    assert report["agibot_layout"] == "PASS"
    assert report["data_state"] == "RAW_COMPATIBLE"
    assert report["ready_to_train"] is False
    payload = site.read_text()
    assert f"RAW_ROOT={data}" in payload
    assert f"WM3D_VGGT_SOURCE_ROOT={paths['source']}" in payload
    assert f"WM3D_VGGT_MODEL_SNAPSHOT={paths['vggt']}" in payload
    assert f"QWEN3_VL_EMBEDDING_PATH={paths['task']}" in payload
    assert "__SET_BY_5B_CONFIGURE__" not in payload
    assert "MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}" in payload
    assert site.stat().st_mode & 0o777 == 0o600


def test_5b_configure_reports_training_ready_for_complete_local_bundle(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    _model_bundle(models)
    data = tmp_path / "bundle"
    control = data / "control"
    control.mkdir(parents=True)
    (control / "public_robot_oxe.yaml").write_text(
        "schema: wm3d_v8_data_profile_v4\nname: fixture\nsources: []\n"
    )
    required = [
        data / "cache/native_p144/task_bank/index.jsonl",
        data / "cache/native_p144/cache_tasks.jsonl",
        data / "cache/native_p144/episode_index.jsonl",
        data / "cache/native_p144/window_index_5b.jsonl",
        data / "cache/native_p144/grouped_normalization_5b.json",
        data / "streaming_metadata/native_p144/metadata_seal_5b.json",
        data / "envs/wm3d-cu128/environment_receipt.json",
        data / "control/runtime_5b_canary1k.yaml",
    ]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    site = control / "5b_canary1k.env"
    result = _run(models, data, data, site)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["data_state"] == "TRAIN_METADATA_READY"
    assert report["environment_ready"] is True
    assert report["runtime_ready"] is True
    assert report["ready_to_train"] is True


def test_5b_configure_reports_control_pack_paths_from_another_machine(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    _model_bundle(models)
    data = tmp_path / "bundle"
    control = data / "control"
    control.mkdir(parents=True)
    (control / "public_robot_oxe.yaml").write_text(
        "schema: wm3d_v8_data_profile_v4\n"
        "name: fixture\n"
        "sources:\n"
        "  - name: fixture\n"
        "    raw_root: /missing/on/this/machine/raw\n"
        "    adapter_config: /missing/on/this/machine/adapter.yaml\n"
        "    manifest: /missing/on/this/machine/manifest.jsonl\n"
    )
    site = control / "5b_canary1k.env"
    result = _run(models, data, data, site)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["data_state"] == "PROFILE_PATH_MISMATCH"
    assert report["ready_to_train"] is False
    assert any("missing raw_root" in row for row in report["missing_training_metadata"])


def test_5b_delivery_doc_uses_two_paths_without_internal_revision_directories() -> None:
    document = (ROOT / "docs/WM3D_5B_SCALING.md").read_text()
    assert "训练操作员" not in document
    assert "操作员" not in document
    assert re.search(r"[0-9a-f]{40}", document) is None
    assert "MODEL_ROOT=/共享目录/模型" in document
    assert "DATA_ROOT=/共享目录/已下载数据" in document
    assert './run_wm3d.sh 5b configure "$MODEL_ROOT" "$DATA_ROOT"' in document
    assert "魔搭" in document
    assert './run_wm3d.sh 5b slurm "$SITE" train 100' in document
