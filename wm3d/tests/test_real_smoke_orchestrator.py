from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts import run_real_smoke as smoke


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_gpu_parser_requires_exactly_two_distinct_indices() -> None:
    assert smoke._parse_gpus("0,7") == (0, 7)
    for invalid in ("0", "0,0", "0,1,2", "-1,0", "gpu0,gpu1"):
        with pytest.raises(Exception):
            smoke._parse_gpus(invalid)


def test_step_receipt_is_no_clobber_and_detects_output_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_wm3d = _write(repo / "run_wm3d.sh", b"#!/bin/sh\nexit 0\n")
    run_wm3d.chmod(0o755)
    work = tmp_path / "work"
    output = work / "result.json"

    def fake_run(command, *, cwd, environment, log) -> None:
        assert command[-2:] == ["fake", "--go"]
        _write(output, b'{"passed":true}\n')

    monkeypatch.setattr(smoke, "_run", fake_run)
    flow = smoke.Orchestrator(
        repo=repo,
        work=work,
        python=Path("/usr/bin/python3"),
        args=object(),
        plan_sha="a" * 64,
        environment={},
    )
    receipt = flow.step(2, "fake", ["fake", "--go"], [(output, "file")])
    sealed = json.loads(receipt.read_text())
    assert sealed["passed"] is True
    assert sealed["outputs"][0]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    flow.step(2, "fake", ["fake", "--go"], [(output, "file")])
    output.write_bytes(b"tampered")
    with pytest.raises(smoke.SmokeError, match="产物发生漂移"):
        flow.step(2, "fake", ["fake", "--go"], [(output, "file")])


def test_safe_vggt_extractor_rejects_link(tmp_path: Path) -> None:
    import tarfile

    archive = tmp_path / "bad.tar.gz"
    prefix = f"vggt-{smoke.VGGT_SOURCE_REVISION}"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo(f"{prefix}/vggt/models/vggt.py")
        member.type = tarfile.SYMTYPE
        member.linkname = "/etc/passwd"
        handle.addfile(member)
    with pytest.raises(smoke.SmokeError, match="禁止 link"):
        smoke._safe_extract_vggt(archive, tmp_path / "source")


def test_checkpoint_validator_checks_every_manifest_payload(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step_00000001"
    payload = _write(checkpoint / "distcp" / "__0_0.distcp", b"real-shard")
    metadata = {
        "runtime_config_sha256": "1" * 64,
        "run_lineage": "lineage",
        "step": 1,
        "sampler_progress": {"next_optimizer_step": 1},
        "world_size": 2,
        "distributed_strategy": "fsdp2",
        "gradient_ownership": {"passed": True},
    }
    metadata_path = _write(
        checkpoint / "metadata.json",
        (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode(),
    )
    manifest = {
        "schema": "wm3d_v8_distributed_checkpoint_v2",
        "step": 1,
        "files": {
            "distcp/__0_0.distcp": {
                "sha256": smoke._sha256(payload),
                "size": payload.stat().st_size,
            },
            "metadata.json": {
                "sha256": smoke._sha256(metadata_path),
                "size": metadata_path.stat().st_size,
            },
        },
    }
    manifest_path = _write(
        checkpoint / "MANIFEST.json",
        (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode(),
    )
    commit = {
        "schema": "wm3d_v8_distributed_checkpoint_commit_v2",
        "step": 1,
        "run_lineage": "lineage",
        "manifest_sha256": smoke._sha256(manifest_path),
        "manifest_content_sha256": smoke._canonical_sha(manifest),
        "metadata_sha256": smoke._sha256(metadata_path),
    }
    _write(
        checkpoint / "COMMITTED.json",
        (json.dumps(commit, sort_keys=True, indent=2) + "\n").encode(),
    )
    smoke._validate_checkpoint(checkpoint, 1, "1" * 64, "lineage")
    payload.write_bytes(b"corrupt-shard")
    with pytest.raises(smoke.SmokeError, match="SHA/size"):
        smoke._validate_checkpoint(checkpoint, 1, "1" * 64, "lineage")


def test_run_wm3d_uses_distributed_preflight_and_has_no_transition() -> None:
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "run_wm3d.sh").read_text(encoding="utf-8")
    assert "preflight) torchrun_preflight_split" in text
    assert '"${app_args[@]}" --preflight-only' in text
    assert "transition)" not in text
    assert "audit_wm3d_stage0_libero_transition.py" not in text
    subprocess.run(["bash", "-n", str(repo / "run_wm3d.sh")], check=True)


def test_smoke_cli_rejects_missing_human_confirmations(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    args = smoke.parse_args(
        ["--work-root", str(tmp_path / "work"), "--operator", "reviewer"]
    )
    with pytest.raises(smoke.SmokeError, match="许可"):
        smoke._validate_args(args, repo)


def test_smoke_inventory_requires_exact_train_val_assignment(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path / "inventory.jsonl",
        (
            '{"episode_id":"aloha_smoke:000000000","split":"train"}\n'
            '{"episode_id":"aloha_smoke:000000030","split":"val"}\n'
        ).encode(),
    )
    receipt = _write(
        tmp_path / "receipt.json",
        json.dumps(
            {
                "selection": {"episode_indices": [0, 30]},
                "split_count": {"test": 0, "train": 1, "val": 1},
            }
        ).encode(),
    )
    smoke._validate_smoke_inventory(manifest, receipt)
    manifest.write_text(
        '{"episode_id":"aloha_smoke:000000000","split":"train"}\n'
        '{"episode_id":"aloha_smoke:000000030","split":"train"}\n'
    )
    with pytest.raises(smoke.SmokeError, match="0=train、30=val"):
        smoke._validate_smoke_inventory(manifest, receipt)
