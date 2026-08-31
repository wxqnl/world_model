from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ("ImitationLearning", "RichInteraction", "ReinforcementLearning")


def _archive(root: Path, prefix: str, *, unsafe: bool = False) -> None:
    payload = root / "payload"
    episode = payload / "lerobot"
    (episode / "meta").mkdir(parents=True, exist_ok=True)
    (episode / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    (episode / "videos/camera").mkdir(parents=True, exist_ok=True)
    (episode / "meta/info.json").write_text(
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
    (episode / "data/chunk-000/episode_000000.parquet").write_bytes(b"fixture")
    (episode / "videos/camera/episode_000000.mp4").write_bytes(b"fixture")
    destination = root / prefix / "part-000.tar"
    destination.parent.mkdir(parents=True)
    with tarfile.open(destination, "w") as handle:
        handle.add(episode, arcname="lerobot")
        if unsafe:
            member = tarfile.TarInfo("unsafe-link")
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp/escape"
            handle.addfile(member)


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/data/check_existing_agibot2026.py"),
            "--snapshot-root",
            str(root),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _extracted(root: Path, prefix: str) -> None:
    episode = root / prefix / "modelscope-cache" / "lerobot"
    (episode / "meta").mkdir(parents=True)
    (episode / "data/chunk-000").mkdir(parents=True)
    (episode / "videos/camera").mkdir(parents=True)
    (episode / "meta/info.json").write_text(
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
    (episode / "data/chunk-000/episode_000000.parquet").write_bytes(b"fixture")
    (episode / "videos/camera/episode_000000.mp4").write_bytes(b"fixture")


def test_existing_agibot2026_snapshot_probe_accepts_official_archive_layout(
    tmp_path: Path,
) -> None:
    for prefix in PREFIXES:
        _archive(tmp_path, prefix)
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["passed"] is True
    assert set(evidence["prefixes"]) == set(PREFIXES)
    assert all(
        row["archive_count"] == 1
        and row["sampled_archives"][0]["has_data"] is True
        and row["sampled_archives"][0]["has_visual"] is True
        for row in evidence["prefixes"].values()
    )


def test_existing_agibot2026_snapshot_probe_fails_closed_on_missing_prefix(
    tmp_path: Path,
) -> None:
    _archive(tmp_path, PREFIXES[0])
    result = _run(tmp_path)
    assert result.returncode != 0
    assert "RichInteraction" in result.stderr


def test_existing_agibot2026_snapshot_probe_rejects_link_members(
    tmp_path: Path,
) -> None:
    for prefix in PREFIXES:
        _archive(tmp_path, prefix, unsafe=prefix == PREFIXES[0])
    result = _run(tmp_path)
    assert result.returncode != 0
    assert "link/device" in result.stderr


def test_existing_agibot2026_snapshot_probe_accepts_extracted_modelscope_layout(
    tmp_path: Path,
) -> None:
    for prefix in PREFIXES:
        _extracted(tmp_path, prefix)
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["passed"] is True
    assert all(
        row["archive_count"] == 0
        and row["extracted_root_count"] == 1
        and row["sampled_extracted_roots"][0]["has_data"] is True
        and row["sampled_extracted_roots"][0]["has_visual"] is True
        for row in evidence["prefixes"].values()
    )
