from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_project_uses_single_unversioned_wm3d_layout() -> None:
    assert ROOT.name == "wm3d"
    assert (ROOT / "run_wm3d.sh").is_file()
    assert (ROOT / "wm3d" / "__init__.py").is_file()
    for retired in (
        "run_v7.sh",
        "run_v8.sh",
        "wm3d_v3",
        "wm3d_v7",
        "wm3d_v8",
    ):
        assert not (ROOT / retired).exists(), retired


def test_every_project_directory_publishes_a_coding_guide() -> None:
    assert (ROOT.parent / "CODING.md").is_file()
    directories = [ROOT]
    directories.extend(
        path
        for path in ROOT.rglob("*")
        if path.is_dir()
        and not any(part.startswith(".") or part == "__pycache__" for part in path.parts)
    )
    missing = [str(path.relative_to(ROOT)) for path in directories if not (path / "CODING.md").is_file()]
    assert not missing, missing


def test_user_facing_entrypoints_are_unversioned() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    entry = (ROOT / "run_wm3d.sh").read_text(encoding="utf-8")
    assert "./run_wm3d.sh" in readme
    assert "wm3d_v7/" not in readme
    assert "wm3d_v8/" not in readme
    assert "run_v7.sh" not in entry
    assert "run_v8.sh" not in entry


def test_5b_operator_handoff_is_discoverable() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    entry = (ROOT / "run_wm3d.sh").read_text(encoding="utf-8")
    assert "docs/WM3D_5B_SCALING.md" in readme
    assert '5b) exec bash scripts/cluster/wm3d_5b.sh' in entry
    assert (ROOT / "configs/cluster/h200_5b.env.example").is_file()
    assert (ROOT / "configs/runtime/h200_64_fsdp2_canary1k.yaml").is_file()
    assert not (ROOT / "configs/runtime/h200_64_fsdp2_validation10k.yaml").exists()
    assert (ROOT / "scripts/data/materialize_oxe_default.py").is_file()
    assert (ROOT / "scripts/tools/report_5b_run.py").is_file()


def test_1b_streaming_handoff_is_discoverable() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    entry = (ROOT / "run_wm3d.sh").read_text(encoding="utf-8")
    assert "docs/WM3D_1B_STREAMING.md" in readme
    assert '1b) exec bash scripts/cluster/wm3d_1b.sh' in entry
    assert (ROOT / "scripts/cluster/wm3d_1b.sh").is_file()
    assert (ROOT / "configs/cluster/h100_1b_streaming.env.example").is_file()
    assert (
        ROOT / "configs/runtime/h100_8_fsdp2_streaming_canary1k.yaml"
    ).is_file()
    assert (
        ROOT / "configs/runtime/h100_8_fsdp2_streaming_formal100k.yaml"
    ).is_file()
