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
