from __future__ import annotations

from pathlib import Path

import pytest

from wm3d_v3.encoders.qwen_vl_encoder import QwenVLEmbed


@pytest.mark.parametrize("revision", [None, "main", "A" * 40, "a" * 39])
def test_qwen_revision_must_be_an_immutable_commit(
    revision: str | None,
) -> None:
    with pytest.raises(RuntimeError, match="40-hex commit SHA"):
        QwenVLEmbed(model_revision=revision, device="cpu")


def test_local_qwen_snapshot_must_match_pinned_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = "a" * 40
    wrong = tmp_path / ("b" * 40)
    wrong.mkdir()
    monkeypatch.setenv("QWEN3_VL_EMBEDDING_PATH", str(wrong))
    with pytest.raises(RuntimeError, match="revision mismatch"):
        QwenVLEmbed(
            model_revision=revision,
            device="cpu",
        )


def test_exact_cached_qwen_snapshot_is_selected_not_newest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = "c" * 40
    hub = tmp_path / "hub"
    exact = (
        hub
        / "models--Qwen--Qwen3-VL-Embedding-2B"
        / "snapshots"
        / revision
    )
    exact.mkdir(parents=True)
    (exact / "modules.json").write_text("[]", encoding="utf-8")
    newer = exact.parent / ("d" * 40)
    newer.mkdir()
    (newer / "modules.json").write_text("[]", encoding="utf-8")
    monkeypatch.delenv("QWEN3_VL_EMBEDDING_PATH", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    assert QwenVLEmbed._resolve_model_path(
        "Qwen/Qwen3-VL-Embedding-2B", revision
    ) == str(exact.resolve())
