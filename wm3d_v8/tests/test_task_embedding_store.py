from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch

from wm3d_v3.data.manifest_contract import sha256_file
from wm3d_v3.data.task_embedding_store import (
    TASK_BANK_SCHEMA,
    TaskEmbeddingStore,
    TaskEmbeddingStoreError,
)


PROFILE_SHA = "1" * 64
ENCODER_SHA = "2" * 64
SOURCE_SHAS = {"a": "3" * 64, "b": "4" * 64}


def _build_bank(root: Path, text: str = "pick up the cup") -> tuple[str, Path]:
    identity = hashlib.sha256(text.encode("utf-8")).hexdigest()
    relative = Path("embeddings") / identity[:2] / f"{identity}.safetensors"
    embedding_path = root / relative
    embedding_path.parent.mkdir(parents=True)
    save_file({"embedding": torch.arange(2048, dtype=torch.float32)}, embedding_path)
    index = root / "index.jsonl"
    row = {
        "schema": TASK_BANK_SCHEMA,
        "text_id": identity,
        "text": text,
        "path": str(relative),
        "sha256": sha256_file(embedding_path),
    }
    index.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "schema": TASK_BANK_SCHEMA,
        "data_profile_sha256": PROFILE_SHA,
        "source_manifest_sha256_by_name": SOURCE_SHAS,
        "encoder_contract_sha256": ENCODER_SHA,
        "unique_text_count": 1,
        "index_path": str(index.absolute()),
        "index_sha256": sha256_file(index),
    }
    (root / "receipt.json").write_text(
        json.dumps(receipt, sort_keys=True), encoding="utf-8"
    )
    return text, embedding_path


def _open(root: Path, **overrides: object) -> TaskEmbeddingStore:
    values = {
        "root": root,
        "index_sha256": sha256_file(root / "index.jsonl"),
        "expected_data_profile_sha256": PROFILE_SHA,
        "expected_source_manifest_sha256_by_name": SOURCE_SHAS,
        "expected_encoder_contract_sha256": ENCODER_SHA,
    }
    values.update(overrides)
    return TaskEmbeddingStore(**values)


def test_store_verifies_full_provenance_and_tensor_abi(tmp_path: Path) -> None:
    text, _path = _build_bank(tmp_path)
    value = _open(tmp_path).get(text)
    assert value.shape == (2048,)
    assert torch.equal(value, torch.arange(2048, dtype=torch.float32))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("expected_data_profile_sha256", "9" * 64),
        ("expected_source_manifest_sha256_by_name", {"a": "9" * 64}),
        ("expected_encoder_contract_sha256", "9" * 64),
    ),
)
def test_store_rejects_provenance_mismatch(
    tmp_path: Path, field: str, value: object
) -> None:
    _build_bank(tmp_path)
    with pytest.raises(TaskEmbeddingStoreError, match="provenance"):
        _open(tmp_path, **{field: value})


def test_store_rehashes_embedding_on_first_open(tmp_path: Path) -> None:
    text, embedding_path = _build_bank(tmp_path)
    store = _open(tmp_path)
    save_file({"embedding": torch.zeros(2048)}, embedding_path)
    with pytest.raises(TaskEmbeddingStoreError, match="SHA"):
        store.get(text)


def test_store_rejects_unknown_instruction(tmp_path: Path) -> None:
    _build_bank(tmp_path)
    with pytest.raises(TaskEmbeddingStoreError, match="absent"):
        _open(tmp_path).get("an instruction absent from the bank")
