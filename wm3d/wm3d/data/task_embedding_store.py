"""SHA-verified read-only task-embedding bank."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from safetensors import safe_open
import torch

from .manifest_contract import sha256_file


class TaskEmbeddingStoreError(RuntimeError):
    pass


TASK_BANK_SCHEMA = "wm3d_v8_task_embedding_bank_v2"


class TaskEmbeddingStore:
    def __init__(
        self,
        *,
        root: Path,
        index_sha256: str,
        expected_data_profile_sha256: str,
        expected_source_manifest_sha256_by_name: Mapping[str, str],
        expected_encoder_contract_sha256: str,
    ):
        self.root = Path(root).resolve(strict=True)
        index = self.root / "index.jsonl"
        if index.is_symlink() or not index.is_file() or sha256_file(index) != index_sha256:
            raise TaskEmbeddingStoreError("task bank index SHA mismatch")
        receipt_path = self.root / "receipt.json"
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise TaskEmbeddingStoreError("task bank receipt is missing/invalid")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        required_receipt = {
            "schema",
            "data_profile_sha256",
            "source_manifest_sha256_by_name",
            "encoder_contract_sha256",
            "unique_text_count",
            "index_path",
            "index_sha256",
        }
        if not isinstance(receipt, dict) or set(receipt) != required_receipt:
            raise TaskEmbeddingStoreError("task bank receipt fields mismatch")
        if (
            receipt["schema"] != TASK_BANK_SCHEMA
            or receipt["data_profile_sha256"] != expected_data_profile_sha256
            or receipt["source_manifest_sha256_by_name"]
            != dict(sorted(expected_source_manifest_sha256_by_name.items()))
            or receipt["encoder_contract_sha256"] != expected_encoder_contract_sha256
            or receipt["index_sha256"] != index_sha256
            or Path(str(receipt["index_path"])).resolve(strict=True) != index
        ):
            raise TaskEmbeddingStoreError("task bank receipt provenance mismatch")
        self.entries: dict[str, tuple[str, str]] = {}
        with index.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                required = {"schema", "text_id", "text", "path", "sha256"}
                if not isinstance(row, dict) or set(row) != required:
                    raise TaskEmbeddingStoreError(f"task bank row {line_number} fields mismatch")
                if row["schema"] != TASK_BANK_SCHEMA:
                    raise TaskEmbeddingStoreError("task bank row schema mismatch")
                text = str(row["text"])
                identity = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if identity != row["text_id"] or identity in self.entries:
                    raise TaskEmbeddingStoreError("task bank identity mismatch/duplicate")
                self.entries[identity] = (str(row["path"]), str(row["sha256"]))
        if not self.entries:
            raise TaskEmbeddingStoreError("task bank is empty")
        if int(receipt["unique_text_count"]) != len(self.entries):
            raise TaskEmbeddingStoreError("task bank receipt cardinality mismatch")
        self._cache: dict[str, torch.Tensor] = {}

    def get(self, text: str) -> torch.Tensor:
        identity = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = self._cache.get(identity)
        if cached is not None:
            return cached
        try:
            relative, expected = self.entries[identity]
        except KeyError as exc:
            raise TaskEmbeddingStoreError("task text is absent from sealed bank") from exc
        path = (self.root / relative).resolve(strict=True)
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise TaskEmbeddingStoreError("task embedding path escapes bank") from exc
        if path.is_symlink() or sha256_file(path) != expected:
            raise TaskEmbeddingStoreError("task embedding SHA mismatch")
        with safe_open(path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"embedding"}:
                raise TaskEmbeddingStoreError("task embedding tensor fields mismatch")
            value = handle.get_tensor("embedding").float()
        if value.shape != (2048,) or not bool(torch.isfinite(value).all()):
            raise TaskEmbeddingStoreError("task embedding ABI is invalid")
        self._cache[identity] = value
        return value
