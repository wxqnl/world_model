"""Prompt parity and cache/runtime incompatibility must fail before model loading."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from wm3d.data.admitted_context import ContextUnion
from wm3d.data.foundation_input import (
    FoundationContextStore, FoundationInputError, build_foundation_input_adapter,
)
from wm3d.data.foundation_prompt import RAW_TASK, WSA_INSTRUCTION, format_wan_prompt
from wm3d.models.foundation_world_model import FoundationWorldModelConfig, FoundationModelError


def _store(root, schema, prompt_format=None):
    root.mkdir()
    payload = {"schema": schema, "tasks": ["pick"]}
    if prompt_format is not None:
        payload["prompt_format"] = prompt_format
    (root / "index.json").write_text(json.dumps(payload))
    save_file({"context": torch.ones(1, 3, 4),
               "mask": torch.ones(1, 3, dtype=torch.bool)},
              str(root / "context.safetensors"))
    return FoundationContextStore.from_directory(root)


def test_official_wsa_prompt_parity():
    path = Path("/data/Minko/reference_repos/WSA_decoder_review_20260825/src/lerobot/policies/WSA_Large/text_cache.py")
    if not path.exists():
        pytest.skip("local upstream reference unavailable")
    spec = importlib.util.spec_from_file_location("wsa_prompt_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for task in ("put the ranch bottle into the pot", "pick up the blue cup and put it into the brown cup."):
        assert format_wan_prompt(task, WSA_INSTRUCTION) == module.build_wsa_large_prompt(task)
        assert format_wan_prompt(task, RAW_TASK) == task


def test_legacy_v1_is_explicit_raw_task(tmp_path):
    store = _store(tmp_path / "v1", "wm3d_foundation_context_store_v1")
    assert store.prompt_format == RAW_TASK
    assert store.lookup(["pick"])[0].shape == (1, 3, 4)


def test_v2_preserves_original_task_lookup_and_prompt_identity(tmp_path):
    store = _store(tmp_path / "v2", "wm3d_foundation_context_store_v2", WSA_INSTRUCTION)
    assert store.prompt_format == WSA_INSTRUCTION
    store.lookup(["pick"])
    with pytest.raises(FoundationInputError, match="absent"):
        store.lookup([format_wan_prompt("pick", WSA_INSTRUCTION)])


def test_mixed_prompt_union_rejected(tmp_path):
    a = _store(tmp_path / "a", "wm3d_foundation_context_store_v1")
    b = _store(tmp_path / "b", "wm3d_foundation_context_store_v2", WSA_INSTRUCTION)
    with pytest.raises(ValueError, match="different prompt"):
        ContextUnion([a, b])


@pytest.mark.parametrize("fmt", ["unknown", None])
def test_v2_rejects_missing_or_unknown_format(tmp_path, fmt):
    with pytest.raises(FoundationInputError):
        _store(tmp_path / "bad", "wm3d_foundation_context_store_v2", fmt)


def test_adapter_rejects_context_model_format_mismatch_before_encoder(tmp_path):
    root = tmp_path / "raw"
    _store(root, "wm3d_foundation_context_store_v1")
    runtime = {"model_profile": {"model": {
        "wan_context_store_path": str(root), "wan_prompt_format": WSA_INSTRUCTION}}}
    with pytest.raises(FoundationInputError, match="differs from model"):
        build_foundation_input_adapter(runtime, device="cpu")


def test_unknown_model_format_rejected():
    with pytest.raises(FoundationModelError, match="prompt format"):
        FoundationWorldModelConfig(wan_prompt_format="unknown").validate()
