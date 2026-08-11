from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.build_wm3d_v8_proprio_sidecars import (
    _extract_bridge_robot_state,
    _publish_bytes_no_clobber,
    _robocasa_model_source_frames,
    _source_manifest_rows,
)
from scripts.preflight_wm3d_v8_stage0_causal_dual_view import (
    _validate_contract_and_objective,
)
from scripts.wm3d_v8_preflight_common import _Checks
from wm3d_v3.data.v8_proprio_contract import (
    V8_EMBODIMENT_VOCAB,
    V8_EMBODIMENT_VOCAB_SHA256,
    V8_PROPRIO_ANCHOR,
    V8_PROPRIO_INDEX_SCHEMA,
    V8_PROPRIO_LAYOUT,
    V8_PROPRIO_SCHEMA,
    V8_PROPRIO_STATS_SCHEMA,
    V8ProprioContractError,
    V8ProprioStore,
    encode_bridge_state,
    encode_droid_state,
    encode_robocasa_state16,
    panda_finger_qpos_to_close01,
    sha256_file,
)
from wm3d_v3.models.action_policy import (
    ActionChunkPolicy,
    ActionChunkPolicyConfig,
)
from wm3d_v3.training.train import (
    action_policy_kwargs_from_targets,
    batch_to_device,
    build_v8_action_policy_contract,
    load_train_config,
    validate_action_pretraining_preflight,
)
from wm3d_v3.training.v8_action_policy_transition import (
    V8ActionPolicyTransitionError,
    action_contract_sha256,
    checkpoint_config_sha256,
    validate_v8_action_policy_contract,
    validate_v8_stage0_checkpoint_payload,
)


def _seal_pending(value, counter: list[int] | None = None):
    counter = counter if counter is not None else [1]
    if isinstance(value, dict):
        return {key: _seal_pending(item, counter) for key, item in value.items()}
    if isinstance(value, list):
        return [_seal_pending(item, counter) for item in value]
    if isinstance(value, str) and value.startswith("PENDING_"):
        digest = f"{counter[0]:064x}"
        counter[0] += 1
        return digest
    return value


def _v3_config() -> dict:
    config = load_train_config(
        Path(
            "configs/"
            "wm3d_v8_stage0_causal_dual_view_unified_action_canary_v3.yaml"
        )
    )
    return _seal_pending(config)


def _write_store(tmp_path: Path) -> tuple[V8ProprioStore, Path]:
    raw = np.asarray(
        [
            [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.2, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    payload_path = tmp_path / "clip.npz"
    np.savez_compressed(
        payload_path,
        schema=np.asarray(V8_PROPRIO_SCHEMA),
        identity=np.asarray("clip"),
        split=np.asarray("train"),
        source=np.asarray("robocasa"),
        embodiment=np.asarray("panda_robocasa_libero"),
        embodiment_id=np.asarray(2, dtype=np.int64),
        source_state_sha256=np.asarray("a" * 64),
        frame_indices=np.asarray([0, 1], dtype=np.int64),
        source_frame_indices=np.asarray([0, 4], dtype=np.int64),
        proprio_raw=raw,
    )
    payload_sha = sha256_file(payload_path)
    index_path = tmp_path / "index.jsonl"
    row = {
        "schema": V8_PROPRIO_INDEX_SCHEMA,
        "identity": "clip",
        "split": "train",
        "source": "robocasa",
        "embodiment": "panda_robocasa_libero",
        "embodiment_id": 2,
        "source_state_sha256": "a" * 64,
        "frame_count": 2,
        "path": str(payload_path),
        "sha256": payload_sha,
    }
    index_path.write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
    )
    index_sha = sha256_file(index_path)
    stats_path = tmp_path / "stats.npz"
    np.savez_compressed(
        stats_path,
        schema=np.asarray(V8_PROPRIO_STATS_SCHEMA),
        split=np.asarray("train"),
        source=np.asarray("robocasa"),
        index_sha256=np.asarray(index_sha),
        embodiment_vocab_sha256=np.asarray(V8_EMBODIMENT_VOCAB_SHA256),
        layout=np.asarray(V8_PROPRIO_LAYOUT),
        mean=np.zeros(10, dtype=np.float32),
        std=np.ones(10, dtype=np.float32),
        sample_count=np.asarray(2, dtype=np.int64),
    )
    store = V8ProprioStore(
        index_path=index_path,
        index_sha256=index_sha,
        stats_path=stats_path,
        stats_sha256=sha256_file(stats_path),
        source="robocasa",
        split="train",
        expected_identities=["clip"],
    )
    return store, payload_path


def test_proprio_builder_is_a_directly_runnable_cli() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_wm3d_v8_proprio_sidecars.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--source-type" in result.stdout


def test_proprio_sidecar_publication_is_atomic_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "clip.npz"
    competing_payload = b"concurrent-writer"

    def concurrent_link(_source, destination, *, follow_symlinks=False):
        assert follow_symlinks is False
        Path(destination).write_bytes(competing_payload)
        raise FileExistsError(destination)

    monkeypatch.setattr(os, "link", concurrent_link)
    with pytest.raises(RuntimeError, match="no-clobber conflict"):
        _publish_bytes_no_clobber(target, b"our-payload")
    assert target.read_bytes() == competing_payload


def test_proprio_builder_rejects_duplicate_source_manifest_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duplicate = SimpleNamespace(clip_id="duplicate")
    monkeypatch.setattr(
        "scripts.build_wm3d_v8_proprio_sidecars.read_manifest",
        lambda _path: [duplicate, duplicate],
    )
    with pytest.raises(RuntimeError, match="duplicate source manifest identity"):
        _source_manifest_rows(tmp_path / "source.jsonl")


def test_real_robot_state_adapters_obey_the_same_proprio10_abi() -> None:
    assert panda_finger_qpos_to_close01(
        np.asarray([0.04, -0.04])
    ) == pytest.approx(0.0)
    assert panda_finger_qpos_to_close01(
        np.asarray([0.0, 0.0])
    ) == pytest.approx(1.0)
    with pytest.raises(V8ProprioContractError, match="outside sealed"):
        panda_finger_qpos_to_close01(np.asarray([0.06, -0.06]))
    assert panda_finger_qpos_to_close01(
        np.asarray([0.0405, -0.0405])
    ) == pytest.approx(0.0)

    robocasa = np.zeros(16, dtype=np.float32)
    robocasa[7:10] = [0.1, 0.2, 0.3]
    robocasa[10:14] = [0.0, 0.0, 0.0, 1.0]
    robocasa[14:16] = [0.04, -0.04]
    encoded = encode_robocasa_state16(robocasa)
    assert encoded.shape == (10,)
    assert encoded[-1] == pytest.approx(0.0, abs=1e-6)

    bridge = encode_bridge_state(
        np.asarray([0, 0, 0, 0, 0, 0, 1.02], dtype=np.float32)
    )
    assert bridge[-1] == pytest.approx(0.0)
    with pytest.raises(V8ProprioContractError, match="Bridge open01"):
        encode_bridge_state(np.asarray([0, 0, 0, 0, 0, 0, 1.06]))

    droid = encode_droid_state(
        np.asarray([0, 0, 0, 0, 0, 0], dtype=np.float32), 1.0
    )
    assert droid[-1] == pytest.approx(1.0)


def test_bridge_builder_reads_only_exact_observation_state() -> None:
    episode = {
        "steps": [
            {"observation": {"state": np.arange(7, dtype=np.float32)}},
            {"observation": {"state": np.arange(7, dtype=np.float32) + 1}},
        ]
    }
    pose, grip = _extract_bridge_robot_state(episode)
    assert pose.shape == (2, 6)
    np.testing.assert_array_equal(grip, np.asarray([6.0, 7.0]))
    with pytest.raises(RuntimeError, match="omits observation.state"):
        _extract_bridge_robot_state({"steps": [{"observation": {}}]})


def test_robocasa_model_index_resolves_exact_sealed_source_frame(
    tmp_path: Path,
) -> None:
    path = tmp_path / "timing.npz"
    np.savez(
        path,
        model_timestamps=np.asarray([16.25, 16.45], dtype=np.float64),
        native_frame_indices=np.arange(325, 334, dtype=np.int64),
        native_fps=np.asarray(20.0, dtype=np.float32),
        source_control_hz=np.asarray(20.0, dtype=np.float32),
        model_control_hz=np.asarray(5.0, dtype=np.float32),
    )
    with np.load(path, allow_pickle=False) as archive:
        frames = _robocasa_model_source_frames(
            archive, identity="clip", world_count=2
        )
    np.testing.assert_array_equal(frames, np.asarray([325, 329]))

    invalid = tmp_path / "invalid_timing.npz"
    np.savez(
        invalid,
        model_timestamps=np.asarray([16.251, 16.45], dtype=np.float64),
        native_frame_indices=np.arange(325, 334, dtype=np.int64),
        native_fps=np.asarray(20.0, dtype=np.float32),
        source_control_hz=np.asarray(20.0, dtype=np.float32),
        model_control_hz=np.asarray(5.0, dtype=np.float32),
    )
    with np.load(invalid, allow_pickle=False) as archive:
        with pytest.raises(RuntimeError, match="non-exact RoboCasa"):
            _robocasa_model_source_frames(
                archive, identity="clip", world_count=2
            )


def test_content_addressed_store_requires_exact_frame_and_payload_hash(
    tmp_path: Path,
) -> None:
    store, payload_path = _write_store(tmp_path)
    sample = store.current("clip", 1)
    assert np.array_equal(sample.raw, sample.normalized)
    assert sample.anchor_frame_index == 1
    assert sample.embodiment_id == 2
    with pytest.raises(V8ProprioContractError, match="no exact proprio frame"):
        store.current("clip", 2)

    with payload_path.open("ab") as handle:
        handle.write(b"tamper")
    replacement = V8ProprioStore(
        index_path=store.index_path,
        index_sha256=store.index_sha256,
        stats_path=store.stats_path,
        stats_sha256=store.stats_sha256,
        source="robocasa",
        split="train",
        expected_identities=["clip"],
    )
    with pytest.raises(V8ProprioContractError, match="payload SHA mismatch"):
        replacement.current("clip", 0)


def test_required_proprio_and_embodiment_are_used_by_policy_and_receive_gradient() -> None:
    torch.manual_seed(31)
    policy = ActionChunkPolicy(
        ActionChunkPolicyConfig(
            token_dim=16,
            task_dim=8,
            hidden=16,
            n_layers=1,
            n_heads=4,
            chunk_layers=1,
            horizon=8,
            max_context=4,
            dropout=0.0,
            lowdim_dim=10,
            require_lowdim_state=True,
            embodiment_vocab_size=3,
            require_embodiment=True,
        )
    )
    context = torch.randn(2, 4, 4, 16)
    task = torch.randn(2, 8)
    lowdim = torch.randn(2, 10, requires_grad=True)
    embodiment = torch.tensor([0, 2], dtype=torch.long)
    output = policy(
        context,
        task,
        lowdim_state=lowdim,
        embodiment_id=embodiment,
    )
    loss = output["base_policy_pose_norm"].square().mean()
    loss = loss + output["base_policy_gripper_logit"].square().mean()
    loss.backward()
    assert lowdim.grad is not None and float(lowdim.grad.abs().sum()) > 0.0
    assert policy.lowdim_proj[1].weight.grad is not None
    assert float(policy.lowdim_proj[1].weight.grad.abs().sum()) > 0.0
    assert policy.embodiment_embed.weight.grad is not None
    assert float(policy.embodiment_embed.weight.grad.abs().sum()) > 0.0

    with pytest.raises(ValueError, match="lowdim_state is required"):
        policy(context, task, embodiment_id=embodiment)
    with pytest.raises(ValueError, match="embodiment_id is required"):
        policy(context, task, lowdim_state=lowdim.detach())


def test_batch_device_path_fails_closed_and_forwards_current_state() -> None:
    batch = {
        "s_in": torch.zeros(1, 16, 1, 2),
        "c": torch.zeros(1, 2),
        "action_tgt": torch.zeros(1, 8, 7),
        "action_tgt_norm": torch.zeros(1, 8, 6),
        "lowdim_state": torch.zeros(1, 10),
        "policy_proprio_raw": torch.zeros(1, 10),
        "embodiment_id": torch.tensor([2]),
        "policy_proprio_stats_key": ["robocasa:" + "a" * 64],
        "policy_proprio_anchor": [V8_PROPRIO_ANCHOR],
        "policy_proprio_frame_index": torch.tensor([15]),
        "action_frame_indices": torch.arange(15, 23, dtype=torch.long)[
            None, :
        ],
    }
    _s, _c, _action, _rgb, target = batch_to_device(
        batch, torch.device("cpu"), 8, direct_policy_only=True
    )
    kwargs = action_policy_kwargs_from_targets(target)
    assert tuple(kwargs["lowdim_state"].shape) == (1, 10)
    assert kwargs["embodiment_id"].tolist() == [2]

    broken = dict(batch)
    broken.pop("embodiment_id")
    with pytest.raises(RuntimeError, match="incomplete V8 policy proprio"):
        batch_to_device(
            broken, torch.device("cpu"), 8, direct_policy_only=True
        )

    wrong_anchor = dict(batch)
    wrong_anchor["policy_proprio_frame_index"] = torch.tensor([14])
    with pytest.raises(RuntimeError, match="first policy action target"):
        batch_to_device(
            wrong_anchor, torch.device("cpu"), 8, direct_policy_only=True
        )


def test_legacy_lowdim_state_does_not_impersonate_v8_proprio() -> None:
    batch = {
        "s_in": torch.zeros(1, 16, 1, 2),
        "c": torch.zeros(1, 2),
        "action_tgt": torch.zeros(1, 8, 7),
        "action_tgt_norm": torch.zeros(1, 8, 6),
        "lowdim_state": torch.zeros(1, 4),
    }
    _s, _c, _action, _rgb, target = batch_to_device(
        batch, torch.device("cpu"), 8, direct_policy_only=True
    )
    assert tuple(target["lowdim_state"].shape) == (1, 4)
    assert "policy_proprio_stats_keys" not in target


def test_v3_config_and_checkpoint_contract_seal_current_state() -> None:
    config = _v3_config()
    assert validate_action_pretraining_preflight(config) is True
    checks = _Checks("structure")
    sources = _validate_contract_and_objective(checks, config)
    assert checks.errors == []
    assert set(sources) == {"oxe_bridge_action", "oxe_droid_action"}

    contract = build_v8_action_policy_contract(config)
    assert contract is not None
    assert contract["schema"] == "wm3d_v8_stage0_action_policy_contract_v3"
    assert contract["proprio"]["schema"] == V8_PROPRIO_SCHEMA
    assert contract["proprio"]["anchor"] == V8_PROPRIO_ANCHOR
    assert contract["proprio"]["embodiment_vocab"] == V8_EMBODIMENT_VOCAB
    assert set(contract["proprio"]["sources"]) == {
        "robocasa",
        "droid",
        "bridge",
    }
    assert validate_v8_action_policy_contract(contract) == contract

    invalid = json.loads(json.dumps(contract))
    invalid["proprio"]["sources"].pop("bridge")
    invalid["contract_sha256"] = action_contract_sha256(invalid)
    with pytest.raises(V8ActionPolicyTransitionError, match="proprio.sources"):
        validate_v8_action_policy_contract(invalid)


def test_checkpoint_config_and_action_contract_versions_are_strictly_bound() -> None:
    v2_config = _seal_pending(
        load_train_config(
            Path(
                "configs/"
                "wm3d_v8_stage0_causal_dual_view_unified_action_canary_v2.yaml"
            )
        )
    )
    v3_config = _v3_config()
    v2_contract = build_v8_action_policy_contract(v2_config)
    v3_contract = build_v8_action_policy_contract(v3_config)
    assert v2_contract is not None and v3_contract is not None

    def payload(config: dict, contract: dict) -> dict:
        sealed = json.loads(json.dumps(config))
        resolved_sha = checkpoint_config_sha256(sealed)
        sealed.setdefault("train", {})["resolved_config_sha256"] = resolved_sha
        return {
            "model": {"action_policy.weight": torch.ones(1)},
            "step": 100,
            "resolved_config_sha256": resolved_sha,
            "action_policy_contract": contract,
            "cfg": sealed,
        }

    with pytest.raises(V8ActionPolicyTransitionError, match="schema mismatch"):
        validate_v8_stage0_checkpoint_payload(payload(v2_config, v3_contract))
    with pytest.raises(V8ActionPolicyTransitionError, match="schema mismatch"):
        validate_v8_stage0_checkpoint_payload(payload(v3_config, v2_contract))


def test_training_preflight_binds_schema_to_proprio_mode_in_both_directions() -> None:
    v3_claiming_v2 = _v3_config()
    v3_claiming_v2["contract"]["schema"] = (
        "wm3d_v8_stage0_causal_dual_view_unified_action_canary_v2"
    )
    v3_claiming_v2["contract"]["action_policy_contract_schema"] = (
        "wm3d_v8_stage0_action_policy_contract_v2"
    )

    v2_claiming_v3 = _seal_pending(
        load_train_config(
            Path(
                "configs/"
                "wm3d_v8_stage0_causal_dual_view_unified_action_canary_v2.yaml"
            )
        )
    )
    v2_claiming_v3["contract"]["schema"] = (
        "wm3d_v8_stage0_causal_dual_view_unified_action_canary_v3"
    )
    v2_claiming_v3["contract"]["action_policy_contract_schema"] = (
        "wm3d_v8_stage0_action_policy_contract_v3"
    )

    for invalid in (v3_claiming_v2, v2_claiming_v3):
        checks = _Checks("structure")
        _validate_contract_and_objective(checks, invalid)
        assert any("v8_proprio_enabled" in error for error in checks.errors)
        with pytest.raises(RuntimeError, match="schema/proprio mode mismatch"):
            validate_action_pretraining_preflight(invalid)
