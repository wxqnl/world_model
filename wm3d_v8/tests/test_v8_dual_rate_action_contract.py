from __future__ import annotations

import numpy as np
import pytest
import torch

from wm3d_v3.data.v7_action_contract import resample_canonical_actions
from wm3d_v3.data.v8_action_contract import (
    ACTION_DIM,
    DYNAMICS_ACTION_DIM,
    POLICY_HISTORY_DIM,
    POLICY_HISTORY_LEN,
    POLICY_HORIZON,
    SUBSTEPS_PER_WORLD,
    PoseStats,
    V8ActionContractError,
    build_coarse_5hz_window_contract,
    build_real_20hz_window_contract,
    compose_base_delta_actions_np,
    compose_base_delta_actions_torch,
    require_v8_pinned_file,
)


def _fine_actions(count: int) -> np.ndarray:
    index = np.arange(count, dtype=np.float32)
    actions = np.zeros((count, ACTION_DIM), dtype=np.float32)
    actions[:, 0] = 0.001 + 0.00001 * index
    actions[:, 1] = 0.0002 * np.sin(index / 7.0)
    actions[:, 2] = -0.0001 * np.cos(index / 5.0)
    actions[:, 3] = 0.0015
    actions[:, 4] = -0.0007
    actions[:, 5] = 0.0005
    actions[:, 6] = np.where((index.astype(np.int64) // 11) % 2 == 0, -1.0, 1.0)
    return actions


def _stats(key: str, scale: float) -> PoseStats:
    return PoseStats(
        mean=np.zeros(6, dtype=np.float32),
        std=np.full(6, scale, dtype=np.float32),
        key=key,
    )


def test_numpy_and_torch_so3_composition_match_and_backpropagate() -> None:
    fine = _fine_actions(24).reshape(6, 4, 7)
    expected = compose_base_delta_actions_np(fine)
    tensor = torch.tensor(fine, dtype=torch.float64, requires_grad=True)
    observed = compose_base_delta_actions_torch(tensor)
    assert np.allclose(observed.detach().numpy(), expected, atol=2.0e-6, rtol=0.0)
    observed[..., :6].square().sum().backward()
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()
    assert float(tensor.grad[..., :6].abs().sum()) > 0.0


def test_real_20hz_window_recovers_world_actions_and_emits_fine_labels() -> None:
    fine = _fine_actions(160)
    world = resample_canonical_actions(fine, source_hz=20.0, target_hz=5.0)
    result = build_real_20hz_window_contract(
        fine_actions=fine,
        world_actions=world,
        world_action_start=15,
        world_horizon=8,
        fine_stats=_stats("robocasa20", 0.02),
        coarse_stats=_stats("robocasa5", 0.08),
    )
    assert result["v8_dynamics_action_cond"].shape == (8, DYNAMICS_ACTION_DIM)
    assert result["policy_action_tgt"].shape == (POLICY_HORIZON, 7)
    assert result["policy_action_valid_mask"].all()
    assert result["policy_action_coarse_valid_mask"].all()
    assert result["action_history"].shape == (
        POLICY_HISTORY_LEN,
        POLICY_HISTORY_DIM,
    )
    assert np.all(result["action_history"][:, 7] == np.float32(0.05))
    assert np.all(result["action_history"][:, 8] == np.float32(1.0))
    # Packed validity and dt occupy the final eight coordinates.
    packed = result["v8_dynamics_action_cond"]
    assert np.all(packed[:, 28:32] == 1.0)
    assert np.all(packed[:, 32:36] == np.float32(0.05))


def test_real_20hz_window_fails_if_sealed_world_action_disagrees() -> None:
    fine = _fine_actions(160)
    world = resample_canonical_actions(fine, source_hz=20.0, target_hz=5.0)
    world[15, 0] += 0.1
    with pytest.raises(V8ActionContractError, match="do not reproduce"):
        build_real_20hz_window_contract(
            fine_actions=fine,
            world_actions=world,
            world_action_start=15,
            world_horizon=8,
            fine_stats=_stats("robocasa20", 0.02),
            coarse_stats=_stats("robocasa5", 0.08),
        )


def test_coarse_oxe_contract_never_fabricates_20hz_labels() -> None:
    episode = resample_canonical_actions(
        _fine_actions(192), source_hz=20.0, target_hz=5.0
    )
    result = build_coarse_5hz_window_contract(
        episode_actions=episode,
        action_indices=np.arange(15, 23),
        coarse_stats=_stats("bridge5", 0.08),
    )
    assert not result["policy_action_valid_mask"].any()
    assert result["policy_action_coarse_valid_mask"].all()
    packed = result["v8_dynamics_action_cond"]
    assert np.all(packed[:, 28] == 1.0)
    assert np.all(packed[:, 29:32] == 0.0)
    assert np.all(packed[:, 32] == np.float32(0.2))
    assert np.all(packed[:, 33:36] == 0.0)
    history = result["action_history"]
    valid_slots = np.flatnonzero(history[:, 8] > 0.5)
    assert valid_slots.tolist() == [3, 7, 11, 15]
    assert np.all(history[valid_slots, 7] == np.float32(0.2))
    assert np.all(history[np.setdiff1d(np.arange(16), valid_slots), :].sum(axis=1) == 0.0)


def test_torch_composition_requires_a_real_slot_per_group() -> None:
    actions = torch.zeros(2, SUBSTEPS_PER_WORLD, ACTION_DIM)
    valid = torch.zeros(2, SUBSTEPS_PER_WORLD, dtype=torch.bool)
    valid[0, 0] = True
    with pytest.raises(V8ActionContractError, match="at least one valid"):
        compose_base_delta_actions_torch(actions, valid)


def test_runtime_normalizer_pin_rejects_mutation_and_symlink(tmp_path) -> None:
    path = tmp_path / "stats.npz"
    path.write_bytes(b"sealed-action-normalizer-v1")
    import hashlib

    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert require_v8_pinned_file(
        path, expected, label="test normalizer"
    ) == path.resolve()
    path.write_bytes(b"mutated-action-normalizer-v2")
    with pytest.raises(V8ActionContractError, match="SHA256 mismatch"):
        require_v8_pinned_file(path, expected, label="test normalizer")
    link = tmp_path / "stats-link.npz"
    link.symlink_to(path)
    with pytest.raises(V8ActionContractError, match="must not be a symlink"):
        require_v8_pinned_file(
            link,
            hashlib.sha256(path.read_bytes()).hexdigest(),
            label="test normalizer",
        )
