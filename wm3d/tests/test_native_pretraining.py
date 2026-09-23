"""Contract/gradient regressions, not a substitute for the real GPU pilot."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from wm3d.data.episode_selection import select_training_entries
from wm3d.models.foundation_native import TrajectoryReadout, TrajectoryStateConditioner, trajectory_loss, _tensor_loss_metrics
from wm3d.models.foundation_world_model import FoundationModelError, foundation_config_from_mapping
from wm3d.training.runtime_contract import validate_runtime_profile


def config():
    root = Path(__file__).resolve().parents[1]
    return foundation_config_from_mapping(yaml.safe_load((root / "configs/model/native_wan.yaml").read_text())["model"])


def test_upstream_float_diagnostics_do_not_break_loss_accumulation():
    parameter = torch.tensor(2.0, requires_grad=True)
    total = parameter.square()
    result = _tensor_loss_metrics(total, {"loss_video": total.detach(), "padding_fraction": 0.25})
    assert result["total"] is total
    for value in result.values():
        assert value.ndim == 0 and torch.isfinite(value)
        torch.zeros_like(value).add_(value.detach())
    result["total"].backward()
    assert parameter.grad.item() == 4.0


def test_native_pretraining_rejects_wsa_checkpoint_and_frozen_new_experts():
    cfg = config()
    cfg.validate()
    for changes in ({"wsa_pretrained_path": "/tmp/WSA-Large"},
                    {"pretrained_interface_init": "pretrained_action_prefix"},
                    {"finetune_mode": "adapters"},
                    {"wan_action_backbone_path": ""},
                    {"native_3d_rollout_loss_weight": 0.05}):
        with pytest.raises(FoundationModelError):
            replace(cfg, **changes).validate()


def test_trajectory_conditioning_preserves_view_horizon_layout_and_gradients():
    cfg = config()
    module = TrajectoryStateConditioner(cfg, hidden_dim=768)
    current = torch.randn(1, 3, 64, 2048, requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    output = module(current, mask).view(1, 3, 8, 64, 768)
    assert output[:, 2].count_nonzero() == 0
    assert torch.equal(output[:, :, 0], output[:, :, 7])  # observation, not future truth
    changed = current.detach().clone()
    changed[:, 1] = changed[:, 1].flip(-1)
    second = module(changed, mask).view_as(output)
    assert torch.equal(output[:, 0], second[:, 0])
    assert not torch.equal(output[:, 1], second[:, 1])
    output.square().mean().backward()
    assert current.grad[:, 2].count_nonzero() == 0
    assert torch.isfinite(module.projection.weight.grad).all()
    assert module.projection.weight.grad.count_nonzero() > 0


def test_trajectory_readout_loss_all_horizons_masked_and_teacher_detached():
    torch.manual_seed(3407)
    readout = TrajectoryReadout(768, 2048)
    hidden = torch.randn(1, 3, 8, 64, 768, requires_grad=True)
    target = torch.randn(1, 3, 8, 64, 2048, requires_grad=True)
    current = torch.randn(1, 3, 64, 2048, requires_grad=True)
    mask = torch.ones(1, 3, 8, dtype=torch.bool)
    mask[:, 2, 3:] = False
    prediction = readout(hidden)
    loss, metrics = trajectory_loss([prediction], target, mask, current)
    loss.backward()
    assert torch.isfinite(loss)
    assert target.grad is None and current.grad is None
    assert hidden.grad[:, 2, 3:].count_nonzero() == 0
    assert (hidden.grad[:, 0].abs().sum(dim=(-1, -2)) > 0).all()
    assert all(torch.isfinite(value) for value in metrics.values())
    assert all(f"native_3d_h{h}_direction" in metrics for h in range(1, 9))


def test_upstream_mask_keeps_native_bridge_inside_rgb_action_and_excludes_future_video():
    from wm3d.models.foundation_world_model import _load_upstream_symbols
    source = Path(config().upstream_source_root)
    if not source.is_dir():
        pytest.skip("upstream WSA source is not installed")
    _load_upstream_symbols(source)
    from lerobot.policies.WSA_Large.core.models.wan22.wsa_large import WSALarge
    from lerobot.policies.WSA_Large.core.models.wan22.wan_video_dit import WanVideoDiT
    # Exercise the actual production mask with the real K8*3*P64 sequence.
    core = object.__new__(WSALarge)
    torch.nn.Module.__init__(core)
    video = object.__new__(WanVideoDiT)
    torch.nn.Module.__init__(video)
    video.video_attention_mask_mode = "first_frame_causal"
    core.video_expert = video
    core.future_3d_expert = SimpleNamespace(da3_num_views=3, query_tokens_per_view=512)
    core.future_3d_view_attention_layout = "robotwin"
    mask = core._build_mot_attention_mask(720, 32, 240, torch.device("cpu"), 1536, (3, 24, 10))
    assert mask[240:720, 720:2256].all()
    assert mask[2256:, 720:2256].all()
    assert mask[720:2256, 2256:].all()
    assert not mask[720:, 240:720].any()
    assert not mask[:240, 240:].any()


def test_reuse_saved_selection_preserves_sources_and_window_proportions(tmp_path):
    data = tmp_path / "data"
    metadata = data / "metadata"
    metadata.mkdir(parents=True)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({
        "schema": "v7_training_episode_selection_v1", "data_root": str(data),
        "episodes": {"oxe_a": ["a"], "robocasa_b": ["b"]},
        "expected_windows": {"oxe_a": 1, "robocasa_b": 2},
    }))
    rows = [SimpleNamespace(source=s, split="train", feature_shard=f)
            for s, f in [("oxe_a", "a"), ("robocasa_b", "b"), ("robocasa_b", "b"), ("robocasa_b", "c")]]
    identity = {"a": ("oxe_a", "a"), "b": ("robocasa_b", "b"), "c": ("robocasa_b", "c")}
    selected, weights = select_training_entries(rows, identity, path, metadata)
    assert len(selected) == 3
    assert weights == {"oxe_a": 1 / 3, "robocasa_b": 2 / 3}
    with pytest.raises(ValueError, match="another data root"):
        select_training_entries(rows, identity, path, tmp_path / "other" / "metadata")


def test_two_node_and_four_node_runtime_contracts():
    root = Path(__file__).resolve().parents[1]
    for filename, world in (("h100_16_native_pilot.yaml", 16), ("h100_32_native.yaml", 32)):
        profile = yaml.safe_load((root / "configs/runtime" / filename).read_text())
        profile["train"].pop("train_episode_allowlist")  # site path is checked at materialization
        validate_runtime_profile(profile)
        assert profile["train"]["global_batch_size"] == world
        assert profile["distributed"]["shard_degree"] == 8
