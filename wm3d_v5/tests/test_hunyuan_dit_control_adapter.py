from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from wm3d_v3.models.hunyuan_dit_control_adapter import (
    HunyuanDiTControlAdapter,
    HunyuanDiTControlConfig,
    HunyuanDiTControlInjector,
    load_hunyuan_dit_control_checkpoint,
    save_hunyuan_dit_control_checkpoint,
)


class AddOneBlock(nn.Module):
    def forward(self, x):
        return x + 1.0


class FakeTransformer(nn.Module):
    def __init__(self, n_double: int = 2, n_single: int = 2):
        super().__init__()
        self.double_blocks = nn.ModuleList([AddOneBlock() for _ in range(n_double)])
        self.single_blocks = nn.ModuleList([AddOneBlock() for _ in range(n_single)])

    def forward(self, img_tokens: torch.Tensor) -> torch.Tensor:
        x = img_tokens
        for block in self.double_blocks:
            x = block(x)
        for block in self.single_blocks:
            x = block(x)
        return x


def tiny_controls(batch: int = 2, horizon: int = 2, token_dim: int = 4):
    g = torch.Generator().manual_seed(7)
    return {
        "pred_tokens": torch.randn(batch, horizon, 4, token_dim, generator=g),
        "depth": torch.randn(batch, horizon, 6, 6, generator=g),
        "motion_hint": torch.randn(batch, horizon, 1, 6, 6, generator=g),
        "contact_hint": torch.randn(batch, horizon, 1, 6, 6, generator=g),
        "rough_rgb": torch.randn(batch, horizon, 3, 6, 6, generator=g),
        "context_rgb": torch.randn(batch, 3, 6, 6, generator=g),
        "action_cond": torch.randn(batch, horizon, 7, generator=g),
        "task_emb": torch.randn(batch, 16, generator=g),
    }


def tiny_adapter(**overrides) -> HunyuanDiTControlAdapter:
    cfg = HunyuanDiTControlConfig(
        token_dim=4,
        token_grid=2,
        hidden=8,
        dit_hidden=8,
        task_dim=16,
        double_blocks=2,
        single_blocks=2,
        **overrides,
    )
    return HunyuanDiTControlAdapter(cfg)


def test_zero_init_residuals_are_zero_and_fake_transformer_is_unchanged():
    adapter = tiny_adapter()
    controls = tiny_controls(batch=2)
    adapter.prepare_controls(**controls)
    img_tokens = torch.randn(2, 8, 8)

    assert torch.count_nonzero(adapter.double_residual(0, img_tokens, latent_shape=None, batch_size=2)) == 0
    assert torch.count_nonzero(adapter.single_residual(0, img_tokens, img_token_len=8, latent_shape=None, batch_size=2)) == 0

    transformer = FakeTransformer(n_double=2, n_single=2)
    expected = transformer(img_tokens)
    injector = HunyuanDiTControlInjector(transformer, adapter)
    with injector.use_controls(**controls):
        actual = transformer(img_tokens)
    assert torch.equal(actual, expected)


def test_cfg_batch_conditional_only_zeroes_uncond_half_and_keeps_cond_half():
    adapter = tiny_adapter()
    controls = tiny_controls(batch=1)
    adapter.prepare_controls(**controls)
    with torch.no_grad():
        adapter.double_gates[0].fill_(1.0)
        adapter.double_projections[0].weight.copy_(torch.eye(8))
        adapter.double_projections[0].bias.zero_()

    img_tokens = torch.zeros(2, 8, 8)
    residual = adapter.double_residual(0, img_tokens, latent_shape=None, batch_size=2)

    assert torch.count_nonzero(residual[0]) == 0
    assert torch.count_nonzero(residual[1]) > 0


def test_injector_hooks_restore_after_context():
    adapter = tiny_adapter()
    transformer = FakeTransformer(n_double=2, n_single=2)
    controls = tiny_controls(batch=1)
    blocks = list(transformer.double_blocks) + list(transformer.single_blocks)
    assert sum(len(block._forward_hooks) for block in blocks) == 0

    injector = HunyuanDiTControlInjector(transformer, adapter)
    with injector.use_controls(**controls):
        assert sum(len(block._forward_hooks) for block in blocks) == 4
        assert adapter.control_state is not None

    assert sum(len(block._forward_hooks) for block in blocks) == 0
    assert adapter.control_state is None


def test_zero_init_control_branch_is_noop_but_trainable():
    adapter = tiny_adapter()
    controls = tiny_controls(batch=1)
    transformer = FakeTransformer(n_double=2, n_single=2)
    img_tokens = torch.randn(1, 8, 8)

    injector = HunyuanDiTControlInjector(transformer, adapter)
    with injector.use_controls(**controls):
        out = transformer(img_tokens)
        loss = out.sum()
    loss.backward()

    assert torch.count_nonzero(adapter.double_projections[0].weight.grad) > 0
    assert torch.count_nonzero(adapter.single_projections[0].weight.grad) > 0


def test_untrained_backend_adapter_infers_transformer_shape():
    from wm3d_v3.video_backends.hunyuan_dit_control_video import (
        HunyuanDiTControlVideoBackend,
        HunyuanDiTControlVideoBackendConfig,
    )

    transformer = FakeTransformer(n_double=3, n_single=1)
    transformer.hidden_size = 8
    backend = HunyuanDiTControlVideoBackend(
        HunyuanDiTControlVideoBackendConfig(allow_untrained_control=True),
        device="cpu",
    )

    adapter = backend.load_control_adapter(transformer)

    assert adapter.cfg.dit_hidden == 8
    assert adapter.cfg.double_blocks == 3
    assert adapter.cfg.single_blocks == 1


def test_dit_control_backend_rejects_multi_clip_generate_controls():
    from wm3d_v3.video_backends.base import VideoConditionBundle
    from wm3d_v3.video_backends.hunyuan_dit_control_video import HunyuanDiTControlVideoBackend

    bundle = VideoConditionBundle(
        context_rgb=torch.zeros(2, 3, 8, 8),
        pred_tokens=torch.zeros(2, 2, 4, 4),
        depth=torch.zeros(2, 2, 8, 8),
    )

    with pytest.raises(RuntimeError, match="exactly one clip"):
        HunyuanDiTControlVideoBackend._controls_from_bundle(bundle)


def test_control_checkpoint_kind_is_accepted_and_other_kinds_rejected(tmp_path: Path):
    adapter = tiny_adapter()
    good = tmp_path / "control.pt"
    save_hunyuan_dit_control_checkpoint(good, adapter, metrics={"loss": 1.0}, wm_ckpt="wm.pt", step=3)

    loaded, payload = load_hunyuan_dit_control_checkpoint(good, device="cpu")
    assert isinstance(loaded, HunyuanDiTControlAdapter)
    assert payload["kind"] == "hunyuan_dit_control_adapter_v1"
    assert payload["metrics"] == {"loss": 1.0}
    assert payload["wm_ckpt"] == "wm.pt"
    assert payload["step"] == 3

    bad = tmp_path / "flow.pt"
    torch.save({"kind": "hunyuan_flow_denoiser_v1", "model": {}, "cfg": {}}, bad)
    with pytest.raises(RuntimeError, match="hunyuan_dit_control_adapter_v1"):
        load_hunyuan_dit_control_checkpoint(bad, device="cpu")

    latent_like = tmp_path / "latent.pt"
    torch.save({"model": {}, "cfg": {"hidden": 128, "n_blocks": 3}}, latent_like)
    with pytest.raises(RuntimeError, match="kind"):
        load_hunyuan_dit_control_checkpoint(latent_like, device="cpu")


def _tiny_geom(batch=2, horizon=3, grid=2):
    g = torch.Generator().manual_seed(7)
    return {
        "point": torch.randn(batch, horizon, grid, grid, 3, generator=g),
        "pose_geom": torch.randn(batch, horizon, 9, generator=g),
    }


def test_point_pose_conditioning_is_live_but_zero_init_noop():
    adapter = tiny_adapter()
    controls = tiny_controls(batch=2)
    geom = _tiny_geom(batch=2, horizon=controls["pred_tokens"].shape[1])

    st_base = adapter.build_control_state(**controls)
    st_geom = adapter.build_control_state(**controls, **geom)
    # native3d geometry must actually change the fused control features...
    assert not torch.allclose(st_base.features, st_geom.features, atol=1e-6)
    # ...yet the DiT-facing residuals stay an exact no-op at init (zero-init output projections).
    adapter.set_control_state(st_geom)
    img_tokens = torch.randn(2, 8, adapter.cfg.dit_hidden)
    assert torch.count_nonzero(adapter.double_residual(0, img_tokens, latent_shape=None, batch_size=2)) == 0
    assert torch.count_nonzero(adapter.single_residual(0, img_tokens, img_token_len=8, latent_shape=None, batch_size=2)) == 0


def test_point_pose_are_optional_and_toggleable():
    # absent geometry -> identical to no-geometry path (backward compatible with depth-only ckpts)
    adapter = tiny_adapter()
    controls = tiny_controls(batch=2)
    a = adapter.build_control_state(**controls)
    b = adapter.build_control_state(**controls, point=None, pose_geom=None)
    assert torch.allclose(a.features, b.features)

    # pose accepts [B,9] and broadcasts across time
    horizon = controls["pred_tokens"].shape[1]
    st = adapter.build_control_state(**controls, pose_geom=torch.randn(2, 9))
    assert st.features.shape[2] == horizon

    # use_point/use_pose=False ignores provided geometry entirely
    off = tiny_adapter(use_point=False, use_pose=False)
    geom = _tiny_geom(batch=2, horizon=horizon)
    c = off.build_control_state(**controls)
    d = off.build_control_state(**controls, **geom)
    assert torch.allclose(c.features, d.features)
