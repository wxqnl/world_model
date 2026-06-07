import torch


def _tiny_joint_model():
    from wm3d_v3.models.action_stream import ActionConfig
    from wm3d_v3.models.dual_stream import DualConfig
    from wm3d_v3.models.joint_model import JointConfig, JointWorldModel
    from wm3d_v3.models.state_stream import StateConfig

    sc = StateConfig(T=2, P=4, D=16, hidden=32, n_layers=1, n_heads=4, k=2,
                     cond_dim=16, action_cond_dim=7)
    ac = ActionConfig(T=2, P=4, D=16, hidden=32, n_layers=1, n_heads=4, k=2,
                      z_dim=8, cond_dim=16, action_cond_dim=7)
    cfg = JointConfig(
        dual=DualConfig(state=sc, action=ac, xattn_layers_state=(), xattn_n_heads=4),
        action_proj_hidden=32,
        action_proj_layers=2,
        geom_hidden=16,
        enable_geom_extra=False,
        enable_pixel=False,
        enable_bridging=False,
        enable_world_prior=True,
        world_prior_hidden=32,
        world_prior_layers=1,
        world_prior_heads=4,
        world_prior_task_dim=16,
        world_prior_action_dim=7,
        world_prior_use_context=True,
        world_prior_use_action=True,
        world_prior_predict_initial=True,
    )
    return JointWorldModel(cfg)


def test_joint_world_model_world_prior_outputs_and_flow_loss():
    from wm3d_v3.losses import LossWeights, compute_losses

    torch.manual_seed(11)
    model = _tiny_joint_model()
    s = torch.randn(2, 2, 4, 16)
    c = torch.randn(2, 16)
    action_cond = torch.randn(2, 2, 7)
    s_tgt = torch.randn(2, 2, 4, 16)
    prior_clean = torch.cat([s[:, -1:, :, :], s_tgt], dim=1)

    out = model(
        s,
        c,
        action_cond=action_cond,
        pixel=False,
        prior_clean_tokens=prior_clean,
    )

    assert out["prior_initial_tokens"].shape == (2, 4, 16)
    assert out["prior_future_tokens"].shape == (2, 2, 4, 16)
    assert out["prior_velocity"].shape == (2, 3, 4, 16)
    assert out["prior_velocity_target"].shape == (2, 3, 4, 16)

    tgt = {
        "s_tgt": s_tgt,
        "s_init_tgt": s[:, -1],
        "action_tgt": torch.randn(2, 2, 7),
        "action_tgt_norm": torch.randn(2, 2, 6),
    }
    losses = compute_losses(
        out,
        tgt,
        LossWeights(
            action=0.0,
            geom_depth=0.0,
            geom_point=0.0,
            geom_pose=0.0,
            idm_reg=0.0,
            world_prior=0.5,
            world_prior_flow=0.7,
        ),
    )

    assert losses["L_world_prior"].item() > 0
    assert losses["L_world_prior_flow"].item() > 0
    losses["L_total"].backward()
    prior_grads = [p.grad for n, p in model.named_parameters() if n.startswith("world_prior.")]
    assert prior_grads
    assert any(g is not None and g.abs().sum() > 0 for g in prior_grads)


def test_condition_dropout_keeps_text_and_drops_context_or_action_per_sample():
    from wm3d_v3.training.train import apply_condition_dropout

    torch.manual_seed(12)
    s = torch.ones(4, 2, 4, 16)
    c = torch.randn(4, 16)
    action_cond = torch.ones(4, 2, 7)
    context_rgb = torch.ones(4, 3, 8, 8)

    s2, c2, action2, rgb2, meta = apply_condition_dropout(
        s,
        c,
        action_cond,
        context_rgb,
        {
            "condition_dropout": {
                "enabled": True,
                "action_p": 1.0,
                "context_p": 1.0,
            }
        },
        training=True,
    )

    assert torch.equal(c2, c)
    assert torch.count_nonzero(s2) == 0
    assert torch.count_nonzero(action2) == 0
    assert torch.count_nonzero(rgb2) == 0
    assert meta["drop_action_frac"].item() == 1.0
    assert meta["drop_context_frac"].item() == 1.0


def test_training_like_prior_path_trains_generation_query_and_decodes_prior_tokens():
    from wm3d_v3.losses import LossWeights, compute_losses

    torch.manual_seed(21)
    model = _tiny_joint_model()
    s = torch.randn(2, 2, 4, 16)
    c = torch.randn(2, 16)
    action_cond = torch.randn(2, 2, 7)
    s_tgt = torch.randn(2, 2, 4, 16)
    prior_clean = torch.cat([s[:, -1:, :, :], s_tgt], dim=1)
    depth_tgt = torch.rand(2, 2, 224, 224)

    out = model(
        s,
        c,
        action_cond=action_cond,
        prior_clean_tokens=prior_clean,
        pixel=False,
    )

    assert out["prior_depth"].shape == (2, 2, 224, 224)
    assert out["prior_hunyuan_tokens"].shape == out["prior_future_tokens"].shape
    losses = compute_losses(
        out,
        {
            "s_tgt": s_tgt,
            "s_init_tgt": s[:, -1],
            "depth_tgt": depth_tgt,
            "action_tgt": torch.randn(2, 2, 7),
            "action_tgt_norm": torch.randn(2, 2, 6),
        },
        LossWeights(
            action=0.0,
            geom_depth=0.0,
            idm_reg=0.0,
            world_prior=0.5,
            world_prior_flow=0.7,
            world_prior_depth=0.2,
        ),
    )
    losses["L_total"].backward()
    assert model.world_prior.query.grad is not None
    assert model.world_prior.query.grad.abs().sum() > 0


def test_world_prior_flow_sampler_supports_condition_modes():
    torch.manual_seed(22)
    model = _tiny_joint_model()
    c = torch.randn(2, 16)
    s = torch.randn(2, 2, 4, 16)
    action_cond = torch.randn(2, 2, 7)

    modes = {
        "text_only": (None, None),
        "text_context": (s, None),
        "text_action": (None, action_cond),
        "full": (s, action_cond),
    }
    for context_tokens, actions in modes.values():
        out = model.generate_world_prior(
            c,
            context_tokens=context_tokens,
            action_cond=actions,
            steps=2,
        )
        assert out["prior_initial_tokens"].shape == (2, 4, 16)
        assert out["prior_future_tokens"].shape == (2, 2, 4, 16)


def test_condition_dropout_explicit_text_only_mode_probability():
    from wm3d_v3.training.train import apply_condition_dropout

    torch.manual_seed(23)
    s = torch.ones(8, 2, 4, 16)
    c = torch.randn(8, 16)
    action_cond = torch.ones(8, 2, 7)
    context_rgb = torch.ones(8, 3, 8, 8)

    s2, c2, action2, rgb2, meta = apply_condition_dropout(
        s,
        c,
        action_cond,
        context_rgb,
        {
            "condition_dropout": {
                "enabled": True,
                "text_only_p": 1.0,
                "action_p": 0.0,
                "context_p": 0.0,
            }
        },
        training=True,
    )

    assert torch.equal(c2, c)
    assert torch.count_nonzero(s2) == 0
    assert torch.count_nonzero(action2) == 0
    assert torch.count_nonzero(rgb2) == 0
    assert meta["drop_text_only_frac"].item() == 1.0


def _tiny_context_pixel_model():
    from wm3d_v3.models.action_stream import ActionConfig
    from wm3d_v3.models.dual_stream import DualConfig
    from wm3d_v3.models.joint_model import JointConfig, JointWorldModel
    from wm3d_v3.models.state_stream import StateConfig

    sc = StateConfig(T=2, P=4, D=16, hidden=32, n_layers=1, n_heads=4, k=2,
                     cond_dim=16, action_cond_dim=7)
    ac = ActionConfig(T=2, P=4, D=16, hidden=32, n_layers=1, n_heads=4, k=2,
                      z_dim=8, cond_dim=16, action_cond_dim=7)
    cfg = JointConfig(
        dual=DualConfig(state=sc, action=ac, xattn_layers_state=(), xattn_n_heads=4),
        action_proj_hidden=32,
        action_proj_layers=2,
        geom_hidden=16,
        enable_geom_extra=False,
        enable_pixel=True,
        enable_context_pixel=True,
        context_pixel_hidden=16,
        context_pixel_action_dim=7,
        context_pixel_task_dim=16,
        context_pixel_use_action=True,
        context_pixel_use_task=True,
        enable_bridging=False,
        enable_world_prior=True,
        world_prior_hidden=32,
        world_prior_layers=1,
        world_prior_heads=4,
        world_prior_task_dim=16,
        world_prior_action_dim=7,
    )
    return JointWorldModel(cfg)


class _FakeLatentDist:
    def __init__(self, latents):
        self._latents = latents

    def mode(self):
        return self._latents


class _FakePosterior:
    def __init__(self, latents):
        self.latent_dist = _FakeLatentDist(latents)


class _FakeVAE:
    dtype = torch.float32

    class config:
        scaling_factor = 1.0

    def __init__(self, channels=4, latent_t=2, latent_hw=8):
        self.channels = channels
        self.latent_t = latent_t
        self.latent_hw = latent_hw

    def encode(self, video):
        bsz = video.shape[0]
        base = video.mean(dim=(1, 2, 3, 4), keepdim=True)
        latents = base.expand(bsz, self.channels, self.latent_t, self.latent_hw, self.latent_hw).contiguous()
        return _FakePosterior(latents)


def test_prior_hunyuan_loss_consumes_prior_tokens_and_depth():
    from wm3d_v3.models.hunyuan_latent_adapter import HunyuanLatentAdapter, HunyuanLatentAdapterConfig
    from wm3d_v3.training.train import compute_hunyuan_latent_loss

    torch.manual_seed(31)
    adapter = HunyuanLatentAdapter(HunyuanLatentAdapterConfig(
        token_dim=16,
        token_grid=2,
        hidden=8,
        latent_channels=4,
        latent_time=2,
        latent_hw=8,
        action_dim=7,
        task_dim=16,
        n_blocks=1,
        use_motion=False,
        use_rough_rgb=False,
    ))
    prior_tokens = torch.randn(2, 2, 4, 16, requires_grad=True)
    prior_depth = torch.rand(2, 2, 224, 224, requires_grad=True)
    out = {
        "pred_tokens": torch.randn(2, 2, 4, 16),
        "depth": torch.rand(2, 2, 224, 224),
        "prior_hunyuan_tokens": prior_tokens,
        "prior_hunyuan_depth": prior_depth,
    }
    losses = compute_hunyuan_latent_loss(
        adapter,
        _FakeVAE(channels=4, latent_t=2, latent_hw=8),
        out,
        {"rgb_tgt_p": torch.rand(2, 2, 3, 32, 32)},
        torch.rand(2, 3, 32, 32),
        torch.randn(2, 2, 7),
        torch.randn(2, 16),
        {
            "enable_prior_hunyuan_latent_loss": True,
            "prior_hunyuan_latent_weight": 1.0,
            "hunyuan_use_rough_rgb": False,
            "hunyuan_latent_mse_weight": 1.0,
            "hunyuan_latent_l1_weight": 0.0,
            "hunyuan_latent_temporal_weight": 0.0,
            "hunyuan_latent_motion_weight": 0.0,
        },
    )
    assert torch.isfinite(losses["L_prior_hunyuan_latent"])
    assert losses["L_prior_hunyuan_latent"].item() > 0
    losses["L_prior_hunyuan_latent"].backward()
    assert prior_tokens.grad is not None and prior_tokens.grad.abs().sum() > 0
    assert prior_depth.grad is not None and prior_depth.grad.abs().sum() > 0


def test_generate_world_prior_text_only_pixel_true_returns_prior_rgb():
    torch.manual_seed(32)
    model = _tiny_context_pixel_model().eval()
    out = model.generate_world_prior(torch.randn(1, 16), steps=1, pixel=True)
    assert out["prior_depth"].shape == (1, 2, 224, 224)
    assert out["prior_rgb"].shape == (1, 2, 3, 256, 256)
    assert torch.isfinite(out["prior_rgb"]).all()
