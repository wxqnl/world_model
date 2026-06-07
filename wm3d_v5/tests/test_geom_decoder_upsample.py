import torch


def test_geom_decoder_resize_conv_outputs_expected_depth_shape():
    from wm3d_v3.models.geom_decoder import GeomDecoder

    model = GeomDecoder(
        token_dim=8,
        token_grid=4,
        hidden=16,
        out_hw=16,
        crop_hw=12,
        enable_extra=False,
        upsample_mode="resize_conv",
    )
    tokens = torch.randn(2, 3, 16, 8)

    out = model(tokens)

    assert out["depth"].shape == (2, 3, 12, 12)
    assert torch.isfinite(out["depth"]).all()


def test_geom_decoder_default_transpose_mode_still_works():
    from wm3d_v3.models.geom_decoder import GeomDecoder

    model = GeomDecoder(
        token_dim=8,
        token_grid=4,
        hidden=16,
        out_hw=16,
        crop_hw=12,
        enable_extra=False,
    )
    tokens = torch.randn(1, 2, 16, 8)

    out = model(tokens)

    assert out["depth"].shape == (1, 2, 12, 12)
