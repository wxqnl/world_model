"""Mathematical contracts for the opt-in video reduction; not model qualification."""
import pytest
import torch

from wm3d.models.foundation_video_objective import valid_view_video_loss


def reduce(pred, mask, *, include=False):
    return valid_view_video_loss(
        pred, torch.zeros_like(pred), mask, ~mask.any(-1),
        temporal_factor=4, include_initial=include,
    )[0]


def test_equal_camera_weight_and_invalid_camera_zero_gradient():
    pred = torch.ones(1, 2, 2, 6, 4, requires_grad=True)
    mask = torch.ones(1, 9, 3, dtype=torch.bool)
    reduce(pred, mask).sum().backward()
    masses = torch.stack((pred.grad[:, :, :, :4].sum(),
                          pred.grad[:, :, :, 4:, :2].sum(),
                          pred.grad[:, :, :, 4:, 2:].sum()))
    torch.testing.assert_close(masses, torch.full((3,), 2 / 3))
    pred.grad = None
    mask[:, :, 2] = False
    reduce(pred, mask).sum().backward()
    assert torch.count_nonzero(pred.grad[:, :, :, 4:, 2:]) == 0
    torch.testing.assert_close(pred.grad[:, :, :, :4].sum(), torch.tensor(1.))
    torch.testing.assert_close(pred.grad[:, :, :, 4:, :2].sum(), torch.tensor(1.))


def test_partial_temporal_camera_block_is_not_a_valid_target():
    pred = torch.ones(1, 1, 2, 6, 4, requires_grad=True)
    mask = torch.ones(1, 9, 3, dtype=torch.bool)
    mask[:, 3, 1] = False
    reduce(pred, mask).sum().backward()
    assert torch.count_nonzero(pred.grad[:, :, 0, 4:, :2]) == 0
    assert torch.count_nonzero(pred.grad[:, :, 1, 4:, :2]) > 0


def test_anchor_alignment_and_per_sample_normalization():
    pred = torch.ones(2, 1, 3, 6, 4)
    pred[1] *= 2
    mask = torch.ones(2, 9, 3, dtype=torch.bool)
    mask[1, :, 1:] = False
    torch.testing.assert_close(reduce(pred, mask, include=True), torch.tensor([1., 4.]))


def test_view_loss_rejects_missing_validity_and_padding_disagreement():
    pred = torch.ones(1, 1, 2, 6, 4)
    mask = torch.zeros(1, 9, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="no complete valid"):
        reduce(pred, mask)
    mask[:] = True
    with pytest.raises(ValueError, match="padding disagrees"):
        valid_view_video_loss(pred, pred, mask, torch.ones(1, 9, dtype=torch.bool),
                              temporal_factor=4, include_initial=False)
    with pytest.raises(ValueError, match="latent steps"):
        reduce(pred[:, :, :1], mask)
