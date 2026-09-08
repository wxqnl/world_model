"""FP32 small trainable interfaces with unchanged BF16 expert computation."""
from __future__ import annotations
import torch
from torch import nn


class FP32TrainableInterface(nn.Module):
    """Preserve the original layer and its hooks; retain FP32 optimizer updates."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        if type(layer) not in (nn.Linear, nn.LayerNorm):
            raise ValueError("Only the released Linear/LayerNorm interfaces are supported")
        self.layer = layer.float()

    @property
    def weight(self) -> torch.Tensor:
        return self.layer.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Training and standalone serving use the same BF16 projection. The
        # small FP32 parameters accumulate updates instead of discarding each
        # sub-ULP step. The original action-time hook remains on the inner layer.
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            return self.layer(x)


def promote_trainable_interfaces(wsa: nn.Module) -> None:
    targets = (
        (wsa.action_expert, "action_encoder"),
        (wsa.action_expert, "head"),
        (wsa.future_3d_expert.output_decoder, "output_norm"),
        (wsa.future_3d_expert.output_decoder, "output_proj"),
        (wsa, "proprio_encoder"),
    )
    if any(type(getattr(parent, name)) not in (nn.Linear, nn.LayerNorm) for parent, name in targets):
        raise ValueError("WSA trainable interface layout changed")
    for parent, name in targets:
        setattr(parent, name, FP32TrainableInterface(getattr(parent, name)))
