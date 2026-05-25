from __future__ import annotations
import torch
import torch.nn as nn


class ActionProjHead(nn.Module):
    """z_a [B, k, z_dim] -> {pose_norm [B,k,6], pose [B,k,6], gripper_logit [B,k]}.

    Outputs pose_norm in standardized space; pose is the de-normalized
    pose_norm * std + mean for downstream eval/demo scripts. No tanh clamp.
    """

    def __init__(self, z_dim: int = 192, hidden: int = 1024, n_layers: int = 5,
                 stats_mean: torch.Tensor | None = None,
                 stats_std: torch.Tensor | None = None):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(z_dim, hidden), nn.GELU()]
        for _ in range(max(0, n_layers - 2)):
            layers.extend([nn.Linear(hidden, hidden), nn.GELU()])
        layers.append(nn.LayerNorm(hidden))
        self.trunk = nn.Sequential(*layers)
        self.pose_norm = nn.Linear(hidden, 6)
        self.grip_head = nn.Linear(hidden, 1)
        if stats_mean is None:
            stats_mean = torch.zeros(6)
        if stats_std is None:
            stats_std = torch.ones(6)
        self.register_buffer("mean", stats_mean.float().view(1, 1, 6))
        self.register_buffer("std", stats_std.float().view(1, 1, 6))

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(z)
        pose_norm = self.pose_norm(h)
        pose = pose_norm * self.std + self.mean
        grip_logit = self.grip_head(h).squeeze(-1)
        return {"pose_norm": pose_norm, "pose": pose, "gripper_logit": grip_logit}
