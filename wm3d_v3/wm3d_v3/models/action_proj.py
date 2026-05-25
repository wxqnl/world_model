from __future__ import annotations
import torch
import torch.nn as nn


class ActionProjHead(nn.Module):
    """z_a [B, k, z_dim] -> {pose [B, k, 6], gripper_logit [B, k]}."""

    def __init__(self, z_dim=192, hidden=768, n_layers=5, action_dim=7):
        super().__init__()
        layers = [nn.Linear(z_dim, hidden), nn.GELU()]
        for _ in range(max(0, n_layers - 2)):
            layers.extend([nn.Linear(hidden, hidden), nn.GELU()])
        self.trunk = nn.Sequential(*layers)
        self.pose_head = nn.Linear(hidden, action_dim - 1)
        self.grip_head = nn.Linear(hidden, 1)

    def forward(self, z):
        h = self.trunk(z)
        pose = torch.tanh(self.pose_head(h)) * 0.1   # actions in OXE are small deltas
        grip_logit = self.grip_head(h).squeeze(-1)
        return {"pose": pose, "gripper_logit": grip_logit}
