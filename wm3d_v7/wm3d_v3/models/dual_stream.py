from __future__ import annotations
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from .state_stream import StateStream, StateConfig
from .action_stream import ActionStream, ActionConfig
from .cross_attn import BidirectionalCrossAttn


@dataclass
class DualConfig:
    state: StateConfig = field(default_factory=StateConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    xattn_layers_state: tuple[int, ...] = (3, 6, 9, 11, 13)
    xattn_n_heads: int = 16


class DualStreamDynamics(nn.Module):
    def __init__(self, cfg: DualConfig):
        super().__init__()
        self.cfg = cfg
        self.state = StateStream(cfg.state)
        self.action = ActionStream(cfg.action)
        nA, nS = cfg.action.n_layers, cfg.state.n_layers
        self.action_steps = [(nA * (i + 1) // nS) - (nA * i // nS) for i in range(nS)]
        self.xattn_blocks = nn.ModuleList([
            BidirectionalCrossAttn(d_state=cfg.state.hidden, d_action=cfg.state.hidden,
                                    n_heads=cfg.xattn_n_heads)
            for _ in cfg.xattn_layers_state
        ])
        self.action_up = (
            nn.Linear(cfg.action.hidden, cfg.state.hidden)
            if cfg.action.hidden != cfg.state.hidden else nn.Identity()
        )
        self.action_down = (
            nn.Linear(cfg.state.hidden, cfg.action.hidden)
            if cfg.action.hidden != cfg.state.hidden else nn.Identity()
        )

    def forward(self, s, c, action_cond=None):
        h_s = self.state.encode(s, c, action_cond=action_cond)
        h_a = self.action.encode(s, c, action_cond=action_cond)
        xattn_set = set(self.cfg.xattn_layers_state)
        xattn_iter = iter(self.xattn_blocks)
        a_i = 0
        for s_i, layer_s in enumerate(self.state.layers):
            h_s = layer_s(h_s)
            steps = self.action_steps[s_i]
            for _ in range(steps):
                if a_i < len(self.action.layers):
                    h_a = self.action.layers[a_i](h_a)
                    a_i += 1
            if s_i in xattn_set:
                blk = next(xattn_iter)
                h_a_up = self.action_up(h_a)
                h_s, h_a_up_new = blk(h_s, h_a_up)
                h_a = self.action_down(h_a_up_new)
        while a_i < len(self.action.layers):
            h_a = self.action.layers[a_i](h_a)
            a_i += 1
        h_s = self.state.norm(h_s)
        h_a = self.action.norm(h_a)
        return {
            "pred_tokens": self.state.decode(h_s, action_cond=action_cond),
            "z_a": self.action.decode(h_a, action_cond=action_cond),
            "h_s": h_s,
            "h_a": h_a,
        }
