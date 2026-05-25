"""Phase 2 scaffolding smoke test.

Confirms FlowMatchingGenerator, PhysicsInference, and the coupling losses run
end-to-end on random tensors, produce finite gradients, and have the
documented shapes. No training is performed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))

from src.phase1.heads import PredictiveHead
from src.phase2.generative import (
    FlowMatchingGenerator, GenerativeConfig, flow_matching_loss,
)
from src.phase2.physics import PhysicsInference, PhysicsConfig
from src.phase2.coupling import coupling_total


def main():
    B, T, P, D = 2, 6, 64, 2048
    gen_cfg = GenerativeConfig(
        token_dim=D, token_grid=8, seq_len=T, cond_text_dim=32,
        hidden_dim=128, n_layers=2, n_heads=4,
    )
    gen = FlowMatchingGenerator(gen_cfg)
    phys_cfg = PhysicsConfig(
        token_dim=D, token_grid=8, context_len=3, hidden_dim=128, n_layers=2,
        n_heads=4, phi_dim=16,
    )
    phys = PhysicsInference(phys_cfg)
    pred = PredictiveHead(token_dim=D, action_dim=7, hidden_dim=64,
                          n_layers=2, n_heads=4, context_len=3,
                          action_embed_dim=16, dropout=0.0, use_actions=False)

    cond = torch.randn(B, gen_cfg.cond_text_dim)
    init = torch.randn(B, P, D)
    z1 = torch.randn(B, T, P, D)

    loss_fm = flow_matching_loss(gen, z1, cond, init)
    assert torch.isfinite(loss_fm)
    loss_fm.backward()

    phi = phys(z1[:, :3])
    assert phi.shape == (B, 16)

    z_gen = torch.randn(B, T, P, D)
    out = coupling_total(pred, phys, z1[:, :3], z_gen, context_len=3)
    assert {"total", "self_consistency", "physics"} == set(out.keys())
    assert torch.isfinite(out["total"])
    out["total"].backward()

    # sample path
    z_sample = gen.sample(cond, init, n_steps=4)
    assert z_sample.shape == (B, T, P, D)
    print("phase2 smoke OK")


if __name__ == "__main__":
    main()
