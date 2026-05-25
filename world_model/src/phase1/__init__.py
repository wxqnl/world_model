"""Phase 1 — predictive head on frozen VGGT tokens.

Public modules:
  cache        — one-time VGGT token caching to disk
  dataset      — (s_t, a_t, s_{t+1}) loader over cached shards
  backbones    — frozen encoders (VGGT, DINOv2) with a common API
  heads        — predictive transformer (D_psi)
  losses       — flow-matching + direct MSE options
  eval         — k-step rollout metrics
  train        — training loop (single- or multi-GPU)

Run scripts live under ``scripts/phase1/``.
"""
