"""Smoke test: predictive head, dataset loader, and eval path run end-to-end
on synthetic data. No VGGT, no disk cache — just enough to catch wiring bugs
before a long GPU run.

Run with: ``python -m pytest tests/test_phase1_smoke.py -q``  (if pytest),
or directly: ``python tests/test_phase1_smoke.py``.
"""
from __future__ import annotations

import sys, json, tempfile
from pathlib import Path

import numpy as np
import torch

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))

from src.phase1.heads import PredictiveHead
from src.phase1 import dataset as dmod
from src.phase1.eval import rollout


def _make_synthetic_shard(
    out_dir: Path, clip_id: str, episode_index: int,
    n: int = 64, P: int = 64, D: int = 2048, A: int = 7,
):
    # Use a simple toy dynamics: tokens drift with action.
    rng = np.random.default_rng(episode_index)
    actions = rng.normal(0, 0.1, size=(n, A)).astype(np.float32)
    tokens = np.zeros((n, P, D), dtype=np.float32)
    tokens[0] = rng.normal(0, 1, size=(P, D)).astype(np.float32)
    for t in range(1, n):
        delta = actions[t - 1, 0] * 0.05  # crude, just so signal exists
        tokens[t] = tokens[t - 1] + delta * rng.normal(0, 1, size=(P, D)).astype(np.float32)
    tokens_bf16 = torch.from_numpy(tokens).to(torch.bfloat16)
    tokens_int16 = tokens_bf16.view(torch.int16).numpy()
    states = rng.normal(0, 1, size=(n, A)).astype(np.float32)
    np.savez_compressed(
        out_dir / f"{clip_id}.npz",
        tokens=tokens_int16,
        depth=np.zeros((0, 1, 1), dtype=np.float16),
        extrinsic=np.zeros((0, 3, 4), dtype=np.float32),
        intrinsic=np.zeros((0, 3, 3), dtype=np.float32),
        conf=np.zeros(n, dtype=np.float32),
        actions=actions,
        states=states,
        frame_ids=np.arange(n, dtype=np.int32),
    )
    (out_dir / f"{clip_id}.json").write_text(json.dumps({
        "clip_id": clip_id, "episode_index": episode_index, "n_frames": n,
        "token_grid": 8, "token_dim": D,
    }))


def main():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        for ep in range(3):
            _make_synthetic_shard(out, f"clip_{ep:02d}", ep, n=48)

        shards = dmod.discover_shards(out)
        assert len(shards) == 3
        train, val = dmod.split_shards(shards, [2])
        assert len(train) == 2 and len(val) == 1

        mean, std = dmod.compute_action_stats(train)
        assert mean.shape == (7,) and std.shape == (7,)

        ds = dmod.NextTokenPairs(train, context_len=4, token_pool="mean",
                                 normalize_actions=True, action_stats=(mean, std))
        batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=8)))
        assert batch["ctx_tokens"].shape == (8, 4, 2048)
        assert batch["tgt_tokens"].shape == (8, 2048)
        assert batch["ctx_actions"].shape == (8, 4, 7)
        assert batch["tgt_action"].shape == (8, 7)

        head = PredictiveHead(token_dim=2048, action_dim=7, hidden_dim=128,
                              n_layers=2, n_heads=4, context_len=4,
                              action_embed_dim=32, dropout=0.0, use_actions=True)
        pred = head(batch["ctx_tokens"], batch["ctx_actions"], batch["tgt_action"])
        assert pred.shape == (8, 2048)

        # quick overfit: loss should drop in 50 steps
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
        first = None; last = None
        for step in range(50):
            pred = head(batch["ctx_tokens"], batch["ctx_actions"], batch["tgt_action"])
            loss = (pred - batch["tgt_tokens"]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if step == 0: first = float(loss)
            last = float(loss)
        assert last < first * 0.7, f"loss didn't drop enough: {first} -> {last}"

        # rollout path (CPU, short horizon)
        per_k = rollout(head, val[0], context_len=4, horizons=[1, 2, 4],
                        token_pool="mean", device="cpu",
                        action_stats=(mean, std))
        assert set(per_k.keys()) == {1, 2, 4}
        for k in (1, 2, 4):
            assert len(per_k[k]["l2"]) > 0

        # no-action variant should still run and produce finite outputs
        head2 = PredictiveHead(token_dim=2048, action_dim=7, hidden_dim=128,
                               n_layers=2, n_heads=4, context_len=4,
                               action_embed_dim=32, dropout=0.0, use_actions=False)
        pred2 = head2(batch["ctx_tokens"], batch["ctx_actions"], batch["tgt_action"])
        assert torch.isfinite(pred2).all()

        print("phase1 smoke OK")


if __name__ == "__main__":
    main()
