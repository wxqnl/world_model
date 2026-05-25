"""Phase 2 feasibility prototype trainer.

Two variants (selected by --variant):
    flow_only    — pure rectified-flow-matching loss on VGGT tokens
    flow_coupled — flow matching + predictor self-consistency + physics moment match

Frozen components:
    * VGGT backbone (we train on cached tokens from Phase 1)
    * CLIP ViT-B/32 text encoder
    * Phase 1 predictor D_psi (loaded from vggt_noact/best.pt)

Trainable:
    * FlowMatchingGenerator (G_theta)
    * PhysicsInference (g_phi) — trained jointly with G_theta through coupling loss

Usage::

    python -m scripts.phase2.train_generative \\
        --cfg configs/phase2/default.yaml --variant flow_coupled --epochs 40
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# make `src` importable when run as `python scripts/phase2/train_generative.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase1.heads import PredictiveHead, count_params
from src.phase2.coupling import coupling_total, predictor_selfconsistency_loss
from src.phase2.dataset import build_datasets, collate
from src.phase2.generative import (
    FlowMatchingGenerator, GenerativeConfig, flow_matching_loss,
)
from src.phase2.physics import (
    PhysicsConfig, PhysicsInference, physics_consistency_loss,
)
from src.phase2.text_encoder import FrozenCLIPText


# ----------------------------------------------------------------- utilities
def set_seed(s: int) -> None:
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_predictor(cfg: dict, device: torch.device) -> PredictiveHead:
    pc = cfg["predictor_ckpt"]
    m = PredictiveHead(
        token_dim=pc["token_dim"], action_dim=pc["action_dim"],
        hidden_dim=pc["hidden_dim"], n_layers=pc["n_layers"], n_heads=pc["n_heads"],
        context_len=pc["context_len"], action_embed_dim=pc["action_embed_dim"],
        dropout=pc["dropout"], use_actions=pc["use_actions"],
    )
    sd = torch.load(pc["path"], map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[predictor] missing={len(missing)} unexpected={len(unexpected)}")
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m.to(device)


def sample_with_grad(
    gen: FlowMatchingGenerator, cond_text: torch.Tensor, init_frame: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    """Like FlowMatchingGenerator.sample but keeps gradients so the coupling
    loss can shape the sampler. Kept short (n_steps=8) during training."""
    B = cond_text.size(0)
    device = cond_text.device
    T, P, D = gen.cfg.seq_len, gen.P, gen.cfg.token_dim
    z = torch.randn(B, T, P, D, device=device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((B,), i * dt, device=device)
        v = gen(z, t, cond_text, init_frame)
        z = z + dt * v
    return z


# ----------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/phase2/default.yaml")
    ap.add_argument("--variant", choices=["flow_only", "flow_coupled"], required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default=None, help="override output dir")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: 2 epochs, 3 batches, 4 sample steps")
    ap.add_argument("--train_sample_steps", type=int, default=8,
                    help="Euler steps when generating z_gen for coupling loss")
    ap.add_argument("--w_sc", type=float, default=None,
                    help="override predictor-self-consistency weight (post-warmup)")
    ap.add_argument("--w_ph", type=float, default=None,
                    help="override physics moment-match weight (post-warmup)")
    ap.add_argument("--sc_warmup_epochs", type=int, default=0,
                    help="epochs with w_sc and w_ph forced to 0 at the start "
                         "(schedule-based coupling; 0 = always on)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.cfg).read_text())
    set_seed(cfg["seed"])

    # apply variant overrides
    var = cfg["variants"][args.variant]
    cfg["coupling"]["w_selfconsistency"] = var["w_selfconsistency"]
    cfg["coupling"]["w_physics"] = var["w_physics"]
    if args.w_sc is not None:
        cfg["coupling"]["w_selfconsistency"] = args.w_sc
    if args.w_ph is not None:
        cfg["coupling"]["w_physics"] = args.w_ph
    do_coupling = (cfg["coupling"]["w_selfconsistency"] > 0
                   or cfg["coupling"]["w_physics"] > 0)
    # Schedule-based coupling: post-warmup weights; before warmup both are 0.
    w_sc_post = cfg["coupling"]["w_selfconsistency"]
    w_ph_post = cfg["coupling"]["w_physics"]
    warmup = max(0, args.sc_warmup_epochs)

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out or f"results/phase2/runs/{args.variant}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------- data
    train_ds, val_ds = build_datasets(cfg)
    print(f"train windows: {len(train_ds)} | val windows: {len(val_ds)}")
    bs = cfg["train"]["batch_size"]
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                          num_workers=cfg["train"]["num_workers"], collate_fn=collate,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False,
                        num_workers=cfg["train"]["num_workers"], collate_fn=collate)

    # -------------- models
    text_enc = FrozenCLIPText(cfg["text_encoder"]["hf_id"],
                              cfg["text_encoder"]["max_length"]).to(device)
    gen_cfg = GenerativeConfig(**cfg["generator"])
    gen = FlowMatchingGenerator(gen_cfg).to(device)
    phys = PhysicsInference(PhysicsConfig(**cfg["physics"])).to(device) if do_coupling else None
    predictor = load_predictor(cfg, device) if do_coupling else None

    print(f"generator params: {count_params(gen)/1e6:.2f} M")
    if phys is not None:
        print(f"physics   params: {count_params(phys)/1e6:.2f} M")

    params = list(gen.parameters())
    if phys is not None:
        params += list(phys.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    steps_per_epoch = len(train_dl)
    epochs = args.epochs or cfg["train"]["epochs"]
    total_steps = steps_per_epoch * epochs
    warm = cfg["train"]["warmup_steps"]

    def lr_at(step: int) -> float:
        if step < warm:
            return step / max(1, warm)
        p = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    # -------------- smoke overrides
    if args.smoke:
        epochs = 2
        max_batches = 3
    else:
        max_batches = 10**9

    # -------------- train
    log: list[dict] = []
    best_val = float("inf")
    global_step = 0
    t0 = time.time()
    for ep in range(epochs):
        gen.train()
        if phys is not None:
            phys.train()
        # epoch-level coupling weights (schedule-based coupling)
        active = ep >= warmup
        w_sc_ep = w_sc_post if active else 0.0
        w_ph_ep = w_ph_post if active else 0.0
        ep_fm, ep_sc, ep_ph, ep_tot, n_batches = 0.0, 0.0, 0.0, 0.0, 0
        for bi, batch in enumerate(train_dl):
            if bi >= max_batches:
                break
            for g in opt.param_groups:
                g["lr"] = cfg["train"]["lr"] * lr_at(global_step)

            z = batch["z"].to(device, non_blocking=True)           # [B, T, P, D]
            init = batch["init"].to(device, non_blocking=True)
            cond = text_enc(batch["text"], device)                 # [B, Dtxt]

            fm = flow_matching_loss(gen, z, cond, init)
            total = fm
            sc_val = ph_val = torch.zeros((), device=device)

            if do_coupling and active:
                n_steps = 4 if args.smoke else args.train_sample_steps
                z_gen = sample_with_grad(gen, cond, init, n_steps=n_steps)
                phys_C = phys.cfg.context_len
                sc_val = predictor_selfconsistency_loss(
                    predictor, z_gen, cfg["coupling"]["context_len"])
                ph_val = physics_consistency_loss(phys, z[:, :phys_C], z_gen[:, :phys_C])
                total = fm + w_sc_ep * sc_val + w_ph_ep * ph_val

            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["train"]["grad_clip"])
            opt.step()

            ep_fm += fm.item()
            ep_sc += sc_val.detach().item() if isinstance(sc_val, torch.Tensor) else float(sc_val)
            ep_ph += ph_val.detach().item() if isinstance(ph_val, torch.Tensor) else float(ph_val)
            ep_tot += total.item(); n_batches += 1; global_step += 1

        # -------------- validation
        gen.eval()
        if phys is not None:
            phys.eval()
        with torch.no_grad():
            v_fm = v_sc = v_ph = 0.0; v_n = 0
            for bi, batch in enumerate(val_dl):
                if args.smoke and bi >= 2:
                    break
                z = batch["z"].to(device)
                init = batch["init"].to(device)
                cond = text_enc(batch["text"], device)
                v_fm += flow_matching_loss(gen, z, cond, init).item()
                if do_coupling:
                    z_gen = sample_with_grad(gen, cond, init,
                                             n_steps=cfg["eval"]["sample_steps"])
                    phys_C = phys.cfg.context_len
                    v_sc += predictor_selfconsistency_loss(
                        predictor, z_gen, cfg["coupling"]["context_len"]).item()
                    v_ph += physics_consistency_loss(
                        phys, z[:, :phys_C], z_gen[:, :phys_C]).item()
                v_n += 1

        row = {
            "epoch": ep,
            "train_fm": ep_fm / max(1, n_batches),
            "train_sc": ep_sc / max(1, n_batches),
            "train_ph": ep_ph / max(1, n_batches),
            "train_total": ep_tot / max(1, n_batches),
            "val_fm": v_fm / max(1, v_n),
            "val_sc": v_sc / max(1, v_n) if do_coupling else None,
            "val_ph": v_ph / max(1, v_n) if do_coupling else None,
            "elapsed": time.time() - t0,
        }
        log.append(row)
        print(f"ep {ep:2d}  fm {row['train_fm']:.4e}  val_fm {row['val_fm']:.4e}  "
              f"sc {row['train_sc']:.4e}  ph {row['train_ph']:.4e}  "
              f"t {row['elapsed']:.0f}s")

        # save best by val_fm (always defined) — for coupled we also report val_sc
        if row["val_fm"] < best_val:
            best_val = row["val_fm"]
            torch.save({
                "gen": gen.state_dict(),
                "phys": phys.state_dict() if phys is not None else None,
                "epoch": ep,
                "val_fm": row["val_fm"],
                "cfg": cfg,
                "variant": args.variant,
            }, out_dir / "best.pt")

        (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))

    # -------------- final summary
    summary = {
        "variant": args.variant,
        "best_val_fm": best_val,
        "final": log[-1] if log else None,
        "epochs": epochs,
        "elapsed": time.time() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done. best_val_fm={best_val:.4e}  elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
