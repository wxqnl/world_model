"""Phase 2 paper-scale trainer (DDP + bf16 + step-level resume + W&B).

Companion to ``scripts.phase2.train_generative`` (the prototype). Trains the
flow-matching generator and (optionally) the physics head with the same
schedule-based-coupling recipe the prototype validated, plus everything
required at scale:

* DDP: generator + physics wrapped on each rank; predictor stays frozen and
  per-rank (no grads, no DDP).
* bfloat16 autocast (no GradScaler — bf16 is range-equivalent to fp32).
* Streaming :class:`StreamingGenerativeWindows` so the 750 GB token cache stays
  on disk.
* Activation checkpointing on the generator backbone (cfg.generator.use_checkpoint).
* Step-level checkpoints (model+opt+step+rng) on rank 0; ``--resume``.
* W&B optional, rank 0 only.

Launch::

    torchrun --nproc_per_node 8 scripts/phase2/train_generative_ddp.py \\
        --cfg configs/phase2/paper_scale.yaml --variant flow_coupled

Use ``scripts/launch_ddp.sh`` to auto-pin to free GPUs on a shared host.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase1.heads import PredictiveHead, count_params
from src.phase2.coupling import predictor_selfconsistency_loss
from src.phase2.dataset import build_datasets, collate
from src.phase2.dataset_streaming import build_streaming_datasets
from src.phase2.generative import (
    FlowMatchingGenerator, GenerativeConfig, flow_matching_loss,
)
from src.phase2.physics import (
    PhysicsConfig, PhysicsInference, physics_consistency_loss,
)
from src.phase2.text_encoder import FrozenCLIPText

log = logging.getLogger("phase2.train_ddp")


# --------------------------------------------------------------------- distributed
def _ddp_setup() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", rank))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


def _is_main(rank: int) -> bool:
    return rank == 0


def _ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _save_ckpt(path: Path, **state) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _latest_ckpt(run_dir: Path) -> Path | None:
    cks = sorted(run_dir.glob("ckpt_step_*.pt"))
    return cks[-1] if cks else None


def set_seed(s: int) -> None:
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# --------------------------------------------------------------------- predictor
def load_predictor(cfg: dict, device: torch.device) -> PredictiveHead:
    pc = cfg["predictor_ckpt"]
    m = PredictiveHead(
        token_dim=pc["token_dim"], action_dim=pc["action_dim"],
        hidden_dim=pc["hidden_dim"], n_layers=pc["n_layers"], n_heads=pc["n_heads"],
        context_len=pc["context_len"], action_embed_dim=pc["action_embed_dim"],
        dropout=pc["dropout"], use_actions=pc["use_actions"],
        use_checkpoint=pc.get("use_checkpoint", False),
    )
    sd = torch.load(pc["path"], map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        log.warning("predictor missing=%d unexpected=%d", len(missing), len(unexpected))
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m.to(device)


def sample_with_grad(
    gen: FlowMatchingGenerator | DDP,
    cond_text: torch.Tensor, init_frame: torch.Tensor, n_steps: int,
) -> torch.Tensor:
    """Eulerian rollout that retains gradients (for L_sc / L_ph)."""
    base = gen.module if isinstance(gen, DDP) else gen
    B = cond_text.size(0)
    device = cond_text.device
    T, P, D = base.cfg.seq_len, base.P, base.cfg.token_dim
    z = torch.randn(B, T, P, D, device=device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((B,), i * dt, device=device)
        v = gen(z, t, cond_text, init_frame)
        z = z + dt * v
    return z


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/phase2/paper_scale.yaml")
    ap.add_argument("--variant", choices=["flow_only", "flow_coupled"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs × 3 batches, 4 sample steps; for testing only")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
                        datefmt="%H:%M:%S")
    rank, world, local = _ddp_setup()
    cfg = yaml.safe_load(Path(args.cfg).read_text())
    set_seed(int(cfg["seed"]) + rank)

    # variant overrides
    var = cfg["variants"][args.variant]
    cfg["coupling"]["w_selfconsistency"] = var["w_selfconsistency"]
    cfg["coupling"]["w_physics"] = var["w_physics"]
    do_coupling = (cfg["coupling"]["w_selfconsistency"] > 0
                   or cfg["coupling"]["w_physics"] > 0)
    w_sc_post = cfg["coupling"]["w_selfconsistency"]
    w_ph_post = cfg["coupling"]["w_physics"]
    warmup_epochs = int(cfg["coupling"].get("warmup_epochs", 0))

    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out or f"results/phase2_scale/runs/{args.variant}")
    if _is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()

    # ---- data
    streaming = bool(cfg["data"].get("streaming", True))
    if streaming:
        # streaming dataset wants val_episode_ids as a list; load from file if needed
        if "val_episode_ids" not in cfg["data"] and "val_episode_ids_file" in cfg["data"]:
            p = Path(cfg["data"]["val_episode_ids_file"])
            cfg["data"]["val_episode_ids"] = list(json.loads(p.read_text())) if p.exists() else []
        train_ds, val_ds = build_streaming_datasets(cfg, rank=rank, world_size=world)
    else:
        train_ds, val_ds = build_datasets(cfg)
    if _is_main(rank):
        log.info("train: %s windows  val: %s windows",
                 len(train_ds), len(val_ds))

    bs = cfg["train"]["batch_size"]
    train_dl = DataLoader(
        train_ds, batch_size=bs, num_workers=cfg["train"]["num_workers"],
        collate_fn=collate, drop_last=True, shuffle=False,
    )
    val_dl = DataLoader(val_ds, batch_size=bs, num_workers=0,
                        collate_fn=collate, shuffle=False)

    # ---- models
    text_enc = FrozenCLIPText(cfg["text_encoder"]["hf_id"],
                              cfg["text_encoder"]["max_length"]).to(device)
    gen = FlowMatchingGenerator(GenerativeConfig(**cfg["generator"])).to(device)
    phys = PhysicsInference(PhysicsConfig(**cfg["physics"])).to(device) if do_coupling else None
    predictor = load_predictor(cfg, device) if do_coupling else None

    if _is_main(rank):
        log.info("generator params: %.2f M", count_params(gen) / 1e6)
        if phys is not None:
            log.info("physics   params: %.2f M", count_params(phys) / 1e6)
        if predictor is not None:
            log.info("predictor params: %.2f M (frozen)", count_params(predictor) / 1e6)

    if world > 1:
        gen = DDP(gen, device_ids=[local],
                  find_unused_parameters=cfg.get("ddp", {}).get("find_unused_parameters", False))
        if phys is not None:
            phys = DDP(phys, device_ids=[local])

    params = list((gen.module if isinstance(gen, DDP) else gen).parameters())
    if phys is not None:
        params += list((phys.module if isinstance(phys, DDP) else phys).parameters())
    opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"],
                            betas=(0.9, 0.95))

    epochs = cfg["train"]["epochs"]
    steps_per_epoch = max(1, len(train_ds) // (bs * max(1, world)))
    total_steps = steps_per_epoch * epochs
    warm = cfg["train"]["warmup_steps"]
    grad_accum = max(1, int(cfg["train"].get("grad_accum", 1)))
    ckpt_every = int(cfg["train"]["ckpt_every_steps"])
    log_every = int(cfg["train"].get("log_every_steps", 50))
    train_sample_steps = int(cfg["train"].get("train_sample_steps", 8))
    amp_dtype = torch.bfloat16 if cfg["train"].get("amp_dtype", "bfloat16") == "bfloat16" else torch.float16

    def lr_at(s: int) -> float:
        if s < warm:
            return s / max(1, warm)
        p = (s - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    # ---- resume
    global_step = 0
    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        ck = _latest_ckpt(out_dir)
        if ck is not None:
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            (gen.module if isinstance(gen, DDP) else gen).load_state_dict(sd["gen"])
            if phys is not None and sd.get("phys") is not None:
                (phys.module if isinstance(phys, DDP) else phys).load_state_dict(sd["phys"])
            opt.load_state_dict(sd["opt"])
            global_step = int(sd["step"]); start_epoch = int(sd["epoch"])
            best_val = float(sd.get("best_val", float("inf")))
            if _is_main(rank):
                log.info("resumed from %s @ step %d", ck.name, global_step)

    if args.smoke:
        epochs = min(epochs, 2)
        max_batches = 3
        train_sample_steps = 4
    else:
        max_batches = 10**9

    # ---- W&B
    wandb_run = None
    if _is_main(rank) and cfg.get("wandb", {}).get("enabled", False):
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg["wandb"].get("project", "geophys-wm"),
                entity=cfg["wandb"].get("entity"),
                name=f"{cfg['wandb'].get('run_name', 'phase2')}-{args.variant}",
                config=cfg, resume="allow",
            )
        except Exception as e:
            log.warning("wandb init failed: %s", e)

    # ---- train loop
    t0 = time.time()
    for ep in range(start_epoch, epochs):
        if streaming and hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(ep)
        active = ep >= warmup_epochs
        w_sc = w_sc_post if active else 0.0
        w_ph = w_ph_post if active else 0.0

        gen.train()
        if phys is not None:
            phys.train()
        opt.zero_grad(set_to_none=True)
        agg = {"fm": 0.0, "sc": 0.0, "ph": 0.0, "tot": 0.0, "n": 0}

        for bi, batch in enumerate(train_dl):
            if bi >= max_batches:
                break
            for g in opt.param_groups:
                g["lr"] = cfg["train"]["lr"] * lr_at(global_step)

            z = batch["z"].to(device, non_blocking=True)
            init = batch["init"].to(device, non_blocking=True)
            cond = text_enc(batch["text"], device)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                fm = flow_matching_loss(gen, z, cond, init)
                total = fm
                sc_val = ph_val = torch.zeros((), device=device)
                if do_coupling and active:
                    z_gen = sample_with_grad(gen, cond, init, n_steps=train_sample_steps)
                    base_phys = phys.module if isinstance(phys, DDP) else phys
                    phys_C = base_phys.cfg.context_len
                    sc_val = predictor_selfconsistency_loss(
                        predictor, z_gen, cfg["coupling"]["context_len"])
                    ph_val = physics_consistency_loss(phys, z[:, :phys_C], z_gen[:, :phys_C])
                    total = fm + w_sc * sc_val + w_ph * ph_val
                total_scaled = total / grad_accum

            total_scaled.backward()
            agg["fm"] += float(fm.detach())
            agg["sc"] += float(sc_val.detach() if isinstance(sc_val, torch.Tensor) else sc_val)
            agg["ph"] += float(ph_val.detach() if isinstance(ph_val, torch.Tensor) else ph_val)
            agg["tot"] += float(total.detach()); agg["n"] += 1

            if (bi + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg["train"]["grad_clip"])
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                if _is_main(rank) and global_step % log_every == 0:
                    log.info("ep %d step %d  fm %.4e  sc %.4e  ph %.4e  tot %.4e  %.0fs",
                             ep, global_step,
                             agg["fm"]/agg["n"], agg["sc"]/agg["n"],
                             agg["ph"]/agg["n"], agg["tot"]/agg["n"],
                             time.time() - t0)
                    if wandb_run is not None:
                        wandb_run.log({
                            "train/fm": agg["fm"]/agg["n"],
                            "train/sc": agg["sc"]/agg["n"],
                            "train/ph": agg["ph"]/agg["n"],
                            "train/total": agg["tot"]/agg["n"],
                            "train/lr": opt.param_groups[0]["lr"],
                            "epoch": ep,
                        }, step=global_step)

                if _is_main(rank) and ckpt_every > 0 and global_step % ckpt_every == 0:
                    _save_ckpt(
                        out_dir / f"ckpt_step_{global_step:08d}.pt",
                        gen=(gen.module if isinstance(gen, DDP) else gen).state_dict(),
                        phys=((phys.module if isinstance(phys, DDP) else phys).state_dict()
                              if phys is not None else None),
                        opt=opt.state_dict(),
                        step=global_step, epoch=ep, best_val=best_val, cfg=cfg,
                        variant=args.variant,
                    )

        # ---- per-epoch val (rank 0)
        if _is_main(rank):
            gen.eval()
            if phys is not None:
                phys.eval()
            with torch.no_grad():
                v = {"fm": 0.0, "sc": 0.0, "ph": 0.0, "n": 0}
                for vbi, batch in enumerate(val_dl):
                    if args.smoke and vbi >= 2:
                        break
                    z = batch["z"].to(device); init = batch["init"].to(device)
                    cond = text_enc(batch["text"], device)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                        v["fm"] += float(flow_matching_loss(gen, z, cond, init))
                        if do_coupling:
                            z_gen = sample_with_grad(gen, cond, init,
                                                     n_steps=cfg["eval"]["sample_steps"])
                            base_phys = phys.module if isinstance(phys, DDP) else phys
                            phys_C = base_phys.cfg.context_len
                            v["sc"] += float(predictor_selfconsistency_loss(
                                predictor, z_gen, cfg["coupling"]["context_len"]))
                            v["ph"] += float(physics_consistency_loss(
                                phys, z[:, :phys_C], z_gen[:, :phys_C]))
                    v["n"] += 1
                vfm = v["fm"]/max(1, v["n"])
            log.info("ep %d  val_fm %.4e  step %d  %.1fs", ep, vfm, global_step, time.time() - t0)
            if wandb_run is not None:
                payload = {"val/fm": vfm, "epoch": ep}
                if do_coupling and v["n"]:
                    payload["val/sc"] = v["sc"]/v["n"]; payload["val/ph"] = v["ph"]/v["n"]
                wandb_run.log(payload, step=global_step)
            if vfm < best_val:
                best_val = vfm
                torch.save({
                    "gen": (gen.module if isinstance(gen, DDP) else gen).state_dict(),
                    "phys": ((phys.module if isinstance(phys, DDP) else phys).state_dict()
                             if phys is not None else None),
                    "epoch": ep, "val_fm": vfm, "cfg": cfg, "variant": args.variant,
                }, out_dir / "best.pt")
        if world > 1:
            dist.barrier()

    if _is_main(rank):
        _save_ckpt(
            out_dir / f"ckpt_step_{global_step:08d}.pt",
            gen=(gen.module if isinstance(gen, DDP) else gen).state_dict(),
            phys=((phys.module if isinstance(phys, DDP) else phys).state_dict()
                  if phys is not None else None),
            opt=opt.state_dict(),
            step=global_step, epoch=epochs, best_val=best_val, cfg=cfg, variant=args.variant,
        )
        (out_dir / "summary.json").write_text(json.dumps({
            "variant": args.variant, "best_val_fm": best_val,
            "epochs": epochs, "elapsed": time.time() - t0, "step": global_step,
        }, indent=2))
        log.info("done. best_val_fm=%.4e  elapsed=%.0fs", best_val, time.time() - t0)
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    try:
        main()
    finally:
        _ddp_cleanup()
