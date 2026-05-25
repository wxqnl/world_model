"""Phase 1 paper-scale trainer (DDP + bf16 + step-level resume + W&B).

Companion to ``src.phase1.train`` (the prototype). Same model and loss; the
difference is everything around it:

* DDP via ``torch.distributed`` (single-node 8×H100 by default; multi-node ok).
* bfloat16 autocast on the forward+backward pass. No GradScaler — bf16 has the
  same dynamic range as fp32, so loss scaling is unnecessary.
* Streaming :class:`StreamingNextTokenPairs` dataset (see
  ``src.phase1.dataset_streaming``) — never holds the 750 GB cache in RAM.
* Step-level checkpointing every ``train.ckpt_every_steps`` steps, saving
  model+optimizer+scheduler-step+rng on rank 0 only. ``--resume`` continues
  from the latest ``ckpt_step_*.pt``.
* W&B logging on rank 0 only (optional; set ``wandb.enabled: true``).
* Activation checkpointing wired through ``head.use_checkpoint`` config.

Launch with::

    torchrun --nproc_per_node 8 -m src.phase1.train_ddp \\
        --cfg configs/phase1/paper_scale.yaml --run vggt_noact

Use ``scripts/launch_ddp.sh`` to auto-pin to free GPUs on a shared host.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .dataset import compute_action_stats, discover_shards, split_shards
from .dataset_streaming import StreamingNextTokenPairs
from .heads import PredictiveHead, count_params

log = logging.getLogger("phase1.train_ddp")


# --------------------------------------------------------------------- distributed
def _ddp_setup() -> tuple[int, int, int]:
    """Returns (rank, world_size, local_rank). Falls back to single-process."""
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


# --------------------------------------------------------------------- helpers
def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def _load_val_ids(cfg: dict) -> list[int]:
    split = cfg["dataset"]["split"]
    if "val_episode_ids" in split:
        return list(split["val_episode_ids"])
    if "val_episode_ids_file" in split:
        p = Path(split["val_episode_ids_file"])
        if p.exists():
            return list(json.loads(p.read_text()))
    log.warning("no validation IDs configured; using empty set")
    return []


def _save_ckpt(path: Path, **state) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _latest_ckpt(run_dir: Path) -> Path | None:
    cks = sorted(run_dir.glob("ckpt_step_*.pt"))
    return cks[-1] if cks else None


# --------------------------------------------------------------------- main run
def run_one(cfg: dict, run_name: str, args) -> dict:
    rank, world, local = _ddp_setup()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    seed = int(cfg["seed"]) + rank
    torch.manual_seed(seed); np.random.seed(seed)

    # ---- shard discovery + split (cheap, all ranks)
    cache_dir = Path(cfg["cache"]["out_dir"])
    shards = discover_shards(cache_dir)
    if not shards:
        raise RuntimeError(f"no shards in {cache_dir}")
    val_ids = _load_val_ids(cfg)
    train_shards, val_shards = split_shards(shards, val_ids)
    if _is_main(rank):
        log.info("train shards: %d  val shards: %d", len(train_shards), len(val_shards))

    # action stats: rank 0 computes, broadcast tensor.
    action_stats = None
    if cfg["dataset"]["normalize_actions"]:
        if _is_main(rank):
            mean, std = compute_action_stats(train_shards)
        else:
            mean = np.zeros(cfg["dataset"]["action_dim"], np.float32)
            std = np.ones(cfg["dataset"]["action_dim"], np.float32)
        if world > 1:
            mt = torch.from_numpy(mean).to(device)
            st = torch.from_numpy(std).to(device)
            dist.broadcast(mt, 0); dist.broadcast(st, 0)
            mean, std = mt.cpu().numpy(), st.cpu().numpy()
        action_stats = (mean, std)

    # ---- datasets (streaming or in-memory)
    C = cfg["head"]["context_len"]
    pool = cfg["head"]["token_pool"]
    use_actions = run_name in ("vggt", "vggt_bigact")

    streaming = bool(cfg["dataset"].get("streaming", True))
    if streaming:
        train_ds = StreamingNextTokenPairs(
            train_shards, C, pool, cfg["dataset"]["normalize_actions"], action_stats,
            rank=rank, world_size=world, seed=int(cfg["seed"]),
        )
        val_ds = StreamingNextTokenPairs(
            val_shards, C, pool, cfg["dataset"]["normalize_actions"], action_stats,
            rank=0, world_size=1, seed=int(cfg["seed"]),
        )
    else:
        from .dataset import NextTokenPairs
        train_ds = NextTokenPairs(train_shards, C, pool,
                                  cfg["dataset"]["normalize_actions"], action_stats)
        val_ds = NextTokenPairs(val_shards, C, pool,
                                cfg["dataset"]["normalize_actions"], action_stats)

    bs = cfg["train"]["batch_windows"]
    train_dl = DataLoader(
        train_ds, batch_size=bs, num_workers=cfg["train"]["num_workers"],
        pin_memory=True, drop_last=True,
        # IterableDataset: shuffle is illegal; per-shard shuffle is internal.
        shuffle=False if streaming else True,
    )
    val_dl = DataLoader(val_ds, batch_size=bs, num_workers=0, drop_last=False, shuffle=False)

    # ---- model
    head = PredictiveHead(
        token_dim=cfg["cache"]["token_dim"],
        action_dim=cfg["dataset"]["action_dim"],
        hidden_dim=cfg["head"]["hidden_dim"],
        n_layers=cfg["head"]["n_layers"],
        n_heads=cfg["head"]["n_heads"],
        context_len=C,
        action_embed_dim=cfg["head"]["action_embed_dim"],
        dropout=cfg["head"]["dropout"],
        use_actions=use_actions,
        use_checkpoint=bool(cfg["head"].get("use_checkpoint", False)),
    ).to(device)
    if _is_main(rank):
        log.info("head params: %.2f M", count_params(head) / 1e6)

    if world > 1:
        head = DDP(head, device_ids=[local],
                   find_unused_parameters=cfg.get("ddp", {}).get("find_unused_parameters", False))

    opt = torch.optim.AdamW(
        head.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        betas=(0.9, 0.95),
    )

    # ---- output dir + resume
    out_root = Path(args.out_root or "results/phase1_scale/runs")
    run_dir = out_root / run_name
    if _is_main(rank):
        run_dir.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()

    global_step = 0
    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        ck = _latest_ckpt(run_dir)
        if ck is not None:
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            (head.module if world > 1 else head).load_state_dict(sd["model"])
            opt.load_state_dict(sd["opt"])
            global_step = int(sd["step"])
            start_epoch = int(sd["epoch"])
            best_val = float(sd.get("best_val", float("inf")))
            if _is_main(rank):
                log.info("resumed from %s @ step %d", ck.name, global_step)

    # ---- W&B (rank 0 only)
    wandb_run = None
    if _is_main(rank) and cfg.get("wandb", {}).get("enabled", False):
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg["wandb"].get("project", "geophys-wm"),
                entity=cfg["wandb"].get("entity"),
                name=f"{cfg['wandb'].get('run_name', 'phase1')}-{run_name}",
                config=cfg, resume="allow",
            )
        except Exception as e:
            log.warning("wandb init failed: %s", e)

    # ---- train
    amp_dtype = torch.bfloat16 if cfg["train"].get("amp_dtype", "bfloat16") == "bfloat16" else torch.float16
    epochs = cfg["train"]["epochs"]
    # Estimate total steps for cosine. With IterableDataset len is approximate.
    steps_per_epoch = max(1, len(train_ds) // (bs * max(1, world)))
    total_steps = steps_per_epoch * epochs
    warmup = cfg["train"]["warmup_steps"]
    grad_accum = max(1, int(cfg["train"].get("grad_accum", 1)))
    ckpt_every = int(cfg["train"]["ckpt_every_steps"])
    log_every = int(cfg["train"].get("log_every_steps", 50))

    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        if streaming:
            train_ds.set_epoch(epoch)
        head.train()
        opt.zero_grad(set_to_none=True)
        ep_losses = []
        for bi, batch in enumerate(train_dl):
            for k in ("ctx_tokens", "ctx_actions", "tgt_action", "tgt_tokens"):
                batch[k] = batch[k].to(device, non_blocking=True)
            lr = cosine_lr(global_step, total_steps, warmup, cfg["train"]["lr"])
            for g in opt.param_groups:
                g["lr"] = lr

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                pred = head(batch["ctx_tokens"], batch["ctx_actions"], batch["tgt_action"])
                loss = (pred.float() - batch["tgt_tokens"].float()).pow(2).mean() / grad_accum

            loss.backward()
            ep_losses.append(float(loss) * grad_accum)
            do_step = (bi + 1) % grad_accum == 0
            if do_step:
                torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["train"]["grad_clip"])
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                if _is_main(rank) and global_step % log_every == 0:
                    msg = f"ep {epoch} step {global_step}  loss {np.mean(ep_losses[-log_every:]):.4e}  lr {lr:.2e}  {time.time()-t0:.0f}s"
                    log.info(msg)
                    if wandb_run is not None:
                        wandb_run.log({"train/loss": np.mean(ep_losses[-log_every:]),
                                       "train/lr": lr, "epoch": epoch}, step=global_step)

                if _is_main(rank) and ckpt_every > 0 and global_step % ckpt_every == 0:
                    _save_ckpt(
                        run_dir / f"ckpt_step_{global_step:08d}.pt",
                        model=(head.module if world > 1 else head).state_dict(),
                        opt=opt.state_dict(),
                        step=global_step, epoch=epoch, best_val=best_val, cfg=cfg,
                    )

        # ---- per-epoch validation (rank 0 only — val_ds is world_size=1)
        if _is_main(rank):
            head.eval()
            with torch.no_grad():
                vs = []
                for batch in val_dl:
                    for k in ("ctx_tokens", "ctx_actions", "tgt_action", "tgt_tokens"):
                        batch[k] = batch[k].to(device, non_blocking=True)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                        pred = head(batch["ctx_tokens"], batch["ctx_actions"], batch["tgt_action"])
                        vs.append((pred.float() - batch["tgt_tokens"].float()).pow(2).mean().item())
                val_loss = float(np.mean(vs)) if vs else float("nan")
            train_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
            log.info("ep %d  train %.4e  val %.4e  step %d  %.1fs",
                     epoch, train_loss, val_loss, global_step, time.time() - t0)
            if wandb_run is not None:
                wandb_run.log({"val/loss": val_loss, "epoch": epoch}, step=global_step)
            if val_loss < best_val:
                best_val = val_loss
                torch.save((head.module if world > 1 else head).state_dict(), run_dir / "best.pt")
        if world > 1:
            dist.barrier()

    # ---- final ckpt
    if _is_main(rank):
        _save_ckpt(
            run_dir / f"ckpt_step_{global_step:08d}.pt",
            model=(head.module if world > 1 else head).state_dict(),
            opt=opt.state_dict(),
            step=global_step, epoch=epochs, best_val=best_val, cfg=cfg,
        )
        if wandb_run is not None:
            wandb_run.finish()

    return {"run_name": run_name, "best_val": best_val,
            "elapsed": time.time() - t0, "step": global_step}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/phase1/paper_scale.yaml")
    p.add_argument("--run", default="vggt_noact",
                   choices=["vggt", "vggt_noact", "vggt_bigact"])
    p.add_argument("--out_root", default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
                        datefmt="%H:%M:%S")
    cfg = yaml.safe_load(open(args.cfg))

    try:
        info = run_one(cfg, args.run, args)
        if int(os.environ.get("RANK", 0)) == 0:
            out_root = Path(args.out_root or "results/phase1_scale/runs")
            (out_root / "_summary.json").write_text(json.dumps(info, indent=2))
    finally:
        _ddp_cleanup()


if __name__ == "__main__":
    main()
