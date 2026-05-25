"""Phase 2 pixel-space feasibility eval.

Answers: "do coupled-generator samples project to more plausible depth maps
than uncoupled ones, when judged by a real-data-trained token->depth decoder?"

Pipeline:
    1. Train a tiny TokenDepthDecoder (pooled tokens -> 64x64 depth) on the
       Phase 1 cache train split. Early stop on val L1.
    2. Load flow_only + flow_coupled generator checkpoints.
    3. For each variant, sample N windows (same seed, same inits) and decode.
    4. Compare against real decoded depth: temporal smoothness, depth
       distribution (histogram), and a qualitative strip.

Outputs under `results/phase2/pixel_eval/`:
    decoder.pt                          (trained decoder weights)
    metrics.json                        (aggregate comparison)
    strip_<variant>.png                 (T=8 decoded strip)
    compare_grid.png                    (3 rows: real / flow_only / flow_coupled)
    hist.png                            (depth distribution per variant)

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m scripts.phase2.pixel_eval \\
        --cfg configs/phase2/default.yaml \\
        --decoder_epochs 60 --n_samples 8
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase1.dataset import Shard, discover_shards, _load_tokens
from src.phase2.dataset import build_datasets, collate
from src.phase2.generative import FlowMatchingGenerator, GenerativeConfig
from src.phase2.pixel_decode import (
    DecoderConfig, TokenDepthDecoder, decoder_loss, normalize_depth,
)
from src.phase2.text_encoder import FrozenCLIPText


# ---------------------------------------------------------------- data helpers
class FrameTokenDepth(Dataset):
    """Per-frame (pooled-tokens, cached-depth) pairs from the Phase 1 cache."""

    def __init__(self, shards: list[Shard]):
        self.shards = shards
        self.index: list[tuple[int, int]] = []
        for si, sh in enumerate(shards):
            for t in range(sh.n_frames):
                self.index.append((si, t))
        self._cache: dict[int, dict] = {}

    def _load(self, si: int) -> dict:
        if si not in self._cache:
            d = np.load(self.shards[si].path)
            self._cache[si] = {"tokens": np.asarray(d["tokens"]),
                               "depth": np.asarray(d["depth"])}
            d.close()
        return self._cache[si]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        si, t = self.index[i]
        sh = self._load(si)
        tok = _load_tokens(sh["tokens"][t])             # [P, D] fp32
        dep = torch.from_numpy(sh["depth"][t]).float()  # [H, W]
        return {"tok": tok, "depth": dep}


def _frame_collate(batch: list[dict]) -> dict:
    return {
        "tok": torch.stack([b["tok"] for b in batch]),
        "depth": torch.stack([b["depth"] for b in batch]),
    }


# ------------------------------------------------------------------ decoder fit
def train_decoder(
    train_ds: FrameTokenDepth,
    val_ds: FrameTokenDepth,
    epochs: int,
    device: torch.device,
    out_dir: Path,
) -> TokenDepthDecoder:
    decoder = TokenDepthDecoder(DecoderConfig()).to(device)
    n_params = sum(p.numel() for p in decoder.parameters()) / 1e6
    print(f"[decoder] params: {n_params:.3f} M")

    opt = torch.optim.AdamW(decoder.parameters(), lr=3e-4, weight_decay=1e-4)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True,
                          num_workers=2, collate_fn=_frame_collate, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False,
                        num_workers=2, collate_fn=_frame_collate)

    best = float("inf")
    log = []
    t0 = time.time()
    for ep in range(epochs):
        decoder.train()
        tr = 0.0; n = 0
        for b in train_dl:
            tok = b["tok"].to(device); dep = b["depth"].to(device)
            pred = decoder(tok)
            loss = decoder_loss(pred, dep)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt.step()
            tr += loss.item(); n += 1
        decoder.eval()
        with torch.no_grad():
            vl = 0.0; vn = 0
            for b in val_dl:
                tok = b["tok"].to(device); dep = b["depth"].to(device)
                pred = decoder(tok)
                vl += decoder_loss(pred, dep).item(); vn += 1
        row = {"ep": ep, "train_l1": tr / max(1, n),
               "val_l1": vl / max(1, vn), "elapsed": time.time() - t0}
        log.append(row)
        if row["val_l1"] < best:
            best = row["val_l1"]
            torch.save(decoder.state_dict(), out_dir / "decoder.pt")
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[decoder] ep {ep:3d}  train {row['train_l1']:.4f}  "
                  f"val {row['val_l1']:.4f}  best {best:.4f}")
    (out_dir / "decoder_log.json").write_text(json.dumps(log, indent=2))
    decoder.load_state_dict(torch.load(out_dir / "decoder.pt",
                                       map_location=device, weights_only=True))
    return decoder


# ------------------------------------------------------------------ generator
def load_generator(ckpt_path: Path, cfg: dict, device: torch.device) -> FlowMatchingGenerator:
    gen_cfg = GenerativeConfig(**cfg["generator"])
    gen = FlowMatchingGenerator(gen_cfg).to(device)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    gen.load_state_dict(sd["gen"])
    gen.eval()
    return gen


@torch.no_grad()
def sample_variant(
    gen: FlowMatchingGenerator,
    cond_text: torch.Tensor,
    init_frame: torch.Tensor,
    n_steps: int,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return gen.sample(cond_text, init_frame, n_steps=n_steps)


# ------------------------------------------------------------------ metrics
def temporal_smoothness(depth_seq: torch.Tensor) -> float:
    """Mean |d_t - d_{t-1}| across a sequence [..., T, H, W], normalized by
    per-sequence median to be scale-free."""
    T = depth_seq.shape[-3]
    med = depth_seq.flatten(-3).median(dim=-1).values.clamp_min(1e-6)
    med = med.view(*med.shape, 1, 1, 1)
    d = depth_seq / med
    diffs = (d[..., 1:, :, :] - d[..., :-1, :, :]).abs().mean().item()
    return diffs


def distribution_stats(depth_seq: torch.Tensor) -> dict[str, float]:
    """Scale-normalized per-sequence stats."""
    med = depth_seq.flatten(-3).median(dim=-1).values.clamp_min(1e-6)
    med = med.view(*med.shape, 1, 1, 1)
    d = (depth_seq / med).flatten()
    return {
        "mean": float(d.mean().item()),
        "std": float(d.std().item()),
        "p10": float(d.quantile(0.10).item()),
        "p50": float(d.quantile(0.50).item()),
        "p90": float(d.quantile(0.90).item()),
    }


def wasserstein_1d(a: torch.Tensor, b: torch.Tensor, n: int = 4096) -> float:
    """Cheap 1-D Wasserstein between two empirical depth distributions on the
    scale-normalized axis. Subsamples to `n` to keep it fast."""
    rng = np.random.default_rng(0)
    aa = a.flatten().cpu().numpy(); bb = b.flatten().cpu().numpy()
    if aa.size > n:
        aa = aa[rng.choice(aa.size, n, replace=False)]
    if bb.size > n:
        bb = bb[rng.choice(bb.size, n, replace=False)]
    aa.sort(); bb.sort()
    # resample to equal length via interpolation of the sorted quantile functions
    m = min(aa.size, bb.size)
    aa = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, aa.size), aa)
    bb = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, bb.size), bb)
    return float(np.abs(aa - bb).mean())


def scale_normalize(depth: torch.Tensor) -> torch.Tensor:
    """Per-sequence scale normalization by median (broadcast-safe)."""
    med = depth.flatten(-3).median(dim=-1).values.clamp_min(1e-6)
    med = med.view(*med.shape, 1, 1, 1)
    return depth / med


# ------------------------------------------------------------------ plotting
def save_strip(depth: np.ndarray, path: Path, title: str) -> None:
    """depth: [T, H, W] for a single sample."""
    T = depth.shape[0]
    fig, axes = plt.subplots(1, T, figsize=(1.8 * T, 2.0))
    vmin = np.percentile(depth, 2); vmax = np.percentile(depth, 98)
    for t in range(T):
        ax = axes[t]
        ax.imshow(depth[t], cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t={t}", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_compare_grid(
    rows: list[tuple[str, np.ndarray]], path: Path, max_samples: int = 4,
) -> None:
    """Stack arbitrary rows of (label, [B, T, H, W]); show first frame per sample."""
    max_samples = min([max_samples] + [arr.shape[0] for _, arr in rows])
    R = len(rows)
    fig, axes = plt.subplots(R, max_samples, figsize=(2.2 * max_samples, 2.0 * R + 0.6))
    if max_samples == 1:
        axes = axes[:, None]
    if R == 1:
        axes = axes[None, :]
    for r, (label, arr) in enumerate(rows):
        for c in range(max_samples):
            ax = axes[r, c]
            img = arr[c, 0]
            vmin = np.percentile(arr[c], 2); vmax = np.percentile(arr[c], 98)
            ax.imshow(img, cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
            if r == 0:
                ax.set_title(f"sample {c}", fontsize=10)
    fig.suptitle("Decoded depth — row: variant, col: sample (t=0)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_hist(entries: list[tuple[str, np.ndarray]], path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.5))
    bins = np.linspace(0, 3.0, 60)  # scale-normalized depth
    for label, arr in entries:
        ax.hist(arr.flatten(), bins=bins, alpha=0.4, label=label, density=True)
    ax.set_xlabel("depth / per-sequence median")
    ax.set_ylabel("density")
    ax.set_title("Scale-normalized decoded depth distribution")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/phase2/default.yaml")
    ap.add_argument("--decoder_epochs", type=int, default=60)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--sample_steps", type=int, default=24)
    ap.add_argument("--out", default="results/phase2/pixel_eval")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--variants", nargs="+",
                    default=["flow_only=results/phase2/runs/flow_only/best.pt",
                             "flow_coupled=results/phase2/runs/flow_coupled/best.pt"],
                    help="name=ckpt_path pairs, space-separated")
    ap.add_argument("--decoder_ckpt", default=None,
                    help="if set, skip decoder training and load this decoder.pt")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.cfg).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # --------------- decoder training data
    shards = discover_shards(cfg["data"]["cache_dir"])
    val_eps = set(cfg["data"]["val_episode_ids"])
    train_sh = [s for s in shards if s.episode_index not in val_eps]
    val_sh = [s for s in shards if s.episode_index in val_eps]
    print(f"shards: train {len(train_sh)}  val {len(val_sh)}")
    train_ds = FrameTokenDepth(train_sh)
    val_ds = FrameTokenDepth(val_sh)
    print(f"frames: train {len(train_ds)}  val {len(val_ds)}")

    # --------------- train or load decoder
    if args.decoder_ckpt is not None:
        decoder = TokenDepthDecoder(DecoderConfig()).to(device)
        decoder.load_state_dict(torch.load(args.decoder_ckpt,
                                           map_location=device, weights_only=True))
        decoder.eval()
        print(f"[decoder] loaded from {args.decoder_ckpt}")
    else:
        decoder = train_decoder(train_ds, val_ds, args.decoder_epochs, device, out_dir)

    # --------------- collect generation inits from val set
    _, gen_val_ds = build_datasets(cfg)
    dl = DataLoader(gen_val_ds, batch_size=args.n_samples, shuffle=False,
                    num_workers=0, collate_fn=collate)
    batch = next(iter(dl))
    z_real = batch["z"].to(device)                       # [B, T, P, D]
    init = batch["init"].to(device)                      # [B, P, D]
    text_enc = FrozenCLIPText(cfg["text_encoder"]["hf_id"],
                              cfg["text_encoder"]["max_length"]).to(device)
    cond = text_enc(batch["text"], device)               # [B, Dtxt]

    B, T, P, D = z_real.shape

    # --------------- sample from each variant
    variants: dict[str, str] = {}
    for spec in args.variants:
        if "=" not in spec:
            raise ValueError(f"--variants entry must be name=path: {spec}")
        name, path = spec.split("=", 1)
        variants[name] = path
    z_gen: dict[str, torch.Tensor] = {}
    for name, path in variants.items():
        gen = load_generator(Path(path), cfg, device)
        z_gen[name] = sample_variant(gen, cond, init, n_steps=args.sample_steps,
                                     seed=args.seed)
        print(f"[sampled] {name}: {tuple(z_gen[name].shape)}")
        del gen; torch.cuda.empty_cache()

    # --------------- decode to depth
    decoder.eval()
    with torch.no_grad():
        def decode_seq(z: torch.Tensor) -> torch.Tensor:
            flat = z.reshape(B * T, P, D)
            d = decoder(flat)
            return d.reshape(B, T, d.shape[-2], d.shape[-1])

        d_real = decode_seq(z_real)
        d_variants = {name: decode_seq(z) for name, z in z_gen.items()}

    # --------------- metrics on scale-normalized decoded depth
    d_real_n = scale_normalize(d_real)
    d_var_n = {name: scale_normalize(d) for name, d in d_variants.items()}

    decoder_val_l1 = None
    dec_log = out_dir / "decoder_log.json"
    if dec_log.exists():
        decoder_val_l1 = float(json.loads(dec_log.read_text())[-1]["val_l1"])

    metrics = {
        "config": {
            "n_samples": args.n_samples, "sample_steps": args.sample_steps,
            "decoder_epochs": args.decoder_epochs, "seed": args.seed,
            "variants": variants,
        },
        "decoder_val_l1": decoder_val_l1,
        "temporal_smoothness": {
            "real": temporal_smoothness(d_real_n),
            **{name: temporal_smoothness(d) for name, d in d_var_n.items()},
        },
        "distribution": {
            "real": distribution_stats(d_real_n),
            **{name: distribution_stats(d) for name, d in d_var_n.items()},
        },
        "wasserstein_vs_real": {
            name: wasserstein_1d(d, d_real_n) for name, d in d_var_n.items()
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # --------------- visualizations
    real_np = d_real_n.cpu().numpy()
    var_np = {name: d.cpu().numpy() for name, d in d_var_n.items()}

    save_strip(real_np[0], out_dir / "strip_real.png", "real (cache) — sample 0")
    for name, arr in var_np.items():
        save_strip(arr[0], out_dir / f"strip_{name}.png", f"{name} — sample 0")

    rows = [("real (cache)", real_np)] + [(name, arr) for name, arr in var_np.items()]
    save_compare_grid(rows, out_dir / "compare_grid.png")
    hist_entries = [("real (cache)", real_np)] + [(name, arr) for name, arr in var_np.items()]
    save_hist(hist_entries, out_dir / "hist.png")

    print("\n=== pixel-space feasibility eval ===")
    ts = metrics["temporal_smoothness"]
    ws = metrics["wasserstein_vs_real"]
    print("temporal smoothness (lower = smoother):")
    print(f"  real         {ts['real']:.4f}")
    for name in var_np:
        print(f"  {name:<20s} {ts[name]:.4f}")
    print("wasserstein vs real depth distribution (lower = closer):")
    for name in var_np:
        print(f"  {name:<20s} {ws[name]:.4f}")
    print(f"\nartifacts written to {out_dir}")


if __name__ == "__main__":
    main()
