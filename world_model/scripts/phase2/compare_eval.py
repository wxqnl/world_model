"""Post-hoc Phase 2 comparison: run both variants through the SAME sampler +
frozen predictor and compute val_sc (predictor self-consistency) on held-out.

Produces results/phase2/comparison.json and plot results/phase2/plot_compare.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase2.coupling import predictor_selfconsistency_loss
from src.phase2.dataset import build_datasets, collate
from src.phase2.generative import FlowMatchingGenerator, GenerativeConfig, flow_matching_loss
from src.phase2.text_encoder import FrozenCLIPText
from scripts.phase2.train_generative import load_predictor, sample_with_grad


def load_gen(ckpt_path: str, device: torch.device) -> FlowMatchingGenerator:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    gcfg = GenerativeConfig(**ck["cfg"]["generator"])
    g = FlowMatchingGenerator(gcfg)
    g.load_state_dict(ck["gen"])
    g.eval()
    return g.to(device)


def eval_one(gen, predictor, val_dl, text_enc, cfg, device, n_steps):
    fm_losses, sc_losses = [], []
    for batch in val_dl:
        z = batch["z"].to(device)
        init = batch["init"].to(device)
        cond = text_enc(batch["text"], device)
        fm_losses.append(flow_matching_loss(gen, z, cond, init).item())
        with torch.no_grad():
            z_gen = sample_with_grad(gen, cond, init, n_steps=n_steps)
        sc_losses.append(predictor_selfconsistency_loss(
            predictor, z_gen, cfg["coupling"]["context_len"]).item())
    return {
        "val_fm": float(np.mean(fm_losses)),
        "val_fm_sem": float(np.std(fm_losses) / np.sqrt(len(fm_losses))),
        "val_sc": float(np.mean(sc_losses)),
        "val_sc_sem": float(np.std(sc_losses) / np.sqrt(len(sc_losses))),
        "n_batches": len(fm_losses),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+",
                    default=["flow_only", "flow_coupled"],
                    help="variant names under results/phase2/runs/")
    ap.add_argument("--out_json", default="results/phase2/comparison.json")
    ap.add_argument("--out_plot", default="results/phase2/plot_compare.png")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path("configs/phase2/default.yaml").read_text())
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_ds = build_datasets(cfg)
    val_dl = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=False, num_workers=0, collate_fn=collate)
    text_enc = FrozenCLIPText(cfg["text_encoder"]["hf_id"],
                              cfg["text_encoder"]["max_length"]).to(device)
    predictor = load_predictor(cfg, device)

    results = {}
    for variant in args.variants:
        ckpt = f"results/phase2/runs/{variant}/best.pt"
        if not Path(ckpt).exists():
            print(f"[warn] missing {ckpt}, skipping"); continue
        gen = load_gen(ckpt, device)
        torch.manual_seed(cfg["seed"])  # same noise across variants
        results[variant] = eval_one(gen, predictor, val_dl, text_enc, cfg, device,
                                    n_steps=cfg["eval"]["sample_steps"])
        print(f"{variant}: {results[variant]}")
        del gen
        torch.cuda.empty_cache()

    if "flow_only" in results:
        base = results["flow_only"]
        deltas = {}
        for v, r in results.items():
            if v == "flow_only":
                continue
            pct_fm = 100.0 * (r["val_fm"] - base["val_fm"]) / max(1e-12, base["val_fm"])
            pct_sc = 100.0 * (base["val_sc"] - r["val_sc"]) / max(1e-12, base["val_sc"])
            deltas[v] = {"val_fm_cost_pct": pct_fm, "val_sc_improvement_pct": pct_sc}
        out = {"per_variant": results, "vs_flow_only_pct": deltas,
               "sample_steps": cfg["eval"]["sample_steps"],
               "val_set_size": len(val_ds)}
    else:
        out = {"per_variant": results, "sample_steps": cfg["eval"]["sample_steps"],
               "val_set_size": len(val_ds)}
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(json.dumps(out.get("vs_flow_only_pct", {}), indent=2))

    # plot
    variants = list(results.keys())
    metrics = ["val_fm", "val_sc"]
    titles = ["val flow-matching loss (lower = better sampler)",
              "val predictor self-consistency (lower = more predictable)"]
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 2.5 * len(variants)), 4))
    colors = plt.cm.tab10.colors
    for i, (m, ti) in enumerate(zip(metrics, titles)):
        means = [results[v][m] for v in variants]
        sems = [results[v][m + "_sem"] for v in variants]
        axes[i].bar(variants, means, yerr=sems, capsize=6,
                    color=[colors[k % len(colors)] for k in range(len(variants))])
        axes[i].set_title(ti, fontsize=10)
        axes[i].set_ylabel(m)
        axes[i].tick_params(axis="x", rotation=25, labelsize=9)
        for j, mu in enumerate(means):
            axes[i].text(j, mu, f"{mu:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=120)
    plt.close()
    print(f"wrote {args.out_json} and {args.out_plot}")


if __name__ == "__main__":
    main()
