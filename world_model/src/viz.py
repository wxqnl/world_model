"""Qualitative visualization helpers. Minimal surface: only what experiments need."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_depth_pair(d1: torch.Tensor, d2: torch.Tensor, out: str | Path, title: str = ""):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    for a, d, t in zip(ax, [d1, d2], ["window 1", "window 2"]):
        dn = d.detach().float().cpu().numpy()
        a.imshow(dn, cmap="turbo")
        a.set_title(t); a.axis("off")
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def save_heatmap_pair(h1: np.ndarray, h2: np.ndarray, out: str | Path, labels=("token Δ", "optical flow")):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    for a, h, lab in zip(ax, [h1, h2], labels):
        a.imshow(h, cmap="magma"); a.set_title(lab); a.axis("off")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def save_histogram(values, out: str | Path, xlabel: str, bins: int = 30):
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    plt.figure(figsize=(6, 4))
    plt.hist(v, bins=bins, edgecolor="k", alpha=0.8)
    plt.xlabel(xlabel); plt.ylabel("count"); plt.tight_layout()
    plt.savefig(out, dpi=120); plt.close()


def save_scatter(x, y, out: str | Path, xlabel: str, ylabel: str):
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.6)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.tight_layout()
    plt.savefig(out, dpi=120); plt.close()
