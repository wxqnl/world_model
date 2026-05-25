"""Exp 4 — Cross-domain probe: repeat exp1 & exp3 on Set C (autonomous driving).

Compares metric distributions between Set A (manipulation) and Set C (AD).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments.exp1_temporal_coherence as exp1
import experiments.exp3_token_dynamics as exp3


def run(cfg_path="configs/default.yaml", out_dir="results", limit=None):
    out_dir = Path(out_dir)

    # Exp 3 on Set A and Set C for comparison
    df3_a = exp3.run(cfg_path, str(out_dir), limit=limit, manifest_key="set_a")
    df3_c = exp3.run(cfg_path, str(out_dir), limit=limit, manifest_key="set_c")

    df = pd.concat([df3_a.assign(domain="manipulation"),
                    df3_c.assign(domain="autonomous_driving")], ignore_index=True)
    df.to_csv(out_dir / "metrics/exp4_cross_domain.csv", index=False)

    plt.figure(figsize=(7, 4))
    for dom, sub in df.groupby("domain"):
        plt.hist(sub["pearson_r"].dropna(), bins=20, alpha=0.5, label=dom)
    plt.xlabel("per-clip Pearson r (token Δ vs flow)"); plt.ylabel("count"); plt.legend()
    plt.tight_layout(); plt.savefig(out_dir / "plots/exp4_domain_comparison.png", dpi=120); plt.close()
    print(df.groupby("domain")["pearson_r"].describe())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.config, a.out, a.limit)
