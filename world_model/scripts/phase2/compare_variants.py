"""Compare Phase 2 variants side-by-side.

Reads train_log.json from each variant directory, produces:
  * results/phase2/ablations.json     — consolidated metrics table
  * results/phase2/plot_ablations.png — per-epoch val_fm / val_sc curves
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_log(run_dir: Path) -> list[dict]:
    log_path = run_dir / "train_log.json"
    if not log_path.exists():
        return []
    return json.loads(log_path.read_text())


def best_summary(log: list[dict]) -> dict:
    if not log:
        return {}
    best_fm = min(log, key=lambda r: r["val_fm"])
    best_sc = min((r for r in log if r.get("val_sc") is not None),
                  key=lambda r: r["val_sc"], default=None)
    return {
        "epochs": len(log),
        "best_val_fm": best_fm["val_fm"],
        "best_val_fm_epoch": best_fm["epoch"],
        "final_val_fm": log[-1]["val_fm"],
        "final_val_sc": log[-1].get("val_sc"),
        "final_val_ph": log[-1].get("val_ph"),
        "best_val_sc": best_sc["val_sc"] if best_sc else None,
        "best_val_sc_epoch": best_sc["epoch"] if best_sc else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="results/phase2/runs")
    ap.add_argument("--variants", nargs="+",
                    default=["flow_only", "flow_coupled", "flow_coupled_sched20",
                             "flow_coupled_w05", "flow_coupled_w025"])
    ap.add_argument("--out_json", default="results/phase2/ablations.json")
    ap.add_argument("--out_plot", default="results/phase2/plot_ablations.png")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    logs: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}
    for v in args.variants:
        log = load_log(runs_dir / v)
        if not log:
            print(f"[warn] no log for {v} at {runs_dir/v}")
            continue
        logs[v] = log
        summary[v] = best_summary(log)

    Path(args.out_json).write_text(json.dumps(summary, indent=2))

    # -------------- plot val_fm and val_sc per epoch
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    colors = plt.cm.tab10.colors
    for i, (v, log) in enumerate(logs.items()):
        eps = [r["epoch"] for r in log]
        val_fm = [r["val_fm"] for r in log]
        val_sc = [r.get("val_sc") for r in log]
        col = colors[i % len(colors)]
        axes[0].plot(eps, val_fm, label=v, color=col, lw=1.5)
        if any(s is not None for s in val_sc):
            axes[1].plot(eps, [s for s in val_sc], label=v, color=col, lw=1.5)

    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("val_fm")
    axes[0].set_title("Flow-matching val loss (lower = better sampler)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val_sc")
    axes[1].set_title("Predictor self-consistency val loss (lower = more predictable)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # -------------- print comparison table
    print("\n=== Phase 2 ablations ===")
    cols = ["variant", "best val_fm @ep", "best val_sc @ep", "final val_sc"]
    print(f"{cols[0]:<24s} {cols[1]:<20s} {cols[2]:<20s} {cols[3]:<14s}")
    for v, s in summary.items():
        bfm = f"{s['best_val_fm']:.4f} @ {s['best_val_fm_epoch']}"
        bsc = (f"{s['best_val_sc']:.4f} @ {s['best_val_sc_epoch']}"
               if s.get("best_val_sc") is not None else "—")
        fsc = f"{s['final_val_sc']:.4f}" if s.get("final_val_sc") is not None else "—"
        print(f"{v:<24s} {bfm:<20s} {bsc:<20s} {fsc:<14s}")
    print(f"\nwrote {args.out_json} and {args.out_plot}")


if __name__ == "__main__":
    main()
