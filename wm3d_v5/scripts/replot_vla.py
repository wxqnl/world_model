"""Re-render VLA plots from the existing report.json + demo npz files (no GPU)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from analyze_vla import plot_quant, plot_trajectories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--demo_dirs", nargs="*", default=[
        "/home/user01/Minko/newwm/results/wm3d_v3/eval/demo",
        "/home/user01/Minko/newwm/results/wm3d_v3/eval/demo_full",
        "/home/user01/Minko/newwm/results/wm3d_v3/eval/demo_long",
    ])
    args = ap.parse_args()
    report = json.loads((args.out / "report.json").read_text())
    plot_dir = args.out / "plots"
    plot_quant(report, plot_dir)
    plot_trajectories([Path(p) for p in args.demo_dirs], plot_dir)
    print(f"plots in {plot_dir}")


if __name__ == "__main__":
    main()
