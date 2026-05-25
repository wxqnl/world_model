"""Generate paper-quality figures from the metrics CSVs."""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "results" / "metrics"
OUT = ROOT / "results" / "paper_figs"
OUT.mkdir(parents=True, exist_ok=True)


def load(name):
    with open(M / name) as f:
        return list(csv.DictReader(f))


def col(rows, key, cast=float, filt=None):
    out = []
    for r in rows:
        if filt and not filt(r):
            continue
        v = r[key]
        if v == "" or v is None:
            continue
        out.append(cast(v))
    return np.array(out)


# ------------------------------------------------------------------ Fig 1
# Exp 1 temporal coherence: 4-panel distribution summary.
e1 = load("exp1_consistency.csv")
depth = col(e1, "depth_rel_err_mean")
rot = col(e1, "rot_deg_mean")
trans = col(e1, "trans_err_mean")
cham = col(e1, "chamfer_mean")

fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
panels = [
    (depth, "Depth rel-err", "green <0.05", 0.05, 0.15),
    (rot, "Rotation (deg)", "green <2°", 2.0, 5.0),
    (trans, "Translation err", None, None, None),
    (cham, "Chamfer", None, None, None),
]
for ax, (v, title, _, g, y) in zip(axes, panels):
    ax.hist(v, bins=12, color="#4C72B0", edgecolor="white", alpha=0.9)
    ax.axvline(float(np.mean(v)), color="#C44E52", lw=1.5, ls="--",
               label=f"mean = {np.mean(v):.3f}")
    ax.axvline(float(np.median(v)), color="#55A868", lw=1.5, ls=":",
               label=f"median = {np.median(v):.3f}")
    if g is not None:
        ax.axvspan(0, g, alpha=0.08, color="green")
        ax.axvspan(g, y, alpha=0.08, color="gold")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("clips")
    ax.legend(fontsize=8, loc="upper right")
fig.suptitle("Figure 1. Temporal coherence across 30 DROID clips "
             "(W=8, S=4 sliding windows; per-clip means).", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig1_temporal_coherence.png")
plt.close(fig)

# ------------------------------------------------------------------ Fig 2
# Exp 3 token-flow correlation distribution (Set A manipulation).
e3 = load("exp3_token_flow_correlation.csv")
r3 = col(e3, "pearson_r")
p3 = col(e3, "p_value")
sig = p3 < 0.05

fig, ax = plt.subplots(figsize=(6.2, 3.8))
bins = np.linspace(0, 1, 11)
ax.hist(r3[~sig], bins=bins, color="#CCCCCC", edgecolor="white",
        label=f"p ≥ 0.05 ({(~sig).sum()} clips)")
ax.hist(r3[sig], bins=bins, color="#4C72B0", edgecolor="white",
        label=f"p < 0.05 ({sig.sum()} clips)")
ax.axvline(0.2, color="#C44E52", lw=1.5, ls="--", label="red threshold (0.2)")
ax.axvline(0.5, color="#55A868", lw=1.5, ls="--", label="green threshold (0.5)")
ax.axvline(float(np.median(r3)), color="black", lw=1.5,
           label=f"median = {np.median(r3):.3f}")
ax.set_xlabel("Pearson r (token L2 Δ  vs  RAFT flow magnitude)")
ax.set_ylabel("clips")
ax.set_xlim(0, 1.0)
ax.set_title("Figure 2. Token-dynamics correlation on manipulation (Set A, n = 30).",
             fontsize=11)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "fig2_token_flow_corr_setA.png")
plt.close(fig)

# ------------------------------------------------------------------ Fig 3
# Cross-domain comparison: box + strip.
e4 = load("exp4_cross_domain.csv")
rA = col(e4, "pearson_r", filt=lambda r: r["domain"] == "manipulation")
rC = col(e4, "pearson_r", filt=lambda r: r["domain"] == "autonomous_driving")
pA = col(e4, "p_value",   filt=lambda r: r["domain"] == "manipulation")
pC = col(e4, "p_value",   filt=lambda r: r["domain"] == "autonomous_driving")

fig, ax = plt.subplots(figsize=(6.2, 4.0))
parts = ax.boxplot([rA, rC], positions=[1, 2], widths=0.55,
                   patch_artist=True, showfliers=False)
for patch, c in zip(parts["boxes"], ["#4C72B0", "#DD8452"]):
    patch.set_facecolor(c); patch.set_alpha(0.35)
for med in parts["medians"]:
    med.set_color("black"); med.set_linewidth(1.5)

rng = np.random.default_rng(0)
for x, vals, ps, c in [(1, rA, pA, "#4C72B0"), (2, rC, pC, "#DD8452")]:
    jitter = rng.normal(0, 0.06, size=len(vals))
    edge = ["black" if p < 0.05 else "#AAAAAA" for p in ps]
    ax.scatter(x + jitter, vals, s=34, color=c, edgecolor=edge, linewidth=0.8, zorder=3)

ax.axhline(0.5, color="#55A868", lw=1.0, ls="--", alpha=0.7)
ax.axhline(0.2, color="#C44E52", lw=1.0, ls="--", alpha=0.7)
ax.axhline(0.0, color="black", lw=0.5, alpha=0.5)
ax.set_xticks([1, 2])
ax.set_xticklabels([f"Manipulation\n(DROID, n={len(rA)})",
                    f"Driving\n(KITTI, n={len(rC)})"])
ax.set_ylabel("Pearson r  (token Δ vs flow)")
ax.set_ylim(-0.35, 0.95)
ax.set_title("Figure 3. Cross-domain token–flow correlation. "
             "Filled markers = p < 0.05.", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig3_cross_domain.png")
plt.close(fig)

# ------------------------------------------------------------------ Fig 4
# Exp 2: flow magnitude vs photometric L1 (quality-vs-motion).
e2 = load("exp2_static_vs_dynamic.csv")
dyn = [r for r in e2 if r["domain"] == "dynamic"]
fm = col(dyn, "flow_mag_mean")
pl = col(dyn, "photometric_l1_mean")
conf = col(dyn, "conf_mean")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
sc = ax1.scatter(fm, pl, c=conf, cmap="viridis", s=50, edgecolor="black", linewidth=0.4)
# OLS fit
if len(fm) > 2:
    m, b = np.polyfit(fm, pl, 1)
    xs = np.linspace(fm.min(), fm.max(), 100)
    ax1.plot(xs, m * xs + b, color="#C44E52", lw=1.5, ls="--",
             label=f"fit: y = {m:.4f}·x + {b:.4f}")
    # Pearson
    r = float(np.corrcoef(fm, pl)[0, 1])
    ax1.text(0.05, 0.95, f"Pearson r = {r:.3f}", transform=ax1.transAxes,
             va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#AAAAAA"))
ax1.set_xlabel("mean optical-flow magnitude (px)")
ax1.set_ylabel("photometric L1 (self-consistency)")
ax1.set_title("(a) Motion vs reconstruction error (Set A, n = 30)", fontsize=10)
ax1.legend(fontsize=8, loc="lower right")
plt.colorbar(sc, ax=ax1, label="mean confidence")

# Static vs dynamic confidence distributions.
sta = [r for r in e2 if r["domain"] == "static"]
conf_s = col(sta, "conf_mean")
parts = ax2.violinplot([conf_s, conf], positions=[1, 2], widths=0.75, showmeans=False,
                       showmedians=True, showextrema=False)
for body, c in zip(parts["bodies"], ["#8172B2", "#4C72B0"]):
    body.set_facecolor(c); body.set_alpha(0.45); body.set_edgecolor("black")
ax2.set_xticks([1, 2]); ax2.set_xticklabels(["Static\n(ScanNet, n=20)", "Dynamic\n(DROID, n=30)"])
ax2.set_ylabel("mean per-frame confidence")
ax2.set_title("(b) Confidence distributions", fontsize=10)

fig.suptitle("Figure 4. Quality-vs-motion diagnostic.", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig4_quality_vs_motion.png")
plt.close(fig)

# ------------------------------------------------------------------ Fig 5
# Per-clip ranking (Exp 3 manipulation): horizontal bar.
order = np.argsort(r3)
fig, ax = plt.subplots(figsize=(6.0, 7.5))
colors = ["#4C72B0" if s else "#BBBBBB" for s in sig[order]]
ax.barh(np.arange(len(r3)), r3[order], color=colors, edgecolor="white")
ax.set_yticks(np.arange(len(r3)))
ax.set_yticklabels([e3[i]["clip_id"] for i in order], fontsize=7)
ax.axvline(0.2, color="#C44E52", lw=1.0, ls="--", label="red (0.2)")
ax.axvline(0.5, color="#55A868", lw=1.0, ls="--", label="green (0.5)")
ax.axvline(float(np.median(r3)), color="black", lw=1.0,
           label=f"median = {np.median(r3):.3f}")
ax.set_xlabel("Pearson r")
ax.set_xlim(0, 1.0)
ax.set_title("Figure 5. Per-clip token-flow correlation (Set A).", fontsize=11)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "fig5_per_clip_ranking.png")
plt.close(fig)

print("wrote:")
for p in sorted(OUT.glob("*.png")):
    print(" ", p.relative_to(ROOT))
