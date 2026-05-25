# Genspark figure prompts

The five data-driven figures (Figures 1–5) are generated from the CSVs by
`scripts/make_paper_figs.py` and live in `results/paper_figs/`. Those are
the authoritative plots for the paper.

The prompts below are for the *conceptual* / hero figures that are not
data-driven — a system diagram and a pipeline flow chart. Paste each
prompt directly into Genspark (or an equivalent diagram generator) and
replace `results/paper_figs/fig0_*.png` when the renders come back.

---

## Prompt A — Figure 0 (hero / system diagram)

> **Style:** Clean academic vector diagram for an ML paper. Flat design,
> white background, muted colour palette (navy `#1F3A5F`, teal `#3C8A8E`,
> warm grey `#7A7870`, accent coral `#D1655A`). Sans-serif labels (Inter
> or similar). No 3D, no gradients, no icons outside the canvas.
>
> **Title above diagram:** "GeoPhys-WM Phase 0 — what we evaluate"
>
> **Composition (left to right):**
>
> 1. **Input column.** Three stacked thumbnails labelled, top to bottom:
>    "Set A — manipulation (DROID, 30 clips)",
>    "Set B — static scenes (ScanNet, 20 scenes)",
>    "Set C — driving (KITTI, 11 drives)".
>    Each thumbnail is a rounded rectangle with a representative sketch
>    (robot arm on table; indoor room; road with cars).
>
> 2. **Backbone column.** One large rounded box labelled
>    **"VGGT-1B (frozen, bf16)"**, with four sub-ports on its right side:
>    `depth`, `point map`, `camera (R, t, K)`, and — highlighted in coral —
>    `aggregator tokens  [T, 1374, 2048]`. Small "❄" snowflake icon in the
>    top-left corner of the box to denote "frozen".
>
> 3. **Diagnostics column.** Four rounded cards, one per experiment, each
>    labelled:
>    - Exp 1 — Temporal coherence (depth / R / t / Chamfer over 8-frame
>      sliding windows)
>    - Exp 2 — Static vs dynamic quality (AbsRel, photometric L1)
>    - **Exp 3 — Token dynamics★** (Pearson r of token ΔL2 vs RAFT flow)
>      — draw a star next to it and a thin coral border to mark it as the
>      critical gate.
>    - Exp 4 — Cross-domain probe (Set A vs Set C)
>
> 4. **Verdict column (far right).** One rounded box labelled
>    "Verdict: cautious proceed" in green, with two sub-bullets:
>    "• median r = 0.47 – 0.52" and "• depth drift = 5.5 %".
>
> **Flow arrows:** Thin arrows from each input thumbnail into the VGGT
> box, from the tokens port out to the four experiment cards, and from
> the four cards into the verdict box. Arrow from Exp 3 is thicker /
> coral-coloured to mark "critical gate".
>
> **Footer line, small grey text:** "Frozen VGGT-1B · no fine-tuning ·
> 60 – 120 frames per clip · 518 × 518 input".
>
> Output: SVG or 3000×1600 PNG, paper-ready, no watermarks.

---

## Prompt B — Figure 0b (single-clip pipeline, optional)

Use this if Prompt A feels too busy; pick whichever reads cleaner.

> **Style:** Horizontal pipeline diagram for an ML paper. White
> background, flat vector, same palette as Prompt A. Sans-serif labels.
>
> **Stages left → right, each a rounded card:**
>
> 1. "Input: 8-frame clip (518 × 518)". Small filmstrip icon showing
>    8 tiny frames.
> 2. "VGGT-1B (frozen) — aggregator tokens [8, 1374, 2048]". Snowflake
>    icon in the top-left corner.
> 3. "Δtokens_t = ‖z_{t+1} − z_t‖₂". Show a small sparkline below.
> 4. "RAFT flow magnitude \|F_t\|". Show a small sparkline below.
> 5. "Pearson r across 7 pairs". Show a miniature scatter plot with a
>    fitted line.
> 6. Result card: **"median r = 0.47 on Set A, 0.52 on Set C  →
>    YELLOW / GREEN"**.
>
> Arrows between stages. Label the arrow from stage 2 → 3 as
> "per-frame token deltas" and from stage 2 → 4 as "RGB frames" (RAFT
> operates on pixels, not tokens).
>
> Output: SVG or 3000×900 PNG.

---

## Prompt C — conceptual cartoon (optional, for slides only)

> **Style:** Clean whiteboard-style illustration, two panels side by
> side, flat vector, soft pastel fills, thin black strokes.
>
> **Left panel, title "What a geometry backbone sees":** a 3D scene
> diagram with a robot arm picking up a block on a table; arrows from
> the scene converge into a small box labelled "scene tokens" with tiny
> depth / points / camera icons coming out of it. Label at the top
> "VGGT ❄  (frozen)".
>
> **Right panel, title "What a dynamics model needs":** the same scene
> but now showing motion — ghosted previous pose of the arm and the
> block, arrows indicating motion vectors, and a callout labelled "Δ
> tokens ≈ motion?" with a small r = 0.47 gauge under it, dial sitting
> between yellow and green.
>
> Output: 2400×1200 PNG.

---

### Notes for the operator
- Use **Prompt A** for the paper's hero figure (page 1).
- **Prompt B** is a good alternative system diagram if A ends up too
  busy.
- **Prompt C** is optional, better suited to a talk/slide deck than the
  PDF.
- Save the chosen render as `results/paper_figs/fig0_system_diagram.png`
  and add a line in `PAPER.md` above Section 1 referencing it
  (e.g., `![Figure 0](results/paper_figs/fig0_system_diagram.png)`).
