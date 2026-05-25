# Phase 2 paper-scale kickoff prompt

_Paste (or link) this document into a fresh Claude Code session that will take over the paper-scale Phase 2 work described in `PHASE2_PAPER_PLAN.md`. It is written to give a cold-start agent the context, priorities, and constraints needed to be useful immediately._

---

## Your role

You are picking up a research project mid-flight. The feasibility study is complete (see Section 1 of this doc for what was verified). Your job is to execute the paper-scale Phase 2 plan in `PHASE2_PAPER_PLAN.md`, starting with infrastructure and data (M0 and M1) and working up the milestone ladder. You are not expected to redesign the research — the architecture, hypotheses, and metrics are settled. You are expected to make solid engineering calls about how to scale the code and data pipelines.

## Must-read docs, in order

Read these before doing anything else. Budget ~30 minutes.

1. **`docs/EXPERIMENTS_EXPLAINED.md`** — plain-language project overview. 10 min. Start here even if you have ML background; it sets the vocabulary everyone else in the repo uses.
2. **`PHASE2_PAPER_PLAN.md`** — the concrete plan you're executing. 10 min.
3. **`PHASE2_REPORT.md`** — the feasibility-prototype writeup, including the pixel-space and ablation addenda. 5 min. Pay attention to the "Limitations" and "Findings" sections — they explain what's known, what's not, and what paper-scale should do differently.
4. **`PHASE1_REPORT.md`** — predictor-head writeup, including the 2026-04-21 action-conditioning triage addendum. 5 min. The surprise finding that actions regressed is a load-bearing part of the paper narrative.

Optional background:
- `REPORT.md` — Phase 0 feasibility diagnostic (static-vs-dynamic scenes, token-flow correlation).
- `PAPER.md` — the workshop-style Phase 0 writeup.

## Non-negotiables carried over from the prototype

These are decisions already made. Don't relitigate them.

1. **Couple against `vggt_noact`**, not `vggt`. It was stronger in Phase 1 and the generator produces no actions anyway. Path: `results/phase1/runs/vggt_noact/best.pt`.
2. **Default coupling schedule is `sched20`-style**: pure flow matching for ~50% of total epochs, then turn on `L_sc` and `L_ph`. Prototype showed this is the only Pareto-winning variant on both token and pixel metrics. The paper-scale sweep varies warmup length around this default, not "sched vs no-sched."
3. **VGGT backbone stays frozen.** This project is deliberately adapter-scale. Do not propose fine-tuning VGGT unless a milestone fails and the fallback requires it.
4. **Storage format for cached tokens is int16-viewed-as-bf16** (see `src/phase1/dataset.py::_load_tokens`). Prototype uses this and it works. For paper scale, consider int8 quantization *per clip* (scale + zero-point in a sidecar JSON) if storage is tight, but benchmark loss impact first.

## Known gotchas from the prototype

These will trip you up if you don't know about them.

1. **`vggt/` directory is gitignored** (it holds a vendored VGGT codebase clone). This gitignore rule *also* catches `results/phase1/runs/vggt/` — when committing run artifacts, you'll need `git add -f` for anything under that path. See commit `80d6554` for the pattern.
2. **Shared GPU policy** — this is a shared box. Check `nvidia-smi` before launching anything. GPUs 0, 1, 4, 5 are usually idle; 2, 3, 6, 7 often busy. Use `CUDA_VISIBLE_DEVICES` explicitly — don't rely on the default.
3. **Shared venv** at `/home/user01/Minko/reskip2/.venv/bin/activate`. Don't create a new one unless you have a reason; dependencies are already pinned.
4. **Long runs must survive tmux close.** The user frequently disconnects the tmux after kicking off work. Pattern: `setsid nohup ./run.sh </dev/null >/dev/null 2>&1 & disown`. Master log goes into the run output directory, not stdout. See `run_phase1.sh` and `run_phase2.sh` for reference.
5. **DataLoader `num_workers` with the current `GenerativeWindows` cache-in-memory shard caching is tricky.** At scale you'll want to drop the in-memory cache and load from shards on every `__getitem__`, with `num_workers ≥ 4`. Prototype uses `num_workers=2` and keeps shards in RAM; that won't hold for 30 K clips.
6. **`float(some_tensor)` warning on a `requires_grad=True` tensor**: always `.detach()` before `.item()`. Prototype had this bug; fixed in commit `3523947`.
7. **CLIP text encoder is frozen** and runs once per batch. Don't include its parameters in the optimizer. See `src/phase2/text_encoder.py`.

## First three tasks (do these before anything else)

### Task 1: Audit and scope (no code changes)
**Goal.** Understand where the code is versus where the plan needs it to be.

1. Read the must-read docs.
2. Inventory the existing codebase: what's in `src/phase{1,2}/`, `scripts/`, `configs/`, `tests/`.
3. Write a short audit note: `PHASE2_SCALE_AUDIT.md`. Sections: (a) what in the existing code needs to change for distributed training, (b) what's reusable as-is, (c) what needs to be built from scratch (policy subpackage, real VGGT decoder wiring, caption pipeline).

**Acceptance criterion.** The audit names every file that will need a non-trivial edit to reach M3 (scaled `G_θ` training). The user should be able to read this and understand the scope delta without reading the code.

### Task 2: Reproducible tiny-scale DDP smoke
**Goal.** Prove distributed training works in this venv on this box before scaling data.

1. Write `scripts/phase2_scale/train_generator_ddp.py` — a DDP-wrapped version of the existing trainer.
2. Launch with `torchrun --nproc_per_node=2` on the 30-clip prototype data, flow_coupled variant, 3 epochs, `--smoke` flag.
3. Verify: both GPUs utilized, loss curve matches single-GPU smoke within 5% at equal batches, checkpoint save/load roundtrips.

**Acceptance criterion.** Smoke run completes with gradient-sync working. Log shows both GPUs active. User can rerun with `CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/phase2_scale/train_generator_ddp.py --smoke --variant flow_coupled` and it works.

**Do not proceed to Task 3 until this is clean.** Many days of debugging can be saved by having a known-good DDP setup before scaling.

### Task 3: Data-acquisition dry run
**Goal.** Ground the data plan in reality before committing to 3 M frames.

1. Download a 100-clip slice of Open X-Embodiment (pick one sub-dataset, e.g., `bridge`). Check the license and the action-space definition — actions vary across OXE sub-datasets and matter for downstream policy work.
2. Run Phase 1's existing `src/phase1/cache.py` on the slice. Time it. Measure storage per clip.
3. Run the Claude API caption pipeline on 10 clips. Verify the captions look sensible. Cost and latency measured.
4. Write `DATA_DRY_RUN.md` with the findings.

**Acceptance criterion.** The user can look at your dry-run report and decide whether to commit to the full 30 K download, or pivot to a smaller initial target.

## Open questions that need a human decision (before you go far)

Flag these back to the user before burning significant compute on them.

1. **Which Open X-Embodiment sub-datasets?** Action spaces differ; can't naively pool. Probably `bridge` + `fractal20220817` + continuing with DROID is the right mix, but user should confirm.
2. **Where does the paper submit?** Plan targets ICLR 2027 or CoRL 2026. Deadline determines schedule urgency.
3. **Compute source — shared cluster or cloud rental?** 4000 H100-hours ≈ $12.5K at on-demand prices. This changes the default scale-up strategy (slower on-prem vs parallelizable on cloud).
4. **Phase 1 action-conditioning: resolve now or defer?** The 2026-04-21 addendum names FiLM and spatial-pool as untried. Plan assumes these are tested before M2; confirm with user whether to run them as part of M0 or treat as an independent mini-experiment.
5. **Captioning budget.** $100 for 30 K clips is cheap but confirm budget authority before sending API calls.

## How to communicate progress to the user

- Keep `PHASE2_SCALE_STATUS.md` updated as a live status doc. Pattern: see `PHASE2_STATUS.md`. Append session-log bullets; check off the M0–M9 milestone list as you complete each.
- Long runs: always write a master orchestrator log, never rely on tmux scrollback. See `run_phase2.sh` for the pattern.
- When scheduling wake-ups for multi-hour work: set them at 1/3 and 2/3 of expected wall-clock so failures get caught early. Do not schedule at expected completion — runs drift, cache misses hurt; schedule a little early, check, reschedule.
- Commit and push after each milestone gate passes. Don't accumulate.

## Memory pointers

The user's Claude Code auto-memory already contains:
- `project_geophys_wm.md` — project state summary (updated through Phase 2 prototype)
- `reference_phase1_status.md` — pointers to `PHASE1_STATUS.md` and `PHASE2_STATUS.md`
- `reference_venv.md` — venv path
- `feedback_gpu_policy.md` — GPU policy

Update these as paper-scale work progresses. Especially: add a new reference memory pointing to `PHASE2_PAPER_PLAN.md` and this kickoff doc.

## First message to the user

After completing Task 1 (the audit), your first real message to the user should be: a one-paragraph summary of the audit's findings, an explicit list of the open questions above (Section "Open questions"), and a recommendation of what to do first. Do not start Task 2 until the user has answered the blocking questions.
