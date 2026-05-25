# Phase 1 — live status & restart guide

**Goal.** Train a small predictive head on frozen tokens and evaluate next-token prediction at horizons 1/2/4/8. Three runs:

1. `vggt` — VGGT tokens, action-conditioned
2. `vggt_noact` — VGGT tokens, actions zeroed (ablation)
3. `dinov2` — DINOv2 tokens, action-conditioned (baseline)

All share `configs/phase1/default.yaml`, 30 epochs, batch 16, head ~20 M params.

## Environment

```bash
source /home/user01/Minko/reskip2/.venv/bin/activate
cd /home/user01/Simo/geophys-feasibility
```

GPU policy on this shared box: GPUs 0, 1, 4, 5 are usually idle; 2, 3, 6, 7 often busy with other users' jobs. `nvidia-smi` first, then pick with `CUDA_VISIBLE_DEVICES=N`.

## Checklist (update as you go)

- [x] VGGT token cache (30/30 clips)  →  `results/phase1/cache_tokens/`
- [x] DINOv2 token cache (30 clips)   →  `results/phase1/cache_tokens_dinov2/`
- [x] Train run: `vggt`                →  `results/phase1/runs/vggt/`
- [x] Train run: `vggt_noact`          →  `results/phase1/runs/vggt_noact/`
- [x] Train run: `dinov2`              →  `results/phase1/runs/dinov2/`
- [x] Eval + bootstrap on all three    →  `results/phase1/runs/<run>/eval_summary.json` (inline at end of train)
- [x] Write-up (comparison table, plots) →  `PHASE1_REPORT.md` + `results/phase1/plot_{rollout,train}.png`

## Commands (copy-paste, safe to rerun — resumable where noted)

### 1. DINOv2 cache (resumable: skips already-cached clips)

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.phase1.cache_dinov2 \
    --cfg configs/phase1/default.yaml \
    2>&1 | tee results/phase1/cache_dinov2.log
```

Expected time: 3–5 min. Output: one `.npz` per clip in `results/phase1/cache_tokens_dinov2/`.

### 2. Training (three runs)

Run sequentially on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.phase1.train \
    --cfg configs/phase1/default.yaml \
    --runs vggt vggt_noact dinov2 \
    2>&1 | tee results/phase1/train.log
```

Or in parallel across 3 idle GPUs (faster):

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.phase1.train --runs vggt       2>&1 | tee results/phase1/train_vggt.log &
CUDA_VISIBLE_DEVICES=1 python -m src.phase1.train --runs vggt_noact 2>&1 | tee results/phase1/train_noact.log &
CUDA_VISIBLE_DEVICES=4 python -m src.phase1.train --runs dinov2     2>&1 | tee results/phase1/train_dinov2.log &
wait
```

Expected: 30–45 min per run. Writes `results/phase1/runs/<run>/{ckpt.pt,log.jsonl,summary.json}` and appends to `results/phase1/runs/_summary.json`.

### 3. Evaluation

Eval runs **inline** at the end of each training run (best-ckpt, full k-step rollout with bootstrap CIs). No separate command needed. Each run writes:

- `results/phase1/runs/<run>/best.pt`
- `results/phase1/runs/<run>/train_log.json`
- `results/phase1/runs/<run>/eval_summary.json`
- `results/phase1/runs/<run>/eval_k{1,2,4,8}.csv`

`eval.py` has no `__main__`; to redo eval standalone, call `src.phase1.eval.evaluate` from Python with a loaded ckpt.

## How to pick up if things break

1. `cat PHASE1_STATUS.md` to see which step is last checked off.
2. `nvidia-smi` to confirm GPU availability.
3. `ls results/phase1/cache_tokens_dinov2/` (30 files expected) — resume cache if missing.
4. `ls results/phase1/runs/` — each run dir contains `ckpt.pt` and `summary.json` when done.
5. Tail the logs in `results/phase1/*.log` to see where a run failed.

Known gotchas:
- The first DINOv2 run downloads weights (~350 MB) from HuggingFace via `timm`.
- `src/vggt_wrapper.py` raises a FutureWarning about `torch.cuda.amp.autocast`; harmless.
- `results/phase1/runs/` currently exists but is empty — training has not been run yet.

## Key results reference

When Phase 1 completes, compare against Phase 0 numbers (in `REPORT.md`):
- Token-flow Pearson r (Phase 0, frozen): **0.47** manipulation / **0.52** driving.
- Depth drift (Phase 0, frozen sliding window): **5.5%**.

Phase 1 success = VGGT beats DINOv2 at next-token prediction AND action conditioning reduces loss vs `vggt_noact`.

---

**Last updated:** see bottom `## Session log` section.

## Session log

<!-- Append one dated bullet per session: what ran, what finished, what to resume. -->

- **2026-04-20** — Handover doc created; VGGT cache confirmed complete (30/30). Next: DINOv2 cache + training.
- **2026-04-20** — Launched `run_phase1.sh` (PGID 2035478, detached via setsid/nohup). GPU1: vggt, GPU4: vggt_noact, GPU0: dinov2 cache → dinov2 train. Master log: `results/phase1/run_phase1.log`. Remaining: monitor completion, then write-up.
- **2026-04-20** — All runs complete in ~7.5 min wall-clock. Best val: vggt=0.0189, vggt_noact=0.0171, dinov2=0.1228. Write-up in `PHASE1_REPORT.md`. Headline: VGGT beats DINOv2 by ~7–8× on L2; action conditioning unexpectedly *regressed* (`vggt_noact` slightly better) — flagged for Phase 2 triage.
