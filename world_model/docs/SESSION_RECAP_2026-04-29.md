# Session recap — 2026-04-29

_Continuation of `SESSION_RECAP_2026-04-23.md`. A future Claude Code session
should read both, then jump to the "How to resume" section at the bottom._

## Short version (one paragraph)

Built and smoke-tested the **paper-scale distributed training pipeline** on top
of the existing prototype. Eight new files, two backward-compatible edits to
existing model code. Phase 1 and Phase 2 each got a parallel "DDP" trainer
(bf16 autocast, gradient checkpointing, step-level resume, optional W&B), a
streaming `IterableDataset` that never holds the 750 GB token cache in RAM, and
a paper-scale config (100 M predictor, 300 M generator). A shared-machine-aware
launcher (`scripts/launch_ddp.sh`) reads `nvidia-smi`, refuses to start if not
enough GPUs are free, and pins `CUDA_VISIBLE_DEVICES`. Wrote a one-page data
acquisition spec for the 750 GB token cache (sources, sizes, captioning,
deliverables checklist) intended to hand to a colleague to start the download.
End-to-end smoke tests passed on a single free GPU; checkpoint save+resume
verified. Original prototype scripts are untouched. No paper-scale data has
been pulled yet — that is the user's next step (or the colleague's).

## What got written this session

All on branch `main`, committed in one PR-shaped commit (see git log for hash).

### Code (modified, backward-compatible)

| File | Change |
|---|---|
| `src/phase1/heads.py` | Added `use_checkpoint: bool = False` flag to `PredictiveHead`. When True (and `training`), wraps each TransformerEncoderLayer with `torch.utils.checkpoint(use_reentrant=False)`. |
| `src/phase2/generative.py` | Added `use_checkpoint: bool = False` to `GenerativeConfig`/`FlowMatchingGenerator`. Same activation-checkpointing wiring as the predictor. |

Defaults preserve prototype behavior. The prototype scripts (`src/phase1/train.py`, `scripts/phase2/train_generative.py`) are untouched.

### Code (new)

| File | Purpose |
|---|---|
| `src/phase1/dataset_streaming.py` | `StreamingNextTokenPairs(IterableDataset)`. Per-shard streaming with `(rank, world_size, worker_id)` partitioning. Opens one shard, emits its (context, target) pairs in randomized order, closes it, moves on. Memory footprint: one shard per worker. Reuses `discover_shards` and `_load_tokens` from the prototype dataset. |
| `src/phase2/dataset_streaming.py` | `StreamingGenerativeWindows(IterableDataset)`. Same partitioning; sliding-window over T=8 frames stride=4. Helper `build_streaming_datasets(cfg, rank, world_size)`. |
| `src/phase1/train_ddp.py` | Phase 1 paper-scale trainer. DDP via `torch.distributed`; bf16 autocast (no GradScaler — bf16 has fp32 dynamic range); step-level checkpoints `ckpt_step_{N:08d}.pt` (rank 0 only); `--resume` picks up the latest. Optional W&B (rank 0). Validation runs on rank 0 only against the full val set. |
| `scripts/phase2/train_generative_ddp.py` | Phase 2 paper-scale trainer. Generator + physics under DDP; predictor stays frozen and per-rank (no grads, no DDP wrap). Schedule-based coupling (`coupling.warmup_epochs`). Same AMP/ckpt/W&B treatment. |
| `scripts/launch_ddp.sh` | GPU-aware torchrun wrapper. Reads `nvidia-smi`, picks GPUs with `memory.used < FREE_MEM_MIB` (default 500) and `utilization.gpu < FREE_UTIL_PCT` (default 5). Refuses if fewer than `--gpus` are free. Honors `PREFER_GPUS=` for biased picks. Single-GPU path skips torchrun + nccl init entirely. |

### Configs (new)

| File | Highlights |
|---|---|
| `configs/phase1/paper_scale.yaml` | Predictor: hidden 1024, 12 layers, 16 heads, action_embed_dim 256, context_len 8, `use_checkpoint: true`. Train: bs/rank=32, 12 epochs, lr 3e-4, warmup 2000, ckpt_every 1000, ema 0.999. Streaming dataset enabled. |
| `configs/phase2/paper_scale.yaml` | Generator: hidden 1280, 20 layers, 16 heads, dropout 0, `use_checkpoint: true`. Physics: hidden 768, 6 layers, phi_dim 128 (sweep 64/128/256). Text encoder: CLIP-L/14. Coupling warmup 6 epochs (sched coupling). bs/rank=16 × grad_accum 2 × world. |

### Docs (new)

| File | Purpose |
|---|---|
| `docs/DATA_ACQUISITION_SPEC.md` | One-page spec for the 750 GB data pull. Sub-datasets and target counts (Bridge 12K, Fractal 8K, KUKA 4K, TACO 3K, JACO 1K, DROID-100 grow-out — total 30K train + 3K val). Download method (TFDS or HF mirror). On-disk layout, captioning recipe (Claude API ≈ $100 for 30K clips), token-cache pipeline reference, LIBERO setup, OOD probe, storage planning, deliverables checklist (A–I), owner-fillable timeline. **This is the doc to share with the colleague who runs the download.** |
| `docs/SESSION_RECAP_2026-04-29.md` | This file. |

## Verified end-to-end

All on a single free GPU (the launcher auto-picked GPU 4 then GPU 0 as other users' jobs came and went). Other users' jobs on the box were untouched.

- **Phase 2 DDP smoke.** 22.59 M generator on the 30-clip prototype cache, `--smoke` flag (2 epochs × 3 batches × 4 sample steps). bf16 autocast active. Loss decreased monotonically (5.49 → 5.29). Val ran. Checkpoints written every 2 steps. Exit clean. ~3 s wall-clock.
- **Phase 1 DDP smoke.** Full 1 epoch on the 30-clip prototype cache (719 steps, bs=4). bf16 autocast active, gradient checkpointing on. train 0.499 → val 0.119. ~90 s wall-clock.
- **Resume round-trip.** Rerun Phase 2 smoke with `--resume`. Loaded `ckpt_step_00000006.pt`, started_epoch=2, exited immediately because epochs=2. Confirmed step+model+opt restoration.
- **Launcher refusal.** When 4 GPUs were busy with other users' jobs, `--gpus 8` correctly refused with a clear "REFUSING TO LAUNCH: only 4/8 GPUs free" message and exit 3.
- **Launcher pinning.** With `--gpus 1`, auto-picked from the free pool, set `CUDA_VISIBLE_DEVICES`, ran without nccl init (single-GPU bypass).

Smoke artifacts cleaned from `/tmp/`. GPU released back to the pool (verified via `nvidia-smi`).

## What this does NOT include (still TODO for paper-scale)

The trainers and configs work; what's missing is the data pipeline and a few research-side items called out in `PHASE2_PAPER_PLAN.md` §8. These are explicit gaps to fill, not blockers on the trainers themselves.

1. **OXE downloader** — `scripts/extract_oxe_frames.py`. The data spec describes the contract; no code yet.
2. **Paper-scale manifest builder** — `scripts/build_paper_scale_manifest.py`. Mirror of `scripts/build_set_a_manifest.py` for the 30K-clip pull.
3. **Caption pipeline** — `scripts/caption_oxe.py`. Rate-limited Claude-API client with resumable JSON checkpoint. Spec'd in §4 of the data acquisition doc.
4. **Token cache verifier** — `scripts/verify_token_cache.py`. Checksum + size audit against the manifest.
5. **MMD physics loss.** Replace `_moments_match_loss` in `src/phase2/physics.py:72`. Prototype explicitly flagged it as a placeholder.
6. **Phase 1 action-conditioning triage variants.** FiLM and spatial-pool variants of `PredictiveHead`, to resolve the open regression from `vggt` vs `vggt_noact`. Until done, paper-scale defaults to `use_actions: false` (the prototype winner).
7. **Real VGGT decoder** in `src/phase2/pixel_eval.py` (currently a learned stand-in). Required for the M4 gate.
8. **Policy harness** — new `src/phase3/policy/` subpackage: BC on VGGT tokens, LIBERO eval, three-arm experiment runner. Required for M6/M7.

## Non-obvious decisions baked in this session

- **bf16, no GradScaler.** H100 native bf16 has fp32 dynamic range, so loss scaling is unnecessary. The trainers explicitly `torch.autocast(dtype=bfloat16)` and skip the scaler. Don't reintroduce one.
- **Predictor stays per-rank, not under DDP.** It's frozen — no gradients flow through it. Wrapping it in DDP would just add an unnecessary all-reduce hook. Each rank gets its own copy.
- **Streaming dataset uses `IterableDataset`, not `Dataset` + `DistributedSampler`.** Reason: shards are large (~25 MB) and there are ~30K of them. `DistributedSampler` would need a global index over all sample positions; `IterableDataset` partitions at shard granularity, which keeps memory bounded.
- **Validation runs on rank 0 only.** The val set is held to ~3K episodes and full eval is fast. Distributing it across ranks would require an all-gather of metrics for marginal speedup. Simpler to barrier other ranks while rank 0 evaluates.
- **Single-GPU path bypasses torchrun.** The launcher avoids `nccl` init for `--gpus 1` because it triggers a "device id unknown" warning under nccl when world_size=1. The trainers' `_ddp_setup` falls back to `(rank=0, world=1)` cleanly when `RANK` env is unset.
- **`coupling.warmup_epochs: 6` (= 50% of 12 epochs).** Matches the `sched20` recipe the prototype validated as Pareto-optimal. Don't relitigate without an experiment.
- **`use_checkpoint: true` is mandatory at paper scale, not optional.** A 300 M generator over seq_len 8 × 64 patches will OOM on H100 without it. Configs default it on; smoke tests confirm forward+backward with it on.

## How to resume (cold-start playbook)

1. Read `SESSION_RECAP_2026-04-23.md` (the prior recap) for prototype context. Then read this file. Then `PHASE2_PAPER_PLAN.md` (10 min). Then `docs/DATA_ACQUISITION_SPEC.md` (5 min).

2. **Decide where the data pull stands.** Open `docs/DATA_ACQUISITION_SPEC.md` §9 (deliverables A–I) and tick off what's done. If A–E aren't ready yet, the trainers cannot run paper-scale; you can only smoke them on the existing 30-clip prototype cache (as this session did).

3. **If data is ready**, the launches are:

   ```bash
   # Token cache (after raw download lands)
   scripts/launch_ddp.sh --gpus 4 -- \
       python -m src.phase1.cache --cfg configs/phase1/paper_scale.yaml --force

   # M2 — Phase 1 predictor at paper scale
   scripts/launch_ddp.sh --gpus 8 -- \
       python -m src.phase1.train_ddp \
           --cfg configs/phase1/paper_scale.yaml --run vggt_noact

   # M3 — Phase 2 generator with coupling
   scripts/launch_ddp.sh --gpus 8 -- \
       python scripts/phase2/train_generative_ddp.py \
           --cfg configs/phase2/paper_scale.yaml --variant flow_coupled

   # baseline arm for the M7 three-arm comparison
   scripts/launch_ddp.sh --gpus 8 -- \
       python scripts/phase2/train_generative_ddp.py \
           --cfg configs/phase2/paper_scale.yaml --variant flow_only
   ```

4. **If data is not ready**, the highest-leverage TODOs are: build `scripts/extract_oxe_frames.py`, then `scripts/build_paper_scale_manifest.py`, then start the OXE download in the background while you implement the captioning script.

5. **Shared-box etiquette.** Always go through `scripts/launch_ddp.sh`. If it refuses, _don't override_ — it means another user's job is on the GPU you'd take. `nvidia-smi` first if you want to see who else is on the box. Tunables: `FREE_MEM_MIB`, `FREE_UTIL_PCT`, `PREFER_GPUS`.

6. **W&B is off by default.** Flip `wandb.enabled: true` in the paper-scale config when you start a real run; rank-0-only logging is already wired.

## Repo state at end of session

- Branch: `main`
- All new code + docs committed and pushed.
- Untracked log files in `results/phase1/*.log` and `results/phase2/*.log` are pre-existing prototype-run logs from the prior session — not changes from this session, deliberately not committed (they sit alongside the prototype run dirs and are not part of the paper-scale pipeline).
- 8×H100 box: idle on GPUs 4–7 at end of session; other users' jobs running on 0–3.
