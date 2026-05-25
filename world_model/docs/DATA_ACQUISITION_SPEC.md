# Paper-scale Data Acquisition Spec

**Project:** GeoPhys-WM (geophys-feasibility)
**Date:** 2026-04-29
**Owner-fillable timeline; everything else is fixed.**

This is the data we need to download, organize, and validate so the paper-scale
training runs (`PHASE2_PAPER_PLAN.md`) can begin. Final on-disk size after VGGT
token caching is **~750 GB int16 / ~375 GB int8**. Raw video downloads are
larger (~3–5 TB) but those bytes get consumed by the caching pass and can be
deleted afterward if disk is tight.

---

## 1. What we need (target totals)

| Bucket | Source | Clips | Frames | Raw GB | Cached GB (int16) |
|---|---|---:|---:|---:|---:|
| **Predictor train** | Open X-Embodiment subset | 30 K train + 3 K val | ~3 M | 2 500 | 750 |
| **Generator train (real)** | Same OXE subset above | (reuse) | (reuse) | — | — |
| **Captions** | Claude API VLM captions for clips lacking task strings | 30 K (text, scene) pairs | — | <1 | <1 |
| **Policy benchmark** | LIBERO (public) | 130 tasks | — | ~50 | n/a (kept at frame level) |
| **OOD probe** | Franka-Kitchen + manual curation | 20 clips | ~3 K | 5 | 1 |

Total cached tokens: **≈ 750 GB**. NVMe-backed local disk is required (not NFS — random reads during training are too slow there).

---

## 2. Open X-Embodiment subset (the 750 GB)

Pull the following sub-datasets only. They are the slices the plan validated against and they cover robot manipulation diversity at the right scale.

| Sub-dataset | Episodes (target) | Notes / license |
|---|---:|---|
| `bridge` | 12 K | Stanford Bridge V2; CC-BY 4.0 |
| `fractal20220817_data` | 8 K | Google "Fractal"; Apache 2.0 |
| `kuka` | 4 K | Google KUKA; Apache 2.0 |
| `taco_play` | 3 K | TACO; CC-BY 4.0 |
| `jaco_play` | 1 K | JACO; CC-BY 4.0 |
| `droid_100` (already cached) | 30 (have) → up to 2 K | already in `data/raw/droid_100/` |
| **Total** | **30 K train + 3 K val** | mixed-license but all open |

**Frame budget per clip.** Cap at 128 frames per clip (`cache.max_frames_per_clip`). Episodes longer than 128 frames are truncated at the start by the cache step; episodes shorter than 8 frames are skipped (window length).

**Action-space note.** The five sub-datasets above have different action conventions. We use action data only inside Phase 1 (the predictor); the generator is text+init-frame conditioned and ignores actions. Phase 1 normalizes per-dataset (mean/std), so action-space differences become a soft prior, not a hard incompatibility. **Do not attempt to unify action spaces during download** — the trainer handles it.

### Download method

OXE is hosted as TFDS shards on Google Cloud Storage and as a Hugging Face mirror:

```bash
# Option A (recommended): TFDS via tensorflow-datasets
pip install tensorflow tensorflow-datasets rlds
python -c "
import tensorflow_datasets as tfds
for ds in ['bridge', 'fractal20220817_data', 'kuka', 'taco_play', 'jaco_play']:
    tfds.load(ds, data_dir='/data/oxe', download=True)
"

# Option B (no TF dependency): Hugging Face mirror
huggingface-cli download \
  --repo-type dataset jxu124/OpenX-Embodiment \
  --local-dir /data/oxe_hf --include 'bridge/**' 'fractal/**' 'kuka/**' 'taco_play/**' 'jaco_play/**'
```

**Network.** Each sub-dataset is 200 GB–1.5 TB raw. Plan ~3.5 TB total raw download. At a 1 Gbit/s line, ~8 hours; at 100 Mbit/s, ~3 days. Resume support: `tfds.load` is idempotent; `huggingface-cli download` is also resumable.

---

## 3. On-disk layout (after download)

```
/data/oxe/                        # raw tfds shards, ~3.5 TB
/data/oxe_extracted/<dataset>/episode_<i>/frame_<j>.jpg   # decoded RGB frames
                                  # written by scripts/extract_oxe_frames.py (TODO)

# Already exists for DROID-100:
/home/user01/Simo/geophys-feasibility/data/raw/droid_100/...

# Final cached token shards (one .npz + one .json per clip):
/data/cache/paper_scale/cache_tokens/<clip_id>.npz   # ~25 MB/clip × 30 K = 750 GB
                                  /<clip_id>.json   # metadata sidecar
```

**Manifest:** `data/manifests/paper_scale.json` — one entry per clip, schema identical to the existing `data/manifests/set_a.json` (clip_id, frames list, fps, meta.task, meta.episode_index). Build it after extraction with `scripts/build_paper_scale_manifest.py` (TODO; mirror `scripts/build_set_a_manifest.py`).

**Validation IDs:** `data/manifests/val_ids.json` — JSON array of ~3 K episode indices reserved for val. Pick by hash on `clip_id` to keep the split reproducible.

---

## 4. Captioning (text conditioning)

About 70 % of OXE clips have a `task` string already. The rest need captions:

* **Tool:** Claude Sonnet 4.6 via the Anthropic API.
* **Input:** 4 evenly-spaced frames per clip (≤ 256×256 each, jpeg quality 75).
* **Prompt:** "Describe in one short sentence what the robot is doing. Use the form 'Robot is …'. Do not describe the background."
* **Cost:** ~$0.003/clip × 30 K = **~$100**.
* **Filter:** drop responses missing one of the keywords {robot, hand, arm, object, grip, pick, place, push, pull, stack}.
* **Output:** `data/manifests/paper_scale_captions.json` — `{clip_id: caption}`. The trainer reads this when `meta.task` is empty.

Script (TODO): `scripts/caption_oxe.py` — rate-limited (50 req/min), resumable from a JSON checkpoint.

---

## 5. Token-cache pipeline (download → cached tokens)

Run after download. Re-uses existing `src/phase1/cache.py` logic, scaled across GPUs:

```bash
# Caches one clip per process; parallelize across 4 GPUs to fit in 120 H100-h.
scripts/launch_ddp.sh --gpus 4 -- \
    python -m src.phase1.cache \
        --cfg configs/phase1/paper_scale.yaml --force
```

**Throughput target.** 3 M frames × ~15 ms VGGT forward × (1/4 GPUs) ≈ 3 hours wall-clock per GPU; budget for ~10 hours including I/O.

**Verification.** After caching, run:

```bash
python scripts/verify_token_cache.py --cache_dir /data/cache/paper_scale/cache_tokens
```

(TODO; should check: shard count == manifest count, every shard has the expected keys, sum of bytes within ±5 % of 750 GB, sha256 stamps stored in `_summary.json`.)

**Optional int8 quantization** (halves the on-disk size to ~375 GB). Per-clip linear scale; reverse on load. Trade ~0.5 % token reconstruction RMSE for 50 % disk savings. Decide based on disk pressure, not training quality — at 375 GB the I/O is also faster.

---

## 6. LIBERO (policy benchmark)

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO /opt/libero
cd /opt/libero && pip install -e .
python -c "from libero.libero import benchmark; benchmark.get_benchmark('libero_object').get_task(0)"  # smoke
```

Pin commit hash to `e83a44e` (or whatever current main is when you check it out — record it). 130 tasks, ~50 GB, kept as raw frames + simulator states (no tokenization).

Held-out tasks for paper: indices `[5, 17, 23, 41, 58, 66, 72, 89, 101, 118]` from `libero_object` benchmark. Reserve before training the policy.

---

## 7. OOD probe (manual)

20 clips total:
* **10 from Franka-Kitchen** (`d4rl/franka_kitchen-v1`, hand-picked for variety in object/scene).
* **10 manually filmed** (or pulled from a different OXE sub-dataset not in our train set, e.g. `nyu_rot_dataset_converted_externally_to_rlds`).

Metadata: `data/manifests/ood_probe.json` — same schema as paper-scale manifest, plus `provenance: "franka_kitchen" | "manual" | "ood_oxe"` to enable per-source breakdown.

---

## 8. Storage planning

| What | Size | Where |
|---|---|---|
| OXE raw downloads (deletable after cache) | ~3.5 TB | `/data/oxe/` (NVMe) |
| OXE extracted frames (deletable after cache) | ~1 TB | `/data/oxe_extracted/` |
| **Token cache (paper_scale)** | **750 GB int16 / 375 GB int8** | `/data/cache/paper_scale/` (NVMe) |
| LIBERO | 50 GB | `/opt/libero/` |
| OOD probe | 5 GB | `/data/ood/` |
| W&B local run dirs (training side-output) | ~50 GB / month | `/data/wandb/` |
| **Peak working set (during cache)** | **~5.3 TB** | NVMe required |
| **Steady-state (after cache, raws deleted)** | **~800 GB** | local OK |

If single-NVMe is < 5 TB: do the cache pass per sub-dataset, deleting raw shards as soon as their tokens are written. Plan accommodates this.

---

## 9. Deliverables checklist (for the colleague doing the download)

- [ ] **A.** `/data/oxe/` populated with the 5 sub-datasets above
- [ ] **B.** `/data/oxe_extracted/` with decoded frames per clip
- [ ] **C.** `data/manifests/paper_scale.json` — 30 K train + 3 K val entries
- [ ] **D.** `data/manifests/val_ids.json` — list of ~3 K val episode IDs
- [ ] **E.** `data/manifests/paper_scale_captions.json` — captions for clips without task strings
- [ ] **F.** LIBERO checked out at pinned commit, smoke test passes
- [ ] **G.** `data/manifests/ood_probe.json` — 20 OOD clips
- [ ] **H.** Disk audit: free space ≥ 800 GB after raw deletion (confirms cache fit)

After A–E land, the team running training will execute `src/phase1/cache.py` to produce `/data/cache/paper_scale/cache_tokens/` (item I).

- [ ] **I.** Token cache produced; `scripts/verify_token_cache.py` passes; total size within ±5 % of 750 GB

---

## 10. Timeline (owner-fillable)

| Item | Owner | Start | Done |
|---|---|---|---|
| A. OXE raw download | _____ | _____ | _____ |
| B. Frame extraction | _____ | _____ | _____ |
| C. paper_scale manifest | _____ | _____ | _____ |
| D. val_ids split | _____ | _____ | _____ |
| E. Captioning (Claude API) | _____ | _____ | _____ |
| F. LIBERO setup | _____ | _____ | _____ |
| G. OOD probe curation | _____ | _____ | _____ |
| H. Disk audit | _____ | _____ | _____ |
| I. Token cache (training team) | _____ | _____ | _____ |

---

## 11. Open questions / decisions needed before kicking off

1. **NVMe location.** Is `/data/` already on NVMe, or do we need to mount one? The token cache needs random-read NVMe.
2. **Hugging Face vs. TFDS** for OXE — pick one and stick with it; don't mix mirrors (frame ordering can drift).
3. **Caption model.** Claude Sonnet 4.6 ($0.003/clip) is the default; if budget is tight, GPT-4o-mini ($0.001/clip) is a fallback. Benchmark a 100-clip sample and compare quality before committing the full 30 K.
4. **Action-space alignment.** Confirm the trainer handles per-dataset normalization. (It does — see `compute_action_stats` in `src/phase1/dataset.py:125`. No work needed at acquisition time.)
5. **Int8 quantization decision** — make this call after measuring the actual cache size on the first sub-dataset. If <500 GB, stay int16.

---

_Reference docs:_ `PHASE2_PAPER_PLAN.md` §3 (data plan), §6 (compute), §8 (impl checklist). _Scale-up tier (not in scope for this download):_ `docs/STRETCH_PLAN.md`.
