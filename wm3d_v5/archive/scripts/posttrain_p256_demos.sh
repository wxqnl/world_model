#!/usr/bin/env bash
set -euo pipefail

TRAIN_SESSION="${TRAIN_SESSION:-wm3d_v3_p256}"
ROOT="${ROOT:-/home/user01/Minko/newwm}"
PROJ="${PROJ:-$ROOT/wm3d_v3}"
CFG="${CFG:-$PROJ/configs/v3_p256_oxe.yaml}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/wm3d_v3_p256}"
CKPT="${CKPT:-$OUT_ROOT/ckpt/best.pt}"
POLL_SECONDS="${POLL_SECONDS:-300}"

SHORT_N_CLIPS="${SHORT_N_CLIPS:-16}"
FULL_TASKS_N="${FULL_TASKS_N:-4}"
FULL_MIN_FRAMES="${FULL_MIN_FRAMES:-32}"
FULL_MAX_FRAMES="${FULL_MAX_FRAMES:-120}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
POST_DIR="${POST_DIR:-$OUT_ROOT/posttrain_$RUN_ID}"
LOG="$POST_DIR/posttrain.log"

mkdir -p "$POST_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] posttrain p256 job started"
echo "train_session=$TRAIN_SESSION"
echo "cfg=$CFG"
echo "ckpt=$CKPT"
echo "post_dir=$POST_DIR"
echo "gpu policy: CUDA_VISIBLE_DEVICES in {0,1,2}; never 4-7"

while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do
  echo "[$(date)] waiting for tmux session $TRAIN_SESSION to finish..."
  sleep "$POLL_SECONDS"
done

while pgrep -f "python.*-m wm3d_v3.training.train --cfg .*v3_p256_oxe.yaml" >/dev/null; do
  echo "[$(date)] training worker still alive; waiting..."
  sleep "$POLL_SECONDS"
done

last_exit="$(grep -o 'TRAIN_EXIT=[0-9]*' "$OUT_ROOT/train.log" 2>/dev/null | tail -n 1 | cut -d= -f2 || true)"
if [[ -n "$last_exit" && "$last_exit" != "0" ]]; then
  echo "[$(date)] training exited with TRAIN_EXIT=$last_exit; aborting posttrain generation"
  exit "$last_exit"
fi
if [[ -z "$last_exit" ]]; then
  echo "[$(date)] WARN: no TRAIN_EXIT line found; continuing with current best checkpoint"
fi

if [[ ! -s "$CKPT" ]]; then
  echo "[$(date)] missing checkpoint: $CKPT"
  exit 1
fi

PYTHONPATH="$PROJ:${PYTHONPATH:-}" CKPT="$CKPT" python - <<'PY'
import os
import torch
ckpt = os.environ["CKPT"]
sd = torch.load(ckpt, map_location="cpu", weights_only=False)
print(f"checkpoint epoch={sd.get('epoch')} val_total={sd.get('val_total')} best_val={sd.get('best_val')}")
PY

CLIP_LIST="$POST_DIR/full_task_clip_ids.txt"
CLIP_META="$POST_DIR/full_task_clip_meta.jsonl"
CFG="$CFG" CLIP_LIST="$CLIP_LIST" CLIP_META="$CLIP_META" \
FULL_TASKS_N="$FULL_TASKS_N" FULL_MIN_FRAMES="$FULL_MIN_FRAMES" FULL_MAX_FRAMES="$FULL_MAX_FRAMES" \
PYTHONPATH="$PROJ:${PYTHONPATH:-}" python - <<'PY'
import json
import os
from pathlib import Path

import torch
import yaml

from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.window_dataset import OXEWindowDataset, WindowConfig, _safe

cfg = yaml.safe_load(Path(os.environ["CFG"]).read_text())
data = cfg["data"]
cache_root = Path(data["cache_root"])
tokens_subdir = data.get("tokens_subdir", "vggt_pooled")
requested = int(os.environ["FULL_TASKS_N"])
min_frames = int(os.environ["FULL_MIN_FRAMES"])
max_frames = int(os.environ["FULL_MAX_FRAMES"])

records = read_manifest(data["manifest"])
wcfg = WindowConfig(
    T=data["T"],
    k=data["k"],
    stride=data["stride"],
    cache_root=cache_root,
    tokens_subdir=tokens_subdir,
)
ds = OXEWindowDataset(records, wcfg)
g = torch.Generator().manual_seed(data["seed"])
perm = torch.randperm(len(ds), generator=g).tolist()
n_val = max(1, int(len(ds) * data["val_frac"]))
val_idx = perm[:n_val]

def has_rollout_cache(rec) -> bool:
    safe = _safe(rec.clip_id)
    return (
        (cache_root / tokens_subdir / f"{safe}.npy").exists()
        and (cache_root / "rgb_256" / f"{safe}.npy").exists()
        and (cache_root / "vggt_geom" / f"{safe}.npz").exists()
    )

def eligible(rec, loose=False) -> bool:
    if rec.n_frames < data["T"] + data["k"]:
        return False
    if not has_rollout_cache(rec):
        return False
    if loose:
        return rec.n_frames <= max(max_frames * 2, data["T"] + data["k"])
    return min_frames <= rec.n_frames <= max_frames

selected = []
seen_clips = set()
seen_datasets = set()

for diverse_pass in (True, False):
    for vi in val_idx:
        ri, _ = ds.index[vi]
        rec = ds.records[ri]
        if rec.clip_id in seen_clips or not eligible(rec):
            continue
        if diverse_pass and rec.dataset in seen_datasets:
            continue
        selected.append(rec)
        seen_clips.add(rec.clip_id)
        seen_datasets.add(rec.dataset)
        if len(selected) >= requested:
            break
    if len(selected) >= requested:
        break

if len(selected) < requested:
    for vi in val_idx:
        ri, _ = ds.index[vi]
        rec = ds.records[ri]
        if rec.clip_id in seen_clips or not eligible(rec, loose=True):
            continue
        selected.append(rec)
        seen_clips.add(rec.clip_id)
        if len(selected) >= requested:
            break

Path(os.environ["CLIP_LIST"]).write_text("\n".join(r.clip_id for r in selected) + ("\n" if selected else ""))
with Path(os.environ["CLIP_META"]).open("w") as f:
    for rec in selected:
        f.write(json.dumps({
            "clip_id": rec.clip_id,
            "dataset": rec.dataset,
            "n_frames": rec.n_frames,
            "task_text": rec.task_text,
        }, ensure_ascii=False) + "\n")
print(f"selected {len(selected)} full-task clips")
for rec in selected:
    print(f"  {rec.dataset} {rec.n_frames:4d} {rec.clip_id} :: {rec.task_text}")
PY

mapfile -t FULL_CLIPS < "$CLIP_LIST"
if [[ "${#FULL_CLIPS[@]}" -eq 0 ]]; then
  echo "[$(date)] no full-task clips selected; aborting"
  exit 1
fi

pids=()
names=()

launch() {
  local name="$1"
  shift
  local logfile="$POST_DIR/${name}.log"
  echo "[$(date)] launching $name; log=$logfile"
  (
    cd "$PROJ"
    "$@"
  ) >"$logfile" 2>&1 &
  pids+=("$!")
  names+=("$name:$logfile")
}

launch short_demo \
  env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PROJ:${PYTHONPATH:-}" \
  python -m wm3d_v3.eval.make_demo_gif \
    --cfg "$CFG" \
    --ckpt "$CKPT" \
    --out_dir "$POST_DIR/demo_gifs" \
    --n_clips "$SHORT_N_CLIPS"

launch full_task_rollout \
  env CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PROJ:${PYTHONPATH:-}" \
  python -m wm3d_v3.eval.make_long_rollout_gif \
    --cfg "$CFG" \
    --ckpt "$CKPT" \
    --out_dir "$POST_DIR/full_task_gifs" \
    --clip_ids "${FULL_CLIPS[@]}" \
    --full \
    --include_context

launch eval \
  env CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$PROJ:${PYTHONPATH:-}" \
  python -m wm3d_v3.eval.run_eval \
    --cfg "$CFG" \
    --ckpt "$CKPT" \
    --out "$POST_DIR/eval_final.json" \
    --max_batches "$EVAL_MAX_BATCHES"

failed=0
for i in "${!pids[@]}"; do
  IFS=: read -r name logfile <<<"${names[$i]}"
  if wait "${pids[$i]}"; then
    echo "[$(date)] finished $name"
  else
    status=$?
    echo "[$(date)] FAILED $name status=$status; see $logfile"
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

PYTHONPATH="$PROJ:${PYTHONPATH:-}" POST_DIR="$POST_DIR" python - <<'PY'
import json
import os
from pathlib import Path

import imageio.v2 as imageio

post = Path(os.environ["POST_DIR"])
gifs = sorted(post.glob("**/*.gif"))
print(f"generated {len(gifs)} gifs")
for gif in gifs[:40]:
    frames = imageio.mimread(gif)
    shape = frames[0].shape if frames else None
    print(f"  {gif} frames={len(frames)} first_shape={shape}")
report = post / "eval_final.json"
if report.exists():
    metrics = json.loads(report.read_text()).get("metrics", {}).get("ALL", {})
    print("eval ALL:", json.dumps(metrics, indent=2))
PY

echo "[$(date)] posttrain p256 generation complete: $POST_DIR"
