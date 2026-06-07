#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

export PYTHONPATH=/data/Minko/world_model/wm3d_v3:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}

LOG_DIR=${LOG_DIR:-/data/Minko/logs/scaling_staged_smoke_v1}
mkdir -p "$LOG_DIR" manifests/scaling_smoke results/scaling_smoke_libero_policy_cache_v1

/data/Minko/.venvs/wm3d/bin/python - <<'PY'
from pathlib import Path
from wm3d_v3.data.manifest import read_manifest, write_manifest

src = Path("manifests/oxe_train.jsonl")
out = Path("manifests/scaling_smoke/oxe_12eps_v1.jsonl")
records = [r for r in read_manifest(src) if r.n_frames >= 24][:12]
write_manifest(out, records)
print({"oxe_smoke_manifest": str(out), "episodes": len(records), "frames": sum(r.n_frames for r in records)})

lib_src = Path("results/wm3d_libero_action_policy_lowdimhist_partial4_start_stride4_v1_cache/manifest.jsonl")
lib_out = Path("results/scaling_smoke_libero_policy_cache_v1/manifest_32.jsonl")
with lib_src.open() as f, lib_out.open("w") as g:
    for i, line in enumerate(f):
        if i >= 32:
            break
        g.write(line)
print({"libero_smoke_manifest": str(lib_out), "windows": 32})
PY

run_oxe_stage() {
  local name="$1"
  local cfg="$2"
  local resume="${3:-}"
  local log="$LOG_DIR/${name}.log"
  echo "=== ${name} ==="
  rm -rf "results/${name}"
  if [[ -n "$resume" ]]; then
    /data/Minko/.venvs/wm3d/bin/torchrun --standalone --nproc_per_node=1 \
      -m wm3d_v3.training.train \
      --cfg "$cfg" \
      --resume "$resume" \
      --reset_optim \
      --print_every 1 \
      > "$log" 2>&1
  else
    /data/Minko/.venvs/wm3d/bin/torchrun --standalone --nproc_per_node=1 \
      -m wm3d_v3.training.train \
      --cfg "$cfg" \
      --print_every 1 \
      > "$log" 2>&1
  fi
  tail -n 20 "$log"
}

run_oxe_stage \
  scaling_smoke_stage_a_world_visual_v1 \
  configs/scaling_smoke_stage_a_world_visual_v1.yaml

run_oxe_stage \
  scaling_smoke_stage_b_oxe_joint_v1 \
  configs/scaling_smoke_stage_b_oxe_joint_v1.yaml \
  results/scaling_smoke_stage_a_world_visual_v1/ckpt/best.pt

run_oxe_stage \
  scaling_smoke_stage_c_oxe_policy_v1 \
  configs/scaling_smoke_stage_c_oxe_policy_v1.yaml \
  results/scaling_smoke_stage_b_oxe_joint_v1/ckpt/best.pt

echo "=== scaling_smoke_stage_d_libero_policy_v1 ==="
rm -rf results/scaling_smoke_stage_d_libero_policy_v1
/data/Minko/.venvs/wm3d/bin/python -m wm3d_v3.training.train_libero_action_policy \
  --cfg configs/scaling_smoke_stage_d_libero_policy_v1.yaml \
  --max_steps 5 \
  --print_every 1 \
  > "$LOG_DIR/scaling_smoke_stage_d_libero_policy_v1.log" 2>&1
tail -n 30 "$LOG_DIR/scaling_smoke_stage_d_libero_policy_v1.log"

echo "=== eval_generation_smoke ==="
EVAL_DIR=results/scaling_smoke_eval_generation_v1
rm -rf "$EVAL_DIR"
mkdir -p "$EVAL_DIR"
/data/Minko/.venvs/wm3d/bin/python -m wm3d_v3.eval.run_eval \
  --cfg configs/scaling_smoke_stage_b_oxe_joint_v1.yaml \
  --ckpt results/scaling_smoke_stage_b_oxe_joint_v1/ckpt/best.pt \
  --out "$EVAL_DIR/eval_stage_b_rgb.json" \
  --max_batches 2 \
  --batch_size 1 \
  > "$LOG_DIR/eval_stage_b_rgb.log" 2>&1
/data/Minko/.venvs/wm3d/bin/python -m wm3d_v3.eval.make_demo_gif \
  --cfg configs/scaling_smoke_stage_b_oxe_joint_v1.yaml \
  --ckpt results/scaling_smoke_stage_b_oxe_joint_v1/ckpt/best.pt \
  --out_dir "$EVAL_DIR/demo_gifs" \
  --n_clips 1 \
  > "$LOG_DIR/make_demo_gif.log" 2>&1
/data/Minko/.venvs/wm3d/bin/python -m wm3d_v3.eval.make_hunyuan_latent_demo \
  --cfg configs/scaling_smoke_stage_b_oxe_joint_v1.yaml \
  --ckpt results/scaling_smoke_stage_b_oxe_joint_v1/ckpt/best.pt \
  --out_dir "$EVAL_DIR/hunyuan_latent_demos" \
  --n_clips 1 \
  > "$LOG_DIR/make_hunyuan_latent_demo.log" 2>&1

find results/scaling_smoke_stage_a_world_visual_v1/ckpt \
     results/scaling_smoke_stage_b_oxe_joint_v1/ckpt \
     results/scaling_smoke_stage_c_oxe_policy_v1/ckpt \
     results/scaling_smoke_stage_d_libero_policy_v1/ckpt \
     "$EVAL_DIR" -maxdepth 2 -type f -print | sort
