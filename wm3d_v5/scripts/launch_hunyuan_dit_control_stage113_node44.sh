#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko

export MASTER_ADDR=172.27.0.6
export MASTER_PORT=29713
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export PYTHONPATH=/data/Minko/world_model/wm3d_v5:/data/Minko/external/HunyuanVideo:${PYTHONPATH:-}

RUN_DIR=/0604-10T-test/wm3d_v5/results/hunyuan_dit_control_v5_1b_stage0base_native3d_oxe_true_dit_stage113_16gpu_20260619_1430
mkdir -p "${RUN_DIR}/logs"

{
  echo "launch_time=$(date -Is)"
  echo "host=$(hostname)"
  echo "node_rank=1"
  echo "run_dir=${RUN_DIR}"
  echo "master=${MASTER_ADDR}:${MASTER_PORT}"
  echo "wm_ckpt=/data/Minko/world_model/wm3d_v5/results/wm3d_v5_p64_1b_stage0_native3d_wsd_4node_v2/ckpt/best.pt"
  echo "wm_cfg=/data/Minko/world_model/wm3d_v5/configs/v5_p64_1b_stage0_native3d_hunyuan_rgb_full8ep_2node_v1.yaml"
  echo "true_hunyuan_dit=1"
  echo "small_rgb_decoder_conditioning=0"
  echo "adapter_use_rough=0"
  echo "rgb_features=0"
  echo "path_type=context"
  echo "velocity_target_sign=-1.0"
  echo "latent_motion_mask_source=gt_rgb"
  echo "velocity_dynamic_weight=12 velocity_static_weight=1"
  env | grep -E '^(CUDA_VISIBLE_DEVICES|NCCL_NVLS_ENABLE|PYTORCH_CUDA_ALLOC_CONF|OMP_NUM_THREADS|PYTHONPATH|MASTER_)=' | sort
} | tee "${RUN_DIR}/logs/node44_env.log"

/data/Minko/.venvs/wm3d/bin/torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=1 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  /data/Minko/world_model/wm3d_v5/scripts/train_hunyuan_dit_control_adapter.py \
  --wm_cfg /data/Minko/world_model/wm3d_v5/configs/v5_p64_1b_stage0_native3d_hunyuan_rgb_full8ep_2node_v1.yaml \
  --wm_ckpt /data/Minko/world_model/wm3d_v5/results/wm3d_v5_p64_1b_stage0_native3d_wsd_4node_v2/ckpt/best.pt \
  --out_dir "${RUN_DIR}" \
  --hunyuan_model_base /data/Minko/models/hunyuan_video \
  --epochs 8 \
  --max_steps 16000 \
  --batch_size_per_gpu 1 \
  --num_workers 2 \
  --lr 5e-6 \
  --weight_decay 0.01 \
  --warmup_steps 500 \
  --grad_clip 1.0 \
  --max_train_windows 0 \
  --max_val_windows 2048 \
  --eval_batches 16 \
  --eval_every_steps 200 \
  --ckpt_every_steps 200 \
  --keep_last_checkpoints 3 \
  --milestone_every_steps 1000 \
  --print_every 10 \
  --path_type context \
  --velocity_mse_weight 1.0 \
  --velocity_l1_weight 0.05 \
  --velocity_target_sign -1.0 \
  --latent_motion_mask_source gt_rgb \
  --latent_motion_threshold 0.02 \
  --latent_motion_dilate 1 \
  --velocity_dynamic_weight 12.0 \
  --velocity_static_weight 1.0 \
  --action_rank_weight 0.0 \
  --use_block_action_film \
  --block_action_film_scale 1.0 \
  --block_action_film_hidden 192 \
  --control_scale 1.0 \
  2>&1 | tee -a "${RUN_DIR}/logs/node44_train.log"
