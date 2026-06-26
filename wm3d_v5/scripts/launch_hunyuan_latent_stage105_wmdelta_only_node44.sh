#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export PYTHONPATH=/data/Minko/world_model/wm3d_v5

source /data/Minko/.venvs/wm3d/bin/activate

RUN_DIR=${RUN_DIR:-/0604-10T-test/wm3d_v5/results/hunyuan_latent_decoder_v5_1b_oxe_stage102step1250_wmdelta_only_stage105_16gpu_20260618_0930}
INIT_CKPT=${INIT_CKPT:-/0604-10T-test/wm3d_v5/results/hunyuan_latent_decoder_v5_1b_oxe_stage100step2250_actionvelocity_stage102_16gpu_20260618_0435/ckpt/eval_step_00001250.pt}
mkdir -p "${RUN_DIR}/logs"

torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=1 \
  --master_addr=172.27.0.6 \
  --master_port=29609 \
  world_model/wm3d_v5/scripts/train_hunyuan_latent_adapter_stage37.py \
  --wm_cfg /0604-10T-test/wm3d_v5/hunyuan_train/oxe_full_hunyuan_joint_v5_1b_stage2_full103965.yaml \
  --wm_ckpt /0604-10T-test/wm3d_v5/results/hunyuan_joint_v5_1b_oxe_stage25step500_latentscale118_hunyuan_owned_rgb_stage28_16gpu_20260615_1620/ckpt/wm_step_00000650.pt \
  --adapter_init_ckpt "${INIT_CKPT}" \
  --out_dir "${RUN_DIR}" \
  --hunyuan_repo /data/Minko/external/HunyuanVideo \
  --hunyuan_model_base /data/Minko/models/hunyuan_video \
  --vae_precision fp32 \
  --precision bf16 \
  --epochs 4 \
  --max_steps 2000 \
  --batch_size_per_gpu 2 \
  --num_workers 2 \
  --lr 3e-6 \
  --weight_decay 0.01 \
  --warmup_steps 80 \
  --grad_clip 1.0 \
  --latent_mse_weight 0.02 \
  --latent_l1_weight 0.002 \
  --latent_dynamic_l1_weight 1.8 \
  --latent_static_l1_weight 0.55 \
  --latent_delta_l1_weight 1.8 \
  --latent_delta_temporal_l1_weight 14.0 \
  --latent_delta_from_first_l1_weight 28.0 \
  --mask_l1_weight 0.0 \
  --mask_bce_weight 0.0 \
  --mask_area_weight 0.0 \
  --mask_tv_weight 0.0 \
  --decoded_l1_weight 0.02 \
  --decoded_motion_l1_weight 90.0 \
  --decoded_temporal_l1_weight 30.0 \
  --decoded_motion_mag_l1_weight 110.0 \
  --decoded_motion_mag_threshold 0.004 \
  --decoded_from_first_l1_weight 160.0 \
  --decoded_static_l1_weight 65.0 \
  --decoded_static_threshold 0.025 \
  --decoded_raw_motion_l1_weight 700.0 \
  --decoded_raw_temporal_l1_weight 240.0 \
  --decoded_raw_motion_mag_l1_weight 800.0 \
  --decoded_raw_from_first_l1_weight 820.0 \
  --decoded_raw_static_l1_weight 320.0 \
  --decoded_raw_losses_roi \
  --decoded_motion_mask_threshold 0.018 \
  --decoded_motion_mask_spatial_dilate 5 \
  --decoded_motion_mask_temporal_dilate 1 \
  --decoded_motion_mask_floor 0.02 \
  --counterfactual_action_mode none \
  --counterfactual_every 1 \
  --counterfactual_latent_rank_weight 0.0 \
  --counterfactual_latent_separation_weight 0.0 \
  --counterfactual_coeff_separation_weight 0.0 \
  --counterfactual_coeff_wrong_abs_weight 0.0 \
  --counterfactual_coeff_opposite_weight 0.0 \
  --counterfactual_static_l1_weight 0.0 \
  --max_train_windows 0 \
  --max_val_windows 1024 \
  --eval_batches 16 \
  --eval_every_steps 250 \
  --ckpt_every_steps 250 \
  --keep_last_checkpoints 3 \
  --milestone_every_steps 1000 \
  --print_every 25 \
  --seed 105 \
  --hidden 256 \
  --n_blocks 6 \
  --output_mode action_latent_velocity \
  --train_wm_delta_only \
  --mask_motion_threshold 0.012 \
  --mask_motion_softness 0.02 \
  --latent_motion_mask_topk 0.22 \
  --latent_motion_mask_floor 0.0 \
  --mask_bias_init -2.0 \
  --mask_temperature 0.9 \
  --mask_min 0.0 \
  --mask_max 1.0 \
  --adapter_use_motion \
  --adapter_use_temporal_memory \
  --adapter_temporal_memory_heads 4 \
  --adapter_temporal_memory_layers 2 \
  --adapter_temporal_memory_mlp_mult 2.0 \
  --adapter_temporal_memory_gate_init 0.35 \
  --motion_region_threshold 0.08 \
  --motion_region_softness 0.04 \
  --motion_region_power 1.2 \
  --motion_region_dilate 1 \
  --motion_region_temporal_dilate 1 \
  --motion_region_topk 0.22 \
  --motion_region_floor 0.0 \
  --motion_region_prior_weight 0.20 \
  --motion_region_bg_ceiling 0.04 \
  --motion_region_mask_mode multiply \
  --direct_delta_static_center_weight 1.0 \
  --direct_delta_temporal_center_weight 0.2 \
  --direct_delta_spatial_highpass_weight 0.15 \
  --direct_delta_spatial_highpass_kernel 5 \
  --action_velocity_direct_delta_scale 0.45 \
  --action_velocity_scale 1.0 \
  --action_velocity_motion_prior_weight 0.45 \
  --action_velocity_motion_prior_floor 0.05 \
  --action_velocity_static_center_weight 1.0 \
  --action_velocity_static_floor 0.12 \
  --action_velocity_static_mask_source combined_delta \
  --action_velocity_static_mask_topk 0.22 \
  --action_velocity_static_mask_threshold 0.02 \
  --action_velocity_static_mask_softness 0.04 \
  --action_velocity_action_gate_weight 0.50 \
  --action_velocity_action_gate_floor 0.05 \
  --action_velocity_action_gate_power 1.0 \
  --action_velocity_action_gate_normalizer 0.20 \
  --action_basis_residual_scale 0.0 \
  --wm_delta_residual_scale 0.25 \
  --wm_delta_blocks 2 \
  --wm_delta_source feature_delta \
  --mask_target_source dynamic \
  --latent_temporal_delta_scale 4.5
