#!/usr/bin/env bash
cd /home/user01/Minko/newwm/wm3d_v3
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/home/user01/Minko/newwm/wm3d_v3
exec torchrun --standalone --nproc_per_node=2 -m wm3d_v3.training.train_pixel_only \
  --cfg "${1:-configs/v3_5_oxe.yaml}" \
  --v3_ckpt "${2:-/home/user01/Minko/newwm/results/wm3d_v3/ckpt/best.pt}" \
  "${@:3}"
