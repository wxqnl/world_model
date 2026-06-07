#!/usr/bin/env bash
cd /home/user01/Minko/newwm/wm3d_v3
export CUDA_VISIBLE_DEVICES=0,1,2,3
exec torchrun --standalone --nproc_per_node=4 -m wm3d_v3.training.train --cfg "${1:-configs/v3_oxe.yaml}" "${@:2}"
