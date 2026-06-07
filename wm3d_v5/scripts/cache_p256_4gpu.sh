#!/usr/bin/env bash
# Launch cache_oxe_p256.py on 4 GPUs (0,1,2,3) in parallel.
set -e
cd /home/user01/Minko/newwm/wm3d_v3
export PYTHONPATH=/home/user01/Minko/newwm/wm3d_v3
mkdir -p /tmp/wm3d_logs
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$shard nohup python scripts/cache_oxe_p256.py \
    --shard $shard --world 4 --token_grid 16 \
    > /tmp/wm3d_logs/cache_p256_sh${shard}.log 2>&1 &
  echo "launched shard $shard pid=$!"
done
wait
echo "all 4 shards done"
