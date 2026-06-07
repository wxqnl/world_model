#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

VENV_BIN=${VENV_BIN:-/data/Minko/.venvs/wm3d/bin}
TORCHRUN=${TORCHRUN:-$VENV_BIN/torchrun}
LOG_DIR=${LOG_DIR:-/data/Minko/logs}
WORKER_HOST=${WORKER_HOST:-root@172.27.0.7}
MASTER_ADDR=${MASTER_ADDR:-172.27.0.6}
MASTER_PORT=${MASTER_PORT:-29620}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0.1411}
GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0.1411}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
NCCL_IB_HCA=${NCCL_IB_HCA:-^mlx5_bond_0}
NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

CFG=${CFG:-configs/v3_p64_500m_full_hunyuan_joint_v1_8gpu.yaml}
RESUME=${RESUME:-results/wm3d_v3_p64_140m_p0_action_policy_oxe_fullpolicy_cached_v4_8gpu/ckpt/best.pt}
MANIFEST=${MANIFEST:-manifests/oxe_all_trainable_cached_rgb_geom_v1.jsonl}
CACHE_ROOT=${CACHE_ROOT:-/data/Minko/datasets/cache/wm3d_v3}
HUNYUAN_REPO=${HUNYUAN_REPO:-/data/Minko/external/HunyuanVideo}
HUNYUAN_MODEL=${HUNYUAN_MODEL:-/data/Minko/models/hunyuan_video}
RUN_NAME=${RUN_NAME:-train_oxe_500m_full_hunyuan_joint_v1_2node}

mkdir -p "$LOG_DIR"

CHECK_SCRIPT=${CHECK_SCRIPT:-/tmp/wm3d_2node_train_check.sh}
cat > "$CHECK_SCRIPT" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

missing=0
require_exe() {
  if [[ ! -x "$1" ]]; then
    echo "missing_exe=$1"
    missing=1
  fi
}
require_file() {
  if [[ ! -s "$1" ]]; then
    echo "missing_file=$1"
    missing=1
  fi
}
require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "missing_dir=$1"
    missing=1
  fi
}

require_exe "$TORCHRUN"
require_file "$CFG"
require_file "$RESUME"
require_file "$MANIFEST"
require_dir "$CACHE_ROOT"
require_dir "$HUNYUAN_REPO"
require_dir "$HUNYUAN_MODEL"

if [[ "$missing" -ne 0 ]]; then
  exit 20
fi

"$VENV_BIN/python" - <<'PY'
import torch
from wm3d_v3.training import train
print('node_check_ok torch=%s cuda=%s gpus=%s' % (torch.__version__, torch.version.cuda, torch.cuda.device_count()))
PY
SH
chmod +x "$CHECK_SCRIPT"

echo "checking_rank0"
TORCHRUN="$TORCHRUN" \
VENV_BIN="$VENV_BIN" \
CFG="$CFG" \
RESUME="$RESUME" \
MANIFEST="$MANIFEST" \
CACHE_ROOT="$CACHE_ROOT" \
HUNYUAN_REPO="$HUNYUAN_REPO" \
HUNYUAN_MODEL="$HUNYUAN_MODEL" \
  "$CHECK_SCRIPT" | sed 's/^/[rank0] /'

echo "checking_rank1"
scp "$CHECK_SCRIPT" "$WORKER_HOST:$CHECK_SCRIPT" >/dev/null
ssh "$WORKER_HOST" \
  "TORCHRUN='$TORCHRUN' VENV_BIN='$VENV_BIN' CFG='$CFG' RESUME='$RESUME' MANIFEST='$MANIFEST' CACHE_ROOT='$CACHE_ROOT' HUNYUAN_REPO='$HUNYUAN_REPO' HUNYUAN_MODEL='$HUNYUAN_MODEL' '$CHECK_SCRIPT'" \
  | sed 's/^/[rank1] /'

ssh "$WORKER_HOST" <<EOF
cd /data/Minko/world_model/wm3d_v3
mkdir -p "$LOG_DIR"
cat > /tmp/run_${RUN_NAME}_rank1.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd /data/Minko/world_model/wm3d_v3
export WM3D_DDP_BACKEND=nccl
export NCCL_IB_DISABLE=$NCCL_IB_DISABLE
export NCCL_IB_HCA=$NCCL_IB_HCA
export NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE
export NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
exec "$TORCHRUN" \\
  --nnodes=2 \\
  --nproc_per_node="$NPROC_PER_NODE" \\
  --node_rank=1 \\
  --master_addr="$MASTER_ADDR" \\
  --master_port="$MASTER_PORT" \\
  -m wm3d_v3.training.train \\
  --cfg "$CFG" \\
  --resume "$RESUME" \\
  --reset_optim \\
  --print_every 25
SH
chmod +x /tmp/run_${RUN_NAME}_rank1.sh
nohup /tmp/run_${RUN_NAME}_rank1.sh >"$LOG_DIR/${RUN_NAME}_rank1.log" 2>&1 &
echo \$! > "$LOG_DIR/${RUN_NAME}_rank1.pid"
EOF

WM3D_DDP_BACKEND=nccl \
NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
NCCL_IB_HCA="$NCCL_IB_HCA" \
NCCL_NVLS_ENABLE="$NCCL_NVLS_ENABLE" \
NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME" \
OMP_NUM_THREADS=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
nohup "$TORCHRUN" \
  --nnodes=2 \
  --nproc_per_node="$NPROC_PER_NODE" \
  --node_rank=0 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  -m wm3d_v3.training.train \
  --cfg "$CFG" \
  --resume "$RESUME" \
  --reset_optim \
  --print_every 25 \
  >"$LOG_DIR/${RUN_NAME}_rank0.log" 2>&1 &

echo "$!" > "$LOG_DIR/${RUN_NAME}_rank0.pid"
echo "started_${RUN_NAME}_rank0_pid=$(cat "$LOG_DIR/${RUN_NAME}_rank0.pid")"
echo "started_${RUN_NAME}_rank1_pid=$(ssh "$WORKER_HOST" "cat '$LOG_DIR/${RUN_NAME}_rank1.pid'")"
