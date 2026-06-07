#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

VENV_BIN=${VENV_BIN:-/data/Minko/.venvs/wm3d/bin}
TORCHRUN=${TORCHRUN:-$VENV_BIN/torchrun}
LOG_DIR=${LOG_DIR:-/data/Minko/logs}
WORKER_HOST=${WORKER_HOST:-root@172.27.0.7}
MASTER_ADDR=${MASTER_ADDR:-172.27.0.6}
MASTER_PORT=${MASTER_PORT:-29612}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0.1411}
GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0.1411}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
NCCL_IB_HCA=${NCCL_IB_HCA:-^mlx5_bond_0}
NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
NCCL_DEBUG=${NCCL_DEBUG:-INFO}

mkdir -p "$LOG_DIR"

cat > /tmp/wm3d_dist_smoke.py <<'PY'
import datetime
import os
import socket

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
backend = os.environ.get("BACKEND", "nccl")

if backend == "nccl":
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    device = torch.device("cpu")

dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=180))
x = torch.tensor([rank + 1.0], device=device)
dist.all_reduce(x, op=dist.ReduceOp.SUM)
expected = world * (world + 1) / 2
if abs(float(x.item()) - expected) > 1e-3:
    raise RuntimeError(f"bad all_reduce rank={rank}: got {float(x.item())}, expected {expected}")
dist.barrier()
if rank == 0:
    print(f"DIST_SMOKE_OK backend={backend} world={world} host={socket.gethostname()} sum={float(x.item())}", flush=True)
dist.destroy_process_group()
PY

scp /tmp/wm3d_dist_smoke.py "$WORKER_HOST:/tmp/wm3d_dist_smoke.py" >/dev/null

ssh "$WORKER_HOST" <<EOF
cd /data/Minko/world_model/wm3d_v3
mkdir -p "$LOG_DIR"
cat > /tmp/run_wm3d_dist_smoke_rank1.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd /data/Minko/world_model/wm3d_v3
export BACKEND=nccl
export NCCL_DEBUG=$NCCL_DEBUG
export NCCL_IB_DISABLE=$NCCL_IB_DISABLE
export NCCL_IB_HCA=$NCCL_IB_HCA
export NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE
export NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
exec "$TORCHRUN" \\
  --nnodes=2 \\
  --nproc_per_node="$NPROC_PER_NODE" \\
  --node_rank=1 \\
  --master_addr="$MASTER_ADDR" \\
  --master_port="$MASTER_PORT" \\
  /tmp/wm3d_dist_smoke.py
SH
chmod +x /tmp/run_wm3d_dist_smoke_rank1.sh
nohup /tmp/run_wm3d_dist_smoke_rank1.sh >"$LOG_DIR/dist_smoke_2node_rank1.log" 2>&1 &
echo \$! > "$LOG_DIR/dist_smoke_2node_rank1.pid"
EOF

sleep 3
set +e
timeout 240 env \
  BACKEND=nccl \
  NCCL_DEBUG="$NCCL_DEBUG" \
  NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
  NCCL_IB_HCA="$NCCL_IB_HCA" \
  NCCL_NVLS_ENABLE="$NCCL_NVLS_ENABLE" \
  NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME" \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  "$TORCHRUN" \
  --nnodes=2 \
  --nproc_per_node="$NPROC_PER_NODE" \
  --node_rank=0 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  /tmp/wm3d_dist_smoke.py >"$LOG_DIR/dist_smoke_2node_rank0.log" 2>&1
rc=$?
set -e

echo "rank0_rc=$rc"
echo "RANK0_LOG"
tail -n 120 "$LOG_DIR/dist_smoke_2node_rank0.log" || true
echo "RANK1_LOG"
ssh "$WORKER_HOST" "tail -n 120 '$LOG_DIR/dist_smoke_2node_rank1.log' || true"

exit "$rc"
