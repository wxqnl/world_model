#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

VENV_BIN=${VENV_BIN:-/data/Minko/.venvs/wm3d/bin}
TORCHRUN=${TORCHRUN:-$VENV_BIN/torchrun}
LOG_DIR=${LOG_DIR:-/data/Minko/logs}
MASTER_ADDR=${MASTER_ADDR:-172.27.0.6}
MASTER_PORT=${MASTER_PORT:-29640}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
WORKER_HOSTS=(${WORKER_HOSTS:-root@172.27.0.7 root@172.27.0.4 root@172.27.0.5})
NNODES=$((1 + ${#WORKER_HOSTS[@]}))

NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0.1411}
GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0.1411}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
NCCL_IB_HCA=${NCCL_IB_HCA:-^mlx5_bond_0}
NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
NCCL_DEBUG=${NCCL_DEBUG:-INFO}

mkdir -p "$LOG_DIR"

cat > /tmp/wm3d_dist_smoke_4node.py <<'PY'
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

dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=300))
x = torch.tensor([rank + 1.0], device=device)
dist.all_reduce(x, op=dist.ReduceOp.SUM)
expected = world * (world + 1) / 2
if abs(float(x.item()) - expected) > 1e-3:
    raise RuntimeError(f"bad all_reduce rank={rank}: got {float(x.item())}, expected {expected}")
dist.barrier()
if rank == 0:
    print(
        f"DIST_SMOKE_OK backend={backend} world={world} "
        f"host={socket.gethostname()} sum={float(x.item())}",
        flush=True,
    )
dist.destroy_process_group()
PY

rank=1
for host in "${WORKER_HOSTS[@]}"; do
  scp /tmp/wm3d_dist_smoke_4node.py "$host:/tmp/wm3d_dist_smoke_4node.py" >/dev/null
  ssh "$host" "mkdir -p '$LOG_DIR'"
  ssh "$host" "cd /data/Minko/world_model/wm3d_v3 && (setsid env BACKEND=nccl NCCL_DEBUG='$NCCL_DEBUG' NCCL_IB_DISABLE='$NCCL_IB_DISABLE' NCCL_IB_HCA='$NCCL_IB_HCA' NCCL_NVLS_ENABLE='$NCCL_NVLS_ENABLE' NCCL_SOCKET_IFNAME='$NCCL_SOCKET_IFNAME' GLOO_SOCKET_IFNAME='$GLOO_SOCKET_IFNAME' CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 '$TORCHRUN' --nnodes='$NNODES' --nproc_per_node='$NPROC_PER_NODE' --node_rank='$rank' --master_addr='$MASTER_ADDR' --master_port='$MASTER_PORT' /tmp/wm3d_dist_smoke_4node.py > '$LOG_DIR/dist_smoke_4node_rank${rank}.log' 2>&1 < /dev/null & echo \$! > '$LOG_DIR/dist_smoke_4node_rank${rank}.pid')"
  rank=$((rank + 1))
done

sleep 3
set +e
timeout 360 env \
  BACKEND=nccl \
  NCCL_DEBUG="$NCCL_DEBUG" \
  NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
  NCCL_IB_HCA="$NCCL_IB_HCA" \
  NCCL_NVLS_ENABLE="$NCCL_NVLS_ENABLE" \
  NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME" \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  "$TORCHRUN" \
  --nnodes="$NNODES" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --node_rank=0 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  /tmp/wm3d_dist_smoke_4node.py >"$LOG_DIR/dist_smoke_4node_rank0.log" 2>&1
rc=$?
set -e

echo "rank0_rc=$rc"
for r in $(seq 0 $((NNODES - 1))); do
  echo "RANK${r}_LOG"
  if [[ "$r" == "0" ]]; then
    tail -n 120 "$LOG_DIR/dist_smoke_4node_rank0.log" || true
  else
    host="${WORKER_HOSTS[$((r - 1))]}"
    ssh "$host" "tail -n 120 '$LOG_DIR/dist_smoke_4node_rank${r}.log' || true"
  fi
done

exit "$rc"
