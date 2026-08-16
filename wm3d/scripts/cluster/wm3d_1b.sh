#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export WM3D_CLUSTER_SCALE=1b
exec bash "${ROOT}/scripts/cluster/wm3d_5b.sh" "$@"
