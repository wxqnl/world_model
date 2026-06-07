#!/usr/bin/env bash
set -euo pipefail

# Create an isolated LIBERO runtime. Do not install these dependencies into the
# WM3D training venv; LIBERO's official stack is older and should call WM3D via
# wm3d_v3.policy.http_policy_server.

ROOT="${ROOT:-/data/Minko}"
LIBERO_ROOT="${LIBERO_ROOT:-$ROOT/benchmarks/LIBERO}"
CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin/conda}"
MAMBA_BIN="${MAMBA_BIN:-$ROOT/tools/micromamba/bin/micromamba}"
ENV_PREFIX="${ENV_PREFIX:-$ROOT/.conda-envs/libero-py38}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$ROOT/.micromamba}"
export MAMBA_ROOT_PREFIX

cd "$ROOT"

if [[ -x "$CONDA_BIN" ]]; then
  ENV_RUN=("$CONDA_BIN" run -p "$ENV_PREFIX")
  if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    "$CONDA_BIN" create -y -p "$ENV_PREFIX" -c conda-forge python=3.8.13 pip
  fi
  "$CONDA_BIN" install -y -p "$ENV_PREFIX" -c conda-forge "cmake<4"
elif [[ ! -x "$MAMBA_BIN" ]]; then
  mkdir -p "$ROOT/tools/micromamba"
  curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C "$ROOT/tools/micromamba" --strip-components=1 bin/micromamba
fi

if [[ ! -d "$LIBERO_ROOT/.git" ]]; then
  mkdir -p "$(dirname "$LIBERO_ROOT")"
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_ROOT"
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$MAMBA_BIN" create -y -p "$ENV_PREFIX" -c conda-forge python=3.8.13 pip
fi
if [[ ! -v ENV_RUN ]]; then
  ENV_RUN=("$MAMBA_BIN" run -p "$ENV_PREFIX")
  "$MAMBA_BIN" install -y -p "$ENV_PREFIX" -c conda-forge "cmake<4"
fi

"${ENV_RUN[@]}" python -m pip install -U pip setuptools wheel
"${ENV_RUN[@]}" python -m pip install -r "$LIBERO_ROOT/requirements.txt"
"${ENV_RUN[@]}" python -m pip install \
  torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
  --extra-index-url https://download.pytorch.org/whl/cu113
"${ENV_RUN[@]}" python -m pip install -e "$LIBERO_ROOT"
"${ENV_RUN[@]}" python - <<'PY'
from pathlib import Path
import robosuite

base = Path(robosuite.__path__[0])
for name in ("macros.py", "macros_private.py"):
    path = base / name
    if not path.exists():
        continue
    text = path.read_text()
    text = text.replace("MUJOCO_GPU_RENDERING = True", "MUJOCO_GPU_RENDERING = False")
    path.write_text(text)
    print(f"set MUJOCO_GPU_RENDERING=False in {path}")
PY

cat <<EOF
LIBERO env ready:
  $ENV_PREFIX

Start WM3D policy server in the WM3D venv, then run LIBERO remote runner in this env.
EOF
