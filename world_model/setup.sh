#!/usr/bin/env bash
# One-shot environment setup. Safe to re-run (idempotent-ish).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d vggt ]; then
  git clone --depth 1 https://github.com/facebookresearch/vggt.git
fi

if [ ! -d .venv ]; then
  uv venv --python 3.11 .venv
fi

source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e ./vggt

mkdir -p weights data/manifests results/{metrics,plots,qualitative,cache}
echo "Setup OK. Activate with: source .venv/bin/activate"
echo "Next: run scripts/smoke_test.py to download VGGT weights (~5GB) and verify."
