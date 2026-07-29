# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       ffmpeg \
       libglib2.0-0 \
       libgl1 \
       libsm6 \
       libxext6 \
       python3 \
       python3-dev \
       python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/wm3d
COPY environments/scale5b/wheelhouse /wheelhouse
RUN cd /wheelhouse \
    && sha256sum --check WHEELHOUSE.SHA256 \
    && sha256sum --check REQUIREMENTS.SHA256 \
    && /opt/wm3d/bin/python -m pip install \
       --no-index \
       --no-deps \
       --find-links /wheelhouse \
       --requirement /wheelhouse/requirements.lock \
    && /opt/wm3d/bin/python -m pip check

COPY wm3d_v3/__init__.py /tmp/wm3d_receipt/wm3d_v3/__init__.py
COPY wm3d_v3/data/__init__.py /tmp/wm3d_receipt/wm3d_v3/data/__init__.py
COPY wm3d_v3/data/scale5b_contracts.py \
     /tmp/wm3d_receipt/wm3d_v3/data/scale5b_contracts.py
COPY wm3d_v3/training/__init__.py \
     /tmp/wm3d_receipt/wm3d_v3/training/__init__.py
COPY wm3d_v3/training/scale5b_environment.py \
     /tmp/wm3d_receipt/wm3d_v3/training/scale5b_environment.py
COPY scripts/scale5b/verify_environment.py \
     /tmp/wm3d_receipt/scripts/scale5b/verify_environment.py
COPY environments/scale5b/environment_contract.json \
     /opt/wm3d/environment_contract.json

RUN PYTHONPATH=/tmp/wm3d_receipt \
    /opt/wm3d/bin/python \
      /tmp/wm3d_receipt/scripts/scale5b/verify_environment.py \
      --contract /opt/wm3d/environment_contract.json \
      --output /opt/wm3d/environment_receipt.json \
    && rm -rf /tmp/wm3d_receipt /wheelhouse

ENV PATH=/opt/wm3d/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

LABEL org.opencontainers.image.title="WM3D-V7 native 5B training runtime"
CMD ["/bin/bash"]
