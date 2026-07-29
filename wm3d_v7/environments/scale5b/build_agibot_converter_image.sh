#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
: "${AGIBOT_CONVERTER_IMAGE_TAG:?Set an immutable internal-registry image tag}"
: "${AGIBOT_CONVERTER_BASE_IMAGE:?Set python:3.10.15-slim-bookworm@sha256:digest}"

if [[ ! "${AGIBOT_CONVERTER_BASE_IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "AGIBOT_CONVERTER_BASE_IMAGE must be digest-pinned" >&2
  exit 2
fi

docker build \
  --file "${SCRIPT_DIR}/Dockerfile.agibot_converter" \
  --build-arg "BASE_IMAGE=${AGIBOT_CONVERTER_BASE_IMAGE}" \
  --label "wm3d.v7.agibot_converter.base_image=${AGIBOT_CONVERTER_BASE_IMAGE}" \
  --tag "${AGIBOT_CONVERTER_IMAGE_TAG}" \
  "${REPO_ROOT}"

docker run --rm "${AGIBOT_CONVERTER_IMAGE_TAG}" \
  /opt/agibot-converter/bin/python \
  /opt/agibot-converter-tools/verify_agibot_converter_environment.py \
  --contract /opt/agibot-converter/environment_contract.json \
  --revision-file /opt/agibot-converter/LEROBOT_REVISION \
  --receipt /opt/agibot-converter/environment_receipt.json

IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${AGIBOT_CONVERTER_IMAGE_TAG}")"
if [[ ! "${IMAGE_ID}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Unexpected image id: ${IMAGE_ID}" >&2
  exit 2
fi
echo "Built ${AGIBOT_CONVERTER_IMAGE_TAG} as ${IMAGE_ID}"
