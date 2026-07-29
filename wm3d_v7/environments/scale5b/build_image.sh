#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
: "${IMAGE_TAG:?Set IMAGE_TAG to the immutable internal-registry tag}"
: "${BASE_IMAGE:?Set BASE_IMAGE to a CUDA 12.8.1 Ubuntu 22.04 image@sha256:digest}"

if [[ ! "${BASE_IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "BASE_IMAGE must be digest-pinned, not a mutable tag: ${BASE_IMAGE}" >&2
  exit 2
fi
if [[ ! -f "${SCRIPT_DIR}/wheelhouse/WHEELHOUSE.SHA256" ]]; then
  echo "Build the sealed wheelhouse before the image" >&2
  exit 2
fi

docker build \
  --file "${SCRIPT_DIR}/Dockerfile" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --label "wm3d.v7.base_image=${BASE_IMAGE}" \
  --tag "${IMAGE_TAG}" \
  "${REPO_ROOT}"

docker run --rm "${IMAGE_TAG}" \
  /opt/wm3d/bin/python \
  -c 'import json; print(json.load(open("/opt/wm3d/environment_receipt.json"))["schema"])'

IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${IMAGE_TAG}")"
echo "Built ${IMAGE_TAG} as ${IMAGE_ID}"
