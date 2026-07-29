# H200 cluster runbook

## Supported formal topologies

| Mode | Nodes | GPUs | HSDP mesh | Expected use |
|---|---:|---:|---|---|
| Recommended | 16x8 H200 | 128 | replicate 16 x shard 8 | 3–5 week target |
| Minimum | 8x8 H200 | 64 | replicate 8 x shard 8 | longer 6–9 week run |

The shard dimension is one NVLink-connected node. Gradients replicate across
nodes over InfiniBand. Formal preflight requires H200, at least 135,000 MiB
per GPU, active IB, zero uncorrected ECC, unique GPU UUIDs, adequate shared
memory/file limits, disk budget, and measured all-reduce throughput.

Do not assume H200 halves H100 runtime. Run a 1,000-step scale canary and use
the measured steady-state seconds/step to finalize wall time.

## 1. Build the qualified runtime

Build on x86_64 Linux. Both the CUDA base image and Python wheels are pinned.

```bash
cd /workspace/wm3d_v7
PYTHON_BIN=python3.10 \
  environments/scale5b/build_wheelhouse.sh \
  /workspace/wm3d_v7/environments/scale5b/wheelhouse

export BASE_IMAGE='nvidia/cuda@sha256:<approved-cuda-12.8.1-cudnn-digest>'
export IMAGE_TAG='registry.internal/wm3d/v7-native5b:release-<sha>'
environments/scale5b/build_image.sh
docker push "${IMAGE_TAG}"
```

The image contains `/opt/wm3d`, an environment contract, and an environment
receipt created only after exact package, PyTorch/CUDA/NCCL, FSDP2, and DCP API
checks pass. Production compute nodes must have no package-install or model
download step.

Convert to the site's immutable runtime format if needed (Apptainer SIF,
Enroot squashfs, or a Slurm OCI image), hash the converted artifact, and record
that hash in the handoff manifest. The site launch adapter must expose:

- `/opt/wm3d/bin/python` and `/opt/wm3d/bin/torchrun`;
- the repository read-only;
- the sealed dataset read-only;
- the run output/checkpoint filesystem read-write;
- the per-node log directory read-write;
- all eight H200 devices and host InfiniBand devices.

## 2. Qualify the release

Inside the final container:

```bash
cd /workspace/wm3d_v7
RUN_GPU_SMOKE=1 \
GPU_SMOKE_ROOT=/scratch/qualification/fsdp2_dcp_<release-sha> \
CUDA_VISIBLE_DEVICES=0,1 \
scripts/scale5b/qualify_release.sh
```

This runs static checks, unit tests, the exact parameter-count assertion, and
a two-GPU FSDP2/HSDP + Distributed Checkpoint save/restore test with a
bit-exact forward result after restore. The test output is evidence; do not
reuse its checkpoint path.

Before formal allocation, additionally run an 8-GPU single-node and a full
64/128-GPU preflight. A two-GPU smoke does not qualify InfiniBand or the
production filesystem.

On one otherwise idle eight-H200 node, run the same exact-restore smoke across
all local ranks:

```bash
cd /workspace/wm3d_v7
export PYTHONPATH=/workspace/wm3d_v7
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  /opt/wm3d/bin/torchrun --standalone --nproc-per-node=8 \
  tests/scale5b_fsdp2_smoke.py \
  --root /scratch/qualification/fsdp2_dcp_8gpu_<release-sha>
```

The checkpoint root must be new and must not be reused as training input.

## 3. Seal code

Use a reviewed release commit/tag. Generate the receipt from the exact mounted
tree:

```bash
cd /workspace/wm3d_v7
/opt/wm3d/bin/python scripts/scale5b/seal_code.py \
  --repo-root /workspace/wm3d_v7 \
  --output /releases/wm3d_v7_native5b_<release-sha>/code_receipt.json
```

The receipt records the git commit, scoped status, and SHA/size of every V7 5B
runtime/config/environment file. Sealing now refuses any dirty file in that
scope, and every later verifier also rejects a dirty receipt. Commit the
reviewed release first; there is no formal `allow-dirty` escape hatch.
Training re-hashes this scope before model construction. This is separate
from the container environment receipt and dataset seal.

## 4. Materialize canary and formal configurations

Materialize inside the final runtime so the current environment is checked:

```bash
export REPO=/workspace/wm3d_v7
export DATASET=/datasets/wm3d_v7_native5b_5650h_v1
export RELEASE=/releases/wm3d_v7_native5b_<release-sha>
export RUN_ROOT=/checkpoints/wm3d_v7_native5b_5b_formal_v1
export RUN_LINEAGE=<64-lowercase-hex-operator-lineage>

cd "${REPO}"
/opt/wm3d/bin/python scripts/scale5b/materialize_config.py \
  --template configs/scale5b/wm3d_v7_native5b_h200.template.yaml \
  --dataset-root "${DATASET}" \
  --code-receipt "${RELEASE}/code_receipt.json" \
  --code-root "${REPO}" \
  --environment-contract /opt/wm3d/environment_contract.json \
  --environment-receipt /opt/wm3d/environment_receipt.json \
  --output-root "${RUN_ROOT}" \
  --output-config "${RELEASE}/formal_128h200.yaml" \
  --run-name wm3d_v7_native5b_5b_formal_v1 \
  --run-lineage "${RUN_LINEAGE}" \
  --world-size 128 \
  --shard-degree 8 \
  --global-batch-size 128 \
  --micro-batch-size 1
```

For 64 GPUs keep global batch 128; materialization derives accumulation 2.
Do not hand-edit the materialized YAML. Change the template, re-seal code, and
materialize a new run lineage.

Before the formal run, materialize the checked-in full-model 1,000-step
canary with a distinct output root and lineage:

```bash
export CANARY_ROOT=/checkpoints/wm3d_v7_native5b_canary1k_v1
export CANARY_LINEAGE=<different-64-lowercase-hex-canary-lineage>

/opt/wm3d/bin/python scripts/scale5b/materialize_config.py \
  --template configs/scale5b/wm3d_v7_native5b_h200_canary1k.template.yaml \
  --dataset-root "${DATASET}" \
  --code-receipt "${RELEASE}/code_receipt.json" \
  --code-root "${REPO}" \
  --environment-contract /opt/wm3d/environment_contract.json \
  --environment-receipt /opt/wm3d/environment_receipt.json \
  --output-root "${CANARY_ROOT}" \
  --output-config "${RELEASE}/canary1k_128h200.yaml" \
  --run-name wm3d_v7_native5b_canary1k_v1 \
  --run-lineage "${CANARY_LINEAGE}" \
  --world-size 128 \
  --shard-degree 8 \
  --global-batch-size 128 \
  --micro-batch-size 1
```

The canary uses the exact formal architecture, dataset seal, loss surface and
topology, but a 1,000-step qualification schedule. It is a disposable
measurement run: **never resume the formal 600k run from a canary
checkpoint**. The formal run starts from its own initialization and lineage.

Materialization is itself a release gate: it re-hashes every control file and
payload manifest bound by the dataset seal, verifies the code and current
container receipts, and rejects any unresolved placeholder. It does not merely
copy digests out of a receipt.

Create one final cross-artifact manifest:

```bash
/opt/wm3d/bin/python scripts/scale5b/create_handoff_manifest.py \
  --config "${RELEASE}/formal_128h200.yaml" \
  --repo-root "${REPO}" \
  --dataset-root "${DATASET}" \
  --asset-root /datasets/wm3d_v7_native5b_encoder_assets_v1 \
  --container-artifact /releases/containers/wm3d_v7_native5b_<sha>.sif \
  --output "${RELEASE}/handoff_manifest.json"
```

This performs deep encoder-asset validation and binds the config, code,
environment, dataset, and final container file in one exclusive receipt.

The default `600,000` optimizer steps expose 76.8 million global samples and
about 442 billion state-token presentations. Treat this as a training budget,
not a runtime promise. Use the 1,000-step scale canary to decide whether to
retain the full budget or issue a new, reviewed template before formal start.

## 5. Full-cluster preflight and launch

The provided Slurm script assumes the site container integration has already
placed `/opt/wm3d` in every task. First submit the canary with a fresh log
directory and rendezvous ID:

```bash
export CONFIG="${RELEASE}/canary1k_128h200.yaml"
export REPO_ROOT="${REPO}"
export LOG_ROOT="/logs/wm3d_v7_native5b_canary1k_$(date +%Y%m%dT%H%M%S)"
export RDZV_ID="wm3d-v7-native5b-canary1k-<release-sha>"
export MASTER_PORT=29400

sbatch --nodes=16 scripts/scale5b/sbatch_native5b_h200.sh
```

Wait for the natural step-1000 stop, verify
`checkpoints/step_00001000/COMMITTED.json`, complete the canary checklist, and
sign off measured HBM/throughput/wall-time. Then submit the independently
materialized formal configuration:

```bash
export CONFIG="${RELEASE}/formal_128h200.yaml"
export REPO_ROOT="${REPO}"
export LOG_ROOT="/logs/wm3d_v7_native5b_5b_formal_v1_$(date +%Y%m%dT%H%M%S)"
export RDZV_ID="wm3d-v7-native5b-formal-v1-<release-sha>"
export MASTER_PORT=29400

sbatch --nodes=16 scripts/scale5b/sbatch_native5b_h200.sh
```

The job first launches a world-size preflight on `MASTER_PORT`, publishes one
exclusive report, then launches training on `MASTER_PORT+1`. It uses exactly
one torchrun process per node and eight workers per node. Torch elastic restart
is disabled.

Required site settings:

- `ulimit -l unlimited`;
- `ulimit -n 1048576`;
- `NCCL_IB_DISABLE=0`;
- 64 GB or more `/dev/shm`;
- 10 TB or more free on dataset and output filesystems at launch;
- no other compute process on assigned GPUs;
- a unique log path and rendezvous ID.

The distributed preflight also parses the complete `nvidia-smi topo -m`
matrix on every host. Every off-diagonal GPU pair must be connected through
NVLink/NVSwitch; a PCIe-only pair fails before the 5B model is constructed.
Dataset and output free-space budgets are checked independently on every
rank.

## 6. Checkpoints and exact resume

Only a directory named `checkpoints/step_XXXXXXXX` with `COMMITTED.json`,
`MANIFEST.json`, `metadata.json`, DCP shards, and all per-rank RNG files is a
checkpoint.

The publication order is:

1. all ranks write DCP state into a unique incomplete directory;
2. all ranks write and fsync their RNG state;
3. rank 0 fsyncs every payload, creates metadata and SHA manifests;
4. rank 0 creates `COMMITTED.json`;
5. rank 0 atomically renames the directory and fsyncs its parent.

Resume uses the same materialized config and topology:

```bash
export RESUME_CHECKPOINT="${RUN_ROOT}/checkpoints/step_00020000"
export LOG_ROOT="/logs/wm3d_v7_native5b_resume_00020000_$(date +%Y%m%dT%H%M%S)"
export RDZV_ID="wm3d-v7-native5b-resume-00020000-<release-sha>"
sbatch --nodes=16 scripts/scale5b/sbatch_native5b_h200.sh
```

Resume fails closed on run lineage, semantic config, dataset receipt, world
size, shard degree, file set, size, or SHA drift. Model, optimizer, schedule
(stateless by optimizer step), sampler address, and per-rank Python/NumPy/CPU
and CUDA RNG state are restored. Topology resharding is disabled for formal
resume.

The launcher accepts only an absolute canonical, non-symlink
`step_XXXXXXXX` directory with a regular `COMMITTED.json`. The checkpoint
verifier additionally rejects unsafe manifest paths and propagates local
model/optimizer/RNG restore failures collectively.

## 7. Operations

Monitor:

- optimizer step and steady-state seconds/step;
- source cycle over each 100-step window;
- total/token/RGB/depth/point/camera/action/contact losses;
- finite gradient norm;
- HBM, utilization, thermals, ECC, IB errors, and filesystem free space;
- DCP incomplete trees and committed checkpoint verification time.

Stop on any nonfinite value, data/schema error, NCCL error, uncorrected ECC,
filesystem threshold breach, or unexpected process/topology change. Preserve
evidence. Never delete an incomplete or old numbered checkpoint as an
automatic recovery action.
