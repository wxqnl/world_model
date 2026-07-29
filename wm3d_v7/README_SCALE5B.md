# WM3D-V7 Native 5B pretraining handoff

This package is the formal large-cluster continuation of **WM3D-V7**. It is
not WM3D-V8, A2, Qwen, Wan, or a VLA retrofit. The world model remains the
owner of explicit future RGB, depth, points, camera state, and robot action.

The default configuration contains exactly **4,956,589,929 parameters** and
uses:

- `T=24` context frames at 5 Hz (4.8 seconds);
- `P=144` native spatial tokens per frame (12x12);
- `K=16` predicted future frames (3.2 seconds);
- external token width `D=2048`, internal state width `2560`;
- a 32-layer state trunk, 24-layer grouped-action trunk, and 10 bridges;
- 30 Hz grouped actions aligned as six substeps per visual frame;
- three RGB views plus masked force, tactile, LiDAR, and proprioception;
- FSDP2/HSDP on 64 or 128 H200 GPUs, BF16 parameters, FP32 reductions;
- transactional Distributed Checkpoint with exact model, optimizer, sampler,
  schedule, topology, and per-rank RNG restoration.

Start with:

1. `docs/scale5b/ARCHITECTURE.md`
2. `docs/scale5b/DATA_PIPELINE.md`
3. `docs/scale5b/CLUSTER_RUNBOOK.md`
4. `docs/scale5b/HANDOFF_CHECKLIST.md`

Formal inputs are immutable and explicit:

- one sealed encoder-asset bundle;
- one sealed dataset receipt;
- one sealed code receipt;
- one qualified environment receipt;
- one materialized training YAML;
- one explicit `step_XXXXXXXX` checkpoint for resume.

There is deliberately no `latest` checkpoint path and no automatic fallback
to a different dataset, topology, model, or software environment.
