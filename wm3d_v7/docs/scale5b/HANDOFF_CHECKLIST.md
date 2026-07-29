# WM3D-V7 native 5B handoff checklist

This checklist is a release gate. A blank or assumed item is a **NO-GO**.

## A. Scope and architecture

- [ ] Release path is `world_model/wm3d_v7`.
- [ ] Dependency guard passes for all sealed Python/YAML/JSON files.
- [ ] No V8, A2, Qwen, Wan, video-generator, or VLA dependency is present.
- [ ] Meta-device count is exactly `4,956,589,929` parameters.
- [ ] `T/P/K/D = 24/144/16/2048`; internal state width is 2560.
- [ ] Explicit RGB, depth, point, confidence, and camera heads are enabled.
- [ ] Grouped native action trunk is enabled from step zero.
- [ ] Future-action no-leak gradient test passes.
- [ ] Missing camera and auxiliary masks pass their unit tests.

## B. Raw data

- [ ] Every raw repository is pinned to an immutable revision.
- [ ] A raw-file size/SHA inventory is archived.
- [ ] License/usage approval exists for every source.
- [ ] `total_frames / fps` is measured per shard; planning hours are replaced.
- [ ] Duplicate versions, simulation repackaging, and cross-source overlap are
      measured before the formal hour total is signed.
- [ ] Train/validation/test splits are episode-level and deterministic.
- [ ] Each physical embodiment has its own action schema.
- [ ] Every camera/action/state/force/tactile/LiDAR column has been inspected
      against actual parquet metadata.
- [ ] Missing-sensor semantics are explicit; no NaN is treated as factual.
- [ ] Contact/gripper supervision is masked only onto discrete action groups.

Record evidence:

| Item | Path / digest / approver |
|---|---|
| Raw inventory | |
| License review | |
| Measured unique hours | |
| Duplicate audit | |
| Source-layout review | |

## C. Offline encoders

- [ ] VGGT source commit is immutable and tracked source is clean.
- [ ] VGGT model revision is immutable.
- [ ] T5 task-model revision is immutable.
- [ ] Portable asset bundle contains no symlink or special file.
- [ ] Deep asset verification passes.
- [ ] Asset receipt SHA is copied into the release manifest.
- [ ] Encoder workers run with network disabled.

## D. Dataset seal

- [ ] Dataset contract compiles with source weights summing to 100.
- [ ] Source scan contains train and validation episodes for every source.
- [ ] Scan measured hours/counts are reviewed.
- [ ] Action statistics use one reviewed global sample budget; increasing
      array shard count does not multiply the merge sample or memory budget.
- [ ] All action-stat partial IDs `[0, N)` exist exactly once and merge.
- [ ] Task bank is 2048-D, finite, and asset-receipt bound.
- [ ] All encoder worker IDs `[0, N)` have committed receipts.
- [ ] No duplicate/missing/extra encoded part exists.
- [ ] All source/split indexes are present.
- [ ] Control verification passes.
- [ ] Deep payload verification passes.
- [ ] Random windows from every validation source decode with finite tensors.
- [ ] Random-window contact masks agree with embodiment discrete groups.
- [ ] Dataset seal SHA and storage snapshot ID are archived.
- [ ] Config materialization re-verifies every control/payload-manifest
      evidence item from the seal without error.

Record evidence:

| Item | Value |
|---|---|
| Contract SHA | |
| Episode-plan SHA | |
| Action-stats SHA | |
| Task-index SHA | |
| Encoder-asset receipt SHA | |
| Dataset-seal SHA | |
| Measured hours | |
| Train/val/test windows | |
| Total payload bytes | |

## E. Runtime and code release

- [ ] CUDA base image is digest-pinned.
- [ ] Wheelhouse SHA checks pass.
- [ ] Container environment receipt passes inside final site runtime.
- [ ] Converted SIF/squashfs/OCI artifact SHA is recorded.
- [ ] Driver supports CUDA 12.8 runtime.
- [ ] Code is a reviewed commit/tag.
- [ ] Code receipt was created from the exact mounted checkout.
- [ ] Scoped git status in the code receipt is reviewed.
- [ ] No training node installs packages or downloads models.
- [ ] `qualify_release.sh` passes without GPU smoke.
- [ ] Two-GPU FSDP2/DCP smoke passes with exact restore difference zero.
- [ ] Eight-GPU node smoke passes.

Record evidence:

| Item | Value |
|---|---|
| Git commit/tag | |
| Code receipt SHA | |
| Environment contract SHA | |
| Environment receipt SHA | |
| Environment fingerprint | |
| Container artifact SHA | |
| Two-GPU smoke receipt/log | |
| Eight-GPU smoke receipt/log | |

## F. Cluster and filesystem

- [ ] Allocation is exactly 8 or 16 nodes with eight H200 GPUs each.
- [ ] Node-local NVLink topology is verified.
- [ ] Automated topology parsing confirms an eight-GPU NVLink/NVSwitch clique
      on every host.
- [ ] Every H200 reports at least 135,000 MiB.
- [ ] Volatile and aggregate uncorrected ECC are zero.
- [ ] No assigned GPU has an external compute process.
- [ ] Active InfiniBand ports and expected 200/400 Gb/s link rate are present.
- [ ] Full-world all-reduce exceeds the accepted threshold.
- [ ] `/dev/shm`, memlock, and file-descriptor limits pass.
- [ ] Dataset and output filesystems each have at least 10 TB free at launch.
- [ ] 200 TB usable capacity (recommended corpus) is provisioned.
- [ ] Repository/dataset are read-only; only logs/output are read-write.
- [ ] Full 64/128-rank `preflight_cluster.py` report is `pass=true`.

## G. Canary

- [ ] Checked-in `wm3d_v7_native5b_h200_canary1k.template.yaml` is
      materialized without hand edits.
- [ ] A separately named 1,000-step canary config/run lineage is materialized.
- [ ] Canary uses the same model, data seal, environment, and topology.
- [ ] Peak HBM has at least 15% safety margin.
- [ ] No OOM, NCCL, data, I/O, ECC, or nonfinite event occurs.
- [ ] Every loss and gradient norm is finite.
- [ ] All six source weights are exact over the 100-step schedule cycle.
- [ ] RGB/depth/point outputs are visually and numerically valid.
- [ ] Every enabled action group receives nonzero finite gradients.
- [ ] Checkpoint save, verify, load, and continued sampler sequence pass.
- [ ] Steady-state seconds/step, input throughput, and projected wall time are
      signed off.
- [ ] Formal training starts from its own initialization/run lineage; no
      canary checkpoint is used as a formal resume parent.

## H. Formal launch

- [ ] Materialized YAML contains no placeholder.
- [ ] Synthetic materialization/tamper regression passes.
- [ ] Run name, output root, and 64-hex run lineage are unique.
- [ ] Training contract SHA verifies.
- [ ] Global batch is exactly the agreed value (default 128).
- [ ] 128 GPUs use accumulation 1; 64 GPUs use accumulation 2.
- [ ] WSD schedule and total-step budget are approved after canary timing.
- [ ] Checkpoint capacity and retention policy are approved.
- [ ] Log directory and rendezvous ID are new.
- [ ] `--max-restarts=0` remains in effect.
- [ ] Operators know that only `step_XXXXXXXX/COMMITTED.json` is authoritative.
- [ ] Operators know never to use a `latest` path.
- [ ] Operators know not to auto-delete incomplete checkpoint evidence.

## I. Resume drill

- [ ] A committed numbered checkpoint is copied/snapshotted safely.
- [ ] Manifest/metadata/commit hashes verify.
- [ ] Resume directory and manifest paths are canonical and contain no symlink
      or path escape.
- [ ] Resume uses the same world size and shard degree.
- [ ] Model and optimizer restore.
- [ ] Schedule resumes from the stored optimizer step.
- [ ] Step-addressed source and sample sequence matches uninterrupted training.
- [ ] Per-rank Python, NumPy, CPU, and CUDA RNG restore.
- [ ] First post-resume metrics are finite and continuous.

## J. Release bundle

Deliver these immutable artifacts to the cluster operator:

1. reviewed V7 source commit/tag;
2. `README_SCALE5B.md` and all `docs/scale5b/` files;
3. container artifact plus SHA;
4. environment contract and receipt;
5. encoder asset bundle and receipt;
6. dataset contract, source layouts, scan receipt, and final dataset seal;
7. code receipt;
8. materialized formal YAML;
9. cluster preflight report;
10. canary logs, metrics, and checkpoint verification report;
11. explicit launch and explicit numbered-resume commands;
12. escalation contacts and the site's non-destructive incident procedure.

Formal training may start only when sections A–H are complete. Resume
qualification additionally requires section I.
