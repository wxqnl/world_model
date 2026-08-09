# V8 Stage0 Causal Dual-View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Replace V8 Stage0 future-leaking VGGT windows with an explicitly causal observed-context/target-only dual-view representation while preserving the complete native-3D and action-policy training recipe.

**Architecture:** Each training window is encoded by two independent VGGT forwards in the same first-observed-camera gauge: a T-frame observed-context forward that is the only source of model inputs, and a T+K-frame target forward whose first T outputs are discarded and whose final K outputs are supervision only. The cache schema, loaders, preflight, and formal configuration all fail closed on this identity; legacy V7/V8 caches remain readable only through their existing configurations.

**Tech Stack:** Python 3.10+, PyTorch, NumPy NPZ caches, VGGT encoder, PCA token codec, pytest, YAML, torchrun/DDP.

## Global Constraints

- Work only in /data/Minko/wm3d_v8_release_worktree on branch codex/v8-stage0-causal-dualview until verification is complete.
- Preserve WM3D native explicit 3D: depth, point, camera pose, confidence, T=16, P=64, D=2048, K=8.
- Context inputs may depend only on the T observed frames; target-forward outputs are supervision_only and never enter s_in, s_wrist, task, action history, or policy state.
- Both forwards use first_observed_camera as the geometry coordinate frame.
- Preserve the full Stage0 action recipe: direct=1.0, flow=0.25, native action no-teacher=0.15, native future no-teacher=0.20, start=0, every=1, flow_use_as_policy=false, grip_owner=delta_composed, factual action conditioning for all five sources.
- Use a fresh run lineage and fresh initialization. Do not resume or warm-start from the old 45k run or the frame-local causal10k prototype.
- Do not overwrite or delete legacy caches, manifests, checkpoints, logs, or results.
- Do not start long formal training as part of this change. Merge only after static tests, a real five-source cache canary, and a bounded training canary pass.

---

## File Map

- wm3d_v3/data/v8_causal_dual_view.py: schema constants, two-forward encoding, array and identity validation.
- wm3d_v3/data/window_dataset.py: OXE dual-view cache loading without changing legacy semantics.
- wm3d_v3/data/v7_compact_dataset.py: RoboCasa compact dual-view window loading without changing legacy semantics.
- wm3d_v3/training/train.py: configuration plumbing for both loaders.
- scripts/cache_robocasa365_v7_compact.py: explicit opt-in dual-view cache production, one atomic NPZ per clip.
- scripts/cache_wm3d_v8_stage0_causal_dual_view_oxe.py: bounded/sharded OXE dual-view cache production from audited RGB/action/task caches.
- scripts/preflight_wm3d_v8_stage0_causal_dual_view.py: fail-closed data and objective preflight.
- configs/wm3d_v8_stage0_causal_dual_view_actionpolicy_canary.yaml: bounded fresh-lineage canary.
- configs/wm3d_v8_stage0_causal_dual_view_actionpolicy_formal.yaml: formal configuration, usable only with sealed full caches.
- scripts/launch_wm3d_v8_stage0_causal_dual_view_canary.sh: resource-checked bounded launcher.
- tests/test_v8_causal_dual_view.py: representation and leakage tests.
- tests/test_v8_causal_dual_view_datasets.py: loader schema/identity/slicing tests.
- tests/test_v8_stage0_causal_dual_view_preflight.py: resolved objective and preflight tests.
- docs/v8_stage0_causal_dual_view.md: operator-facing cache, verification, and launch guide.

### Task 1: Representation Contract and Leakage-Proof Encoder

**Files:**
- Create: wm3d_v3/data/v8_causal_dual_view.py
- Create: tests/test_v8_causal_dual_view.py

**Interfaces:**
- Produces: CAUSAL_DUAL_VIEW_SCHEMA, CAUSAL_DUAL_VIEW_REPRESENTATION, encode_causal_dual_view(frames, encoder, codec, T, k), validate_causal_dual_view_archive(npz, T, k, paired_views).
- Consumes: an encoder callable returning pooled tokens and explicit geometry; a codec exposing encode(tokens).

- [ ] **Step 1: Write failing identity and leakage tests**

    def test_context_is_invariant_to_future_pixels():
        first = np.zeros((16, 8, 8, 3), dtype=np.uint8)
        future_a = np.zeros((8, 8, 8, 3), dtype=np.uint8)
        future_b = np.full((8, 8, 8, 3), 255, dtype=np.uint8)
        out_a = encode_causal_dual_view(
            np.concatenate([first, future_a]), FakeMixingEncoder(), IdentityCodec(), 16, 8
        )
        out_b = encode_causal_dual_view(
            np.concatenate([first, future_b]), FakeMixingEncoder(), IdentityCodec(), 16, 8
        )
        np.testing.assert_array_equal(out_a["context_codes"], out_b["context_codes"])
        assert not np.array_equal(out_a["future_codes"], out_b["future_codes"])

    def test_contract_rejects_legacy_schema(tmp_path):
        with np.load(write_legacy_npz(tmp_path), allow_pickle=False) as archive:
            with pytest.raises(ValueError, match="causal dual-view schema"):
                validate_causal_dual_view_archive(archive, T=16, k=8, paired_views=False)

- [ ] **Step 2: Run the focused test and confirm it fails because the module is absent**

    PYTHONPATH=. /root/miniconda3/envs/starvla/bin/python -m pytest tests/test_v8_causal_dual_view.py -q

Expected: collection error for wm3d_v3.data.v8_causal_dual_view.

- [ ] **Step 3: Implement the minimal two-forward contract**

    CAUSAL_DUAL_VIEW_SCHEMA = "wm3d_v8_stage0_causal_dual_view_v1"
    CAUSAL_DUAL_VIEW_REPRESENTATION = "wm3d_v8_vggt_observed_context_target_split_v1"
    GEOMETRY_COORDINATE_FRAME = "first_observed_camera"

    def encode_causal_dual_view(frames, encoder, codec, T=16, k=8):
        if len(frames) != T + k:
            raise ValueError(f"expected {T + k} frames, got {len(frames)}")
        context = encoder(frames[:T], reference_frame=0)
        target = encoder(frames, reference_frame=0)
        return {
            "context_codes": codec.encode(context["pooled"])["codes"],
            "future_codes": codec.encode(target["pooled"][T:T + k])["codes"],
            "future_depth_patch": target["depth_patch"][T:T + k],
            "future_point_patch": target["point_patch"][T:T + k],
            "future_pose_enc": target["pose_enc"][T:T + k],
        }

The implementation must validate finite values, exact T/K/P/D-compatible shapes, context_future_leakage=false, target_usage=supervision_only, and geometry_coordinate_frame=first_observed_camera. It must never concatenate the target forward back into context output.

- [ ] **Step 4: Run focused tests**

    PYTHONPATH=. /root/miniconda3/envs/starvla/bin/python -m pytest tests/test_v8_causal_dual_view.py -q

Expected: all tests pass.

- [ ] **Step 5: Commit**

    git add wm3d_v3/data/v8_causal_dual_view.py tests/test_v8_causal_dual_view.py
    git commit -m "feat(v8): add causal dual-view cache contract"

### Task 2: OXE Loader Integration

**Files:**
- Modify: wm3d_v3/data/window_dataset.py
- Modify: wm3d_v3/training/train.py
- Create: tests/test_v8_causal_dual_view_datasets.py

**Interfaces:**
- Consumes: validate_causal_dual_view_archive(npz, T, k, paired_views).
- Produces: WindowConfig.causal_dual_view_required and WindowConfig.causal_dual_view_representation; samples whose s_in comes only from context_codes and whose s_tgt/depth/point/pose come only from future arrays.

- [ ] **Step 1: Add failing OXE loader tests**

Create a synthetic manifest, RGB/actions/task caches, and one dual-view NPZ. Assert:

    sample = dataset[0]
    np.testing.assert_array_equal(sample["s_in"].numpy(), expected_context)
    np.testing.assert_array_equal(sample["s_tgt"].numpy(), expected_future)
    assert not np.shares_memory(sample["s_in"].numpy(), expected_future)

Also assert that legacy pooled-only archives, representation mismatches, target_usage other than supervision_only, or context_future_leakage=true raise ValueError before a sample is returned.

- [ ] **Step 2: Run focused loader test and confirm failure**

    PYTHONPATH=. /root/miniconda3/envs/starvla/bin/python -m pytest tests/test_v8_causal_dual_view_datasets.py -k oxe -q

Expected: failure because WindowConfig has no causal dual-view fields.

- [ ] **Step 3: Add explicit opt-in fields and loader branch**

Add fields with legacy-safe defaults:

    causal_dual_view_required: bool = False
    causal_dual_view_representation: str | None = None

When enabled, load context_codes/context_scale and future_codes/future_scale, validate the archive contract and exact clip/start identity, and assign only context to s_in. Do not fall back to pooled or vggt_pooled.

- [ ] **Step 4: Thread fields through _window_config and _build_mixed_oxe_source**

Both construction paths must propagate the exact YAML values. A missing representation while required is true must fail during dataset construction.

- [ ] **Step 5: Run focused and legacy tests**

    PYTHONPATH=. /root/miniconda3/envs/starvla/bin/python -m pytest tests/test_v8_causal_dual_view_datasets.py -k oxe -q
    PYTHONPATH=. /root/miniconda3/envs/starvla/bin/python -m pytest tests -q

Expected: focused tests and the existing suite pass without changing legacy sample semantics.

- [ ] **Step 6: Commit**

    git add wm3d_v3/data/window_dataset.py wm3d_v3/training/train.py tests/test_v8_causal_dual_view_datasets.py
    git commit -m "feat(v8): load causal dual-view OXE windows"

### Task 3: RoboCasa Compact Loader Integration

**Files:**
- Modify: wm3d_v3/data/v7_compact_dataset.py
- Modify: wm3d_v3/training/train.py
- Modify: tests/test_v8_causal_dual_view_datasets.py
- Modify: tests/test_v7_compact_dataset.py

**Interfaces:**
- Consumes: a per-clip archive containing window_starts [W], anchor_context_codes [W,T,P,C], wrist_context_codes [W,T,P,C], anchor_future_codes [W,K,P,C], and explicit future geometry [W,K,...].
- Produces: V7CompactDatasetConfig.causal_dual_view_required and exact start-to-window lookup.

- [ ] **Step 1: Add failing compact loader tests**

Use a two-window synthetic archive. Assert that start=0 and start=2 select different W entries, that wrist dropout affects only context, and that future arrays cannot influence s_in. Assert duplicate window_starts, absent starts, wrong clip identity, and old schema all fail closed when causal_dual_view_required=true.

- [ ] **Step 2: Run focused test and confirm failure**

    PYTHONPATH=. /root/miniconda3/envs/starvla/bin/python -m pytest tests/test_v8_causal_dual_view_datasets.py -k compact -q

- [ ] **Step 3: Implement the compact dual-view branch**

Build the dataset index from row.window_starts, verify it equals the archive window_starts set, map each requested start to exactly one W index, decode context/future codes with their matching scales, and retain existing action normalization, physical signed actions, close01 grip target, task embedding, RGB sidecar, and view-mask behavior.

- [ ] **Step 4: Thread configuration through v7_mixed and v7_compact builders**

Use explicit keys compact_causal_dual_view_required and compact_causal_dual_view_representation. Legacy configs keep false and remain byte-for-byte semantically unchanged.

- [ ] **Step 5: Run focused and regression tests**

    PYTHONPATH=. /root/miniconda3/envs/starvla/bin/python -m pytest tests/test_v8_causal_dual_view_datasets.py tests/test_v7_compact_dataset.py -q

- [ ] **Step 6: Commit**

    git add wm3d_v3/data/v7_compact_dataset.py wm3d_v3/training/train.py tests/test_v8_causal_dual_view_datasets.py tests/test_v7_compact_dataset.py
    git commit -m "feat(v8): load causal dual-view RoboCasa clips"

### Task 4: Cache Producers

**Files:**
- Modify: scripts/cache_robocasa365_v7_compact.py
- Create: scripts/cache_wm3d_v8_stage0_causal_dual_view_oxe.py
- Create: tests/test_v8_causal_dual_view_cache_builders.py

**Interfaces:**
- Consumes: encode_causal_dual_view and audited input manifests/caches.
- Produces: atomic NPZ files plus JSONL indices with content hashes, exact selection identity, source, split, clip/start, T/K/P/D, and causal contract fields.

- [ ] **Step 1: Add failing producer tests with a future-mixing fake encoder**

For RoboCasa, assert one clip produces one archive with a W dimension and exact sorted window_starts. For OXE, assert each selected window is atomic and the index has no missing, extra, or duplicate identity. Mutating only future RGB must leave every context code byte-identical.

- [ ] **Step 2: Run producer tests and confirm failure**

    PYTHONPATH=. /root/miniconda3/envs/starvla/bin/python -m pytest tests/test_v8_causal_dual_view_cache_builders.py -q

- [ ] **Step 3: Add explicit RoboCasa opt-in mode**

Add --causal-dual-view without changing the default legacy cache. In this mode, decode each selected clip once, enumerate valid starts with WINDOW_STRIDE=2, run the anchor two-forward contract per window, run the observed-only wrist forward for context, and write one atomic NPZ per clip. Record input/selection/config/output SHA256 values and reject an existing non-identical destination.

- [ ] **Step 4: Implement the OXE producer**

Read the sealed OXE manifest and existing rgb_256, canonical action, task embedding, and identity caches. Partition by stable hash across shard-id/num-shards, write via temporary file plus os.replace, and write an atomic shard index/commit report. Existing non-identical files are conflicts; identical files are accepted without overwrite.

- [ ] **Step 5: Run producer tests**

    PYTHONPATH=. /root/miniconda3/envs/starvla/bin/python -m pytest tests/test_v8_causal_dual_view_cache_builders.py -q

- [ ] **Step 6: Commit**

    git add scripts/cache_robocasa365_v7_compact.py scripts/cache_wm3d_v8_stage0_causal_dual_view_oxe.py tests/test_v8_causal_dual_view_cache_builders.py
    git commit -m "feat(v8): build causal dual-view Stage0 caches"

### Task 5: Formal Objective Preflight and Configurations

**Files:**
- Create: scripts/preflight_wm3d_v8_stage0_causal_dual_view.py
- Create: configs/wm3d_v8_stage0_causal_dual_view_actionpolicy_canary.yaml
- Create: configs/wm3d_v8_stage0_causal_dual_view_actionpolicy_formal.yaml
- Create: tests/test_v8_stage0_causal_dual_view_preflight.py

**Interfaces:**
- Consumes: the new cache indices and existing action-policy preflight.
- Produces: a JSON receipt with pass, errors, warnings, resolved_config_sha256, cache contract hashes, five-source coverage, action objective values, and fresh run_lineage.

- [ ] **Step 1: Add failing resolved-config tests**

Resolve each new YAML and assert all action values from Global Constraints exactly. Assert the formal and canary run names are new, no resume path exists, stage_transition is fresh_init, all five source weights are 35/15/10/20/20, and every cache source requires the new schema.

- [ ] **Step 2: Run focused test and confirm failure**

    PYTHONPATH=. /root/miniconda3/envs/starvla/bin/python -m pytest tests/test_v8_stage0_causal_dual_view_preflight.py -q

- [ ] **Step 3: Implement the preflight**

Reuse existing action-policy checks, then add full index scans for schema/representation/identity/shape/finiteness, content-addressed receipts, source coverage, train/val disjointness, and context_future_leakage=false. Any legacy cache or missing seal is an error, never a warning.

- [ ] **Step 4: Add canary and formal configs**

Inherit the existing formal architecture and objective. Override only cache paths, explicit contract gates, fresh lineage/output/log paths, and bounded max_steps for canary. The formal config points to full sealed caches and remains fail-closed until those paths exist.

- [ ] **Step 5: Run tests and dry-run preflight against synthetic fixtures**

    PYTHONPATH=. /root/miniconda3/envs/starvla/bin/python -m pytest tests/test_v8_stage0_causal_dual_view_preflight.py -q

- [ ] **Step 6: Commit**

    git add scripts/preflight_wm3d_v8_stage0_causal_dual_view.py configs/wm3d_v8_stage0_causal_dual_view_actionpolicy_canary.yaml configs/wm3d_v8_stage0_causal_dual_view_actionpolicy_formal.yaml tests/test_v8_stage0_causal_dual_view_preflight.py
    git commit -m "feat(v8): gate causal Stage0 configuration"

### Task 6: Real Five-Source Cache Canary

**Files:**
- Create at runtime outside Git: runtime/v8_stage0_causal_dual_view_canary/
- Create at runtime outside Git: logs/v8_stage0_causal_dual_view_canary/

**Interfaces:**
- Consumes: one audited bounded selection from DROID, Bridge, RoboCasa atomic, composite, and MG.
- Produces: sealed cache/index/validation receipts and context-invariance evidence.

- [ ] **Step 1: Read-only resource and input audit**

On node43/node44/node41, inspect GPU compute processes, ECC, /data free, required VGGT weights, RGB/action/task inputs, and source manifests. Select only idle GPUs and never stop other processes.

- [ ] **Step 2: Build a deterministic bounded selection**

Select at least two train and one val window per source where available. Write the selection hash and exact source counts before encoding.

- [ ] **Step 3: Run cache producers**

Use distinct canary output roots. Run RoboCasa and OXE cache commands with max-clips/max-windows bounds and explicit shard identity. Do not write into formal cache roots.

- [ ] **Step 4: Validate and seal**

Verify every NPZ SHA, schema, identity, shapes, finiteness, no duplicates/missing/extra, source coverage, and index-content hashes. Re-encode one sample per source after mutating only future RGB and prove context codes are byte-identical while at least one future target changes.

- [ ] **Step 5: Save the exact command transcript and receipt hashes**

Record environment, model revision, codec hash, input selection hash, output index hash, and validation result under logs/v8_stage0_causal_dual_view_canary.

### Task 7: Bounded Fresh-Initialization Training Canary

**Files:**
- Create: scripts/launch_wm3d_v8_stage0_causal_dual_view_canary.sh
- Modify: docs/v8_stage0_causal_dual_view.md

**Interfaces:**
- Consumes: passed canary preflight receipt and sealed five-source canary cache.
- Produces: fresh-lineage checkpoints at fixed hard stops and a training health report.

- [ ] **Step 1: Implement a fail-closed launcher**

The launcher checks the exact preflight receipt hash, confirms selected GPUs are idle, verifies ECC=0 and free space, refuses resume/latest, and starts only the bounded canary config.

- [ ] **Step 2: Run 0 to 20 steps**

Verify one launcher and expected direct workers, finite total/RGB/depth/point/fusion/direct/flow/native-no-teacher losses, nonzero finite gradients for every intended trainable prefix, factual action conditioning for each source, and exact mix sampling.

- [ ] **Step 3: Review step 20 checkpoint**

Require a stable numbered ZIP checkpoint with model/optimizer/scheduler/sampler/RNG state and validate it without trusting latest.

- [ ] **Step 4: Resume exactly 20 to 100**

Resume only from the passed numbered step-20 checkpoint in the same lineage and verify model/optimizer/scheduler/sampler/RNG continuity.

- [ ] **Step 5: Hard-stop and review step 100**

Require no OOM/CUDA/NCCL/nonfinite/Traceback/data errors, stable loss windows, expected source mix, explicit 3D targets active, and action objectives active from step 0. Save a hashed canary report.

- [ ] **Step 6: Commit launcher and documentation**

    git add scripts/launch_wm3d_v8_stage0_causal_dual_view_canary.sh docs/v8_stage0_causal_dual_view.md
    git commit -m "docs(v8): document causal Stage0 operation"

### Task 8: Final Verification and Merge

**Files:**
- Modify only if needed: README.md
- Merge target: v8

**Interfaces:**
- Consumes: passed static suite, cache canary receipt, and training canary receipt.
- Produces: reviewed v8 merge with no long training side effect.

- [ ] **Step 1: Run the complete project check**

    PYTHON_BIN=/root/miniconda3/envs/starvla/bin/python ./run_v8.sh check

Expected: all tests pass; no new warnings beyond the existing two baseline warnings.

- [ ] **Step 2: Inspect the complete diff and repository status**

    git diff 568fa4b...HEAD --check
    git status --short
    git log --oneline --decorate 568fa4b..HEAD

Expected: no whitespace errors, no runtime artifacts tracked, and only approved V8 Stage0 files changed.

- [ ] **Step 3: Verify legacy behavior**

Run legacy compact and OXE loader tests with causal mode disabled, resolve the prior formal config, and confirm its values have not changed. Confirm no old cache/checkpoint/result path was modified.

- [ ] **Step 4: Merge into v8**

Switch to v8 only after all preceding evidence passes, merge codex/v8-stage0-causal-dualview with a merge commit, and rerun ./run_v8.sh check on v8.

- [ ] **Step 5: Report exact evidence**

Report branch/merge commit IDs, test counts, five-source cache receipt hashes, canary checkpoint/report hashes, and any intentionally deferred item. Do not start formal long training automatically.
