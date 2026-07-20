# WorldArena Context-Pyramid Validation Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a protocol-locked five-video validation diagnostic that compares the current WorldArena renderer with six fixed context-pyramid renderers, scores official image/JEPA/dynamic/smoothness metrics plus GT PSNR, and emits an auditable GO/NO-GO result without touching episodes 40–49.

**Architecture:** A pure renderer/protocol module owns panel selection, the locked grid, rendering, PSNR, and aggregate selection. A separate generation CLI loads the already-trained `formal1000_diverse3` checkpoint once, generates five native factual rollouts from episodes 36–39, and renders baseline plus six candidates from the same native cache. A scoring CLI prepares each variant independently for the pinned WorldArena tools, runs only the requested four official metrics, aggregates the gate, and writes contact sheets after selection.

**Tech Stack:** Python 3.11, NumPy, OpenCV, PyTorch, PyYAML, pytest, the existing WM3D native S1 rollout code, pinned WorldArena `a918b93f`, pinned local JEDi/V-JEPA weights.

## Global Constraints

- Run every project command on `New-H100-3` from `/data/Minko/world_model/wm3d_v7`.
- Use checkpoint `/data/Minko/world_model/wm3d_v7/results/benchmarks/worldarena_bimanual_adapt/formal1000_diverse3/ckpt/step_00001000.pt`.
- Read only episodes 36, 37, 38, and 39; encountering episodes 40–49 in the selected panel is fatal before model loading or video decoding.
- Select lexicographic task indices `[0, 12, 24, 36, 49]` with episodes `[36, 37, 38, 39, 36]`.
- Inference may read only first RGB, instruction, and actions. Future GT is evaluation-only after all seven video sets exist.
- The parameter grid is exactly the Cartesian product of `alpha=[0.50,0.75,1.00]` and ramps `[(0.02,0.08),(0.04,0.12)]`, with sigma `1.0` and native comparison size `64`.
- Select on five-video aggregates only. GO requires PSNR gain at least `0.25 dB`, image quality no lower, and JEPA/dynamic/smoothness each at least `0.97 * baseline`.
- PSNR gains within `0.02 dB` are tied; break ties by smaller alpha and then lexicographic ramp.
- A GO result authorizes review only. Do not generate or rescore test500.
- Do not alter `scripts/eval_worldarena_s1.py`, `scripts/prepare_worldarena_official_data.py`, the completed formal result card, or any existing formal metric output.
- Run metric jobs only on node43 GPUs 0–3; do not use GPUs 4–7 for this diagnostic.
- Write all artifacts below `results/benchmarks/worldarena_bimanual_adapt/context_pyramid_val5/`.

---

## File Map

- Create `scripts/worldarena_context_pyramid_val.py`: pure protocol, renderer, PSNR, metric parsing, and GO/NO-GO selection functions.
- Create `scripts/eval_worldarena_context_pyramid_val5.py`: strict five-record native generation and seven-way rendering CLI.
- Create `scripts/run_worldarena_context_pyramid_val5.py`: official data preparation, metric launch/validation, aggregation, provenance, and contact sheets.
- Create `configs/benchmarks/worldarena_context_pyramid_val5.yaml`: immutable paths, pinned weights, GPU allowlist, and gate constants.
- Create `tests/test_worldarena_context_pyramid_val.py`: unit tests for selection, rendering, PSNR, and gate logic.
- Create `tests/test_worldarena_context_pyramid_val_pipeline.py`: synthetic end-to-end artifact/guard smoke test that loads neither model nor official data.

### Task 1: Pure Protocol and Renderer Core

**Files:**
- Create: `scripts/worldarena_context_pyramid_val.py`
- Create: `tests/test_worldarena_context_pyramid_val.py`

**Interfaces:**
- Consumes: JSONL rows with `id`, `task`, `episode`, `video_file`, `hdf5_file`, and `instruction_file`.
- Produces: `RenderConfig`, `locked_grid()`, `select_locked_panel(rows)`, `variant_name(config)`, `render_baseline(initial_rgb, native_rgb)`, `render_context_pyramid(initial_rgb, native_rgb, config)`, `aligned_video_psnr(pred_path, gt_path)`, and `select_candidate(baseline, candidates)`.

- [ ] **Step 1: Write failing tests for the strict panel and fixed grid**

```python
def fake_rows():
    return [
        {"id": f"{task}:episode{ep}", "task": task, "episode": ep}
        for task in reversed([f"task_{i:02d}" for i in range(50)])
        for ep in range(50)
    ]


def test_locked_panel_is_deterministic_and_val_only():
    panel = select_locked_panel(fake_rows())
    assert [(row["task"], row["episode"]) for row in panel] == [
        ("task_00", 36), ("task_12", 37), ("task_24", 38),
        ("task_36", 39), ("task_49", 36),
    ]


def test_locked_grid_has_exactly_six_global_configs():
    assert [(c.alpha, c.low, c.high, c.sigma, c.native_size) for c in locked_grid()] == [
        (0.50, 0.02, 0.08, 1.0, 64), (0.50, 0.04, 0.12, 1.0, 64),
        (0.75, 0.02, 0.08, 1.0, 64), (0.75, 0.04, 0.12, 1.0, 64),
        (1.00, 0.02, 0.08, 1.0, 64), (1.00, 0.04, 0.12, 1.0, 64),
    ]
```

- [ ] **Step 2: Run the protocol tests and verify RED**

Run: `python -m pytest tests/test_worldarena_context_pyramid_val.py -k 'locked_panel or locked_grid' -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'scripts.worldarena_context_pyramid_val'`.

- [ ] **Step 3: Implement immutable protocol objects and guards**

```python
@dataclass(frozen=True, order=True)
class RenderConfig:
    alpha: float
    low: float
    high: float
    sigma: float = 1.0
    native_size: int = 64

    def __post_init__(self) -> None:
        if self.alpha not in (0.5, 0.75, 1.0):
            raise ProtocolError("alpha is outside the locked grid")
        if (self.low, self.high) not in ((0.02, 0.08), (0.04, 0.12)):
            raise ProtocolError("motion ramp is outside the locked grid")
        if self.sigma != 1.0 or self.native_size != 64:
            raise ProtocolError("sigma/native size must remain 1.0/64")


def locked_grid() -> tuple[RenderConfig, ...]:
    return tuple(
        RenderConfig(alpha, low, high)
        for alpha in (0.5, 0.75, 1.0)
        for low, high in ((0.02, 0.08), (0.04, 0.12))
    )


def select_locked_panel(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_identity = {(str(r["task"]), int(r["episode"])): dict(r) for r in rows}
    tasks = sorted({str(r["task"]) for r in rows})
    if len(tasks) != 50:
        raise ProtocolError(f"expected 50 tasks, got {len(tasks)}")
    chosen = [(tasks[i], ep) for i, ep in zip((0, 12, 24, 36, 49), (36, 37, 38, 39, 36))]
    if any(ep not in (36, 37, 38, 39) for _, ep in chosen):
        raise ProtocolError("selected panel escaped episodes 36-39")
    panel = [by_identity[key] for key in chosen]
    if len({str(r["id"]) for r in panel}) != 5:
        raise ProtocolError("panel must contain five unique ids")
    return panel
```

- [ ] **Step 4: Run the protocol tests and verify GREEN**

Run: `python -m pytest tests/test_worldarena_context_pyramid_val.py -k 'locked_panel or locked_grid' -q`

Expected: `2 passed`.

- [ ] **Step 5: Add failing renderer property tests**

```python
def test_zero_motion_injects_exact_residual_and_full_motion_injects_none():
    initial = np.zeros((64, 64, 3), np.float32)
    initial[::2, ::2] = 1.0
    static = np.repeat(initial[None], 2, axis=0)
    moved = np.repeat((1.0 - initial)[None], 2, axis=0)
    cfg = RenderConfig(1.0, 0.02, 0.08)
    static_out = render_context_pyramid(initial, static, cfg, output_size=(128, 96))
    moved_out = render_context_pyramid(initial, moved, cfg, output_size=(128, 96))
    high = cv2.resize(initial, (128, 96), interpolation=cv2.INTER_CUBIC)
    moved_low = np.stack([
        cv2.resize(frame, (128, 96), interpolation=cv2.INTER_CUBIC) for frame in moved
    ])
    assert np.allclose(static_out, np.repeat(high[None], 2, axis=0), atol=1e-5)
    assert np.allclose(moved_out, moved_low, atol=1e-5)


def test_renderer_is_finite_bounded_and_does_not_mutate_inputs():
    rng = np.random.default_rng(7)
    initial = rng.random((91, 117, 3), dtype=np.float32)
    native = rng.random((3, 3, 73, 85), dtype=np.float32)
    initial_copy, native_copy = initial.copy(), native.copy()
    output = render_context_pyramid(initial, native, RenderConfig(0.75, 0.04, 0.12))
    assert output.shape == (3, 480, 640, 3)
    assert np.isfinite(output).all() and output.min() >= 0 and output.max() <= 1
    assert np.array_equal(initial, initial_copy) and np.array_equal(native, native_copy)
```

- [ ] **Step 6: Run renderer tests and verify RED**

Run: `python -m pytest tests/test_worldarena_context_pyramid_val.py -k renderer -q`

Expected: FAIL because `render_context_pyramid` is not defined.

- [ ] **Step 7: Implement baseline and context-pyramid rendering**

```python
def _thwc(native_rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(native_rgb, dtype=np.float32)
    if value.ndim != 4:
        raise ProtocolError("native RGB must be rank four")
    if value.shape[1] == 3:
        value = np.moveaxis(value, 1, -1)
    if value.shape[-1] != 3 or not np.isfinite(value).all():
        raise ProtocolError("native RGB must be finite T,H,W,3 or T,3,H,W")
    return value


def render_context_pyramid(initial_rgb, native_rgb, config, output_size=(640, 480)):
    initial = np.asarray(initial_rgb, dtype=np.float32)
    native = _thwc(native_rgb)
    if initial.ndim != 3 or initial.shape[-1] != 3 or not np.isfinite(initial).all():
        raise ProtocolError("initial RGB must be finite H,W,3")
    native64 = np.stack([cv2.resize(f, (64, 64), interpolation=cv2.INTER_AREA) for f in native])
    context64 = cv2.resize(initial, (64, 64), interpolation=cv2.INTER_AREA)
    context_high = cv2.resize(initial, output_size, interpolation=cv2.INTER_CUBIC)
    context_low_up = cv2.resize(context64, output_size, interpolation=cv2.INTER_CUBIC)
    residual = context_high - context_low_up
    output = []
    for frame in native64:
        distance = np.mean(np.abs(frame - context64), axis=-1)
        mask = np.clip((distance - config.low) / (config.high - config.low), 0.0, 1.0)
        mask = cv2.GaussianBlur(mask, (0, 0), config.sigma)
        mask = cv2.resize(mask, output_size, interpolation=cv2.INTER_LINEAR)[..., None]
        low_up = cv2.resize(frame, output_size, interpolation=cv2.INTER_CUBIC)
        output.append(np.clip(low_up + config.alpha * (1.0 - mask) * residual, 0.0, 1.0))
    return np.stack(output).astype(np.float32)
```

- [ ] **Step 8: Run all renderer/protocol tests and commit**

Run: `python -m pytest tests/test_worldarena_context_pyramid_val.py -q`

Expected: all current tests pass.

Commit:

```bash
git add scripts/worldarena_context_pyramid_val.py tests/test_worldarena_context_pyramid_val.py
git commit -m "feat: add locked WorldArena context renderer"
```

### Task 2: GT PSNR and Aggregate GO/NO-GO Gate

**Files:**
- Modify: `scripts/worldarena_context_pyramid_val.py`
- Modify: `tests/test_worldarena_context_pyramid_val.py`

**Interfaces:**
- Consumes: frame-aligned generated/GT video paths and aggregate dictionaries with keys `psnr`, `image_quality`, `jepa_similarity`, `dynamic_degree`, `motion_smoothness`.
- Produces: `aligned_video_psnr(pred_path: Path, gt_path: Path) -> dict[str, Any]` and `select_candidate(baseline: Mapping[str,float], candidates: Mapping[str,Mapping[str,float]]) -> dict[str,Any]`.

- [ ] **Step 1: Add failing frame-count and aggregate-only gate tests**

```python
def test_psnr_rejects_frame_count_mismatch(tmp_path):
    pred = tmp_path / "pred.mp4"
    gt = tmp_path / "gt.mp4"
    write_video(pred, [np.zeros((16, 16, 3), np.uint8)] * 3)
    write_video(gt, [np.zeros((16, 16, 3), np.uint8)] * 4)
    with pytest.raises(ProtocolError, match="frame count mismatch"):
        aligned_video_psnr(pred, gt)


def test_select_candidate_uses_all_five_aggregate_values_and_tie_breaks():
    baseline = {"psnr": 20.0, "image_quality": 0.5, "jepa_similarity": 0.8,
                "dynamic_degree": 0.4, "motion_smoothness": 0.6, "coverage": 5}
    candidates = {
        "a050_l002_h008": {**baseline, "psnr": 20.30},
        "a075_l002_h008": {**baseline, "psnr": 20.31},
        "a100_l002_h008": {**baseline, "psnr": 21.00, "jepa_similarity": 0.70},
    }
    result = select_candidate(baseline, candidates)
    assert result["decision"] == "GO"
    assert result["selected"] == "a050_l002_h008"
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `python -m pytest tests/test_worldarena_context_pyramid_val.py -k 'psnr or select_candidate' -q`

Expected: FAIL because the scorer and selector are not defined.

- [ ] **Step 3: Implement aligned PSNR and deterministic gate selection**

```python
def aligned_video_psnr(pred_path: Path, gt_path: Path) -> dict[str, Any]:
    pred = read_video_rgb(pred_path)
    gt = read_video_rgb(gt_path)
    if len(pred) != len(gt):
        raise ProtocolError(f"frame count mismatch: {len(pred)} != {len(gt)}")
    resized = np.stack([cv2.resize(frame, (pred.shape[2], pred.shape[1]), interpolation=cv2.INTER_AREA) for frame in gt])
    mse = np.mean((pred.astype(np.float64) - resized.astype(np.float64)) ** 2, axis=(1, 2, 3))
    values = np.where(mse == 0.0, 100.0, 10.0 * np.log10((255.0 ** 2) / mse))
    if not np.isfinite(values).all():
        raise ProtocolError("PSNR contains NaN/Inf")
    return {"mean": float(values.mean()), "per_frame": values.tolist(), "frames": len(values)}


def select_candidate(baseline, candidates):
    required = ("psnr", "image_quality", "jepa_similarity", "dynamic_degree", "motion_smoothness")
    if int(baseline.get("coverage", -1)) != 5:
        raise ProtocolError("baseline coverage must equal five")
    passing = []
    audits = {}
    for name, values in candidates.items():
        if int(values.get("coverage", -1)) != 5:
            raise ProtocolError(f"candidate {name} coverage must equal five")
        checks = {
            "psnr_gain": values["psnr"] - baseline["psnr"] >= 0.25,
            "image_quality": values["image_quality"] >= baseline["image_quality"],
            "jepa_similarity": values["jepa_similarity"] >= 0.97 * baseline["jepa_similarity"],
            "dynamic_degree": values["dynamic_degree"] >= 0.97 * baseline["dynamic_degree"],
            "motion_smoothness": values["motion_smoothness"] >= 0.97 * baseline["motion_smoothness"],
        }
        audits[name] = checks
        if all(checks.values()):
            passing.append(name)
    if not passing:
        return {"decision": "NO-GO", "selected": None, "checks": audits}
    best_gain = max(candidates[name]["psnr"] - baseline["psnr"] for name in passing)
    tied = [name for name in passing if best_gain - (candidates[name]["psnr"] - baseline["psnr"]) <= 0.02]
    selected = min(tied, key=lambda name: parse_variant_name(name))
    return {"decision": "GO", "selected": selected, "checks": audits}
```

- [ ] **Step 4: Run the complete unit test and commit**

Run: `python -m pytest tests/test_worldarena_context_pyramid_val.py -q`

Expected: all tests pass.

Commit:

```bash
git add scripts/worldarena_context_pyramid_val.py tests/test_worldarena_context_pyramid_val.py
git commit -m "feat: gate context renderer on val aggregates"
```

### Task 3: Strict Five-Record Native Generation and Seven-Way Rendering

**Files:**
- Create: `scripts/eval_worldarena_context_pyramid_val5.py`
- Create: `tests/test_worldarena_context_pyramid_val_pipeline.py`

**Interfaces:**
- Consumes: `--config`, `--device 0`, full clean50 manifest, fixed adapt checkpoint, and existing functions `load_checkpoint_for_eval`, `canonical_bimanual_actions`, `validate_temporal_contract`, `_load_instruction`, `_model_episode`, and `_write_h264_atomic` from `scripts/eval_worldarena_s1.py`.
- Produces: `panel.jsonl`, `panel_audit.json`, `native/<id>.npz`, `native/<id>.json`, and `rendered/<variant>/<task>_episode<episode>.mp4` for `baseline` plus six locked variants.

- [ ] **Step 1: Write failing tests for pre-load leakage guard and synthetic seven-way coverage**

```python
def test_build_panel_audit_contains_only_val_rows_and_hashes(tmp_path):
    rows = make_50_task_rows()
    panel, audit = build_panel_audit(rows)
    assert [r["episode"] for r in panel] == [36, 37, 38, 39, 36]
    assert audit["future_gt_used_for_inference"] is False
    assert len(audit["manifest_row_sha256"]) == 5
    assert not any("episode4" in identity for identity in audit["ids"])


def test_render_native_cache_writes_baseline_plus_six_variants(tmp_path):
    initial = np.zeros((48, 64, 3), np.uint8)
    native = np.zeros((2, 3, 64, 64), np.float32)
    written = render_native_cache(initial, native, tmp_path, "task_00_episode36.mp4", fps=10)
    assert set(written) == {"baseline", *(variant_name(c) for c in locked_grid())}
    assert all(path.is_file() for path in written.values())
```

- [ ] **Step 2: Run the pipeline tests and verify RED**

Run: `python -m pytest tests/test_worldarena_context_pyramid_val_pipeline.py -k 'panel_audit or render_native' -q`

Expected: collection fails because `eval_worldarena_context_pyramid_val5` does not exist.

- [ ] **Step 3: Implement panel/audit and reusable cache rendering before model code**

```python
def build_panel_audit(rows):
    panel = select_locked_panel(rows)
    raw = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in panel]
    audit = {
        "schema": "wm3d_v7_worldarena_context_pyramid_val5_panel_v1",
        "allowed_episodes": [36, 37, 38, 39],
        "forbidden_test_episodes": [40, 49],
        "ids": [str(row["id"]) for row in panel],
        "manifest_row_sha256": [hashlib.sha256(value.encode()).hexdigest() for value in raw],
        "future_gt_used_for_inference": False,
    }
    return panel, audit


def render_native_cache(initial_rgb, native_rgb, root, name, fps):
    variants = {"baseline": render_baseline(initial_rgb, native_rgb)}
    variants.update({variant_name(cfg): render_context_pyramid(initial_rgb, native_rgb, cfg) for cfg in locked_grid()})
    written = {}
    first = cv2.resize(cv2.cvtColor(initial_rgb, cv2.COLOR_RGB2BGR), (640, 480), interpolation=cv2.INTER_LINEAR)
    for variant, rgb in variants.items():
        path = Path(root) / "rendered" / variant / name
        frames = [first] + [cv2.cvtColor(np.rint(frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR) for frame in rgb]
        _write_h264_atomic(frames, path, fps)
        written[variant] = path
    return written
```

- [ ] **Step 4: Run the synthetic tests and verify GREEN**

Run: `python -m pytest tests/test_worldarena_context_pyramid_val_pipeline.py -k 'panel_audit or render_native' -q`

Expected: `2 passed`.

- [ ] **Step 5: Add a failing test proving the model loader cannot run before the panel guard**

```python
def test_main_validates_panel_before_loading_model(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(module, "read_manifest", lambda _: make_unsafe_49_task_rows())
    monkeypatch.setattr(module, "load_checkpoint_for_eval", lambda *a, **k: events.append("loaded"))
    with pytest.raises(ProtocolError, match="expected 50 tasks"):
        module.run_generation(make_args(tmp_path))
    assert events == []
```

- [ ] **Step 6: Implement strict generation ordering and inference audit**

```python
def run_generation(args):
    rows = read_manifest(args.manifest)
    panel, panel_audit = build_panel_audit(rows)
    validate_source_files(panel)
    atomic_json(args.output_root / "panel_audit.json", panel_audit)
    atomic_jsonl(args.output_root / "panel.jsonl", panel)
    model, tokenizer, action_mean, action_std, load_audit = load_checkpoint_for_eval(
        args.checkpoint, torch.device(f"cuda:{args.device}"), expected_adaptation="worldarena_bimanual"
    )
    for row in panel:
        initial_rgb, left, right, future_frames, fps = load_inference_inputs(row)
        tokenized = tokenizer.tokenize([initial_rgb], _load_instruction(row))
        predicted, rollout_audit = _model_episode(
            model=model, task_embedding=tokenized.task_emb.to(model.device),
            initial_state=tokenized.context_tokens.to(model.device),
            initial_context_rgb=tokenized.context_rgb.to(model.device),
            left_physical=left, right_physical=right, action_mean=action_mean,
            action_std=action_std, device=model.device,
        )
        if predicted.shape[0] != future_frames:
            raise ProtocolError("native prediction length mismatch")
        atomic_npz(args.output_root / "native" / f"{row['id']}.npz", initial_rgb=initial_rgb, native_rgb=predicted)
        render_native_cache(initial_rgb, predicted, args.output_root, output_name(row), fps)
        atomic_json(args.output_root / "native" / f"{row['id']}.json", {
            "id": row["id"], "episode": row["episode"], "no_future_ground_truth": True,
            "inference_inputs": ["initial_rgb", "instruction", "physical_actions"],
            "rollout": rollout_audit, "load": load_audit,
        })
```

- [ ] **Step 7: Run pipeline tests, static forbidden-pattern audit, and commit**

Run:

```bash
python -m pytest tests/test_worldarena_context_pyramid_val_pipeline.py -q
python -m pytest tests/test_worldarena_context_pyramid_val.py -q
python - <<'PY'
from pathlib import Path
text = Path('scripts/eval_worldarena_context_pyramid_val5.py').read_text()
assert 'load_test_manifest' not in text
assert 'split == "test"' not in text and "split == 'test'" not in text
print('generation leakage static audit: PASS')
PY
```

Expected: all tests pass and the audit prints `PASS`.

Commit:

```bash
git add scripts/eval_worldarena_context_pyramid_val5.py tests/test_worldarena_context_pyramid_val_pipeline.py
git commit -m "feat: generate locked WorldArena val renderer panel"
```

### Task 4: Official Metric Preparation, Four-GPU Runner, and Report

**Files:**
- Create: `scripts/run_worldarena_context_pyramid_val5.py`
- Create: `configs/benchmarks/worldarena_context_pyramid_val5.yaml`
- Modify: `tests/test_worldarena_context_pyramid_val_pipeline.py`

**Interfaces:**
- Consumes: seven rendered directories, `panel.jsonl`, pinned WorldArena repo/weights, node43 GPUs `[0,1,2,3]`.
- Produces: per-variant official frame trees; `metrics/<variant>/{image_quality,dynamic_degree,motion_smoothness,jepa_similarity}`; `psnr/<variant>.json`; `selection_report.json`; `provenance/*.json`; and `contact_sheets/*.jpg`.

- [ ] **Step 1: Write failing tests for val-only summaries, metric coverage, and no-test audit**

```python
def test_prepare_variant_summary_accepts_exact_val5_and_rejects_test_episode(tmp_path):
    panel = make_selected_panel(tmp_path)
    summary = prepare_variant_summary(panel, tmp_path / "rendered" / "baseline", tmp_path / "prepared")
    assert len(summary) == 5
    assert all(Path(item["gt_path"]).parts[-5].startswith("task_") for item in summary)
    panel[0]["episode"] = 40
    with pytest.raises(ProtocolError, match="episodes 36-39"):
        prepare_variant_summary(panel, tmp_path / "rendered" / "baseline", tmp_path / "bad")


def test_validate_diagnostic_tree_requires_five_unique_videos(tmp_path):
    tree = make_synthetic_official_tree(tmp_path, count=4)
    with pytest.raises(ProtocolError, match="coverage"):
        validate_diagnostic_tree(tree, expected_count=5)


def test_assert_no_test_episode_reference_scans_report_values():
    with pytest.raises(ProtocolError, match="test episode reference"):
        assert_no_test_episode_reference({"video": "/x/task_episode40.mp4"})
```

- [ ] **Step 2: Run runner tests and verify RED**

Run: `python -m pytest tests/test_worldarena_context_pyramid_val_pipeline.py -k 'prepare_variant or diagnostic_tree or no_test' -q`

Expected: FAIL because the runner functions do not exist.

- [ ] **Step 3: Implement diagnostic-only summary/frame preparation without changing formal preparer**

```python
def prepare_variant_summary(panel, prediction_dir, output_root):
    if len(panel) != 5 or len({str(row["id"]) for row in panel}) != 5:
        raise ProtocolError("diagnostic panel coverage must equal five")
    if any(int(row["episode"]) not in (36, 37, 38, 39) for row in panel):
        raise ProtocolError("diagnostic rows must use episodes 36-39")
    summary = []
    for row in panel:
        name = output_name(row)
        pred = Path(prediction_dir) / name
        gt = Path(row["video_file"]).resolve(strict=True)
        if not pred.is_file():
            raise ProtocolError(f"missing prediction: {pred}")
        shaped = Path(output_root) / "structured_gt" / row["task"] / "source" / "video" / "frames" / gt.name
        safe_symlink(gt, shaped)
        image = Path(output_root) / "initial" / f"{Path(name).stem}.png"
        write_first_frame(gt, image)
        summary.append({"gt_path": str(shaped.absolute()), "image": str(image.absolute()),
                        "prompt": [instruction(row)], "generated_name": name, "id": row["id"]})
    return summary
```

- [ ] **Step 4: Implement explicit four-GPU scheduling and pinned metric commands**

```python
STANDARD = ("image_quality", "dynamic_degree", "motion_smoothness")


def build_jobs(variants, config, root):
    jobs = []
    for index, variant in enumerate(variants):
        gpu = index % 4
        jobs.append({"variant": variant, "gpu": gpu, "kind": "standard", "metrics": STANDARD,
                     "command": [sys.executable, "scripts/run_worldarena_metric_queue.py",
                                 "--config", str(config_for_variant(config, variant, root)),
                                 "--worldarena-repo", config["worldarena"]["repo"],
                                 "--run-root", str(root / "metrics" / variant),
                                 "--gpu", str(gpu), "--expected-count", "5", "--metrics", *STANDARD]})
        jobs.append({"variant": variant, "gpu": gpu, "kind": "jepa",
                     "command": jedi_command(config, variant, root, gpu)})
    return jobs
```

The executor groups jobs by physical GPU, starts four worker processes, and runs each GPU's standard job followed by its JEDi job. It sets `CUDA_VISIBLE_DEVICES` to the assigned physical GPU, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and JEDi working directory `/data/Minko/models/worldarena_v1/jedi`. A non-zero return code is fatal and is written to provenance before raising.

- [ ] **Step 5: Add the pinned config**

```yaml
schema: wm3d_v7_worldarena_context_pyramid_val5_v1
checkpoint: /data/Minko/world_model/wm3d_v7/results/benchmarks/worldarena_bimanual_adapt/formal1000_diverse3/ckpt/step_00001000.pt
manifest: /data/Minko/world_model/wm3d_v7/manifests/benchmarks/worldarena_clean50_paper_v1.jsonl
output_root: /data/Minko/world_model/wm3d_v7/results/benchmarks/worldarena_bimanual_adapt/context_pyramid_val5
allowed_episodes: [36, 37, 38, 39]
forbidden_test_episodes: [40, 49]
expected_coverage: 5
metric_devices: [0, 1, 2, 3]
worldarena:
  repo: /data/Minko/external/WorldArena
  commit: a918b93f8533a4e9452a224c0ec54d27e527c4bb
weights:
  raft: /data/Minko/models/worldarena_v1/raft/RAFT/models/raft-things.pth
  vfimamba: /data/Minko/models/worldarena_v1/vfimamba/model.pkl
  musiq: /data/Minko/models/worldarena_v1/musiq/musiq_spaq_ckpt-358bb6af.pth
  jedi_dir: /data/Minko/models/worldarena_v1/jedi
gate:
  min_psnr_gain_db: 0.25
  min_image_quality_ratio: 1.0
  min_jepa_ratio: 0.97
  min_dynamic_ratio: 0.97
  min_smoothness_ratio: 0.97
  psnr_tie_db: 0.02
```

- [ ] **Step 6: Implement aggregation and report/contact-sheet creation**

```python
def aggregate_variant(root, variant, panel):
    values = {
        metric: read_official_mean(root / "metrics" / variant, metric, expected_count=5)
        for metric in STANDARD
    }
    values["jepa_similarity"] = read_jedi_score(root / "metrics" / variant, expected_count=5)
    psnr = [aligned_video_psnr(root / "rendered" / variant / output_name(row), Path(row["video_file"])) for row in panel]
    values.update({"psnr": float(np.mean([item["mean"] for item in psnr])), "coverage": 5})
    atomic_json(root / "psnr" / f"{variant}.json", {"aggregate": values["psnr"], "details": psnr})
    return values


def finalize(root, panel):
    variants = ["baseline", *(variant_name(cfg) for cfg in locked_grid())]
    aggregates = {name: aggregate_variant(root, name, panel) for name in variants}
    selection = select_candidate(aggregates["baseline"], {k: v for k, v in aggregates.items() if k != "baseline"})
    report = {"schema": "wm3d_v7_worldarena_context_pyramid_val5_result_v1",
              "decision": selection["decision"], "selected": selection["selected"],
              "aggregates": aggregates, "gate": selection["checks"],
              "ids": [row["id"] for row in panel], "test500_authorized": False}
    assert_no_test_episode_reference(report)
    atomic_json(root / "selection_report.json", report)
    write_contact_sheets(root, panel, selection["selected"])
    return report
```

- [ ] **Step 7: Run all diagnostic tests and commit**

Run:

```bash
python -m pytest tests/test_worldarena_context_pyramid_val.py tests/test_worldarena_context_pyramid_val_pipeline.py -q
python -m py_compile scripts/worldarena_context_pyramid_val.py scripts/eval_worldarena_context_pyramid_val5.py scripts/run_worldarena_context_pyramid_val5.py
```

Expected: every test passes and all three CLIs compile.

Commit:

```bash
git add scripts/run_worldarena_context_pyramid_val5.py configs/benchmarks/worldarena_context_pyramid_val5.yaml tests/test_worldarena_context_pyramid_val_pipeline.py
git commit -m "feat: score WorldArena context renderer val panel"
```

### Task 5: Run the Five-Video Diagnostic on Node43

**Files:**
- No source edits expected.
- Create runtime artifacts only under `results/benchmarks/worldarena_bimanual_adapt/context_pyramid_val5/`.

**Interfaces:**
- Consumes: completed Tasks 1–4 and free node43 GPUs 0–3.
- Produces: a verified `selection_report.json`, provenance, metric files, and five contact sheets.

- [ ] **Step 1: Verify node43 capacity and immutable inputs**

Run:

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
df -h /data/Minko
sha256sum results/benchmarks/worldarena_bimanual_adapt/formal1000_diverse3/ckpt/step_00001000.pt \
  manifests/benchmarks/worldarena_clean50_paper_v1.jsonl
```

Expected: GPUs 0–3 have enough free memory, disk has at least 20 GB free, and both hashes are recorded in `provenance/input_hashes.txt`.

- [ ] **Step 2: Generate native rollouts once and render all seven variants**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/eval_worldarena_context_pyramid_val5.py \
  --config configs/benchmarks/worldarena_context_pyramid_val5.yaml \
  --device 0
```

Expected: five native caches, seven directories with five playable MP4s each, and every native audit says `no_future_ground_truth: true`.

- [ ] **Step 3: Prove no test record was read or emitted**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path('results/benchmarks/worldarena_bimanual_adapt/context_pyramid_val5')
panel = [json.loads(line) for line in (root / 'panel.jsonl').read_text().splitlines() if line]
assert len(panel) == 5
assert [row['episode'] for row in panel] == [36, 37, 38, 39, 36]
for path in root.rglob('*'):
    assert 'episode40' not in str(path) and 'episode49' not in str(path)
print('val-only runtime audit: PASS')
PY
```

Expected: `val-only runtime audit: PASS`.

- [ ] **Step 4: Run official metrics with four physical GPUs**

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_worldarena_context_pyramid_val5.py \
  --config configs/benchmarks/worldarena_context_pyramid_val5.yaml \
  --prepare --metrics --gpus 0 1 2 3
```

Expected: 21 standard metric results plus seven JEDi results, all with five unique videos and no non-zero subprocess return code.

- [ ] **Step 5: Aggregate, select, and build visual evidence**

Run:

```bash
python scripts/run_worldarena_context_pyramid_val5.py \
  --config configs/benchmarks/worldarena_context_pyramid_val5.yaml \
  --finalize
```

Expected: `selection_report.json` has decision `GO` or `NO-GO`, includes all seven aggregate rows, has `test500_authorized: false`, and writes five non-empty JPEG contact sheets.

- [ ] **Step 6: Final verification before reporting**

Run:

```bash
python -m pytest tests/test_worldarena_context_pyramid_val.py tests/test_worldarena_context_pyramid_val_pipeline.py -q
python - <<'PY'
import json, math
from pathlib import Path
root = Path('results/benchmarks/worldarena_bimanual_adapt/context_pyramid_val5')
report = json.loads((root / 'selection_report.json').read_text())
assert report['decision'] in {'GO', 'NO-GO'}
assert len(report['aggregates']) == 7
assert all(v['coverage'] == 5 for v in report['aggregates'].values())
assert all(math.isfinite(float(x)) for v in report['aggregates'].values() for k, x in v.items() if k != 'coverage')
assert len(list((root / 'contact_sheets').glob('*.jpg'))) == 5
assert report['test500_authorized'] is False
print(json.dumps({'decision': report['decision'], 'selected': report['selected'], 'aggregates': report['aggregates']}, indent=2))
PY
```

Expected: tests pass and the command prints one auditable GO/NO-GO decision. Report only this validation result; do not start test500.

---

## Self-Review Record

- Spec coverage: panel selection, episode guard, no-future-GT inference, exact grid, renderer equation, official four-metric comparison, aligned PSNR, aggregate gate, tie-break, provenance, coverage checks, visualization, GPU allowlist, and no-test500 rule each map to a task above.
- Placeholder scan: the plan contains no deferred implementation markers; every code-producing task names exact files, interfaces, commands, expected failure, implementation, verification, and commit.
- Type consistency: all later tasks consume the exact `RenderConfig`, `locked_grid`, `variant_name`, `render_baseline`, `render_context_pyramid`, `aligned_video_psnr`, and `select_candidate` interfaces introduced in Tasks 1–2.
- Scope control: no existing formal evaluator/preparer/result is modified, and no optical flow, Wan renderer, checkpoint training, per-task tuning, or test episode appears in execution.
