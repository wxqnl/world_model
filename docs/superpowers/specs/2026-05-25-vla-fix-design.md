# 2026-05-25 — VLA Fix Design (wm3d_v3 → wm3d_v3_vla)

## Status
- Owner: tancilon
- Compute: GPU 2,3 (cards 0,1 reserved for v3.5 training)
- Branch base: `wm3d-v3` from `dac02a6`
- Spec authored after the [VLA findings](../../../results/wm3d_v3/eval/vla_analysis/VLA_FINDINGS.md) showed `best.pt` was statistically indistinguishable from a zero-action baseline on bridge and at random-level on gripper switching.

## Goals
Move the v3 VLA path from "degenerate to zero" to "useful predictor", without disturbing the already-validated RGB / depth heads.

### Strict success gates (Stage A)
1. `bridge pose_mse_overall` < 0.000590  (≥30% below zero-pred baseline 0.000843)
2. `bridge grip_switch_f1` ≥ 0.60
3. `fractal pose_mse_overall` < 0.01011  (≥15% below zero-pred baseline 0.01189)

All three must pass for Stage A to be considered complete. Any single failure triggers Stage B automatically.

## Constraints
- Keep PixelDecoder and GeomDecoder behaviour frozen (numerically equal RGB/depth outputs)
- Must warm-start from `results/wm3d_v3/ckpt/best.pt`
- Must coexist with v3.5 training (GPU 0,1) — only GPU 2,3 available for Stage A
- GPU policy [[gpu_policy_newwm]]: only GPUs 0–3 usable in this directory

## Root-cause analysis (recap)
Re-evaluating `best.pt` on the full val subset (4182 windows) versus zero-pred / dataset-mean baselines:

| | v3 model | zero-pred | mean-pred | improvement |
| - | - | - | - | - |
| bridge pose_mse  | 0.000841 | 0.000843 | 0.000840 | **0.1%** |
| fractal pose_mse | 0.01134  | 0.01189  | 0.01182  | 4.6% |
| bridge switch F1 | 0.250 | — | — | random |
| fractal switch F1 | n/a (0 switches) | — | — | n/a |

Four contributing causes:
1. `L_pose_a = mse_loss` in raw ±0.1 space — bridge converges with ~0.00085 magnitude, drowned by RGB gradient (L_rgb_l1 × 1.0 ≈ 0.04)
2. `ActionProjHead` clamps output with `tanh × 0.1` — predicting all-zero already minimises bridge L_pose
3. No per-axis standardisation — drz variance 5× larger than dx, gradient steers only big axes
4. `idm_reg = 0.01 · (z_a)^2.mean()` pushes the action latent to zero, removing the bottleneck signal
5. ActionStream sees only `s_in[t..t+T-1]` — has no inverse-dynamics view, only forward conditioning

## Stage A — Surgical fix (target)

### A.1 Data plumbing
- New cache file: `cache/wm3d_v3/action_stats.npz` with keys `mean[7]`, `std[7]`, `pos_rate[1]`. Computed by a one-time script over all `cache/wm3d_v3/actions/*.npy`.
- `OXEWindowDataset.__getitem__` adds:
  - `action_tgt_norm[k,6] = (action_tgt[..., :6] - mean[:6]) / std[:6]`
  - keeps `action_tgt[k,7]` as raw (used for gripper + analysis)
  - `stats` is loaded once at dataset construction (not per item)

### A.2 Model changes (`wm3d_v3/models/action_proj.py` + new `aux_idm.py`)
- `ActionProjHead` rewrite:
  - 5-layer MLP, hidden 1024 (was 768), GELU + LayerNorm after trunk
  - `pose_norm = Linear(1024, 6)` (no tanh, unbounded)
  - `gripper_logit = Linear(1024, 1)` (unchanged behaviour)
  - Module exposes `mean[6]`, `std[6]` as registered buffers so `denormalize(pose_norm) = pose_norm * std + mean` is available for downstream demo/eval scripts
- New `AuxIDM` module (only used during training):
  - input: `s_last [B,64,2048]` (= `s_in[:, -1]`), `s_fut_last [B,64,2048]` (= `pred_tokens[:, -1]`, detached? no — we need gradient through DualStream too)
  - Linear(4096 → 1024), 3-layer transformer encoder
  - query buffer `[1, k, 1024]`, transformer decoder (1 layer)
  - heads: `aux_pose_norm[B,k,6]`, `aux_grip[B,k]`
  - param count: ~12M (small, throwaway)
- `JointWorldModel`:
  - New optional sub-module `self.aux_idm` (constructed iff `cfg.enable_aux_idm`)
  - Forward returns `aux_pose_norm` and `aux_grip` when `aux_idm=True` is passed
  - `pose` in the existing output dict becomes the **de-normalized** value (so existing `analyze_vla.py` and demo scripts don't change). The standardized form is exposed as `pose_norm` for the loss.

### A.3 Loss redesign (`wm3d_v3/losses.py`)
- New `compute_losses_vla(out, tgt, w, lpips_fn=None)`:
  - `L_pose = huber_loss(pose_norm, action_tgt_norm, delta=1.0)` (mean over B,k,6)
  - `L_grip = focal_bce(grip_logit, grip_tgt, alpha=0.25, gamma=2.0, pos_weight=stats.pos_rate)`
  - `L_aux = huber(aux_pose_norm, action_tgt_norm, δ=1.0) + 0.5*focal_bce(aux_grip, grip_tgt, ...)`
  - `L_total = w.action_pose*L_pose + w.action_grip*L_grip + w.aux_idm*L_aux`
  - When `freeze_other=True`: RGB/depth loss terms are **not computed at all** (faster than computing then zero-weighting)
- `LossWeights` extended: `action_pose=10, action_grip=2, aux_idm=5`; legacy fields (`cos`, `geom_depth`, `rgb_l1`, …) default to 0 and are only consulted in non-VLA mode

### A.4 Training script (`wm3d_v3/training/train_vla.py`, new file)
- Imports `JointWorldModel`, builds with config flag `enable_aux_idm=true`
- `--resume_from path/to/best.pt` loads only matching state-dict keys (the new `pose_norm` layer replaces the old `pose_head/grip_head`, AuxIDM is fresh)
- Two-phase schedule:
  - Phase A.1 (epochs 0–2): `param.requires_grad=False` for `pixel`, `geom`, `dual.state_stream`, `dual.action_stream`, `dual.cross_attn`. Only `action_proj` and `aux_idm` train.
  - Phase A.2 (epochs 3–9): unfreeze `dual.action_stream` and the dual-stream cross-attention layers. `state_stream` stays frozen (the world model dynamics are validated).
- Optimizer: `AdamW` `lr=5e-5`, cosine to `lr=5e-6`, `warmup_steps=500`, `betas=(0.9, 0.95)`, `weight_decay=0.02`
- `batch_size_per_gpu=8`, DDP on GPUs 2,3 (`CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2`)
- Ckpt every epoch (smaller model graph, ckpt cheap). best.pt selected by `val/L_pose_norm_huber + 0.5 * val/L_grip_focal`
- TensorBoard at `results/wm3d_v3_vla/tb`, ckpt at `results/wm3d_v3_vla/ckpt`

### A.5 Config (`wm3d_v3/configs/v3_vla.yaml`)
Inherits from `v3_oxe.yaml` then overrides:
```yaml
model:
  enable_pixel: true       # still computed for forward pass, but no loss
  enable_aux_idm: true
  action_proj_hidden: 1024
train:
  epochs: 10
  batch_size_per_gpu: 8
  lr: 5.0e-5
  warmup_steps: 500
  freeze_phases:
    - { epoch_start: 0, freeze: [pixel, geom, dual.state_stream, dual.action_stream, dual.cross_attn] }
    - { epoch_start: 3, freeze: [pixel, geom, dual.state_stream] }
loss:
  action_pose: 10.0
  action_grip: 2.0
  aux_idm: 5.0
  # all geom/rgb weights default 0
out:
  root: /home/user01/Minko/newwm/results/wm3d_v3_vla
  tb_dir: tb
  ckpt_dir: ckpt
```

### A.6 Evaluation
- After A.1 (epoch 2): `analyze_vla.py --max_batches 100 --device cuda:0` (CUDA_VISIBLE_DEVICES=2). Quick sanity check.
- After A.2 final: full `analyze_vla.py` → produces `report.json` and plot set under `results/wm3d_v3_vla/eval/vla_analysis/`
- Compare against `results/wm3d_v3/eval/vla_analysis/report.json` (baseline). Save side-by-side table to `VLA_FINDINGS.md`.

### A.7 Risks & mitigations
| Risk | Mitigation |
| - | - |
| Standardization breaks downstream demo scripts | `JointWorldModel.forward` returns de-normalized `pose`; only `pose_norm` is new |
| Focal BCE numerically unstable on long constant-gripper streaks | Clamp `logit` and use `binary_cross_entropy_with_logits` underneath |
| AuxIDM trivially copies last GT action via the `pred_tokens` shortcut | AuxIDM never sees ground truth, only `s_in[:,-1]` and `pred_tokens[:,-1]`. The latter is conditioned on `z_a` which is what we want to *shape*, so this gradient is constructive |
| Lr=5e-5 too low / too high | Quick sweep at epoch 0: print `grad_norm` first 50 steps — if < 1e-4 raise lr, if > 5 lower it. Document tuned value in TB |
| 12-18h is too long if A clearly failing | Eval at end of every epoch (full val). If epoch 4 pose MSE > 0.9× zero-baseline, abort A early and start B |

## Stage B — True IDM (triggered on A failure)

### Trigger condition
Any of the strict gates fails after A.2 epoch 9. Or early abort if epoch ≥ 4 and `pose_mse_overall_bridge > 0.000800` (5% below zero).

### B.1 Architecture
Replace `ActionStream` with `IDMStream`:
- Input: concat of `s_in[t..t+T-1]` and `pred_tokens[t+T..t+T+k-1]` along time axis → `[B, T+k, 64, 2048]`. Position embedding distinguishes "observed" vs "predicted".
- Self-attn over the full T+k frames; cross-attn from `dec_q[k]` over the joined sequence.
- Same z_a → ActionProjHead → pose / gripper output.

### B.2 Training
- From scratch (no warm-start; the IDM needs to shape its own dynamics)
- 4 GPU DDP on cards 0-3 (wait for v3.5 to finish on 0,1; checkpoint timestamps show < 12h remain)
- 20 epochs, ~30h, same loss recipe as A
- Same eval gate

### B.3 Decision after B
- If B passes: declare VLA done, write final report, merge `wm3d-v3-vla` branch
- If B fails: escalate to discussion (likely scope expansion — diffusion head, larger action latent, more OXE shards) — do NOT trigger autonomously

## Deliverables checklist
- [ ] `cache/wm3d_v3/action_stats.npz` (one-time script run)
- [ ] `wm3d_v3/data/window_dataset.py` adds `action_tgt_norm` field
- [ ] `wm3d_v3/models/action_proj.py` rewrite (no tanh, register buffer for mean/std)
- [ ] `wm3d_v3/models/aux_idm.py` new
- [ ] `wm3d_v3/models/joint_model.py` wires AuxIDM, de-normalization
- [ ] `wm3d_v3/losses.py` adds `compute_losses_vla` + `focal_bce`
- [ ] `wm3d_v3/training/train_vla.py` new with phased freezing
- [ ] `wm3d_v3/configs/v3_vla.yaml` new
- [ ] `wm3d_v3/scripts/train_v3_vla.sh` launcher (CUDA_VISIBLE_DEVICES=2,3 torchrun)
- [ ] Eval pass after A.1, A.2; comparison table appended to `VLA_FINDINGS.md`
- [ ] (Conditional) Stage B implementation if A fails the gates

## Out of scope
- Diffusion action head (deferred until Stage B also fails)
- Adding new datasets beyond bridge + fractal
- Touching v3.5 (separate experiment, lane 0,1)
- Anything related to v4 (separate codebase)
