# VLA Stage D — Native-3D Scene-Flow + Teacher-Forcing IDM

Date: 2026-05-26
Branch: `wm3d-vla-fix` (continuation)
Prior art: A `e708225` → B `7f7c57a` → C `ab8a372` (see `results/wm3d_v3_vla_c/eval/vla_analysis/FINAL_VLA_REPORT.md`)

## Context

Stage C (`JointWorldModelC`, depth-augmented IDM, 15 epoch / 6h on 2× H100) failed all three strict gates but was the first stage to break out of zero-baseline noise:

- bridge pose MSE: -13.9% vs zero-baseline (target -30%)
- bridge grip switch F1: 0.108 (target ≥ 0.60)
- fractal pose MSE: -13.1% (target -15%, missed by ~2 pp)

Per-axis: translation dy/dz learn (-19/-25%), rotation drx/dry/drz stay near zero-baseline (-2 to -7%). The residual gap was diagnosed as **four structural limits not addressed in A/B/C**:

1. IDM reads from the model's blurry predicted future (`pred_tokens`), not the real future — there is no clean target for "look at the visual change, recover the action."
2. VGGT pooled tokens (64 patches over 224×224) destroy sub-patch motion. Wrist rotations cause sub-patch displacements.
3. Depth maps are largely invariant to in-plane rotation; native-3D depth signal carries no angular information.
4. Multi-task gradient remains dominated by dense pixel-level losses (state_mse, geom_depth). Raising pose loss weight to 10× did not move the gradient balance enough.
5. (small / fixable) Gripper `pos_weight` was tuned against "currently closed" rate (~0.54), but the gate measures **switch detection**, whose event rate is ~0.046.

## Goal

Stage D attempts the final push **within the native-3D constraint**: no raw RGB re-encoding, no third-party motion encoders (RAFT, etc.), only VGGT-derived 3D quantities. Targets the same three gates as A/B/C.

## Hard constraint

All input signals to the model must be derived from the VGGT 3D foundation backbone or its direct geometric outputs. The thesis of "native 3D world model" is preserved.

## Architecture changes (5)

### Change 1: Re-cache VGGT `point_map`

Currently cached: pooled tokens [n,64,2048] + dense depth [n,224,224].
Add: **`point_map`** [n,224,224,3] in world coordinates, plus `intrinsic` [n,3,3] per-frame.

Storage: ~200 GB additional (fp16) on the OXE manifest. Run on idle cards as a one-shot batch job.

Format: `cache/wm3d_v3/vggt_pointmap/<safe_id>.npz` with keys `{"point": fp16[n,H,W,3], "intrinsic": fp32[n,3,3]}`.

### Change 2: SceneFlowEncoder

Compute per-frame 3D scene flow:
```
flow[t] = point_map[t+1] - point_map[t]    # [T-1, 224, 224, 3]
```

Encoder: 4-stage stride-2 CNN (3→16→32→64→128), adaptive pool to 8×8, linear to `d_flow=256`. ~0.3M params. Mirrors the existing `DepthEncoder` shape but takes 3-channel 3D-flow input.

**For the last frame** (no t+1 available in `s_in`), use zero-flow.

For the future stream into IDM: flow over (pred_t, pred_{t+1}) using backprojected `pmap_pred`.

Output token per patch: concatenated alongside the existing `depth_tok` and pooled VGGT token. IDM's `extra_token_dim` extends from 256 to 512.

### Change 3: Teacher-forcing IDM

Replace the IDMStream's `pred` input depending on training phase:

| epoch range | `fut_for_idm` | `depth_for_idm` | `pmap_for_idm` |
| - | - | - | - |
| 0-4 (teacher) | `s_tgt` (GT) | `depth_tgt` (GT) | `pmap_tgt` (GT cache) |
| 5-9 (anneal) | `mix(α) = (1-α)·s_tgt + α·pred` where α = (epoch-5)/5 | same mix | same mix |
| 10-14 (pred) | `pred` | `depth_pred` | backprojected from `depth_pred + intrinsic` |

State-stream loss (`L_state`, `L_cos`) and depth loss (`L_depth`) are computed against the **pred path** in all phases — the teacher path is only consumed by IDM. This keeps the world-model objectives honest.

### Change 4: Late-stage phased freezing

Epoch 10 onward: freeze `state`, `geom`, `depth_enc`, `flow_enc`. Only `idm` and `action_proj` continue training. Pose loss weight bumped from 10 to 30 in this phase.

Rationale: the dense pixel-level losses dominate gradient mass; freezing them during the final fine-tune lets IDM specialize on action.

### Change 5: Gripper supervision fix

Two changes to the gripper objective:

- **`pos_weight` recomputed against switch event rate**, not closed-state rate. From Stage C eval: switch_rate ≈ 0.046 on bridge → `pos_weight ≈ (1 - 0.046) / 0.046 ≈ 20.7`. Compute once on the train manifest like the action stats.
- **Add `L_switch`** auxiliary loss: focal-BCE on first-differenced logits against ground-truth state-change indicators.

```python
gp_diff   = grip_logit[:, 1:] - grip_logit[:, :-1]          # [B, k-1]
gt_switch = (grip_tgt[:,1:] != grip_tgt[:,:-1]).float()
L_switch  = focal_bce(gp_diff.abs(), gt_switch, alpha=0.25, gamma=2.0)
```

Weight 1.0. Added to total loss alongside L_grip.

## Backprojection helper

`pmap_pred` for inference / phase 3 is computed from predicted depth + cached intrinsic, **not** from a new neural head:

```python
def backproject(depth: Tensor, K: Tensor) -> Tensor:
    """[B, T, H, W] + [B, T, 3, 3] -> [B, T, H, W, 3] camera-frame points."""
    H, W = depth.shape[-2:]
    v, u = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    uv1 = stack([u, v, ones_like(u)]).float()                # [3, H, W]
    K_inv = inverse(K)                                       # [B, T, 3, 3]
    rays = einsum("btij,jhw->bthwi", K_inv, uv1)            # [B, T, H, W, 3]
    return rays * depth.unsqueeze(-1)                        # camera-frame
```

Note: this is *camera-frame* points. The GT cached `point_map` is *world-frame*. We will compare both options during smoke test (Task D-3) and pick whichever gives stable scene flow under bf16.

## Loss recipe

```
L_total = w_pose * L_pose
        + w_grip * L_grip
        + w_switch * L_switch
        + w_state * L_state
        + w_cos * L_cos
        + w_depth * L_depth
```

| weight | epoch 0-9 | epoch 10-14 |
| - | - | - |
| w_pose | 10.0 | **30.0** |
| w_grip | 2.0 | 2.0 |
| w_switch | 1.0 | 1.0 |
| w_state | 1.0 | 0.0 (frozen) |
| w_cos | 0.1 | 0.0 (frozen) |
| w_depth | 0.5 | 0.0 (frozen) |

L_pose: Huber(δ=1.0) in standardized space (unchanged from C).
L_grip: focal-BCE(α=0.25, γ=2.0, pos_weight≈20.7).
L_switch: focal-BCE on differenced logits.

## Gates (unchanged from A/B/C)

1. **bridge** `pose_mse_overall` < 0.000590 (≥30% below baseline 0.000843)
2. **bridge** `grip_switch_f1` ≥ 0.60
3. **fractal** `pose_mse_overall` < 0.01011 (≥15% below baseline 0.01189)

### Intermediate gates (added)

| checkpoint | expected | failure interpretation |
| - | - | - |
| after re-cache | 100-frame sanity: point map consistent, scene flow non-degenerate | cache pipeline broken |
| ep 5 (peak teacher) | bridge pose ≤ -20% (teacher-forcing ceiling) | structural — even GT future cannot drive learning, **STOP** |
| ep 10 (start pred) | bridge pose ≤ -22% | annealing too fast, extend mix phase |
| ep 14 (final) | three strict gates | reach failure → confirm native-3D ceiling and ship final report |

## Risks

1. **Re-cache wall time**: 22k episodes × ~30 frames each. VGGT inference: ~1.5s/frame on H100 → ~280h serial, ~16-18h on a 2-GPU split. Can run in background on cards 0/1 if available, otherwise queue after current v3.5 training. **Mitigation**: subset cache (fractal-only first to validate, ~50 GB) and only proceed if scene flow looks usable.
2. **fp16 scene flow noise**: differencing fp16 point maps amplifies quantization error. **Mitigation**: light 3×3 box filter on flow before encoding; fall back to fp32 cache if filter doesn't fix it (storage 2× → 400 GB).
3. **Teacher-forcing inference gap**: model leans on clean GT during epochs 0-4 and degrades when switched to pred. **Mitigation**: annealing schedule is gradual (5 epochs), and ep 10 intermediate gate catches collapse.
4. **Backprojected pmap_pred quality**: depends on `depth_pred` accuracy. If depth predictions are poor, scene flow on pmap_pred is junk. **Mitigation**: compare backproject vs new PointDecoder head during smoke test; can add a small PointDecoder if needed (out of scope for default plan, listed as fallback).
5. **GPU availability**: cards 2/3 currently occupied by DreamerVLA preprocessing. Stage D training waits until that finishes or finds cards 0/1 free.

## Out of scope (explicitly)

- Raw RGB encoding (violates native-3D constraint)
- Optical flow from RAFT or any pretrained flow model (third-party, not native-3D)
- Action-conditioned state generation (architectural overhaul — v4 territory, separate effort)
- Diffusion / flow-matching action head (independent improvement vector, deferred)
- Proprioception channel (would change dataset contract; deferred)

## Files

### New
- `wm3d_v3/scripts/cache_vggt_pointmap.py` — VGGT pointmap re-cache (batched)
- `wm3d_v3/scripts/compute_switch_stats.py` — gripper switch rate over train manifest
- `wm3d_v3/wm3d_v3/models/scene_flow_encoder.py` — SceneFlowEncoder + backproject helper
- `wm3d_v3/wm3d_v3/models/joint_model_d.py` — JointWorldModelD wrapper
- `wm3d_v3/wm3d_v3/training/train_vla_d.py` — Stage D trainer (teacher-forcing schedule + phased freeze)
- `wm3d_v3/configs/v3_vla_d.yaml`
- `wm3d_v3/scripts/train_v3_vla_d.sh`
- `wm3d_v3/tests/test_stage_d.py`

### Modified
- `wm3d_v3/wm3d_v3/data/window_dataset.py` — emit `pmap_in`, `pmap_tgt`, `intrinsic`
- `wm3d_v3/wm3d_v3/models/idm_stream.py` — accept extended `extra_token_dim`
- `wm3d_v3/wm3d_v3/losses.py` — add `L_switch`
- `wm3d_v3/scripts/analyze_vla.py` — `--variant d` support
- `wm3d_v3/configs/v3_vla_c.yaml` — (no changes; D has its own config)

## Success criteria (write final report regardless)

Stage D passes gates → declare VLA success, native-3D thesis validated.
Stage D fails gates → declare the native-3D ceiling, hand off to v4 architectural redesign.

Either way the v3 effort closes here.
