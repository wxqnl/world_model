# Hunyuan Integration Plan for wm3d_v3

Date: 2026-05-29
Status: planning note
Scope: use wm3d_v3 world-model outputs to condition a pretrained Hunyuan video model for higher-fidelity RGB future prediction.

## 1. Current Baseline

The current `p64_138m_actioncond_full` model is an action-conditioned future prediction model. It is useful as an in-domain prediction result, but it should not yet be presented as a strong generalization benchmark.

Current inputs:

- Past visual context encoded as cached VGGT pooled tokens: `T=16, P=64, D=2048`.
- Cached task embedding from Qwen task vectors.
- Future action condition for `k=8` steps: normalized 6-DoF delta pose plus binary gripper.

Current outputs:

- `pred_tokens`: future VGGT token predictions, shape `[B, k, 64, 2048]`.
- `depth`: predicted future depth, shape `[B, k, 224, 224]`.
- `rgb`: rough future RGB from PixelDecoder, shape `[B, k, 3, 256, 256]`.
- `pose` and `gripper_logit`: action-head outputs.

The current RGB path is not video diffusion. It is a supervised PixelDecoder trained with L1 plus LPIPS. The RGB output is useful for layout, visual grounding, and quick diagnostics, but it is expected to be blurry.

## 2. Integration Principle

Do not directly treat wm3d_v3 `pred_tokens` as Hunyuan video latents in the first implementation. VGGT tokens and Hunyuan VAE/DiT latents live in different representation spaces. A direct latent-to-latent regression is likely to be brittle.

Preferred design:

```text
past RGB + task text + future actions
        -> wm3d_v3 world model
        -> pred_tokens + predicted depth + rough RGB
        -> Hunyuan control adapter / LoRA
        -> high-fidelity future RGB video
```

The world model should provide state, geometry, action-conditioned motion, and rough layout. Hunyuan should provide high-frequency appearance and video realism.

## 3. Proposed Architecture

### 3.1 Frozen World Model

Use `wm3d_v3_p64_138m_actioncond_full/ckpt/best.pt` as the first conditioning model. Run it in eval mode to produce `pred_tokens`, `depth`, and optional rough `rgb`. Freeze wm3d_v3 for the first Hunyuan experiments so the effect of the conditioning signal is isolated.

### 3.2 Hunyuan Control Adapter

Train a small adapter around the pretrained Hunyuan model while keeping most Hunyuan weights frozen.

Candidate conditioning branches:

1. Rough RGB control branch
   - Input: PixelDecoder rough RGB video.
   - Encoder: lightweight 2D/3D CNN or temporal ConvNet.
   - Injection: residual control features into Hunyuan denoiser blocks.

2. Depth control branch
   - Input: predicted depth video.
   - Encoder: depth normalization plus lightweight CNN/3D CNN.
   - Injection: ControlNet-style residuals or block-wise conditioning.

3. Token condition branch
   - Input: `pred_tokens [B,k,64,2048]`.
   - Encoder: linear projection plus a small temporal/spatial transformer.
   - Output: cross-attention tokens for Hunyuan, or FiLM/scale-shift conditioning per denoiser block.

4. Text/action condition
   - Use task text through Hunyuan native text encoder if available.
   - Optionally pass future action embeddings through the same condition stack.

First train only Hunyuan LoRA weights, control adapter weights, and VGGT-token projection layers. Keep wm3d_v3, Hunyuan base, and Hunyuan VAE frozen.

## 4. Training Data and Split

Use the OXE window format initially:

- Input context: past 16 frames.
- Target: future 8 frames.
- Action condition: future 8 actions.
- Optional long rollout after short-horizon training works.

Before claiming generalization, change validation from random window split to episode-level split. Prefer a held-out dataset or held-out robot split when enough cached data is available.

## 5. Training Objective

Keep Hunyuan original diffusion denoising objective as the main loss.

Training loop:

1. Sample an OXE window.
2. Run frozen wm3d_v3 to get `pred_tokens`, `depth`, and optional rough `rgb`.
3. Encode target future RGB using Hunyuan VAE.
4. Apply the diffusion noising schedule.
5. Train Hunyuan LoRA/control adapter to predict noise or velocity conditioned on world-model outputs.

Optional auxiliary losses can include low-weight decoded RGB loss, temporal consistency loss, or depth consistency loss. Do not over-weight direct pixel losses, or Hunyuan may collapse toward the blurry PixelDecoder output.

## 6. Ablation Plan

Run these in order:

1. Hunyuan baseline: context/text only, no wm3d_v3 condition.
2. Hunyuan + rough RGB only.
3. Hunyuan + predicted depth only.
4. Hunyuan + rough RGB + predicted depth.
5. Hunyuan + `pred_tokens` only.
6. Hunyuan + `pred_tokens` + depth.
7. Hunyuan + `pred_tokens` + depth + rough RGB.

Key questions:

- Does `pred_tokens` add useful state information beyond rough RGB and depth?
- Does rough RGB help, or does it bias Hunyuan toward blurry output?
- Does predicted depth improve object placement and motion consistency?

## 7. PixelDecoder Role After Hunyuan

PixelDecoder should not be the final RGB generator once Hunyuan is integrated. Keep it for:

- Auxiliary visual grounding during world-model training.
- Fast debugging without running Hunyuan.
- Optional rough RGB condition for Hunyuan.

If ablations show rough RGB is not useful, freeze or remove PixelDecoder from the Hunyuan path and rely on depth plus token conditioning.

## 8. Evaluation

Short-horizon metrics:

- RGB L1.
- LPIPS.
- Motion-region L1.
- FVD or another video perceptual metric if practical.

Long-horizon rollout:

- Autoregressive wm3d_v3 token rollout.
- Hunyuan refinement per predicted future chunk.
- Full-task comparison GIFs.

Generalization checks:

- Held-out episodes.
- Held-out tasks.
- Held-out robot/dataset if possible.

Qualitative deliverables:

- Side-by-side GIFs: PixelDecoder rough RGB, Hunyuan RGB, GT RGB.
- Depth visualization next to generated RGB.
- Action-conditioned rollout sheets.

## 9. Risks

1. Validation leakage from window-level split. Fix with episode-level split before major claims.
2. Latent mismatch between VGGT tokens and Hunyuan latents. Use adapters before attempting direct latent alignment.
3. Rough RGB blur bias. Use condition dropout or weaker RGB control if Hunyuan copies blur.
4. Compute cost. Start with frozen Hunyuan plus LoRA/control adapter.
5. Missing task embeddings in full OXE cache. Backfill Qwen/task embeddings before large text-conditioned training.

## 10. Implementation Milestones

### Milestone A: Split and Baseline Cleanup

- Add episode-level train/val split support.
- Freeze current p64 wm3d_v3 checkpoint as the conditioning baseline.
- Produce a fixed validation clip list.

### Milestone B: Offline Condition Cache

Cache wm3d_v3 outputs for selected windows:

- `pred_tokens`
- `depth`
- `rough_rgb`
- action condition

This avoids running wm3d_v3 inside every Hunyuan training step during early experiments.

### Milestone C: Minimal Hunyuan Adapter

- Implement rough RGB + depth control adapter.
- Freeze Hunyuan base.
- Train adapter/LoRA on short horizon `k=8` clips.

### Milestone D: Token Conditioning

- Add `pred_tokens` projection to Hunyuan cross-attention or block-wise FiLM.
- Compare against rough RGB/depth-only control.

### Milestone E: Long Rollout Demo

- Use wm3d_v3 autoregressive token rollout.
- Use Hunyuan chunk refinement for each predicted future segment.
- Generate full-task comparison GIFs.

### Milestone F: Optional Joint Fine-Tuning

Only after adapter evidence is positive:

- Unfreeze small parts of wm3d_v3 or train a lightweight token adapter end-to-end.
- Keep Hunyuan base mostly frozen; expand LoRA rank if needed.

## 11. Recommended First Experiment

```text
Hunyuan frozen base + LoRA/control adapter
conditions = predicted depth + rough RGB
world model = frozen p64_138m_actioncond_full best.pt
target = future 8 RGB frames
split = episode-level heldout
```

Then add `pred_tokens` as a separate ablation. This gives the fastest answer to whether wm3d_v3 improves Hunyuan video generation beyond text/context alone.
