# tau0-style wm3d_v3 Roadmap

Date: 2026-06-01

Scope: keep the core of this project as a 3D-native, action-conditioned world model, while borrowing the strongest interface ideas from tau0-WM: shared video-action representation, action-conditioned simulation, dense progress evaluation, and proposal-evaluation-revision at inference time.

## 1. One-sentence Direction

wm3d_v3 should not become "a blurry RGB predictor plus a video upsampler." It should become a 3D-native action-conditioned simulator that produces structured controls for a high-fidelity video backend and, later, evaluates and improves robot actions.

```text
past RGB / VGGT tokens / Qwen task / robot state
        |
        v
3D World Core
  simulate(context, task, candidate_action)
    -> future VGGT tokens
    -> future depth
    -> motion/contact soft control
    -> progress/plausibility/reward
        |
        v
Video Backend: Hunyuan / Wan / tau0-compatible renderer
  condition = context RGB + depth + motion/contact + action + task + optional tokens
        |
        v
high-fidelity future RGB
        |
        v
VLA interface
  propose actions -> simulate/evaluate -> revise -> execute
```

## 2. Current Implementation Review

Current strongest checkpoint:

- `p64_context_motion`
- Checkpoint: `results/wm3d_v3_p64_140m_actioncond_context_motion/ckpt/best.pt`
- Base: `p64_context_renderer` warm-start plus new motion-control head.
- Data: `manifests/oxe_train.jsonl`, 21,942 clips, about 800k frames.
- Cache coverage on `oxe_train`: 100% for `vggt_pooled`, `vggt_geom`, `rgb_256`, `actions`, and `qwen_taskemb`.

Current model interface:

- `JointWorldModel` in `wm3d_v3/models/joint_model.py`
- Inputs:
  - past VGGT tokens `[B,T,P,D]`
  - Qwen task embedding `[B,2048]`
  - optional future action condition `[B,k,7]`
  - last context RGB for context renderer
- Outputs:
  - `pred_tokens`
  - `depth`
  - `pose`, `gripper_logit`
  - diagnostic `rgb`
  - `motion_hint`, `motion_logit`, `rgb_blend` when context-motion is enabled

Measured context-motion improvement over context-renderer on the same 440 validation windows:

| Model | RGB L1 | LPIPS | Motion L1 |
|---|---:|---:|---:|
| context-renderer | 0.02411 | 0.10488 | 0.05911 |
| context-motion | 0.02354 | 0.10238 | 0.05766 |

Motion hint diagnostic on the same quick split:

- threshold 0.5: precision 0.778, recall 0.785, IoU 0.641, F1 0.782.

Important interpretation:

- The change is directionally correct.
- The magnitude is small.
- Current RGB decoder should be treated as a diagnostic renderer, not the final RGB path.
- `motion_hint` is useful, but currently it is supervised by `GT RGB - context RGB`; it is closer to a dynamic-region mask than a clean object/contact mask.

## 3. What We Learn From tau0-WM

tau0-WM's useful lesson is not "use video diffusion everywhere." The useful lesson is interface design:

1. VAM: a policy-facing action generator that predicts executable action chunks while sharing predictive video representations.
2. ACVS: an action-conditioned simulator that predicts future outcomes and dense task progress for candidate actions.
3. Heterogeneous supervision masks: robot data supervises action and video; human/ego videos supervise visual future only; failure/progress data supervises task progress.
4. Test-time computation: propose multiple actions, evaluate/simulate consequences, revise before execution.

Mapping to wm3d_v3:

| tau0-WM concept | wm3d_v3 equivalent |
|---|---|
| shared video-action representation | VGGT 3D token dynamics + Qwen task + action tokens |
| video diffusion backbone | external Hunyuan/Wan backend, not the core world model |
| action-conditioned simulator | current `action_cond -> pred_tokens/depth/motion_hint` path |
| dense task-progress/reward | missing; add progress/plausibility head |
| VAM action proposer | missing; add `propose(context, task)` mode |
| heterogeneous supervision masks | needed for OXE + UMI + human video expansion |

## 4. Non-negotiable Project Core

Keep:

- VGGT-derived 3D token state as the main world representation.
- Qwen task embeddings/text as semantic condition.
- Robot action conditioning as a first-class input.
- Depth and motion/contact controls as structured physical outputs.
- Video backend as renderer/refiner, not as the only world state.

Do not:

- Over-invest in the current Conv/UNet RGB decoder as the final generator.
- Claim progress from random-window validation alone.
- Treat action reconstruction under teacher-forced future action as VLA capability.
- Couple the project permanently to one video backend. Hunyuan can be the first backend, but the interface should also allow Wan/tau0-style backends.

## 5. Two Interfaces We Need

### 5.1 ACVS / simulator, near-term

```python
simulate(
    context_tokens,
    task_embedding,
    candidate_action_chunk,
    context_rgb=None,
) -> {
    pred_tokens,
    depth,
    motion_hint,
    contact_hint,
    progress,
    plausibility,
    rough_rgb,
}
```

This is close to what we already have. It must become robust, action-sensitive, and measurable.

### 5.2 VAM / proposer, mid-term

```python
propose(
    context_tokens,
    task_embedding,
    robot_state=None,
    n_candidates=K,
) -> {
    action_chunks,
    action_logprob_or_score,
}
```

This does not exist yet. It is required before we can honestly say we have a tau0-like VLA.

## 6. Professional Benchmark Plan

Demo GIFs remain useful for debugging, but claims should be based on benchmark tables.

### 6.1 World-model prediction benchmarks

Use episode-level split only.

Metrics:

- VGGT token prediction:
  - MSE
  - cosine distance
  - rollout drift over 1, 2, 4, 8 chunks
- Depth:
  - AbsRel
  - RMSE
  - delta accuracy if scale is reliable
  - temporal depth consistency
- RGB diagnostic/rendered prediction:
  - L1
  - LPIPS
  - FVD on generated clips
  - temporal optical-flow consistency
- Motion/contact controls:
  - motion IoU, Dice/F1, precision/recall
  - contact-region L1
  - gripper/object-region metrics when masks are available or pseudo-labeled
- Long-horizon stability:
  - per-chunk degradation curve
  - token drift
  - depth drift
  - motion false-positive/false-negative growth

Required splits:

- random-window split: only for continuity with old numbers.
- episode-level in-dataset split: primary offline split.
- held-out dataset split: e.g. train on Bridge, validate Fractal, and vice versa.
- held-out task text split when enough task coverage exists.

### 6.2 Counterfactual action-sensitivity benchmarks

This is the key benchmark for proving "world model" rather than "video predictor."

For each validation window, compare:

- real action chunk
- zero action chunk
- shuffled action from another clip
- sign-flipped pose action
- scaled action magnitude
- gripper toggled

Metrics:

- action sensitivity score:
  - `||simulate(real_action) - simulate(shuffled_action)||`
  - measured in tokens, depth, motion, and rendered RGB
- action correctness gap:
  - prediction under real action should be closer to GT than prediction under counterfactual actions
- counterfactual ranking:
  - rank real action among K candidate actions by future prediction loss or progress score
  - report top-1, top-3, MRR, NDCG, AUC
- gripper/contact response:
  - contact/motion near gripper should change when gripper channel changes

This benchmark is more important than a prettier GIF.

### 6.3 Video backend benchmarks

For Hunyuan/Wan renderer experiments:

- compare backend conditions:
  1. context only
  2. context + depth
  3. context + depth + motion hint
  4. context + depth + motion hint + action
  5. context + depth + motion hint + action + pred_tokens
  6. optional rough RGB, with condition dropout
- report:
  - LPIPS
  - FVD
  - temporal consistency
  - motion-region L1
  - object/contact-region metrics
  - human inspection only as secondary evidence

Guardrail:

- Do not let rough RGB dominate, because it can copy blur from the diagnostic decoder.

### 6.4 VLA downstream benchmarks

Final proof should be closed-loop success rate, not only future-frame quality.

Recommended order:

1. LIBERO
   - Suites: Spatial, Object, Goal, Long / LIBERO-10.
   - Metric: official success rate over rollouts.
   - Reason: common VLA benchmark, cheap to run, fast iteration.
2. CALVIN
   - Metric: average completed instruction chain length / long-horizon success.
   - Reason: long-horizon language-conditioned manipulation.
3. SimplerEnv
   - Focus: Bridge/WidowX and Fractal/Google Robot.
   - Reason: closest public benchmark family to our current OXE data sources.
4. Real-robot micro-benchmark, if hardware is available
   - 5-10 tasks with strict initial-state protocol.
   - Report success rate, progress score, and failure mode taxonomy.

VLA baselines:

- behavior cloning / action head only
- current wm3d action head without simulator reranking
- proposer only
- proposer + ACVS ranking
- proposer + ACVS ranking + video backend
- proposer + ACVS ranking + revise loop

## 7. Experiment Roadmap

### Phase 0: Evaluation cleanup

Goal: make future results trustworthy.

Implementation:

- Add episode-level split builder:
  - `wm3d_v3/data/splits.py`
  - output `manifests/splits/oxe_episode_split_seed*.json`
- Update training/eval configs to accept explicit train/val clip ids.
- Add evaluation scripts:
  - `wm3d_v3/eval/action_sensitivity.py`
  - `wm3d_v3/eval/long_horizon_metrics.py`
  - `wm3d_v3/eval/motion_control_metrics.py`

Experiments:

- E0.1: current `p64_context_motion` on random-window split.
- E0.2: same checkpoint on episode-level split.
- E0.3: held-out dataset split, Bridge -> Fractal and Fractal -> Bridge.

Success criteria:

- We know how much old validation was inflated.
- We have action-sensitivity scores for current baseline.

### Phase 1: Independent ControlHead

Goal: move control signals out of the RGB decoder.

Implementation:

- Add `wm3d_v3/models/control_head.py`.
- Inputs:
  - `pred_tokens`
  - `depth`
  - `context_rgb`
  - `action_cond`
  - `task_emb`
- Outputs:
  - `motion_hint`
  - `contact_hint`
  - `occlusion_hint`
  - `control_confidence`
- Keep `ContextResidualPixelDecoder` only for diagnostic `rough_rgb`.

Training:

- Warm-start from `p64_context_motion`.
- Freeze or low-LR the dual-stream core for the first experiment.
- Train control head 8-12 epochs.

Losses:

- motion BCE/Dice from RGB-difference pseudo-label
- depth-edge consistency
- gripper-local motion/contact emphasis
- temporal smoothness with sharp changes allowed near action peaks
- optional pseudo-contact from gripper proximity if available

Success criteria:

- motion IoU > 0.641 on episode-level validation.
- action sensitivity improves over current checkpoint.
- long rollout drift does not worsen.

### Phase 2: Progress and plausibility head

Goal: make the simulator useful for evaluating candidate actions.

Implementation:

- Add `wm3d_v3/models/progress_head.py`.
- Outputs:
  - `progress[k]`
  - `terminal_success_logit`
  - `plausibility_logit`

Supervision:

- Positive: matched real context/action/future.
- Negative:
  - shuffled action with same context
  - shuffled future with same action
  - time-reversed future
  - wrong task text
  - large action perturbation
- Weak temporal ranking:
  - later successful frames should rank higher than earlier frames when task text implies progress.

Metrics:

- real-vs-negative AUC
- top-k real-action ranking among candidates
- correlation between progress and future prediction quality

Success criteria:

- real action ranks top-1 or top-3 among counterfactual candidates significantly above chance.
- progress score separates matched vs mismatched futures.

### Phase 3: Control bundle cache

Goal: decouple world-model compute from video-backend training.

Script:

- `scripts/cache_control_bundle.py`

Cache fields:

- `context_rgb`
- `target_rgb`
- `task_text`
- `task_emb`
- `action_tgt`
- `action_cond`
- `pred_tokens`
- `depth`
- `motion_hint`
- `contact_hint`
- `rough_rgb`
- `progress/plausibility`
- metadata: dataset, clip_id, start, split

Format:

- Prefer sharded `.npz` or WebDataset tar shards.
- Keep enough metadata to reproduce split and metrics.

Success criteria:

- Hunyuan/Wan adapter training can load bundles without running wm3d_v3 online.

### Phase 4: Video backend integration

Goal: use a pretrained video model as renderer/refiner.

Implementation:

- Add backend abstraction:
  - `wm3d_v3/video_backends/base.py`
  - `wm3d_v3/video_backends/hunyuan_i2v.py`
  - later `wm3d_v3/video_backends/wan_i2v.py`
- First run inference-only:
  - context image/video + task prompt -> generated future video.
- Then add adapter:
  - depth encoder
  - motion/contact encoder
  - action/task FiLM or cross-attn
  - optional token projection

Ablations:

1. context only
2. context + depth
3. context + depth + motion/contact
4. context + depth + motion/contact + action
5. context + depth + motion/contact + action + tokens
6. optional rough RGB with dropout

Success criteria:

- renderer improves LPIPS/FVD and object/contact-region metrics without reducing action sensitivity.
- generated video follows candidate action differences.

### Phase 5: VAM/action proposer

Goal: turn the model from only a simulator into a VLA-capable policy.

Implementation:

- Add `propose` mode in the action stream or a separate `ActionProposer`.
- Training modes:
  - action-conditioned simulation batch
  - action-masked proposal batch
  - mixed supervision masks, tau0-style
- Output:
  - continuous action chunk distribution
  - action confidence/logprob

Evaluation:

- behavior cloning action error
- closed-loop LIBERO success
- proposer-only vs proposer + ACVS reranking

Success criteria:

- proposer-only is competitive with BC baseline.
- proposer + ACVS ranking improves success rate over proposer-only.

### Phase 6: Test-time propose-evaluate-revise

Goal: reproduce the tau0-style benefit in our 3D-native architecture.

Loop:

1. propose K action chunks
2. simulate each candidate
3. rank by progress, plausibility, and visual consistency
4. optionally condition a second proposal on the best imagined future
5. execute the selected first action chunk

Metrics:

- success rate
- average number of simulated candidates
- latency
- improvement over single-sample policy

Success criteria:

- measurable success-rate gain on LIBERO/SimplerEnv or real-robot micro-benchmark.
- latency remains acceptable for chunked control.

## 8. Concrete Experiment Table

| ID | Goal | Model | Split | Main metric | Expected decision |
|---|---|---|---|---|---|
| E0.1 | continuity baseline | current `p64_context_motion` | random window | existing metrics | compare to old numbers |
| E0.2 | leakage check | current `p64_context_motion` | episode | metric drop | define honest baseline |
| E0.3 | dataset transfer | current `p64_context_motion` | Bridge/Fractal held-out | token/depth/motion | estimate generalization |
| E1.1 | independent control | frozen core + ControlHead | episode | motion IoU/action sensitivity | decide if control head works |
| E1.2 | unfreeze top dynamics | partial unfreeze + ControlHead | episode | long rollout drift | decide whether dynamics needs tuning |
| E2.1 | progress negatives | progress head | episode | real-vs-negative AUC | decide if ranking signal exists |
| E2.2 | candidate ranking | progress + simulator | episode | real action top-k/MRR | decide if ACVS is useful |
| E3.1 | cache bundles | best simulator | episode | throughput/integrity | unblock video backend |
| E4.1 | Hunyuan inference | backend only | fixed clips | qualitative + metrics | validate runtime |
| E4.2 | depth/motion adapter | frozen backend + adapter | episode | FVD/LPIPS/motion L1 | decide video-control value |
| E5.1 | VAM proposer | proposer | LIBERO | success rate | establish VLA baseline |
| E6.1 | reranking | proposer + ACVS | LIBERO/SimplerEnv | success-rate delta | prove world-model value |

## 9. Answers to the Two Questions

### Q1. Can we use a more professional benchmark than visual demo?

Yes. We should use three layers of benchmarks:

1. Offline world-model benchmark:
   - episode-level split
   - token/depth/RGB/motion metrics
   - long-horizon drift
   - action counterfactual sensitivity
2. Renderer benchmark:
   - FVD, LPIPS, temporal consistency, motion/contact-region metrics
   - condition ablations for depth, motion, action, tokens
3. Downstream VLA benchmark:
   - LIBERO success rate
   - CALVIN long-horizon score
   - SimplerEnv Bridge/Fractal because it matches our OXE data sources best

The most important new benchmark is action counterfactual sensitivity. If the model cannot distinguish real candidate actions from shuffled or perturbed actions, it is not a useful robot world model even if the GIF looks good.

### Q2. Can this become a VLA like tau0?

Yes, but not with the current checkpoint alone.

Current `p64_context_motion` is closer to an ACVS simulator:

```text
context + task + candidate future action -> imagined future
```

It is not yet a full VLA because the future action is teacher-forced during training and inference. To become VLA-capable, we must add:

1. an action proposer:
   - `context + task -> action chunk distribution`
2. an evaluator:
   - `context + task + candidate action -> future + progress/plausibility`
3. a test-time loop:
   - propose -> simulate/evaluate -> revise -> execute
4. closed-loop benchmarks:
   - LIBERO/CALVIN/SimplerEnv success rate

The advantage of our route over tau0 is that our predictive core is 3D-native through VGGT tokens and depth. The risk is that we currently have much less heterogeneous action/progress data. So the near-term target should be:

```text
3D-native ACVS first, VAM proposer second, video renderer third as a replaceable backend.
```

This preserves the center of the project while still learning the right tau0 interface.

## 10. Immediate Next Work

Start with these in order:

1. Implement episode-level split and fixed validation clip lists.
2. Implement action counterfactual sensitivity eval.
3. Implement independent `ControlHead`.
4. Train `p64_control_head` from `p64_context_motion`.
5. Add progress/plausibility head with shuffled-action negatives.
6. Build `control_bundle` cache.
7. Add video backend wrapper and run Hunyuan/Wan inference-only.
8. Train first depth+motion adapter.
9. Add action proposer and LIBERO closed-loop evaluation.

Do not start heavy Hunyuan adapter training before steps 1-3 are done; otherwise we will not know whether improvements come from the world model, the renderer, or validation leakage.

## 11. External References

- tau0-WM project: https://finch.agibot.com/research/tau0-wm
- tau0-WM code: https://github.com/sii-research/tau-0-wm
- tau0-WM Hugging Face: https://huggingface.co/sii-research/tau-0-wm
- VLA evaluation leaderboard: https://allenai.github.io/vla-evaluation-harness/leaderboard/
- CALVIN benchmark paper: https://arxiv.org/abs/2112.03227

