# WM3D Closed-Loop Status - 2026-06-02

## Current Best Scaffold

Architecture sketch generated for the current design discussion:

```text
report_assets/wm3d_current_next_architecture_imagen.png
```

## Closed-Loop Gate Update - 2026-06-03

Current status: the WM3D scaffold is not complete enough for formal full
training. The world core, action policy heads, optional RGB/Hunyuan path, and
LIBERO runner/harness exist, but the learned closed-loop policy still fails the
minimal hdf5-init task-1 gate.

Comparable gate settings:

```text
camera_size=128
warmup_steps=0
context_T=16
action_history_len=16
send_lowdim=true
send_object_state=true
send_plan_state=true
plan_state_dim=17
send_progress=true
trace_object_state=true
```

Latest gate results:

| candidate | result |
|---|---|
| v18 eval-step sweep 100..800 | no success; best stage_score 0.75, butter not placed |
| v20 stage3-waypoint eval-step sweep 100..800 | no success; best stage_score 0.75, butter xy about 0.233 |
| v16b force stage3 gripper closed | no success; stage_score 0.75, butter unchanged |
| v16b expert suffix from step190 | no success; stage_score 0.5 |
| v16b expert suffix from step160 | no success; stage_score 0.5 |
| v16b terminal NN reference | no success; stage_score 0.75 |
| v16b terminal linear reference | no success; stage_score 0.75 |

Artifacts:

```text
results/wm3d_v18_ckpt_sweep_cam128_warm0/
results/wm3d_v20_ckpt_sweep_cam128_warm0/
results/wm3d_v16b_force_stage3_grip_task1_cam128_warm0/
results/wm3d_v16b_expert_suffix190_task1_cam128_warm0/
results/wm3d_v16b_expert_suffix160_task1_cam128_warm0/
results/wm3d_v16b_direct_terminal_nn_task1_cam128_warm0/
results/wm3d_v16b_direct_terminal_linear_task1_cam128_warm0/
```

Interpretation:

```text
The failure is not caused by video generation, Hunyuan, or gripper convention
alone. It is a closed-loop action/transport problem around the second object.
Open-loop hdf5 expert suffixes do not recover once v16b has drifted, so the next
mainline step must be online teacher/intervention data or a stronger
closed-loop planner/selector. More local residual or waypoint-only training is
unlikely to move the architecture forward.
```

New utility added:

```text
wm3d_v3/benchmarks/libero_remote_runner.py
  --expert_action_override_from_step

wm3d_v3/policy/world_model_policy.py
wm3d_v3/policy/token_policy.py
wm3d_v3/policy/http_policy_server.py
  selection_mode=direct_terminal_linear
```

Tests:

```text
pytest tests/test_control_progress_heads.py -q
11 passed

pytest tests/test_libero_object_contact_eval.py tests/test_control_progress_heads.py -q
14 passed
```

Current best P0 closed-loop scaffold:

```text
configs/v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce.yaml
results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/ckpt/best.pt
```

This checkpoint is the first one where both offline TTC ranking and the policy
probe choose actions that are better than the anchor candidate under the current
offline action-error metric.

| Metric | pairwise | candrank | candrank CE |
|---|---:|---:|---:|
| world `L_state_mse` | 0.0167275 | 0.0167275 | 0.0167275 |
| world `rgb_L1` | 0.0161417 | 0.0161417 | 0.0161417 |
| progress abs err | 0.0884349 | 0.0590709 | 0.0698882 |
| TTC anchor pose L1 | 0.2574103 | 0.2582230 | 0.2555776 |
| TTC ranked pose L1 | 0.2723065 | 0.2609601 | 0.2535250 |
| TTC ranked - anchor | +0.0148962 | +0.0027371 | -0.0020526 |
| TTC oracle match | 0.5125 | 0.5828 | 0.7141 |
| policy anchor pose L1 | 0.2586655 | 0.2590892 | 0.2569799 |
| policy selected pose L1 | 0.2752191 | 0.2603165 | 0.2561648 |
| policy selected - anchor | +0.0165536 | +0.0012273 | -0.0008151 |
| policy oracle match | 0.5969 | 0.7719 | 0.9305 |

Interpretation: CE improved the selector/ranker enough to pass the current
offline closed-loop gate. It did not improve the candidate distribution itself:
the best available candidate got worse than candrank. The next useful work is
therefore better evaluator/success supervision and benchmark integration, not
more local RGB decoder tuning.

## Direct Action Policy Stage - 2026-06-02

Why this stage was added:

```text
expert hdf5 action replay succeeds from the same LIBERO init state,
but WM3D proposer/anchor rollouts fail from that init state.
```

So the current bottleneck is closed-loop action policy quality, not LIBERO
environment wiring and not the terminal success API.

New direct policy path:

```text
wm3d_v3/models/action_policy.py
wm3d_v3/training/train_libero_action_policy.py
configs/v3_p64_140m_actioncond_context_motion_p0_heads_direct_policy.yaml
configs/libero_action_policy_direct_partial4_dense_v1.yaml
```

The new `ActionChunkPolicy` is separate from the proposer/ranker path. It keeps
the WM3D world backbone/checkpoint structure, but adds a direct BC head:

```text
context VGGT tokens + Qwen task embedding
  -> ActionChunkPolicy
  -> direct [8, 7] action chunk
```

Inference now supports:

```text
selection_mode=ranked  # old proposer -> simulate -> rank path
selection_mode=anchor  # old candidate-0 path
selection_mode=direct  # new direct action policy path
```

Adapter fixes made during this stage:

```text
LIBERO gripper mapping: policy closed01 -> env -1/1
hdf5-init rollout default warmup: 0 steps
trace now records policy_action, env action, policy_gripper, env_gripper
```

Dense expert cache:

```text
results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/system_eval_best/libero_expert_cache_partial4_2048/
windows: 2048
tasks: 4 completed LIBERO hdf5 tasks
```

Direct policy training:

```text
checkpoint best:
  results/wm3d_libero_action_policy_direct_partial4_dense_v1/ckpt/best.pt
  step: 300
  val L_total: 1.0661
  val policy_first_pose_l1: 0.4687
  val policy_grip_acc: 0.9018

checkpoint latest:
  results/wm3d_libero_action_policy_direct_partial4_dense_v1/ckpt/latest.pt
  step: 1000
  val L_total: 1.4776
  val policy_first_pose_l1: 0.4715
  val policy_grip_acc: 0.9248
```

Interpretation: the small 4-task dense cache overfits after about step 300.
The train loss keeps falling, but validation total gets worse.

Strict hdf5-init closed-loop tests:

```text
task: LIBERO task 1
demo: demo_0
init state: hdf5 expert init_state
warmup_steps: 0
gripper_mode: closed01_to_libero
max_steps: 300
```

Results:

| checkpoint | success | reward steps | mean action norm | artifact |
|---|---:|---:|---:|---|
| direct best snapshot | 0 / 1 | 0 | 1.0438 | `results/wm3d_libero_action_policy_direct_partial4_dense_v1/libero_remote_rollout_hdf5init_task1_demo0_direct_snapshot_300step_summary.json` |
| direct latest | 0 / 1 | 0 | 1.1076 | `results/wm3d_libero_action_policy_direct_partial4_dense_v1/libero_remote_rollout_hdf5init_task1_demo0_direct_latest_300step_summary.json` |

Expert action alignment diagnostics:

| checkpoint | pose L1 first10 | pose L1 first50 | pose L1 mean | grip match first50 | grip match mean |
|---|---:|---:|---:|---:|---:|
| direct best snapshot | 0.1231 | 0.2380 | 0.2060 | 1.0000 | 0.4729 |
| direct latest | 0.0781 | 0.1707 | 0.2122 | 1.0000 | 0.5698 |

Artifacts:

```text
results/wm3d_libero_action_policy_direct_partial4_dense_v1/hdf5init_direct_snapshot_expert_action_compare.json
results/wm3d_libero_action_policy_direct_partial4_dense_v1/hdf5init_direct_latest_expert_action_compare.json
```

Important conclusion:

```text
Fixing gripper convention and exact hdf5 init is not enough.
The direct BC head improves first-action imitation but still cannot maintain a
successful long-horizon closed-loop trajectory.
```

The next action-policy stage should not be more RGB decoder work. It should be
a stronger closed-loop BC policy stage:

```text
1. train on full continuous LIBERO expert trajectories, not sparse sampled windows;
2. add action-history / proprio / ee-state conditioning where available;
3. train with receding-horizon chunk execution and scheduled action dropout;
4. evaluate first on hdf5-init expert-covered tasks, then on suite init states;
5. only after action success improves, reconnect evaluator/ranker and Hunyuan RGB decoder.
```

## Start-Padded Lowdim/History Policy Stage - 2026-06-02

Root-cause update:

```text
The previous expert cache did not train episode-start decisions.
First existing window: frames[0:16] -> actions[16:24]
Online hdf5-init first decision: first_frame repeated 16x -> action[0]
```

This mismatch explains why direct BC could look plausible offline yet fail from
the real initial state.

New data/model path:

```text
wm3d_v3/benchmarks/libero_start_windows.py
wm3d_v3/benchmarks/libero_expert_cache.py  # supports negative context_start padding
wm3d_v3/models/action_policy.py           # optional lowdim + action_history tokens
wm3d_v3/benchmarks/libero_remote_runner.py # sends lowdim/action_history online
```

Policy conditioning added:

```text
visual context: VGGT tokens from 16 RGB frames
task: Qwen3-VL task embedding
lowdim_state: 12D = eef_pos(3) + gripper_qpos(2) + joint_pos(7)
action_history: previous 16 raw 7D env actions
```

Single-demo overfit test:

```text
source demo:
  LIBERO task 1 demo_0
  LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket

start-padded windows:
  results/wm3d_libero_action_policy_direct_partial4_dense_v1/task1_demo0_start_stride1.jsonl
  windows: 251

token cache:
  results/wm3d_libero_action_policy_direct_partial4_dense_v1/task1_demo0_start_stride1_cache/

config:
  configs/libero_action_policy_lowdimhist_task1_demo0_overfit_v1.yaml
```

Closed-loop result:

| checkpoint | step | hdf5-init success | success step | artifact |
|---|---:|---:|---:|---|
| lowdimhist step100 snapshot | 100 | 1 / 1 | 238 | `results/wm3d_libero_action_policy_lowdimhist_task1_demo0_overfit_v1/libero_remote_rollout_hdf5init_task1_demo0_lowdimhist_step100ish_300step_summary.json` |
| lowdimhist offline-best | 700 | 0 / 1 | n/a | `results/wm3d_libero_action_policy_lowdimhist_task1_demo0_overfit_v1/libero_remote_rollout_hdf5init_task1_demo0_lowdimhist_best_300step_summary.json` |

The successful checkpoint is:

```text
results/wm3d_libero_action_policy_lowdimhist_task1_demo0_overfit_v1/ckpt/best_rollout_snapshot_step100ish.pt
```

Key interpretation:

```text
This is the first real closed-loop WM3D/VLA-style success in LIBERO.
The framework can execute a task when trained on start-aligned windows with
lowdim state and action history.
```

Important caveat:

```text
The offline-best checkpoint has lower action L1 but fails closed-loop.
Therefore checkpoint selection cannot rely on cached action L1 alone.
The next gate must include rollout success or a rollout-correlated metric.
```

Next scaling step started:

```text
partial4 start-padded JSONL:
  results/wm3d_libero_action_policy_lowdimhist_task1_demo0_overfit_v1/libero_partial4_start_stride4.jsonl
  demos: 200
  windows: 11647

cache process:
  pid file: /data/Minko/logs/libero_partial4_start_stride4_lowdimhist_cache.pid
  log: /data/Minko/logs/libero_partial4_start_stride4_lowdimhist_cache.log
  output dir: results/wm3d_libero_action_policy_lowdimhist_partial4_start_stride4_v1_cache/

training config prepared:
  configs/libero_action_policy_lowdimhist_partial4_start_stride4_v1.yaml
```

## What The Model Does Now

The implemented loop is:

```text
cached observation tokens + Qwen task embedding
  -> WM3D transformer world core
  -> action proposer generates K action chunks
  -> world core imagines each candidate
  -> progress/terminal/plausibility heads score the imagined futures
  -> policy returns selected 7D action chunk
```

The action-facing API is implemented in:

```text
wm3d_v3/policy/token_policy.py
wm3d_v3/policy/world_model_policy.py
```

The policy can output a 7D robot action:

```text
6D delta pose + gripper_closed
```

with pose denormalized through the checkpoint action-stat buffers when available.

## What It Still Cannot Claim

This is not yet a complete VLA. It does not yet have all of:

1. online RGB/depth observation -> VGGT/Qwen tokenization inside an external env,
2. benchmark-specific action conversion and gripper convention handling,
3. real simulator/robot execution,
4. true success/failure labels from the environment,
5. real benchmark success rate.

The system report correctly marks professional benchmarks unavailable:

```text
LIBERO: unavailable
CALVIN: unavailable
SimplerEnv: unavailable
WorldArena: unavailable
```

LIBERO/CALVIN are the better next professional benchmark targets for robotic
manipulation. WorldArena-style tasks can be useful later, but they are less
direct evidence of robot VLA competence.

## Benchmark Loop Progress

The benchmark runner now supports:

```text
mock
offline_replay
libero  # in-process adapter, requires LIBERO runtime dependencies
```

`offline_replay` exercises the same adapter/policy path on real cached OXE
validation windows. It is not a simulator success benchmark, but it verifies
that the policy runner can consume real WM3D tokens and emit correctly scaled
actions.

Latest offline replay run:

```text
adapter: offline_replay
checkpoint: candrank CE best
tasks: 80 OXE val windows
pose_l1_norm: 0.2287380
pose_l1_raw: 0.0139379
grip_match: 0.8125
success_rate: 0.9375  # thresholded offline first-action proxy, not env success
```

Output artifact:

```text
results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/system_eval_best/offline_replay_adapter_80.json
```

LIBERO progress:

```text
official source: /data/Minko/benchmarks/LIBERO
official commit: 8f1084e
isolated runtime: /data/Minko/.conda-envs/libero-py38
probe artifact: results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/system_eval_best/libero_probe.json
task API: available
suite: libero_10
num_tasks: 10
first task: put both the alphabet soup and the tomato sauce in the basket
env API: available in the isolated LIBERO runtime
```

The current WM3D venv still does not import the LIBERO runtime, by design.
LIBERO runs in Python 3.8.13 with `robosuite==1.4.0`, `robomimic==0.2.0`,
`bddl==1.0.1`, and `torch==1.11.0+cu113`. Xvfb is installed and robosuite is
configured with `MUJOCO_GPU_RENDERING=False`, so LIBERO runs with
`xvfb-run ... MUJOCO_GL=glfw`.

The project now has a cross-environment bridge:

```text
wm3d_v3/policy/http_policy_server.py
wm3d_v3/benchmarks/libero_remote_runner.py
scripts/setup_libero_micromamba_env.sh
scripts/run_libero_remote_smoke.sh
```

Intended run shape:

```text
WM3D Python 3.10 env:
  start policy HTTP server with VGGT + Qwen + WM3D checkpoint

LIBERO Python 3.8 env:
  run LIBERO env and call /act on the WM3D policy server
```

This avoids forcing old LIBERO simulator dependencies into the current training
environment.

Real LIBERO remote-loop smoke has run:

```text
artifact: results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/system_eval_best/libero_remote_rollout_1step.json
task: libero_10 task 0, init state 0
steps: 1
success_rate: 0.0
seconds: 11.52

artifact: results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/system_eval_best/libero_remote_rollout_20step.json
task: libero_10 task 0, init state 0
steps: 20
success_rate: 0.0
seconds: 14.45

artifact: results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/system_eval_best/libero_remote_rollout_100step.json
task: libero_10 task 0, init state 0
steps: 100
success_rate: 0.0
seconds: 48.08

artifact: results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/system_eval_best/libero_probe_py38.json
task API: true
env API: true

artifact: results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/system_eval_best/libero_remote_rollout_trace_5step.json
steps: 5
contains: per-step selected 7D action, action norm, reward, done, success
```

New trace-data path, added after the first remote-loop smoke:

```text
artifact: results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/system_eval_best/libero_remote_rollout_tasks2_30step_trace_v2.json
summary:  results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/system_eval_best/libero_remote_rollout_tasks2_30step_trace_v2_summary.json
jsonl:    results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/system_eval_best/libero_remote_rollout_tasks2_30step_trace_v2.jsonl
frames:   results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/system_eval_best/libero_frames_tasks2_30step_trace_v2/

episodes: 2
trace_steps: 60
success_rate: 0.0
candidate_score_steps: 60
action_chunk_steps: 60
frame_steps: 6
mean_action_norm: 1.0011422
nonzero_reward_steps: 0
```

This v2 trace format records the policy ranking signal for every environment
step:

```text
selected_idx
selected_score
candidate_scores
first 7D action
full action_chunk_raw
optional frame_path
reward / done / success
```

The important interpretation is not that the current policy is good. It is not:
the two-task LIBERO sample is still zero-success and zero-reward. The important
engineering gate is that benchmark rollouts can now be converted into a stable
training/debug format:

```text
wm3d_v3/benchmarks/libero_trace_summary.py
scripts/run_libero_remote_smoke.sh  # now also writes summary + JSONL
```

Current training signal from this rollout:

```text
benchmark_feedback: true
action_trace: true
policy_candidate_scores: true
action_chunks: true
frame_retokenization: true
failure_only_supervision: true
binary_success_supervision: false
```

So this is usable for negative mining and policy diagnostics now. It is not yet
enough for strong evaluator/proposer learning because there are no successful
LIBERO trajectories in the sampled trace.

Positive LIBERO expert-data path is now partially available:

```text
dataset dir: /data/Minko/benchmarks/LIBERO/datasets/libero_10
download status: 4 / 10 hdf5 files complete
disk used: 4.8G
source: yifengzhu-hf/LIBERO-datasets, folder libero_10
```

Completed hdf5 files:

```text
KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5
LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo.hdf5
LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5
STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5
```

Each completed file has 50 expert demonstrations, 7D actions, and RGB
observations:

```text
obs keys include: agentview_rgb, eye_in_hand_rgb, ee_pos, ee_ori, ee_states,
                  gripper_states, joint_states
action shape: [episode_len, 7]
```

Robomimic/LIBERO dataset reader check passed on the first completed file:

```text
SequenceDataset len: 12434
n_demos: 50
sample action shape: [10, 7]
```

Expert demonstration JSONL export is implemented and verified:

```text
wm3d_v3/benchmarks/libero_demo_export.py

summary: results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/system_eval_best/libero10_expert_windows_partial4_summary.json
jsonl:   results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/system_eval_best/libero10_expert_windows_partial4.jsonl

files: 4
demos: 200
windows: 5466
T: 16
k: 8
stride: 8
positive_success_windows: 5466
```

The expert JSONL is compact: it stores `hdf5_path`, `demo_id`,
`context_start`, `target_start`, `action_chunk`, `terminal_success_tgt=1`, and
`benchmark_success=true`. It does not duplicate RGB; frames are referenced in
the source hdf5 and can be re-tokenized by the next training dataset.

## LIBERO Mixed-Supervision Smoke

The first real benchmark-supervision training path now runs end to end:

```text
LIBERO expert hdf5
  -> expert JSONL windows
  -> VGGT/Qwen cached WM3D token windows
  -> progress/evaluator + action_proposer fine-tune
  -> checkpoint with LIBERO action stats

LIBERO policy failure rollout
  -> per-step frame/action trace
  -> VGGT/Qwen cached negative windows
  -> terminal/evaluator negative supervision
```

New cache/training files:

```text
wm3d_v3/benchmarks/libero_expert_cache.py
wm3d_v3/benchmarks/libero_rollout_cache.py
wm3d_v3/training/train_libero_success_p0.py
configs/libero_success_p0_smoke.yaml
configs/libero_success_p0_mixed_smoke.yaml
```

Expert cache smoke:

```text
cache dir: results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/system_eval_best/libero_expert_cache_smoke
windows: 16
token shape: [16, 64, 2048]
action shape: [8, 7]
LIBERO action stats:
  mean: [-0.0051, 0.0617, -0.0770, 0.0045, 0.0020, -0.0275]
  std:  [0.3004, 0.3418, 0.3946, 0.0393, 0.0532, 0.0712]
  gripper pos_rate: 0.5454
```

Failure cache smoke:

```text
rollout: results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/system_eval_best/libero_remote_rollout_task1_12step_frames_v3.json
cache dir: results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/system_eval_best/libero_failure_cache_smoke
windows: 12
terminal_success_tgt: 0
proposer_weight: 0
```

Two training smokes were run:

```text
positive-only checkpoint:
results/wm3d_libero_success_p0_smoke_v1/ckpt/best.pt

mixed positive+failure checkpoint:
results/wm3d_libero_success_p0_mixed_smoke_v1/ckpt/best.pt
```

Evaluation on the same 16 positive + 12 negative cached windows:

| Checkpoint | terminal positive mean | terminal negative mean | gap | proposer pose |
|---|---:|---:|---:|---:|
| base + LIBERO stats | 0.547 | 0.945 | -0.398 | 0.195 |
| positive-only smoke | 0.997 | 0.997 | -0.001 | 0.111 |
| mixed smoke | 0.836 | 0.008 | 0.828 | 0.115 |

Interpretation:

1. The base checkpoint has the wrong benchmark success semantics on LIBERO:
   it rates failure rollout actions higher than expert actions.
2. Positive-only fine-tuning improves proposer/action imitation but collapses
   terminal success to "everything succeeds".
3. Mixed expert-positive + rollout-failure supervision fixes the key evaluator
   semantics in this smoke: expert windows score high, failed policy windows
   score low, and proposer pose fit still improves over base.

This is the first verified benchmark-supervision loop. It is still a smoke, not
a final VLA policy: only 16 expert windows and 12 failure windows were cached,
and no broad LIBERO success-rate improvement has been claimed yet.

## Partial4 Mixed Stage

The smoke was scaled to a larger partial LIBERO-10 stage using the four
available expert hdf5 files:

```text
expert cache:
results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/system_eval_best/libero_expert_cache_partial4_256
windows: 256
size: 1.1G

failure rollout:
results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/system_eval_best/libero_remote_rollout_tasks3_40step_frames_v4.json
episodes: 3
trace_steps: 120
success_rate: 0.0

failure cache:
results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/system_eval_best/libero_failure_cache_tasks3_120
windows: 120
size: 527M
```

Training config and checkpoint:

```text
configs/libero_success_p0_partial4_mixed_v1.yaml
results/wm3d_libero_success_p0_partial4_mixed_v1/ckpt/best.pt
```

Training kept the VGGT-native world core frozen and trained only:

```text
progress_head.
action_proposer.
```

Evaluation on 256 positive + 120 negative cached windows:

| Checkpoint | terminal positive mean | terminal negative mean | gap | proposer pose |
|---|---:|---:|---:|---:|
| base + LIBERO stats | 0.448 | 0.936 | -0.489 | 0.299 |
| mixed smoke | 0.778 | 0.019 | 0.759 | 0.246 |
| partial4 mixed v1 | 0.999 | 0.0006 | 0.999 | 0.167 |

This confirms the partial4 stage learned the cached benchmark success semantics
and improved expert-action imitation on the cached data.

Real LIBERO rollouts are still zero-success:

```text
checkpoint: results/wm3d_libero_success_p0_partial4_mixed_v1/ckpt/best.pt

rollout: results/wm3d_libero_success_p0_partial4_mixed_v1/libero_remote_rollout_tasks3_100step.json
tasks: 0, 1, 2
episodes: 3
steps: 300
success_rate: 0.0

rollout: results/wm3d_libero_success_p0_partial4_mixed_v1/libero_remote_rollout_expert_tasks_1_3_5_6_100step.json
tasks: 1, 3, 5, 6  # all have downloaded expert hdf5 files
episodes: 4
steps: 400
success_rate: 0.0
```

Action diagnostics:

```text
results/wm3d_libero_success_p0_partial4_mixed_v1/online_rollout_action_diagnostics.json

expert-covered rollout mean action norm: 0.813
expert-covered rollout gripper mean: 0.7175
nonzero reward steps: 0
```

Expert replay sanity check passed:

```text
wm3d_v3/benchmarks/libero_expert_replay.py
artifact: results/wm3d_libero_success_p0_partial4_mixed_v1/expert_replay_task1_demo0.json
task: LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket
demo: demo_0
success: true
success_at: 247 steps
```

Interpretation:

1. LIBERO env/action-space wiring is valid; expert hdf5 actions replay to
   success in the same simulator.
2. The current WM3D policy now learns cached success/failure semantics, but this
   does not yet produce a successful closed-loop policy.
3. The next bottleneck is action policy quality over long horizons: we need
   stronger behavior-cloning / receding-horizon action supervision and likely
   online rollout aggregation, not only terminal-score separation.

The zero success rate is not surprising. The current action head is still trained
from offline pseudo targets and OXE action scale, not from LIBERO success
rollouts. The important gate now passed is the real loop itself:

```text
LIBERO env observation -> WM3D HTTP policy -> selected 7D action -> LIBERO env step
```

## Next Engineering Priorities

1. Finish `libero_10` download or continue with the current partial expert set
   for a larger action-policy stage.

2. Add an action-policy stage that explicitly optimizes first-action and
   receding-horizon action imitation on expert rollouts, then evaluates on
   expert-covered tasks before broad LIBERO evaluation.

3. Keep mixed success/failure terminal supervision, but do not treat cached
   terminal separation as sufficient for VLA success. The next gate is real
   nonzero reward or success on expert-covered LIBERO tasks.

4. Train a real P0-success/proposer stage on the larger mixed LIBERO data, then
   run broader LIBERO evaluation: more tasks,
   multiple init states, and official horizon once runtime cost is acceptable.

4. Keep Hunyuan/Wan as the future RGB renderer/refiner path. It should improve
   visual output quality, but it should not replace the VGGT-native world core.

5. After the real benchmark loop works, fine-tune proposer/evaluator on target
   task data so the system becomes a VLA-style action policy instead of only an
   offline world-model scaffold.

## Partial4 and Task1-Only Gate Update - 2026-06-02 17:25 CST

Architecture discussion asset:

```text
report_assets/wm3d_architecture_current_next_2026-06-02.png
```

Training script update:

```text
wm3d_v3/training/train_libero_action_policy.py
```

The direct action-policy trainer now supports:

```yaml
train:
  keep_eval_ckpts: true
  eval_ckpt_dir: ckpt/eval_steps
```

This is required because the single-demo experiment already showed that
offline-best action L1 can fail closed-loop while an earlier rollout snapshot
succeeds. The training loop must preserve eval-time checkpoints for rollout
selection.

### Partial4 Start-Padded Lowdim/History Result

Dataset:

```text
results/wm3d_libero_action_policy_lowdimhist_task1_demo0_overfit_v1/libero_partial4_start_stride4.jsonl
demos: 200
tasks: 4
windows: 11647
```

Sharded cache:

```text
results/wm3d_libero_action_policy_lowdimhist_partial4_start_stride4_sharded_v1_cache/
total size: about 50G
```

Training:

```text
config: configs/libero_action_policy_lowdimhist_partial4_start_stride4_sharded_v1.yaml
output: results/wm3d_libero_action_policy_lowdimhist_partial4_start_stride4_sharded_v1/
best checkpoint: ckpt/best.pt
best step: 1500
best val L_total: 0.2588577
```

Strict hdf5-init closed-loop gate on task1/demo0:

| checkpoint | rollout success | pose L1 first10 | pose L1 first50 | pose L1 mean | grip switch |
|---|---:|---:|---:|---:|---|
| step250ish | 0 / 1 | 0.1014 | 0.0944 | 0.2535 | not decisive |
| step1250ish | 0 / 1 | 0.0995 | 0.1965 | 0.1931 | policy 45 vs expert 50 |
| step1500/best | 0 / 1 | 0.0824 | 0.1182 | 0.2219 | policy 45 vs expert 50 |

Artifacts:

```text
results/wm3d_libero_action_policy_lowdimhist_partial4_start_stride4_sharded_v1/libero_remote_rollout_hdf5init_task1_demo0_best_300step_summary.json
results/wm3d_libero_action_policy_lowdimhist_partial4_start_stride4_sharded_v1/hdf5init_best_expert_action_compare.json
```

Interpretation:

```text
The 4-task start-padded lowdim/history policy improves early pose imitation but
still has zero closed-loop success. Offline validation cannot select a useful
closed-loop checkpoint.
```

### Task1-Only Control Experiment

Purpose:

```text
Determine whether partial4 failure is mostly cross-task/task-conditioning
interference, or whether the current direct BC head is not enough even for
50 demos of one task.
```

Filtered cache manifest:

```text
results/wm3d_libero_action_policy_lowdimhist_task1_start_stride4_from_partial4_v1_cache/manifest.jsonl
task: LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket
demos: 50
windows: 3186
instruction: put both the cream cheese box and the butter in the basket
```

Config:

```text
configs/libero_action_policy_lowdimhist_task1_start_stride4_from_partial4_v1.yaml
```

Training result:

| checkpoint | step | val L_total | first pose L1 | pose L1 | grip acc | first grip loss |
|---|---:|---:|---:|---:|---:|---:|
| best | 1000 | 0.3548 | 0.1802 | 0.3051 | 0.9567 | 0.0622 |
| latest | 1200 | 0.3834 | 0.1781 | 0.3026 | 0.9577 | 0.0912 |

Closed-loop hdf5-init gates:

| checkpoint | rollout success | pose L1 first10 | pose L1 first50 | pose L1 mean | grip switch |
|---|---:|---:|---:|---:|---|
| step100 | 0 / 1 | 0.0650 | 0.1418 | 0.2177 | never switched |
| step500 | 0 / 1 | 0.0895 | 0.1956 | 0.2503 | policy 37 vs expert 50 |
| best/step1000 | 0 / 1 | 0.0702 | 0.1593 | 0.2184 | policy 68 vs expert 50 |

Artifacts:

```text
results/wm3d_libero_action_policy_lowdimhist_task1_start_stride4_from_partial4_v1/libero_remote_rollout_hdf5init_task1_demo0_step100_300step_summary.json
results/wm3d_libero_action_policy_lowdimhist_task1_start_stride4_from_partial4_v1/libero_remote_rollout_hdf5init_task1_demo0_step500_300step_summary.json
results/wm3d_libero_action_policy_lowdimhist_task1_start_stride4_from_partial4_v1/libero_remote_rollout_hdf5init_task1_demo0_best_300step_summary.json
results/wm3d_libero_action_policy_lowdimhist_task1_start_stride4_from_partial4_v1/hdf5init_step100_expert_action_compare.json
results/wm3d_libero_action_policy_lowdimhist_task1_start_stride4_from_partial4_v1/hdf5init_step500_expert_action_compare.json
results/wm3d_libero_action_policy_lowdimhist_task1_start_stride4_from_partial4_v1/hdf5init_best_expert_action_compare.json
```

Conclusion:

```text
The problem is not only cross-task language conditioning. Even single-task
50-demo BC with VGGT tokens + Qwen task embedding + lowdim state + action
history fails the strict hdf5-init closed-loop gate.
```

The single-demo overfit success remains important because it proves the runtime
loop, action-space conversion, and policy API can solve the task. The 50-demo
and 4-task failures show the missing piece is the policy/training objective:
the current direct BC head does not model task phase and gripper timing robustly
enough for long-horizon closed-loop execution.

### Next Required Change

The next aligned step is not more vanilla BC training. The next model/training
stage should add rollout-correlated control supervision:

```text
1. Add progress/phase targets from expert trajectories.
2. Train a gripper/event head separately from continuous pose regression.
3. Add rollout checkpoint gates as first-class selection criteria.
4. Use failed rollout traces for DAgger-style hard negative/state recovery data.
5. Keep the 3D-native VGGT/Qwen/lowdim/action-history interface unchanged so
   the later Hunyuan RGB decoder and VLA fine-tuning path remain compatible.
```

### Grip-Transition Loss Probe - 2026-06-02 17:45 CST

Code changes:

```text
wm3d_v3/training/train_libero_success_p0.py
  - exposes progress_tgt from target_start / episode_len when manifest provides it

wm3d_v3/training/train_libero_action_policy.py
  - adds optional grip_transition_weight
  - reports policy_grip_transition_acc and policy_grip_transition_rate
  - keeps old behavior when grip_transition_weight is 0
```

Config:

```text
configs/libero_action_policy_lowdimhist_task1_start_stride4_griptransition_v1.yaml
```

Training result:

| checkpoint | step | val L_total | first pose L1 | pose L1 | grip acc | transition acc |
|---|---:|---:|---:|---:|---:|---:|
| step500 | 500 | 0.4951 | 0.2111 | 0.3721 | 0.9345 | 0.7097 |
| best | 700 | 0.4568 | 0.1988 | 0.3301 | 0.9508 | 0.7419 |
| step800 | 800 | 0.4626 | 0.1865 | 0.3185 | 0.9537 | 0.7742 |
| latest | 1000 | 0.4836 | 0.1842 | 0.3145 | 0.9537 | 0.8065 |

Closed-loop gates:

| checkpoint | rollout success | pose L1 first10 | pose L1 first50 | pose L1 mean | grip switch |
|---|---:|---:|---:|---:|---|
| step500 | 0 / 1 | 0.1000 | 0.1341 | 0.2478 | policy 48 vs expert 50 |
| best/step700 | 0 / 1 | 0.1127 | 0.1733 | 0.2192 | never switched |
| step800 | 0 / 1 | 0.0903 | 0.1340 | 0.1845 | policy 46 vs expert 50 |

Artifacts:

```text
results/wm3d_libero_action_policy_lowdimhist_task1_start_stride4_griptransition_v1/libero_remote_rollout_hdf5init_task1_demo0_step500_300step_summary.json
results/wm3d_libero_action_policy_lowdimhist_task1_start_stride4_griptransition_v1/libero_remote_rollout_hdf5init_task1_demo0_best_300step_summary.json
results/wm3d_libero_action_policy_lowdimhist_task1_start_stride4_griptransition_v1/libero_remote_rollout_hdf5init_task1_demo0_step800_300step_summary.json
results/wm3d_libero_action_policy_lowdimhist_task1_start_stride4_griptransition_v1/hdf5init_step500_expert_action_compare.json
results/wm3d_libero_action_policy_lowdimhist_task1_start_stride4_griptransition_v1/hdf5init_best_expert_action_compare.json
results/wm3d_libero_action_policy_lowdimhist_task1_start_stride4_griptransition_v1/hdf5init_step800_expert_action_compare.json
```

Interpretation:

```text
The transition-weighted loss directly improves the failed gripper-timing
behavior. Step500 and step800 switch near the expert switch step. However, the
rollout still has zero success, and pose errors along the main translation axes
remain too large after closed-loop drift.
```

Updated next step:

```text
Do not continue tuning gripper-only BCE. The next useful stage is a
rollout-recovery policy objective:

1. collect failed rollout states from these gates;
2. align each failed state to the nearest expert state or hdf5 trajectory phase;
3. add recovery windows to the training cache;
4. train the same WM3D action policy on expert + recovery states;
5. select checkpoints by hdf5-init rollout success first, offline loss second.
```

### Rollout-Recovery Data Probe - 2026-06-02 18:10 CST

New recovery cache builder:

```text
wm3d_v3/benchmarks/libero_rollout_recovery_cache.py
```

Purpose:

```text
Turn failed closed-loop states into training samples:
  failed rollout RGB context + online lowdim/action history
  -> align to expert hdf5 phase
  -> target expert action chunk
```

Collection rollout:

```text
policy: results/wm3d_libero_action_policy_lowdimhist_task1_start_stride4_griptransition_v1/ckpt/eval_steps/eval_step_000800.pt
rollout: results/wm3d_libero_action_policy_lowdimhist_task1_recovery_step800_v1/libero_remote_rollout_hdf5init_task1_demo0_step800_fullframes_300step.json
frames saved: 300
success: 0 / 1
```

#### Recovery v1: nearest-lowdim alignment

Cache:

```text
results/wm3d_libero_action_policy_lowdimhist_task1_recovery_step800_v1_cache/
windows: 96
align_mode: nearest_lowdim
sample_weight: 6.0
mean align distance: 0.7699
```

Config:

```text
configs/libero_action_policy_lowdimhist_task1_recovery_step800_v1.yaml
```

Training:

| checkpoint | step | val L_total | first pose L1 | pose L1 | grip acc | transition acc |
|---|---:|---:|---:|---:|---:|---:|
| best/latest | 600 | 0.2286 | 0.1475 | 0.2726 | 0.9837 | 0.3605 |

Gate:

| checkpoint | rollout success | reward steps | pose L1 first10 | pose L1 first50 | pose L1 mean | grip switch |
|---|---:|---:|---:|---:|---:|---|
| best/step600 | 0 / 1 | 0 | 0.0531 | 0.2593 | 0.2548 | policy 94 vs expert 50 |

Observation:

```text
Nearest-lowdim recovery made the first action almost expert-perfect, but
non-monotonic alignment polluted later phases. Some late failed states aligned
back to early expert indices, and online gripper switch became too late.
```

Artifacts:

```text
results/wm3d_libero_action_policy_lowdimhist_task1_recovery_step800_v1/libero_remote_rollout_hdf5init_task1_demo0_best_300step_summary.json
results/wm3d_libero_action_policy_lowdimhist_task1_recovery_step800_v1/hdf5init_best_expert_action_compare.json
```

#### Recovery v2: time-aligned phase labels

Cache:

```text
results/wm3d_libero_action_policy_lowdimhist_task1_recovery_step800_timealign_v2_cache/
windows: 96
align_mode: time
sample_weight: 6.0
mean align distance: 1.4072
```

Config:

```text
configs/libero_action_policy_lowdimhist_task1_recovery_step800_timealign_v2.yaml
```

Training:

| checkpoint | step | val L_total | first pose L1 | pose L1 | grip acc | transition acc |
|---|---:|---:|---:|---:|---:|---:|
| best | 400 | 0.2690 | 0.1497 | 0.2859 | 0.9809 | 0.4022 |
| latest | 600 | 0.3044 | 0.1569 | 0.2953 | 0.9839 | 0.3478 |

Gate:

| checkpoint | rollout success | reward steps | pose L1 first10 | pose L1 first50 | pose L1 mean | grip switch |
|---|---:|---:|---:|---:|---:|---|
| best/step400 | 0 / 1 | 0 | 0.0629 | 0.2149 | 0.2678 | policy 109 vs expert 50 |

Artifacts:

```text
results/wm3d_libero_action_policy_lowdimhist_task1_recovery_step800_timealign_v2/libero_remote_rollout_hdf5init_task1_demo0_best_300step_summary.json
results/wm3d_libero_action_policy_lowdimhist_task1_recovery_step800_timealign_v2/hdf5init_best_expert_action_compare.json
```

Conclusion:

```text
Simple recovery labels are not enough. They improve initial action imitation but
do not solve long-horizon state recovery. The next policy stage needs explicit
phase/progress conditioning and a better oracle signal for off-expert states.
```

Updated next step:

```text
1. Make progress_tgt a real input/head target, not just a cached field.
2. Add a phase-conditioned policy path so the action head knows where the task
   should be, independent of ambiguous lowdim nearest-neighbor matches.
3. Generate recovery labels with monotonic expert alignment or model-predictive
   relabeling, then gate by rollout success.
4. Keep the VGGT/Qwen/lowdim/action-history interface unchanged, because it
   remains the correct VLA-compatible model boundary.
```

## 2026-06-02 Update: video generation is optional

Leader constraint accepted:

```text
Video/RGB generation is a renderer branch, not a required world-model output.
Action output and latent/geometric prediction must run without activating RGB
video generation.
```

Implemented switches:

```text
wm3d_v3/training/train.py
  --no_pixel
  train.enable_pixel_loss: false
  -> never activates RGB renderer/loss/LPIPS and freezes pixel/context-pixel
     modules so DDP does not see unused trainable video params.

wm3d_v3/eval/run_eval.py
  --skip_rgb_metrics
  -> evaluates pred_tokens/depth/action/progress/proposer metrics without
     forward(pixel=True), RGB metrics, or LPIPS import.

wm3d_v3/eval/system_harness.py
  --no_video
  -> runs world-core/action-sensitivity/TTC/policy/offline-replay scaffold
     without demo GIF generation or RGB eval.
```

Smoke:

```text
command:
  python -m wm3d_v3.eval.system_harness
    --cfg configs/v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce.yaml
    --ckpt results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/ckpt/best.pt
    --out_dir results/_smoke_no_video_harness
    --max_eval_batches 1
    --max_action_batches 1
    --max_ttc_batches 1
    --max_policy_batches 1
    --max_offline_replay_tasks 2
    --no_video
    --skip_libero_probe

result:
  system_scaffold_complete: true
  video_generation_active: false
  rgb_metrics_active: false
  world_core_eval: true
  action_counterfactual: true
  offline_ttc: true
  policy_action_output: true
  offline_replay_adapter: true

world_core_eval_ALL:
  L_state_mse: 0.012205
  L_depth_rel_L1: 0.009384
  grip_acc: 1.0
  progress_abs_err: 0.047415
```

Verification:

```text
py_compile:
  wm3d_v3/eval/run_eval.py
  wm3d_v3/eval/system_harness.py
  wm3d_v3/training/train.py

pytest:
  tests/test_eval_config.py
  tests/test_context_renderer_integration.py
  tests/test_control_progress_heads.py

result:
  12 passed
```

## 2026-06-02 Update: phase/progress action policy experiments

Reason for this round:

```text
The action policy needs a phase/control condition that remains useful when RGB
video generation is disabled. This follows the tau0-style split: policy/control,
world prediction, and renderer are separate modules.
```

Implementation:

```text
wm3d_v3/models/action_policy.py
  added optional progress_state for direct action policy.
  token mode is kept for old experiments.
  summary mode is the new adapter path: no extra token, no pos_embed shape
  change, progress residual is added to the policy summary.

wm3d_v3/training/train_libero_success_p0.py
  fixed LIBERO cache progress_tgt normalization.
  old task1 cache stored absolute target_start in progress_tgt; dataset now
  normalizes target_start by episode_len or hdf5 action length.

wm3d_v3/policy/*
wm3d_v3/benchmarks/libero_remote_runner.py
  HTTP policy path and LIBERO runner pass optional progress_state.

configs/*phase_summary*.yaml
  new summary-adapter phase config for VLA/control path.
```

Critical bugs found:

```text
1. progress_tgt bug:
   task1 start cache had progress values 0..312, while online sent 0..1.
   After fixing dataset normalization, both main/recovery manifests are 0..1
   with mean around 0.496.

2. LayerNorm(1) bug:
   token progress projection used LayerNorm over a scalar, which maps every
   progress value to the same zero-normalized input. This made den130 and
   den257 rollouts identical.

3. token-mode ckpt loading issue:
   adding a progress token changes pos_embed from 20 to 21 tokens, so the old
   policy pos_embed is skipped. Summary adapter fixes this: pos_embed remains
   (1, 20, 768), skipped_action_policy_keys=0 when loading the old policy.
```

Offline and online results:

| run | structural note | best first pose L1 | best pose L1 | online success | gripper switch |
|---|---|---:|---:|---:|---|
| v1 token | bad absolute progress mix | 0.1586 | 0.2830 | 0 / 1 | 219 |
| v2 token | fixed progress, but LayerNorm(1) ignores progress | 0.1693 | 0.2966 | 0 / 1 | 97 |
| v3 token | no LayerNorm(1), but pos_embed skipped | 0.1724 | 0.2917 | 0 / 1 | never closes |
| v4 summary | adapter, pos_embed loads cleanly | 0.1596 | 0.2833 | 0 / 1 | 118 |
| v5 summary grip-refine | v4 + high transition loss | 0.1527 | 0.2809 | 0 / 1 | 254 |
| prior griptransition step800 | no explicit progress | n/a | n/a | 0 / 1 | 46 |

Interpretation:

```text
The correct architecture direction is v4 summary adapter, not token-mode phase.
It preserves the old policy checkpoint and makes progress genuinely influence
pose. However, explicit progress conditioning alone does not solve online
LIBERO success. The remaining failure is not RGB/video-related; it is closed-loop
state/recovery alignment.

Increasing gripper transition loss is not enough and can make the gripper more
conservative. The next useful step is not more local action-head tuning. It is
to build a better recovery/DAgger-style data loop: run failed online traces,
align current lowdim/RGB-token state to expert or oracle corrective actions,
cache those states with correct action history/progress, then retrain and gate
by LIBERO success.
```

Current best candidate:

```text
results/wm3d_libero_action_policy_lowdimhist_task1_phasecond_v4_summary/ckpt/best.pt
configs/v3_p64_140m_actioncond_context_motion_p0_heads_direct_policy_lowdimhist_phase_summary.yaml
configs/libero_action_policy_lowdimhist_task1_phasecond_v4_summary.yaml
results/wm3d_libero_action_policy_lowdimhist_task1_phasecond_v4_summary/libero_remote_rollout_hdf5init_task1_demo0_best_progress_300step.json
```

Next action:

```text
1. Keep v4 summary adapter as the phase/progress architecture.
2. Stop optimizing v5-style transition loss in isolation.
3. Add a recovery data generator that relabels failed online states with
   nearest/monotonic expert corrective actions and preserves online action
   history/progress.
4. Evaluate with no-video system_harness plus LIBERO online success; demo GIF is
   optional and not a gate.
```

## 2026-06-02 Update: monotonic recovery data loop

Implementation:

```text
wm3d_v3/benchmarks/libero_rollout_recovery_cache.py
  added monotonic nearest-lowdim relabeling.
  added monotonic_slack and max_align_distance filters.

wm3d_v3/benchmarks/libero_remote_runner.py
scripts/run_libero_remote_smoke.sh
  added optional pose_scale/max_pose_norm deployment calibration.
```

New recovery cache:

```text
source rollout:
  results/wm3d_libero_action_policy_lowdimhist_task1_phasecond_v4_summary/libero_remote_rollout_hdf5init_task1_demo0_best_progress_fullframes_300step.json

cache:
  results/wm3d_libero_action_policy_lowdimhist_task1_v4_summary_recovery_nearest_mono_v1_cache/manifest.jsonl

summary:
  cached_windows: 128
  skipped_windows: 0
  align_mode: nearest_lowdim
  monotonic: true
  max_align_distance: 4.0
  align_distance_mean: 2.2554
  align_distance_max: 3.5186
```

v6 training:

| run | init | recovery data | first pose L1 | pose L1 | transition acc | online success | switch |
|---|---|---|---:|---:|---:|---:|---:|
| v4 summary | griptransition step800 | old recovery/time | 0.1596 | 0.2833 | 0.375 | 0 / 1 | 118 |
| v6 summary recovery mono | v4 best | new monotonic nearest-lowdim | 0.1429 | 0.2453 | 0.924 | 0 / 1 | 56 |
| v6 pose_scale=0.5 | v6 best | deployment calibration | n/a | n/a | n/a | 0 / 1 | 53 |

Interpretation:

```text
The monotonic recovery loop is the first change in this phase that clearly moves
the closed-loop behavior in the right direction: gripper timing is now close to
the old griptransition model, while offline pose and transition metrics are much
better than v4.

However, online reward remains 0 even with pose_scale=0.5. So the remaining
failure is not simply gripper timing or action magnitude. The next bottleneck is
contact/object-state alignment: the robot reaches/moves, but the corrective data
does not yet guarantee object contact and placement. The next recovery cache
should include more failed traces and filter/weight windows by contact-relevant
state, not just lowdim nearest-neighbor distance.
```

Current best for next iteration:

```text
results/wm3d_libero_action_policy_lowdimhist_task1_phasecond_v6_summary_recovery_mono/ckpt/best.pt
results/wm3d_libero_action_policy_lowdimhist_task1_phasecond_v6_summary_recovery_mono/libero_remote_rollout_hdf5init_task1_demo0_best_progress_300step.json
results/wm3d_libero_action_policy_lowdimhist_task1_phasecond_v6_summary_recovery_mono/libero_remote_rollout_hdf5init_task1_demo0_best_progress_pose05_300step.json
```

## 2026-06-02 Update: object-state/contact-aware recovery probe

Stability checks:

```text
pytest tests -q
  54 passed

system_harness --no_video smoke:
  system_scaffold_complete: true
  world_core_eval: true
  action_counterfactual: true
  offline_ttc: true
  policy_action_output: true
  offline_replay_adapter: true
```

Object-state finding:

```text
The hdf5 demo only stores ee_pos/ee_ori/gripper/joint/rgb, but the online
LIBERO env exposes object-state and per-object poses:
  object-state: 112D
  cream_cheese_1_pos / butter_1_pos / basket_1_pos / ...

Therefore object/contact alignment should be part of benchmark/recovery data,
not a change to the WM3D world core.
```

Implementation:

```text
wm3d_v3/benchmarks/libero_remote_runner.py
  --trace_object_state
  records 112D object_state in step_trace.

wm3d_v3/benchmarks/libero_object_state_reference.py
  replays expert demo in LIBERO py38 and exports expert lowdim/object-state
  references.

wm3d_v3/benchmarks/libero_rollout_recovery_cache.py
  supports expert_object_state_npz and object_state_weight for alignment.
```

Object-aware cache:

```text
expert ref:
  results/wm3d_libero_action_policy_lowdimhist_task1_object_state_ref_demo0.npz
  steps: 247
  object_dim: 112

cache:
  results/wm3d_libero_action_policy_lowdimhist_task1_v6_recovery_object_mono_v1_cache/manifest.jsonl
  cached_windows: 59
  skipped_windows: 69
  object_state_weight: 0.5
  align_distance_mean: 1.5293
  align_distance_max: 4.0931
```

v7/v8 results:

| run | data | first pose L1 | pose L1 | transition acc | online success | switch |
|---|---|---:|---:|---:|---:|---:|
| v6 | expert + lowdim monotonic recovery | 0.1429 | 0.2453 | 0.924 | 0 / 1 | 56 |
| v7 | expert + object-aware recovery | 0.1488 | 0.2802 | 0.453 | 0 / 1 | 59 |
| v8 | expert + lowdim recovery + object-aware recovery | 0.1527 | 0.2509 | 0.787 | 0 / 1 | 54 |

Conclusion:

```text
The object-state path is useful infrastructure, but single-demo object-aware
relabeling is not enough. v6 remains the best current checkpoint because it has
the strongest offline pose/transition balance and online gripper timing.

The WM3D scaffold is now functionally assembled for P0/P1 experiments:
  world core
  optional RGB/video branch
  action/progress policy
  policy HTTP server
  LIBERO online runner
  no-video system harness
  lowdim/object-state recovery cache generation

But it is not ready for full formal training as a complete VLA/world-model
system, because the online benchmark gate is still 0/1 on the task1 hdf5-init
test. Starting a large full training run now would mostly scale a policy that
has not passed the closed-loop contact benchmark.
```

Recommended next gate before formal full training:

```text
1. Generate recovery caches from multiple failed rollouts, not just one v4/v6
   trace.
2. Use object-state to measure phase/contact progress and filter bad relabels.
3. Add a contact-aware offline evaluator that checks object-to-basket/object-to-
   gripper distances from online object-state traces.
4. Require at least one hdf5-init LIBERO success before starting broad formal
   full training.
```

## Update: Named-Object Diagnostics And Handoff Sweep

Status after the latest P0 completion pass:

```text
System scaffold:
  PASS in no-video dual-env harness.
  world_core_eval/action_counterfactual/TTC/policy_action/offline_replay all true.
  LIBERO task/env probe true when harness uses:
    wm3d_python=/data/Minko/.venvs/wm3d/bin/python
    libero_python=/data/Minko/.conda-envs/libero-py38/bin/python

Closed-loop formal-training gate:
  NOT PASSED.
  Learned policy still has 0/1 success on task1 hdf5-init.
```

New infrastructure:

```text
wm3d_v3/benchmarks/libero_remote_runner.py
  --trace_object_state now also records named_poses:
    robot0/eef, target object pos/quat, target_to_robot0_eef vectors.
  --expert_action_hdf5 / --expert_action_prefix_steps / --expert_action_full_replay
    support expert-prefix handoff diagnostics.

wm3d_v3/benchmarks/libero_object_contact_eval.py
  Computes contact/place stage metrics from named object poses:
    gripper-object distance
    object-basket xy distance
    per-object contact/receptacle hits
    stage_score and failure diagnosis
```

Key results:

| run | execution path | success | stage score | diagnosis |
|---|---|---:|---:|---|
| v6 ranked | proposer/ranker selected action | 0 / 1 | 0.00 | no target contact |
| v6 direct | direct action_policy | 0 / 1 | 0.25 | cream contact only |
| v9 direct | v6 + direct recovery cache | 0 / 1 | 0.25 | cream contact only |
| expert full replay through remote runner | expert actions | 1 / 1 | 1.00 | runner/action convention valid |
| expert60 -> v9 direct | handoff after early grasp | 0 / 1 | 0.50 | cream placed, butter untouched |
| expert120 -> v9 direct | handoff after longer first-object phase | 0 / 1 | 0.50 | cream placed, butter untouched |
| expert180 -> v9 direct | handoff in second-object phase | 0 / 1 | 0.75 | both contacted, butter not placed |
| expert220 -> v9 direct | handoff near final stage | 1 / 1 | 1.00 | final release/settle can complete |
| v10 direct | v9 + handoff60/180 stage recovery | 0 / 1 | 0.25 | cream contact only; gripper transition regressed |
| v10 step300 direct | v10 eval checkpoint | 0 / 1 | 0.25 | same gate failure |
| v11 direct | object_state-conditioned policy | 0 / 1 | 0.50 | cream placed, butter untouched |
| v12 direct | v11 + pure-failure recovery | 0 / 1 | 0.75 | both contacted, butter not placed |
| v13 step400 direct | 8D stage/target/subgoal plan_state | 0 / 1 | 0.75 | stage tracker worked, butter not placed |
| v13 step400 force-grip diagnostic | forced closed gripper in stage3 | 0 / 1 | 0.75 | gripper alone is insufficient |
| v14 step500 direct | 17D target-geometry plan_state | 0 / 1 | 0.75 | earlier gripper close, but eef-butter min dist still too large |
| v15 best direct | v14 failure-tail DAgger | 0 / 1 | 0.50 | overfit/regressed second-object approach |

Concrete interpretation:

```text
1. The model scaffold is assembled as a P0/P1 prototype, and optional video can
   stay inactive for action/prediction evaluation.
2. The ranked/proposer execution path should not be the default online policy
   right now. It selected bad actions early:
     ranked first40 pose L1 ~= 0.435, grip match ~= 0.075
     direct first40 pose L1 ~= 0.089, grip match ~= 1.000
3. The direct action policy is the correct current online path, but it is not a
   complete VLA yet. It reaches the first object but fails multi-object stage
   switching and second-object transport/place.
4. The runner and action convention are not the blocker: expert full replay
   through the same remote runner succeeds.
5. The immediate next training target should be stage/phase-complete recovery
   around the second-object transition, not another generic large-scale run.
6. The first stage-recovery attempt (v10) improved offline pose metrics but did
   not improve pure online success. It regressed gripper transition behavior in
   pure closed-loop, so the next change should add an explicit stage/subgoal
   signal or object-centric policy input/head instead of simply adding more
   weighted recovery windows.
7. Object-state and plan-state conditioning are useful but not sufficient:
   v11 moved the gate from 0.25 to 0.50, and v12/v13/v14 reached 0.75.
   The remaining failure is local closed-loop control for the second object:
   the policy can switch stages and contact butter, but it does not reliably
   close at the right geometry or move the object into the basket.
8. Continuing cache-only DAgger is now a low-value path. v15's high-weight
   failure-tail correction regressed from 0.75 to 0.50, so the next change
   should be architectural: add an object-centric/local action head or
   waypoint-style subgoal controller, while preserving the existing world core
   and optional RGB/video branch.
```

## 2026-06-03 Update: Waypoint Residual Experiments

Implementation added:

```text
ActionChunkPolicy optional waypoint head:
  policy_enable_waypoint_head
  policy_waypoint_active_stages
  policy_waypoint_mode=residual/direct/aux

Policy selection diagnostic:
  selection_mode=plan_waypoint

Main train/eval builder parity:
  training/train.py now passes object_state, plan_state, local residual,
  and waypoint policy fields consistently with eval/run_eval.py.
```

Verification:

```text
pytest tests -q
  61 passed, 9 warnings
```

Comparable LIBERO gate settings must be fixed:

```text
camera_size=128
warmup_steps=0
context_T=16
action_history_len=16
send_lowdim=true
send_object_state=true
send_plan_state=true
plan_state_dim=17
send_progress=true
trace_object_state=true
```

Using `camera_size=224` and `warmup_steps=5` changes the first action and drops
the current-code v16b baseline to stage_score 0.00, so those runs are not
comparable to the earlier v16b 0.75 gate.

New results under the comparable `camera128/warm0` gate:

| run | execution path | success | stage score | diagnosis |
|---|---|---:|---:|---|
| v16b best direct | local residual, old comparable rollout | 0 / 1 | 0.75 | cream placed, butter contacted, butter not placed |
| v19 best direct | all-stage waypoint residual | 0 / 1 | not used | invalid gate initially run at camera224/warm5; regressed early behavior |
| v20 best direct | stage3-only waypoint residual | 0 / 1 | 0.75 | preserved gate but did not improve butter placement |

v20 details:

```text
cream contact: true, contact_step=44
cream receptacle: true, receptacle_step=114
butter contact: true, contact_step=191
butter receptacle: false
butter min_eef_dist: 0.0375
butter final_receptacle_xy_dist: 0.2339
```

Interpretation:

```text
1. The active-stage waypoint head is safe enough as an opt-in scaffold but is
   not an improvement over v16b.
2. Supervised residual/waypoint BC from mixed caches is not solving the final
   butter placement failure.
3. The next useful direction is not another residual cache experiment. The next
   P0 change should explicitly model a terminal place/release controller or a
   learned evaluator/selector trained from mixed success and failure traces,
   while keeping RGB/Hunyuan video optional.
```

Decision:

```text
Do not start formal full training yet.

Formal training should start only after the pure learned direct policy gets at
least one hdf5-init LIBERO task1 success, or after the benchmark harness is
expanded to multiple tasks with an explicitly accepted lower gate.
```
